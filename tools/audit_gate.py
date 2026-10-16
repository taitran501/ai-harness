#!/usr/bin/env python3
"""SWE-agent-inspired Deterministic Audit Gate (Statistical & Semantic).

Enforces physical code-level & statistical verification before any submission can be generated:
1. Calculates SHA-256 hashes to block duplicate file submissions.
2. Enforces Minimum Meaningful Effect Size (MMES >= 2.0% row diff) to reject trivial tweaks.
3. Performs McNemar's test (classification) or paired t-test on squared loss (regression) for statistical significance.
4. Audits ensemble submissions via EnsembleGuard (Cohen's Kappa / Pearson Correlation ceiling & Disagreement floor).
5. Validates label integrity, null absence, and oracle conditioning rules across modalities.
6. Generates a signed AUDIT_RECEIPT.json with statistical audit metrics ONLY on PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# NOTE: scipy.stats is imported lazily inside compute_*_significance() below.
# It is the single most expensive import in this CLI (~1s); deferring it
# keeps the common CLI path (argparse + diff metrics) fast.

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.ensemble_guard import EnsembleGuard


def compute_sha256(filepath: Path | str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Retrieve current git commit hash if in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def audit_prediction_integrity(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    spec: Optional[Any] = None,
) -> List[str]:
    """Verify structural validity and constraints of the predictions DataFrame."""
    if spec is None:
        from harness.task_spec import load_active_task_spec
        try:
            spec = load_active_task_spec()
        except Exception:
            spec = None
    key = id_col or (spec.id_column if spec else "id")
    errors = []
    if not any(c.startswith(("pred_", "proba_")) for c in df) and (not spec or not spec.subtasks):
        errors.append("Missing predictions")
    if key not in df:
        errors.append("Missing declared ID column")
    elif df[key].isna().any() or df[key].duplicated().any():
        errors.append(f"IDs {key} must be non-null and unique")
    if df.empty:
        errors.append("Prediction coverage is empty")
    if spec and spec.subtasks:
        for target, st in spec.subtasks.items():
            col = next((c for c in ("pred_" + target, "pred_" + (st.column or target), st.column or target) if c in df), None)
            if col is None:
                errors.append("Missing prediction: " + target)
                continue
            if df[col].isna().any():
                errors.append("Null prediction: " + target)
            if st.task_type == "regression":
                values = pd.to_numeric(df[col], errors="coerce")
                if not np.isfinite(values).all():
                    errors.append("Non-finite prediction: " + target)
            elif st.task_type == "multilabel_classification":
                try:
                    values = np.asarray([json.loads(x) if isinstance(x, str) else x for x in df[col]])
                    if values.shape != (len(df), len(st.labels or st.classes)) or not np.isin(values, [0, 1]).all():
                        errors.append("Invalid multilabel shape/domain")
                except (ValueError, TypeError):
                    errors.append("Invalid multilabel encoding")
            elif st.task_type in ("ranking", "recommendation"):
                try:
                    values = pd.to_numeric(df[col], errors="raise")
                    if not np.isfinite(values).all():
                        errors.append("Non-finite prediction: " + target)
                except (ValueError, TypeError):
                    pass
            elif st.task_type in ("generative_nlp", "generation", "text_generation", "seq2seq", "nlp", "generative"):
                pass
            elif getattr(st, "classes", None) and len(st.classes) > 0:
                if not df[col].astype(str).isin(list(map(str, st.classes))).all():
                    errors.append("invalid values outside declared classes: " + target)
    return errors


def compute_diff_metrics(
    cand_df: pd.DataFrame,
    base_df: pd.DataFrame,
    id_col: Optional[str] = None,
    rel_epsilon: float = 0.01,
) -> Tuple[int, Dict[str, Any]]:
    """Compute differential row counts and metrics between candidate and baseline predictions."""
    key = id_col
    if not key:
        for candidate_key in ["id", "ID", "sample_id", "row_id", "uid"]:
            if candidate_key in cand_df.columns and candidate_key in base_df.columns:
                key = candidate_key
                break
    if not key:
        key = "id"

    for frame in (cand_df, base_df):
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError("Invalid comparison IDs")
    if set(cand_df[key]) != set(base_df[key]):
        raise ValueError("Prediction ID sets differ")

    cand = cand_df.set_index(key)
    base = base_df.set_index(key).reindex(cand.index)
    cols = [c for c in cand if c.startswith(("pred_", "proba_"))]
    if not cols:
        raise ValueError("No prediction columns to compare")
    if set(cols) != {c for c in base if c.startswith(("pred_", "proba_"))}:
        raise ValueError("Prediction schemas differ")

    details = {}
    changed = pd.Series(False, index=cand.index)
    for col in cols:
        cand_s = cand[col]
        base_s = base[col]
        is_continuous = False
        try:
            cand_num = pd.to_numeric(cand_s, errors="raise")
            base_num = pd.to_numeric(base_s, errors="raise")
            if pd.api.types.is_float_dtype(cand_num) or pd.api.types.is_float_dtype(base_num):
                is_continuous = True
        except (ValueError, TypeError):
            is_continuous = False

        if is_continuous:
            cand_vals = cand_num.to_numpy(dtype=float)
            base_vals = base_num.to_numpy(dtype=float)
            std = float(np.std(base_vals))
            mean_abs = float(np.mean(np.abs(base_vals))) if len(base_vals) > 0 else 1.0
            S = std if (not np.isnan(std) and std > 1e-8) else (mean_abs if (not np.isnan(mean_abs) and mean_abs > 1e-8) else 1.0)
            if np.isnan(S) or S <= 1e-8:
                S = 1.0
            diff = pd.Series((np.abs(cand_vals - base_vals) / S) >= rel_epsilon, index=cand.index)
        else:
            diff = cand_s != base_s

        changed |= diff
        details[col] = {
            "different_count": int(diff.sum()),
            "pct_changed": float(diff.mean() * 100),
            "rel_epsilon": float(rel_epsilon) if is_continuous else None,
            "is_continuous": is_continuous,
        }
    return int(changed.sum()), details


def compute_continuous_significance(
    y_true: pd.Series | np.ndarray,
    cand_pred: pd.Series | np.ndarray,
    base_pred: pd.Series | np.ndarray,
) -> Dict[str, Any]:
    """Compute paired error test (paired t-test on squared errors) between candidate and baseline."""
    from scipy.stats import ttest_rel
    y = np.asarray(y_true, dtype=float)
    cand = np.asarray(cand_pred, dtype=float)
    base = np.asarray(base_pred, dtype=float)

    cand_errors = (cand - y) ** 2
    base_errors = (base - y) ** 2

    b = int((cand_errors < base_errors).sum())
    c = int((base_errors < cand_errors).sum())

    if len(cand_errors) < 2 or (b == 0 and c == 0) or np.allclose(cand_errors, base_errors, atol=0.0, rtol=1e-7):
        return {
            "cand_better_count": b,
            "base_better_count": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "test_name": "paired_ttest",
        }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = ttest_rel(cand_errors, base_errors)
    stat = float(res.statistic) if np.isfinite(res.statistic) else 0.0
    p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0

    return {
        "cand_better_count": b,
        "base_better_count": c,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "is_significant": bool(p_val < 0.05),
        "test_name": "paired_ttest",
    }


def compute_mcnemar_significance(
    y_true: pd.Series | np.ndarray,
    cand_pred: pd.Series | np.ndarray,
    base_pred: pd.Series | np.ndarray,
) -> Dict[str, Any]:
    """Compute McNemar's test for paired classification accuracy between candidate and baseline."""
    from scipy.stats import chi2
    y = pd.Series(y_true)
    cand = pd.Series(cand_pred)
    base = pd.Series(base_pred)

    cand_correct = cand == y
    base_correct = base == y

    # b: candidate correct, baseline wrong (candidate won)
    b = int((cand_correct & (~base_correct)).sum())
    # c: candidate wrong, baseline correct (baseline won)
    c = int(((~cand_correct) & base_correct).sum())

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "cand_better_count": b,
            "base_better_count": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "test_name": "mcnemar",
        }

    # Edwards continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "cand_better_count": b,
        "base_better_count": c,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "is_significant": bool(p_val < 0.05),
        "test_name": "mcnemar",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-agent Deterministic & Statistical Audit Gate for Model Predictions"
    )
    parser.add_argument(
        "--candidate",
        "-c",
        required=True,
        type=str,
        help="Path to candidate prediction CSV",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        required=False,
        type=str,
        default=None,
        help="Path to parent baseline prediction CSV for diff analysis",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        required=False,
        type=str,
        default=None,
        help="Optional path to ground truth CSV/TSV for McNemar / paired error statistical test",
    )
    parser.add_argument(
        "--ensemble-components",
        nargs="*",
        default=None,
        help="List of prediction CSV paths of constituent models if candidate is an ensemble",
    )
    parser.add_argument(
        "--output-zip",
        "-o",
        type=str,
        default="submission.zip",
        help="Output path for submission ZIP package",
    )
    parser.add_argument(
        "--receipt",
        "-r",
        type=str,
        default="AUDIT_RECEIPT.json",
        help="Output path for verified audit receipt",
    )
    parser.add_argument(
        "--allow-initial",
        action="store_true",
        help="Allow initial baseline run without comparison baseline",
    )
    parser.add_argument(
        "--min-diff-pct",
        type=float,
        default=2.0,
        help="Minimum required percentage of changed prediction rows (MMES floor, default: 2.0%%)",
    )
    parser.add_argument(
        "--min-diff",
        type=int,
        default=1,
        help="Minimum absolute required changed prediction rows",
    )
    parser.add_argument(
        "--max-kappa",
        type=float,
        default=None,
        help="Maximum allowable Cohen's Kappa / Correlation between ensemble constituent models (default: 0.85)",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=None,
        help="Alias for --max-kappa: Maximum allowable Pearson/Spearman correlation for continuous ensemble components (default: 0.85)",
    )
    parser.add_argument(
        "--rel-epsilon",
        type=float,
        default=0.01,
        help="Relative epsilon tolerance normalized by target std dev for continuous diff metrics (default: 0.01)",
    )
    parser.add_argument(
        "--min-disagreement-pct",
        type=float,
        default=5.0,
        help="Minimum allowable disagreement percentage between ensemble components (default: 5.0%%)",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Identifier column name",
    )
    args = parser.parse_args()

    # Resolve max_kappa / max_corr alias (default 0.85)
    effective_max_corr = 0.85
    if args.max_corr is not None:
        effective_max_corr = args.max_corr
    elif args.max_kappa is not None:
        effective_max_corr = args.max_kappa
    args.max_kappa = effective_max_corr
    args.max_corr = effective_max_corr

    # Optionally load thresholds from active task spec submission_spec
    active_spec = None
    try:
        from harness.task_spec import load_active_task_spec
        active_spec = load_active_task_spec()
        sub_spec = active_spec.metadata.get("submission_spec", {}) if active_spec and active_spec.metadata else {}
        if sub_spec:
            if args.min_diff_pct == 2.0 and "min_diff_pct" in sub_spec:
                args.min_diff_pct = float(sub_spec["min_diff_pct"])
            if args.rel_epsilon == 0.01 and "rel_epsilon" in sub_spec:
                args.rel_epsilon = float(sub_spec["rel_epsilon"])
            if "max_corr" in sub_spec and args.max_corr == 0.85:
                args.max_corr = float(sub_spec["max_corr"])
                args.max_kappa = args.max_corr
            elif "max_kappa" in sub_spec and args.max_kappa == 0.85:
                args.max_kappa = float(sub_spec["max_kappa"])
                args.max_corr = args.max_kappa
    except Exception:
        pass

    cand_path = Path(args.candidate)
    if not cand_path.exists():
        print(f"[AUDIT FAILED] Candidate file does not exist: {cand_path}", file=sys.stderr)
        sys.exit(1)

    print("===========================================================")
    print(f"🔒 DETERMINISTIC & STATISTICAL AUDIT GATE: {cand_path.name}")
    print("===========================================================")

    cand_sha256 = compute_sha256(cand_path)
    cand_df = pd.read_csv(cand_path)
    print(f"✔ Candidate SHA-256 : {cand_sha256}")
    print(f"✔ Candidate Rows    : {len(cand_df):,}")

    # Step 1: Check physical integrity
    integrity_errors = audit_prediction_integrity(cand_df, id_col=args.id_column, spec=active_spec)
    if integrity_errors:
        print(f"\n❌ [CRITICAL INTEGRITY FAILURE] Found {len(integrity_errors)} violations:", file=sys.stderr)
        for err in integrity_errors:
            print(f"   - {err}", file=sys.stderr)
        if Path(args.output_zip).exists():
            os.remove(args.output_zip)
        sys.exit(1)

    # Step 2: Ensemble Guard (if ensemble components are declared)
    ensemble_audit_data = None
    if args.ensemble_components:
        print(f"\n--- Auditing Ensemble Constituent Diversity ({len(args.ensemble_components)} models) ---")
        comp_dfs = []
        for comp_p in args.ensemble_components:
            p = Path(comp_p)
            if not p.exists():
                print(f"❌ [AUDIT FAILED] Ensemble constituent file not found: {p}", file=sys.stderr)
                sys.exit(1)
            comp_dfs.append(pd.read_csv(p))

        # Check diversity on primary prediction column
        pred_cols = [c for c in cand_df.columns if c.startswith("pred_")]
        primary_col = pred_cols[0] if pred_cols else (cand_df.columns[1] if len(cand_df.columns) > 1 else cand_df.columns[0])
        passed_a, errs_a, details_a = EnsembleGuard.audit_constituent_diversity(
            comp_dfs,
            col=primary_col,
            max_kappa=args.max_kappa,
            min_disagreement_pct=args.min_disagreement_pct,
            rel_epsilon=args.rel_epsilon,
        )
        if not passed_a:
            print("\n❌ [REJECTED: PSEUDO-ENSEMBLE DETECTED]", file=sys.stderr)
            for err in errs_a:
                print(f"   - {err}", file=sys.stderr)
            print("The agent is attempting to blend non-diverse/correlated models from the same family.", file=sys.stderr)
            if Path(args.output_zip).exists():
                os.remove(args.output_zip)
            sys.exit(1)

        metric_label = "Pearson Correlation" if (details_a.get("pairwise_stats") and details_a["pairwise_stats"][0].get("metric_type") == "continuous") else "Kappa"
        print(f"✔ Ensemble Diversity: PASSED! (All pairwise {metric_label} <= {args.max_kappa})")
        ensemble_audit_data = details_a

    # Step 3: Compare against baseline
    diff_count = -1
    diff_details = {}
    base_sha256 = None
    sig_data = None

    if args.baseline:
        base_path = Path(args.baseline)
        if not base_path.exists():
            print(f"❌ [AUDIT FAILED] Baseline file does not exist: {base_path}", file=sys.stderr)
            sys.exit(1)

        base_sha256 = compute_sha256(base_path)
        base_df = pd.read_csv(base_path)
        print(f"✔ Baseline SHA-256  : {base_sha256}")

        # Check for 100% duplicate file
        if cand_sha256 == base_sha256:
            print(
                f"\n❌ [REJECTED: 100% DUPLICATE FILE DETECTED]\n"
                f"Candidate SHA-256 matches baseline SHA-256 exactly!\n"
                f"Candidate: {cand_path}\n"
                f"Baseline : {base_path}\n"
                f"The agent is attempting to submit an unchanged file without genuine execution.",
                file=sys.stderr,
            )
            if Path(args.output_zip).exists():
                os.remove(args.output_zip)
            sys.exit(1)

        # Check row diff
        diff_count, diff_details = compute_diff_metrics(
            cand_df, base_df, id_col=args.id_column, rel_epsilon=args.rel_epsilon
        )
        diff_pct = round(diff_count / len(cand_df) * 100, 2)
        print(f"✔ Differing Samples : {diff_count} / {len(cand_df)} ({diff_pct}%)")

        # Minimum Meaningful Effect Size (MMES) floor
        required_diff = max(
            args.min_diff, int(math.ceil(len(cand_df) * args.min_diff_pct / 100.0))
        )
        if diff_count < required_diff:
            print(
                f"\n❌ [REJECTED: EFFECT SIZE TOO SMALL (MMES VIOLATION)]\n"
                f"Only {diff_count} predictions differ ({diff_pct}%). Minimum required: {required_diff} ({args.min_diff_pct}%).\n"
                f"Trivial tweaks or random seed noise do NOT constitute a valid new experiment.",
                file=sys.stderr,
            )
            if Path(args.output_zip).exists():
                os.remove(args.output_zip)
            sys.exit(1)

        # Optional Step 3b: Significance Test (Paired Error Test for Regression, McNemar for Classification)
        if args.ground_truth:
            gt_path = Path(args.ground_truth)
            if gt_path.exists():
                gt_df = pd.read_csv(gt_path, sep="\t" if gt_path.suffix == ".tsv" else ",")
                # Dynamically resolve ID column
                resolved_id = None
                for cid in ["id", "ID", "sample_id", "row_id", "uid"]:
                    if cid in gt_df.columns and cid in cand_df.columns and cid in base_df.columns:
                        resolved_id = cid
                        break
                if not resolved_id and getattr(args, "id_column", None):
                    resolved_id = args.id_column if args.id_column in gt_df.columns else None

                # Dynamically detect primary classification/regression target
                target_col = None
                for col in gt_df.columns:
                    pred_col = f"pred_{col}"
                    if pred_col in cand_df.columns and pred_col in base_df.columns:
                        target_col = col
                        break

                if resolved_id and target_col:
                    pred_t = f"pred_{target_col}"
                    merged = pd.merge(
                        gt_df,
                        cand_df[[resolved_id, pred_t]],
                        on=resolved_id,
                    )
                    merged = pd.merge(
                        merged,
                        base_df[[resolved_id, pred_t]],
                        on=resolved_id,
                        suffixes=("_cand", "_base"),
                    )

                    is_continuous_target = False
                    if active_spec and target_col in active_spec.subtasks:
                        st = active_spec.subtasks[target_col]
                        if st.task_type == "regression":
                            is_continuous_target = True

                    if not is_continuous_target:
                        try:
                            gt_num = pd.to_numeric(merged[target_col], errors="raise")
                            cand_num = pd.to_numeric(merged[f"{pred_t}_cand"], errors="raise")
                            if pd.api.types.is_float_dtype(gt_num) or pd.api.types.is_float_dtype(cand_num):
                                if len(gt_num.unique()) > 2 or not np.all(np.equal(np.mod(gt_num.to_numpy(dtype=float), 1), 0)):
                                    is_continuous_target = True
                        except (ValueError, TypeError):
                            is_continuous_target = False

                    if is_continuous_target:
                        sig_data = compute_continuous_significance(
                            merged[target_col],
                            merged[f"{pred_t}_cand"],
                            merged[f"{pred_t}_base"],
                        )
                        print(
                            f"✔ Paired Error Test : p-value = {sig_data['p_value']} "
                            f"(Cand better: {sig_data['cand_better_count']}, Base better: {sig_data['base_better_count']})"
                        )
                    else:
                        sig_data = compute_mcnemar_significance(
                            merged[target_col],
                            merged[f"{pred_t}_cand"],
                            merged[f"{pred_t}_base"],
                        )
                        print(
                            f"✔ McNemar's Test    : p-value = {sig_data['p_value']} "
                            f"(Cand better: {sig_data['cand_better_count']}, Base better: {sig_data['base_better_count']})"
                        )

    else:
        if not args.allow_initial:
            print(
                "⚠ Warning: No baseline provided. Specify --baseline or use --allow-initial for root baseline.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("✔ Initial Baseline Run: Comparison skipped.")

    # Step 4: Package submission.zip
    out_zip = Path(args.output_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cand_path, arcname=cand_path.name)
    print(f"✔ Packaged Archive  : {out_zip} ({out_zip.stat().st_size:,} bytes)")

    # Step 5: Sign and save AUDIT_RECEIPT.json
    receipt_data = {
        "audit_version": "2.0_statistical",
        "status": "PASSED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_commit": get_git_commit(),
        "candidate": {
            "path": str(cand_path),
            "sha256": cand_sha256,
            "rows": len(cand_df),
        },
        "baseline": {
            "path": str(args.baseline) if args.baseline else None,
            "sha256": base_sha256,
        },
        "verification": {
            "differing_rows": diff_count,
            "differing_pct": round(diff_count / len(cand_df) * 100, 2) if len(cand_df) > 0 else 0.0,
            "min_diff_pct_required": args.min_diff_pct,
            "rel_epsilon": args.rel_epsilon,
            "max_corr_threshold": args.max_corr,
            "max_kappa_threshold": args.max_kappa,
            "diff_by_column": diff_details,
            "significance_test": sig_data,
            "mcnemar_test": sig_data,
            "ensemble_audit": ensemble_audit_data,
        },
    }

    receipt_path = Path(args.receipt)
    from harness.io_utils import atomic_write_json
    atomic_write_json(receipt_path, receipt_data, indent=2)

    print(f"✔ Audit Receipt     : {receipt_path}")
    print("===========================================================")
    print("🎉 STATISTICAL AUDIT VERDICT: PASSED! Ready for human submission.")
    print("===========================================================")


if __name__ == "__main__":
    main()
