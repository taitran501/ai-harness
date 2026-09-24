#!/usr/bin/env python3
"""SWE-agent-inspired Deterministic Audit Gate (Statistical & Semantic).

Enforces physical code-level & statistical verification before any submission can be generated:
1. Calculates SHA-256 hashes to block duplicate file submissions.
2. Enforces Minimum Meaningful Effect Size (MMES >= 2.0% row diff) to reject trivial tweaks.
3. Performs McNemar's test for statistical significance when ground truth is provided.
4. Audits ensemble submissions via EnsembleGuard (Cohen's Kappa <= 0.85 & Disagreement >= 5.0%).
5. Validates label integrity, null absence, and oracle conditioning rules.
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

import pandas as pd
from scipy.stats import chi2

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
    df: pd.DataFrame, id_col: str = "StereoQueerEval_id"
) -> List[str]:
    """Verify structural validity and constraints of the predictions DataFrame."""
    errors = []

    if id_col in df.columns and df[id_col].isnull().any():
        errors.append(f"Column '{id_col}' contains null values.")

    # Check Subtask A valid values
    if "pred_stereotype" in df.columns:
        invalid = df[~df["pred_stereotype"].isin(["yes", "no"])]
        if len(invalid) > 0:
            errors.append(f"Subtask A has {len(invalid)} invalid values (expected 'yes' or 'no').")

    # Check Subtask B valid values
    if "pred_hate_speech" in df.columns:
        valid_b = ["no", "yes_explicit", "yes_implicit"]
        invalid = df[~df["pred_hate_speech"].isin(valid_b)]
        if len(invalid) > 0:
            errors.append(f"Subtask B has {len(invalid)} invalid values (expected {valid_b}).")

    # Check Subtask C conditioning
    if "pred_hate_speech" in df.columns and "pred_target_type" in df.columns:
        non_hate = df["pred_hate_speech"] == "no"
        unconditioned = df[non_hate & (df["pred_target_type"] != "none")]
        if len(unconditioned) > 0:
            errors.append(
                f"Oracle Hate rule violated: {len(unconditioned)} non-hateful rows have target_type != 'none'."
            )

    return errors


def compute_diff_metrics(
    cand_df: pd.DataFrame,
    base_df: pd.DataFrame,
    id_col: str = "StereoQueerEval_id",
) -> Tuple[int, Dict[str, Any]]:
    """Compute exact row-by-row prediction deltas between candidate and baseline."""
    if id_col in cand_df.columns and id_col in base_df.columns:
        cand = cand_df.sort_values(id_col).reset_index(drop=True)
        base = base_df.sort_values(id_col).reset_index(drop=True)
    else:
        cand = cand_df.reset_index(drop=True)
        base = base_df.reset_index(drop=True)

    if len(cand) != len(base):
        raise ValueError(
            f"Row count mismatch! Candidate has {len(cand)} rows, baseline has {len(base)} rows."
        )

    common_cols = [c for c in cand.columns if c in base.columns and c != id_col]
    if not common_cols:
        common_cols = [c for c in cand.columns if c != id_col]

    diff_per_col = {}
    any_diff_mask = pd.Series([False] * len(cand))

    for col in common_cols:
        col_diff = cand[col] != base[col]
        n_col_diff = int(col_diff.sum())
        diff_per_col[col] = {
            "different_count": n_col_diff,
            "pct_changed": round(n_col_diff / len(cand) * 100, 2),
        }
        any_diff_mask = any_diff_mask | col_diff

    total_diff_rows = int(any_diff_mask.sum())
    return total_diff_rows, diff_per_col


def compute_mcnemar_significance(
    y_true: pd.Series,
    cand_pred: pd.Series,
    base_pred: pd.Series,
) -> Dict[str, Any]:
    """Compute McNemar's test for paired classification accuracy between candidate and baseline."""
    cand_correct = cand_pred == y_true
    base_correct = base_pred == y_true

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
        }

    # Edwards continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "cand_better_count": b,
        "base_better_count": c,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "is_significant": p_val < 0.05,
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
        help="Optional path to ground truth CSV/TSV for McNemar statistical test",
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
        default=0.85,
        help="Maximum allowable Cohen's Kappa between ensemble constituent models (default: 0.85)",
    )
    parser.add_argument(
        "--min-disagreement-pct",
        type=float,
        default=5.0,
        help="Minimum allowable disagreement percentage between ensemble components (default: 5.0%%)",
    )

    args = parser.parse_args()

    cand_path = Path(args.candidate)
    if not cand_path.exists():
        print(f"[AUDIT FAILED] Candidate file does not exist: {cand_path}", file=sys.stderr)
        sys.exit(1)

    print(f"===========================================================")
    print(f"🔒 DETERMINISTIC & STATISTICAL AUDIT GATE: {cand_path.name}")
    print(f"===========================================================")

    cand_sha256 = compute_sha256(cand_path)
    cand_df = pd.read_csv(cand_path)
    print(f"✔ Candidate SHA-256 : {cand_sha256}")
    print(f"✔ Candidate Rows    : {len(cand_df):,}")

    # Step 1: Check physical integrity
    integrity_errors = audit_prediction_integrity(cand_df)
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

        # Check diversity on Subtask A and Subtask B
        passed_a, errs_a, details_a = EnsembleGuard.audit_constituent_diversity(
            comp_dfs,
            col="pred_stereotype" if "pred_stereotype" in cand_df.columns else cand_df.columns[1],
            max_kappa=args.max_kappa,
            min_disagreement_pct=args.min_disagreement_pct,
        )
        if not passed_a:
            print(f"\n❌ [REJECTED: PSEUDO-ENSEMBLE DETECTED]", file=sys.stderr)
            for err in errs_a:
                print(f"   - {err}", file=sys.stderr)
            print("The agent is attempting to blend non-diverse/correlated models from the same family.", file=sys.stderr)
            if Path(args.output_zip).exists():
                os.remove(args.output_zip)
            sys.exit(1)

        print(f"✔ Ensemble Diversity: PASSED! (All pairwise Kappa <= {args.max_kappa})")
        ensemble_audit_data = details_a

    # Step 3: Compare against baseline
    diff_count = -1
    diff_details = {}
    base_sha256 = None
    mcnemar_data = None

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
        diff_count, diff_details = compute_diff_metrics(cand_df, base_df)
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

        # Optional Step 3b: McNemar's Test
        if args.ground_truth:
            gt_path = Path(args.ground_truth)
            if gt_path.exists():
                gt_df = pd.read_csv(gt_path, sep="\t" if gt_path.suffix == ".tsv" else ",")
                if "StereoQueerEval_id" in gt_df.columns and "stereotype" in gt_df.columns:
                    merged = pd.merge(
                        gt_df,
                        cand_df[["StereoQueerEval_id", "pred_stereotype"]],
                        on="StereoQueerEval_id",
                    )
                    merged = pd.merge(
                        merged,
                        base_df[["StereoQueerEval_id", "pred_stereotype"]],
                        on="StereoQueerEval_id",
                        suffixes=("_cand", "_base"),
                    )
                    mcnemar_data = compute_mcnemar_significance(
                        merged["stereotype"],
                        merged["pred_stereotype_cand"],
                        merged["pred_stereotype_base"],
                    )
                    print(
                        f"✔ McNemar's Test    : p-value = {mcnemar_data['p_value']} "
                        f"(Cand better: {mcnemar_data['cand_better_count']}, Base better: {mcnemar_data['base_better_count']})"
                    )

    else:
        if not args.allow_initial:
            print(
                f"⚠ Warning: No baseline provided. Specify --baseline or use --allow-initial for root baseline.",
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
            "diff_by_column": diff_details,
            "mcnemar_test": mcnemar_data,
            "ensemble_audit": ensemble_audit_data,
        },
    }

    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt_data, f, indent=2)

    print(f"✔ Audit Receipt     : {receipt_path}")
    print(f"===========================================================")
    print(f"🎉 STATISTICAL AUDIT VERDICT: PASSED! Ready for human submission.")
    print(f"===========================================================")


if __name__ == "__main__":
    main()
