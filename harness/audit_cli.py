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
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: scipy.stats is imported lazily inside compute_*_significance() below.
# It is the single most expensive import in this CLI (~1s); deferring it
# keeps the common CLI path (argparse + diff metrics) fast.



from harness.ensemble_guard import EnsembleGuard
from harness.audit_gate import (compute_sha256, get_git_commit, audit_prediction_integrity, compute_diff_metrics, compute_continuous_significance, compute_mcnemar_significance)


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
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--rule-id", default=None)
    parser.add_argument("--max-spearman", type=float, default=.995)
    parser.add_argument("--receipt-scope", choices=("validation", "submission"), default="validation")
    parser.add_argument("--competition", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--approval-id", default=None)
    parser.add_argument("--user-approved", action="store_true")
    args = parser.parse_args()
    from harness.io_utils import atomic_write_json
    # Replace a stale success before inspecting this candidate, including on failure.
    for output in (args.receipt, args.output_zip):
        for parent in Path(output).resolve().parents:
            manifest_path = parent / "artifact_manifest.json"
            if manifest_path.is_file():
                manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "COMPLETED":
                    parser.error("Audit outputs must be outside completed immutable runs")
    if "--receipt" in sys.argv or "-r" in sys.argv or Path(args.receipt).exists():
        atomic_write_json(args.receipt, {"audit_version": "3.0_scoped", "scope": args.receipt_scope,
                                        "status": "FAILED", "candidate": {"path": args.candidate}, "dispatch_authorized": False})
    if args.receipt_scope == "submission":
        from harness.submission_gate import SubmissionGate
        if not args.competition:
            parser.error("--receipt-scope submission requires --competition and sealed experiment evidence")
        try:
            gate = SubmissionGate(args.project_root)
            receipt, _, _, _ = gate.evaluate(args.competition, args.candidate, exp_id=args.exp_id,
                rule_id=args.rule_id, approval_id=args.approval_id, user_approved=args.user_approved,
                max_spearman=args.max_spearman)
            atomic_write_json(args.receipt, receipt)
            print(__import__("json").dumps(receipt, allow_nan=False))
            raise SystemExit(0 if receipt["status"] == "pass" else 1)
        except ValueError as exc:
            print(__import__("json").dumps({"ok": False, "error": str(exc), "error_code": getattr(exc, "code", "AUDIT_ERROR")}), file=sys.stderr)
            raise SystemExit(1) from exc

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
        if args.project_root:
            from harness.project import HarnessProject
            active_spec = load_active_task_spec(HarnessProject.discover(args.project_root).task_config_path)
        else:
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
        "audit_version": "3.0_scoped",
        "scope": "validation",
        "dispatch_authorized": False,
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
    print("Validation audit PASSED. Dispatch requires a fresh submission-gate authorization.")
    print("===========================================================")


if __name__ == "__main__":
    main()
