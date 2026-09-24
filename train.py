#!/usr/bin/env python3
"""Unified training and evaluation CLI runner for StereoQueerEval experiments."""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import sklearn
import yaml

from src.data import IDENTITIES, load_dataset, load_splits, merge_data_and_splits
from src.metrics import evaluate_predictions, format_cv_summary_table
from src.models import build_model
from src.utils import (
    get_git_commit_hash,
    load_yaml_config,
    render_experiment_card,
    save_json,
    set_seed,
    setup_logger,
    update_experiments_leaderboard,
)
from harness.validator import ExperimentValidator
from harness.tree_journal import Node, TreeJournal
from harness.diagnostics import DiagnosticEngine
from harness.grounding_gate import GroundingGate, EpistemicGroundingError


def merge_configs(base: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge experiment configuration over base configuration."""
    merged = base.copy()
    for k, v in exp.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def run_experiment(args: argparse.Namespace) -> None:
    # 1. Load Configurations
    base_cfg = load_yaml_config(args.base_config) if Path(args.base_config).exists() else {}
    exp_cfg = load_yaml_config(args.config)
    cfg = merge_configs(base_cfg, exp_cfg)

    # 2. Epistemic Grounding & Assumption Register Pre-Flight Gate (Anti-Outdated & Anti-Hallucination)
    passed, grounding_errors, receipt = GroundingGate.verify_config(
        cfg, config_path=args.config, raise_on_error=True
    )

    exp_info = cfg.get("experiment", {})
    exp_id = exp_info.get("id", Path(args.config).stem)
    paths = cfg.get("paths", {})
    artifacts_root = Path(paths.get("artifacts_dir", "artifacts"))
    exp_dir = artifacts_root / exp_id

    # 3. Immutability Guard
    if exp_dir.exists() and (exp_dir / "metrics.json").exists() and not args.force:
        print(
            f"ERROR: Experiment directory '{exp_dir}' already exists with completed metrics.\n"
            f"Experiments are immutable by contract. Please define a new experiment ID or pass --force.",
            file=sys.stderr,
        )
        sys.exit(1)

    exp_dir.mkdir(parents=True, exist_ok=True)
    GroundingGate.write_receipt(receipt, exp_dir / "GROUNDING_RECEIPT.json")

    # 4. Setup Logging
    log_file = exp_dir / "run.log"
    logger = setup_logger(log_file)
    logger.info("=" * 60)
    logger.info(f"Starting Experiment: {exp_id}")
    logger.info(f"Config path: {args.config}")
    logger.info("=" * 60)

    # 4. Environment & Reproducibility Metadata
    seed = cfg.get("runtime", {}).get("seed", 42)
    set_seed(seed)
    git_hash = get_git_commit_hash()
    logger.info(f"Git commit hash: {git_hash}")
    logger.info(f"Random seed: {seed}")

    runtime_meta = {
        "git_commit_hash": git_hash,
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    cfg["resolved_runtime"] = runtime_meta
    with open(exp_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # 5. Load Dataset and Splits
    data_dir = paths.get("data_dir", "dataset")
    split_path = paths.get("split_path", "splits/fold_assignments.tsv")
    logger.info(f"Loading datasets from '{data_dir}'...")
    df, checksums = load_dataset(data_dir)
    logger.info(f"Loaded {len(df):,} total samples across languages.")

    logger.info(f"Loading split assignments from '{split_path}'...")
    splits_df = load_splits(split_path)
    merged_df = merge_data_and_splits(df, splits_df)

    if args.debug:
        logger.warning("Debug mode enabled: sampling 50 instances per language.")
        merged_df = merged_df.groupby("language").head(50).reset_index(drop=True)

    # 6. Folds and Input Variants
    target_folds = [args.fold] if args.fold is not None else cfg.get("data", {}).get("folds", [0, 1, 2, 3, 4])
    input_variants = cfg.get("data", {}).get("input_variants", ["comment"])
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "tfidf_logistic_regression")
    model_params = model_cfg.get("params", {})

    logger.info(f"Target folds: {target_folds}")
    logger.info(f"Input variants: {input_variants}")
    logger.info(f"Model architecture: {model_name}")

    detailed_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    oof_records: List[Dict[str, Any]] = []

    # 7. Training & Evaluation Loop
    for dev_fold in target_folds:
        logger.info(f"\n--- Processing Fold {dev_fold} ---")
        for language, lang_df in merged_df.groupby("language", sort=True):
            train_part = lang_df[lang_df["fold"] != dev_fold].reset_index(drop=True)
            dev_part = lang_df[lang_df["fold"] == dev_fold].reset_index(drop=True)

            if len(dev_part) == 0:
                continue

            for variant in input_variants:
                # Instantiate model
                model = build_model(model_name, **model_params)

                # Physical Pre-flight Checks (Enforced by Code, not Docs)
                max_params = cfg.get("constraints", {}).get("max_params_limit", 150_000_000)
                param_count = ExperimentValidator.assert_parameter_limit(model, max_params=max_params)
                ExperimentValidator.assert_no_video_data_leakage(train_part, dev_part, group_column="yt_title")

                model.fit(train_part, variant=variant)

                # Predict
                preds = model.predict(dev_part, variant=variant)

                # Store OOF Predictions
                for i, row_id in enumerate(dev_part["StereoQueerEval_id"]):
                    id_flags = [
                        IDENTITIES[j]
                        for j in range(len(IDENTITIES))
                        if preds["pred_identities"][i, j] == 1
                    ]
                    oof_records.append(
                        {
                            "StereoQueerEval_id": row_id,
                            "language": language,
                            "fold": dev_fold,
                            "input_variant": variant,
                            "pred_stereotype": preds["pred_stereotype"][i],
                            "pred_hate_speech": preds["pred_hate_speech"][i],
                            "pred_target_type": preds["pred_target_type"][i],
                            "pred_identities": ",".join(id_flags) if id_flags else "none",
                        }
                    )

                # Evaluate Metrics
                eval_metrics = evaluate_predictions(
                    dev_part,
                    preds["pred_stereotype"],
                    preds["pred_hate_speech"],
                    preds["pred_target_type"],
                    preds["pred_identities"],
                )

                metric_record = {
                    "experiment_id": exp_id,
                    "language": language,
                    "input_variant": variant,
                    "dev_fold": dev_fold,
                    "train_rows": len(train_part),
                    "dev_rows": len(dev_part),
                    **eval_metrics,
                }
                detailed_results.append(metric_record)

                summary_rows.append(
                    {
                        "language": language,
                        "input_variant": variant,
                        "dev_fold": dev_fold,
                        "train_rows": len(train_part),
                        "dev_rows": len(dev_part),
                        "a_macro_f1": eval_metrics["subtask_a"]["macro_f1"],
                        "b_macro_f1": eval_metrics["subtask_b"]["macro_f1"],
                        "c_target_type_macro_f1_oracle_hate": eval_metrics[
                            "subtask_c_target_type_oracle_hate"
                        ]["macro_f1"],
                        "c_identity_macro_f1_oracle_hate": eval_metrics[
                            "subtask_c_identity_oracle_hate"
                        ]["macro_f1"],
                        "c_identity_micro_f1_oracle_hate": eval_metrics[
                            "subtask_c_identity_oracle_hate"
                        ]["micro_f1"],
                    }
                )

                logger.info(
                    f"[{language}] {variant:13} (Fold {dev_fold}) | "
                    f"A-F1: {eval_metrics['subtask_a']['macro_f1']:.4f} | "
                    f"B-F1: {eval_metrics['subtask_b']['macro_f1']:.4f} | "
                    f"C-Type: {eval_metrics['subtask_c_target_type_oracle_hate']['macro_f1']:.4f} | "
                    f"C-Id: {eval_metrics['subtask_c_identity_oracle_hate']['macro_f1']:.4f}"
                )

    # 8. Save Artifacts
    logger.info("\nSaving experiment artifacts...")
    save_json(exp_dir / "metrics.json", detailed_results)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(exp_dir / "summary.csv", index=False)

    oof_df = pd.DataFrame(oof_records)
    oof_df.to_csv(exp_dir / "oof_predictions.csv", index=False)

    summary_md = format_cv_summary_table(summary_df)
    (exp_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    card_md = render_experiment_card(cfg, summary_df, git_hash)
    (exp_dir / "EXPERIMENT_CARD.md").write_text(card_md, encoding="utf-8")

    # Generate Diagnostics & Failure Mode Slices
    logger.info("Computing failure diagnostics & error slices...")
    diag_summary = DiagnosticEngine.save_diagnostics_bundle(
        eval_df=merged_df,
        oof_df=oof_df,
        output_dir=exp_dir,
        exp_id=exp_id,
    )
    logger.info(f"Diagnostics written to: {exp_dir / 'DIAGNOSTICS.md'}")
    if diag_summary.get("critical_bottlenecks"):
        logger.warning("Critical bottlenecks detected:")
        for b in diag_summary["critical_bottlenecks"]:
            logger.warning(f"  - {b}")

    logger.info(f"Artifacts successfully written to: {exp_dir}")

    # 9. Update EXPERIMENTS.md Leaderboard
    if not args.skip_leaderboard and len(target_folds) == 5:
        logger.info("Updating central experiment leaderboard...")
        leaderboard_path = paths.get("leaderboard_path", "EXPERIMENTS.md")
        agg = summary_df.groupby(["language", "input_variant"])[
            ["a_macro_f1", "b_macro_f1", "c_target_type_macro_f1_oracle_hate", "c_identity_macro_f1_oracle_hate"]
        ].mean()

        leaderboard_rows = []
        branch = exp_info.get("branch", "general")
        parent_id = exp_info.get("parent_id", "none")
        for variant in input_variants:
            try:
                en_a = agg.loc[("EN", variant), "a_macro_f1"]
                en_b = agg.loc[("EN", variant), "b_macro_f1"]
                en_c_t = agg.loc[("EN", variant), "c_target_type_macro_f1_oracle_hate"]
                en_c_i = agg.loc[("EN", variant), "c_identity_macro_f1_oracle_hate"]
                it_a = agg.loc[("IT", variant), "a_macro_f1"]
                nl_a = agg.loc[("NL", variant), "a_macro_f1"]
            except KeyError:
                continue

            leaderboard_rows.append(
                {
                    "exp_id": f"{exp_id}_{variant}",
                    "branch": branch,
                    "parent_id": parent_id,
                    "date": exp_info.get("date", datetime.date.today().isoformat()),
                    "model": model_name,
                    "context": variant,
                    "en_macro_a": en_a,
                    "en_macro_b": en_b,
                    "en_c_type": en_c_t,
                    "en_c_id": en_c_i,
                    "it_macro_a": it_a,
                    "nl_macro_a": nl_a,
                    "artifact_path": f"`artifacts/{exp_id}`",
                    "notes": exp_info.get("description", ""),
                }
            )

        if leaderboard_rows:
            update_experiments_leaderboard(leaderboard_path, leaderboard_rows)
            logger.info("Leaderboard updated successfully.")

    # 10. Automatic TreeJournal Registration (AIDE-style Solution Tree)
    if not args.skip_leaderboard and len(target_folds) == 5:
        logger.info("Automatically recording experiment in TreeJournal...")
        journal = TreeJournal()
        overall_f1 = float(summary_df[["a_macro_f1", "b_macro_f1", "c_target_type_macro_f1_oracle_hate"]].mean().mean())
        node = Node(
            id=exp_id,
            parent_id=exp_info.get("parent_id") if exp_info.get("parent_id") != "none" else None,
            branch=exp_info.get("branch", "baseline"),
            atomic_hypothesis=exp_info.get("hypothesis", ""),
            intervention=exp_info.get("intervention", ""),
            metrics={
                "overall_composite": round(overall_f1, 4),
                "model": model_name,
                "subtask_a": {"macro_f1": round(float(summary_df["a_macro_f1"].mean()), 4)},
                "subtask_b": {"macro_f1": round(float(summary_df["b_macro_f1"].mean()), 4)},
                "subtask_c": {
                    "type_macro_f1": round(float(summary_df["c_target_type_macro_f1_oracle_hate"].mean()), 4),
                    "id_macro_f1": round(float(summary_df["c_identity_macro_f1_oracle_hate"].mean()), 4),
                },
            },
            param_count=param_count if "param_count" in locals() else None,
            commit_hash=git_hash,
            status="COMPLETED",
            artifact_dir=str(exp_dir),
            notes=exp_info.get("description", ""),
            diagnostics_summary=diag_summary if "diag_summary" in locals() else {},
        )
        journal.append(node)
        logger.info(f"TreeJournal successfully updated with node: '{node.id}'")

    logger.info("Experiment run finished.")


def main():
    parser = argparse.ArgumentParser(description="StereoQueerEval experiment runner.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration.")
    parser.add_argument("--base-config", default="configs/base.yaml", help="Path to base configuration.")
    parser.add_argument("--fold", type=int, default=None, help="Run only on a specific fold (0-4).")
    parser.add_argument("--debug", action="store_true", help="Debug smoke test on small subset.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing experiment artifacts.")
    parser.add_argument("--skip-leaderboard", action="store_true", help="Skip updating EXPERIMENTS.md.")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
