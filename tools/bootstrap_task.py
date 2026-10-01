#!/usr/bin/env python3
"""Inventory local competition data and prepare an unverified onboarding draft.

Bootstrap records all discovered files and preserves task context, but it does
not infer a competition contract, assign data roles, generate splits, or create
a baseline. The IDE agent and user review official sources before experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.io_utils import atomic_write_text, atomic_write_json
from harness.skill_injector import inject_skill_file
from harness.task_spec import (
    extract_archives_in_directory,
    load_dataframe_file,
    UniversalTaskSpec,
)


def extract_context_from_markdown(task_md_path: Path) -> Dict[str, Any]:
    """Extract key-value task attributes from a markdown context document."""
    if not task_md_path.exists():
        return {}

    content = task_md_path.read_text(encoding="utf-8")
    extracted: Dict[str, Any] = {}

    patterns = {
        "name": [
            r"\*\*Competition\s*(?:/\s*Benchmark)?\s*Name\*\*:\s*([^\n\r]+)",
            r"#\s*Competition(?:\s*Name)?:\s*([^\n\r]+)",
        ],
        "platform": [
            r"\*\*Platform\*\*:\s*([^\n\r]+)",
        ],
        "modality": [
            r"\*\*Modality\*\*:\s*([^\n\r]+)",
        ],
        "task_type": [
            r"\*\*Task\s*Type\*\*:\s*([^\n\r]+)",
        ],
        "target_columns": [
            r"\*\*Target\s*Columns?\*\*:\s*([^\n\r]+)",
        ],
        "id_column": [
            r"\*\*Primary\s*ID\s*Column\*\*:\s*([^\n\r]+)",
            r"\*\*ID\s*Column\*\*:\s*([^\n\r]+)",
        ],
        "group_column": [
            r"\*\*Anti-Leakage\s*Group\s*Column\*\*:\s*([^\n\r]+)",
            r"\*\*Group\s*Column\*\*:\s*([^\n\r]+)",
        ],
        "primary_metric": [
            r"\*\*Primary\s*Metric\*\*:\s*([^\n\r]+)",
            r"\*\*Evaluation\s*Metric\*\*:\s*([^\n\r]+)",
        ],
        "direction": [
            r"\*\*Optimization\s*Direction\*\*:\s*([^\n\r]+)",
        ],
        "submission_file": [
            r"\*\*Submission\s*File\s*Name\*\*:\s*([^\n\r]+)",
        ],
    }

    for key, regexes in patterns.items():
        for pattern in regexes:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                # Clean brackets or comments
                val = re.sub(r"\[.*?\]", "", val).strip()
                val = val.strip("`'\"")
                if val and val.lower() not in ("none", "n/a", "null", ""):
                    extracted[key] = val
                break

    return extracted


def find_data_files(dataset_dir: Path) -> Dict[str, Any]:
    """Inventory dataset files without guessing which file defines the task."""
    extract_archives_in_directory(dataset_dir)

    all_files: List[Path] = []
    exts = (".csv", ".tsv", ".parquet", ".pq", ".feather", ".jsonl", ".json", ".sqlite", ".db")
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                all_files.append(p)

    all_files = sorted(all_files, key=lambda path: path.as_posix().casefold())
    sample_candidates = [
        path for path in all_files
        if "sample_sub" in path.name.lower() or "submission" in path.name.lower()
    ]
    test_candidates = [
        path for path in all_files
        if path not in sample_candidates and "test" in path.name.lower()
    ]
    train_candidates = [
        path for path in all_files
        if path not in sample_candidates
        and path not in test_candidates
        and "train" in path.name.lower()
    ]

    # Only expose a convenience selection when it is unambiguous.  The complete
    # inventory remains available so multilingual/multi-split tasks are not
    # silently reduced to whichever file happened to be scanned last.
    train_file = train_candidates[0] if len(train_candidates) == 1 else None
    test_file = test_candidates[0] if len(test_candidates) == 1 else None
    sub_file = sample_candidates[0] if len(sample_candidates) == 1 else None

    return {
        "train": train_file,
        "test": test_file,
        "sample_submission": sub_file,
        "train_candidates": train_candidates,
        "test_candidates": test_candidates,
        "sample_submission_candidates": sample_candidates,
        "files": all_files,
    }


def analyze_dataset_schema(df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristically inspect columns to identify ID, target, and modality."""
    cols = list(df.columns)

    # 1. Resolve ID column
    id_col = context.get("id_column")
    if not id_col or id_col not in cols:
        id_candidates = [
            c for c in cols
            if c.lower() in ("id", "sample_id", "row_id", "uid", "uuid", "passengerid")
            or c.lower().endswith("_id")
            or c.lower().endswith("id")
        ]
        # Check uniqueness
        unique_cands = [c for c in id_candidates if df[c].nunique() == len(df)]
        id_col = unique_cands[0] if unique_cands else (id_candidates[0] if id_candidates else cols[0])

    # 2. Resolve Group column (for anti-leakage splits)
    group_col = context.get("group_column")
    if not group_col or group_col not in cols:
        group_candidates = [
            c for c in cols
            if c != id_col and (
                c.lower() in ("group", "group_id", "cluster", "patient_id", "user_id", "session_id", "video_id")
                or "group" in c.lower() or "user" in c.lower()
            )
        ]
        group_col = group_candidates[0] if group_candidates else None

    # 3. Resolve Target column
    target_col = context.get("target_columns")
    if target_col and "," in target_col:
        target_col = [t.strip() for t in target_col.split(",") if t.strip() in cols]
        target_col = target_col[0] if target_col else None

    if not target_col or target_col not in cols:
        target_candidates = [
            c for c in cols
            if c != id_col and c != group_col and (
                c.lower() in ("target", "label", "class", "outcome", "ground_truth", "y", "score", "price")
                or "target" in c.lower() or "label" in c.lower()
            )
        ]
        target_col = target_candidates[0] if target_candidates else cols[-1]

    # 4. Resolve Modality
    modality = context.get("modality")
    if not modality:
        text_cols = []
        for c in cols:
            if c not in (id_col, group_col, target_col) and df[c].dtype == object:
                sample_lens = df[c].dropna().astype(str).str.len()
                if len(sample_lens) > 0 and sample_lens.mean() > 40:
                    text_cols.append(c)
        modality = "text" if text_cols else "tabular"

    # 5. Resolve Task Type & Classes
    task_type = context.get("task_type")
    classes: List[str] = []
    if not task_type:
        y_valid = df[target_col].dropna()
        n_unique = y_valid.nunique()
        if n_unique <= 2:
            task_type = "binary_classification"
            classes = [str(x) for x in sorted(y_valid.unique().tolist())]
        elif n_unique <= 50 and (y_valid.dtype == object or pd.api.types.is_integer_dtype(y_valid)):
            task_type = "multiclass_classification"
            classes = [str(x) for x in sorted(y_valid.unique().tolist())]
        else:
            task_type = "regression"

    # 6. Resolve Primary Metric
    metric = context.get("primary_metric")
    direction = context.get("direction")
    if not metric:
        if task_type == "binary_classification":
            metric = "roc_auc"
            direction = "maximize"
        elif task_type == "multiclass_classification":
            metric = "macro_f1"
            direction = "maximize"
        elif task_type == "regression":
            metric = "rmse"
            direction = "minimize"
        else:
            metric = "macro_f1"
            direction = "maximize"

    if not direction:
        direction = "minimize" if metric.lower() in ("rmse", "mae", "mse", "logloss", "loss") else "maximize"

    return {
        "id_column": id_col,
        "group_column": group_col,
        "target_column": target_col,
        "modality": modality,
        "task_type": task_type,
        "classes": classes,
        "primary_metric": metric,
        "direction": direction,
    }


def generate_task_yaml(
    task_name: str,
    platform: str,
    schema: Dict[str, Any],
    train_rel_path: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Construct UniversalTaskSpec compatible YAML definition."""
    task_dict = {
        "task": {
            "name": task_name,
            "platform": platform,
            "task_type": schema["task_type"],
            "modality": schema["modality"],
            "primary_metric": schema["primary_metric"],
            "direction": schema["direction"],
            "description": f"Autonomously bootstrapped task '{task_name}' on {platform}.",
            "data_config": {
                "train_path": train_rel_path,
                "id_column": schema["id_column"],
                "target_column": schema["target_column"],
            },
            "subtasks": [
                {
                    "id": schema["target_column"],
                    "name": schema["target_column"],
                    "task_type": schema["task_type"],
                    "column": schema["target_column"],
                    "primary_metric": schema["primary_metric"],
                    "direction": schema["direction"],
                    "weight": 1.0,
                    "classes": schema["classes"],
                }
            ],
            "validation": {
                "strategy": "kfold",
                "num_folds": 5,
                "split_file": "splits/fold_assignments.tsv",
            },
        }
    }

    if schema["group_column"]:
        task_dict["task"]["data_config"]["group_column"] = schema["group_column"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(task_dict, f, sort_keys=False)

    return task_dict


def generate_baseline_config(
    task_name: str,
    schema: Dict[str, Any],
    train_rel_path: str,
    output_path: Path,
) -> Path:
    """Generate a GroundingGate-compliant baseline experiment config."""
    is_text = schema["modality"] == "text"
    is_reg = schema["task_type"] == "regression"
    model_name = "tfidf_logistic_regression" if is_text else ("ridge" if is_reg else "logistic_regression")
    model_family = "linear"
    approach_type = "ml"

    exp_payload = {
        "experiment": {
            "id": "exp000_baseline",
            "branch": "baseline",
            "parent_id": "none",
            "problem_addressed": f"Establish official empirical benchmark for {task_name} using standard regularized baseline.",
            "hypothesis": f"A baseline {model_name} pipeline provides a verified lower bound for {schema['primary_metric']}.",
            "intervention": "Default regularized linear baseline with stratified/grouped cross-validation.",
            "description": f"Automated baseline for {task_name}",
        },
        "task": {
            "name": task_name,
            "type": schema["task_type"],
            "primary_metric": schema["primary_metric"],
            "direction": schema["direction"],
            "approach_type": approach_type,
            "model_family": model_family,
        },
        "grounding": {
            "source_type": "verified_baseline",
            "source_id": "canonical_linear_baseline",
            "retrieval_summary": f"Canonical regularized baseline for {schema['modality']} modeling.",
            "verified_via": "local_registry",
        },
        "assumption_register": {
            "known_facts": [
                "Data splits adhere to validation assignments in splits/fold_assignments.tsv.",
            ],
            "assumptions": [
                "Features are linearly or logistically separable with L2 regularization.",
            ],
            "uncertainty_level": "low",
            "verification_probe": "Run smoke test: ml-harness-run --project-root . --config configs/exp000_baseline.yaml --debug --fold 0",
        },
        "constraints": {
            "max_params_limit": 150000000,
        },
        "data": {
            "train_path": train_rel_path,
            "folds": [0, 1, 2, 3, 4],
        },
        "model": {
            "name": model_name,
            "params": {"C": 1.0} if not is_reg else {"alpha": 1.0},
        },
        "training": {
            "seed": 42,
            "debug": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp_payload, f, sort_keys=False)
    return output_path


def generate_validation_splits(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    group_col: Optional[str],
    task_type: str,
    output_path: Path,
    num_folds: int = 5,
) -> Path:
    """Generate leakage-proof K-Fold split assignments."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

    df = df.copy().reset_index(drop=True)
    n_samples = len(df)
    folds = np.zeros(n_samples, dtype=int)

    if n_samples < 2:
        splits_df = pd.DataFrame({id_col: df[id_col], "fold": 0})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        splits_df.to_csv(output_path, sep="\t", index=False)
        return output_path

    if group_col and group_col in df.columns:
        n_groups = df[group_col].nunique()
        actual_folds = min(num_folds, n_groups)
        if actual_folds >= 2:
            gkf = GroupKFold(n_splits=actual_folds)
            groups = df[group_col].fillna("unknown").astype(str)
            for fold, (_, val_idx) in enumerate(gkf.split(df, groups=groups)):
                folds[val_idx] = fold
    elif task_type in ("binary_classification", "multiclass_classification"):
        y = df[target_col].fillna("missing").astype(str)
        min_class_count = int(y.value_counts().min()) if len(y.value_counts()) > 0 else 0
        actual_folds = min(num_folds, min_class_count)
        if actual_folds >= 2:
            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            for fold, (_, val_idx) in enumerate(skf.split(df, y=y)):
                folds[val_idx] = fold
        else:
            actual_folds = min(num_folds, n_samples)
            if actual_folds >= 2:
                kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
                for fold, (_, val_idx) in enumerate(kf.split(df)):
                    folds[val_idx] = fold
    else:
        actual_folds = min(num_folds, n_samples)
        if actual_folds >= 2:
            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
            for fold, (_, val_idx) in enumerate(kf.split(df)):
                folds[val_idx] = fold

    splits_df = pd.DataFrame({
        id_col: df[id_col],
        "fold": folds,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, sep="\t", index=False)
    return output_path


def bootstrap_competition_task(
    dataset_dir: Path | str = "dataset",
    context_file: Path | str = "docs/TASK.md",
    output_config: Path | str = "configs/task.yaml",
    project_root: Optional[Path | str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a reviewable task draft and inventory without inventing a contract.

    Dataset names and observed columns are not sufficient evidence for a metric,
    task type, split policy, platform, or submission contract.  Bootstrap leaves
    those fields explicitly unconfirmed and never creates a baseline artifact.
    """
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT.resolve()
    data_dir = Path(dataset_dir).resolve() if Path(dataset_dir).is_absolute() else (root / dataset_dir).resolve()
    ctx_path = Path(context_file).resolve() if Path(context_file).is_absolute() else (root / context_file).resolve()
    cfg_path = Path(output_config).resolve() if Path(output_config).is_absolute() else (root / output_config).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    if cfg_path.exists() and not overwrite:
        raise FileExistsError(
            f"Task config already exists: {cfg_path}. Review it first or pass --force to replace it with an onboarding draft."
        )

    # 1. Inventory every local data file. Do not choose a single language/split
    # merely because its name sorts last.
    data_files = find_data_files(data_dir)
    files: List[Path] = data_files["files"]
    if not files:
        raise FileNotFoundError(f"No supported dataset files found in '{data_dir}'.")

    # 2. Preserve the supplied task brief as prompt context. If it is outside
    # the workspace, copy it into the task-owned context directory so rendered
    # prompts remain workspace-relative and portable.
    context = extract_context_from_markdown(ctx_path)
    try:
        context_rel = ctx_path.relative_to(root).as_posix() if ctx_path.is_file() else None
    except ValueError:
        context_dir = root / ".ml-harness" / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        context_copy = context_dir / "TASK.md"
        context_copy.write_text(ctx_path.read_text(encoding="utf-8"), encoding="utf-8")
        context_rel = context_copy.relative_to(root).as_posix()

    def relative_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path.resolve())

    inventory = [
        {
            "path": relative_path(path),
            "role": "unclassified",
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    task_name = context.get("name") or "Task contract pending review"
    explicit_targets = [
        item.strip() for item in str(context.get("target_columns") or "").split(",") if item.strip()
    ]
    data_config: Dict[str, Any] = {
        "candidate_files": inventory,
        "target_columns": explicit_targets,
    }
    for key in ("id_column", "group_column"):
        if context.get(key):
            data_config[key] = context[key]
    if len(explicit_targets) == 1:
        data_config["target_column"] = explicit_targets[0]

    task_data: Dict[str, Any] = {
        "name": task_name,
        "task_name": task_name,
        "platform": context.get("platform") or "unconfirmed",
        "task_type": context.get("task_type") or "unconfirmed",
        "type": context.get("task_type") or "unconfirmed",
        "modality": context.get("modality") or "unconfirmed",
        "primary_metric": context.get("primary_metric") or "unconfirmed",
        "direction": context.get("direction") or "unconfirmed",
        "contract_status": "needs_review",
        "description": "Task details are supplied as context and must be checked against official competition sources before execution.",
        "data_config": data_config,
        "subtasks": [],
        "competition_rules": {},
        "submission_spec": {},
        "validation": {"strategy": "needs_review"},
        "competition_context_files": [context_rel] if context_rel else [],
        "contract_sources": ([{"path": context_rel, "status": "user_supplied_context"}] if context_rel else []),
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"task": task_data}, stream, sort_keys=False, allow_unicode=True)

    onboarding_path = root / ".ml-harness" / "onboarding.json"
    atomic_write_json(onboarding_path, {
        "schema_version": 1,
        "status": "NEEDS_REVIEW",
        "task_config": relative_path(cfg_path),
        "context_file": context_rel,
        "contract_status": "needs_review",
        "candidate_files": inventory,
        "pending_contract_fields": [
            field for field, value in (
                ("platform", task_data["platform"]),
                ("task_type", task_data["task_type"]),
                ("modality", task_data["modality"]),
                ("primary_metric", task_data["primary_metric"]),
                ("direction", task_data["direction"]),
                ("validation", "unconfirmed"),
                ("submission_spec", "unconfirmed"),
            ) if value == "unconfirmed" or field in {"validation", "submission_spec"}
        ],
        "next_action": "Review the task brief and verify the official metric, rules, validation, and submission contract.",
    })

    # 3. Inject a setup-first skill. It will explicitly state that this config
    # is a draft and that no baseline or split has been established.
    skill_output = root / ".agents" / "skills" / "ml-harness" / "SKILL.md"
    injected_skill = inject_skill_file(config_path=cfg_path, output_path=skill_output)

    pending_fields = [
        field for field, value in (
            ("platform", task_data["platform"]),
            ("task_type", task_data["task_type"]),
            ("modality", task_data["modality"]),
            ("primary_metric", task_data["primary_metric"]),
            ("direction", task_data["direction"]),
        ) if value == "unconfirmed"
    ]
    report = {
        "status": "NEEDS_REVIEW",
        "task_name": task_name,
        "contract_status": "needs_review",
        "platform": task_data["platform"],
        "modality": task_data["modality"],
        "task_type": task_data["task_type"],
        "primary_metric": task_data["primary_metric"],
        "direction": task_data["direction"],
        "candidate_file_count": len(inventory),
        "candidate_files": inventory,
        "pending_contract_fields": pending_fields,
        "context_file": context_rel,
        "config_file": relative_path(cfg_path),
        "onboarding_file": relative_path(onboarding_path),
        "skill_file": relative_path(injected_skill),
        "baseline_config": None,
        "splits_file": None,
        "next_action": "Review the task brief and verify the official contract before creating splits or running a baseline.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap task onboarding and data ingestion for ML-Harness")
    parser.add_argument("--project-root", type=str, default=None, help="Competition workspace root")
    parser.add_argument("--dataset", type=str, default="dataset", help="Directory containing dataset files or archives")
    parser.add_argument("--context", type=str, default="docs/TASK.md", help="Markdown context document describing task")
    parser.add_argument("--output", type=str, default="configs/task.yaml", help="Destination path for generated task.yaml")
    parser.add_argument("--force", action="store_true", help="Replace an existing task config with an unverified onboarding draft")
    args = parser.parse_args()

    try:
        from harness.project import resolve_project_root

        project_root = Path(args.project_root).expanduser().resolve() if args.project_root else resolve_project_root()
        report = bootstrap_competition_task(
            dataset_dir=args.dataset,
            context_file=args.context,
            output_config=args.output,
            project_root=project_root,
            overwrite=args.force,
        )
        print("\nTask workspace inventoried. Contract review is required before setup is complete:")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"\n❌ Error during task bootstrapping: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
