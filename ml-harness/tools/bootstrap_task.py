#!/usr/bin/env python3
"""Autonomous Task Bootstrapper and Data Ingestion Engine for ML-Harness.

Enables zero-touch task onboarding:
1. Discovers and auto-extracts data archives (.zip, .tar.gz) in dataset/
2. Inspects arbitrary data formats (.csv, .tsv, .parquet, .feather, .jsonl, .sqlite)
3. Synthesizes task specification from docs/TASK.md or heuristics
4. Generates configs/task.yaml conforming to UniversalTaskSpec
5. Generates leakage-proof validation folds in splits/fold_assignments.tsv
6. Synchronizes the AI Agent skill in .agents/skills/ml-harness/SKILL.md
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


def find_data_files(dataset_dir: Path) -> Dict[str, Optional[Path]]:
    """Scan dataset directory and group candidate files into train, test, and sample submission."""
    extract_archives_in_directory(dataset_dir)

    all_files: List[Path] = []
    exts = (".csv", ".tsv", ".parquet", ".pq", ".feather", ".jsonl", ".json", ".sqlite", ".db")
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                all_files.append(p)

    train_file = None
    test_file = None
    sub_file = None

    for f in all_files:
        name_lower = f.name.lower()
        if "sample_sub" in name_lower or "submission" in name_lower:
            sub_file = f
        elif "test" in name_lower:
            test_file = f
        elif "train" in name_lower:
            train_file = f

    # Fallback if no explicit train file
    if not train_file and all_files:
        non_sub_test = [f for f in all_files if f != sub_file and f != test_file]
        if non_sub_test:
            # Pick largest file as train
            train_file = max(non_sub_test, key=lambda p: p.stat().st_size)

    return {
        "train": train_file,
        "test": test_file,
        "sample_submission": sub_file,
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
            "inputs": [{"name": "features", "type": "text" if schema["modality"] == "text" else "tabular", "source_columns": schema.get("feature_columns", [])}],
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
            "name": "baseline",
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
) -> Dict[str, Any]:
    """End-to-end task onboarding pipeline."""
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT.resolve()
    data_dir = Path(dataset_dir).resolve() if Path(dataset_dir).is_absolute() else (root / dataset_dir).resolve()
    ctx_path = Path(context_file).resolve() if Path(context_file).is_absolute() else (root / context_file).resolve()
    cfg_path = Path(output_config).resolve() if Path(output_config).is_absolute() else (root / output_config).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    # 1. Discover data files & extract archives
    data_files = find_data_files(data_dir)
    train_file = data_files["train"]
    if not train_file or not train_file.exists():
        raise FileNotFoundError(f"No suitable train dataset found in '{data_dir}'.")

    # 2. Extract Context from Markdown
    context = extract_context_from_markdown(ctx_path)
    task_name = context.get("name") or train_file.stem.capitalize()
    platform = context.get("platform") or "kaggle"

    # 3. Load & Analyze Dataset Schema
    df_train = load_dataframe_file(train_file)
    schema = analyze_dataset_schema(df_train, context)
    schema["feature_columns"] = [c for c in df_train if c not in {schema["target_column"], schema["id_column"], schema["group_column"], "fold"}]

    # 4. Generate Task YAML Spec
    try:
        rel_train = str(train_file.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        rel_train = str(train_file).replace("\\", "/")

    task_yaml_data = generate_task_yaml(
        task_name=task_name,
        platform=platform,
        schema=schema,
        train_rel_path=rel_train,
        output_path=cfg_path,
    )

    # 5. Generate Grounded Baseline Config
    baseline_cfg_path = cfg_path.parent / "exp000_baseline.yaml"
    generate_baseline_config(
        task_name=task_name,
        schema=schema,
        train_rel_path=rel_train,
        output_path=baseline_cfg_path,
    )

    # 6. Generate Validation Splits
    splits_dir = cfg_path.parent.parent / "splits" if "configs" in cfg_path.parts else root / "splits"
    splits_path = splits_dir / "fold_assignments.tsv"
    generate_validation_splits(
        df=df_train,
        id_col=schema["id_column"],
        target_col=schema["target_column"],
        group_col=schema["group_column"],
        task_type=schema["task_type"],
        output_path=splits_path,
        num_folds=5,
    )

    # 7. Inject AI Agent Skill
    skill_output = root / ".agents" / "skills" / "ml-harness" / "SKILL.md"
    injected_skill = inject_skill_file(config_path=cfg_path, output_path=skill_output)

    try:
        splits_rel = str(splits_path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        splits_rel = str(splits_path).replace("\\", "/")

    try:
        cfg_rel = str(cfg_path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        cfg_rel = str(cfg_path).replace("\\", "/")

    try:
        baseline_rel = str(baseline_cfg_path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        baseline_rel = str(baseline_cfg_path).replace("\\", "/")

    try:
        skill_rel = str(injected_skill.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        skill_rel = str(injected_skill).replace("\\", "/")

    report = {
        "status": "SUCCESS",
        "task_name": task_name,
        "platform": platform,
        "modality": schema["modality"],
        "task_type": schema["task_type"],
        "primary_metric": schema["primary_metric"],
        "direction": schema["direction"],
        "id_column": schema["id_column"],
        "target_column": schema["target_column"],
        "group_column": schema["group_column"],
        "samples_count": len(df_train),
        "train_file": rel_train,
        "splits_file": splits_rel,
        "config_file": cfg_rel,
        "baseline_config": baseline_rel,
        "skill_file": skill_rel,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap task onboarding and data ingestion for ML-Harness")
    parser.add_argument("--project-root", type=str, default=None, help="Competition workspace root")
    parser.add_argument("--dataset", type=str, default="dataset", help="Directory containing dataset files or archives")
    parser.add_argument("--context", type=str, default="docs/TASK.md", help="Markdown context document describing task")
    parser.add_argument("--output", type=str, default="configs/task.yaml", help="Destination path for generated task.yaml")
    args = parser.parse_args()

    try:
        from harness.project import resolve_project_root

        project_root = Path(args.project_root).expanduser().resolve() if args.project_root else resolve_project_root()
        report = bootstrap_competition_task(
            dataset_dir=args.dataset,
            context_file=args.context,
            output_config=args.output,
            project_root=project_root,
        )
        print("\n✔ Autonomous Task Onboarding Completed Successfully:")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"\n❌ Error during task bootstrapping: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
