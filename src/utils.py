"""Utility functions for reproducibility, logging, configuration, and artifact management."""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def set_seed(seed: int = 42) -> None:
    """Set global seed for Python, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash() -> str:
    """Return the current git commit hash and dirty status, or 'unversioned'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            dirty = "-dirty" if status_res.stdout.strip() else ""
            return f"{commit_hash}{dirty}"
    except Exception:
        pass
    return "unversioned"


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_json(path: str | Path, payload: Any) -> None:
    """Save payload as a formatted JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def setup_logger(log_file: Optional[Path | str] = None) -> logging.Logger:
    """Configure console and optional file logging."""
    logger = logging.getLogger("stereoqueer")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def render_experiment_card(
    config: Dict[str, Any],
    summary_df: pd.DataFrame,
    git_hash: str,
) -> str:
    """Render a formal markdown Experiment Card (EXPERIMENT_CARD.md)."""
    exp = config.get("experiment", {})
    exp_id = exp.get("id", "unnamed_experiment")
    branch = exp.get("branch", "general")
    parent_id = exp.get("parent_id", "none")
    date_str = exp.get("date", datetime.date.today().isoformat())
    problem = exp.get("problem_addressed", exp.get("description", "Not specified."))
    hypothesis = exp.get("hypothesis", "Not specified.")
    intervention = exp.get("intervention", "Not specified.")

    lines = [
        f"# Experiment Card: `{exp_id}`",
        "",
        "## 1. Lineage & Metadata",
        f"- **Experiment ID**: `{exp_id}`",
        f"- **Branch / Category**: `{branch}`",
        f"- **Parent Baseline**: `{parent_id}`",
        f"- **Date**: `{date_str}`",
        f"- **Git Commit**: `{git_hash}`",
        f"- **Problem Addressed**: {problem}",
        "",
        "## 2. Hypothesis",
        f"> {hypothesis}",
        "",
        "## 3. Intervention (Control vs Treatment)",
        f"- **Controlled Baseline**: `{parent_id}`",
        f"- **Intervention Applied**: {intervention}",
        "",
        "## 4. Cross-Validation Results",
        "",
        "| Language | Input Variant | A Macro-F1 | B Macro-F1 | C Type Macro-F1 | C Id Macro-F1 | C Id Micro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    metric_cols = [
        "a_macro_f1",
        "b_macro_f1",
        "c_target_type_macro_f1_oracle_hate",
        "c_identity_macro_f1_oracle_hate",
        "c_identity_micro_f1_oracle_hate",
    ]

    if not summary_df.empty:
        if "language" in summary_df.columns and "input_variant" in summary_df.columns and all(c in summary_df.columns for c in metric_cols):
            grouped = (
                summary_df.groupby(["language", "input_variant"], sort=True)[metric_cols]
                .mean()
                .reset_index()
            )
            for _, row in grouped.iterrows():
                lines.append(
                    f"| {row['language']} | {row['input_variant']} | "
                    f"{row['a_macro_f1']:.4f} | {row['b_macro_f1']:.4f} | "
                    f"{row['c_target_type_macro_f1_oracle_hate']:.4f} | "
                    f"{row['c_identity_macro_f1_oracle_hate']:.4f} | "
                    f"{row['c_identity_micro_f1_oracle_hate']:.4f} |"
                )
        else:
            lines.append(summary_df.to_markdown(index=False))

    lines += [
        "",
        "## 5. Post-Mortem & Verdict",
        "",
        "- **Outcome Status**: `[PENDING REVIEW]` (Update to `[CONFIRMED]`, `[REFUTED]`, or `[INCONCLUSIVE]` upon human inspection).",
        "- **Key Finding**: Document whether the intervention improved the target metrics and whether trade-offs occurred.",
        "- **Next Action**: State whether this branch should be promoted to the Hall of Fame, refined, or deprecated into the Idea Graveyard.",
        "",
    ]
    return "\n".join(lines)


def update_experiments_leaderboard(
    leaderboard_path: Path | str,
    new_rows: List[Dict[str, Any]],
) -> None:
    """Append or update experiment rows in the central EXPERIMENTS.md leaderboard."""
    path = Path(leaderboard_path)
    if not path.exists():
        header = (
            "# StereoQueerEval Experiment Leaderboard\n\n"
            "| Exp ID | Branch | Parent | Date | Model / Backbone | Context | EN Macro A | EN Macro B | EN C Type/Id | IT Macro A | NL Macro A | Artifact Path | Notes |\n"
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|\n"
        )
        content = header
    else:
        content = path.read_text(encoding="utf-8")

    formatted_lines = []
    for r in new_rows:
        date_str = r.get("date", datetime.date.today().isoformat())
        exp_id = r.get("exp_id", "")
        branch = r.get("branch", "general")
        parent = r.get("parent_id", "none")
        model = r.get("model", "")
        context = r.get("context", "")
        en_a = f"{r.get('en_macro_a', 0.0):.4f}" if "en_macro_a" in r else "N/A"
        en_b = f"{r.get('en_macro_b', 0.0):.4f}" if "en_macro_b" in r else "N/A"
        en_c = f"{r.get('en_c_type', 0.0):.4f} / {r.get('en_c_id', 0.0):.4f}" if "en_c_type" in r else "N/A"
        it_a = f"{r.get('it_macro_a', 0.0):.4f}" if "it_macro_a" in r else "N/A"
        nl_a = f"{r.get('nl_macro_a', 0.0):.4f}" if "nl_macro_a" in r else "N/A"
        art_path = r.get("artifact_path", f"`artifacts/{exp_id}`")
        notes = r.get("notes", "")

        line = (
            f"| `{exp_id}` | `{branch}` | `{parent}` | {date_str} | {model} | {context} | {en_a} | {en_b} | "
            f"{en_c} | {it_a} | {nl_a} | {art_path} | {notes} |"
        )
        formatted_lines.append(line)

    if formatted_lines:
        new_block = "\n".join(formatted_lines) + "\n"
        if not content.endswith("\n"):
            content += "\n"
        path.write_text(content + new_block, encoding="utf-8")
