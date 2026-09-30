"""Atomic File I/O Utilities for ML-Harness.

Prevents file corruption and race conditions when multiple processes (CLI,
web dashboard, background runners) read and write shared JSON and text states.
Uses same-directory temporary files combined with atomic `os.replace()`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_write_text(
    path: Path | str,
    text: str,
    encoding: str = "utf-8",
) -> None:
    """Atomically write text to disk using a temporary file and os.replace()."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file must be created on the same filesystem/directory for atomic rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.tmp_",
        text=True,
    )
    try:
        with open(tmp_fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_path, str(target))
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_json(
    path: Path | str,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Atomically serialize data to a JSON file on disk."""
    dumped = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(path, dumped, encoding=encoding)


def get_project_root() -> Path:
    """Finds the root directory of the active project or competition workspace.

    Shared resolver honors an explicit root or an attached workspace manifest.
    """
    from harness.project import resolve_project_root

    return resolve_project_root()


def set_seed(seed: int = 42) -> None:
    """Set global seed for Python, NumPy, and PyTorch (if installed)."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash(cwd: Optional[Path | str] = None) -> str:
    """Return the current git commit hash and dirty status, or 'unversioned'."""
    import subprocess
    work_dir = str(cwd) if cwd else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=work_dir,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
                cwd=work_dir,
            )
            dirty = "-dirty" if status_res.stdout.strip() else ""
            return f"{commit_hash}{dirty}"
    except Exception:
        pass
    return "unversioned"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    import yaml
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_json(path: str | Path, payload: Any) -> None:
    """Save payload as a formatted JSON file using atomic write."""
    atomic_write_json(path, payload, indent=2, ensure_ascii=False)


def setup_logger(
    log_file_or_name: Optional[Union[Path, str]] = None,
    log_file: Optional[Union[Path, str]] = None,
    name: str = "harness",
) -> Any:
    """Configure console and optional file logging."""
    import logging
    import sys

    actual_name = name
    actual_file = log_file

    if log_file_or_name is not None:
        if isinstance(log_file_or_name, Path) or (
            isinstance(log_file_or_name, str)
            and ("/" in log_file_or_name or "\\" in log_file_or_name or log_file_or_name.endswith(".log") or log_file_or_name.endswith(".txt"))
        ):
            actual_file = log_file_or_name
        else:
            actual_name = str(log_file_or_name)

    logger = logging.getLogger(actual_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if actual_file:
        log_path = Path(actual_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def render_experiment_card(
    config: dict[str, Any],
    summary_df: Any,
    git_hash: str,
    task_spec: Optional[Any] = None,
) -> str:
    """Render a formal, task-agnostic markdown Experiment Card (EXPERIMENT_CARD.md)."""
    import datetime
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
    ]

    if summary_df is not None and hasattr(summary_df, "empty") and not summary_df.empty:
        try:
            lines.append(summary_df.to_markdown(index=False))
        except Exception:
            lines.append(str(summary_df))
    else:
        lines.append("No numerical cross-validation summary table available.")

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
    new_rows: list[dict[str, Any]],
    task_spec: Optional[Any] = None,
) -> None:
    """Append or update experiment rows in the central EXPERIMENTS.md leaderboard."""
    import datetime
    path = Path(leaderboard_path)
    task_name = getattr(task_spec, "name", "Experiment") if task_spec else "Experiment"
    primary_metric_name = getattr(task_spec, "primary_metric", "Score") if task_spec else "Score"

    if not path.exists():
        header = (
            f"# {task_name} Leaderboard\n\n"
            f"| Exp ID | Branch | Parent | Date | Model | Context | {primary_metric_name} | Artifact Path | Notes |\n"
            "|---|---|---|---|---|---|---:|---|---|\n"
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
        score_val = r.get("score") or r.get("primary_metric") or r.get("composite_score") or r.get("metric_value")
        score_str = f"{float(score_val):.4f}" if score_val is not None else "N/A"
        art_path = r.get("artifact_path", f"`artifacts/{exp_id}`")
        notes = r.get("notes", "")

        line = (
            f"| `{exp_id}` | `{branch}` | `{parent}` | {date_str} | {model} | {context} | {score_str} | {art_path} | {notes} |"
        )
        formatted_lines.append(line)

    if formatted_lines:
        new_block = "\n".join(formatted_lines) + "\n"
        if not content.endswith("\n"):
            content += "\n"
        atomic_write_text(path, content + new_block)

