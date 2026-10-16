"""Shared test fixtures for the ai-harness test suite.

Centralizes the two workspace patterns that used to be copy-pasted across
test files:

1. ``make_workspace`` — a minimal synthetic competition workspace
   (``configs/task.yaml`` + ``dataset/train.csv`` + attached project).
2. ``workspace_with_repo_configs`` — a temp workspace containing the repo's
   real ``configs/`` directory (needed by agent-bridge tests that rely on
   the attached task spec).

Keep this module dependency-light: it must import as fast as the test
modules that use it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from harness.project import attach_project

SOURCE_ROOT = Path(__file__).resolve().parent.parent

#: Tiny text-classification dataset used by most bridge/runner fixtures.
TRAIN_CSV_4ROW = (
    "id,text,label\n"
    "0,nice dog,0\n"
    "1,kind dog,0\n"
    "2,awful cat,1\n"
    "3,mean cat,1\n"
)


def task_document(
    task_id: str = "fixture-task",
    name: str = "Fixture Task",
    primary_metric: str = "accuracy",
    direction: str = "maximize",
    task_type: str = "classification",
    num_folds: int = 2,
    target_column: str = "label",
    train_path: str = "dataset/train.csv",
) -> Dict[str, Any]:
    """Build a minimal task.yaml document (classification, kfold)."""
    return {
        "task": {
            "id": task_id,
            "name": name,
            "task_type": task_type,
            "type": task_type,
            "primary_metric": primary_metric,
            "direction": direction,
            "data_config": {
                "train_path": train_path,
                "target_column": target_column,
            },
            "validation": {"strategy": "kfold", "num_folds": num_folds},
        }
    }


def make_workspace(
    prefix: str = "aih-fix-",
    with_train_file: bool = True,
    train_csv: str = TRAIN_CSV_4ROW,
    task: Optional[Dict[str, Any]] = None,
    train_path: str = "dataset/train.csv",
) -> Tuple[Path, Path]:
    """Create a temp workspace: ``configs/task.yaml`` + dataset + attach_project.

    Returns ``(root, task_config_path)``. Callers own cleanup of ``root``.
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    (root / "configs").mkdir()
    (root / "dataset").mkdir()
    if with_train_file:
        (root / "dataset" / "train.csv").write_text(train_csv, encoding="utf-8")
    doc = task if task is not None else task_document(train_path=train_path)
    task_config = root / "configs" / "task.yaml"
    task_config.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    attach_project(root, task_config)
    return root, task_config


def workspace_with_repo_configs(prefix: str = "aih-repocfg-") -> Path:
    """Temp workspace with the repo's real ``configs/`` directory copied in.

    Used by agent-bridge tests that need the genuinely attached task spec.
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    shutil.copytree(SOURCE_ROOT / "configs", root / "configs")
    return root


def copy_repo_configs(dest_root: Path) -> Path:
    """Copy the repo's real ``configs/`` into an existing temp workspace root."""
    dest = Path(dest_root) / "configs"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_ROOT / "configs", dest, dirs_exist_ok=True)
    return dest
