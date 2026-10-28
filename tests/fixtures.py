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


def workspace_with_verified_pair(prefix="aih-policy-"):
    """Disposable sealed classification evidence for research/dispatch invariants."""
    import pandas as pd
    from harness.decision_rules import ResearchPolicy
    from harness.io_utils import atomic_write_json
    doc = task_document()
    doc["task"]["subtasks"] = [{"id": "label", "column": "label", "task_type": "binary_classification",
                              "classes": [0, 1], "primary_metric": "accuracy", "direction": "maximize"}]
    doc["task"]["submission_spec"] = {"required_columns": ["id", "label"]}
    root, config = make_workspace(prefix=prefix, task=doc,
        train_csv="id,text,label\n" + "".join(f"{i},text {i},{i % 2}\n" for i in range(40)))
    pd.DataFrame({"id": range(40, 80), "text": ["test"] * 40}).to_csv(root / "dataset/test.csv", index=False)
    policy = ResearchPolicy(minimum_effect={"value": .05, "scale": "absolute"}, max_comparisons=1,
        bootstrap_resamples=1000, require_confirmation_for_adaptive_search=False)
    atomic_write_json(root / ".ai-harness/policies/research_policy.json", policy.to_dict())
    make_submission_evidence(root, "control", labels=[0] * 40)
    return root, config


def make_submission_evidence(root, exp_id, *, labels=None, cfg=None):
    """Seal genuine plugin-recomputed synthetic OOF and test predictions."""
    import pandas as pd
    from harness.artifacts import contract_identity, digest, seal_file_map, sha256
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.project import HarnessProject
    from harness.task_spec import load_active_task_spec
    project = HarnessProject.discover(root)
    spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(project.root, project.task_config_path, spec, project.plugin_id or "default")
    plugin = load_task_plugin(project.plugin_id, context)
    truth = pd.read_csv(root / "dataset/train.csv")
    labels = list(truth.label) if labels is None else labels
    frame = pd.DataFrame({"id": truth.id, "pred_label": labels, "fold": truth.id % 2, "input_variant": "fixture"})
    cfg = cfg or {"experiment": {"id": exp_id}, "validation": {"partition": "oof"}}
    run_dir = project.artifacts_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    frame.to_csv(run_dir / "oof_predictions.csv", index=False)
    truth.to_csv(run_dir / "evaluation_truth.csv", index=False)
    test_frame = pd.DataFrame({"id": range(40, 80), "pred_label": labels})
    test_frame.to_csv(run_dir / "test_predictions.csv", index=False)
    metrics = plugin.evaluate_predictions(context, truth, {"pred_label": labels})
    metrics.update(experiment_id=exp_id, selection_variant="fixture")
    (run_dir / "metrics.json").write_text(__import__("json").dumps(metrics), encoding="utf-8")
    manifest = dict(schema_version=2, provenance_version=1, provenance="local_run",
        status="COMPLETED", run_kind="full", run_id=exp_id, exp_id=exp_id,
        **contract_identity(spec, plugin), data_sha256=digest(truth.to_dict("list")), split_sha256=digest("fixture-split"),
        resolved_config_sha256=sha256(run_dir / "run_config.yaml"), files=seal_file_map(run_dir))
    (run_dir / "artifact_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    return run_dir
