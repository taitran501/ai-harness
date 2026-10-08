"""Immutable run evidence shared by execution and downstream consumers."""

from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import pandas as pd

REQUIRED = (
    "run_config.yaml",
    "metrics.json",
    "oof_predictions.csv",
    "evaluation_truth.csv",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def contract_identity(task_spec, plugin=None):
    from harness.plugins import TaskPlugin

    scorer = plugin.evaluate_predictions if plugin else TaskPlugin.evaluate_predictions
    if plugin and type(plugin).evaluate_predictions is TaskPlugin.evaluate_predictions:
        scorer = TaskPlugin.evaluate_predictions
    try:
        implementation = inspect.getsource(scorer)
        if scorer is TaskPlugin.evaluate_predictions:
            from harness import evaluation
            from harness import task_spec as task_module

            implementation += inspect.getsource(evaluation) + inspect.getsource(
                task_module
            )
    except (OSError, TypeError):
        raise ValueError("Scorer implementation cannot be identified")
    return {
        "task_sha256": digest(task_spec.metadata),
        "scorer_sha256": digest(implementation),
    }


def validate_run(directory, *, task_spec=None, plugin=None, expected=None):
    directory = Path(directory)
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("LEGACY_UNVERIFIED: missing manifest v2")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "COMPLETED"
        or manifest.get("run_kind") != "full"
    ):
        raise ValueError("Run is not completed, verified full validation")
    for field in (
        "run_id",
        "exp_id",
        "task_sha256",
        "scorer_sha256",
        "data_sha256",
        "split_sha256",
        "resolved_config_sha256",
    ):
        if not manifest.get(field):
            raise ValueError(f"Missing run identity: {field}")
    checks = dict(expected or {})
    if task_spec is not None:
        checks.update(contract_identity(task_spec, plugin))
    for field, value in checks.items():
        if manifest.get(field) != value:
            raise ValueError(f"Run identity mismatch: {field}")
    files = manifest.get("files", {})
    for name in REQUIRED:
        if name not in files:
            raise ValueError(f"Missing required artifact: {name}")
    for name, value in files.items():
        if (
            Path(name).name != name
            or not (directory / name).is_file()
            or sha256(directory / name) != value
        ):
            raise ValueError(f"Artifact hash mismatch: {name}")
    import yaml, importlib

    cfg = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
    hook = cfg.get("paths", {}).get("metrics_evaluator")
    if hook:
        module, name = hook.rsplit(".", 1)
        if manifest.get("custom_scorer_sha256") != digest(
            inspect.getsource(getattr(importlib.import_module(module), name))
        ):
            raise ValueError("Custom scorer implementation changed")
    if sha256(directory / "run_config.yaml") != manifest["resolved_config_sha256"]:
        raise ValueError("Resolved configuration identity mismatch")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    from harness.evaluation import utility

    utility(metrics["primary_score"], metrics["direction"])
    if metrics.get("experiment_id") != manifest["exp_id"]:
        raise ValueError("Metrics experiment identity mismatch")
    return manifest


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def validate_test_prediction(path, *, task_spec, plugin=None):
    path = Path(path).resolve()
    manifest = validate_run(path.parent, task_spec=task_spec, plugin=plugin)
    if (
        path.name != "test_predictions.csv"
        or path.name not in manifest["files"]
        or manifest.get("prediction_splits", {}).get(path.name) != "test"
    ):
        raise ValueError(
            "Submission requires proven test predictions; OOF is validation evidence"
        )
    return manifest
