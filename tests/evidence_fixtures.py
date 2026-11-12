"""Explicit synthetic evidence for isolated tests; never upgrades workspace history."""

import json
from pathlib import Path

from harness.artifacts import contract_identity, digest, sha256
from harness.evaluation import utility



def seal_fixture(folder, spec, score=0.75, **identity):
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 2,
        "provenance_version": 1,
        "provenance": "local_run",
        "experiment_id": folder.name,
        "primary_metric": spec.primary_metric,
        "primary_score": score,
        "direction": spec.direction,
        "utility": score if spec.direction == "maximize" else -score,
        "n_samples": 2,
        "selection_variant": "features",
        spec.primary_metric: score,
    }
    (folder / "metrics.json").write_text(json.dumps(metrics))
    for name, contents in {
        "run_config.yaml": "experiment: {}\n",
        "oof_predictions.csv": "id,pred_target\n1,0\n2,1\n",
        "evaluation_truth.csv": "id,target\n1,0\n2,1\n",
    }.items():
        if not (folder / name).exists():
            (folder / name).write_text(contents)
    manifest = {
        "schema_version": 2,
        "provenance_version": 1,
        "provenance": "local_run",
        "run_id": "test-" + folder.name,
        "exp_id": folder.name,
        "run_kind": "full",
        "status": "COMPLETED",
        **contract_identity(spec),
        "data_sha256": digest("fixture"),
        "split_sha256": digest("fixture-split"),
        "resolved_config_sha256": sha256(folder / "run_config.yaml"),
        "files": {
            p.name: sha256(p)
            for p in folder.iterdir()
            if p.is_file() and p.name != "artifact_manifest.json"
        },
        **identity,
    }
    (folder / "artifact_manifest.json").write_text(json.dumps(manifest))
    return metrics


def seal_into(folder: Path, spec, score: float = 0.8) -> Path:
    """Write complete mock predictions, truth, config, and sealed manifest."""
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 2,
        "experiment_id": folder.name,
        "primary_metric": spec.primary_metric,
        "primary_score": score,
        "direction": spec.direction,
        "utility": utility(score, spec.direction),
        "n_samples": 2,
        "selection_variant": "features",
        spec.primary_metric: score,
    }
    (folder / "metrics.json").write_text(json.dumps(metrics))
    for name, contents in {
        "run_config.yaml": "experiment: {}\n",
        "oof_predictions.csv": "id,pred_target\n1,0\n2,1\n",
        "evaluation_truth.csv": "id,target\n1,0\n2,1\n",
    }.items():
        (folder / name).write_text(contents)
    seal_fixture(folder, spec, score)
    return folder


def drift_metrics(folder: Path, spec, score: float = 0.85) -> None:
    """Rewrite metrics.json with modified primary score to simulate artifact drift."""
    metrics = {
        "schema_version": 2,
        "experiment_id": folder.name,
        "primary_metric": spec.primary_metric,
        "primary_score": score,
        "direction": spec.direction,
        "utility": utility(score, spec.direction),
        "n_samples": 2,
        "selection_variant": "features",
        spec.primary_metric: score,
    }
    (folder / "metrics.json").write_text(json.dumps(metrics))


def make_project(tmp):
    """Create a minimal attached project with an artifacts dir and return (root, spec, artifacts)."""
    import yaml
    from harness.task_spec import UniversalTaskSpec
    from tests.fixtures import text_task_document

    root = Path(tmp)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    doc = text_task_document()
    (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    spec = UniversalTaskSpec.from_dict(doc)
    manifest = root / ".ml-harness"
    manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "project.yaml").write_text(
        yaml.safe_dump({"task_config": "configs/task.yaml", "artifacts_dir": "artifacts"}),
        encoding="utf-8",
    )
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return root, spec, artifacts


_seal_into = seal_into
_drift_metrics = drift_metrics
_make_project = make_project

