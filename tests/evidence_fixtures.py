"""Explicit synthetic evidence for isolated tests; never upgrades workspace history."""

import json
from harness.artifacts import contract_identity, sha256, digest


def seal_fixture(folder, spec, score=0.75, **identity):
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 2,
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
