"""Backend model-resource planning, verification, and evidence preservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
from unittest.mock import patch

import yaml

from harness.model_assets import BackendResourcePlan, ModelResolver
from harness.project import attach_project
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import task_document


def _model_plan(*, provider: str = "huggingface", source_uri: str | None = None):
    definition = {
        "schema_version": 1,
        "model_id": "fixture-model",
        "kind": "pretrained",
        "family": "fixture backbone",
        "task_types": ["classification"],
        "modalities": ["text"],
        "source": {
            "provider": provider,
            "repository": "example/fixture-model",
            "revision": "0123456789abcdef",
            "source_uri": source_uri or "hf://example/fixture-model@0123456789abcdef",
        },
        "artifacts": {
            "manifest_ref": "fixture-manifest",
            "bundle_digest": "a" * 64,
        },
        "parameter_facts": {"exact_param_count": 10, "count_basis": "logical_total"},
        "release": {"public_release_date": None},
        "license": {"expression": "MIT"},
        "parents": [],
        "verification_refs": [],
    }
    root = {
        "slot": "predictor",
        "role": "predictor",
        "model_id": "fixture-model",
        "definition_digest": "b" * 64,
        "artifact_bundle_digest": "a" * 64,
        "definition": definition,
    }
    return SimpleNamespace(model_lock={"schema_version": 1, "task_id": "fixture-task", "roots": [root]})


def _manifest_for(contents: bytes = b"fixture model weights"):
    return {
        "schema_version": 1,
        "files": [
            {
                "relative_path": "weights/model.safetensors",
                "role": "weights",
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        ],
    }


def _scope(root: Path, **overrides):
    value = {
        "root": root,
        "offline": True,
        "model_manifests": {"fixture-manifest": _manifest_for()},
    }
    value.update(overrides)
    return value


def _modal_workspace(root: Path) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "dataset").mkdir(exist_ok=True)
    (root / "dataset" / "train.csv").write_text("id,val\n1,2\n", encoding="utf-8")
    task_path = root / "configs" / "task.yaml"
    task_path.write_text(yaml.safe_dump(task_document(), sort_keys=False), encoding="utf-8")
    attach_project(root, task_path, force=True)
    config_path = root / "configs" / "exp001.yaml"
    config_path.write_text("experiment:\n  id: exp001\n", encoding="utf-8")
    return config_path


def test_local_resource_plan_resolves_explicit_root_and_verifies_manifest_files(tmp_path: Path):
    model_root = tmp_path / "assets" / "fixture-model"
    model_file = model_root / "weights" / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"fixture model weights")
    resolver = ModelResolver()

    plan = resolver.plan_backend_resources(
        _model_plan(),
        "local",
        _scope(
            tmp_path,
            model_locations={"predictor": {"path": "assets/fixture-model"}},
        ),
    )

    assert plan.backend == "local"
    assert plan.mount_expectations["predictor"] == model_root.resolve()
    assert plan.offline is True
    assert resolver.verify_runtime_resources(plan) is True


def test_local_huggingface_plan_uses_the_pinned_snapshot_revision(tmp_path: Path):
    hf_home = tmp_path / "hf-home"
    snapshot = hf_home / "hub" / "models--example--fixture-model" / "snapshots" / "0123456789abcdef"
    model_file = snapshot / "weights" / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"fixture model weights")
    newer_snapshot = hf_home / "hub" / "models--example--fixture-model" / "snapshots" / "newer"
    newer_snapshot.mkdir(parents=True)
    (newer_snapshot / "weights").mkdir()
    (newer_snapshot / "weights" / "model.safetensors").write_bytes(b"different weights")

    plan = ModelResolver().plan_backend_resources(
        _model_plan(), "local", _scope(tmp_path, hf_cache_dir=hf_home)
    )

    assert plan.mount_expectations["predictor"] == snapshot.resolve()
    assert ModelResolver().verify_runtime_resources(plan) is True


def test_kaggle_plan_uses_dataset_mount_and_metadata_unions_dataset_sources(tmp_path: Path):
    resource_plan = ModelResolver().plan_backend_resources(
        _model_plan(),
        "kaggle",
        _scope(
            tmp_path,
            model_locations={
                "predictor": {
                    "kind": "dataset",
                    "dataset": "owner/fixture-weights",
                    "relative_path": "release-v1",
                    "version": 3,
                }
            },
        ),
    )

    mount = resource_plan.mount_expectations["predictor"].as_posix()
    assert mount == "/kaggle/input/fixture-weights/release-v1"
    assert resource_plan.required_sources[0]["dataset_slug"] == "owner/fixture-weights"
    assert resource_plan.required_sources[0]["provider_version"] == 3

    with patch.object(KaggleKernelManager, "get_authenticated_username", return_value="owner"):
        metadata = KaggleKernelManager.generate_kernel_metadata(
            "fixture-kernel",
            dataset_sources=["owner/task-data", "owner/shared"],
            model_sources=["owner/shared", "owner/fixture-weights", "none", ""],
        )
    assert metadata["dataset_sources"] == [
        "owner/task-data",
        "owner/shared",
        "owner/fixture-weights",
    ]
    # Model weights are datasets, so Kaggle mounts them through dataset_sources.
    assert metadata["model_sources"] == []


def test_modal_plan_mounts_named_model_volume_and_preserves_governance_outputs(tmp_path: Path):
    config_path = _modal_workspace(tmp_path)
    resource_plan = ModelResolver().plan_backend_resources(
        _model_plan(),
        "modal",
        _scope(
            tmp_path,
            model_locations={
                "predictor": {
                    "kind": "volume",
                    "volume_name": "approved-models",
                    "generation": "generation-7",
                    "relative_path": "generation-7/fixture-model",
                }
            },
        ),
    )
    assert resource_plan.mount_expectations["predictor"].as_posix() == (
        "/vol/models/generation-7/fixture-model"
    )

    script = generate_modal_script(
        str(config_path),
        project_root=tmp_path,
        volume_name="fixture-task-data",
        model_resource_plan=resource_plan,
    )
    compile(script, "generated_modal.py", "exec")
    assert "'approved-models', create_if_missing=False" in script
    assert 'volume_mounts["/vol/models"] = model_volume' in script
    for name in (
        "model_lock.json",
        "model_policy_receipt.json",
        "loaded_models.json",
        "model_outputs.json",
        "model_lineage.json",
    ):
        assert f'"{name}"' in script
    assert "MODEL_RESOURCE_PLAN" in script


def test_offline_plan_fails_for_missing_weights_without_network_fallback(tmp_path: Path):
    (tmp_path / "missing" / "model").mkdir(parents=True)
    resource_plan = ModelResolver().plan_backend_resources(
        _model_plan(),
        "local",
        _scope(
            tmp_path,
            model_locations={"predictor": {"path": "missing/model"}},
        ),
    )
    resolver = ModelResolver()

    with patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")) as urlopen:
        assert resolver.verify_runtime_resources(resource_plan) is False

    urlopen.assert_not_called()
    assert any("Offline model weight file is missing" in error for error in resolver.verification_errors)


def test_kaggle_export_keeps_sealed_model_governance_files(tmp_path: Path):
    from harness.task_spec import load_active_task_spec

    task_path = tmp_path / "configs" / "task.yaml"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(yaml.safe_dump(task_document(), sort_keys=False), encoding="utf-8")
    spec = load_active_task_spec(task_path)
    run = tmp_path / "run" / "exp001"
    run.mkdir(parents=True)
    (run / "model_lock.json").write_text('{"roots": []}', encoding="utf-8")
    (run / "model_policy_receipt.json").write_text('{"status": "eligible"}', encoding="utf-8")
    seal_fixture(run, spec)
    output = tmp_path / "output"
    output.mkdir()

    from harness.kaggle_runtime import export_run

    export_run(run, output, {"exp_id": "exp001", "task_id": "fixture-task"})

    assert (output / "model_lock.json").read_bytes() == (run / "model_lock.json").read_bytes()
    assert (output / "model_policy_receipt.json").read_bytes() == (
        run / "model_policy_receipt.json"
    ).read_bytes()
    assert "model_lock.json" in json.loads((output / "artifact_manifest.json").read_text())["files"]


def test_backend_resource_plan_json_round_trip_preserves_posix_mounts(tmp_path: Path):
    plan = ModelResolver().plan_backend_resources(
        _model_plan(),
        "kaggle",
        _scope(
            tmp_path,
            model_locations={"predictor": {"dataset": "owner/fixture-weights"}},
        ),
    )

    restored = BackendResourcePlan.from_dict(plan.to_dict())

    assert restored.backend == plan.backend
    assert restored.offline is True
    assert restored.mount_expectations["predictor"].as_posix() == (
        "/kaggle/input/fixture-weights"
    )


def test_kaggle_code_bundle_excludes_model_weights_and_attaches_model_dataset(tmp_path: Path):
    config_path = _modal_workspace(tmp_path)
    weight_dir = tmp_path / "configs" / "weights"
    weight_dir.mkdir()
    (weight_dir / "model.safetensors").write_bytes(b"do not put weights in code ZIP")
    package = tmp_path / "package"
    runtime = {
        "core": {"distribution": "ai-harness", "version": "0.2.0"},
        "plugin": {"id": "default", "version": "1", "install": None},
        "requirements": [],
        "bundle_files": [],
    }

    with patch.object(KaggleKernelManager, "get_authenticated_username", return_value="owner"):
        KaggleKernelManager.setup_kernel_package(
            package,
            str(config_path),
            "exp001",
            "owner/task-data",
            project_root=tmp_path,
            task_config_path=tmp_path / "configs" / "task.yaml",
            runtime_manifest=runtime,
            enable_internet=False,
            model_sources=["owner/model-assets"],
        )

    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    bundle = json.loads((package / "code-bundle.json").read_text(encoding="utf-8"))
    bundle_path = package / bundle["directory"] / bundle["filename"]
    assert "owner/model-assets" in metadata["dataset_sources"]
    with zipfile.ZipFile(bundle_path) as archive:
        assert "configs/weights/model.safetensors" not in archive.namelist()
