from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# ===========================================================================
# Source: tests/test_model_registry.py
# ===========================================================================
from pathlib import Path
from harness.model_assets import ModelDefinition, ModelStore
from harness.model_registry import builtin_model_names

"""Bounded storage tests for model asset definitions and legacy model names."""

def _definition_registry() -> dict:
    return {
        "schema_version": 1,
        "model_id": "  test-model  ",
        "kind": "PRETRAINED",
        "family": "  Test LLM  ",
        "task_types": ["generation"],
        "modalities": ["text"],
        "source": {
            "provider": "huggingface",
            "repository": "example/test-model",
            "revision": "abcdef0123456789",
            "source_uri": "hf://example/test-model@abcdef0123456789",
        },
        "artifacts": {"manifest_ref": None, "bundle_digest": None},
        "parameter_facts": {
            "exact_param_count": 100,
            "count_basis": "LOGICAL_TOTAL",
            "active_param_count": None,
            "trainable_param_count": 100,
            "evidence_refs": ["count-report"],
        },
        "release": {"public_release_date": "2026-01-01", "evidence_refs": []},
        "license": {"expression": "Apache-2.0", "evidence_refs": []},
        "parents": [],
        "verification_refs": [],
    }

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_load_valid_model_definition_yaml_under_size_limit(self):
        tmp_path = self.tmp_path
        path = tmp_path / "model.yaml"
        path.write_text(yaml.safe_dump(_definition_registry(), sort_keys=False), encoding="utf-8")
    
        model = ModelStore().load_catalog_file(path)
    
        assert model.model_id == "test-model"
        assert model.kind == "pretrained"
        assert model.parameter_facts.count_basis == "logical_total"

    def test_rejects_model_definition_larger_than_16_kib(self):
        tmp_path = self.tmp_path
        path = tmp_path / "oversized.yaml"
        path.write_bytes(b" " * (16 * 1024 + 1))
    
        with pytest.raises(ValueError, match="exceeds 16 KiB limit"):
            ModelStore().load_catalog_file(path)

    def test_rejects_unknown_nested_fields_and_tensor_cards(self):
        tmp_path = self.tmp_path
        definition = _definition_registry()
        definition["parameter_facts"]["tensor_card"] = {"weights": [1, 2, 3]}
        path = tmp_path / "extra-field.yaml"
        path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    
        with pytest.raises(ValueError, match="Unknown field.*tensor_card"):
            ModelStore().load_catalog_file(path)

    def test_rejects_yaml_aliases_and_duplicate_keys(self):
        tmp_path = self.tmp_path
        alias_path = tmp_path / "alias.yaml"
        alias_path.write_text("source: &source {provider: local}\ncopy: *source\n", encoding="utf-8")
        duplicate_path = tmp_path / "duplicate.yaml"
        duplicate_path.write_text("model_id: first\nmodel_id: second\n", encoding="utf-8")
    
        with pytest.raises(ValueError, match="aliases are not allowed"):
            ModelStore().load_catalog_file(alias_path)
        with pytest.raises(ValueError, match="Duplicate YAML key"):
            ModelStore().load_catalog_file(duplicate_path)

    def test_json_and_yaml_definitions_parse_to_same_normalized_record(self):
        tmp_path = self.tmp_path
        definition = _definition_registry()
        yaml_path = tmp_path / "model.yaml"
        json_path = tmp_path / "model.json"
        yaml_path.write_text(yaml.safe_dump(definition), encoding="utf-8")
        json_path.write_text(json.dumps(definition), encoding="utf-8")
        store = ModelStore()
    
        assert store.load_catalog_file(yaml_path).to_dict() == store.load_catalog_file(json_path).to_dict()

    def test_compute_digest_uses_canonical_normalized_json(self):
        first = ModelDefinition.from_dict(_definition_registry())
        reordered = dict(reversed(list(_definition_registry().items())))
        second = ModelDefinition.from_dict(reordered)
        canonical = json.dumps(
            first.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    
        assert first.compute_digest() == second.compute_digest()
        assert first.compute_digest() == hashlib.sha256(canonical).hexdigest()
        assert len(first.compute_digest()) == 64

    def test_register_definition_writes_atomically_at_content_addressed_path(self):
        tmp_path = self.tmp_path
        store = ModelStore()
        model = ModelDefinition.from_dict(_definition_registry())
    
        digest = store.register_definition(model, tmp_path)
        object_path = tmp_path / ".ai-harness" / "models" / "objects" / f"{digest}.json"
    
        assert object_path.is_file()
        assert json.loads(object_path.read_text(encoding="utf-8")) == model.to_dict()
        assert list(object_path.parent.iterdir()) == [object_path]
        assert store.load_definition(digest, tmp_path).to_dict() == model.to_dict()

    def test_load_definition_returns_none_for_absent_digest_and_rejects_bad_digest(self):
        tmp_path = self.tmp_path
        store = ModelStore()
        assert store.load_definition("0" * 64, tmp_path) is None
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            store.load_definition("../model", tmp_path)

    def test_load_definition_detects_content_tampering(self):
        tmp_path = self.tmp_path
        store = ModelStore()
        digest = store.register_definition(ModelDefinition.from_dict(_definition_registry()), tmp_path)
        path = tmp_path / ".ai-harness" / "models" / "objects" / f"{digest}.json"
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["family"] = "Tampered family"
        path.write_text(json.dumps(changed), encoding="utf-8")
    
        with pytest.raises(ValueError, match="digest mismatch"):
            store.load_definition(digest, tmp_path)

    def test_existing_builtin_registry_still_exposes_sklearn_and_lightgbm_models(self):
        names = builtin_model_names()
        assert "logistic_regression" in names
        assert "lightgbm" in names


# ===========================================================================
# Source: tests/test_model_resolution.py
# ===========================================================================
from pathlib import Path
from harness.model_assets import BackendResourcePlan, ModelResolver
from harness.project import attach_project
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import task_document

"""Backend model-resource planning, verification, and evidence preservation."""

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

class TestModelResolution(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_local_resource_plan_resolves_explicit_root_and_verifies_manifest_files(self):
        tmp_path = self.tmp_path
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

    def test_local_huggingface_plan_uses_the_pinned_snapshot_revision(self):
        tmp_path = self.tmp_path
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

    def test_kaggle_plan_uses_dataset_mount_and_metadata_unions_dataset_sources(self):
        tmp_path = self.tmp_path
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

    def test_modal_plan_mounts_named_model_volume_and_preserves_governance_outputs(self):
        tmp_path = self.tmp_path
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

    def test_offline_plan_fails_for_missing_weights_without_network_fallback(self):
        tmp_path = self.tmp_path
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

    def test_kaggle_export_keeps_sealed_model_governance_files(self):
        tmp_path = self.tmp_path
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

    def test_backend_resource_plan_json_round_trip_preserves_posix_mounts(self):
        tmp_path = self.tmp_path
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

    def test_kaggle_code_bundle_excludes_model_weights_and_attaches_model_dataset(self):
        tmp_path = self.tmp_path
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


# ===========================================================================
# Source: tests/test_model_execution_plan.py
# ===========================================================================
from pathlib import Path
from harness.cloud_runner import _prepare_remote_run, run_via_kaggle
from harness.kernel_dispatch import dispatch_kernel
from harness.model_assets import (
    ModelAdmissionError,
    enforce_model_admission,
    prepare_or_validate_model_plan,
    resolve_effective_config,
)
from harness.project import HarnessProject
from harness.runner import UniversalRunner
from harness.task_spec import load_active_task_spec
from tests.fixtures import TRAIN_CSV_4ROW

"""Execution-plan admission and cloud side-effect regression tests."""

def _definition_exec(count: int = 8) -> dict:
    return {
        "schema_version": 1,
        "model_id": "fixture-model",
        "kind": "checkpoint",
        "family": "fixture classifier",
        "task_types": ["classification"],
        "modalities": ["text"],
        "source": {
            "provider": "local",
            "repository": "fixture-model",
            "revision": "workspace-v1",
            "source_uri": "workspace://fixture-model",
        },
        "artifacts": {"manifest_ref": None, "bundle_digest": None},
        "parameter_facts": {
            "exact_param_count": count,
            "count_basis": "logical_total",
            "trainable_param_count": 2,
            "active_param_count": None,
            "evidence_refs": ["fixture-count"],
        },
        "release": {"public_release_date": None, "evidence_refs": []},
        "license": {"expression": "MIT", "evidence_refs": ["fixture-license"]},
        "parents": [],
        "verification_refs": ["fixture-verification"],
    }

def _governed_workspace(tmp_path: Path, *, limit: int = 10) -> tuple[Path, Path, Path]:
    task = task_document()
    task["task"]["competition_rules"] = {"max_total_parameters": limit}
    task["task"]["model_governance"] = {
        "schema_version": 1,
        "mode": "required",
        "catalog": "configs/models.yaml",
    }
    root = Path(tmp_path) / "workspace"
    (root / "configs").mkdir(parents=True)
    (root / "dataset").mkdir()
    (root / "dataset" / "train.csv").write_text(TRAIN_CSV_4ROW, encoding="utf-8")
    task_config = root / "configs" / "task.yaml"
    task_config.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    attach_project(root, task_config)
    model_path = root / "configs" / "fixture-model.yaml"
    model_path.write_text(yaml.safe_dump(_definition_exec(), sort_keys=False), encoding="utf-8")
    (root / "configs" / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "models": {"predictor-model": {"definition": "configs/fixture-model.yaml"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "configs" / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"branch": "fixture-branch"},
                "model_assets": {
                    "schema_version": 1,
                    "roots": [
                        {
                            "slot": "predictor",
                            "ref": "predictor-model",
                            "phases": ["training", "inference"],
                        }
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = root / "configs" / "exp001.yaml"
    config_path.write_text(
        yaml.safe_dump({"experiment": {"id": "exp001"}}, sort_keys=False),
        encoding="utf-8",
    )
    return root, Path(task_config), config_path

class TestModelExecutionPlan(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_base_inheritance_produces_identical_local_and_cloud_model_plans(self):
        tmp_path = self.tmp_path
        root, task_path, config_path = _governed_workspace(tmp_path)
        project = HarnessProject.discover(root, task_path)
        args = SimpleNamespace(base_config=None)
    
        local_effective = resolve_effective_config(project, config_path, args)
        cloud_effective = resolve_effective_config(project, config_path, args)
        spec = load_active_task_spec(task_path)
        local_plan = prepare_or_validate_model_plan(
            spec, local_effective, ModelStore(), project=project
        )
        cloud_plan = prepare_or_validate_model_plan(
            spec, cloud_effective, ModelStore(), project=project
        )
        enforce_model_admission(local_plan)
        enforce_model_admission(cloud_plan)
    
        assert local_effective["experiment"]["branch"] == "fixture-branch"
        assert local_plan.lock_digest == cloud_plan.lock_digest
        assert local_plan.model_lock == cloud_plan.model_lock
        assert local_plan.model_lock["roots"][0]["model_id"] == "fixture-model"

    def test_cloud_preparation_uses_the_same_effective_plan_as_local(self):
        tmp_path = self.tmp_path
        root, task_path, config_path = _governed_workspace(tmp_path)
        project = HarnessProject.discover(root, task_path)
        effective = resolve_effective_config(project, config_path, SimpleNamespace(base_config=None))
        spec = load_active_task_spec(task_path)
        local_plan = prepare_or_validate_model_plan(spec, effective, ModelStore(), project=project)
    
        with (
            patch.dict(
                os.environ,
                {"HARNESS_PROJECT_ROOT": str(root), "HARNESS_TASK_CONFIG": str(task_path)},
            ),
            patch("harness.cloud_runner._warn_if_framework_checkout_dirty"),
            patch(
                "harness.cloud_runner.CloudPreflightInspector.run_full_preflight_check",
                return_value=(True, [], effective),
            ) as preflight,
            patch(
                "harness.citation_audit.enforce_citation_admission",
                return_value={"admission": "allowed"},
            ),
            patch(
                "harness.cloud_policy.provider_preflight",
                return_value={"cli_installed": True, "authenticated": True, "version_supported": True},
            ),
            patch("harness.cloud_runner.declared_data_roots", return_value=[]),
            patch("harness.cloud_runner.dataset_fingerprint", return_value={"fingerprint": "fixture"}),
            patch("harness.cloud_runner.UploadConsentStore.is_granted", return_value=True),
            patch("harness.cloud_runner.resolve_remote_runtime", return_value={"core": {}}),
        ):
            context = _prepare_remote_run(
                str(config_path), "kaggle", root, effective_config=effective
            )
    
        assert preflight.call_args.kwargs["effective_config"] == effective
        assert context["model_plan"].lock_digest == local_plan.lock_digest

    def test_over_limit_model_is_rejected_before_plugin_or_model_setup(self):
        tmp_path = self.tmp_path
        root, task_path, config_path = _governed_workspace(tmp_path, limit=5)
        project = HarnessProject.discover(root, task_path)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        args = SimpleNamespace(
            project_root=str(root),
            task_config=str(task_path),
            base_config=None,
        )
    
        with patch("harness.runner.load_task_plugin") as load_plugin:
            with pytest.raises(ModelAdmissionError) as error:
                UniversalRunner(cfg, str(config_path), args)
    
        assert error.value.code == "MODEL_PARAMETER_LIMIT_EXCEEDED"
        load_plugin.assert_not_called()
        assert not (project.artifacts_dir / "exp001").exists()

    def test_direct_kernel_dispatch_rejects_before_provider_push(self):
        tmp_path = self.tmp_path
        root, task_path, config_path = _governed_workspace(tmp_path, limit=5)
        notebook = root / "notebook.ipynb"
        notebook.write_text('{"cells": []}', encoding="utf-8")
        client = SimpleNamespace(kernels_push=lambda *args, **kwargs: pytest.fail("provider push called"))
    
        with pytest.raises(ModelAdmissionError) as error:
            dispatch_kernel(
                str(notebook),
                "fixture-owner/model-gated",
                "Governed model gate",
                client=client,
                project_root=root,
                task_config=task_path,
                config_path=config_path,
            )
    
        assert error.value.code == "MODEL_PARAMETER_LIMIT_EXCEEDED"

    def test_validator_keeps_trainable_budget_separate_from_logical_total(self):
        from harness.validator import ExperimentValidator
    
        class Parameter:
            def __init__(self, count, requires_grad):
                self.count = count
                self.requires_grad = requires_grad
    
            def numel(self):
                return self.count
    
        class Model:
            def parameters(self):
                return [Parameter(8, False), Parameter(2, True)]
    
        assert ExperimentValidator.count_trainable_parameters(Model()) == 2
        assert ExperimentValidator.count_logical_parameters(Model()) == 10
    
        with pytest.raises(ValueError, match="MODEL_PARAMETER_LIMIT_EXCEEDED"):
            ExperimentValidator.assert_logical_parameter_limit(Model(), max_params=9)

    def test_kaggle_dry_run_does_not_upload_data_or_dispatch_kernel(self):
        tmp_path = self.tmp_path
        root, task_path, _ = _governed_workspace(tmp_path)
        project = HarnessProject.discover(root, task_path)
        stage_dir = tmp_path / "aih-kaggle-job_test"
        stage_dir.mkdir()
        context = {
            "project": project,
            "exp_id": "exp-dry-run",
            "config_rel": "configs/exp001.yaml",
            "config_sha256": "config-hash",
            "effective_config_sha256": "effective-hash",
            "task_id": "fixture-task",
            "task": {"runtime": {}},
            "runtime": {"core": {}},
            "data_config": {},
            "external_kaggle_dataset": "fixture-owner/fixture-data",
            "fingerprint": {"fingerprint": "fixture"},
        }
    
        def setup_package(target_dir, *args, **kwargs):
            (Path(target_dir) / "kernel-metadata.json").write_text(
                '{"id":"fixture-owner/fixture-kernel"}', encoding="utf-8"
            )
    
        with (
            patch("harness.cloud_runner._prepare_remote_run", return_value=context),
            patch("harness.cloud_runner.tempfile.mkdtemp", return_value=str(stage_dir)),
            patch("harness.cloud_runner.KaggleKernelManager.ensure_cli_ready"),
            patch("harness.cloud_runner.KaggleKernelManager.setup_kernel_package", side_effect=setup_package),
            patch("harness.cloud_runner.KaggleKernelManager.push_dataset_to_kaggle") as upload_dataset,
            patch("harness.cloud_runner.KaggleKernelManager.wait_for_dataset") as wait_dataset,
            patch("harness.cloud_runner.KaggleKernelManager.push_code_bundle") as upload_code,
            patch("harness.cloud_runner.subprocess.run") as push_kernel,
        ):
            result = run_via_kaggle(
                "configs/exp001.yaml", project_root=root, dry_run=True
            )
    
        assert result["dry_run"] is True
        upload_dataset.assert_not_called()
        wait_dataset.assert_not_called()
        upload_code.assert_not_called()
        push_kernel.assert_not_called()


# ===========================================================================
# Source: tests/test_model_lineage_and_submission.py
# ===========================================================================
from pathlib import Path
from harness.artifacts import seal_file_map
from harness.candidate_preflight import load_preflight_receipt, run_candidate_preflight
from harness.model_assets.lineage import (
    evaluate_candidate_model_policy,
    model_policy_digest,
    resolve_candidate_model_closure,
)
from harness.submission_gate import SubmissionGate, SubmissionGateError
from tests.fixtures import make_submission_evidence, make_workspace, workspace_with_verified_pair

"""Candidate model closure and SubmissionGate binding regressions."""

def _canonical_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _definition_lineage(model_id: str, count: int) -> dict:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "kind": "checkpoint",
        "family": "fixture classifier",
        "task_types": ["classification"],
        "modalities": ["text"],
        "source": {
            "provider": "local",
            "repository": model_id,
            "revision": "workspace-v1",
            "source_uri": f"workspace://{model_id}",
        },
        "artifacts": {"manifest_ref": None, "bundle_digest": None},
        "parameter_facts": {
            "exact_param_count": count,
            "count_basis": "logical_total",
            "active_param_count": None,
            "trainable_param_count": count,
            "evidence_refs": [f"{model_id}-count"],
        },
        "release": {"public_release_date": None, "evidence_refs": []},
        "license": {"expression": "MIT", "evidence_refs": [f"{model_id}-license"]},
        "parents": [],
        "verification_refs": [f"{model_id}-verified"],
    }

def _source(exp_id: str, model_id: str, count: int, policy: dict) -> dict:
    definition = _definition_lineage(model_id, count)
    model = ModelDefinition.from_dict(definition)
    rules_digest = model_policy_digest(policy)
    lock = {
        "schema_version": 1,
        "task_id": "fixture-task",
        "governance_mode": "required",
        "roots": [
            {
                "slot": "predictor",
                "role": "predictor",
                "ref": model_id,
                "phases": ["inference"],
                "model_id": model_id,
                "definition_digest": model.compute_digest(),
                "artifact_bundle_digest": None,
                "definition": definition,
                "parents": [],
                "verification_refs": [f"{model_id}-verified"],
            }
        ],
        "policy_digest": rules_digest,
    }
    lock["model_lock_digest"] = _canonical_digest(lock)
    receipt = {
        "schema_version": 1,
        "task_id": "fixture-task",
        "governance_mode": "required",
        "required": True,
        "declared": True,
        "status": "eligible",
        "eligible": True,
        "failure_code": None,
        "reasons": [],
        "facts_evaluated": {},
        "policy_digest": rules_digest,
        "model_lock_digest": lock["model_lock_digest"],
    }
    return {
        "exp_id": exp_id,
        "source_manifest_digest": hashlib.sha256(exp_id.encode()).hexdigest(),
        "model_lock_digest": lock["model_lock_digest"],
        "model_lock": lock,
        "model_policy_receipt": receipt,
    }

def _governed_task() -> dict:
    document = task_document()
    document["task"]["subtasks"] = [
        {
            "id": "label",
            "column": "label",
            "task_type": "binary_classification",
            "classes": [0, 1],
            "primary_metric": "accuracy",
            "direction": "maximize",
        }
    ]
    document["task"]["submission_spec"] = {"required_columns": ["id", "label"]}
    document["task"]["competition_rules"] = {"max_total_parameters": 10}
    document["task"]["model_governance"] = {"schema_version": 1, "mode": "required"}
    return document

def _seal_model_records(run_dir: Path, roots: list[dict], policy: dict) -> str:
    rules_digest = model_policy_digest(policy)
    lock = {
        "schema_version": 1,
        "task_id": "fixture-task",
        "governance_mode": "required",
        "roots": roots,
        "policy_digest": rules_digest,
    }
    lock["model_lock_digest"] = _canonical_digest(lock)
    receipt = {
        "schema_version": 1,
        "task_id": "fixture-task",
        "governance_mode": "required",
        "required": True,
        "declared": True,
        "status": "eligible",
        "eligible": True,
        "failure_code": None,
        "reasons": ["Each declared component is individually within the parameter limit."],
        "facts_evaluated": {},
        "policy_digest": rules_digest,
        "model_lock_digest": lock["model_lock_digest"],
    }
    (run_dir / "model_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    (run_dir / "model_policy_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_lock_digest"] = lock["model_lock_digest"]
    manifest["model_policy_status"] = "eligible"
    manifest["files"] = seal_file_map(run_dir)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return lock["model_lock_digest"]

class TestModelLineageAndSubmission(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_aggregate_policy_rejects_individually_compliant_over_cap_blend(self):
        policy = {"max_total_parameters": 10}
        sources = [_source("run-a", "model-a", 6, policy), _source("run-b", "model-b", 6, policy)]
        recipe = {"mode": "weighted_blend", "source_ids": ["run-a", "run-b"], "details": {"weights": [0.5, 0.5]}}
    
        closure = resolve_candidate_model_closure(sources, recipe)
        assessment = evaluate_candidate_model_policy(closure, policy)
    
        assert closure.verified
        assert len(closure.models) == 2
        assert assessment.eligible is False
        assert assessment.failure_code == "MODEL_PARAMETER_LIMIT_EXCEEDED"
        assert assessment.facts_evaluated["total_parameters"] == 12

    def test_preflight_receipt_v3_binds_subject_lock_and_policy(self):
        root, _ = workspace_with_verified_pair(prefix="aih-model-preflight-")
        try:
            from harness.project import HarnessProject
    
            project = HarnessProject.discover(root)
            candidate = project.artifacts_dir / "control" / "test_predictions.csv"
            incumbent = root / "incumbent.csv"
            shutil.copyfile(candidate, incumbent)
    
            receipt = run_candidate_preflight(
                root, candidate, incumbent, id_col="id", pred_col="pred_label", source_exp_id="control"
            )
            loaded = load_preflight_receipt(root, receipt["preflight_id"])
    
            assert receipt["schema_version"] == 3
            assert len(receipt["candidate_subject_digest"]) == 64
            assert len(receipt["model_lock_digest"]) == 64
            assert len(receipt["model_policy_digest"]) == 64
            assert receipt["model_compliance_status"] == "not_required"
            assert loaded == receipt
            assert loaded["candidate_subject_digest"] == receipt["candidate_subject_digest"]
        finally:
            shutil.rmtree(root)

    def test_preflight_reader_keeps_v1_receipts_readable(self):
        root, _ = make_workspace(prefix="aih-model-preflight-v1-")
        try:
            receipt_id = "preflight-" + "a" * 32
            receipt = {
                "schema_version": 1,
                "preflight_id": receipt_id,
                "candidate": {"path": "candidate.csv", "sha256": "a" * 64, "semantic_hash": "b" * 64, "row_count": 1},
                "incumbent": {"path": "incumbent.csv", "sha256": "c" * 64, "semantic_hash": "d" * 64, "row_count": 1},
                "similarity": {
                    "spearman_rho": 0.0,
                    "pearson_r": 0.0,
                    "mean_abs_delta": 0.0,
                    "max_abs_delta": 0.0,
                    "p95_abs_delta": 0.0,
                    "rank_swap_count": 0,
                    "rank_swap_ratio": 0.0,
                },
                "classification": "DIVERSE_CANDIDATE",
                "recommendation": "legacy receipt",
                "verdict": "PASS",
                "created_at": "2026-11-03T00:00:00+00:00",
            }
            output = root / ".ai-harness" / "artifacts" / "preflight"
            output.mkdir(parents=True)
            (output / f"{receipt_id}.json").write_text(json.dumps(receipt), encoding="utf-8")
    
            assert load_preflight_receipt(root, receipt_id) == receipt
        finally:
            shutil.rmtree(root)

    def test_submission_gate_rejects_over_cap_model_closure_before_permit(self):
        tmp_path = self.tmp_path
        del tmp_path
        task = _governed_task()
        train_csv = "id,text,label\n" + "".join(
            f"{index},text {index},{index % 2}\n" for index in range(40)
        )
        root, _ = make_workspace(
            prefix="aih-model-gate-cap-", task=task, train_csv=train_csv
        )
        try:
            pd.DataFrame({"id": range(40, 80), "text": ["test"] * 40}).to_csv(
                root / "dataset" / "test.csv", index=False
            )
            run_dir = make_submission_evidence(root, "control", labels=[0] * 40)
            policy = task["task"]["competition_rules"]
            roots = []
            for model_id in ("model-a", "model-b"):
                definition = _definition_lineage(model_id, 6)
                model = ModelDefinition.from_dict(definition)
                roots.append(
                    {
                        "slot": model_id,
                        "role": model_id,
                        "ref": model_id,
                        "phases": ["inference"],
                        "model_id": model_id,
                        "definition_digest": model.compute_digest(),
                        "artifact_bundle_digest": None,
                        "definition": definition,
                        "parents": [],
                        "verification_refs": [f"{model_id}-verified"],
                    }
                )
            _seal_model_records(run_dir, roots, policy)
    
            gate = SubmissionGate(root)
            predictions = pd.read_csv(run_dir / "test_predictions.csv")
            payload = gate.plugin.build_submission(gate.context, predictions, root / "submission.csv")
    
            with pytest.raises(SubmissionGateError, match="MODEL_POLICY_VIOLATION"):
                gate.authorize_dispatch(
                    "fixture-comp",
                    payload,
                    exp_id="control",
                    user_approved=True,
                    quota={"daily_quota": 5, "num_today": 0, "has_remaining_quota": True},
                )
            assert not (root / ".ai-harness" / "submission_events.jsonl").exists()
        finally:
            shutil.rmtree(root)

    def test_dispatch_permit_persists_preflight_subject_and_model_lock_binding(self):
        root, _ = workspace_with_verified_pair(prefix="aih-model-gate-binding-")
        try:
            from harness.competition_state import CompetitionStateMachine
    
            state_machine = CompetitionStateMachine(root)
            state_machine.state["gates"].update(baseline_locked=True, baseline_exp_id="control")
            state_machine.save()
            gate = SubmissionGate(root)
            predictions = pd.read_csv(gate.project.artifacts_dir / "control" / "test_predictions.csv")
            payload = gate.plugin.build_submission(gate.context, predictions, root / "submission.csv")
            incumbent = root / "incumbent.csv"
            shutil.copyfile(payload, incumbent)
            preflight = run_candidate_preflight(
                root,
                payload,
                incumbent,
                id_col="id",
                pred_col="label",
                source_exp_id="control",
            )
    
            authorization = gate.authorize_dispatch(
                "fixture-comp",
                payload,
                exp_id="control",
                user_approved=True,
                preflight_id=preflight["preflight_id"],
                quota={"daily_quota": 5, "num_today": 0, "has_remaining_quota": True},
            )
    
            intent = authorization["intent"]
            assert intent["candidate_subject_digest"] == preflight["candidate_subject_digest"]
            assert intent["model_lock_digest"] == preflight["model_lock_digest"]
            assert intent["model_policy_digest"] == preflight["model_policy_digest"]
            assert intent["preflight_id"] == preflight["preflight_id"]
        finally:
            shutil.rmtree(root)


# ===========================================================================
# Source: tests/test_model_cli_and_bridge.py
# ===========================================================================
from pathlib import Path
from harness.agent_bridge import AgentBridge, _model_assets_hud_summary
from harness.bridge_common import (
    ActionObservation,
    ApprovalTier,
    TypedActionCall,
    TypedActionName,
    get_action_schema,
)
from harness.model_assets.cli import rebuild_model_index
from harness.cli import main as harness_main

def _definition_cli() -> dict:
    return {
        "schema_version": 1,
        "model_id": "fixture-model",
        "kind": "pretrained",
        "family": "fixture text model",
        "task_types": ["classification"],
        "modalities": ["text"],
        "source": {
            "provider": "local",
            "repository": "models/fixture-model",
            "revision": "workspace",
            "source_uri": "file://models/fixture-model",
        },
        "artifacts": {
            "manifest_ref": "configs/model-manifest.json",
            "bundle_digest": "a" * 64,
        },
        "parameter_facts": {
            "exact_param_count": 42,
            "count_basis": "logical_total",
            "evidence_refs": ["fixture-count-report"],
        },
        "release": {"public_release_date": "2026-01-01", "evidence_refs": ["release-note"]},
        "license": {"expression": "Apache-2.0", "evidence_refs": ["license-page"]},
        "parents": [],
        "verification_refs": [],
    }

def _workspace(tmp_path: Path, *, with_model: bool = True) -> tuple[Path, Path, str | None]:
    task = task_document()
    task["task"]["competition_rules"] = {
        "max_total_parameters": 100,
        "model_weight_release_cutoff": "2026-06-01",
    }
    task["task"]["model_governance"] = {"schema_version": 1, "mode": "required"}
    root, task_path = make_workspace(task=task, root_path=tmp_path / "workspace")

    model_digest = None
    if with_model:
        model_dir = root / "models" / "fixture-model"
        model_dir.mkdir(parents=True)
        payload = b"fixture model weights"
        weight_path = model_dir / "weights.bin"
        weight_path.write_bytes(payload)
        manifest = {
            "schema_version": 1,
            "files": [
                {
                    "relative_path": "weights.bin",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
        (root / "configs" / "model-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        model_path = root / "configs" / "model.yaml"
        model_path.write_text(yaml.safe_dump(_definition_cli(), sort_keys=False), encoding="utf-8")
        model_digest = ModelStore().register_definition(
            ModelStore().load_catalog_file(model_path), root
        )
        exp_config = {
            "experiment": {"id": "exp001"},
            "model_assets": {
                "schema_version": 1,
                "roots": [
                    {
                        "slot": "predictor",
                        "ref": model_digest,
                        "phases": ["training", "inference"],
                    }
                ],
            },
        }
        (root / "configs" / "exp001.yaml").write_text(
            yaml.safe_dump(exp_config, sort_keys=False), encoding="utf-8"
        )
    return root, task_path, model_digest

def _json_result(capsys) -> dict:
    return json.loads(capsys.readouterr().out)

def _dispatch_model_action(root: Path, action: TypedActionName, params: dict) -> ActionObservation:
    bridge = AgentBridge.__new__(AgentBridge)
    task_config = root / "configs" / "task.yaml"
    bridge.project = HarnessProject.discover(root, task_config)
    bridge.project_root = root
    call = TypedActionCall(
        step_index=1,
        action=action.value,
        approval_tier=ApprovalTier.AUTO.value,
        params=params,
    )
    return bridge._dispatch_typed_action(call, None, None, None)

class TestModelCliAndBridge(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._td.name)
        self._stdout_buf = io.StringIO()
        self._stderr_buf = io.StringIO()
        self._stdout_redirect = contextlib.redirect_stdout(self._stdout_buf)
        self._stderr_redirect = contextlib.redirect_stderr(self._stderr_buf)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        class _Capsys:
            def __init__(inner):
                pass
            def readouterr(inner):
                out = self._stdout_buf.getvalue()
                err = self._stderr_buf.getvalue()
                self._stdout_buf.seek(0)
                self._stdout_buf.truncate(0)
                self._stderr_buf.seek(0)
                self._stderr_buf.truncate(0)
                return SimpleNamespace(out=out, err=err)
        self._capsys = _Capsys()

    def tearDown(self):
        self._stderr_redirect.__exit__(None, None, None)
        self._stdout_redirect.__exit__(None, None, None)
        self._td.cleanup()

    def test_model_cli_register_list_inspect_and_validate_config(self):
        tmp_path = self.tmp_path
        capsys = self._capsys
        root, _, _ = _workspace(tmp_path, with_model=False)
        model_path = root / "configs" / "model.yaml"
        model_path.write_text(yaml.safe_dump(_definition_cli(), sort_keys=False), encoding="utf-8")
    
        assert harness_main(["model", "register", "--file", str(model_path), "--project-root", str(root)]) == 0
        registered = _json_result(capsys)
        assert registered["status"] == "registered"
        assert registered["artifact_verification"] == "pending"
        assert registered["eligibility"] == "not assessed"
        digest = registered["definition_digest"]
    
        assert harness_main(["model", "list", "--project-root", str(root), "--limit", "1"]) == 0
        listed = _json_result(capsys)
        assert listed["pagination"] == {"limit": 1, "offset": 0, "total": 1, "next_offset": None}
        assert listed["models"][0]["definition_digest"] == digest
    
        assert harness_main(["model", "inspect", digest, "--project-root", str(root)]) == 0
        inspected = _json_result(capsys)
        assert inspected["model"]["release"]["public_release_date"] == "2026-01-01"
        assert inspected["artifact_verification"] == "pending"
    
        exp = {
            "experiment": {"id": "exp001"},
            "model_assets": {"schema_version": 1, "roots": [{"slot": "predictor", "ref": digest}]},
        }
        (root / "configs" / "exp001.yaml").write_text(yaml.safe_dump(exp), encoding="utf-8")
        assert harness_main(
            ["model", "validate-config", "--config", "configs/exp001.yaml", "--project-root", str(root)]
        ) == 0
        plan = _json_result(capsys)
        assert plan["status"] == "eligible"
        assert plan["checks"]["eligible"] is True
        assert plan["model_lock"]["roots"][0]["model_id"] == "fixture-model"

    def test_model_cli_verify_hashes_artifact_and_resolve_is_dry_run(self):
        tmp_path = self.tmp_path
        capsys = self._capsys
        root, _, digest = _workspace(tmp_path)
        assert digest
        assert harness_main(["model", "verify", digest, "--project-root", str(root)]) == 0
        verified = _json_result(capsys)
        assert verified["status"] == "verified"
        assert verified["checks"]["artifact_files"] == "passed"
        assert verified["checks"]["files_checked"] == 1
    
        assert harness_main(
            ["model", "resolve", "--config", "configs/exp001.yaml", "--backend", "local", "--dry-run", "--project-root", str(root)]
        ) == 0
        resolved = _json_result(capsys)
        assert resolved["dry_run"] is True
        assert resolved["resource_plan"]["backend"] == "local"
        assert resolved["resource_plan"]["mount_expectations"]["predictor"].endswith("models/fixture-model")

    def test_model_bridge_actions_are_typed_and_dispatch_to_services(self):
        tmp_path = self.tmp_path
        root, _, digest = _workspace(tmp_path)
        assert digest
        assert TypedActionName.LIST_MODEL_ASSETS.value in AgentBridge._ACTION_REGISTRY
        assert get_action_schema("list_model_assets")["optional"]["limit"] == 20
        assert get_action_schema("register_model_asset")["parameter_types"]["file"] == "path"
    
        listed = _dispatch_model_action(root, TypedActionName.LIST_MODEL_ASSETS, {"limit": 10})
        inspected = _dispatch_model_action(
            root, TypedActionName.INSPECT_MODEL_ASSET, {"model_ref": digest}
        )
        registered = _dispatch_model_action(
            root,
            TypedActionName.REGISTER_MODEL_ASSET,
            {"file": str(root / "configs" / "model.yaml")},
        )
        validated = _dispatch_model_action(
            root,
            TypedActionName.VALIDATE_MODEL_PLAN,
            {"config_path": "configs/exp001.yaml"},
        )
    
        assert [item.status for item in (listed, inspected, registered, validated)] == [
            "SUCCESS",
            "SUCCESS",
            "SUCCESS",
            "SUCCESS",
        ]
        assert listed.data["models"][0]["model_id"] == "fixture-model"
        assert inspected.data["definition_digest"] == digest
        assert registered.data["artifact_verification"] == "pending"
        assert validated.data["model_policy_receipt"]["eligible"] is True

    def test_model_hud_is_compact_and_uses_index_and_selected_lock_only(self):
        tmp_path = self.tmp_path
        root, task_path, digest = _workspace(tmp_path)
        assert digest
        project = HarnessProject.discover(root, task_path)
        task_spec = load_active_task_spec(task_path)
        index = rebuild_model_index(root, task_spec=task_spec, task_id=project.task_id)
    
        run_dir = root / ".ai-harness" / "artifacts" / "exp001"
        run_dir.mkdir(parents=True)
        lock = {
            "model_lock_digest": "b" * 64,
            "roots": [
                {
                    "slot": "predictor",
                    "model_id": "fixture-model",
                    "phases": ["training", "inference"],
                    "definition": _definition_cli(),
                }
            ],
        }
        (run_dir / "model_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        (run_dir / "model_policy_receipt.json").write_text(
            json.dumps({"status": "eligible"}), encoding="utf-8"
        )
    
        assert index["registered_count"] == 1
        with (
            patch.object(Path, "glob", side_effect=AssertionError("HUD enumerated registry files")),
            patch(
                "harness.model_assets.resolver._sha256_file",
                side_effect=AssertionError("HUD hashed model weights"),
            ),
        ):
            summary = _model_assets_hud_summary(
                root,
                project.task_id,
                "exp001",
                task_spec.competition_rules,
            )
    
        assert summary["registered"] == 1
        assert summary["eligible"] == 1
        assert summary["unverified"] == 1
        assert summary["cutoff_status"] == "ok"
        assert summary["summary"].startswith("Models [fixture-task]: 1 registered; 1 eligible, 1 unverified.")
        assert len(summary["summary"].split()) <= 80

if __name__ == "__main__":
    unittest.main()
