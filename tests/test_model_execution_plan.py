"""Execution-plan admission and cloud side-effect regression tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from harness.cloud_runner import _prepare_remote_run, run_via_kaggle
from harness.kernel_dispatch import dispatch_kernel
from harness.model_assets import (
    ModelAdmissionError,
    ModelStore,
    enforce_model_admission,
    prepare_or_validate_model_plan,
    resolve_effective_config,
)
from harness.project import HarnessProject, attach_project
from harness.runner import UniversalRunner
from harness.task_spec import load_active_task_spec
from tests.fixtures import TRAIN_CSV_4ROW, task_document


def _definition(count: int = 8) -> dict:
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
    model_path.write_text(yaml.safe_dump(_definition(), sort_keys=False), encoding="utf-8")
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


def test_base_inheritance_produces_identical_local_and_cloud_model_plans(tmp_path):
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


def test_cloud_preparation_uses_the_same_effective_plan_as_local(tmp_path):
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


def test_over_limit_model_is_rejected_before_plugin_or_model_setup(tmp_path):
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


def test_direct_kernel_dispatch_rejects_before_provider_push(tmp_path):
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


def test_validator_keeps_trainable_budget_separate_from_logical_total():
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


def test_kaggle_dry_run_does_not_upload_data_or_dispatch_kernel(tmp_path):
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
