from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import yaml

from harness.agent_bridge import AgentBridge, _model_assets_hud_summary
from harness.bridge_common import (
    ActionObservation,
    ApprovalTier,
    TypedActionCall,
    TypedActionName,
    get_action_schema,
)
from harness.model_assets.cli import rebuild_model_index
from harness.model_assets.store import ModelStore
from harness.project import HarnessProject
from harness.task_spec import load_active_task_spec
from harness.cli import main as harness_main
from tests.fixtures import make_workspace, task_document


def _definition() -> dict:
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
        model_path.write_text(yaml.safe_dump(_definition(), sort_keys=False), encoding="utf-8")
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


def test_model_cli_register_list_inspect_and_validate_config(tmp_path, capsys):
    root, _, _ = _workspace(tmp_path, with_model=False)
    model_path = root / "configs" / "model.yaml"
    model_path.write_text(yaml.safe_dump(_definition(), sort_keys=False), encoding="utf-8")

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


def test_model_cli_verify_hashes_artifact_and_resolve_is_dry_run(tmp_path, capsys):
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


def test_model_bridge_actions_are_typed_and_dispatch_to_services(tmp_path):
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


def test_model_hud_is_compact_and_uses_index_and_selected_lock_only(tmp_path):
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
                "definition": _definition(),
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

