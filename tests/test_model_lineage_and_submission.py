"""Candidate model closure and SubmissionGate binding regressions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from harness.artifacts import seal_file_map
from harness.candidate_preflight import load_preflight_receipt, run_candidate_preflight
from harness.model_assets import ModelDefinition
from harness.model_assets.lineage import (
    evaluate_candidate_model_policy,
    model_policy_digest,
    resolve_candidate_model_closure,
)
from harness.submission_gate import SubmissionGate, SubmissionGateError
from tests.fixtures import make_submission_evidence, make_workspace, task_document, workspace_with_verified_pair


def _canonical_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _definition(model_id: str, count: int) -> dict:
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
    definition = _definition(model_id, count)
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


def test_aggregate_policy_rejects_individually_compliant_over_cap_blend():
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


def test_preflight_receipt_v3_binds_subject_lock_and_policy(tmp_path):
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


def test_preflight_reader_keeps_v1_receipts_readable(tmp_path):
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


def test_submission_gate_rejects_over_cap_model_closure_before_permit(tmp_path):
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
            definition = _definition(model_id, 6)
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


def test_dispatch_permit_persists_preflight_subject_and_model_lock_binding():
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

