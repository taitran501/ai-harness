"""Regression tests for the repair evidence-integrity gate (PR-B).

Reviewer scenario C1: a doctored `metrics.json` (or OOF/truth) was being
re-sealed by `ai-harness repair` — the tampered score became the sealed
truth again. After the fix:

* evidence drift (metrics.json / OOF / predictions / truth) is NEVER
  re-sealed by default: the run stays visibly broken (exit 3);
* `--verify` re-derives the primary score from on-disk OOF + truth through
  the runner's own scoring path and re-seals only on an exact match;
* `--force` is an operator override, always recorded in the append-only
  `repair_log.jsonl` outside the sealed bundle;
* every re-seal leaves `repaired: true` + `repaired_at` markers in the
  manifest (re-seal provenance).

The fixture below is a minimal *regression* run (rmse) whose primary score
is fully re-derivable from `oof_predictions.csv` + `evaluation_truth.csv`
through the default declarative plugin — the same code path the runner used
to produce the recorded score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from harness.artifacts import sha256, validate_run
from harness import repair as repair_mod
from tests.fixtures import make_workspace

# ---------------------------------------------------------------------------
# Fixture: a minimal sealed regression run
# ---------------------------------------------------------------------------

TRUTH_CSV = "id,y\n0,0.0\n1,1.0\n2,2.0\n3,3.0\n"
# pred errors vs truth [0,1,2,3]: [0.5, 0.5, -0.5, 0.5] -> rmse = 0.5
OOF_CSV = "id,pred_y,input_variant\n0,0.5,v1\n1,1.5,v1\n2,1.5,v1\n3,3.5,v1\n"
EXPECTED_SCORE = 0.5


def _regression_task_doc():
    return {
        "task": {
            "id": "repair-integrity-task",
            "name": "Repair Integrity Task",
            "task_type": "regression",
            "type": "regression",
            "primary_metric": "rmse",
            "direction": "minimize",
            "data_config": {
                "train_path": "dataset/train.csv",
                "target_column": "y",
                "id_column": "id",
            },
            "validation": {"strategy": "kfold", "num_folds": 2},
            "subtasks": [
                {
                    "id": "target",
                    "name": "target",
                    "task_type": "regression",
                    "column": "y",
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "weight": 1.0,
                }
            ],
        }
    }


def _write_run(root: Path, exp_id: str) -> Path:
    """Write one sealed, verified run folder with a re-derivable score."""
    artifacts_dir = root / ".ai-harness" / "artifacts" / "repair-integrity-task"
    run_dir = artifacts_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = run_dir / "run_config.yaml"
    run_config.write_text(
        yaml.safe_dump({"model": {"name": "fixture-lr"}, "estimator": "linear"}, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "evaluation_truth.csv").write_text(TRUTH_CSV, encoding="utf-8")
    (run_dir / "oof_predictions.csv").write_text(OOF_CSV, encoding="utf-8")
    card = run_dir / "EXPERIMENT_CARD.md"
    card.write_text(f"# {exp_id}\nfixture evidence card\n", encoding="utf-8")

    metrics = {
        "primary_score": EXPECTED_SCORE,
        "direction": "minimize",
        "experiment_id": exp_id,
        "selection_variant": "v1",
        "subtask_scores": {"target": EXPECTED_SCORE},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": f"run-{exp_id}",
        "exp_id": exp_id,
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run_config),
        "files": {
            p.name: sha256(p)
            for p in (
                run_config,
                run_dir / "metrics.json",
                run_dir / "oof_predictions.csv",
                run_dir / "evaluation_truth.csv",
                card,
            )
        },
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir


@pytest.fixture()
def workspace(tmp_path: Path):
    root, _task_config = make_workspace(task=_regression_task_doc())
    return root


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_clean_run_verifies(workspace: Path):
    run_dir = _write_run(workspace, "exp001_clean")
    validate_run(run_dir)  # no exception
    assert repair_mod.classify_run(run_dir)["state"] == "verified"


def test_rederivation_matches_recorded_score(workspace: Path):
    """The gate's re-derivation path reproduces the runner's score exactly."""
    run_dir = _write_run(workspace, "exp002_rederive")
    from harness.oof_revalidation import rederive_primary_score
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.project import HarnessProject
    from harness.task_spec import load_active_task_spec

    project = HarnessProject.discover(workspace)
    spec = load_active_task_spec(project.task_config_path)
    plugin_context = TaskPluginContext(
        project.root, project.task_config_path, spec, project.plugin_id or "default"
    )
    plugin = load_task_plugin(project.plugin_id, plugin_context)

    report = rederive_primary_score(
        run_dir, task_spec=spec, plugin=plugin, plugin_context=plugin_context
    )
    assert report["verifiable"] is True
    assert report["ok"] is True
    assert report["recorded"] == EXPECTED_SCORE
    assert report["recomputed"] == pytest.approx(EXPECTED_SCORE)


# ---------------------------------------------------------------------------
# C1: tampered evidence must NOT be laundered by default repair
# ---------------------------------------------------------------------------


def test_tampered_metrics_blocked_by_default(workspace: Path, capsys):
    run_dir = _write_run(workspace, "exp003_tampered")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["primary_score"] = 9.99  # doctored score
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rc = repair_mod.run_repair(project_root=str(workspace), exp=["exp003_tampered"])
    assert rc == 3  # blocked by the evidence integrity gate

    out = capsys.readouterr().out
    assert "blocked_evidence_drift" in out

    # the tampered score was NOT re-sealed: validation still fails,
    # the manifest is untouched (no repaired marker)
    with pytest.raises(ValueError):
        validate_run(run_dir)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("repaired") is None
    # the declared hash is still the ORIGINAL (clean) one, not the tampered bytes
    original = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert original["primary_score"] == 9.99  # file on disk is tampered...
    assert manifest["files"]["metrics.json"] != sha256(metrics_path)  # ...but not sealed

    # audit trail recorded the blocked attempt
    log_path = (
        workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
    )
    assert log_path.is_file()
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["status"] == "blocked_evidence_drift" for e in entries)
    blocked = next(e for e in entries if e["status"] == "blocked_evidence_drift")
    assert "metrics.json" in blocked["evidence_files"]


def test_benign_drift_still_repairs(workspace: Path, capsys):
    """Non-evidence drift keeps the classic repair behavior (+ audit trail)."""
    run_dir = _write_run(workspace, "exp004_benign")
    card = run_dir / "EXPERIMENT_CARD.md"
    card.write_text("# updated card\n", encoding="utf-8")

    rc = repair_mod.run_repair(project_root=str(workspace), exp=["exp004_benign"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Repaired 1" in out

    validate_run(run_dir)  # restored
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repaired"] is True
    assert manifest["repaired_at"]
    assert manifest["files"]["EXPERIMENT_CARD.md"] == sha256(card)

    log_path = (
        workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
    )
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["exp_dir"] == "exp004_benign"
    assert entries[0]["mode"] == "resave"
    assert "EXPERIMENT_CARD.md" in entries[0]["files_changed"]


# ---------------------------------------------------------------------------
# --verify: re-derivation gate
# ---------------------------------------------------------------------------


def test_verify_reseals_when_score_matches(workspace: Path, capsys):
    """Evidence hash drift with an UNCHANGED score passes --verify."""
    run_dir = _write_run(workspace, "exp005_verify_ok")
    metrics_path = run_dir / "metrics.json"
    # re-serialize with extra whitespace: hash drifts, score is intact
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True), encoding="utf-8")

    rc = repair_mod.run_repair(
        project_root=str(workspace), exp=["exp005_verify_ok"], verify=True
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Repaired 1" in out
    assert "recorded=" in out  # metric check printed

    validate_run(run_dir)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repaired"] is True
    assert manifest["files"]["metrics.json"] == sha256(metrics_path)

    log_path = (
        workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
    )
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    check = entries[0]["metric_check"]
    assert check["verifiable"] is True
    assert check["ok"] is True
    assert check["recorded"] == EXPECTED_SCORE
    assert check["recomputed"] == pytest.approx(EXPECTED_SCORE)


def test_verify_blocks_metric_mismatch(workspace: Path, capsys):
    """A doctored score fails re-derivation: both values are reported."""
    run_dir = _write_run(workspace, "exp006_mismatch")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["primary_score"] = 0.1  # doctored
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rc = repair_mod.run_repair(
        project_root=str(workspace), exp=["exp006_mismatch"], verify=True
    )
    assert rc == 3
    out = capsys.readouterr().out
    assert "metric_mismatch" in out
    assert "recorded=0.1" in out
    assert f"recomputed={EXPECTED_SCORE}" in out

    # NOT re-sealed
    with pytest.raises(ValueError):
        validate_run(run_dir)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("repaired") is None

    log_path = (
        workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
    )
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["status"] == "metric_mismatch" for e in entries)
    mismatch = next(e for e in entries if e["status"] == "metric_mismatch")
    assert mismatch["metric_check"]["recorded"] == 0.1
    assert mismatch["metric_check"]["recomputed"] == pytest.approx(EXPECTED_SCORE)


def test_force_reseals_but_is_audited(workspace: Path, capsys):
    """--force is an explicit operator override, never silent."""
    run_dir = _write_run(workspace, "exp007_force")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["primary_score"] = 0.1  # doctored
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rc = repair_mod.run_repair(
        project_root=str(workspace), exp=["exp007_force"], force=True
    )
    assert rc == 0
    validate_run(run_dir)  # the bytes are now sealed (operator's call)

    log_path = (
        workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
    )
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["forced"] is True
    assert "exp007_force" in entries[0]["exp_dir"]


def test_missing_evidence_file_blocked_and_not_verifiable(workspace: Path, capsys):
    """Deleting an evidence file is also evidence drift — and not re-derivable."""
    run_dir = _write_run(workspace, "exp008_missing")
    (run_dir / "oof_predictions.csv").unlink()

    # default: blocked
    rc = repair_mod.run_repair(project_root=str(workspace), exp=["exp008_missing"])
    assert rc == 3

    # --verify: the score cannot be re-derived without the evidence
    rc = repair_mod.run_repair(
        project_root=str(workspace), exp=["exp008_missing"], verify=True
    )
    out = capsys.readouterr().out
    assert rc == 3
    assert "not_verifiable" in out

    with pytest.raises(ValueError):
        validate_run(run_dir)


# ---------------------------------------------------------------------------
# Audit trail properties
# ---------------------------------------------------------------------------


def test_repair_log_lives_outside_every_sealed_bundle(workspace: Path, capsys):
    run_dir = _write_run(workspace, "exp009_log")
    card = run_dir / "EXPERIMENT_CARD.md"
    card.write_text("v2\n", encoding="utf-8")
    repair_mod.run_repair(project_root=str(workspace), exp=["exp009_log"])
    capsys.readouterr()

    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert "repair_log.jsonl" not in manifest["files"]

    # a second repair appends (append-only), never overwrites
    card.write_text("v3\n", encoding="utf-8")
    repair_mod.run_repair(project_root=str(workspace), exp=["exp009_log"])
    capsys.readouterr()
    log_path = (
        workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
    )
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["exp_dir"] == "exp009_log"
    assert json.loads(lines[1])["exp_dir"] == "exp009_log"
