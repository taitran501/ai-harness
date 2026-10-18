"""Regression tests for recursive artifact sealing (PR-B, finding M2).

Reviewer scenario M2: pipeline runs write per-stage evidence under
``stages/*.log`` which was NEVER included in the sealed ``files`` map
(top-level ``iterdir`` only). The fix seals the full recursive surface:
top-level files keep bare names, nested files use POSIX relative paths
(``stages/train.log``), and every consumer (validate_run, repair, import,
Kaggle export/import, notebook validation) accepts safe relative paths
while still rejecting traversal.

Backward compatibility: old top-level-only manifests validate exactly as
before, and repair never adopts undeclared files (bundle exclusivity).
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.artifacts import seal_file_map, sha256, validate_run
from harness import import_local
from harness.kaggle_runtime import export_run

# ---------------------------------------------------------------------------
# seal_file_map
# ---------------------------------------------------------------------------


def _make_run(root: Path, name: str, nested: bool = True) -> Path:
    run = root / name
    (run / "stages").mkdir(parents=True)
    (run / "run_config.yaml").write_text("model: {name: fixture}\n", encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({"primary_score": 0.5, "direction": "minimize", "experiment_id": name}),
        encoding="utf-8",
    )
    (run / "oof_predictions.csv").write_text("id,pred_y\n0,0.5\n", encoding="utf-8")
    (run / "evaluation_truth.csv").write_text("id,y\n0,0.0\n", encoding="utf-8")
    if nested:
        (run / "stages" / "fold0").mkdir(parents=True)
        (run / "stages" / "train.log").write_text("stage train ok\n", encoding="utf-8")
        (run / "stages" / "fold0" / "stdout.log").write_text(
            "fold output\n", encoding="utf-8"
        )
    return run


def test_seal_file_map_is_recursive_with_relpath_keys(tmp_path: Path):
    run = _make_run(tmp_path, "exp_rec")
    files = seal_file_map(run)
    assert "metrics.json" in files  # top-level keeps bare name
    assert "stages/train.log" in files  # nested sealed with POSIX relpath
    assert "stages/fold0/stdout.log" in files  # deeper nesting works
    assert "artifact_manifest.json" not in files  # manifest never seals itself
    for name, value in files.items():
        assert value == sha256(run / name)


def test_validate_run_accepts_recursive_manifest(tmp_path: Path):
    run = _make_run(tmp_path, "exp_val")
    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": "exp_val",
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "files": seal_file_map(run),
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validate_run(run)  # nested sealed files verify


def test_validate_run_detects_nested_tamper(tmp_path: Path):
    """The exact reviewer blind spot: stages/*.log could previously be
    rewritten freely. Now the sealed hash catches it."""
    run = _make_run(tmp_path, "exp_tamper")
    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": "exp_tamper",
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "files": seal_file_map(run),
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validate_run(run)

    (run / "stages" / "train.log").write_text("rewritten by an actor\n", encoding="utf-8")
    try:
        validate_run(run)
    except ValueError as exc:
        assert "stages/train.log" in str(exc)
    else:
        raise AssertionError("nested tamper was not detected")


def test_validate_run_rejects_traversal_and_absolute_keys(tmp_path: Path):
    run = _make_run(tmp_path, "exp_trav", nested=False)
    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": "exp_trav",
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "files": seal_file_map(run),
    }
    for evil in ("../outside.log", "..\\outside.log", "stages/../../escape.log"):
        manifest["files"][evil] = sha256(run / "metrics.json")
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            validate_run(run)
        except ValueError:
            pass
        else:
            raise AssertionError(f"traversal key accepted: {evil}")
        del manifest["files"][evil]


def test_old_top_level_manifest_still_validates(tmp_path: Path):
    """Backward compatibility: pre-recursive manifests (bare names only)."""
    run = _make_run(tmp_path, "exp_old", nested=False)
    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": "exp_old",
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "files": {
            p.name: sha256(p) for p in run.iterdir() if p.is_file()
        },
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validate_run(run)


# ---------------------------------------------------------------------------
# import_local: safe relpaths accepted, traversal rejected
# ---------------------------------------------------------------------------


def test_import_content_verified_accepts_nested(tmp_path: Path):
    run = _make_run(tmp_path, "exp_imp")
    files = seal_file_map(run)
    manifest = {"files": files}
    (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert import_local._content_verified(run, manifest) is None


def test_import_content_verified_rejects_traversal(tmp_path: Path):
    run = _make_run(tmp_path, "exp_imp2", nested=False)
    manifest = {"files": {"../evil.log": "deadbeef"}}
    assert import_local._content_verified(run, manifest) == "Manifest file name unsafe: ../evil.log"


# ---------------------------------------------------------------------------
# kaggle_runtime export: nested outputs are materialized
# ---------------------------------------------------------------------------


def test_export_run_materializes_nested_files(tmp_path: Path):
    run = _make_run(tmp_path, "exp_exp")
    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": "exp_exp",
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "files": seal_file_map(run),
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out = tmp_path / "kaggle_output"
    export_run(run, out, {"exp_id": "exp_exp"})
    assert (out / "stages" / "train.log").is_file()
    assert (out / "stages" / "fold0" / "stdout.log").is_file()
    validate_run(out)
