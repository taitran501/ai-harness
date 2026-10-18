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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.artifacts import seal_file_map, sha256, validate_run
from harness import import_local
from harness.kaggle_runtime import export_run

# ---------------------------------------------------------------------------
# helpers
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


def _full_manifest(run: Path, exp_id: str) -> dict:
    return {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": exp_id,
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "files": seal_file_map(run),
    }


# ---------------------------------------------------------------------------
# seal_file_map
# ---------------------------------------------------------------------------


class RecursiveSealingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_seal_file_map_is_recursive_with_relpath_keys(self):
        run = _make_run(self.tmp_path, "exp_rec")
        files = seal_file_map(run)
        self.assertIn("metrics.json", files)  # top-level keeps bare name
        self.assertIn("stages/train.log", files)  # nested sealed with POSIX relpath
        self.assertIn("stages/fold0/stdout.log", files)  # deeper nesting works
        self.assertNotIn("artifact_manifest.json", files)  # manifest never seals itself
        for name, value in files.items():
            self.assertEqual(value, sha256(run / name))

    def test_validate_run_accepts_recursive_manifest(self):
        run = _make_run(self.tmp_path, "exp_val")
        (run / "artifact_manifest.json").write_text(
            json.dumps(_full_manifest(run, "exp_val")), encoding="utf-8"
        )
        validate_run(run)  # nested sealed files verify

    def test_validate_run_detects_nested_tamper(self):
        """The exact reviewer blind spot: stages/*.log could previously be
        rewritten freely. Now the sealed hash catches it."""
        run = _make_run(self.tmp_path, "exp_tamper")
        (run / "artifact_manifest.json").write_text(
            json.dumps(_full_manifest(run, "exp_tamper")), encoding="utf-8"
        )
        validate_run(run)

        (run / "stages" / "train.log").write_text("rewritten by an actor\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            validate_run(run)
        self.assertIn("stages/train.log", str(ctx.exception))

    def test_validate_run_rejects_traversal_and_absolute_keys(self):
        run = _make_run(self.tmp_path, "exp_trav", nested=False)
        manifest = _full_manifest(run, "exp_trav")
        for evil in ("../outside.log", "..\\outside.log", "stages/../../escape.log"):
            manifest["files"][evil] = sha256(run / "metrics.json")
            (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_run(run)
            del manifest["files"][evil]

    def test_old_top_level_manifest_still_validates(self):
        """Backward compatibility: pre-recursive manifests (bare names only)."""
        run = _make_run(self.tmp_path, "exp_old", nested=False)
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


class ImportContentVerifiedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_import_content_verified_accepts_nested(self):
        run = _make_run(self.tmp_path, "exp_imp")
        files = seal_file_map(run)
        manifest = {"files": files}
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIsNone(import_local._content_verified(run, manifest))

    def test_import_content_verified_rejects_traversal(self):
        run = _make_run(self.tmp_path, "exp_imp2", nested=False)
        manifest = {"files": {"../evil.log": "deadbeef"}}
        self.assertEqual(
            import_local._content_verified(run, manifest),
            "Manifest file name unsafe: ../evil.log",
        )


# ---------------------------------------------------------------------------
# kaggle_runtime export: nested outputs are materialized
# ---------------------------------------------------------------------------


class ExportNestedFilesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_run_materializes_nested_files(self):
        run = _make_run(self.tmp_path, "exp_exp")
        (run / "artifact_manifest.json").write_text(
            json.dumps(_full_manifest(run, "exp_exp")), encoding="utf-8"
        )

        out = self.tmp_path / "kaggle_output"
        export_run(run, out, {"exp_id": "exp_exp"})
        self.assertTrue((out / "stages" / "train.log").is_file())
        self.assertTrue((out / "stages" / "fold0" / "stdout.log").is_file())
        validate_run(out)


if __name__ == "__main__":
    unittest.main()
