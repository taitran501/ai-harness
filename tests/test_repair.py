"""Tests for `ai-harness repair` — re-sealing drifted artifact manifests.

The core scenario from the conversation audit: a completed run's artifact bytes
get modified after the run (IDE background process, manual edit, cloud re-import).
`validate_run()` then raises "Artifact hash mismatch" and the run becomes
invisible to research_memory. `repair` recomputes the per-file `files` map from
the bytes on disk and re-seals the manifest, without touching run identity.
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from harness.repair import (
    classify_run,
    repair_run_dir,
    scan_artifacts,
    main,
    build_parser,
)
from harness.evaluation import utility
from tests.evidence_fixtures import seal_fixture
from tests.test_research_continuity import task_document


def _seal_into(folder: Path, spec, score=0.8):
    """Write a complete, valid sealed run (mirrors what runner + seal_fixture produce)."""
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
    # seal_fixture adds the manifest with the correct per-file hashes
    seal_fixture(folder, spec, score)
    return folder


def _drift_metrics(folder: Path, spec, score=0.85):
    """Re-write metrics.json with a full-but-changed structure (keeps direction etc.)."""
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


def _make_project(tmp: str):
    """Create a minimal attached project with an artifacts dir and return it."""
    root = Path(tmp)
    (root / "configs").mkdir()
    (root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()))
    spec = __import__("harness.task_spec", fromlist=["UniversalTaskSpec"]).UniversalTaskSpec.from_dict(
        task_document()
    )
    # attach manifest so _find_artifacts_dir resolves the artifacts dir
    manifest = root / ".ml-harness"
    manifest.mkdir()
    (manifest / "project.yaml").write_text(
        yaml.safe_dump({"task_config": "configs/task.yaml", "artifacts_dir": "artifacts"}),
        encoding="utf-8",
    )
    artifacts = root / "artifacts"
    artifacts.mkdir()
    return root, spec, artifacts


class TestClassifyRun(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-repair-")
        self.root, self.spec, self.artifacts = _make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _seal(self, exp_id="exp_a"):
        return _seal_into(self.artifacts / exp_id, self.spec, 0.8)

    def test_verified_run_is_verified(self):
        folder = self._seal()
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "verified")

    def test_drifted_file_detected(self):
        folder = self._seal()
        # modify a declared file after sealing → its hash no longer matches
        _drift_metrics(folder, self.spec, 0.9)
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "drifted")
        self.assertIn("metrics.json", cls["drifted"])
        self.assertEqual(cls["missing_on_disk"], [])

    def test_missing_file_detected(self):
        folder = self._seal()
        (folder / "oof_predictions.csv").unlink()
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "drifted")
        self.assertIn("oof_predictions.csv", cls["missing_on_disk"])

    def test_unverifiable_not_drift(self):
        folder = self._seal()
        # break an identity field (not a file hash) → unverifiable, not drifted
        manifest_path = folder / "artifact_manifest.json"
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["status"] = "RUNNING"  # validate_run requires COMPLETED
        manifest_path.write_text(json.dumps(m))
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "unverifiable")
        self.assertIn("not a file-hash drift", cls["note"])


class TestRepair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-repair-fix-")
        self.root, self.spec, self.artifacts = _make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _seal(self, exp_id="exp_a"):
        return _seal_into(self.artifacts / exp_id, self.spec, 0.8)

    def test_repair_refreshes_hash_and_restores_validation(self):
        from harness.artifacts import validate_run

        folder = self._seal()
        # drift the bytes
        _drift_metrics(folder, self.spec, 0.85)
        self.assertEqual(classify_run(folder)["state"], "drifted")

        res = repair_run_dir(folder)
        self.assertTrue(res["now_verified"], res)
        # identity fields must be untouched
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "test-exp_a")
        self.assertEqual(manifest["status"], "COMPLETED")
        # and the run now validates
        validate_run(folder)

    def test_repair_does_not_add_undeclared_files(self):
        folder = self._seal()
        # an undeclared file appears on disk; repair must NOT add it to files
        (folder / "stray_file.txt").write_text("noise")
        _drift_metrics(folder, self.spec, 0.85)
        repair_run_dir(folder)
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("stray_file.txt", manifest["files"])
        self.assertIn("metrics.json", manifest["files"])

    def test_repair_drop_missing_removes_deleted_files(self):
        folder = self._seal()
        (folder / "oof_predictions.csv").unlink()
        res = repair_run_dir(folder, drop_missing=True)
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("oof_predictions.csv", manifest["files"])
        self.assertIn("oof_predictions.csv", res["dropped_missing"])

    def test_repair_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            repair_run_dir(self.artifacts / "nope")

    def test_repair_no_manifest_raises(self):
        folder = self.artifacts / "exp_nomanifest"
        folder.mkdir()
        (folder / "metrics.json").write_text("{}")
        with self.assertRaises(ValueError):
            repair_run_dir(folder)


class TestScanAndCli(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-repair-cli-")
        self.root, self.spec, self.artifacts = _make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _seal(self, exp_id):
        return _seal_into(self.artifacts / exp_id, self.spec, 0.8)

    def test_scan_counts_states(self):
        self._seal("good")
        bad = self._seal("bad")
        _drift_metrics(bad, self.spec, 0.99)
        scan = scan_artifacts(self.artifacts)
        self.assertEqual(scan["runs_total"], 2)
        self.assertEqual(len(scan["drifted"]), 1)
        self.assertEqual(scan["drifted"][0]["exp_dir"], "bad")
        self.assertIn("good", scan["verified"])

    def test_check_reports_without_writing(self):
        bad = self._seal("bad")
        before = (bad / "artifact_manifest.json").read_text(encoding="utf-8")
        _drift_metrics(bad, self.spec, 0.99)
        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            code = main(["--project-root", str(self.root), "--check", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["drifted"], 1)
        # manifest unchanged by --check
        self.assertEqual(before, (bad / "artifact_manifest.json").read_text(encoding="utf-8"))

    def test_repair_all_restores_drifted_run(self):
        from contextlib import redirect_stdout
        from io import StringIO

        bad = self._seal("bad")
        _drift_metrics(bad, self.spec, 0.99)
        buf = StringIO()
        with redirect_stdout(buf):
            code = main(["--project-root", str(self.root), "--all", "--json"])
        self.assertEqual(code, 0, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 1)
        self.assertTrue(payload["details"][0]["now_verified"])
        from harness.artifacts import validate_run

        validate_run(bad)

    def test_repair_specific_exp(self):
        from contextlib import redirect_stdout
        from io import StringIO

        self._seal("good")
        bad = self._seal("bad")
        _drift_metrics(bad, self.spec, 0.99)
        buf = StringIO()
        with redirect_stdout(buf):
            code = main(["--project-root", str(self.root), "--exp", "bad", "--json"])
        self.assertEqual(code, 0, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 1)
        self.assertEqual(payload["details"][0]["exp_dir"], "bad")

    def test_repair_nothing_to_do(self):
        from contextlib import redirect_stdout
        from io import StringIO

        self._seal("good")
        buf = StringIO()
        with redirect_stdout(buf):
            code = main(["--project-root", str(self.root), "--all", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 0)

    def test_not_attached_returns_2(self):
        with tempfile.TemporaryDirectory() as empty:
            from contextlib import redirect_stdout
            from io import StringIO

            buf = StringIO()
            with redirect_stdout(buf):
                code = main(["--project-root", empty, "--json"])
            self.assertEqual(code, 2)
            self.assertIn("error", json.loads(buf.getvalue()))


class TestParser(unittest.TestCase):
    def test_parser_flags(self):
        parser = build_parser()
        args = parser.parse_args(["--check", "--drop-missing", "--json", "--exp", "x"])
        self.assertTrue(args.check)
        self.assertTrue(args.drop_missing)
        self.assertTrue(args.json)
        self.assertEqual(args.exp, ["x"])
        self.assertFalse(args.all)

    def test_cli_registers_repair_subcommand(self):
        from harness.cli import build_parser as cli_parser

        p = cli_parser()
        args = p.parse_args(["repair"])
        self.assertEqual(args.command, "repair")


if __name__ == "__main__":
    unittest.main()
