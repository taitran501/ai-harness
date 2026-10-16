"""Local verified-run import (P1) and cross-workspace adoption scan (P2).

Covers the rsna-knee adoption gap: runs executed on a remote backend whose
outputs were downloaded to top-level directories (kernel_output_*,
kaggle_results) never reached the artifact tree, leaving the next agent
convo with zero visible evidence.

Guarantees under test:
- strict identity match -> import into the live artifact tree;
- task/scorer identity drift -> quarantine with IMPORT_RECEIPT, never mixed
  into the live tree, content still hash-verified;
- tampered / missing files -> rejected, nothing copied;
- existing evidence is never overwritten;
- --scan classifies candidates without side effects;
- adoption-scan aggregates per-workspace verdicts + unimported counts.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness.import_local import import_verified_local_run, scan_importable_runs
from harness.project import HarnessProject
from harness.task_spec import load_active_task_spec
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace


def _spec_for(root: Path):
    return load_active_task_spec(HarnessProject.discover(root).task_config_path)


def _make_run(root: Path, name: str, *, stale_task: bool = False, stale_scorer: bool = False) -> Path:
    """Create a verified local run dir (fresh identity unless stale)."""
    folder = root / name
    spec = _spec_for(root)
    seal_fixture(folder, spec)
    if stale_task or stale_scorer:
        manifest_path = folder / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stale_task:
            manifest["task_sha256"] = "0" * 64
            manifest["scorer_sha256"] = "1" * 64
        else:
            manifest["scorer_sha256"] = "1" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class ScanImportableRunsTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = make_workspace()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_scan_classifies_strict_stale_invalid_and_imported(self):
        _make_run(self.root, "kernel_output_exp001")
        _make_run(self.root, "kernel_output_exp002", stale_task=True)
        broken = self.root / "kernel_output_exp003"
        broken.mkdir()
        (broken / "artifact_manifest.json").write_text("{}", encoding="utf-8")
        entries = scan_importable_runs(self.root)
        by_dir = {e["dir"]: e for e in entries}
        self.assertEqual(by_dir["kernel_output_exp001"]["verdict"], "importable")
        self.assertEqual(by_dir["kernel_output_exp002"]["verdict"], "quarantine")
        self.assertIn("drift", by_dir["kernel_output_exp002"]["detail"])
        self.assertEqual(by_dir["kernel_output_exp003"]["verdict"], "invalid")
        self.assertNotIn("dataset", by_dir)  # no manifest -> not a candidate

    def test_scan_marks_already_imported(self):
        source = _make_run(self.root, "kernel_output_exp001")
        result = import_verified_local_run(self.root, source)
        entries = scan_importable_runs(self.root)
        by_dir = {e["dir"]: e for e in entries}
        self.assertEqual(by_dir["kernel_output_exp001"]["verdict"], "already_imported")
        self.assertEqual(result["quarantined"], False)


class ImportVerifiedLocalRunTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = make_workspace()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_strict_import_lands_in_live_tree_with_receipt(self):
        source = _make_run(self.root, "kernel_output_exp001")
        result = import_verified_local_run(self.root, source)
        target = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp001"
        self.assertEqual(result["target"], str(target))
        self.assertFalse(result["quarantined"])
        self.assertTrue((target / "metrics.json").is_file())
        receipt = json.loads((target / "IMPORT_RECEIPT.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["content_verified"])
        self.assertFalse(receipt["quarantined"])
        self.assertEqual(receipt["identity"]["task_sha256"], "match")
        # journal audit entry recorded
        journal = json.loads((self.root / ".ai-harness" / "artifacts" / "fixture-task" / "journal.json").read_text())
        self.assertEqual(journal[-1]["event"], "local_artifact_import")

    def test_stale_identity_quarantined_not_live(self):
        source = _make_run(self.root, "kernel_output_exp002", stale_task=True)
        result = import_verified_local_run(self.root, source)
        live = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp002"
        quarantine = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "quarantine" / "kernel_output_exp002"
        self.assertTrue(quarantine.is_dir())
        self.assertFalse(live.is_dir())
        self.assertTrue(result["quarantined"])
        self.assertIn("task_sha256", result["drift"])
        receipt = json.loads((quarantine / "IMPORT_RECEIPT.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["content_verified"])
        self.assertIn("drift", receipt["identity"]["task_sha256"])
        self.assertEqual(receipt["reason"], "identity drift vs attached task spec")

    def test_reject_stale_raises(self):
        source = _make_run(self.root, "kernel_output_exp002", stale_task=True)
        with self.assertRaisesRegex(ValueError, "identity drift"):
            import_verified_local_run(self.root, source, reject_stale=True)

    def test_tampered_file_rejected_nothing_copied(self):
        source = _make_run(self.root, "kernel_output_exp001")
        (source / "oof_predictions.csv").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            import_verified_local_run(self.root, source)
        target = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp001"
        self.assertFalse(target.exists())

    def test_missing_manifest_file_rejected(self):
        source = _make_run(self.root, "kernel_output_exp001")
        (source / "metrics.json").unlink()
        with self.assertRaisesRegex(ValueError, "content integrity"):
            import_verified_local_run(self.root, source)

    def test_never_overwrites_existing_evidence(self):
        source = _make_run(self.root, "kernel_output_exp001")
        import_verified_local_run(self.root, source)
        source2 = self.root / "kernel_output_exp001_copy"
        shutil.copytree(source, source2)
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            import_verified_local_run(self.root, source2)
        # original receipt untouched
        receipt = json.loads(
            (self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp001" / "IMPORT_RECEIPT.json").read_text()
        )
        self.assertEqual(receipt["source_dir"], str(source))

    def test_unsafe_manifest_filename_rejected(self):
        source = _make_run(self.root, "kernel_output_exp001")
        manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["../evil.csv"] = "0" * 64
        (source / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            import_verified_local_run(self.root, source)


class AdoptionScanTests(unittest.TestCase):
    def test_aggregates_workspaces_and_unimported_counts(self):
        parent = Path(tempfile.mkdtemp(prefix="aih-adoption-scan-"))
        ws_a_target = parent / "ws_a"
        ws_b_target = parent / "ws_b"
        try:
            ws_a, _ = make_workspace(prefix="aih-scan-a-")
            shutil.move(str(ws_a), str(ws_a_target))
            ws_b, _ = make_workspace(prefix="aih-scan-b-")
            shutil.move(str(ws_b), str(ws_b_target))
            # unimported verified run in ws_a
            _make_run(ws_a_target, "kernel_output_exp001", stale_task=True)

            from harness.cli import _run_adoption_scan

            import argparse
            import contextlib
            import io

            args = argparse.Namespace(root=str(parent), include=None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _run_adoption_scan(args)
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            by_name = {r["workspace"]: r for r in payload["workspaces"]}
            self.assertEqual(len(by_name), 2)
            row_a = by_name["ws_a"]
            self.assertEqual(row_a["unimported_count"], 1)
            self.assertEqual(row_a["unimported_runs"][0]["verdict"], "quarantine")
            self.assertEqual(by_name["ws_b"]["unimported_count"], 0)
            self.assertEqual(payload["summary"]["no_activity"], 2)
        finally:
            shutil.rmtree(str(parent), ignore_errors=True)

    def test_imported_source_record_written(self):
        root, _ = make_workspace()
        source = _make_run(root, "kernel_output_exp001", stale_task=True)
        import_verified_local_run(root, source)
        task_root = root / ".ai-harness" / "artifacts" / "fixture-task"
        payload = json.loads((task_root / "imported_sources.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["source_dir"], "kernel_output_exp001")
        self.assertTrue(payload[0]["quarantined"])
        # duplicate import of a different source adds, same source does not duplicate
        import_verified_local_run(root, _make_run(root, "kaggle_results", stale_scorer=True))
        payload = json.loads((task_root / "imported_sources.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 2)

    def test_quarantined_source_is_already_imported_in_scan(self):
        root, _ = make_workspace()
        source = _make_run(root, "kernel_output_exp001", stale_task=True)
        import_verified_local_run(root, source)
        entries = scan_importable_runs(root)
        by_dir = {e["dir"]: e for e in entries}
        self.assertEqual(by_dir["kernel_output_exp001"]["verdict"], "already_imported")

    def test_recovered_dir_not_flagged_as_bypass(self):
        from harness.hygiene import scan_workspace_hygiene

        root, _ = make_workspace()
        source = _make_run(root, "kernel_output_exp001", stale_task=True)
        # a second manual unsealed push (recovered finding needs >2 run dirs to aggregate)
        (root / "kaggle_sub_exp002_apex").mkdir()
        # a third manual unsealed push that was NOT recovered
        (root / "kaggle_sub_exp999_apex").mkdir()
        (root / "kaggle_sub_exp999_apex" / "out.csv").write_text("x\n")
        import_verified_local_run(root, source)

        findings = scan_workspace_hygiene(root)["findings"]
        kinds = [f["kind"] for f in findings]
        self.assertIn("recovered_run_output", kinds)
        self.assertIn("bypassed_run_output", kinds)
        # recovered dir must NOT appear in the bypass finding
        for f in findings:
            if f["kind"] == "bypassed_run_output":
                self.assertNotIn("kernel_output_exp001", f["detail"])
                self.assertIn("kaggle_sub_exp002_apex", f["detail"])
                self.assertIn("kaggle_sub_exp999_apex", f["detail"])


if __name__ == "__main__":
    unittest.main()
