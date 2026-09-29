"""Unit tests for Submission Lifecycle Tracking and Cloud Pre-flight Inspection."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from harness.cloud_inspector import CloudPreflightInspector
from harness.submission_manager import SubmissionEntry, SubmissionManager
from harness.templates.kaggle_template import KaggleKernelManager


class SubmissionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)
        self.sub_dir = self.p / "submissions"
        self.mgr = SubmissionManager(submissions_dir=str(self.sub_dir))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_id_generation(self):
        self.assertEqual(self.mgr.get_next_submission_id(), "sub_001")
        entry = SubmissionEntry(
            id="sub_001_exp001",
            exp_id="exp001",
            created_at="2026-09-24",
            status="LOCAL_VERIFIED",
        )
        self.mgr.submissions[entry.id] = entry
        self.assertEqual(self.mgr.get_next_submission_id(), "sub_002")

    def test_record_official_score_and_divergence(self):
        entry = SubmissionEntry(
            id="sub_001_exp001",
            exp_id="exp001",
            created_at="2026-09-24",
            status="LOCAL_VERIFIED",
            local_cv_scores={"subtask_a_macro": 0.7000, "subtask_b_macro": 0.5500},
        )
        self.mgr.submissions[entry.id] = entry

        # Case 1: Normal score (LB = 0.7100, delta = +0.0100)
        updated = self.mgr.record_official_score(
            sub_id="sub_001_exp001",
            scores={"subtask_a_macro": 0.7100, "subtask_b_macro": 0.5400},
            split_type="public",
        )
        self.assertEqual(updated.status, "SCORED")
        self.assertEqual(updated.score_deltas["subtask_a_macro"], 0.0100)
        self.assertEqual(updated.score_deltas["subtask_b_macro"], -0.0100)
        self.assertFalse(updated.divergence_warning)

        # Case 2: Divergence (LB drops to 0.6200, delta = -0.0800 < -0.05)
        updated_div = self.mgr.record_official_score(
            sub_id="sub_001_exp001",
            scores={"subtask_a_macro": 0.6200, "subtask_b_macro": 0.5400},
        )
        self.assertTrue(updated_div.divergence_warning)
        self.assertEqual(updated_div.score_deltas["subtask_a_macro"], -0.0800)

    def test_render_submissions_markdown(self):
        entry = SubmissionEntry(
            id="sub_001_exp001",
            exp_id="exp001",
            created_at="2026-09-24",
            status="LOCAL_VERIFIED",
            local_cv_scores={"subtask_a_macro": 0.6962, "subtask_b_macro": 0.5238},
        )
        self.mgr.submissions[entry.id] = entry
        md = self.mgr.render_submissions_markdown()

        self.assertIn("Submissions & Leaderboard Registry", md)
        self.assertIn("sub_001_exp001", md)
        self.assertIn("LOCAL_VERIFIED", md)
        self.assertTrue((self.sub_dir / "SUBMISSIONS.md").exists())


class CloudPreflightInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inspect_config(self):
        cfg_path = self.p / "test_cfg.yaml"
        # Invalid config missing branch
        cfg_path.write_text("experiment:\n  id: exp999\n", encoding="utf-8")
        ok, errs, _ = CloudPreflightInspector.inspect_config(cfg_path)
        self.assertFalse(ok)
        self.assertTrue(any("branch" in e for e in errs))

        # Valid config
        cfg_path.write_text("experiment:\n  id: exp999\n  branch: feat\n", encoding="utf-8")
        ok, errs, _ = CloudPreflightInspector.inspect_config(cfg_path)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

    def test_inspect_python_syntax(self):
        bad_py = self.p / "bad.py"
        bad_py.write_text("def broken_func(\n", encoding="utf-8")

        good_py = self.p / "good.py"
        good_py.write_text("def fine_func():\n    return 42\n", encoding="utf-8")

        ok, errs = CloudPreflightInspector.inspect_python_syntax([bad_py, good_py])
        self.assertFalse(ok)
        self.assertEqual(len(errs), 1)
        self.assertIn("Syntax error", errs[0])


class KaggleKernelManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_generation(self):
        meta = KaggleKernelManager.generate_kernel_metadata("test-kernel", enable_gpu=True)
        uname = KaggleKernelManager.get_authenticated_username()
        self.assertEqual(meta["id"], f"{uname}/test-kernel")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(meta["enable_internet"], "true")

    def test_migrate_downloaded_artifacts(self):
        download_d = self.p / "download"
        download_d.mkdir()

        # Create essential files and non-essential large checkpoint
        (download_d / "oof_predictions.csv").write_text("id,pred\n1,yes\n", encoding="utf-8")
        (download_d / "metrics.json").write_text("{}", encoding="utf-8")
        (download_d / "heavy_model.pt").write_text("BINARY DUMP", encoding="utf-8")

        target_d = self.p / "artifacts" / "exp_test"
        migrated = KaggleKernelManager.migrate_downloaded_artifacts(download_d, target_d)

        self.assertIn("oof_predictions.csv", migrated)
        self.assertIn("metrics.json", migrated)
        self.assertNotIn("heavy_model.pt", migrated)
        self.assertTrue((target_d / "oof_predictions.csv").exists())
        self.assertFalse((target_d / "heavy_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
