"""Unit tests for the ML Experiment Harness (TreeJournal, Validator, Audit Gate)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from harness.tree_journal import Node, TreeJournal
from harness.validator import ExperimentValidator
from tools.audit_gate import (
    audit_prediction_integrity,
    compute_diff_metrics,
    compute_sha256,
)


class TreeJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.temp_dir.name) / "test_journal.json"
        self.journal = TreeJournal(storage_path=str(self.journal_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_and_load(self):
        node = Node(
            verified=True,eligible=True,
            id="test_node_1",
            parent_id=None,
            branch="baseline",
            atomic_hypothesis="Test hypothesis",
            metrics={"overall_composite": 0.75},
            param_count=1000,
        )
        self.journal.append(node)

        # Reload from disk
        reloaded = TreeJournal(storage_path=str(self.journal_path))
        self.assertIn("test_node_1", reloaded.nodes)
        self.assertEqual(reloaded.nodes["test_node_1"].get_score(), 0.75)

    def test_50_nodes_context_compression(self):
        """Verify that 50 experiments are compressed into a compact summary without bloat."""
        for i in range(50):
            node = Node(
            verified=True,eligible=True,
                id=f"exp{i:03d}_test",
                parent_id=f"exp{max(0, i-1):03d}_test" if i > 0 else None,
                branch="feat",
                atomic_hypothesis=f"Atomic change #{i} tested on validation set.",
                metrics={"overall_composite": 0.5 + (i * 0.005)},
                param_count=10_000_000,
            )
            self.journal.append(node)

        summary = self.journal.generate_compact_summary(max_nodes=10)
        self.assertIn("Total Runs: 50", summary)
        # Verify line count is bounded and token-friendly
        self.assertLess(len(summary.splitlines()), 25)

    def test_recommend_next_branch_respects_max_params(self):
        # Add a node exceeding parameter limits
        heavy_node = Node(
            verified=True,eligible=True,
            id="exp_heavy",
            branch="model",
            metrics={"overall_composite": 0.99},
            param_count=500_000_000,
        )
        # Add a light compliant node
        light_node = Node(
            verified=True,eligible=True,
            id="exp_light",
            branch="baseline",
            metrics={"overall_composite": 0.70},
            param_count=100_000_000,
        )
        self.journal.append(heavy_node)
        self.journal.append(light_node)

        # Recommendation with max_params=150M must NOT recommend the heavy node
        rec = self.journal.recommend_next_branch(max_params=150_000_000)
        self.assertEqual(rec["parent_id"], "exp_light")


class ValidatorTests(unittest.TestCase):
    def test_assert_parameter_limit_exceeded(self):
        class DummyModel:
            def num_parameters(self):
                return 200_000_000

        model = DummyModel()
        with self.assertRaises(ValueError):
            ExperimentValidator.assert_parameter_limit(model, max_params=150_000_000)

    def test_assert_parameter_limit_passed(self):
        class DummyModel:
            def num_parameters(self):
                return 50_000_000

        model = DummyModel()
        count = ExperimentValidator.assert_parameter_limit(model, max_params=150_000_000)
        self.assertEqual(count, 50_000_000)

    def test_assert_no_video_leakage_detected(self):
        train_df = pd.DataFrame({"yt_title": ["Video A", "Video B", "Video C"]})
        val_df = pd.DataFrame({"yt_title": ["Video C", "Video D"]})  # Video C leaked!

        with self.assertRaises(AssertionError):
            ExperimentValidator.assert_no_video_data_leakage(train_df, val_df)

    def test_assert_no_video_leakage_clean(self):
        train_df = pd.DataFrame({"yt_title": ["Video A", "Video B"]})
        val_df = pd.DataFrame({"yt_title": ["Video C", "Video D"]})
        # Should not raise
        ExperimentValidator.assert_no_video_data_leakage(train_df, val_df)

    def test_assert_subtask_c_oracle_conditioning(self):
        clean_df = pd.DataFrame(
            {
                "hate_speech": ["no", "yes_explicit"],
                "target_type": ["none", "individual"],
            }
        )
        ExperimentValidator.assert_subtask_c_oracle_conditioning(clean_df)

        violating_df = pd.DataFrame(
            {
                "hate_speech": ["no", "yes_explicit"],
                "target_type": ["group", "individual"],  # 'no' has 'group'!
            }
        )
        with self.assertRaises(AssertionError):
            ExperimentValidator.assert_subtask_c_oracle_conditioning(violating_df)


class AuditGateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_diff_metrics(self):
        cand_df = pd.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "pred_label": ["pos", "neg", "pos"],
            }
        )
        base_df = pd.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "pred_label": ["pos", "pos", "pos"],  # row 2 differs
            }
        )
        diff_count, details = compute_diff_metrics(cand_df, base_df,id_col="sample_id")
        self.assertEqual(diff_count, 1)
        self.assertEqual(details["pred_label"]["different_count"], 1)

    def test_audit_gate_cli_duplicate_failure(self):
        """CLI must exit code 1 when candidate matches baseline 100%."""
        df = pd.DataFrame(
            {
                "sample_id": [1, 2],
                "pred_label": ["pos", "neg"],
                "pred_secondary": ["no", "no"],
                "pred_category": ["none", "none"],
            }
        )
        cand_file = self.dir_path / "cand.csv"
        base_file = self.dir_path / "base.csv"
        df.to_csv(cand_file, index=False)
        df.to_csv(base_file, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--id-column", "sample_id",
            "--candidate",
            str(cand_file),
            "--baseline",
            str(base_file),
            "--output-zip",
            str(self.dir_path / "out.zip"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("DUPLICATE FILE DETECTED", res.stderr)
        self.assertFalse((self.dir_path / "out.zip").exists())

    def test_audit_gate_cli_valid_pass(self):
        """CLI must exit code 0, create zip, and write receipt when differing predictions exist."""
        cand_df = pd.DataFrame(
            {
                "sample_id": [1, 2],
                "pred_label": ["pos", "pos"],  # row 2 changed
                "pred_secondary": ["no", "no"],
                "pred_category": ["none", "none"],
            }
        )
        base_df = pd.DataFrame(
            {
                "sample_id": [1, 2],
                "pred_label": ["pos", "neg"],
                "pred_secondary": ["no", "no"],
                "pred_category": ["none", "none"],
            }
        )
        cand_file = self.dir_path / "cand.csv"
        base_file = self.dir_path / "base.csv"
        out_zip = self.dir_path / "out.zip"
        receipt = self.dir_path / "receipt.json"

        cand_df.to_csv(cand_file, index=False)
        base_df.to_csv(base_file, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--id-column", "sample_id",
            "--candidate",
            str(cand_file),
            "--baseline",
            str(base_file),
            "--output-zip",
            str(out_zip),
            "--receipt",
            str(receipt),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(out_zip.exists())
        self.assertTrue(receipt.exists())

        with open(receipt, "r") as f:
            r_data = json.load(f)
        self.assertEqual(r_data["status"], "PASSED")
        self.assertEqual(r_data["verification"]["differing_rows"], 1)


if __name__ == "__main__":
    unittest.main()
