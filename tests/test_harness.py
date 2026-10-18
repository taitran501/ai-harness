"""Unit tests for the ML Experiment Harness (TreeJournal, Validator, Audit Gate)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import Node, TreeJournal
from harness.validator import ExperimentValidator


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

    def test_multi_parent_dag_lineage(self):
        """Verify that multi-parent nodes (e.g. ensembles/merges) correctly track DAG edges and children."""
        n1 = Node(verified=True, eligible=True, id="m1", branch="baseline", metrics={"overall_composite": 0.80})
        n2 = Node(verified=True, eligible=True, id="m2", branch="model", metrics={"overall_composite": 0.82})
        ens = Node(
            verified=True,
            eligible=True,
            id="ens_blend",
            parent_ids=["m1", "m2"],
            branch="ens",
            metrics={"overall_composite": 0.85},
        )
        self.journal.append(n1)
        self.journal.append(n2)
        self.journal.append(ens)

        # Children query for m1 and m2 must both include ens_blend
        ch_m1 = self.journal.get_children("m1")
        ch_m2 = self.journal.get_children("m2")
        self.assertTrue(any(c.id == "ens_blend" for c in ch_m1))
        self.assertTrue(any(c.id == "ens_blend" for c in ch_m2))

        # Graph data should contain edges from both m1 and m2
        g = self.journal.get_graph_data()
        edges = g["edges"]
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        self.assertIn(("m1", "ens_blend"), edge_pairs)
        self.assertIn(("m2", "ens_blend"), edge_pairs)


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

    def test_assert_no_group_leakage_detected(self):
        train_df = pd.DataFrame({"group_id": ["A", "B", "C"]})
        val_df = pd.DataFrame({"group_id": ["C", "D"]})

        with self.assertRaises(AssertionError):
            ExperimentValidator.assert_no_group_data_leakage(train_df, val_df, group_column="group_id")

    def test_assert_no_group_leakage_clean(self):
        train_df = pd.DataFrame({"group_id": ["A", "B"]})
        val_df = pd.DataFrame({"group_id": ["C", "D"]})
        ExperimentValidator.assert_no_group_data_leakage(train_df, val_df, group_column="group_id")

    def test_assert_task_conditional_integrity(self):
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "name": "conditional_example",
                "task_type": "multi_task_classification",
                "primary_metric": "macro_f1",
                "subtasks": [
                    {"id": "primary", "column": "primary", "task_type": "binary_classification"},
                    {
                        "id": "secondary",
                        "column": "secondary",
                        "task_type": "multiclass_classification",
                        "condition": {
                            "parent_target": "primary",
                            "operator": "==",
                            "values": "active",
                            "fallback_value": "none",
                        },
                    },
                ],
            }
        })
        clean_df = pd.DataFrame(
            {
                "primary": ["inactive", "active"],
                "secondary": ["none", "choice_a"],
            }
        )
        ExperimentValidator.assert_task_conditional_integrity(clean_df, spec=spec)

        violating_df = pd.DataFrame(
            {
                "primary": ["inactive", "active"],
                "secondary": ["choice_b", "choice_a"],
            }
        )
        with self.assertRaises(AssertionError):
            ExperimentValidator.assert_task_conditional_integrity(violating_df, spec=spec)


class AuditGateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

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
