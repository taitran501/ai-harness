"""Unit tests for hardened correctness invariants.

Covers:
1. Stacking regression preserves continuous fractional targets (no integer truncation or logit conversion).
2. Stacking ID alignment rejects duplicate, null, or mismatched IDs.
3. RankAverager rejects non-positive or zero-sum weights.
4. Pipeline stage resolver rejects path traversal and resolves duplicate names cleanly.
5. Competition spec parsing fails closed on malformed YAML (raises StateIntegrityError).
6. Candidate preflight enforces exact ID coverage and rejects duplicate IDs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.candidate_preflight import run_candidate_preflight
from harness.competition_state import CompetitionStateMachine
from harness.run_pipeline import RunPipelineMixin
from harness.stacking import L2LogitStacker, RankAverager, align_prediction_frames
from harness.state_store import StateIntegrityError


class TestStackingInvariants(unittest.TestCase):
    def test_regression_fractional_labels(self):
        """Continuous regression targets must not be truncated to integers or passed to logit."""
        rng = np.random.RandomState(42)
        n = 60
        y = np.linspace(0.1, 0.9, n)
        X = np.column_stack([y + rng.normal(0, 0.01, size=n), y + rng.normal(0, 0.02, size=n)])

        stacker = L2LogitStacker(C=1.0, task_type="regression", n_splits=5, random_state=42)
        oof_preds, _ = stacker.fit_predict(X, y)

        self.assertEqual(len(oof_preds), n)
        # Predictions must be continuous floats, not collapsed to [0.0]
        self.assertFalse(np.allclose(oof_preds, 0.0))
        # Correlation with true continuous target should be high
        corr = float(np.corrcoef(oof_preds, y)[0, 1])
        self.assertGreater(corr, 0.9)

    def test_align_prediction_frames_duplicate_ids_rejected(self):
        """align_prediction_frames must reject prediction frames with duplicate IDs."""
        ref_df = pd.DataFrame({"id": ["a", "b"], "target": [1.0, 2.0]})
        dup_df = pd.DataFrame({"id": ["a", "a", "b"], "pred": [0.1, 0.2, 0.3]})

        with self.assertRaises(ValueError) as ctx:
            align_prediction_frames([dup_df], ref_df, id_column="id")
        self.assertIn("duplicate or null IDs", str(ctx.exception))

    def test_align_prediction_frames_mismatched_ids_rejected(self):
        """align_prediction_frames must reject prediction frames missing reference IDs."""
        ref_df = pd.DataFrame({"id": ["a", "b", "c"], "target": [1.0, 2.0, 3.0]})
        partial_df = pd.DataFrame({"id": ["a", "b"], "pred": [0.1, 0.2]})

        with self.assertRaises(ValueError) as ctx:
            align_prediction_frames([partial_df], ref_df, id_column="id")
        self.assertIn("ID mismatch", str(ctx.exception))

    def test_rank_averager_zero_weights_rejected(self):
        """RankAverager.blend must reject zero-sum or all-zero weights."""
        arr1 = np.array([0.1, 0.5, 0.9])
        arr2 = np.array([0.2, 0.4, 0.8])

        with self.assertRaises(ValueError) as ctx:
            RankAverager.blend([arr1, arr2], weights=[0.0, 0.0])
        self.assertIn("sum to a positive finite value", str(ctx.exception))


class TestPipelineInvariants(unittest.TestCase):
    def test_pipeline_stage_traversal_rejected(self):
        """Stage names containing path separators or '..' must be rejected."""
        mixin = RunPipelineMixin()
        with self.assertRaises(ValueError) as ctx:
            mixin._resolve_pipeline_stages({"pipeline": [{"name": "../../outside", "entrypoint": "a.py"}]})
        self.assertIn("path separators or '..'", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            mixin._resolve_pipeline_stages({"pipeline": [{"name": "sub/stage", "entrypoint": "a.py"}]})
        self.assertIn("path separators or '..'", str(ctx.exception))

    def test_pipeline_stage_collision_resolution(self):
        """Multiple duplicate stage names must be resolved to unique names without collisions."""
        mixin = RunPipelineMixin()
        stages = mixin._resolve_pipeline_stages({
            "pipeline": [
                {"name": "train", "entrypoint": "a.py"},
                {"name": "train_1", "entrypoint": "b.py"},
                {"name": "train", "entrypoint": "c.py"},
            ]
        })
        names = [s["name"] for s in stages]
        self.assertEqual(len(set(names)), 3)
        self.assertEqual(names, ["train", "train_1", "train_2"])

    def test_pipeline_lexical_ids_preserved(self):
        """read_prediction_frame must preserve leading zeros in string IDs like '001'."""
        from harness.evaluation import read_prediction_frame
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            csv_path.write_text("id,pred\n001,0.25\n002,0.75\n", encoding="utf-8")
            df = read_prediction_frame(csv_path, expected_ids=["001", "002"], id_column="id")
            self.assertEqual(list(df["id"]), ["001", "002"])
            self.assertEqual(type(df["id"].iloc[0]), str)


class TestCompetitionStateInvariants(unittest.TestCase):
    def test_corrupt_yaml_fails_closed(self):
        """A corrupt or unparseable task config must raise StateIntegrityError, not fallback silently."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "configs"
            config_dir.mkdir()
            task_yaml = config_dir / "task.yaml"
            # Write invalid YAML syntax
            task_yaml.write_text("task:\n  [unbalanced syntax: %%%", encoding="utf-8")

            with self.assertRaises(StateIntegrityError) as ctx:
                CompetitionStateMachine(tmp_path)
            self.assertIn("SPEC_PARSE_FAILED", str(ctx.exception))


class TestCandidatePreflightInvariants(unittest.TestCase):
    def test_mismatched_id_coverage_rejected(self):
        """Candidate and incumbent predictions must cover the exact same set of IDs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cand_csv = tmp_path / "cand.csv"
            inc_csv = tmp_path / "inc.csv"

            # Candidate has 2 IDs, incumbent has 3
            pd.DataFrame({"id": ["a", "b"], "pred": [0.1, 0.2]}).to_csv(cand_csv, index=False)
            pd.DataFrame({"id": ["a", "b", "c"], "pred": [0.1, 0.2, 0.3]}).to_csv(inc_csv, index=False)

            with self.assertRaises(ValueError) as ctx:
                run_candidate_preflight(tmp_path, cand_csv, inc_csv)
            self.assertIn("ID sets do not match", str(ctx.exception))

    def test_duplicate_ids_rejected(self):
        """Candidate predictions with duplicate IDs must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cand_csv = tmp_path / "cand.csv"
            inc_csv = tmp_path / "inc.csv"

            pd.DataFrame({"id": ["a", "a"], "pred": [0.1, 0.2]}).to_csv(cand_csv, index=False)
            pd.DataFrame({"id": ["a", "b"], "pred": [0.1, 0.2]}).to_csv(inc_csv, index=False)

            with self.assertRaises(ValueError) as ctx:
                run_candidate_preflight(tmp_path, cand_csv, inc_csv)
            self.assertIn("Duplicate IDs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
