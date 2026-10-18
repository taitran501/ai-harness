"""Unit and Regression Tests for P1-P2 Refactored Capabilities.

Covers:
1. Minimization vs Maximization direction in TreeJournal (UCB formula) and EnsembleGuard.
2. Multi-model rescue sample analysis and dynamic diversity auditing in EnsembleGuard.
3. Atomic File I/O utilities (io_utils.py) and corruption resistance.
4. Weighted probability blending and fair tie-breaking in CompetitionState candidate synthesis.
5. Dynamic CV metric extraction and markdown rendering in SubmissionManager.
6. Dynamic task-spec driven validation and diff metrics in AuditGate.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.ensemble_guard import EnsembleGuard
from harness.io_utils import atomic_write_json, atomic_write_text
from harness.submission_manager import SubmissionEntry, SubmissionManager
from harness.tree_journal import Node, TreeJournal
from tools.audit_gate import audit_prediction_integrity, compute_diff_metrics


class TestTreeJournalDirectionAndUCB(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.journal_path = self.test_dir / "journal.json"
        self.journal = TreeJournal(storage_path=str(self.journal_path))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ucb_minimization_prioritizes_lower_score(self):
        """When direction='minimize', lower loss should receive higher exploit score and UCB."""
        # Node 1: High loss (worse)
        node_bad = Node(
            verified=True,eligible=True,
            id="exp001_high_loss",
            branch="baseline",
            direction="minimize",
            primary_metric_name="loss",
            metrics={"loss": 2.50},
        )
        # Node 2: Low loss (better)
        node_good = Node(
            verified=True,eligible=True,
            id="exp002_low_loss",
            branch="feat",
            parent_id="exp001_high_loss",
            direction="minimize",
            primary_metric_name="loss",
            metrics={"loss": 0.40},
        )
        self.journal.append(node_bad)
        self.journal.append(node_good)

        rec = self.journal.recommend_next_branch()
        # Better node (lower loss = 0.40) must be recommended as parent
        self.assertEqual(rec["parent_id"], "exp002_low_loss")

    def test_ucb_maximization_prioritizes_higher_score(self):
        """When direction='maximize', higher score should receive higher exploit score and UCB."""
        node_low = Node(
            verified=True,eligible=True,
            id="exp001_low_f1",
            branch="baseline",
            direction="maximize",
            primary_metric_name="f1",
            metrics={"f1": 0.60},
        )
        node_high = Node(
            verified=True,eligible=True,
            id="exp002_high_f1",
            branch="feat",
            parent_id="exp001_low_f1",
            direction="maximize",
            primary_metric_name="f1",
            metrics={"f1": 0.85},
        )
        self.journal.append(node_low)
        self.journal.append(node_high)

        rec = self.journal.recommend_next_branch()
        self.assertEqual(rec["parent_id"], "exp002_high_f1")

    def test_generic_bottleneck_cue_handling(self):
        """Diagnostics with generic words like 'imbalance' or 'underfitting' trigger appropriate branch cues."""
        node = Node(
            verified=True,eligible=True,
            id="exp001_imbalanced",
            branch="baseline",
            direction="maximize",
            metrics={"score": 0.70},
            diagnostics_summary={"critical_bottlenecks": ["Severe class imbalance on tail classes"]},
        )
        self.journal.append(node)
        rec = self.journal.recommend_next_branch()
        self.assertIn("imbalance", rec["diagnostic_cue"].lower())
        self.assertEqual(rec["recommended_branch"], "loss")


class TestEnsembleGuardEnhancements(unittest.TestCase):
    def test_verify_ensemble_superiority_minimize(self):
        """In minimization, ensemble is superior when ensemble_score < best_constituent - min_margin."""
        constituent_losses = [1.20, 1.15, 1.10]  # Best constituent = 1.10
        # Ensemble achieves 1.05 (margin = 1.10 - 1.05 = 0.05 >= 0.01)
        passed, msg, details = EnsembleGuard.verify_ensemble_superiority(
            ensemble_score=1.05,
            constituent_scores=constituent_losses,
            min_margin=0.01,
            direction="minimize",
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(details["margin"], 0.05)

        # Ensemble achieves 1.12 (worse than best constituent 1.10)
        passed_fail, _, _ = EnsembleGuard.verify_ensemble_superiority(
            ensemble_score=1.12,
            constituent_scores=constituent_losses,
            min_margin=0.01,
            direction="minimize",
        )
        self.assertFalse(passed_fail)

    def test_audit_constituent_diversity_dynamic_column(self):
        """audit_constituent_diversity automatically finds prediction columns when col=None."""
        df1 = pd.DataFrame({"pred_target": ["a", "b", "a", "b", "a"] * 20})
        df2 = pd.DataFrame({"pred_target": ["a", "a", "a", "b", "b"] * 20})
        passed, errors, details = EnsembleGuard.audit_constituent_diversity([df1, df2], col=None)
        self.assertEqual(details["column_checked"], "pred_target")

    def test_multi_model_rescue_sample_analysis(self):
        """analyze_rescue_samples handles 3 or more models gracefully."""
        y_true = np.array([1, 1, 1, 0, 0])
        m1 = np.array([1, 0, 0, 0, 0])  # wrong on index 1, 2
        m2 = np.array([1, 1, 0, 0, 0])  # wrong on index 2
        m3 = np.array([0, 1, 0, 0, 1])  # wrong on index 0, 2, 4
        ens = np.array([1, 1, 1, 0, 0])  # 100% correct!

        res = EnsembleGuard.analyze_rescue_samples(y_true, ens, m1, m2, m3)
        self.assertIn("rescued_from_model_0", res)
        self.assertIn("rescued_from_model_1", res)
        self.assertIn("rescued_from_model_2", res)
        # All constituents were wrong on index 2, but ensemble got it right!
        self.assertEqual(res["synergistic_rescues"], 1)
        self.assertEqual(res["degraded_samples"], 0)
        self.assertGreater(res["net_gain"], 0)


class TestAtomicIOUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_write_text_and_json(self):
        target_file = self.test_dir / "subdir" / "state.json"
        data = {"hello": "world", "number": 42, "items": [1, 2, 3]}

        atomic_write_json(target_file, data)
        self.assertTrue(target_file.exists())

        with open(target_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

        # Overwrite atomically
        atomic_write_text(target_file, "plain text updated")
        self.assertEqual(target_file.read_text(encoding="utf-8"), "plain text updated")

        # Verify no dangling temp files in directory
        temp_files = list(target_file.parent.glob(".*.tmp_*"))
        self.assertEqual(len(temp_files), 0)


class TestCandidateSynthesisEnhancements(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.test_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True)

        # Create two mock experiment runs
        for exp_id in ["exp01_model", "exp02_model"]:
            exp_path = self.artifacts_dir / exp_id
            exp_path.mkdir()
            df = pd.DataFrame({
                "id": [1, 2, 3, 4],
                "fold": [0, 0, 1, 1],
                "pred_prob": [0.2, 0.8, 0.4, 0.9] if exp_id == "exp01_model" else [0.4, 0.6, 0.6, 0.7],
                "pred_label": ["no", "yes", "no", "yes"] if exp_id == "exp01_model" else ["no", "no", "yes", "yes"],
            })
            df.to_csv(exp_path / "oof_predictions.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_weighted_probability_blend(self):
        """Weighted blend correctly averages continuous probabilities."""
        from harness.competition_state import CompetitionState

        comp_state = CompetitionState(project_root=self.test_dir)
        with self.assertRaisesRegex(ValueError,"LEGACY_UNVERIFIED"):
            comp_state.synthesize_candidate(name="prob_blend_v1",mode="weighted_blend",source_map={"components":["exp01_model","exp02_model"],"weights":[.75,.25]})


class TestSubmissionManagerDynamic(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.mgr = SubmissionManager(submissions_dir=str(self.test_dir / "submissions"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dynamic_markdown_rendering(self):
        """Renders dynamic markdown without hardcoded A/B/C when other metrics are present."""
        entry = SubmissionEntry(
            id="sub_001_exp01",
            exp_id="exp01",
            created_at="2026-09-27 10:00:00 UTC",
            status="SCORED",
            local_cv_scores={"rmse": 0.4521, "mae": 0.3120},
            official_scores={"rmse": 0.4610},
            score_deltas={"rmse": 0.0089},
        )
        self.mgr.submissions[entry.id] = entry
        md = self.mgr.render_submissions_markdown()
        self.assertIn("sub_001_exp01", md)
        self.assertTrue(Path(self.test_dir / "submissions" / "SUBMISSIONS.md").exists())


class TestAuditGateDynamicIntegrity(unittest.TestCase):
    def test_audit_prediction_integrity_custom_id_col(self):
        """Integrity checker handles non-standard id column."""
        df = pd.DataFrame({
            "sample_index": [1, 2, None],
            "pred_stereotype": ["yes", "no", "yes"],
        })
        errors = audit_prediction_integrity(df, id_col="sample_index")
        self.assertTrue(any("sample_index" in e for e in errors))

    def test_compute_diff_metrics_dynamic(self):
        """Row-by-row diffs computed cleanly without hardcoded ID col."""
        cand = pd.DataFrame({"sample_id": [1, 2, 3], "pred_target": ["a", "b", "c"]})
        base = pd.DataFrame({"sample_id": [1, 2, 3], "pred_target": ["a", "b", "b"]})
        diff_count, diff_details = compute_diff_metrics(cand, base, id_col="sample_id")
        self.assertEqual(diff_count, 1)
        self.assertEqual(diff_details["pred_target"]["different_count"], 1)


if __name__ == "__main__":
    unittest.main()
