"""Unit tests for the Anti-Degeneration System (Ensemble Guard, Diagnostics, Statistical Audit Gate)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.diagnostics import DiagnosticEngine
from harness.ensemble_guard import EnsembleGuard
from harness.tree_journal import Node, TreeJournal
from tools.audit_gate import compute_mcnemar_significance


class EnsembleGuardTests(unittest.TestCase):
    def test_pairwise_diversity_identical(self):
        """Identical predictions should have kappa=1.0 and 0% disagreement."""
        preds = pd.Series(["yes", "no", "yes", "no", "yes"] * 20)
        div = EnsembleGuard.compute_pairwise_diversity(preds, preds)
        self.assertEqual(div["cohen_kappa"], 1.0)
        self.assertEqual(div["disagreement_count"], 0)
        self.assertEqual(div["disagreement_pct"], 0.0)

    def test_audit_constituent_diversity_rejects_pseudo_ensemble(self):
        """Audit should reject pseudo-ensembles where kappa > 0.85 or disagreement < 5%."""
        base = ["yes"] * 96 + ["no"] * 4
        # Slightly modified (only 1 label flipped out of 100 -> 99% agreement)
        tweak = ["yes"] * 95 + ["no"] * 5

        df1 = pd.DataFrame({"sample_id": range(100), "pred_label": base})
        df2 = pd.DataFrame({"sample_id": range(100), "pred_label": tweak})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_label", max_kappa=0.85, min_disagreement_pct=5.0
        )
        self.assertFalse(passed)
        self.assertTrue(any("Excessive correlation" in e or "Insufficient disagreement" in e for e in errs))

    def test_audit_constituent_diversity_passes_diverse_models(self):
        """Audit should pass diverse models with reasonable disagreement and kappa <= 0.85."""
        m1 = (["pos"] * 40 + ["neg"] * 60)
        m2 = (["pos"] * 55 + ["neg"] * 45)  # Disagreement ~ 15%

        df1 = pd.DataFrame({"sample_id": range(100), "pred_label": m1})
        df2 = pd.DataFrame({"sample_id": range(100), "pred_label": m2})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_label", max_kappa=0.85, min_disagreement_pct=5.0
        )
        self.assertTrue(passed)
        self.assertEqual(len(errs), 0)

    def test_verify_ensemble_superiority(self):
        """Assert ensemble must beat best constituent by at least min_margin."""
        # Case 1: Inferior or marginal
        passed, msg, stats = EnsembleGuard.verify_ensemble_superiority(
            ensemble_score=0.702, constituent_scores=[0.700, 0.690], min_margin=0.005
        )
        self.assertFalse(passed)
        self.assertIn("REJECTED", msg)

        # Case 2: Meaningful improvement (+0.010 >= +0.005)
        passed, msg, stats = EnsembleGuard.verify_ensemble_superiority(
            ensemble_score=0.710, constituent_scores=[0.700, 0.690], min_margin=0.005
        )
        self.assertTrue(passed)
        self.assertGreaterEqual(stats["margin"], 0.005)

    def test_analyze_rescue_samples(self):
        """Verify rescue sample counting."""
        y_true = np.array([1, 1, 0, 0, 1])
        base_a = np.array([0, 1, 0, 1, 0])  # A wrong on 0, 3, 4
        base_b = np.array([1, 0, 0, 0, 0])  # B wrong on 1, 4
        ens    = np.array([1, 1, 0, 0, 0])  # Ens right on 0, 1, 2, 3; wrong on 4

        rescue = EnsembleGuard.analyze_rescue_samples(y_true, ens, base_a, base_b)
        self.assertEqual(rescue["rescued_from_model_0"], 2)  # Samples 0 and 3
        self.assertEqual(rescue["rescued_from_model_1"], 1)  # Sample 1
        self.assertEqual(rescue["degraded_samples"], 0)
        self.assertEqual(rescue["net_gain"], 3)


class StatisticalAuditGateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mcnemar_significance(self):
        y_true = pd.Series(["yes"] * 50 + ["no"] * 50)
        # Candidate gets 10 samples right where baseline got them wrong
        cand_pred = pd.Series(["yes"] * 50 + ["no"] * 50)
        base_pred = pd.Series(["no"] * 10 + ["yes"] * 40 + ["no"] * 50)

        res = compute_mcnemar_significance(y_true, cand_pred, base_pred)
        self.assertEqual(res["cand_better_count"], 10)
        self.assertEqual(res["base_better_count"], 0)
        self.assertLess(res["p_value"], 0.05)
        self.assertTrue(res["is_significant"])

    def test_cli_rejects_trivial_effect_size(self):
        """CLI should reject a candidate differing by less than 2.0% (MMES violation)."""
        n = 200
        base_df = pd.DataFrame({
            "sample_id": [f"id_{i}" for i in range(n)],
            "pred_label": ["neg"] * n,
            "pred_secondary": ["no"] * n,
            "pred_category": ["none"] * n,
        })
        # Modify only 1 row out of 200 (0.5% < 2.0%)
        cand_df = base_df.copy()
        cand_df.loc[0, "pred_label"] = "pos"

        base_csv = self.p / "baseline.csv"
        cand_csv = self.p / "candidate.csv"
        base_df.to_csv(base_csv, index=False)
        cand_df.to_csv(cand_csv, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--id-column", "sample_id",
            "--candidate", str(cand_csv),
            "--baseline", str(base_csv),
            "--output-zip", str(self.p / "out.zip"),
            "--receipt", str(self.p / "receipt.json"),
            "--min-diff-pct", "2.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("EFFECT SIZE TOO SMALL", proc.stderr)

    def test_cli_rejects_pseudo_ensemble_components(self):
        """CLI should reject an ensemble when constituent models are nearly identical."""
        n = 100
        comp1_df = pd.DataFrame({
            "sample_id": [f"id_{i}" for i in range(n)],
            "pred_label": ["pos"] * 95 + ["neg"] * 5,
            "pred_secondary": ["no"] * n,
            "pred_category": ["none"] * n,
        })
        comp2_df = comp1_df.copy()
        comp2_df.loc[0, "pred_label"] = "neg"  # 99% identical

        cand_df = comp1_df.copy()

        c1_csv = self.p / "comp1.csv"
        c2_csv = self.p / "comp2.csv"
        cand_csv = self.p / "cand.csv"
        comp1_df.to_csv(c1_csv, index=False)
        comp2_df.to_csv(c2_csv, index=False)
        cand_df.to_csv(cand_csv, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--id-column", "sample_id",
            "--candidate", str(cand_csv),
            "--allow-initial",
            "--ensemble-components", str(c1_csv), str(c2_csv),
            "--output-zip", str(self.p / "out.zip"),
            "--receipt", str(self.p / "receipt.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("PSEUDO-ENSEMBLE DETECTED", proc.stderr)


class DiagnosticEngineTests(unittest.TestCase):
    def test_confusion_and_bottleneck_detection(self):
        """Verify DiagnosticEngine identifies categorical and low-support label failures."""
        eval_df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(10)],
            "sentiment": ["positive"] * 4 + ["negative"] * 6,
            "label_rare": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "label_common": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })
        oof_df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(10)],
            "pred_sentiment": ["positive"] * 10,
            "pred_label_rare": [0] * 10,
            "pred_label_common": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })

        diag = DiagnosticEngine.generate_diagnostics(eval_df, oof_df)
        self.assertIn("sentiment", diag["target_confusions"])
        self.assertTrue(diag["minority_labels"]["rare"]["is_zero_recall"])
        self.assertGreaterEqual(len(diag["critical_bottlenecks"]), 1)


class TreeJournalAntiDegenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.temp_dir.name) / "journal.json"
        self.journal = TreeJournal(storage_path=str(self.journal_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_locks_ens_when_only_one_model_family_exists(self):
        """Recommendation must disallow 'ens' if only 1 model family exists in the tree."""
        node1 = Node(
            verified=True,eligible=True,
            id="baseline_tfidf_1",
            parent_id=None,
            branch="baseline",
            atomic_hypothesis="TF-IDF baseline",
            metrics={"overall_composite": 0.65, "model": "tfidf"},
        )
        node2 = Node(
            verified=True,eligible=True,
            id="baseline_tfidf_2",
            parent_id="baseline_tfidf_1",
            branch="feat",
            atomic_hypothesis="TF-IDF with title",
            metrics={"overall_composite": 0.64, "model": "tfidf"},
        )
        self.journal.append(node1)
        self.journal.append(node2)

        self.assertEqual(self.journal.count_distinct_model_families(), 1)
        rec = self.journal.recommend_next_branch()
        self.assertNotEqual(rec["recommended_branch"], "ens")

    def test_diagnostic_cues_influence_recommendation(self):
        """Diagnostic bottleneck on rare identity should prioritize 'loss' branch."""
        node = Node(
            verified=True,eligible=True,
            id="node_with_rare_class_issue",
            parent_id=None,
            branch="baseline",
            atomic_hypothesis="Initial run",
            metrics={"overall_composite": 0.60, "model": "tfidf"},
            diagnostics_summary={"critical_bottlenecks": ["Zero recall on rare identity classes: ['a', 'i']."]},
        )
        self.journal.append(node)

        rec = self.journal.recommend_next_branch()
        self.assertEqual(rec["recommended_branch"], "loss")
        self.assertIn("Focal Loss", rec["diagnostic_cue"])


if __name__ == "__main__":
    unittest.main()
