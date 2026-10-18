"""Dedicated Unit Test Suite for SWE-agent Deterministic & Statistical Audit Gate.

Tests:
1. Discrete exact != row comparison with backward compatibility.
2. Continuous relative epsilon tolerance (machine epsilon noise ignored, micro-jitter rejected, genuine delta detected).
3. Constant / zero-variance baseline fallback without ZeroDivisionError.
4. Continuous paired t-test statistical significance.
5. Prediction integrity for regression, ranking, and generative LLM tasks without classes.
6. End-to-end CLI regression candidate receipt generation and micro-jitter rejection.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.task_spec import SubtaskSpec, UniversalTaskSpec
from tools.audit_gate import (
    audit_prediction_integrity,
    compute_continuous_significance,
    compute_diff_metrics,
)


def make_task_spec(
    task_type: str,
    classes: list = None,
    primary_metric: str = "rmse",
    direction: str = "minimize",
) -> UniversalTaskSpec:
    """Helper to synthesize task specification fixtures."""
    classes = classes if classes is not None else []
    modality_map = {
        "regression": "tabular",
        "ranking": "ranking",
        "generative": "nlp",
        "generative_nlp": "nlp",
        "text_generation": "nlp",
        "multiclass_classification": "tabular",
    }
    return UniversalTaskSpec.from_dict({
        "task": {
            "name": f"fixture_{task_type}",
            "modality": modality_map.get(task_type, "tabular"),
            "primary_metric": primary_metric,
            "direction": direction,
            "data": {"id_column": "id"},
            "inputs": [{"name": "feat", "type": "tabular", "source_columns": ["x"]}],
            "subtasks": [{
                "id": "target",
                "column": "target",
                "task_type": task_type,
                "primary_metric": primary_metric,
                "direction": direction,
                "classes": classes,
            }],
        }
    })


class TestAuditGateDiffMetrics(unittest.TestCase):
    """Verifies differential metrics between candidate and baseline predictions."""

    def test_compute_diff_metrics_discrete_exact(self):
        """Discrete columns use exact != comparison."""
        base = pd.DataFrame({
            "id": range(10),
            "pred_label": ["cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird", "cat"],
        })
        cand = base.copy()
        # Flip only row 0: 'cat' -> 'dog'
        cand.loc[0, "pred_label"] = "dog"

        diff_count, details = compute_diff_metrics(cand, base, id_col="id")
        self.assertEqual(diff_count, 1)
        self.assertEqual(details["pred_label"]["different_count"], 1)
        self.assertAlmostEqual(details["pred_label"]["pct_changed"], 10.0)
        self.assertFalse(details["pred_label"]["is_continuous"])
        self.assertIsNone(details["pred_label"]["rel_epsilon"])

    def test_compute_diff_metrics_continuous_machine_epsilon_ignored(self):
        """Machine epsilon (1e-15) noise is ignored and evaluates to 0 changed rows."""
        rng = np.random.default_rng(42)
        values = rng.normal(loc=100.0, scale=15.0, size=100)
        base = pd.DataFrame({"id": range(100), "pred_val": values})
        # Add float representation noise (1e-15) to every single row
        cand = pd.DataFrame({"id": range(100), "pred_val": values + 1e-15})

        diff_count, details = compute_diff_metrics(cand, base, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 0)
        self.assertEqual(details["pred_val"]["different_count"], 0)
        self.assertEqual(details["pred_val"]["pct_changed"], 0.0)
        self.assertTrue(details["pred_val"]["is_continuous"])
        self.assertEqual(details["pred_val"]["rel_epsilon"], 0.01)

    def test_compute_diff_metrics_continuous_rejects_micro_jitter(self):
        """Micro-jitter (1e-8) does NOT trigger false MMES pass."""
        rng = np.random.default_rng(42)
        values = rng.normal(loc=50.0, scale=10.0, size=200)
        base = pd.DataFrame({"id": range(200), "pred_val": values})
        # Add micro-jitter between -1e-8 and +1e-8
        jitter = rng.uniform(-1e-8, 1e-8, size=200)
        cand = pd.DataFrame({"id": range(200), "pred_val": values + jitter})

        diff_count, details = compute_diff_metrics(cand, base, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 0)
        self.assertEqual(details["pred_val"]["different_count"], 0)

    def test_compute_diff_metrics_continuous_detects_real_delta(self):
        """Genuine 5% standard deviation shift is correctly detected."""
        rng = np.random.default_rng(42)
        sigma = 10.0
        values = rng.normal(loc=50.0, scale=sigma, size=100)
        base = pd.DataFrame({"id": range(100), "pred_val": values})
        cand = base.copy()

        # Shift 15 rows (15%) by 5% of sigma (0.5 units)
        shift = 0.05 * sigma  # 0.5
        cand.loc[:14, "pred_val"] = values[:15] + shift

        diff_count, details = compute_diff_metrics(cand, base, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 15)
        self.assertAlmostEqual(details["pred_val"]["pct_changed"], 15.0)

    def test_compute_diff_metrics_zero_variance_fallback(self):
        """Zero-variance baseline does not divide by zero and falls back to mean scale."""
        base = pd.DataFrame({"id": range(20), "pred_val": [50.0] * 20})
        cand = base.copy()
        # Shift 4 rows by 2.0 (4% shift relative to mean 50.0 >= rel_epsilon 0.01)
        cand.loc[:3, "pred_val"] = 52.0

        diff_count, details = compute_diff_metrics(cand, base, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 4)
        self.assertAlmostEqual(details["pred_val"]["pct_changed"], 20.0)

    def test_compute_diff_metrics_constant_zero_baseline(self):
        """Constant zero baseline falls back to 1.0 denominator."""
        n = 50
        base = pd.DataFrame({"id": list(range(n)), "pred_val": np.zeros(n)})
        cand = pd.DataFrame({"id": list(range(n)), "pred_val": np.ones(n) * 0.02})
        # std is 0.0, mean_abs is 0.0 -> S falls back to 1.0. |0.02 - 0| / 1.0 = 0.02 >= 0.01
        diff_count, _ = compute_diff_metrics(cand, base, rel_epsilon=0.01)
        self.assertEqual(diff_count, 50)

    def test_compute_diff_metrics_id_and_schema_validation(self):
        """Invalid IDs or schemas raise ValueError."""
        a = pd.DataFrame({"id": [1, 2], "pred_a": [1.0, 2.0]})
        b = pd.DataFrame({"id": [1, 3], "pred_a": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            compute_diff_metrics(a, b, id_col="id")

        c = pd.DataFrame({"id": [1, 2], "pred_b": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            compute_diff_metrics(a, c, id_col="id")

    def test_compute_diff_metrics_small_scale_low_variance_detects_relative_shift(self):
        """Baseline with std <= 1e-8 and mean_abs ~ 1e-6 does NOT inflate S to 1.0, detecting 20% shift."""
        n = 100
        rng = np.random.default_rng(42)
        base_vals = rng.normal(1e-6, 5e-9, size=n)
        cand_vals = base_vals + 2e-7  # 20% shift relative to mean 1e-6 >= rel_epsilon 0.01

        base_df = pd.DataFrame({"id": range(n), "pred_target": base_vals})
        cand_df = pd.DataFrame({"id": range(n), "pred_target": cand_vals})

        diff_count, details = compute_diff_metrics(cand_df, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, n)
        self.assertEqual(details["pred_target"]["different_count"], n)
        self.assertAlmostEqual(details["pred_target"]["pct_changed"], 100.0)
        self.assertTrue(details["pred_target"]["is_continuous"])

    def test_compute_diff_metrics_small_scale_low_variance_sub_epsilon_noise_rejected(self):
        """Baseline with std <= 1e-8 and mean_abs ~ 1e-6 rejects 0.1% sub-epsilon shift."""
        n = 100
        rng = np.random.default_rng(42)
        base_vals = rng.normal(1e-6, 5e-9, size=n)
        cand_vals = base_vals + 1e-9  # 0.1% shift < rel_epsilon 0.01

        base_df = pd.DataFrame({"id": range(n), "pred_target": base_vals})
        cand_df = pd.DataFrame({"id": range(n), "pred_target": cand_vals})

        diff_count, details = compute_diff_metrics(cand_df, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 0)
        self.assertEqual(details["pred_target"]["different_count"], 0)
        self.assertAlmostEqual(details["pred_target"]["pct_changed"], 0.0)


class TestAuditGateContinuousSignificance(unittest.TestCase):
    """Verifies continuous paired error significance test."""

    def test_continuous_significance_paired_ttest_basic(self):
        """Paired t-test calculates sample counts and p-value on squared loss differentials."""
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0] * 10)
        # Baseline is consistently off by 5.0
        base_pred = y_true + 5.0
        # Candidate is consistently off by 1.0 (significantly better)
        cand_pred = y_true + 1.0

        res = compute_continuous_significance(y_true, cand_pred, base_pred)
        self.assertEqual(res["test_name"], "paired_ttest")
        self.assertEqual(res["cand_better_count"], 50)
        self.assertEqual(res["base_better_count"], 0)
        self.assertLess(res["p_value"], 0.01)
        self.assertTrue(res["is_significant"])

    def test_continuous_significance_identical_predictions(self):
        """Identical predictions return p-value=1.0 and is_significant=False."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])

        res = compute_continuous_significance(y_true, pred, pred)
        self.assertEqual(res["statistic"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertFalse(res["is_significant"])
        self.assertEqual(res["cand_better_count"], 0)
        self.assertEqual(res["base_better_count"], 0)

    def test_continuous_significance_noisy_unimproved_difference(self):
        """Equal noise differences evaluate to p-value > 0.05 (not significant)."""
        rng = np.random.default_rng(123)
        y_true = pd.Series(rng.normal(0, 1, 100))
        # Both models have similar random noise
        cand_pred = y_true + rng.normal(0, 1, 100)
        base_pred = y_true + rng.normal(0, 1, 100)

        res = compute_continuous_significance(y_true, cand_pred, base_pred)
        self.assertIn("p_value", res)
        self.assertIn("cand_better_count", res)
        self.assertIn("base_better_count", res)

    def test_continuous_significance_small_scale_superior_candidate_significant(self):
        """Small-scale targets (y in [1e-6, 1e-5]) achieve p < 0.05 when candidate is strictly superior."""
        n = 500
        rng = np.random.default_rng(42)
        y_true = rng.uniform(1e-6, 1e-5, size=n)
        # Candidate error is 1e-8 (near perfect)
        cand_pred = y_true + 1e-8
        # Baseline error is 5e-6 (500x larger)
        base_pred = y_true + 5e-6

        res = compute_continuous_significance(y_true, cand_pred, base_pred)
        self.assertEqual(res["test_name"], "paired_ttest")
        self.assertEqual(res["cand_better_count"], n)
        self.assertEqual(res["base_better_count"], 0)
        self.assertTrue(res["is_significant"], f"Expected significant improvement, got p={res['p_value']}")
        self.assertLess(res["p_value"], 0.05)
        self.assertLess(res["statistic"], 0.0)

    def test_continuous_significance_small_scale_identical_not_significant(self):
        """Small-scale identical predictions return p = 1.0 and is_significant = False."""
        n = 100
        rng = np.random.default_rng(42)
        y_true = rng.uniform(1e-6, 1e-5, size=n)
        pred = y_true + 1e-7

        res = compute_continuous_significance(y_true, pred, pred)
        self.assertEqual(res["statistic"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertFalse(res["is_significant"])
        self.assertEqual(res["cand_better_count"], 0)
        self.assertEqual(res["base_better_count"], 0)

    def test_continuous_significance_small_scale_machine_epsilon_not_significant(self):
        """Small-scale predictions with only machine epsilon noise return is_significant = False."""
        n = 100
        rng = np.random.default_rng(42)
        y_true = rng.uniform(1e-6, 1e-5, size=n)
        base_pred = y_true + 1e-7
        # Machine epsilon relative noise
        cand_pred = base_pred + (np.finfo(float).eps * base_pred)

        res = compute_continuous_significance(y_true, cand_pred, base_pred)
        self.assertFalse(res["is_significant"])
        self.assertEqual(res["p_value"], 1.0)


class TestAuditGatePredictionIntegrity(unittest.TestCase):
    """Verifies prediction integrity across regression, ranking, and generative modalities."""

    def test_prediction_integrity_regression_finite_values(self):
        """Regression subtasks enforce finite numeric values."""
        spec = make_task_spec("regression")
        # Valid finite floats pass
        df_valid = pd.DataFrame({"id": [1, 2, 3], "pred_target": [1.5, -2.3, 0.0]})
        self.assertEqual(audit_prediction_integrity(df_valid, spec=spec), [])

        # Non-finite values fail
        df_nan = pd.DataFrame({"id": [1, 2, 3], "pred_target": [1.5, np.nan, 0.0]})
        self.assertTrue(any("Null" in e for e in audit_prediction_integrity(df_nan, spec=spec)))

        df_inf = pd.DataFrame({"id": [1, 2, 3], "pred_target": [1.5, np.inf, 0.0]})
        self.assertTrue(any("Non-finite" in e for e in audit_prediction_integrity(df_inf, spec=spec)))

    def test_prediction_integrity_ranking_task_without_classes(self):
        """Ranking subtasks without classes are NOT rejected for missing class domain."""
        spec = make_task_spec("ranking", classes=[])
        df_ranking = pd.DataFrame({"id": [101, 102, 103], "pred_target": [1, 3, 2]})
        errors = audit_prediction_integrity(df_ranking, spec=spec)
        self.assertEqual(errors, [])

    def test_prediction_integrity_generative_task_without_classes(self):
        """Generative NLP subtasks without classes accept arbitrary text strings."""
        spec = make_task_spec("generative", classes=[])
        df_gen = pd.DataFrame({
            "id": [1, 2],
            "pred_target": [
                "The quick brown fox jumps over the lazy dog.",
                "Reinforcement learning from human feedback.",
            ],
        })
        errors = audit_prediction_integrity(df_gen, spec=spec)
        self.assertEqual(errors, [])

    def test_prediction_integrity_multimodal_spec(self):
        """Multimodal spec with ranking and generative subtasks passes."""
        spec = UniversalTaskSpec(
            name="MultiModalTask",
            task_type="multimodal",
            primary_metric="spearman_corr",
            subtasks={
                "rank_target": SubtaskSpec("rank_target", "Rank", "ranking", column="rank_target", classes=[]),
                "gen_target": SubtaskSpec("gen_target", "Gen", "generative_nlp", column="gen_target", classes=[]),
            },
        )
        valid_df = pd.DataFrame({
            "id": ["1", "2"],
            "pred_rank_target": [0.85, 0.12],
            "pred_gen_target": ["Sample response 1", "Sample response 2"],
        })
        errors = audit_prediction_integrity(valid_df, spec=spec)
        self.assertEqual(errors, [])

    def test_prediction_integrity_multiclass_enforces_classes(self):
        """Multiclass subtasks strictly reject labels outside declared classes."""
        spec = make_task_spec("multiclass_classification", classes=["cat", "dog"])
        df_invalid = pd.DataFrame({"id": [1, 2], "pred_target": ["cat", "elephant"]})
        errors = audit_prediction_integrity(df_invalid, spec=spec)
        self.assertTrue(any("outside declared classes" in e for e in errors))


class TestAuditGateCLIEndToEnd(unittest.TestCase):
    """Verifies end-to-end CLI execution on regression targets."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_regression_candidate_generates_receipt(self):
        """End-to-end CLI runs cleanly on continuous regression and writes signed receipt."""
        n = 100
        rng = np.random.default_rng(42)
        y = rng.normal(100.0, 15.0, n)
        base_preds = y + rng.normal(0, 5.0, n)

        # Candidate improves on 20% of rows (20 rows shifted closer to truth)
        cand_preds = base_preds.copy()
        cand_preds[:20] = y[:20] + rng.normal(0, 0.5, 20)

        base_df = pd.DataFrame({"id": range(n), "pred_target": base_preds})
        cand_df = pd.DataFrame({"id": range(n), "pred_target": cand_preds})
        gt_df = pd.DataFrame({"id": range(n), "target": y})

        base_csv = self.p / "base.csv"
        cand_csv = self.p / "cand.csv"
        gt_csv = self.p / "gt.csv"
        out_zip = self.p / "submission.zip"
        receipt_json = self.p / "receipt.json"

        base_df.to_csv(base_csv, index=False)
        cand_df.to_csv(cand_csv, index=False)
        gt_df.to_csv(gt_csv, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--id-column", "id",
            "--candidate", str(cand_csv),
            "--baseline", str(base_csv),
            "--ground-truth", str(gt_csv),
            "--output-zip", str(out_zip),
            "--receipt", str(receipt_json),
            "--min-diff-pct", "2.0",
            "--rel-epsilon", "0.01",
            "--max-corr", "0.85",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertTrue(out_zip.exists())
        self.assertTrue(receipt_json.exists())

        receipt = json.loads(receipt_json.read_text())
        self.assertEqual(receipt["status"], "PASSED")
        self.assertGreaterEqual(receipt["verification"]["differing_rows"], 2)
        self.assertEqual(receipt["verification"]["rel_epsilon"], 0.01)
        self.assertEqual(receipt["verification"]["max_corr_threshold"], 0.85)
        col_info = receipt["verification"]["diff_by_column"]["pred_target"]
        self.assertTrue(col_info["is_continuous"])
        self.assertIn("mcnemar_test", receipt["verification"])
        self.assertIn("significance_test", receipt["verification"])

    def test_cli_regression_rejects_micro_jitter_candidate(self):
        """CLI rejects candidate with only 1e-8 jitter as MMES violation."""
        n = 100
        rng = np.random.default_rng(42)
        base_preds = rng.normal(100.0, 15.0, n)
        cand_preds = base_preds + rng.uniform(-1e-8, 1e-8, n)

        base_df = pd.DataFrame({"id": range(n), "pred_target": base_preds})
        cand_df = pd.DataFrame({"id": range(n), "pred_target": cand_preds})

        base_csv = self.p / "base.csv"
        cand_csv = self.p / "cand.csv"
        base_df.to_csv(base_csv, index=False)
        cand_df.to_csv(cand_csv, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--id-column", "id",
            "--candidate", str(cand_csv),
            "--baseline", str(base_csv),
            "--output-zip", str(self.p / "out.zip"),
            "--receipt", str(self.p / "receipt.json"),
            "--min-diff-pct", "2.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("EFFECT SIZE TOO SMALL", proc.stderr)

    def test_cli_rejects_monotonic_pseudo_ensemble_via_spearman(self):
        """CLI audit gate rejects non-linear monotonic constituent models (Spearman = 1.0 > 0.85)."""
        n = 100
        x = np.linspace(1.0, 10.0, n)
        y = np.exp(x)  # Monotonic exponential: Spearman = 1.0, Pearson ~ 0.72 < 0.85
        ens = (x + y) / 2.0

        f_cand = self.p / "ens_mono.csv"
        f_c1 = self.p / "c1_mono.csv"
        f_c2 = self.p / "c2_mono.csv"
        pd.DataFrame({"id": range(n), "pred_target": ens}).to_csv(f_cand, index=False)
        pd.DataFrame({"id": range(n), "pred_target": x}).to_csv(f_c1, index=False)
        pd.DataFrame({"id": range(n), "pred_target": y}).to_csv(f_c2, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--allow-initial",
            "--candidate", str(f_cand),
            "--ensemble-components", str(f_c1), str(f_c2),
            "--output-zip", str(self.p / "out.zip"),
            "--receipt", str(self.p / "receipt.json"),
            "--max-corr", "0.85",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("PSEUDO-ENSEMBLE DETECTED", proc.stderr)
        self.assertTrue("Spearman" in proc.stderr or "Spearman Correlation" in proc.stderr)


if __name__ == "__main__":
    unittest.main()
