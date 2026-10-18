"""Dedicated Unit Test Suite for Ensemble Diversity and Superiority Guard.

Tests:
1. Scale invariance across large numbers ($200k house prices) and small numbers (0.005 rates).
2. Constant predictions and zero-variance robustness without ZeroDivisionError.
3. Pearson and Spearman rank correlation on linear and non-linear monotonic data.
4. Numeric string series (object dtype) handling without Cohen's Kappa failure.
5. Continuous rescue sample analysis (synergistic rescues, single model rescues, degraded samples).
6. Diversity audit reporting across continuous regression and discrete classification regimes.
7. Backward compatibility on classification models and superiority verification.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from harness.ensemble_guard import EnsembleGuard


class TestEnsembleGuardScaleInvariance(unittest.TestCase):
    """Verifies that pairwise diversity is invariant to the absolute scale of continuous targets."""

    def test_scale_invariance_large_numbers_house_prices(self):
        """$200k house prices with $0.06 noise do NOT trigger 100% false disagreement."""
        rng = np.random.default_rng(42)
        # Large scale: house prices with mean $250,000 and std $50,000
        preds_a = rng.normal(loc=250000.0, scale=50000.0, size=100)
        # Negligible 6-cent noise
        preds_b = preds_a + 0.06

        div = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_b, rel_epsilon=0.05)
        self.assertEqual(div["metric_type"], "continuous")
        self.assertEqual(div["disagreement_count"], 0)
        self.assertEqual(div["disagreement_pct"], 0.0)

        # Genuine difference: 20 rows shifted by $10,000 (20% of std >= 5% rel_epsilon)
        preds_c = preds_a.copy()
        preds_c[:20] += 10000.0
        div_c = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_c, rel_epsilon=0.05)
        self.assertEqual(div_c["disagreement_count"], 20)
        self.assertAlmostEqual(div_c["disagreement_pct"], 20.0)

    def test_scale_invariance_small_numbers_rates(self):
        """Small target scale (0.005) evaluates relative difference accurately."""
        rng = np.random.default_rng(42)
        sigma = 0.001
        preds_a = rng.normal(loc=0.005, scale=sigma, size=100)
        # Microscopic noise (1e-7) -> 0% disagreement
        preds_b = preds_a + 1e-7
        div_b = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_b, rel_epsilon=0.05)
        self.assertEqual(div_b["disagreement_count"], 0)
        self.assertEqual(div_b["disagreement_pct"], 0.0)

        # Meaningful shift: 15 rows shifted by 0.0003 (30% of sigma >= 5% rel_epsilon)
        preds_c = preds_a.copy()
        preds_c[:15] += 0.0003
        div_c = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_c, rel_epsilon=0.05)
        self.assertEqual(div_c["disagreement_count"], 15)
        self.assertAlmostEqual(div_c["disagreement_pct"], 15.0)

    def test_constant_predictions_no_zero_division(self):
        """Constant predictions do not raise ZeroDivisionError."""
        preds_a = pd.Series([10.0] * 50)
        preds_b = pd.Series([10.0] * 50)
        div = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_b)
        self.assertEqual(div["pearson_corr"], 1.0)
        self.assertEqual(div["cohen_kappa"], 1.0)
        self.assertEqual(div["disagreement_count"], 0)
        self.assertEqual(div["disagreement_pct"], 0.0)

    def test_pairwise_diversity_discrete_diverse(self):
        """Discrete diverse predictions yield valid kappa and disagreement."""
        m1 = pd.Series(["pos"] * 40 + ["neg"] * 60)
        m2 = pd.Series(["pos"] * 55 + ["neg"] * 45)
        div = EnsembleGuard.compute_pairwise_diversity(m1, m2)
        self.assertEqual(div["metric_type"], "discrete")
        self.assertLessEqual(div["cohen_kappa"], 0.85)
        self.assertGreaterEqual(div["disagreement_pct"], 5.0)


class TestEnsembleGuardCorrelation(unittest.TestCase):
    """Verifies Pearson and Spearman rank correlation computation."""

    def test_linear_data_pearson_and_spearman(self):
        """Linear data produces both Pearson and Spearman correlation of 1.0."""
        x = np.linspace(1.0, 100.0, 50)
        y = 2.5 * x + 10.0
        div = EnsembleGuard.compute_pairwise_diversity(x, y)

        self.assertAlmostEqual(div["pearson_corr"], 1.0, places=4)
        self.assertAlmostEqual(div["spearman_corr"], 1.0, places=4)
        # Preserved backward-compatibility alias
        self.assertAlmostEqual(div["cohen_kappa"], 1.0, places=4)

    def test_nonlinear_monotonic_data_spearman_higher_than_pearson(self):
        """Non-linear monotonic relationship yields Spearman 1.0 but lower Pearson."""
        x = np.linspace(1.0, 10.0, 50)
        y = np.exp(x)  # Exponential growth
        div = EnsembleGuard.compute_pairwise_diversity(x, y)

        # Spearman is rank-order invariant -> 1.0
        self.assertAlmostEqual(div["spearman_corr"], 1.0, places=3)
        # Pearson linear correlation is much lower
        self.assertLess(div["pearson_corr"], 0.80)

    def test_pairwise_diversity_pearson_and_spearman_cubic(self):
        """Monotonic cubic series has Spearman=1.0 and Pearson < 1.0."""
        x = pd.Series(np.linspace(1.0, 10.0, 50))
        y = pd.Series(x.values ** 3)
        div = EnsembleGuard.compute_pairwise_diversity(x, y)
        self.assertEqual(div["metric_type"], "continuous")
        self.assertEqual(div["spearman_corr"], 1.0)
        self.assertLess(div["pearson_corr"], 1.0)
        self.assertGreater(div["pearson_corr"], 0.85)

    def test_pairwise_diversity_microscopic_anti_correlated_no_false_1_0(self):
        """Microscopic inversely correlated vectors report r ~ -1.0 and MUST NOT report false 1.0."""
        x = np.linspace(1e-9, 9e-9, 50)
        y = np.linspace(9e-9, 1e-9, 50)
        div = EnsembleGuard.compute_pairwise_diversity(x, y, rel_epsilon=0.05)

        self.assertEqual(div["metric_type"], "continuous")
        self.assertLess(div["pearson_corr"], -0.90)
        self.assertLess(div["spearman_corr"], -0.90)
        self.assertNotEqual(div["pearson_corr"], 1.0)
        self.assertNotEqual(div["spearman_corr"], 1.0)
        self.assertNotEqual(div["cohen_kappa"], 1.0)
        self.assertGreater(div["disagreement_pct"], 50.0)

    def test_pairwise_diversity_microscopic_constant_different_reports_zero(self):
        """Microscopic different constant vectors report r = 0.0 and 100% disagreement."""
        n = 50
        a = pd.Series([1e-9] * n)
        b = pd.Series([2e-9] * n)
        div = EnsembleGuard.compute_pairwise_diversity(a, b)

        self.assertEqual(div["pearson_corr"], 0.0)
        self.assertEqual(div["spearman_corr"], 0.0)
        self.assertEqual(div["cohen_kappa"], 0.0)
        self.assertEqual(div["disagreement_count"], n)
        self.assertEqual(div["disagreement_pct"], 100.0)

    def test_pairwise_diversity_microscopic_identical_reports_one(self):
        """Microscopic identical vectors report r = 1.0 and 0% disagreement."""
        x = np.linspace(1e-9, 9e-9, 50)
        div = EnsembleGuard.compute_pairwise_diversity(x, x.copy())

        self.assertEqual(div["pearson_corr"], 1.0)
        self.assertEqual(div["spearman_corr"], 1.0)
        self.assertEqual(div["disagreement_count"], 0)
        self.assertEqual(div["disagreement_pct"], 0.0)


class TestEnsembleGuardNumericStringDtype(unittest.TestCase):
    """Verifies series containing numeric strings in object dtype."""

    def test_numeric_string_series_object_dtype(self):
        """Numeric strings are coerced to float and evaluated under continuous regime."""
        preds_a = pd.Series(["10.5", "20.3", "30.1", "40.9", "50.0"], dtype=object)
        preds_b = pd.Series(["10.6", "20.2", "30.0", "41.0", "50.1"], dtype=object)

        div = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_b)
        self.assertEqual(div["metric_type"], "continuous")
        self.assertIn("pearson_corr", div)
        self.assertIn("spearman_corr", div)
        self.assertGreater(div["pearson_corr"], 0.99)
        self.assertEqual(div["cohen_kappa"], div["pearson_corr"])

    def test_true_categorical_string_series(self):
        """Non-numeric string labels evaluate under discrete Cohen's Kappa regime."""
        preds_a = pd.Series(["cat", "dog", "bird"] * 10)
        preds_b = pd.Series(["cat", "dog", "bird"] * 10)

        div = EnsembleGuard.compute_pairwise_diversity(preds_a, preds_b)
        self.assertEqual(div["metric_type"], "discrete")
        self.assertAlmostEqual(div["cohen_kappa"], 1.0)
        self.assertNotIn("pearson_corr", div)


class TestEnsembleGuardContinuousRescueSamples(unittest.TestCase):
    """Verifies continuous rescue sample analysis with error margin reductions."""

    def test_continuous_rescue_samples_synergistic(self):
        """Ensemble error strictly lower than both constituents qualifies as synergistic rescue."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Model A is biased +5.0; Model B is biased -5.0
        model_a = y_true + 5.0
        model_b = y_true - 5.0
        # Ensemble averages perfectly to ground truth
        ensemble = y_true.copy()

        res = EnsembleGuard.analyze_rescue_samples(y_true, ensemble, model_a, model_b)
        self.assertEqual(res["rescued_from_model_0"], 5)
        self.assertEqual(res["rescued_from_model_1"], 5)
        self.assertEqual(res["synergistic_rescues"], 5)
        self.assertEqual(res["degraded_samples"], 0)
        self.assertEqual(res["net_gain"], 10)
        self.assertEqual(res["mode"], "continuous")

    def test_continuous_rescue_samples_single_model_rescue(self):
        """Ensemble correcting one inaccurate constituent model."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        # Model A has large error on sample 0
        model_a = np.array([25.0, 20.0, 30.0, 40.0])
        # Model B is accurate on all samples
        model_b = y_true.copy()
        # Ensemble stays close to Model B
        ensemble = np.array([11.0, 20.0, 30.0, 40.0])

        res = EnsembleGuard.analyze_rescue_samples(y_true, ensemble, model_a, model_b)
        self.assertGreaterEqual(res["rescued_from_model_0"], 1)
        self.assertEqual(res["rescued_from_model_1"], 0)

    def test_continuous_rescue_samples_degraded(self):
        """Ensemble worse than both constituents is counted as degraded."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        model_a = y_true.copy()
        model_b = y_true.copy()
        # Degraded ensemble
        ensemble = y_true + 15.0

        res = EnsembleGuard.analyze_rescue_samples(y_true, ensemble, model_a, model_b)
        self.assertEqual(res["degraded_samples"], 4)
        self.assertEqual(res["synergistic_rescues"], 0)
        self.assertLess(res["net_gain"], 0)

    def test_continuous_rescue_samples_small_scale_low_variance_scaling(self):
        """Rescue sample analysis on small scale (mean ~ 1e-6, std <= 1e-8) scales delta margin correctly."""
        n = 50
        y_true = np.full(n, 1e-6)
        # Model A and Model B have opposing errors of 5e-7
        model_a = y_true + 5e-7
        model_b = y_true - 5e-7
        # Ensemble perfectly cancels bias
        ensemble = y_true.copy()

        res = EnsembleGuard.analyze_rescue_samples(y_true, ensemble, model_a, model_b, rel_epsilon=0.05)
        self.assertEqual(res["mode"], "continuous")
        self.assertEqual(res["synergistic_rescues"], n)
        self.assertLess(res["delta_margin"], 1e-6)
        self.assertGreater(res["delta_margin"], 0.0)

    def test_rescue_samples_delta_margin_microscopic_scale(self):
        """Microscopic scale targets (y in [1e-6, 1e-5]) preserve non-zero delta_margin."""
        rng = np.random.default_rng(42)
        y_true = rng.uniform(1e-6, 1e-5, 50)
        ens = y_true.copy()
        m0 = y_true + 1e-7
        m1 = y_true - 1e-7

        res = EnsembleGuard.analyze_rescue_samples(y_true, ens, m0, m1, rel_epsilon=0.05)
        self.assertEqual(res["mode"], "continuous")
        self.assertGreater(res["delta_margin"], 0.0)


class TestEnsembleGuardAuditDiversity(unittest.TestCase):
    """Verifies audit_constituent_diversity reporting and thresholds."""

    def test_audit_constituent_diversity_regression_reporting(self):
        """Regression constituent diversity checks report Pearson correlation in error messages."""
        n = 50
        rng = np.random.default_rng(42)
        base = rng.normal(100.0, 15.0, n)
        # Nearly identical constituent (high correlation > 0.85)
        tweak = base + rng.normal(0, 0.01, n)

        df1 = pd.DataFrame({"id": range(n), "pred_target": base})
        df2 = pd.DataFrame({"id": range(n), "pred_target": tweak})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_target", max_kappa=0.85, min_disagreement_pct=5.0
        )
        self.assertFalse(passed)
        # Verifies message mentions Pearson Correlation, not Cohen's Kappa
        self.assertTrue(any("Pearson Correlation =" in e or "Pearson Correlation" in e for e in errs))
        self.assertFalse(any("Cohen's Kappa =" in e for e in errs))
        self.assertTrue(any("rel_eps=0.05" in e for e in errs))
        self.assertEqual(details["metric_type"], "continuous")

    def test_audit_constituent_diversity_passes_orthogonal_regression_models(self):
        """Orthogonal regression models pass diversity audit."""
        n = 100
        rng = np.random.default_rng(42)
        m1 = rng.normal(100.0, 15.0, n)
        # Moderate correlation (~0.50) with substantial variance
        m2 = 0.5 * m1 + 0.5 * rng.normal(100.0, 15.0, n)

        df1 = pd.DataFrame({"id": range(n), "pred_target": m1})
        df2 = pd.DataFrame({"id": range(n), "pred_target": m2})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_target", max_kappa=0.85, min_disagreement_pct=5.0
        )
        self.assertTrue(passed)
        self.assertEqual(len(errs), 0)

    def test_audit_constituent_diversity_rejects_monotonic_exponential(self):
        """Monotonic exponential pseudo-ensemble (Spearman = 1.0 > 0.85, Pearson ~ 0.72) is rejected."""
        n = 100
        x = np.linspace(1.0, 10.0, n)
        y = np.exp(x)

        df1 = pd.DataFrame({"id": range(n), "pred_target": x})
        df2 = pd.DataFrame({"id": range(n), "pred_target": y})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_target", max_kappa=0.85
        )
        self.assertFalse(passed, "EnsembleGuard failed to reject rank-identical monotonic models!")
        self.assertTrue(any("pseudo-ensembling detected" in e for e in errs))
        self.assertTrue(any("Spearman" in e for e in errs))
        self.assertEqual(details["pairwise_stats"][0]["spearman_corr"], 1.0)
        self.assertLess(details["pairwise_stats"][0]["pearson_corr"], 0.85)

    def test_audit_constituent_diversity_rejects_monotonic_logarithmic(self):
        """Monotonic logarithmic pseudo-ensemble (Spearman = 1.0 > 0.85, Pearson ~ 0.82) is rejected."""
        n = 100
        x = np.linspace(1.0, 1000.0, n)
        y = np.log(x)

        df1 = pd.DataFrame({"id": range(n), "pred_target": x})
        df2 = pd.DataFrame({"id": range(n), "pred_target": y})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_target", max_kappa=0.85
        )
        self.assertFalse(passed)
        self.assertTrue(any("Spearman" in e for e in errs))


class TestEnsembleGuardBackwardCompatibility(unittest.TestCase):
    """Verifies existing classification contracts remain 100% operational."""

    def test_classification_pairwise_diversity(self):
        """Classification strings retain Cohen's Kappa score."""
        preds = pd.Series(["yes", "no", "yes", "no"] * 20)
        div = EnsembleGuard.compute_pairwise_diversity(preds, preds)
        self.assertEqual(div["metric_type"], "discrete")
        self.assertEqual(div["cohen_kappa"], 1.0)
        self.assertEqual(div["disagreement_count"], 0)
        self.assertEqual(div["disagreement_pct"], 0.0)

    def test_classification_audit_constituent_diversity(self):
        """Classification pseudo-ensemble is rejected with Cohen's Kappa messaging."""
        df1 = pd.DataFrame({"id": range(100), "pred_label": ["yes"] * 96 + ["no"] * 4})
        df2 = pd.DataFrame({"id": range(100), "pred_label": ["yes"] * 95 + ["no"] * 5})

        passed, errs, details = EnsembleGuard.audit_constituent_diversity(
            [df1, df2], col="pred_label", max_kappa=0.85, min_disagreement_pct=5.0
        )
        self.assertFalse(passed)
        self.assertTrue(any("Cohen's Kappa" in e for e in errs))
        self.assertEqual(details["metric_type"], "discrete")

    def test_analyze_rescue_samples_discrete_backward_compatible(self):
        """Discrete classification rescue analysis retains exact integer behavior."""
        y_true = np.array([1, 1, 0, 0, 1])
        base_a = np.array([0, 1, 0, 1, 0])
        base_b = np.array([1, 0, 0, 0, 0])
        ens = np.array([1, 1, 0, 0, 0])

        rescue = EnsembleGuard.analyze_rescue_samples(y_true, ens, base_a, base_b)
        self.assertEqual(rescue["mode"], "discrete")
        self.assertEqual(rescue["rescued_from_model_0"], 2)
        self.assertEqual(rescue["rescued_from_model_1"], 1)
        self.assertEqual(rescue["synergistic_rescues"], 0)
        self.assertEqual(rescue["degraded_samples"], 0)
        self.assertEqual(rescue["net_gain"], 3)

    def test_verify_ensemble_superiority(self):
        """Superiority verification operates correctly for maximize and minimize."""
        # Maximize: pass if margin >= min_margin
        p_max, _, _ = EnsembleGuard.verify_ensemble_superiority(0.85, [0.80, 0.79], min_margin=0.01, direction="maximize")
        self.assertTrue(p_max)

        # Minimize: pass if best_constituent - ensemble >= min_margin
        p_min, _, _ = EnsembleGuard.verify_ensemble_superiority(0.40, [0.45, 0.48], min_margin=0.01, direction="minimize")
        self.assertTrue(p_min)


if __name__ == "__main__":
    unittest.main()
