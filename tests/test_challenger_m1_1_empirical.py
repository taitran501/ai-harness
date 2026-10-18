r"""Empirical Challenger Verification Suite for Milestone 1.

Author: Challenger M1 1
Target: tools/audit_gate.py and harness/ensemble_guard.py
Milestone: Milestone 1 (Backend Audit Gate & EnsembleGuard Generalization)

Empirically challenges the implementation across 5 required adversarial vectors:
1. Machine epsilon noise ($10^{-16}$) vs micro-jitter ($10^{-8}$) vs genuine effect ($0.05 \times \sigma$).
2. Extreme scales: $y \in [10^7, 10^8]$ vs $y \in [10^{-6}, 10^{-5}]$.
3. Constant / zero-variance predictions and zero-variance loss differentials.
4. Non-linear monotonic transformations (verifying Spearman $\approx 1.0$ vs Pearson $< 1.0$).
5. Continuous sample rescue analysis under identical vs perturbed vs superior models.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.ensemble_guard import EnsembleGuard
from tools.audit_gate import (
    compute_continuous_significance,
    compute_diff_metrics,
)


class TestAdversarialVector1NoiseVsMicroJitterVsGenuine(unittest.TestCase):
    """Vector 1: Machine epsilon noise vs micro-jitter vs genuine effect."""

    def setUp(self):
        self.rng = np.random.default_rng(20261009)
        self.n = 500
        self.sigma = 25.0
        self.base_vals = self.rng.normal(loc=150.0, scale=self.sigma, size=self.n)
        self.base_df = pd.DataFrame({"id": range(self.n), "pred_val": self.base_vals})

    def test_audit_gate_machine_epsilon_noise_zero_diff(self):
        """Machine epsilon noise (1e-16 to eps) MUST yield exactly 0 differing rows."""
        eps = np.finfo(float).eps  # ~2.22e-16
        for noise_val in [1e-16, eps, 2 * eps]:
            cand_vals = self.base_vals + noise_val
            cand_df = pd.DataFrame({"id": range(self.n), "pred_val": cand_vals})
            diff_count, details = compute_diff_metrics(
                cand_df, self.base_df, id_col="id", rel_epsilon=0.01
            )
            self.assertEqual(diff_count, 0, f"Machine epsilon noise {noise_val} triggered {diff_count} diffs")
            self.assertEqual(details["pred_val"]["different_count"], 0)
            self.assertEqual(details["pred_val"]["pct_changed"], 0.0)

    def test_audit_gate_micro_jitter_rejected(self):
        """Micro-jitter (1e-8, 1e-9) MUST yield 0 differing rows under rel_epsilon=0.01."""
        for scale in [1e-8, 1e-9, 5e-8]:
            jitter = self.rng.uniform(-scale, scale, size=self.n)
            cand_vals = self.base_vals + jitter
            cand_df = pd.DataFrame({"id": range(self.n), "pred_val": cand_vals})
            diff_count, details = compute_diff_metrics(
                cand_df, self.base_df, id_col="id", rel_epsilon=0.01
            )
            self.assertEqual(diff_count, 0, f"Micro jitter scale {scale} triggered {diff_count} diffs")
            self.assertEqual(details["pred_val"]["pct_changed"], 0.0)

    def test_audit_gate_genuine_effect_detected_with_exact_proportion(self):
        """Genuine effect (0.05 * sigma) MUST be accurately detected on shifted subset."""
        # Shift exactly 15% of rows (75 rows) by 0.05 * sigma
        shift = 0.05 * self.sigma
        cand_vals = self.base_vals.copy()
        cand_vals[:75] += shift

        cand_df = pd.DataFrame({"id": range(self.n), "pred_val": cand_vals})
        diff_count, details = compute_diff_metrics(
            cand_df, self.base_df, id_col="id", rel_epsilon=0.01
        )
        self.assertEqual(diff_count, 75)
        self.assertAlmostEqual(details["pred_val"]["pct_changed"], 15.0)

    def test_audit_gate_sub_threshold_boundary_rejected(self):
        """Effect below relative epsilon threshold (0.009 * sigma < 0.01) must NOT trigger diff."""
        # 0.009 * sigma is strictly below rel_epsilon = 0.01
        shift = 0.009 * self.sigma
        cand_vals = self.base_vals + shift
        cand_df = pd.DataFrame({"id": range(self.n), "pred_val": cand_vals})
        diff_count, _ = compute_diff_metrics(
            cand_df, self.base_df, id_col="id", rel_epsilon=0.01
        )
        self.assertEqual(diff_count, 0)

    def test_ensemble_guard_pairwise_noise_vs_microjitter_vs_genuine(self):
        """EnsembleGuard disagreement: 0% on epsilon/jitter, 100% on genuine 0.05*sigma."""
        a = self.base_vals
        # Epsilon
        div_eps = EnsembleGuard.compute_pairwise_diversity(a, a + 1e-16, rel_epsilon=0.05)
        self.assertEqual(div_eps["disagreement_count"], 0)
        self.assertEqual(div_eps["disagreement_pct"], 0.0)

        # Micro-jitter
        jitter = self.rng.uniform(-1e-8, 1e-8, size=self.n)
        div_jit = EnsembleGuard.compute_pairwise_diversity(a, a + jitter, rel_epsilon=0.05)
        self.assertEqual(div_jit["disagreement_count"], 0)
        self.assertEqual(div_jit["disagreement_pct"], 0.0)

        # Genuine effect (0.06 * sigma >= 0.05 rel_epsilon)
        div_gen = EnsembleGuard.compute_pairwise_diversity(a, a + 0.06 * self.sigma, rel_epsilon=0.05)
        self.assertEqual(div_gen["disagreement_count"], self.n)
        self.assertEqual(div_gen["disagreement_pct"], 100.0)


class TestAdversarialVector2ExtremeScales(unittest.TestCase):
    """Vector 2: Extreme scales: y in [10^7, 10^8] vs y in [10^-6, 10^-5]."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n = 300

    def test_extreme_large_scale_macro_invariance(self):
        """Scale [10^7, 10^8] (e.g. GDP / transaction totals)."""
        # Mean 5e7, sigma 1e7
        sigma = 1e7
        base_vals = self.rng.normal(loc=5e7, scale=sigma, size=self.n)
        base_df = pd.DataFrame({"id": range(self.n), "pred_val": base_vals})

        # Dollar-level noise ($1.00 on $50M) must NOT trigger differing rows
        cand_df_noise = pd.DataFrame({"id": range(self.n), "pred_val": base_vals + 1.0})
        diff_count, _ = compute_diff_metrics(cand_df_noise, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 0)

        # Genuine effect of 200,000 (0.02 * sigma >= 0.01 rel_eps) on 50 rows
        cand_vals = base_vals.copy()
        cand_vals[:50] += 0.02 * sigma
        cand_df_shift = pd.DataFrame({"id": range(self.n), "pred_val": cand_vals})
        diff_count_shift, _ = compute_diff_metrics(cand_df_shift, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count_shift, 50)

        # EnsembleGuard on large scale
        div_guard = EnsembleGuard.compute_pairwise_diversity(base_vals, base_vals + 5.0, rel_epsilon=0.05)
        self.assertEqual(div_guard["disagreement_count"], 0)

        # Continuous significance on large scale
        y_true = base_vals
        cand_preds = base_vals + 1e5
        base_preds = base_vals + 5e5
        sig = compute_continuous_significance(y_true, cand_preds, base_preds)
        self.assertEqual(sig["cand_better_count"], self.n)
        self.assertTrue(sig["is_significant"])
        self.assertLess(sig["p_value"], 0.001)

    def test_extreme_small_scale_micro_invariance(self):
        """Scale [10^-6, 10^-5] (e.g. microscopic rates, biochemical concentrations)."""
        sigma = 2e-6
        base_vals = self.rng.normal(loc=5e-6, scale=sigma, size=self.n)
        base_df = pd.DataFrame({"id": range(self.n), "pred_val": base_vals})

        # Noise of 1e-12 must NOT trigger differing rows
        cand_df_noise = pd.DataFrame({"id": range(self.n), "pred_val": base_vals + 1e-12})
        diff_count, _ = compute_diff_metrics(cand_df_noise, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 0)

        # Genuine effect: shift 60 rows by 0.05 * sigma (1e-7)
        cand_vals = base_vals.copy()
        cand_vals[:60] += 0.05 * sigma
        cand_df_shift = pd.DataFrame({"id": range(self.n), "pred_val": cand_vals})
        diff_count_shift, _ = compute_diff_metrics(cand_df_shift, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count_shift, 60)

        # EnsembleGuard on microscopic scale
        div_noise = EnsembleGuard.compute_pairwise_diversity(base_vals, base_vals + 1e-12, rel_epsilon=0.05)
        self.assertEqual(div_noise["disagreement_count"], 0)

        # Continuous significance on small scale:
        # Standard small scale (where squared errors exceed atol 1e-8):
        y_std = self.rng.uniform(0.01, 0.05, self.n)
        cand_std = y_std + 0.001
        base_std = y_std + 0.010
        sig_std = compute_continuous_significance(y_std, cand_std, base_std)
        self.assertEqual(sig_std["cand_better_count"], self.n)
        self.assertTrue(sig_std["is_significant"])

        # Microscopic scale (y in [10^-6, 10^-5]) paired t-test correctly verifies significance:
        y_micro = base_vals
        cand_micro = base_vals + 1e-8
        base_micro = base_vals + 1e-7
        sig_micro = compute_continuous_significance(y_micro, cand_micro, base_micro)
        self.assertEqual(sig_micro["cand_better_count"], self.n)
        self.assertTrue(sig_micro["is_significant"])
        self.assertLess(sig_micro["p_value"], 0.05)


class TestAdversarialVector3ConstantZeroVariance(unittest.TestCase):
    """Vector 3: Constant / zero-variance predictions and paired differences."""

    def test_constant_identical_vectors(self):
        """Identical constant vectors evaluate cleanly without division by zero."""
        n = 100
        for val in [0.0, 1.0, -42.5, 1e8]:
            cand = pd.DataFrame({"id": range(n), "pred_val": [val] * n})
            base = pd.DataFrame({"id": range(n), "pred_val": [val] * n})
            diff_count, details = compute_diff_metrics(cand, base, id_col="id")
            self.assertEqual(diff_count, 0)
            self.assertEqual(details["pred_val"]["different_count"], 0)

            div = EnsembleGuard.compute_pairwise_diversity(cand["pred_val"], base["pred_val"])
            self.assertEqual(div["pearson_corr"], 1.0)
            self.assertEqual(div["spearman_corr"], 1.0)
            self.assertEqual(div["cohen_kappa"], 1.0)
            self.assertEqual(div["disagreement_count"], 0)

    def test_constant_different_vectors(self):
        """Different constant vectors report 0 correlation and 100% disagreement."""
        n = 50
        cand = pd.Series([10.0] * n)
        base = pd.Series([20.0] * n)
        div = EnsembleGuard.compute_pairwise_diversity(cand, base)
        self.assertEqual(div["pearson_corr"], 0.0)
        self.assertEqual(div["spearman_corr"], 0.0)
        self.assertEqual(div["disagreement_count"], n)
        self.assertEqual(div["disagreement_pct"], 100.0)

    def test_constant_difference_significance_test(self):
        """When candidate and baseline have constant error differential, t-test handles 0 variance."""
        n = 50
        y = np.linspace(10, 50, n)
        # base has squared error 25 everywhere, cand has squared error 1 everywhere
        base = y + 5.0
        cand = y + 1.0

        # Loss differential is strictly constant: (1 - 25) = -24 everywhere
        sig = compute_continuous_significance(y, cand, base)
        self.assertEqual(sig["cand_better_count"], n)
        self.assertEqual(sig["base_better_count"], 0)
        self.assertIn("p_value", sig)
        self.assertIn("statistic", sig)

    def test_both_exact_zero_constant(self):
        """Both predictions are zero everywhere."""
        n = 40
        cand = pd.DataFrame({"id": range(n), "pred_val": [0.0] * n})
        base = pd.DataFrame({"id": range(n), "pred_val": [0.0] * n})
        diff_count, _ = compute_diff_metrics(cand, base, id_col="id")
        self.assertEqual(diff_count, 0)

        div = EnsembleGuard.compute_pairwise_diversity(cand["pred_val"], base["pred_val"])
        self.assertEqual(div["pearson_corr"], 1.0)
        self.assertEqual(div["disagreement_count"], 0)


class TestAdversarialVector4MonotonicTransformations(unittest.TestCase):
    """Vector 4: Non-linear monotonic transformations (Spearman ~ 1.0 vs Pearson < 1.0)."""

    def setUp(self):
        self.n = 100
        self.x = np.linspace(1.0, 10.0, self.n)

    def test_exponential_monotonic_transformation(self):
        """Exponential transformation: Spearman == 1.0, Pearson < 0.80."""
        y = np.exp(self.x)
        div = EnsembleGuard.compute_pairwise_diversity(self.x, y)
        self.assertAlmostEqual(div["spearman_corr"], 1.0, places=4)
        self.assertLess(div["pearson_corr"], 0.80)
        self.assertGreater(div["pearson_corr"], 0.40)

    def test_cubic_monotonic_transformation(self):
        """Cubic transformation: Spearman == 1.0, Pearson < 0.95."""
        y = self.x ** 3
        div = EnsembleGuard.compute_pairwise_diversity(self.x, y)
        self.assertAlmostEqual(div["spearman_corr"], 1.0, places=4)
        self.assertLess(div["pearson_corr"], 0.95)
        self.assertGreater(div["pearson_corr"], 0.85)

    def test_logarithmic_monotonic_transformation(self):
        """Logarithmic transformation: Spearman == 1.0, Pearson < 0.85."""
        x_wide = np.linspace(1.0, 1000.0, self.n)
        y = np.log(x_wide)
        div = EnsembleGuard.compute_pairwise_diversity(x_wide, y)
        self.assertAlmostEqual(div["spearman_corr"], 1.0, places=4)
        self.assertLess(div["pearson_corr"], 0.85)

    def test_sigmoid_monotonic_transformation(self):
        """Sigmoid / logistic transformation: Spearman == 1.0, Pearson < 1.0."""
        x_centered = np.linspace(-6.0, 6.0, self.n)
        y = 1.0 / (1.0 + np.exp(-x_centered))
        div = EnsembleGuard.compute_pairwise_diversity(x_centered, y)
        self.assertAlmostEqual(div["spearman_corr"], 1.0, places=4)
        self.assertLess(div["pearson_corr"], 1.0)
        self.assertGreater(div["spearman_corr"], div["pearson_corr"])

    def test_non_monotonic_parabola_breaks_spearman(self):
        """Symmetric parabola: both Spearman and Pearson collapse near 0."""
        x_sym = np.linspace(-10.0, 10.0, 101)
        y = x_sym ** 2
        div = EnsembleGuard.compute_pairwise_diversity(x_sym, y)
        self.assertLess(abs(div["spearman_corr"]), 0.05)
        self.assertLess(abs(div["pearson_corr"]), 0.05)

    def test_monotonic_exponential_rejected_by_constituent_diversity_audit(self):
        """Monotonic exponential pseudo-ensemble is rejected by audit_constituent_diversity."""
        y = np.exp(self.x)
        df1 = pd.DataFrame({"id": range(self.n), "pred_target": self.x})
        df2 = pd.DataFrame({"id": range(self.n), "pred_target": y})
        passed, errs, _ = EnsembleGuard.audit_constituent_diversity([df1, df2], col="pred_target", max_kappa=0.85)
        self.assertFalse(passed)
        self.assertTrue(any("Spearman" in e for e in errs))


class TestAdversarialVector5RescueSampleAnalysis(unittest.TestCase):
    """Vector 5: Continuous rescue sample analysis under identical vs perturbed vs superior models."""

    def setUp(self):
        self.rng = np.random.default_rng(777)
        self.n = 200
        self.y_true = self.rng.normal(100.0, 20.0, self.n)
        self.sigma_y = float(np.std(self.y_true))

    def test_identical_models_yield_zero_rescues_and_zero_gain(self):
        """Identical models (ens == m0 == m1): exactly 0 rescues and 0 net gain."""
        m0 = self.y_true + self.rng.normal(0, 5.0, self.n)
        m1 = m0.copy()
        ens = m0.copy()

        rescue = EnsembleGuard.analyze_rescue_samples(self.y_true, ens, m0, m1, rel_epsilon=0.05)
        self.assertEqual(rescue["mode"], "continuous")
        self.assertEqual(rescue["rescued_from_model_0"], 0)
        self.assertEqual(rescue["rescued_from_model_1"], 0)
        self.assertEqual(rescue["synergistic_rescues"], 0)
        self.assertEqual(rescue["degraded_samples"], 0)
        self.assertEqual(rescue["net_gain"], 0)

    def test_perturbed_models_sub_threshold_yield_zero_rescues(self):
        """Perturbed models with noise smaller than delta margin yield 0 rescues."""
        m0 = self.y_true + 5.0
        # Perturbation is 0.001 (far below delta = 0.05 * 20.0 = 1.0)
        m1 = m0 + 0.001
        ens = (m0 + m1) / 2.0  # Still ~5.0 error

        rescue = EnsembleGuard.analyze_rescue_samples(self.y_true, ens, m0, m1, rel_epsilon=0.05)
        self.assertEqual(rescue["rescued_from_model_0"], 0)
        self.assertEqual(rescue["rescued_from_model_1"], 0)
        self.assertEqual(rescue["synergistic_rescues"], 0)
        self.assertEqual(rescue["net_gain"], 0)

    def test_superior_synergistic_ensemble_yields_full_rescues(self):
        """Synergistic ensemble (averaging out opposing biases) yields 100% synergistic rescues."""
        # Model 0 overpredicts by 10.0; Model 1 underpredicts by 10.0
        m0 = self.y_true + 10.0
        m1 = self.y_true - 10.0
        # Ensemble averages perfectly to ground truth
        ens = (m0 + m1) / 2.0  # Exactly y_true

        rescue = EnsembleGuard.analyze_rescue_samples(self.y_true, ens, m0, m1, rel_epsilon=0.05)
        self.assertEqual(rescue["rescued_from_model_0"], self.n)
        self.assertEqual(rescue["rescued_from_model_1"], self.n)
        self.assertEqual(rescue["synergistic_rescues"], self.n)
        self.assertEqual(rescue["degraded_samples"], 0)
        self.assertEqual(rescue["net_gain"], 2 * self.n)

    def test_degraded_ensemble_yields_degraded_samples(self):
        """Degraded ensemble with larger error yields degraded_samples > 0 and negative net gain."""
        m0 = self.y_true + 1.0
        m1 = self.y_true - 1.0
        # Terrible ensemble
        ens = self.y_true + 25.0

        rescue = EnsembleGuard.analyze_rescue_samples(self.y_true, ens, m0, m1, rel_epsilon=0.05)
        self.assertEqual(rescue["degraded_samples"], self.n)
        self.assertEqual(rescue["synergistic_rescues"], 0)
        self.assertEqual(rescue["rescued_from_model_0"], 0)
        self.assertEqual(rescue["rescued_from_model_1"], 0)
        self.assertEqual(rescue["net_gain"], -self.n)


class TestAdversarialVectorCLIStressAndEdgeCases(unittest.TestCase):
    """Stress tests for CLI execution, missing/non-standard schemas, and edge parameters."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_rejects_identical_candidate_sha256(self):
        """Audit gate strictly blocks candidate file that has identical sha256 to baseline."""
        df = pd.DataFrame({"id": range(10), "pred_val": range(10)})
        f_cand = self.p / "cand.csv"
        f_base = self.p / "base.csv"
        df.to_csv(f_cand, index=False)
        df.to_csv(f_base, index=False)

        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--candidate", str(f_cand),
            "--baseline", str(f_base),
            "--output-zip", str(self.p / "out.zip"),
            "--receipt", str(self.p / "receipt.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("100% DUPLICATE FILE DETECTED", proc.stderr)

    def test_cli_rejects_pseudo_ensemble_correlated_components(self):
        """Audit gate blocks ensemble candidate if constituents are highly correlated."""
        n = 100
        rng = np.random.default_rng(42)
        base = rng.normal(100.0, 10.0, n)
        # Highly correlated constituent (> 0.85)
        c1 = base
        c2 = base + rng.normal(0, 0.05, n)
        ens = (c1 + c2) / 2.0

        f_cand = self.p / "ens.csv"
        f_c1 = self.p / "c1.csv"
        f_c2 = self.p / "c2.csv"
        pd.DataFrame({"id": range(n), "pred_target": ens}).to_csv(f_cand, index=False)
        pd.DataFrame({"id": range(n), "pred_target": c1}).to_csv(f_c1, index=False)
        pd.DataFrame({"id": range(n), "pred_target": c2}).to_csv(f_c2, index=False)

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
        self.assertIn("Pearson Correlation", proc.stderr)


if __name__ == "__main__":
    unittest.main()
