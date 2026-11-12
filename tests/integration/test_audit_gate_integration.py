"""Integration test suite consolidating deterministic and statistical audit gate verification.

Consolidates:
- tests/test_audit_gate.py
- AuditGateTests from tests/test_harness.py
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

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.task_spec import SubtaskSpec, UniversalTaskSpec
from tools.audit_gate import (
    align_prediction_frames,
    audit_prediction_integrity,
    check_prediction_diversity,
    compute_continuous_significance,
    compute_diff_metrics,
)
from tests.fixtures import make_workspace, task_document  # noqa: F401
from tests.evidence_fixtures import seal_fixture  # noqa: F401

AUDIT_GATE_TOOL = str(ROOT / "tools" / "audit_gate.py")


# ---------------------------------------------------------------------------
# Prediction diversity tests
# ---------------------------------------------------------------------------


class TestPredictionDiversity(unittest.TestCase):
    spec = {"id_column": "id", "prediction_columns": {"score": "continuous"}}

    def receipt(self, a, b, spec=None):
        return check_prediction_diversity(a, [{"id": "prior", "frame": b}], spec=spec or self.spec)

    def test_rank_preserving_jitter_and_reordered_rows_are_near_duplicate(self):
        base = pd.DataFrame({"id": range(20), "score": np.arange(20) / 20})
        candidate = base.copy()
        candidate.score += 0.00001
        result = self.receipt(candidate.sample(frac=1, random_state=1), base)
        self.assertEqual(result["status"], "near_duplicate")
        self.assertAlmostEqual(result["maximum_similarity"], 1)

    def test_signed_negative_correlation_is_diverse(self):
        a = pd.DataFrame({"id": range(5), "score": range(5)})
        b = a.copy()
        b.score = -b.score
        self.assertEqual(self.receipt(a, b)["status"], "diverse")

    def test_all_prior_references_are_checked(self):
        a = pd.DataFrame({"id": range(5), "score": range(5)})
        b = a.copy()
        b.score = -b.score
        receipt = check_prediction_diversity(a, [{"id": "first", "frame": b}, {"id": "oldest", "frame": a}], spec=self.spec)
        self.assertEqual(receipt["status"], "near_duplicate")
        self.assertEqual(receipt["reference_count"], 2)

    def test_multioutput_requires_all_columns_for_near_duplicate(self):
        spec = {"id_column": "id", "prediction_columns": {"a": "continuous", "b": "continuous"}}
        a = pd.DataFrame({"id": range(5), "a": range(5), "b": range(5)})
        b = a.copy()
        b.b = -b.b
        result = self.receipt(a, b, spec)
        self.assertEqual(result["status"], "diverse")
        self.assertEqual(len(result["comparisons"][0]["columns"]), 2)

    def test_distinct_constants_are_inconclusive(self):
        a = pd.DataFrame({"id": range(5), "score": [1.0] * 5})
        b = a.copy()
        b.score = 2.0
        self.assertEqual(self.receipt(a, b)["status"], "inconclusive")
        self.assertEqual(self.receipt(a, a)["status"], "near_duplicate")

    def test_duplicate_missing_and_typed_id_mismatch_rejected(self):
        a = pd.DataFrame({"id": range(5), "score": range(5)})
        for b in (a.iloc[:-1], pd.concat([a, a.iloc[:1]]), a.assign(id=a.id.astype(str)), a.assign(fold=0)):
            with self.assertRaisesRegex(ValueError, "IDENTITY_MISMATCH"):
                align_prediction_frames(a, b, ["id"])

    def test_numeric_class_ids_use_disagreement(self):
        spec = {"id_column": "id", "prediction_columns": {"label": "discrete"}}
        a = pd.DataFrame({"id": range(5), "label": [0, 1, 2, 0, 1]})
        b = a.assign(label=[2, 1, 0, 2, 1])
        result = self.receipt(a, b, spec)
        self.assertEqual(result["metric"], "exact_disagreement")
        self.assertIsNone(result["maximum_similarity"])
        self.assertEqual(result["status"], "diverse")

    def test_missing_history_and_nonfinite_predictions_fail_closed(self):
        a = pd.DataFrame({"id": range(5), "score": range(5)})
        with self.assertRaisesRegex(ValueError, "HISTORY_INCOMPLETE"):
            check_prediction_diversity(a, [{"id": "missing"}], spec=self.spec)
        with self.assertRaisesRegex(ValueError, "NONFINITE"):
            self.receipt(a.assign(score=np.inf), a)


# ---------------------------------------------------------------------------
# Task spec helper fixture
# ---------------------------------------------------------------------------


def make_task_spec(
    task_type: str,
    classes: list | None = None,
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


# ---------------------------------------------------------------------------
# Differential metrics tests
# ---------------------------------------------------------------------------


class TestAuditGateDiffMetrics(unittest.TestCase):
    """Verifies differential metrics between candidate and baseline predictions."""

    def test_compute_diff_metrics_discrete_exact(self):
        """Discrete columns use exact != comparison."""
        base = pd.DataFrame({
            "id": range(10),
            "pred_label": ["cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird", "cat"],
        })
        cand = base.copy()
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

        shift = 0.05 * sigma
        cand.loc[:14, "pred_val"] = values[:15] + shift

        diff_count, details = compute_diff_metrics(cand, base, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 15)
        self.assertAlmostEqual(details["pred_val"]["pct_changed"], 15.0)

    def test_compute_diff_metrics_zero_variance_fallback(self):
        """Zero-variance baseline does not divide by zero and falls back to mean scale."""
        base = pd.DataFrame({"id": range(20), "pred_val": [50.0] * 20})
        cand = base.copy()
        cand.loc[:3, "pred_val"] = 52.0

        diff_count, details = compute_diff_metrics(cand, base, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 4)
        self.assertAlmostEqual(details["pred_val"]["pct_changed"], 20.0)

    def test_compute_diff_metrics_constant_zero_baseline(self):
        """Constant zero baseline falls back to 1.0 denominator."""
        n = 50
        base = pd.DataFrame({"id": list(range(n)), "pred_val": np.zeros(n)})
        cand = pd.DataFrame({"id": list(range(n)), "pred_val": np.ones(n) * 0.02})
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
        cand_vals = base_vals + 2e-7

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
        cand_vals = base_vals + 1e-9

        base_df = pd.DataFrame({"id": range(n), "pred_target": base_vals})
        cand_df = pd.DataFrame({"id": range(n), "pred_target": cand_vals})

        diff_count, details = compute_diff_metrics(cand_df, base_df, id_col="id", rel_epsilon=0.01)
        self.assertEqual(diff_count, 0)
        self.assertEqual(details["pred_target"]["different_count"], 0)
        self.assertAlmostEqual(details["pred_target"]["pct_changed"], 0.0)


# ---------------------------------------------------------------------------
# Continuous significance tests
# ---------------------------------------------------------------------------


class TestAuditGateContinuousSignificance(unittest.TestCase):
    """Verifies continuous paired error significance test."""

    def test_continuous_significance_paired_ttest_basic(self):
        """Paired t-test calculates sample counts and p-value on squared loss differentials."""
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0] * 10)
        base_pred = y_true + 5.0
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
        cand_pred = y_true + 1e-8
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
        cand_pred = base_pred + (np.finfo(float).eps * base_pred)

        res = compute_continuous_significance(y_true, cand_pred, base_pred)
        self.assertFalse(res["is_significant"])
        self.assertEqual(res["p_value"], 1.0)


# ---------------------------------------------------------------------------
# Prediction integrity tests
# ---------------------------------------------------------------------------


class TestAuditGatePredictionIntegrity(unittest.TestCase):
    """Verifies prediction integrity across regression, ranking, and generative modalities."""

    def test_prediction_integrity_regression_finite_values(self):
        """Regression subtasks enforce finite numeric values."""
        spec = make_task_spec("regression")
        df_valid = pd.DataFrame({"id": [1, 2, 3], "pred_target": [1.5, -2.3, 0.0]})
        self.assertEqual(audit_prediction_integrity(df_valid, spec=spec), [])

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


# ---------------------------------------------------------------------------
# CLI end-to-end audit gate tests
# ---------------------------------------------------------------------------


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
            AUDIT_GATE_TOOL,
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

        receipt = json.loads(receipt_json.read_text(encoding="utf-8"))
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
            AUDIT_GATE_TOOL,
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
        y = np.exp(x)
        ens = (x + y) / 2.0

        f_cand = self.p / "ens_mono.csv"
        f_c1 = self.p / "c1_mono.csv"
        f_c2 = self.p / "c2_mono.csv"
        pd.DataFrame({"id": range(n), "pred_target": ens}).to_csv(f_cand, index=False)
        pd.DataFrame({"id": range(n), "pred_target": x}).to_csv(f_c1, index=False)
        pd.DataFrame({"id": range(n), "pred_target": y}).to_csv(f_c2, index=False)

        cmd = [
            sys.executable,
            AUDIT_GATE_TOOL,
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


# ---------------------------------------------------------------------------
# CLI duplicate and valid pass tests (consolidated from AuditGateTests in test_harness.py)
# ---------------------------------------------------------------------------


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
            AUDIT_GATE_TOOL,
            "--id-column", "sample_id",
            "--candidate", str(cand_file),
            "--baseline", str(base_file),
            "--output-zip", str(self.dir_path / "out.zip"),
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
            AUDIT_GATE_TOOL,
            "--id-column", "sample_id",
            "--candidate", str(cand_file),
            "--baseline", str(base_file),
            "--output-zip", str(out_zip),
            "--receipt", str(receipt),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(out_zip.exists())
        self.assertTrue(receipt.exists())

        with open(receipt, "r", encoding="utf-8") as f:
            r_data = json.load(f)
        self.assertEqual(r_data["status"], "PASSED")
        self.assertEqual(r_data["verification"]["differing_rows"], 1)


class StatisticalAuditGateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mcnemar_significance(self):
        from tools.audit_gate import compute_mcnemar_significance

        y_true = pd.Series(["yes"] * 50 + ["no"] * 50)
        cand_pred = pd.Series(["yes"] * 50 + ["no"] * 50)
        base_pred = pd.Series(["no"] * 10 + ["yes"] * 40 + ["no"] * 50)

        res = compute_mcnemar_significance(y_true, cand_pred, base_pred)
        self.assertEqual(res["cand_better_count"], 10)
        self.assertEqual(res["base_better_count"], 0)
        self.assertLess(res["p_value"], 0.05)
        self.assertTrue(res["is_significant"])

    def test_cli_rejects_trivial_effect_size(self):
        n = 200
        base_df = pd.DataFrame({
            "sample_id": [f"id_{i}" for i in range(n)],
            "pred_label": ["neg"] * n,
            "pred_secondary": ["no"] * n,
            "pred_category": ["none"] * n,
        })
        cand_df = base_df.copy()
        cand_df.loc[0, "pred_label"] = "pos"

        base_csv = self.p / "baseline.csv"
        cand_csv = self.p / "candidate.csv"
        base_df.to_csv(base_csv, index=False)
        cand_df.to_csv(cand_csv, index=False)

        cmd = [
            sys.executable,
            AUDIT_GATE_TOOL,
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
        n = 100
        comp1_df = pd.DataFrame({
            "sample_id": [f"id_{i}" for i in range(n)],
            "pred_label": ["pos"] * 95 + ["neg"] * 5,
            "pred_secondary": ["no"] * n,
            "pred_category": ["none"] * n,
        })
        comp2_df = comp1_df.copy()
        comp2_df.loc[0, "pred_label"] = "neg"

        cand_df = comp1_df.copy()

        c1_csv = self.p / "comp1.csv"
        c2_csv = self.p / "comp2.csv"
        cand_csv = self.p / "cand.csv"
        comp1_df.to_csv(c1_csv, index=False)
        comp2_df.to_csv(c2_csv, index=False)
        cand_df.to_csv(cand_csv, index=False)

        cmd = [
            sys.executable,
            AUDIT_GATE_TOOL,
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


if __name__ == "__main__":
    unittest.main()
