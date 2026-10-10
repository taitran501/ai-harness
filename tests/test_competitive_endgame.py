"""Unit Test Suite for Competitive ML Endgame Modules.

Tests:
1. to_clipped_logit: Numerical stability and boundary clipping.
2. align_prediction_frames: Scrambled index recovery and ID alignment.
3. L2LogitStacker: Nested cross-validation and L2 regularized shrinkage.
4. RankAverager: Scale-invariant percentile rank blending.
5. TeacherLogitPrior: Auxiliary dataset prior extraction.
6. SubmissionPreflightValidator: Pre-submission schema and probability bounds.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from harness.endgame_submission import SubmissionPreflightValidator
from harness.foundation_models import MultiScaleBinner, HierarchicalDiscretizer, TeacherLogitPrior
from harness.stacking import (
    L2LogitStacker,
    RankAverager,
    align_prediction_frames,
    to_clipped_logit,
)


class TestClippedLogit(unittest.TestCase):
    def test_boundary_handling(self):
        probs = [0.0, 1.0, 0.5, 1e-18, 1.0 - 1e-18]
        logits = to_clipped_logit(probs, logit_bounds=(-15.0, 15.0))
        self.assertEqual(len(logits), 5)
        self.assertAlmostEqual(logits[2], 0.0, places=5)
        self.assertEqual(logits[0], -15.0)
        self.assertEqual(logits[1], 15.0)
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_pass_through_existing_logits(self):
        raw_logits = np.array([-20.0, 0.0, 20.0])
        clipped = to_clipped_logit(raw_logits, logit_bounds=(-12.0, 12.0))
        self.assertEqual(clipped[0], -12.0)
        self.assertEqual(clipped[1], 0.0)
        self.assertEqual(clipped[2], 12.0)


class TestPredictionAligner(unittest.TestCase):
    def test_scrambled_id_alignment(self):
        ref_df = pd.DataFrame({"id": [10, 20, 30, 40], "feat": [1, 2, 3, 4]})
        # Model A is reversed
        df_a = pd.DataFrame({"id": [40, 30, 20, 10], "pred": [0.4, 0.3, 0.2, 0.1]})
        # Model B is shuffled with string array
        df_b = pd.DataFrame({"id": [20, 40, 10, 30], "proba": ["[0.8, 0.2]", "[0.6, 0.4]", "[0.9, 0.1]", "[0.7, 0.3]"]})

        aligned, names = align_prediction_frames([df_a, df_b], ref_df, id_column="id")
        self.assertEqual(aligned.shape, (4, 2))
        # Check alignment matches ref_df [10, 20, 30, 40]
        np.testing.assert_allclose(aligned[:, 0], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(aligned[:, 1], [0.1, 0.2, 0.3, 0.4])

    def test_missing_id_raises(self):
        ref_df = pd.DataFrame({"id": [1, 2, 3]})
        bad_df = pd.DataFrame({"id": [1, 2], "pred": [0.1, 0.2]})
        with self.assertRaises(ValueError):
            align_prediction_frames([bad_df], ref_df, id_column="id")


class TestL2LogitStacker(unittest.TestCase):
    def test_nested_stacking_improves_or_matches_auc(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.integers(0, 2, size=n)

        # 3 candidate models with varying noise
        m1 = np.clip(y * 0.7 + rng.normal(0, 0.3, size=n), 0.01, 0.99)
        m2 = np.clip(y * 0.6 + rng.normal(0, 0.4, size=n), 0.01, 0.99)
        m3 = np.clip(y * 0.5 + rng.normal(0, 0.5, size=n), 0.01, 0.99)

        X_oof = np.column_stack([m1, m2, m3])
        X_test = np.column_stack([m1[:50], m2[:50], m3[:50]])

        stacker = L2LogitStacker(C=0.01, n_splits=3, random_state=42)
        oof_pred, test_pred = stacker.fit_predict(X_oof, y, X_test=X_test)

        self.assertEqual(len(oof_pred), n)
        self.assertEqual(len(test_pred), 50)
        self.assertTrue(np.all((oof_pred >= 0.0) & (oof_pred <= 1.0)))

        auc_m1 = roc_auc_score(y, m1)
        auc_stack = roc_auc_score(y, oof_pred)
        # Stack should match or exceed individual baseline
        self.assertGreaterEqual(auc_stack, auc_m1 - 0.02)

        # Feature importance table
        fi = stacker.get_feature_importance()
        self.assertEqual(len(fi), 3)
        self.assertIn("weight", fi.columns)


class TestRankAverager(unittest.TestCase):
    def test_rank_percentile_blend(self):
        # Model 1 has high mean (0.8), Model 2 has low mean (0.2)
        m1 = np.array([0.7, 0.8, 0.9, 0.95])
        m2 = np.array([0.1, 0.2, 0.3, 0.4])

        blended = RankAverager.blend([m1, m2], weights=[0.5, 0.5])
        self.assertEqual(len(blended), 4)
        # Ranks must be monotonic
        self.assertTrue(np.all(np.diff(blended) > 0))
        self.assertAlmostEqual(blended[0], 0.25)
        self.assertAlmostEqual(blended[3], 1.0)


class TestTeacherLogitPrior(unittest.TestCase):
    def test_teacher_prior_extraction(self):
        aux_df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat2": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 0, 0, 1, 1, 1],
        })
        target_df = pd.DataFrame({
            "feat1": [2.5, 5.5],
            "feat2": ["a", "b"],
        })

        teacher = TeacherLogitPrior(target_column="target", model_type="logistic_regression")
        teacher.fit(aux_df)

        meta_feats = teacher.transform(target_df)
        self.assertEqual(len(meta_feats), 2)
        self.assertIn("teacher_prob", meta_feats.columns)
        self.assertIn("teacher_logit", meta_feats.columns)
        self.assertTrue(np.all(np.isfinite(meta_feats["teacher_logit"])))


class TestMultiScaleBinner(unittest.TestCase):
    def test_multi_scale_binning(self):
        s = pd.Series([15, 235, 1250], name="metric")
        res = MultiScaleBinner.bin_scales(s, scales=(10, 100, 1000))
        self.assertEqual(res.shape, (3, 4))
        self.assertIn("metric_exact", res.columns)
        self.assertIn("metric_scale_10", res.columns)
        self.assertIn("metric_scale_100", res.columns)
        self.assertIn("metric_scale_1000", res.columns)
        self.assertEqual(res["metric_scale_10"].iloc[0], 1)
        self.assertEqual(res["metric_scale_100"].iloc[1], 2)
        self.assertEqual(res["metric_scale_1000"].iloc[2], 1)

    def test_backwards_compatible_alias(self):
        s = pd.Series([12, 45, 102], name="dist")
        res = HierarchicalDiscretizer.decompose_distance(s, col_name="val")
        self.assertIn("val_exact", res.columns)
        self.assertEqual(res["val_exact"].iloc[0], 12)


class TestSubmissionPreflightValidator(unittest.TestCase):
    def test_valid_and_invalid_submissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref_path = tmp / "test.csv"
            pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]}).to_csv(ref_path, index=False)

            # Valid sub
            valid_sub = tmp / "valid.csv"
            pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.5, 0.9]}).to_csv(valid_sub, index=False)
            rep = SubmissionPreflightValidator.audit_submission(valid_sub, ref_path)
            self.assertTrue(rep.is_valid)
            self.assertEqual(len(rep.errors), 0)

            # Invalid sub: NaN present
            nan_sub = tmp / "nan.csv"
            pd.DataFrame({"id": [1, 2, 3], "target": [0.1, np.nan, 0.9]}).to_csv(nan_sub, index=False)
            rep_nan = SubmissionPreflightValidator.audit_submission(nan_sub, ref_path)
            self.assertFalse(rep_nan.is_valid)
            self.assertIn("NaN/null", rep_nan.errors[0])

            # Invalid sub: out of bounds
            oob_sub = tmp / "oob.csv"
            pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 1.5, 0.9]}).to_csv(oob_sub, index=False)
            rep_oob = SubmissionPreflightValidator.audit_submission(oob_sub, ref_path)
            self.assertFalse(rep_oob.is_valid)


if __name__ == "__main__":
    unittest.main()
