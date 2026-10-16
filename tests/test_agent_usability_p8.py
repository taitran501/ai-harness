"""Tests for Phase 8: Universal Modalities Expansion (Forecasting, CV, Speech, NLP Generation, Financial Purged Splits)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.data_strategy import DataStrategyEngine, SplitMode
from harness.endgame_submission import SubmissionPreflightValidator
from harness.evaluation import (
    compute_bleu,
    compute_box_iou,
    compute_cer,
    compute_dice_score,
    compute_exact_match,
    compute_iou_score,
    compute_mrr,
    compute_pearson,
    compute_pinball_loss,
    compute_rmsle,
    compute_smape,
    compute_wape,
    compute_wer,
    default_metric_direction,
    evaluate,
    mask_to_rle,
    rle_to_mask,
)
from harness.task_spec import UniversalTaskSpec


class TestAgentUsabilityP8(unittest.TestCase):
    """Verify universal modality support across forecasting, computer vision, speech, and financial splits."""

    def test_forecasting_metrics(self):
        """Verify RMSLE, SMAPE, WAPE, Pinball loss, and Pearson correlation."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 32.0, 38.0])

        rmsle = compute_rmsle(y_true, y_pred)
        self.assertGreater(rmsle, 0.0)
        self.assertLess(rmsle, 0.2)

        smape = compute_smape(y_true, y_pred)
        self.assertGreater(smape, 0.0)
        self.assertLess(smape, 0.2)

        wape = compute_wape(y_true, y_pred)
        self.assertGreater(wape, 0.0)
        self.assertLess(wape, 0.1)

        pinball = compute_pinball_loss(y_true, y_pred, alpha=0.5)
        self.assertGreater(pinball, 0.0)

        pearson = compute_pearson(y_true, y_pred)
        self.assertGreater(pearson, 0.95)

    def test_ranking_and_ir_metrics(self):
        """Verify Mean Reciprocal Rank (MRR)."""
        y_true = np.array([0, 0, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.9, 0.4])  # Rank 1
        self.assertAlmostEqual(compute_mrr(y_true, y_pred), 1.0)

        y_pred_2 = np.array([0.1, 0.9, 0.5, 0.2])  # Rank 2
        self.assertAlmostEqual(compute_mrr(y_true, y_pred_2), 0.5)

    def test_computer_vision_segmentation_rle_and_dice(self):
        """Verify RLE encoding, decoding, Dice score, and IoU."""
        # 10x10 binary mask with a 3x3 square
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:6] = 1

        rle = mask_to_rle(mask)
        self.assertIsInstance(rle, str)
        self.assertGreater(len(rle), 0)

        recovered = rle_to_mask(rle, (10, 10))
        np.testing.assert_array_equal(mask, recovered)

        # Perfect match
        dice = compute_dice_score(mask, recovered)
        self.assertAlmostEqual(dice, 1.0, places=4)
        iou = compute_iou_score(mask, recovered)
        self.assertAlmostEqual(iou, 1.0, places=4)

        # Bounding box IoU
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_box_iou(box1, box2), 1.0)
        box3 = [60, 60, 100, 100]  # Disjoint
        self.assertAlmostEqual(compute_box_iou(box1, box3), 0.0)

    def test_speech_and_text_generation_metrics(self):
        """Verify Word Error Rate (WER), Character Error Rate (CER), and BLEU."""
        refs = ["the quick brown fox", "hello world"]
        hyps = ["the fast brown fox", "hello world"]

        wer = compute_wer(refs, hyps)
        self.assertGreater(wer, 0.0)
        self.assertLess(wer, 0.3)

        cer = compute_cer(refs, hyps)
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 0.3)

        bleu = compute_bleu(refs, hyps)
        self.assertGreater(bleu, 0.4)

        em = compute_exact_match(refs, hyps)
        self.assertEqual(em, 0.5)

    def test_default_metric_direction(self):
        """Verify automated direction inference."""
        self.assertEqual(default_metric_direction("rmsle"), "minimize")
        self.assertEqual(default_metric_direction("wer"), "minimize")
        self.assertEqual(default_metric_direction("cer"), "minimize")
        self.assertEqual(default_metric_direction("pinball_loss"), "minimize")
        self.assertEqual(default_metric_direction("logloss"), "minimize")
        self.assertEqual(default_metric_direction("dice"), "maximize")
        self.assertEqual(default_metric_direction("iou"), "maximize")
        self.assertEqual(default_metric_direction("bleu"), "maximize")
        self.assertEqual(default_metric_direction("pr_auc"), "maximize")
        self.assertEqual(default_metric_direction("balanced_accuracy"), "maximize")

    def test_purged_group_time_series_split(self):
        """Verify financial PurgedGroupTimeSeriesSplit enforces temporal embargo."""
        df = pd.DataFrame({
            "id": range(100),
            "date_id": np.repeat(range(20), 5),  # 20 distinct dates
            "feature": np.random.randn(100),
        })
        splits = DataStrategyEngine.generate_splits(
            df,
            mode=SplitMode.PURGED_GROUP_TIME_SERIES,
            n_splits=3,
            group_col="date_id",
            holdout_ratio=0.1,  # 10% embargo buffer
        )
        self.assertGreaterEqual(len(splits), 2)
        for tr_idx, val_idx in splits:
            tr_dates = df.loc[tr_idx, "date_id"]
            val_dates = df.loc[val_idx, "date_id"]
            # Strict chronological ordering: max(train) < min(val)
            self.assertLess(tr_dates.max(), val_dates.min())
            # Embargo gap check: there must be skipped dates between max(train) and min(val)
            self.assertGreater(val_dates.min() - tr_dates.max(), 1)

    def test_multilabel_stratified_kfold(self):
        """Verify iterative multi-label stratification preserves label presence across folds."""
        Y = np.zeros((100, 3), dtype=int)
        Y[:20, 0] = 1  # 20 samples with label 0
        Y[15:35, 1] = 1  # 20 samples with label 1 (some overlap)
        Y[70:85, 2] = 1  # 15 samples with label 2

        df = pd.DataFrame(Y, columns=["label_a", "label_b", "label_c"])
        df["id"] = range(100)

        splits = DataStrategyEngine.generate_splits(
            df,
            mode=SplitMode.MULTILABEL_STRATIFIED_KFOLD,
            n_splits=5,
            target_cols=["label_a", "label_b", "label_c"],
        )
        self.assertEqual(len(splits), 5)
        for tr_idx, val_idx in splits:
            # Each validation fold must receive samples from each label
            self.assertGreater(df.loc[val_idx, "label_a"].sum(), 0)
            self.assertGreater(df.loc[val_idx, "label_b"].sum(), 0)
            self.assertGreater(df.loc[val_idx, "label_c"].sum(), 0)

    def test_submission_preflight_validator_string_predictions(self):
        """Verify SubmissionPreflightValidator accepts valid string predictions when probability_mode=False."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ref_path = tmp_path / "test.csv"
            pd.DataFrame({"id": [1, 2, 3]}).to_csv(ref_path, index=False)

            sub_path = tmp_path / "submission.csv"
            # String classification labels
            pd.DataFrame({"id": [1, 2, 3], "prediction": ["cat", "dog", "bird"]}).to_csv(sub_path, index=False)

            report = SubmissionPreflightValidator.audit_submission(
                sub_path,
                ref_path,
                id_column="id",
                target_column="prediction",
                probability_mode=False,
            )
            self.assertTrue(report.is_valid)
            self.assertEqual(len(report.errors), 0)

    def test_evaluate_segmentation_and_speech_tasks(self):
        """Verify UniversalTaskSpec evaluate on CV segmentation and ASR speech tasks."""
        # Test 1: Segmentation
        seg_spec_dict = {
            "task": {
                "name": "lung_seg",
                "task_type": "segmentation",
                "primary_metric": "dice",
                "id_column": "id",
                "subtasks": {
                    "mask": {
                        "task_type": "segmentation",
                        "primary_metric": "dice",
                    }
                },
            }
        }
        spec = UniversalTaskSpec.from_dict(seg_spec_dict)
        truth = pd.DataFrame({"id": [1, 2], "mask": [[1, 1, 0, 0], [0, 1, 1, 0]]})
        preds = {"mask": [[1, 1, 0, 0], [0, 1, 0, 0]]}
        res = evaluate(spec, truth, preds)
        self.assertIn("primary_score", res)
        self.assertGreater(res["primary_score"], 0.7)

        # Test 2: Speech Recognition
        asr_spec_dict = {
            "task": {
                "name": "voice_transcription",
                "task_type": "speech_recognition",
                "primary_metric": "wer",
                "direction": "minimize",
                "id_column": "id",
                "subtasks": {
                    "text": {
                        "task_type": "speech_recognition",
                        "primary_metric": "wer",
                        "direction": "minimize",
                    }
                },
            }
        }
        asr_spec = UniversalTaskSpec.from_dict(asr_spec_dict)
        truth_asr = pd.DataFrame({"id": [1, 2], "text": ["hello world", "good morning"]})
        preds_asr = {"text": ["hello world", "good evening"]}
        res_asr = evaluate(asr_spec, truth_asr, preds_asr)
        self.assertIn("primary_score", res_asr)
        self.assertLess(res_asr["primary_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
