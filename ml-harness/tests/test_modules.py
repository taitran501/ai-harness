"""Unit tests for src modules: data, metrics, models, and configuration utilities."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import (
    build_text,
    load_splits,
    parse_target,
    IDENTITIES,
    TARGET_TYPES,
)
from src.metrics import (
    compute_single_label_metrics,
    compute_subtask_c_identities,
    evaluate_predictions,
)
from src.models import build_model
from src.utils import get_git_commit_hash, set_seed
from train import merge_configs


class DataModuleTests(unittest.TestCase):
    def test_parse_target_none(self):
        target_type, identities = parse_target("none")
        self.assertIsNone(target_type)
        self.assertEqual(identities, [])

    def test_parse_target_valid_multilabel(self):
        target_type, identities = parse_target("group_l,g,lgbtqia+")
        self.assertEqual(target_type, "group")
        self.assertEqual(identities, ["l", "g", "lgbtqia+"])

    def test_parse_target_invalid_type(self):
        with self.assertRaises(ValueError):
            parse_target("community_l")

    def test_parse_target_invalid_identity(self):
        with self.assertRaises(ValueError):
            parse_target("individual_unknown")

    def test_build_text_variants(self):
        df = pd.DataFrame(
            [
                {
                    "yt_title": "Sample Title",
                    "yt_description": "Sample Description",
                    "yt_comment": "Sample Comment",
                }
            ]
        )
        self.assertEqual(build_text(df, "comment").iloc[0], "Sample Comment")
        self.assertEqual(
            build_text(df, "title_comment").iloc[0],
            "[TITLE] Sample Title [COMMENT] Sample Comment",
        )
        self.assertEqual(
            build_text(df, "full_context").iloc[0],
            "[TITLE] Sample Title [DESCRIPTION] Sample Description [COMMENT] Sample Comment",
        )

    def test_canonical_splits_integrity(self):
        split_path = Path("splits/fold_assignments.tsv")
        if not split_path.exists():
            self.skipTest("splits/fold_assignments.tsv not yet present")
        df = load_splits(split_path)
        self.assertGreater(len(df), 0)
        # Verify video_key never crosses folds within language
        leakage = df.groupby(["language", "video_key"])["fold"].nunique().max()
        self.assertEqual(leakage, 1)


class MetricsModuleTests(unittest.TestCase):
    def test_single_label_metrics(self):
        y_true = ["yes", "no", "yes", "yes"]
        y_pred = ["yes", "no", "no", "yes"]
        metrics = compute_single_label_metrics(y_true, y_pred, ["no", "yes"])
        self.assertIn("macro_f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_subtask_c_identities(self):
        true_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]])
        pred_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        metrics = compute_subtask_c_identities(true_mat, pred_mat)
        self.assertIn("macro_f1", metrics)
        self.assertIn("micro_f1", metrics)
        self.assertIn("per_label_f1", metrics)
        self.assertEqual(metrics["per_label_f1"]["l"], 1.0)


class ModelsModuleTests(unittest.TestCase):
    def test_build_model_registered(self):
        model = build_model("tfidf_logistic_regression")
        self.assertIsNotNone(model)

    def test_build_model_unknown(self):
        with self.assertRaises(ValueError):
            build_model("unregistered_model_architecture")


class ConfigMergeTests(unittest.TestCase):
    def test_merge_configs(self):
        base = {"runtime": {"seed": 42, "n_splits": 5}, "data": {"variant": "comment"}}
        exp = {"runtime": {"seed": 123}, "model": {"name": "bert"}}
        merged = merge_configs(base, exp)
        self.assertEqual(merged["runtime"]["seed"], 123)
        self.assertEqual(merged["runtime"]["n_splits"], 5)
        self.assertEqual(merged["data"]["variant"], "comment")
        self.assertEqual(merged["model"]["name"], "bert")

    def test_render_experiment_card(self):
        from src.utils import render_experiment_card
        config = {
            "experiment": {
                "id": "exp014_loss_focal",
                "branch": "loss",
                "parent_id": "exp010_model_xlmr",
                "hypothesis": "Focal loss will improve minority classes.",
                "intervention": "Switch BCE to Focal Loss.",
            }
        }
        summary_df = pd.DataFrame(
            [
                {
                    "language": "EN",
                    "input_variant": "comment",
                    "a_macro_f1": 0.75,
                    "b_macro_f1": 0.60,
                    "c_target_type_macro_f1_oracle_hate": 0.70,
                    "c_identity_macro_f1_oracle_hate": 0.35,
                    "c_identity_micro_f1_oracle_hate": 0.65,
                }
            ]
        )
        card = render_experiment_card(config, summary_df, "test_hash_123")
        self.assertIn("exp014_loss_focal", card)
        self.assertIn("Focal loss will improve minority classes.", card)
        self.assertIn("test_hash_123", card)
        self.assertIn("0.7500", card)


if __name__ == "__main__":
    unittest.main()

