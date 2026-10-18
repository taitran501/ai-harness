"""Tests for Phase 7: Non-NLP Modality Expansion (Tabular, Time-Series, Media)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.data_strategy import DataStrategyEngine, SplitMode
from harness.models import RandomForestModel, RobustTabularTransformer
from harness.task_spec import InputVariantSpec, UniversalTaskSpec


class TestAgentUsabilityP7(unittest.TestCase):
    """Verify non-NLP enhancements across time series, tabular transformers, and media path resolution."""

    def test_data_strategy_time_series_splits(self):
        """Verify that TimeSeriesSplit produces expanding-window leakage-free splits."""
        df = pd.DataFrame({"id": range(100), "val": range(100)})
        for mode in [
            SplitMode.TIME_SERIES,
            SplitMode.TEMPORAL,
            "time_series",
            "timeseries_split",
            "temporal",
        ]:
            splits = DataStrategyEngine.generate_splits(df, mode=mode, n_splits=5)
            self.assertEqual(len(splits), 5)
            for train_idx, val_idx in splits:
                # Train and val must be strictly disjoint
                self.assertEqual(len(set(train_idx).intersection(set(val_idx))), 0)
                # In time-series, all train indices must precede the maximum val index
                self.assertLess(max(train_idx), max(val_idx))

    def test_robust_tabular_transformer_mixed_types(self):
        """Verify RobustTabularTransformer handles mixed numeric, string categories, and NaNs."""
        train_df = pd.DataFrame({
            "num_feat": [10.0, np.nan, 30.0, 40.0],
            "cat_feat": ["apple", "banana", "apple", np.nan],
        })
        transformer = RobustTabularTransformer()
        transformer.fit(train_df)

        transformed_train = transformer.transform(train_df)
        self.assertIsInstance(transformed_train, np.ndarray)
        self.assertEqual(transformed_train.shape, (4, 2))
        # Check that NaN in num_feat was imputed with median (30.0)
        self.assertEqual(transformed_train[1, 0], 30.0)
        # Check that NaN in cat_feat was imputed with -1.0
        self.assertEqual(transformed_train[3, 1], -1.0)

        # Test inference with unseen categories and missing columns
        test_df = pd.DataFrame({
            "num_feat": [np.nan, 50.0],
            "cat_feat": ["cherry", "banana"],  # 'cherry' is unseen
        })
        transformed_test = transformer.transform(test_df)
        self.assertEqual(transformed_test.shape, (2, 2))
        # Unseen 'cherry' must be mapped to -1.0
        self.assertEqual(transformed_test[0, 1], -1.0)
        # Seen 'banana' must be mapped to valid non-negative float
        self.assertGreaterEqual(transformed_test[1, 1], 0.0)

    def test_random_forest_model_with_string_categories(self):
        """Verify RandomForestModel fits and predicts on string categories without ValueError."""
        train_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "brand": ["Toyota", "Ford", "Toyota", "BMW", "Ford", "BMW"],
            "mileage": [10000, 25000, np.nan, 5000, 18000, 30000],
            "price_cat": ["mid", "low", "mid", "high", "low", "high"],
        })
        spec_dict = {
            "task": {
                "name": "car_pricing",
                "task_type": "multiclass_classification",
                "primary_metric": "macro_f1",
                "id_column": "id",
                "inputs": [
                    {
                        "name": "tabular_feats",
                        "type": "tabular",
                        "source_columns": ["brand", "mileage"],
                    }
                ],
                "subtasks": {
                    "price_cat": {
                        "name": "price_cat",
                        "task_type": "multiclass_classification",
                        "classes": ["low", "mid", "high"],
                    }
                },
            }
        }
        spec = UniversalTaskSpec.from_dict(spec_dict)
        model = RandomForestModel(spec, n_estimators=10)
        model.fit(train_df, variant="tabular_feats")

        # Predict on unseen test data with new categories
        test_df = pd.DataFrame({
            "id": [7, 8],
            "brand": ["Audi", "Ford"],  # 'Audi' unseen
            "mileage": [12000, np.nan],
        })
        preds = model.predict(test_df, variant="tabular_feats")
        self.assertIn("price_cat", preds)
        self.assertIn("proba_price_cat", preds)
        self.assertEqual(len(preds["price_cat"]), 2)
        self.assertEqual(len(preds["proba_price_cat"]), 2)

    def test_resolve_media_paths(self):
        """Verify InputVariantSpec.resolve_media_paths finds media across directories and extensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img_dir = tmp_path / "images"
            img_dir.mkdir(parents=True)
            (img_dir / "sample_01.jpg").write_text("dummy image 1")
            (img_dir / "sample_02.png").write_text("dummy image 2")

            df = pd.DataFrame({
                "img_filename": ["sample_01.jpg", "sample_02"],
            })
            variant = InputVariantSpec(
                name="media_input",
                type="image",
                source_columns=["img_filename"],
            )

            resolved = variant.resolve_media_paths(
                df,
                project_root=tmp_path,
                column="img_filename",
                extensions=[".png", ".jpg"],
            )
            self.assertEqual(len(resolved), 2)
            self.assertTrue(Path(resolved.iloc[0]).exists())
            self.assertTrue(resolved.iloc[0].endswith("sample_01.jpg"))
            self.assertTrue(Path(resolved.iloc[1]).exists())
            self.assertTrue(resolved.iloc[1].endswith("sample_02.png"))

    def test_runner_time_series_strategy(self):
        """Verify Runner._resolve_validation_splits executes cleanly with time_series strategy."""
        import argparse

        import yaml

        from harness.runner import Runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_df = pd.DataFrame({"id": list(range(20)), "feature": list(range(20)), "target": [0, 1] * 10})
            train_df.to_csv(root / "train.csv", index=False)

            task_yaml = root / "task.yaml"
            task_dict = {
                "task": {
                    "name": "ts_task",
                    "task_type": "binary_classification",
                    "primary_metric": "macro_f1",
                    "id_column": "id",
                    "data": {"train_path": "train.csv"},
                    "inputs": [{"name": "feat", "type": "tabular", "source_columns": ["feature"]}],
                    "subtasks": {"target": {"task_type": "binary_classification", "classes": [0, 1]}},
                    "validation": {"strategy": "time_series", "num_folds": 3},
                }
            }
            task_yaml.write_text(yaml.safe_dump(task_dict), encoding="utf-8")

            exp_yaml = root / "exp.yaml"
            exp_dict = {
                "task": {"name": "ts_task"},
                "model": {"name": "random_forest"},
                "validation": {"strategy": "time_series", "num_folds": 3},
            }
            exp_yaml.write_text(yaml.safe_dump(exp_dict), encoding="utf-8")

            args = argparse.Namespace(
                project_root=str(root),
                task_config=str(task_yaml),
                debug=False,
                probe_samples=None,
                fold=None,
                job_id="test_ts",
                force_import=False,
            )
            runner = Runner(exp_dict, exp_yaml, args)
            splits = runner._resolve_validation_splits(train_df, "time_series")
            self.assertEqual(len(splits), 3)
            for fold, tr, val in splits:
                self.assertFalse(tr.empty)
                self.assertFalse(val.empty)
                self.assertEqual(len(set(tr["id"]).intersection(set(val["id"]))), 0)


if __name__ == "__main__":
    unittest.main()
