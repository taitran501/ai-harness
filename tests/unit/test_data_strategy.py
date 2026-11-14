"""Consolidated unit test suite for cross-validation splits, leakage audit, and data strategies.

Consolidates:
- tests/test_harness_universal.py (TestDataStrategyAndLeakageAuditor)
- tests/agent_bridge/test_usability_phase7.py (test_data_strategy_time_series_splits)
- tests/agent_bridge/test_usability_phase8.py (test_purged_group_time_series_split, test_multilabel_stratified_kfold)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from harness.data_strategy import (
    DataLeakageError,
    DataStrategyEngine,
    LeakageAuditor,
    SplitMode,
)


class TestDataStrategyAndLeakageAuditor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(12)],
            "video_id": ["v1", "v1", "v1", "v2", "v2", "v3", "v3", "v4", "v4", "v5", "v6", "v6"],
            "text": [f"Sample comment {i}" for i in range(12)],
            "label": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        })

    def test_group_kfold_no_leakage(self):
        splits = DataStrategyEngine.generate_splits(
            self.df, mode=SplitMode.GROUP_KFOLD, n_splits=3, group_col="video_id"
        )
        self.assertEqual(len(splits), 3)

        for train_idx, val_idx in splits:
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]
            report = LeakageAuditor.audit_splits(train_df, val_df, group_col="video_id")
            self.assertFalse(report["leakage_detected"])

    def test_leakage_auditor_raises_on_overlap(self):
        train_df = self.df.iloc[:6]
        val_df = self.df.iloc[5:10]

        with self.assertRaises(DataLeakageError):
            LeakageAuditor.audit_splits(train_df, val_df, group_col="video_id")

    def test_manifest_generation(self):
        splits = DataStrategyEngine.generate_splits(
            self.df, mode=SplitMode.GROUP_KFOLD, n_splits=3, group_col="video_id"
        )
        manifest = DataStrategyEngine.create_split_manifest(
            self.df, splits, id_col="id", group_col="video_id", target_cols=["label"]
        )
        self.assertEqual(manifest["total_samples"], 12)
        self.assertEqual(manifest["n_folds"], 3)
        self.assertIn("val_distributions", manifest["folds"][0])

    def test_data_strategy_time_series_splits(self):
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
                self.assertEqual(len(set(train_idx).intersection(set(val_idx))), 0)
                self.assertLess(max(train_idx), max(val_idx))

    def test_purged_group_time_series_split(self):
        df = pd.DataFrame({
            "id": range(100),
            "date_id": np.repeat(range(20), 5),
            "feature": np.random.randn(100),
        })
        splits = DataStrategyEngine.generate_splits(
            df,
            mode=SplitMode.PURGED_GROUP_TIME_SERIES,
            n_splits=3,
            group_col="date_id",
            holdout_ratio=0.1,
        )
        self.assertGreaterEqual(len(splits), 2)
        for tr_idx, val_idx in splits:
            tr_dates = df.loc[tr_idx, "date_id"]
            val_dates = df.loc[val_idx, "date_id"]
            self.assertLess(tr_dates.max(), val_dates.min())
            self.assertGreater(val_dates.min() - tr_dates.max(), 1)

    def test_multilabel_stratified_kfold(self):
        Y = np.zeros((100, 3), dtype=int)
        Y[:20, 0] = 1
        Y[15:35, 1] = 1
        Y[70:85, 2] = 1

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
            self.assertGreater(df.loc[val_idx, "label_a"].sum(), 0)
            self.assertGreater(df.loc[val_idx, "label_b"].sum(), 0)
            self.assertGreater(df.loc[val_idx, "label_c"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
