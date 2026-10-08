"""Unit tests for Universal ML-Harness components:
- Task Spec & Conditional Engine
- Data Strategy & Leakage Auditor
- Feature Pipeline
- Tuning & Calibration Engine
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.data_strategy import (
    DataLeakageError,
    DataStrategyEngine,
    LeakageAuditor,
    SplitMode,
)
from harness.feature_pipeline import (
    SparseNgramFeaturizer,
    TabularMetadataExtractor,
    TextTemplateComposer,
    UniversalFeaturePipeline,
)
from harness.task_spec import (
    ConditionalRoutingRule,
    InputVariantSpec,
    SubtaskSpec,
    UniversalTaskSpec,
)
from harness.tuning_engine import ProbabilityCalibrator, ThresholdOptimizer


class TestUniversalTaskSpec(unittest.TestCase):
    def test_load_from_yaml(self):
        task_yaml = Path("configs/task.yaml")
        spec = UniversalTaskSpec.from_yaml(task_yaml)
        self.assertTrue(len(spec.name) > 0)
        self.assertIsNotNone(spec.primary_metric)

        # Check multi-task preset if present
        preset_yaml = Path("task_plugins/stereoqueereval/examples/task.yaml")
        if preset_yaml.exists():
            multi_spec = UniversalTaskSpec.from_yaml(preset_yaml)
            self.assertEqual(multi_spec.name, "StereoQueerEval")
            self.assertIn("subtask_a", multi_spec.subtasks)
            self.assertIn("subtask_b", multi_spec.subtasks)
            self.assertIn("subtask_c_type", multi_spec.subtasks)
            self.assertIn("subtask_c_identity", multi_spec.subtasks)

            cond = multi_spec.subtasks["subtask_c_type"].condition
            self.assertIsNotNone(cond)
            self.assertEqual(cond.parent_target, "subtask_b")
            self.assertEqual(cond.operator, "not_in")
            self.assertEqual(cond.fallback_value, "none")

    def test_conditional_routing(self):
        preset_yaml = Path("task_plugins/stereoqueereval/examples/task.yaml")
        if not preset_yaml.exists():
            return
        spec = UniversalTaskSpec.from_yaml(preset_yaml)

        raw_preds = {
            "subtask_b": np.array(["no", "yes_explicit", "no", "yes_implicit"]),
            "subtask_c_type": np.array(["group", "individual", "individual", "group"]),
            "subtask_c_identity": np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]),
        }

        routed = spec.apply_routing(raw_preds, n_samples=4)

        # Row 0 had subtask_b == "no" -> subtask_c_type should be "none", identities all 0
        self.assertEqual(routed["subtask_c_type"][0], "none")
        np.testing.assert_array_equal(routed["subtask_c_identity"][0], np.zeros(9))

        # Row 1 had subtask_b == "yes_explicit" -> preserves predictions
        self.assertEqual(routed["subtask_c_type"][1], "individual")
        self.assertEqual(routed["subtask_c_identity"][1, 1], 1)

        # Row 2 had subtask_b == "no" -> subtask_c_type should be "none"
        self.assertEqual(routed["subtask_c_type"][2], "none")
        np.testing.assert_array_equal(routed["subtask_c_identity"][2], np.zeros(9))

        # Row 3 had subtask_b == "yes_implicit" -> preserves predictions
        self.assertEqual(routed["subtask_c_type"][3], "group")
        self.assertEqual(routed["subtask_c_identity"][3, 3], 1)


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
            # Must pass without error
            report = LeakageAuditor.audit_splits(train_df, val_df, group_col="video_id")
            self.assertFalse(report["leakage_detected"])

    def test_leakage_auditor_raises_on_overlap(self):
        train_df = self.df.iloc[:6]
        val_df = self.df.iloc[5:10]  # row 5 (video_id 'v3') overlaps

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


class TestFeaturePipeline(unittest.TestCase):
    def test_text_composer(self):
        df = pd.DataFrame({
            "title": ["Hello World", "Test Title"],
            "comment": ["Nice video!", "Bad audio."],
        })
        composer = TextTemplateComposer(
            template="[T] {title} [C] {comment}", source_columns=["title", "comment"]
        )
        composed = composer.transform(df)
        self.assertEqual(composed[0], "[T] Hello World [C] Nice video!")
        self.assertEqual(composed[1], "[T] Test Title [C] Bad audio.")

    def test_sparse_featurizer(self):
        texts = pd.Series([
            "this is a clean comment about lgbt community",
            "another text comment for evaluation of models",
            "short comment",
            "yet another sample sentence with multiple words",
        ])
        featurizer = SparseNgramFeaturizer(min_df=1, word_max_features=50, char_max_features=50)
        X = featurizer.fit_transform(texts)
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)


class TestTuningEngine(unittest.TestCase):
    def test_binary_threshold_optimizer(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.4, 0.45, 0.6, 0.8, 0.9, 0.3, 0.25, 0.7])

        thresh, f1 = ThresholdOptimizer.optimize_binary_threshold(y_true, y_prob)
        self.assertGreater(thresh, 0.0)
        self.assertLess(thresh, 1.0)
        self.assertGreaterEqual(f1, 0.8)

    def test_probability_calibrator(self):
        y_prob = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.9])
        y_true = np.array([0, 0, 0, 1, 1, 1])

        calib = ProbabilityCalibrator(method="sigmoid")
        calib.fit(y_prob, y_true)
        calibrated = calib.transform(y_prob)

        self.assertEqual(calibrated.shape, (6, 1))
        # Monotonically increasing or well calibrated
        self.assertLess(calibrated[0, 0], calibrated[-1, 0])


if __name__ == "__main__":
    unittest.main()
