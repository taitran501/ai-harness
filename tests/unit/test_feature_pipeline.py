"""Consolidated unit test suite for feature transformers, text composing, and tabular pipelines.

Consolidates:
- tests/test_harness_universal.py (TestFeaturePipeline)
- tests/agent_bridge/test_usability_phase7.py (test_robust_tabular_transformer_mixed_types)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from harness.feature_pipeline import (
    SparseNgramFeaturizer,
    TextTemplateComposer,
)
from harness.models import RobustTabularTransformer


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

    def test_robust_tabular_transformer_mixed_types(self):
        train_df = pd.DataFrame({
            "num_feat": [10.0, np.nan, 30.0, 40.0],
            "cat_feat": ["apple", "banana", "apple", np.nan],
        })
        transformer = RobustTabularTransformer()
        transformer.fit(train_df)

        transformed_train = transformer.transform(train_df)
        self.assertIsInstance(transformed_train, np.ndarray)
        self.assertEqual(transformed_train.shape, (4, 2))
        self.assertEqual(transformed_train[1, 0], 30.0)
        self.assertEqual(transformed_train[3, 1], -1.0)

        test_df = pd.DataFrame({
            "num_feat": [np.nan, 50.0],
            "cat_feat": ["cherry", "banana"],
        })
        transformed_test = transformer.transform(test_df)
        self.assertEqual(transformed_test.shape, (2, 2))
        self.assertEqual(transformed_test[0, 1], -1.0)
        self.assertGreaterEqual(transformed_test[1, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
