"""Universal Feature Pipeline for ML-Harness.

Provides modular text composition, sparse n-gram extraction (word + char),
dense transformer embeddings, and tabular metadata feature engineering.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


class TextTemplateComposer(BaseEstimator, TransformerMixin):
    """Composes structured text inputs from multiple columns using templates."""

    def __init__(self, template: str, source_columns: List[str]):
        self.template = template
        self.source_columns = source_columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> List[str]:
        output = []
        for _, row in X[self.source_columns].iterrows():
            clean_dict = {col: str(row.get(col, "") or "").strip() for col in self.source_columns}
            output.append(self.template.format(**clean_dict))
        return output


class SparseNgramFeaturizer(BaseEstimator, TransformerMixin):
    """Word and character n-gram TF-IDF featurizer."""

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 40_000,
        char_ngram_range: Tuple[int, int] = (3, 5),
        char_max_features: int = 60_000,
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self.word_ngram_range = word_ngram_range
        self.word_max_features = word_max_features
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        self.union = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=self.word_ngram_range,
                        min_df=self.min_df,
                        max_features=self.word_max_features,
                        sublinear_tf=self.sublinear_tf,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=self.char_ngram_range,
                        min_df=self.min_df,
                        max_features=self.char_max_features,
                        sublinear_tf=self.sublinear_tf,
                    ),
                ),
            ]
        )

    def fit(self, texts: List[str] | pd.Series, y=None):
        self.union.fit(texts)
        return self

    def transform(self, texts: List[str] | pd.Series):
        return self.union.transform(texts)


class TabularMetadataExtractor(BaseEstimator, TransformerMixin):
    """Extracts lexical and metadata statistical features from text and columns."""

    def __init__(self, text_col: str, include_language: bool = True):
        self.text_col = text_col
        self.include_language = include_language
        self.scaler = StandardScaler()

    def _extract_stats(self, texts: pd.Series) -> np.ndarray:
        stats = []
        for t in texts.fillna(""):
            t_str = str(t)
            char_len = len(t_str)
            words = t_str.split()
            word_count = len(words)
            upper_count = sum(1 for c in t_str if c.isupper())
            punct_count = len(re.findall(r"[^\w\s]", t_str))

            stats.append(
                [
                    char_len,
                    word_count,
                    upper_count / (char_len + 1e-5),
                    punct_count / (word_count + 1e-5),
                ]
            )
        return np.array(stats, dtype=float)

    def fit(self, X: pd.DataFrame, y=None):
        features = self._extract_stats(X[self.text_col])
        self.scaler.fit(features)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        features = self._extract_stats(X[self.text_col])
        return self.scaler.transform(features)


class UniversalFeaturePipeline:
    """Combines text featurizers, sparse representations, and tabular features."""

    def __init__(
        self,
        use_sparse_ngrams: bool = True,
        use_tabular_metadata: bool = False,
        sparse_params: Optional[Dict[str, Any]] = None,
    ):
        self.use_sparse_ngrams = use_sparse_ngrams
        self.use_tabular_metadata = use_tabular_metadata
        self.sparse_params = sparse_params or {}

        self.sparse_featurizer = (
            SparseNgramFeaturizer(**self.sparse_params) if use_sparse_ngrams else None
        )
        self.tabular_extractor = None

    def fit(self, text_series: pd.Series, df: Optional[pd.DataFrame] = None) -> UniversalFeaturePipeline:
        if self.sparse_featurizer:
            self.sparse_featurizer.fit(text_series)
        return self

    def transform(self, text_series: pd.Series, df: Optional[pd.DataFrame] = None):
        components = []
        if self.sparse_featurizer:
            components.append(self.sparse_featurizer.transform(text_series))

        if len(components) == 1:
            return components[0]
        elif len(components) > 1:
            return hstack(components).tocsr()
        raise RuntimeError("No feature components were activated in UniversalFeaturePipeline.")
