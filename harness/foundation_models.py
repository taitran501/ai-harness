"""Tabular Foundation Models and Knowledge Transfer Adapters.

Provides high-leverage adapters for non-tree tabular learning paradigms:
1. TeacherLogitPrior: Extracts cross-dataset Bayesian priors from external/original
   datasets via trained Teacher models, preventing generator drift while preserving
   strong conditional probability signals.
2. TabularModelAdapter: Standardized interface for deep tabular networks (RealMLP)
   and in-context foundation models (TabPFN, LimiX, TabICL).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TeacherLogitPrior:
    """Extracts teacher model probability and logit priors from auxiliary datasets.

    Solves the 'Negative Results Law' in competitive ML: Appending original data
    directly contaminates synthetic distributions and causes test degradation.
    Instead, this adapter trains on auxiliary data and extracts smooth conditional
    log-odds onto the target dataset.
    """

    def __init__(
        self,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        clip_bounds: Tuple[float, float] = (-15.0, 15.0),
    ):
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.model_type = model_type
        self.model_params = model_params or {}
        self.clip_bounds = clip_bounds
        self.teacher_model_: Any = None
        self.is_fitted_: bool = False

    def fit(self, auxiliary_df: pd.DataFrame) -> "TeacherLogitPrior":
        """Fits teacher model on auxiliary reference data."""
        if self.target_column not in auxiliary_df.columns:
            raise ValueError(f"Target column '{self.target_column}' missing from auxiliary DataFrame")

        y = auxiliary_df[self.target_column]
        if y.dtype == object or y.dtype.name == "category":
            # Map binary strings e.g. satisfied/neutral
            unique_vals = sorted(y.dropna().unique())
            val_map = {val: idx for idx, val in enumerate(unique_vals)}
            y_arr = y.map(val_map).values.astype(int)
        else:
            y_arr = y.values.astype(int)

        if self.feature_columns is None:
            cols = [c for c in auxiliary_df.columns if c != self.target_column]
        else:
            cols = [c for c in self.feature_columns if c in auxiliary_df.columns]

        X = auxiliary_df[cols].copy()

        # Handle categoricals
        cat_cols = [c for c in X.columns if X[c].dtype.kind not in "biuf"]
        for c in cat_cols:
            X[c] = X[c].astype("category")

        if self.model_type.lower() == "lightgbm":
            try:
                import lightgbm as lgb
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "learning_rate": 0.03,
                    "num_leaves": 63,
                    "max_depth": 8,
                    "min_child_samples": 20,
                    "verbose": -1,
                    "random_state": 42,
                    **self.model_params,
                }
                dtrain = lgb.Dataset(X, label=y_arr)
                self.teacher_model_ = lgb.train(params, dtrain, num_boost_round=params.get("n_estimators", 300))
            except ImportError:
                self.teacher_model_ = self._fit_fallback_pipeline(X, y_arr, cat_cols)
        else:
            self.teacher_model_ = self._fit_fallback_pipeline(X, y_arr, cat_cols)

        self.feature_columns = cols
        self.is_fitted_ = True
        return self

    def _fit_fallback_pipeline(self, X: pd.DataFrame, y: np.ndarray, cat_cols: List[str]) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(X, y)
        return pipe

    def transform(self, df: pd.DataFrame, prefix: str = "teacher") -> pd.DataFrame:
        """Generates prior probabilities and logits as new meta-features."""
        if not self.is_fitted_ or self.teacher_model_ is None:
            raise RuntimeError("Teacher model must be fitted before transform")

        cols = [c for c in self.feature_columns if c in df.columns]
        X = df[cols].copy()
        cat_cols = [c for c in X.columns if X[c].dtype.kind not in "biuf"]
        for c in cat_cols:
            X[c] = X[c].astype("category")

        if hasattr(self.teacher_model_, "predict_proba"):
            preds = self.teacher_model_.predict_proba(X)[:, 1]
        elif hasattr(self.teacher_model_, "predict"):
            preds = self.teacher_model_.predict(X)
        else:
            raise RuntimeError("Teacher model has neither predict_proba nor predict")

        p = np.clip(np.asarray(preds, dtype=np.float64), 1e-15, 1.0 - 1e-15)
        logits = np.clip(np.log(p / (1.0 - p)), self.clip_bounds[0], self.clip_bounds[1])

        out = pd.DataFrame(index=df.index)
        out[f"{prefix}_prob"] = p
        out[f"{prefix}_logit"] = logits
        return out


class HierarchicalDiscretizer:
    """Decomposes continuous mileage/corridor metrics into multi-scale discrete keys.

    Generates macro-regional bands, airport corridor clusters, and route keys.
    """

    @staticmethod
    def decompose_distance(
        distance_series: pd.Series,
        col_name: str = "flight_distance",
    ) -> pd.DataFrame:
        """Decomposes a continuous distance into exact, corridor, and macro-band keys."""
        s = distance_series.fillna(-1).astype(int)
        df = pd.DataFrame(index=distance_series.index)
        df[f"{col_name}_exact"] = s
        df[f"{col_name}_corridor_10"] = s // 10
        df[f"{col_name}_macro_100"] = s // 100
        df[f"{col_name}_band_1000"] = s // 1000
        return df
