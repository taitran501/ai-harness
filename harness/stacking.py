"""Competitive ML Layer-2 Stacking and Multi-Paradigm Ensemble Module.

Provides robust, leakage-safe ensembling algorithms designed for high-tier
competition performance:
1. L2LogitStacker: Regularized shrinkage stacking on clipped logits (C=0.003)
   to eliminate multicollinearity and prevent validation-to-test degradation.
2. RankAverager: Monotonic, scale-invariant rank percentile averaging
   optimal for rank-based metrics such as ROC-AUC and Gini.
3. PredictionAligner: Guarantees strict natural index and ID row alignment
   between Out-of-Fold (OOF) arrays and ground truth tables.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold


def to_clipped_logit(
    probs: Union[np.ndarray, pd.Series, List[float]],
    clip_epsilon: float = 1e-15,
    logit_bounds: Tuple[float, float] = (-15.0, 15.0),
) -> np.ndarray:
    """Transforms probabilities into bounded numerical logits.

    Args:
        probs: Array-like probabilities in [0, 1].
        clip_epsilon: Boundary buffer to prevent log(0) or division by zero.
        logit_bounds: Tuple (min_val, max_val) to bound the logit space.

    Returns:
        1D float64 array of clipped log-odds.
    """
    arr = np.asarray(probs, dtype=np.float64)
    # If already in logit space (negative values or values > 1), clip directly
    if arr.min() < 0.0 or arr.max() > 1.0:
        return np.clip(arr, logit_bounds[0], logit_bounds[1])
    p_clip = np.clip(arr, clip_epsilon, 1.0 - clip_epsilon)
    logits = np.log(p_clip / (1.0 - p_clip))
    return np.clip(logits, logit_bounds[0], logit_bounds[1])


def align_prediction_frames(
    prediction_frames: List[pd.DataFrame],
    reference_df: pd.DataFrame,
    id_column: str = "id",
    pred_column: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Aligns multiple prediction DataFrames to the exact ID order of a reference DataFrame.

    Prevents catastrophic indexing bugs where fold-concatenated OOF predictions
    are mismatched against natural training row orders.

    Args:
        prediction_frames: List of candidate prediction DataFrames.
        reference_df: Reference DataFrame containing the target 'id' sequence.
        id_column: Column name for unique identifier.
        pred_column: Optional target prediction column name. If None, auto-detected.

    Returns:
        Tuple of (2D array of aligned predictions [n_samples, n_models], list of model names).
    """
    if id_column not in reference_df.columns:
        raise ValueError(f"id_column '{id_column}' missing from reference DataFrame")

    target_ids = reference_df[id_column].values
    n_samples = len(target_ids)
    aligned_matrix = []
    model_names = []

    for idx, df in enumerate(prediction_frames):
        if id_column not in df.columns:
            raise ValueError(f"id_column '{id_column}' missing in prediction DataFrame at index {idx}")

        # Auto-detect prediction column if not specified
        col = pred_column
        if col is None or col not in df.columns:
            candidates = [
                c for c in df.columns
                if c not in (id_column, "fold", "target", "label", "satisfaction_true")
            ]
            if not candidates:
                raise ValueError(f"No valid prediction column found in frame index {idx}")
            col = candidates[0]

        # Re-index strictly to target ID sequence
        indexed = df.set_index(id_column)
        if not set(target_ids).issubset(indexed.index):
            missing_count = len(set(target_ids) - set(indexed.index))
            raise ValueError(f"Prediction frame index {idx} missing {missing_count} IDs from reference")

        reindexed_series = indexed.loc[target_ids, col]

        # Parse string representations of array if present (e.g. "[0.12, 0.88]")
        first_val = reindexed_series.iloc[0]
        if isinstance(first_val, str) and first_val.startswith("["):
            val_arr = reindexed_series.apply(
                lambda s: float(s.strip("[]").split(",")[1]) if "," in s else float(s.strip("[]"))
            ).values.astype(np.float64)
        else:
            val_arr = reindexed_series.values.astype(np.float64)

        aligned_matrix.append(val_arr)
        name = getattr(df, "name", f"model_{idx:02d}_{col}")
        model_names.append(name)

    return np.column_stack(aligned_matrix), model_names


class L2LogitStacker:
    """Stratified Nested Meta-Stacker with L2 Regularization on Clipped Logits.

    Solves extreme multicollinearity across 10-100 correlated model predictions
    by optimizing an L2 shrinkage penalty (default C=0.003) on bounded logit features.
    """

    def __init__(
        self,
        C: float = 0.003,
        n_splits: int = 5,
        random_state: int = 42,
        clip_bounds: Tuple[float, float] = (-15.0, 15.0),
        task_type: str = "classification",
    ):
        self.C = C
        self.n_splits = n_splits
        self.random_state = random_state
        self.clip_bounds = clip_bounds
        self.task_type = task_type
        self.meta_models_: List[Any] = []
        self.feature_names_: List[str] = []
        self.oof_predictions_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None

    def fit_predict(
        self,
        X_oof: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fits meta-stacker across nested folds and predicts OOF and test probabilities.

        Args:
            X_oof: 2D array of out-of-fold member predictions (probabilities or logits).
            y: 1D ground truth binary array.
            X_test: Optional 2D array of member test predictions.
            feature_names: Optional names of member models.

        Returns:
            Tuple of (oof_stacked_predictions, test_stacked_predictions).
        """
        X_oof = np.asarray(X_oof, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_models = X_oof.shape
        self.feature_names_ = feature_names or [f"model_{i}" for i in range(n_models)]

        # Transform inputs into logit representation
        Z_oof = np.column_stack([
            to_clipped_logit(X_oof[:, i], logit_bounds=self.clip_bounds)
            for i in range(n_models)
        ])

        if X_test is not None:
            X_test = np.asarray(X_test, dtype=np.float64)
            Z_test = np.column_stack([
                to_clipped_logit(X_test[:, i], logit_bounds=self.clip_bounds)
                for i in range(n_models)
            ])
            n_test = len(X_test)
            test_accum = np.zeros(n_test, dtype=np.float64)
        else:
            Z_test = None
            test_accum = None

        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_stacked = np.zeros(n_samples, dtype=np.float64)
        self.meta_models_ = []
        coefs = []

        for fold, (idx_tr, idx_va) in enumerate(cv.split(Z_oof, y)):
            if self.task_type == "classification":
                clf = LogisticRegression(
                    C=self.C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self.random_state + fold,
                )
                clf.fit(Z_oof[idx_tr], y[idx_tr])
                oof_stacked[idx_va] = clf.predict_proba(Z_oof[idx_va])[:, 1]
                coefs.append(clf.coef_[0])
                if Z_test is not None:
                    test_accum += clf.predict_proba(Z_test)[:, 1] / float(self.n_splits)
            else:
                reg = Ridge(alpha=1.0 / (2.0 * self.C), random_state=self.random_state + fold)
                reg.fit(Z_oof[idx_tr], y[idx_tr])
                oof_stacked[idx_va] = reg.predict(Z_oof[idx_va])
                coefs.append(reg.coef_)
                if Z_test is not None:
                    test_accum += reg.predict(Z_test) / float(self.n_splits)

            self.meta_models_.append(clf if self.task_type == "classification" else reg)

        self.oof_predictions_ = oof_stacked
        self.coefficients_ = np.mean(coefs, axis=0)
        return oof_stacked, test_accum

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns sorted feature importance table by absolute meta-weight."""
        if self.coefficients_ is None:
            raise RuntimeError("Model has not been fitted yet")
        df = pd.DataFrame({
            "model": self.feature_names_,
            "weight": self.coefficients_,
            "abs_weight": np.abs(self.coefficients_),
        })
        return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


class RankAverager:
    """Scale-Invariant Monotonic Rank Averager for Metric Area Optimization.

    Averages predictions across different calibration scales using empirical
    percentile ranking:
        r_i = rank(p_i) / N
        final = sum(w_m * r_i_m)
    """

    @staticmethod
    def rank_percentiles(arr: np.ndarray) -> np.ndarray:
        """Converts an array into uniformly distributed percentile ranks in (0, 1]."""
        arr = np.asarray(arr, dtype=np.float64)
        series = pd.Series(arr)
        return series.rank(pct=True).values

    @classmethod
    def blend(
        cls,
        prediction_arrays: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Computes weighted rank average across multiple prediction arrays.

        Args:
            prediction_arrays: List of 1D prediction arrays of equal length.
            weights: Optional list of non-negative weights summing to 1.0.

        Returns:
            1D array of blended percentile ranks.
        """
        n_members = len(prediction_arrays)
        if n_members == 0:
            raise ValueError("Empty list of prediction arrays")

        length = len(prediction_arrays[0])
        for idx, a in enumerate(prediction_arrays):
            if len(a) != length:
                raise ValueError(f"Array length mismatch at index {idx}: expected {length}, got {len(a)}")

        if weights is None:
            w = np.ones(n_members, dtype=np.float64) / float(n_members)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != n_members:
                raise ValueError(f"Weights length {len(w)} does not match models {n_members}")
            w = w / np.sum(w)

        blended = np.zeros(length, dtype=np.float64)
        for i in range(n_members):
            ranks = cls.rank_percentiles(prediction_arrays[i])
            blended += w[i] * ranks

        return blended
