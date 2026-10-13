"""Universal Tuning, Calibration, and Threshold Optimization Engine for AI-Harness.

Implements per-class and per-group out-of-fold threshold search, Platt/Isotonic
probability calibration, and cascaded error control for hierarchical tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class ThresholdOptimizer:
    """Finds optimal decision thresholds on out-of-fold predictions to maximize F1."""

    @staticmethod
    def optimize_binary_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric_average: str = "binary",
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        step: float = 0.02,
    ) -> Tuple[float, float]:
        """Search threshold maximizing F1 on true vs predicted probability."""
        best_threshold = 0.5
        best_f1 = -1.0

        for thresh in np.arange(min_threshold, max_threshold, step):
            preds = (y_prob >= thresh).astype(int)
            score = f1_score(y_true, preds, average=metric_average, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(thresh)

        return best_threshold, best_f1

    @staticmethod
    def optimize_multilabel_thresholds(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        labels: List[str],
        default_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Optimize threshold independently per label for multi-label classification."""
        best_thresholds: Dict[str, float] = {}

        n_labels = y_true.shape[1]
        for idx in range(n_labels):
            label_name = labels[idx] if idx < len(labels) else f"label_{idx}"
            col_true = y_true[:, idx]
            col_prob = y_probs[:, idx]

            # If label has zero support or almost zero support in OOF
            if col_true.sum() < 2:
                best_thresholds[label_name] = default_threshold
                continue

            thresh, _ = ThresholdOptimizer.optimize_binary_threshold(col_true, col_prob)
            best_thresholds[label_name] = thresh

        return best_thresholds

    @staticmethod
    def optimize_group_thresholds(
        df: pd.DataFrame,
        group_col: str,
        true_col: str,
        prob_col: str,
    ) -> Dict[Any, float]:
        """Optimize thresholds separately for each declared group or cohort."""
        group_thresholds = {}
        for group_val, group_sub in df.groupby(group_col):
            y_t = group_sub[true_col].to_numpy()
            y_p = group_sub[prob_col].to_numpy()
            t_star, _ = ThresholdOptimizer.optimize_binary_threshold(y_t, y_p)
            group_thresholds[group_val] = t_star
        return group_thresholds


class ProbabilityCalibrator:
    """Calibrates predicted probabilities using Platt scaling or Isotonic Regression."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrators: Dict[int, Any] = {}

    def fit(self, y_probs: np.ndarray, y_true: np.ndarray):
        """Fit calibration mapping on 1D or 2D probabilities."""
        if y_probs.ndim == 1:
            y_probs = y_probs.reshape(-1, 1)

        n_classes = y_probs.shape[1]
        for c in range(n_classes):
            probs_c = y_probs[:, c]
            if n_classes == 1:
                true_c = (y_true == 1).astype(int) if y_true.ndim <= 1 else y_true[:, 0]
            else:
                true_c = y_true[:, c] if y_true.ndim > 1 else (y_true == c).astype(int)

            if self.method == "sigmoid":
                calib = LogisticRegression(C=1.0, solver="lbfgs")
                calib.fit(probs_c.reshape(-1, 1), true_c)
            elif self.method == "isotonic":
                calib = IsotonicRegression(out_of_bounds="clip")
                calib.fit(probs_c, true_c)
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")

            self.calibrators[c] = calib
        return self

    def transform(self, y_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if y_probs.ndim == 1:
            y_probs = y_probs.reshape(-1, 1)

        calibrated = np.zeros_like(y_probs)
        for c, calib in self.calibrators.items():
            probs_c = y_probs[:, c]
            if self.method == "sigmoid":
                calibrated[:, c] = calib.predict_proba(probs_c.reshape(-1, 1))[:, 1]
            else:
                calibrated[:, c] = calib.predict(probs_c)

        return calibrated
