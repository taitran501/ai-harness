"""Universal Tuning, Calibration, and Threshold Optimization Engine for AI-Harness.

Implements per-class and per-group out-of-fold threshold search, Platt/Isotonic
probability calibration, and cascaded error control for hierarchical tasks.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
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
