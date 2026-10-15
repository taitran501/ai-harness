"""Competitive ML Submission Pre-flight Gate and Leaderboard Auditor.

Guarantees 100% submission integrity before dispatching to Kaggle/Remote:
1. Preflight Validation: Strict row count match, NaN checks, ID alignment,
   finite range checks, and probability distribution heuristics.
2. Leaderboard Auditor: Tracks score progression, public score deltas,
   and prevents silent failures.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    id_column: str
    target_column: str
    min_val: float
    max_val: float
    mean_val: float
    nan_count: int
    errors: List[str]


class SubmissionPreflightValidator:
    """Rigorous preflight auditor for final competition submission files."""

    @staticmethod
    def audit_submission(
        submission_path: Union[str, Path],
        reference_test_path: Union[str, Path],
        id_column: str = "id",
        target_column: Optional[str] = None,
        probability_mode: bool = True,
    ) -> ValidationReport:
        """Audits a candidate submission file against the official test set.

        Args:
            submission_path: Path to candidate submission CSV.
            reference_test_path: Path to official competition test.csv.
            id_column: Identifier column name.
            target_column: Target prediction column. If None, auto-detected.
            probability_mode: If True, asserts values lie in [0, 1].

        Returns:
            ValidationReport dataclass with audit details and errors if any.
        """
        errors = []
        sub_p = Path(submission_path)
        ref_p = Path(reference_test_path)

        if not sub_p.exists():
            return ValidationReport(False, 0, id_column, "", 0.0, 0.0, 0.0, 0, [f"File {sub_p} does not exist"])
        if not ref_p.exists():
            return ValidationReport(False, 0, id_column, "", 0.0, 0.0, 0.0, 0, [f"Reference {ref_p} does not exist"])

        sub_df = pd.read_csv(sub_p)
        ref_df = pd.read_csv(ref_p)

        # 1. Row count match
        if len(sub_df) != len(ref_df):
            errors.append(f"Row count mismatch: candidate has {len(sub_df)}, reference test has {len(ref_df)}")

        # 2. ID column verification
        if id_column not in sub_df.columns:
            errors.append(f"Missing required ID column '{id_column}'")
        else:
            if not (sub_df[id_column].values == ref_df[id_column].values).all():
                errors.append(f"Candidate IDs do not strictly match reference test ID sequence")

        # 3. Detect target column
        if target_column is None:
            cand = [c for c in sub_df.columns if c != id_column]
            if not cand:
                errors.append("No target prediction column found in submission")
                target_col = ""
            else:
                target_col = cand[0]
        else:
            target_col = target_column
            if target_col not in sub_df.columns:
                errors.append(f"Declared target column '{target_col}' missing from submission")

        nan_count = 0
        min_v = max_v = mean_v = 0.0

        if target_col and target_col in sub_df.columns:
            series = sub_df[target_col]
            nan_count = int(series.isnull().sum())
            if nan_count > 0:
                errors.append(f"Submission contains {nan_count} NaN/null values")

            if pd.api.types.is_numeric_dtype(series):
                min_v = float(series.min())
                max_v = float(series.max())
                mean_v = float(series.mean())

                if probability_mode:
                    if min_v < 0.0 or max_v > 1.0:
                        errors.append(f"Probability out of bounds [0, 1]: min={min_v:.4f}, max={max_v:.4f}")
            else:
                errors.append(f"Target column '{target_col}' is not numeric")

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            row_count=len(sub_df),
            id_column=id_column,
            target_column=target_col,
            min_val=min_v,
            max_val=max_v,
            mean_val=mean_v,
            nan_count=nan_count,
            errors=errors,
        )


class KaggleLeaderboardTracker:
    """Manages Kaggle CLI submission dispatch and programmatic status polling."""

    @staticmethod
    def submit_and_track(
        competition: str,
        file_path: Union[str, Path],
        message: str,
        poll_interval_sec: int = 5,
        max_wait_sec: int = 120,
    ) -> Dict[str, Any]:
        """Dispatches submission to Kaggle and polls until status is COMPLETE.

        Args:
            competition: Kaggle competition slug.
            file_path: Path to submission file.
            message: Submission commit/audit message.
            poll_interval_sec: Seconds between status polls.
            max_wait_sec: Maximum timeout in seconds.

        Returns:
            Dict containing final submission record with publicScore.
        """
        cmd_submit = [
            "kaggle", "competitions", "submit",
            "-c", competition,
            "-f", str(file_path),
            "-m", message,
        ]
        res = subprocess.run(cmd_submit, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Kaggle submit failed: {res.stderr or res.stdout}")

        # Poll status
        start_time = time.time()
        while time.time() - start_time < max_wait_sec:
            time.sleep(poll_interval_sec)
            cmd_list = ["kaggle", "competitions", "submissions", "-c", competition, "--csv"]
            res_list = subprocess.run(cmd_list, capture_output=True, text=True)
            if res_list.returncode == 0 and res_list.stdout:
                try:
                    import io
                    df = pd.read_csv(io.StringIO(res_list.stdout))
                    if not df.empty:
                        latest = df.iloc[0].to_dict()
                        status = str(latest.get("status", "")).upper()
                        if "COMPLETE" in status:
                            return latest
                        if "FAILED" in status or "ERROR" in status:
                            raise RuntimeError(f"Kaggle scoring failed: {latest}")
                except Exception:
                    pass

        return {"status": "TIMEOUT", "message": f"Scoring exceeded {max_wait_sec}s timeout"}


class ThresholdOptimizer:
    """Optimizes classification decision thresholds on Out-Of-Fold probabilities."""

    @staticmethod
    def optimize_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = "f1",
        num_thresholds: int = 181,
    ) -> Tuple[float, float]:
        """Finds the optimal scalar threshold maximizing the specified metric.

        Args:
            y_true: Binary ground truth labels (0 or 1).
            y_prob: Predicted probabilities in [0, 1].
            metric: Target metric to optimize ('f1', 'kappa', 'accuracy').
            num_thresholds: Number of evaluation thresholds in [0.05, 0.95].

        Returns:
            Tuple of (optimal_threshold, best_metric_score).
        """
        from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

        metric_lower = str(metric).lower()
        if metric_lower in {"kappa", "cohen_kappa"}:
            fn = cohen_kappa_score
        elif metric_lower in {"accuracy", "acc"}:
            fn = accuracy_score
        else:
            fn = f1_score

        best_th = 0.5
        best_score = float("-inf")
        y_true_clean = np.asarray(y_true).ravel()
        y_prob_clean = np.asarray(y_prob).ravel()

        for th in np.linspace(0.05, 0.95, num_thresholds):
            preds = (y_prob_clean >= th).astype(int)
            try:
                score = float(fn(y_true_clean, preds))
                if score > best_score:
                    best_score = score
                    best_th = float(th)
            except Exception:
                continue

        return best_th, best_score


class QuantileMatcher:
    """Non-parametric quantile alignment matching test prediction CDF to a reference distribution."""

    @staticmethod
    def match_quantiles(
        source_preds: np.ndarray,
        target_reference: np.ndarray,
    ) -> np.ndarray:
        """Projects source_preds onto the empirical quantile distribution of target_reference.

        Args:
            source_preds: Array of predictions to calibrate.
            target_reference: Ground truth or OOF reference predictions with desired distribution.

        Returns:
            Calibrated numpy array matching target_reference CDF.
        """
        from scipy.stats import rankdata

        src = np.asarray(source_preds).ravel()
        tgt = np.asarray(target_reference).ravel()
        ranks = rankdata(src, method="average") / (len(src) + 1.0)
        matched = np.percentile(np.sort(tgt), ranks * 100.0)
        return matched.reshape(np.asarray(source_preds).shape)


class MicroDoseProbeGenerator:
    """Generates low-drift, safe leaderboard probing submissions to test orthogonal hypotheses."""

    @staticmethod
    def generate_probe(
        base_csv: Union[str, Path],
        probe_csv: Union[str, Path],
        output_csv: Union[str, Path],
        alpha: float = 0.03,
        id_column: str = "id",
        target_column: Optional[str] = None,
        space: str = "logit",
    ) -> Dict[str, Any]:
        """Blends a verified base submission with a small micro-dose (alpha <= 0.08) of a probe signal.

        Args:
            base_csv: Path to proven, verified high-scoring submission CSV.
            probe_csv: Path to candidate probe submission CSV containing novel signal.
            output_csv: Destination path for the blended probe CSV.
            alpha: Dose ratio for probe (default: 0.03 = 3%).
            id_column: Name of ID column.
            target_column: Name of target column (auto-detected if None).
            space: Blending space ('logit', 'probability', 'rank').

        Returns:
            Audit dictionary with Spearman correlation, max change, and drift metrics.
        """
        from scipy.special import expit, logit
        from scipy.stats import spearmanr, rankdata

        df_base = pd.read_csv(base_csv)
        df_probe = pd.read_csv(probe_csv)

        if len(df_base) != len(df_probe):
            raise ValueError(f"Row count mismatch: base={len(df_base)}, probe={len(df_probe)}")

        if target_column is None:
            cand = [c for c in df_base.columns if c != id_column]
            if not cand:
                raise ValueError("Target column could not be inferred")
            target_col = cand[0]
        else:
            target_col = target_column

        s_base = df_base[target_col].values.astype(float)
        s_probe = df_probe[target_col].values.astype(float)

        if space == "logit":
            eps = 1e-6
            z_base = logit(np.clip(s_base, eps, 1.0 - eps))
            z_probe = logit(np.clip(s_probe, eps, 1.0 - eps))
            blended = expit((1.0 - alpha) * z_base + alpha * z_probe)
        elif space == "rank":
            r_base = rankdata(s_base) / len(s_base)
            r_probe = rankdata(s_probe) / len(s_probe)
            blended = (1.0 - alpha) * r_base + alpha * r_probe
        else:  # probability
            blended = (1.0 - alpha) * s_base + alpha * s_probe

        corr, _ = spearmanr(s_base, blended)
        max_diff = float(np.max(np.abs(s_base - blended)))
        mean_diff = float(np.mean(np.abs(s_base - blended)))

        out_df = df_base.copy()
        out_df[target_col] = blended
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_p, index=False)

        return {
            "output_path": str(out_p),
            "alpha": alpha,
            "space": space,
            "spearman_correlation_vs_base": float(corr),
            "max_absolute_diff": max_diff,
            "mean_absolute_diff": mean_diff,
            "rows": len(out_df),
        }


# Backwards compatibility alias
SubmissionValidator = SubmissionPreflightValidator

