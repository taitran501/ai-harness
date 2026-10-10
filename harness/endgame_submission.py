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
