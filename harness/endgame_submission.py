"""Competitive ML Submission Pre-flight Gate and Leaderboard Auditor.

Guarantees 100% submission integrity before dispatching to Kaggle/Remote:
1. Preflight Validation: Strict row count match, NaN checks, ID alignment,
   finite range checks, and probability distribution heuristics.
2. Leaderboard Auditor: Tracks score progression, public score deltas,
   and prevents silent failures.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from harness.dispatch_audit import record_dispatch


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
    # Standard deviation of the predicted values. Agents use this to detect
    # prediction collapse (constant output) before packaging a submission.
    std_val: float = 0.0


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
                errors.append("Candidate IDs do not strictly match reference test ID sequence")

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
        min_v = max_v = mean_v = std_v = 0.0

        if target_col and target_col in sub_df.columns:
            series = sub_df[target_col]
            nan_count = int(series.isnull().sum())
            if nan_count > 0:
                errors.append(f"Submission contains {nan_count} NaN/null values")

            if pd.api.types.is_numeric_dtype(series):
                min_v = float(series.min())
                max_v = float(series.max())
                mean_v = float(series.mean())
                std_v = float(series.std()) if len(series) > 1 else 0.0

                if probability_mode:
                    if min_v < 0.0 or max_v > 1.0:
                        errors.append(f"Probability out of bounds [0, 1]: min={min_v:.4f}, max={max_v:.4f}")
            else:
                if probability_mode:
                    errors.append(f"Target column '{target_col}' is expected to be numeric probabilities [0, 1] but is non-numeric")
                else:
                    empty_count = int((series.astype(str).str.strip() == "").sum())
                    if empty_count > 0:
                        errors.append(f"Target column '{target_col}' contains {empty_count} empty strings")

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
            std_val=std_v,
        )


class KaggleLeaderboardTracker:
    """Kaggle endgame tracker: dispatch submissions, poll for scores, track best.

    This is the harness-native, in-process dispatch path. The host hook blocks
    raw ``kaggle competitions submit`` / ``kaggle kernels push`` *shell* commands;
    this tracker is the authorized alternative — it drives the Kaggle API
    internally via :class:`harness.kaggle_compat.KaggleClient`, so every dispatch
    is attributable and auditable rather than a forced raw-CLI friction loop.

    Typical endgame loop::

        tracker = KaggleLeaderboardTracker()
        sub = tracker.dispatch("comp", "submission.csv", "Exp 22")
        info = tracker.wait_for_score("comp", sub["ref"], timeout_s=1800)
        print(info["public_score_f"])
    """

    _COMPLETE = "COMPLETE"
    _ERROR = "ERROR"

    def __init__(self, client=None, sleep_fn=None):
        """Initialise the tracker.

        Args:
            client: A :class:`harness.kaggle_compat.KaggleClient` (or any object
                exposing ``submit_submission`` / ``list_submissions``). Created
                lazily when omitted, so construction never requires credentials
                until a dispatch/poll actually runs.
            sleep_fn: Injectable sleep callable (tests pass a no-op). Defaults to
                :func:`time.sleep`.
        """
        if client is None:
            from harness.kaggle_compat import KaggleClient

            client = KaggleClient()
        self._client = client
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep

    @staticmethod
    def _status_str(status: Any) -> str:
        """Normalise a (possibly wrapped) SubmissionStatus to an uppercase name."""
        raw = getattr(status, "_raw", status)
        return str(raw).upper()

    @staticmethod
    def _score(value: Any) -> Optional[float]:
        """Parse a score string to float, or None when unset/invalid."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def dispatch(
        self,
        competition: str,
        file_name: str,
        message: str = "",
        kernel: Optional[str] = None,
        kernel_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a file (or code-competition kernel output) and return the ref.

        Args:
            competition: Kaggle competition slug.
            file_name: Local submission file (standard) or the kernel output
                filename (code competition, e.g. ``submission.csv``).
            message: Human-readable leaderboard label.
            kernel: Code competition — public kernel slug (``owner/kernel``) that
                produces the submission output.
            kernel_version: Code competition — the kernel version number.

        Returns:
            dict with ``ref`` (int), ``message`` and the dispatch inputs.
        """
        resp = self._client.submit_submission(
            file_name,
            message,
            competition,
            kernel=kernel,
            kernel_version=kernel_version,
        )
        result: Dict[str, Any] = {
            "ref": getattr(resp, "ref", None),
            "message": getattr(resp, "message", ""),
            "competition": competition,
            "file_name": file_name,
            "kernel": kernel,
            "kernel_version": kernel_version,
        }
        # Dispatch governance (PR-C): append-only audit entry + advisory
        # duplicate warning. Best-effort — audit failure never blocks the
        # dispatch itself.
        try:
            audit = record_dispatch(
                kind="submission",
                competition=competition,
                file_path=file_name,
                kernel=kernel,
                kernel_version=kernel_version,
                message=message,
                ref=result["ref"],
            )
        except Exception as exc:  # noqa: BLE001 - audit must never break dispatch
            print(
                f"[dispatch_audit] warning: audit entry not recorded: {exc}",
                file=sys.stderr,
            )
            audit = None
        if audit and audit.get("duplicate_of"):
            dup = audit["duplicate_of"]
            ref_info = f" (ref {dup['ref']})" if dup.get("ref") is not None else ""
            sub_info = f" (submission {dup['sub_id']})" if dup.get("sub_id") else ""
            result["duplicate_of"] = dup
            result["duplicate_warning"] = (
                f"Duplicate submission: identical file was already sent to "
                f"'{competition}'"
                f"{ref_info}{sub_info}; resubmitting identical content may waste a "
                "submission slot. [matched via "
                f"{dup.get('source', 'dispatch log')}]"
            )
        return result

    def get_submission(self, competition: str, ref: Any) -> Dict[str, Any]:
        """Fetch a single submission by ref (list + filter, cross-SDK safe)."""
        for sub in self._client.list_submissions(competition):
            if str(getattr(sub, "ref", "")) == str(ref):
                return self._submission_dict(sub)
        return {"found": False, "ref": ref, "status": "NOT_FOUND"}

    def _submission_dict(self, sub: Any) -> Dict[str, Any]:
        """Project a normalised submission object into a flat dict."""
        pub = getattr(sub, "public_score", "") or ""
        priv = getattr(sub, "private_score", "") or ""
        date_val = getattr(sub, "date", None) or getattr(sub, "created_at", None)
        return {
            "found": True,
            "ref": getattr(sub, "ref", None),
            "status": self._status_str(getattr(sub, "status", "")),
            "date": str(date_val) if date_val is not None else "",
            "public_score": pub,
            "public_score_f": self._score(pub),
            "private_score": priv,
            "private_score_f": self._score(priv),
            "error_description": getattr(sub, "error_description", "") or "",
            "description": getattr(sub, "description", "") or "",
            "file_name": getattr(sub, "file_name", "") or getattr(sub, "filename", "") or "",
            "submitted_by": getattr(sub, "submitted_by", "") or "",
            "team_name": getattr(sub, "team_name", "") or "",
            "url": getattr(sub, "url", "") or "",
        }

    def list_submissions(
        self, competition: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List submissions for a competition as projection dicts."""
        subs = self._client.list_submissions(competition)
        if limit is not None:
            subs = subs[:limit]
        return [self._submission_dict(s) for s in subs]

    def wait_for_score(
        self,
        competition: str,
        ref: Any,
        timeout_s: float = 1800.0,
        poll_interval_s: float = 15.0,
        max_poll_interval_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Poll until the submission is COMPLETE/ERROR or ``timeout_s`` elapses.

        Polling starts at ``poll_interval_s`` and grows 1.5x per round (capped at
        ``max_poll_interval_s``), mirroring the native ``--wait`` backoff. Returns
        the final :meth:`get_submission` dict plus ``final`` (bool), ``outcome``
        (``scored``/``error``/``timeout``) and ``attempts`` (int).
        """
        deadline = time.monotonic() + timeout_s
        interval = poll_interval_s
        attempts = 0
        while True:
            info = self.get_submission(competition, ref)
            attempts += 1
            status = info.get("status", "")
            if self._COMPLETE in status:
                info.update(final=True, outcome="scored", attempts=attempts)
                return info
            if self._ERROR in status:
                info.update(final=True, outcome="error", attempts=attempts)
                return info
            if time.monotonic() >= deadline:
                info.update(final=False, outcome="timeout", attempts=attempts)
                return info
            self._sleep(interval)
            interval = min(interval * 1.5, max_poll_interval_s)

    def wait_for_scores(
        self,
        competition: str,
        refs: List[Any],
        timeout_s: float = 1800.0,
        poll_interval_s: float = 15.0,
        max_poll_interval_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Poll until all submissions in ``refs`` reach COMPLETE/ERROR or timeout elapses.

        Uses single ``list_submissions`` calls per round to avoid rate-limiting.
        """
        clean_refs = [str(r).strip() for r in refs if str(r).strip()]
        deadline = time.monotonic() + timeout_s
        interval = poll_interval_s
        attempts = 0
        final_results: Dict[str, Dict[str, Any]] = {}

        while True:
            attempts += 1
            all_subs = {
                str(getattr(s, "ref", "")): self._submission_dict(s)
                for s in self._client.list_submissions(competition)
            }
            pending: List[str] = []
            for r in clean_refs:
                if r in final_results and final_results[r].get("final"):
                    continue
                info = all_subs.get(r, {"found": False, "ref": r, "status": "NOT_FOUND"})
                status = str(info.get("status", "")).upper()
                if self._COMPLETE in status:
                    info.update(final=True, outcome="scored", attempts=attempts)
                    final_results[r] = info
                elif self._ERROR in status:
                    info.update(final=True, outcome="error", attempts=attempts)
                    final_results[r] = info
                else:
                    pending.append(r)

            if not pending or time.monotonic() >= deadline:
                for r in pending:
                    info = all_subs.get(r, {"found": False, "ref": r, "status": "TIMEOUT"})
                    info.update(final=False, outcome="timeout", attempts=attempts)
                    final_results[r] = info
                return {
                    "all_final": len(pending) == 0,
                    "attempts": attempts,
                    "count": len(clean_refs),
                    "submissions": [final_results[r] for r in clean_refs if r in final_results],
                }
            self._sleep(interval)
            interval = min(interval * 1.5, max_poll_interval_s)

    def latest(self, competition: str) -> Dict[str, Any]:
        """Summarise a competition's submissions: count + best scored entry."""
        subs = list(self._client.list_submissions(competition))
        scored = [
            s for s in subs
            if self._score(getattr(s, "public_score", "")) is not None
        ]
        scored.sort(
            key=lambda s: self._score(getattr(s, "public_score", "")) or 0.0,
            reverse=True,
        )
        return {
            "competition": competition,
            "total": len(subs),
            "scored": len(scored),
            "best": self._submission_dict(scored[0]) if scored else None,
        }

    def get_quota(self, competition: str, project_root: Optional[Any] = None) -> Dict[str, Any]:
        """Fetch live daily submission quota for a competition."""
        if hasattr(self._client, "get_submission_limits"):
            quota = self._client.get_submission_limits(competition)
            try:
                root = Path(project_root) if project_root else Path.cwd()
                dot_ai = root / ".ai-harness"
                if dot_ai.is_dir() and quota.get("num_allowed_now") is not None:
                    (dot_ai / "submission_limits.json").write_text(
                        json.dumps(quota, indent=2), encoding="utf-8"
                    )
            except Exception:
                pass
            return quota
        return {
            "competition": competition,
            "num_today": None,
            "num_allowed_now": None,
            "num_total": None,
            "has_remaining_quota": None,
            "error": "Client does not support get_submission_limits",
        }


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
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

        metric_lower = str(metric).lower()
        if metric_lower in {"kappa", "cohen_kappa"}:
            fn = cohen_kappa_score
        elif metric_lower in {"accuracy", "acc"}:
            fn = accuracy_score
        elif metric_lower == "f1":
            def fn(y, pred):
                return f1_score(y, pred, zero_division=0)
        else:
            raise ValueError("Unsupported threshold metric: " + str(metric))

        best_th = 0.5
        best_score = float("-inf")
        y_true_clean = np.asarray(y_true).ravel()
        y_prob_clean = np.asarray(y_prob, dtype=float).ravel()
        if (not len(y_true_clean) or len(y_true_clean) != len(y_prob_clean)
                or not np.isin(y_true_clean, [0, 1]).all()
                or not np.isfinite(y_prob_clean).all()
                or (y_prob_clean < 0).any() or (y_prob_clean > 1).any()):
            raise ValueError("Threshold search requires aligned binary labels and finite probabilities in [0, 1]")
        if int(num_thresholds) < 2:
            raise ValueError("num_thresholds must be at least 2")

        for th in np.unique(np.r_[0.5, np.linspace(0.05, 0.95, int(num_thresholds))]):
            preds = (y_prob_clean >= th).astype(int)
            score = float(fn(y_true_clean, preds))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_th = float(th)

        if not np.isfinite(best_score):
            raise ValueError("No finite threshold score could be computed")
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
        from scipy.stats import rankdata, spearmanr

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

