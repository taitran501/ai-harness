"""Ensemble Diversity and Superiority Guard.

Prevents AI agents from submitting pseudo-ensembles (averaging or voting over nearly identical
models from the same family) by enforcing physical statistical orthogonality:
1. Pairwise Cohen's Kappa (classification) / Pearson & Spearman Correlation (regression) ceiling.
2. Prediction Disagreement Rate (disagreement >= 5.0% floor) with scale-invariant continuous tolerance.
3. Statistical Superiority: Ensemble score must exceed the best constituent by at least delta.
4. Rescue Sample Analysis: Proves that the ensemble physically corrects constituent errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


class EnsembleGuard:
    """Enforces diversity and genuine performance superiority for ensembles."""

    @staticmethod
    def _is_continuous_predictions(
        a_raw: np.ndarray,
        b_raw: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect whether prediction arrays represent continuous values or discrete classes."""
        # 1. Booleans are always discrete
        if a_raw.dtype == bool or b_raw.dtype == bool:
            return False, None, None

        # 2. Both are already floating-point
        if np.issubdtype(a_raw.dtype, np.floating) and np.issubdtype(b_raw.dtype, np.floating):
            return True, a_raw.astype(float), b_raw.astype(float)

        # 3. Attempt robust numeric conversion (handles object series, float strings, numeric arrays)
        try:
            s_a = pd.to_numeric(pd.Series(a_raw), errors="raise")
            s_b = pd.to_numeric(pd.Series(b_raw), errors="raise")
        except (ValueError, TypeError):
            # Contains non-numeric strings (e.g. "yes", "no", "cat") -> discrete
            return False, None, None

        # 4. If either is float dtype, or any value has a fractional part -> continuous
        is_float_type = np.issubdtype(s_a.dtype, np.floating) or np.issubdtype(s_b.dtype, np.floating)
        arr_a = s_a.to_numpy(dtype=float)
        arr_b = s_b.to_numpy(dtype=float)

        has_fractions = bool(np.any(np.mod(arr_a, 1) != 0) or np.any(np.mod(arr_b, 1) != 0))
        if is_float_type or has_fractions:
            return True, arr_a, arr_b

        # 5. Integer-only data: distinguish discrete classes (low cardinality) vs continuous targets (high cardinality)
        unique_a = len(np.unique(arr_a))
        unique_b = len(np.unique(arr_b))
        if unique_a > 20 or unique_b > 20:
            # High cardinality integers (e.g. house prices or count regression) -> continuous
            return True, arr_a, arr_b

        # Low cardinality integers (e.g. classes 0, 1) -> discrete
        return False, None, None

    @classmethod
    def compute_pairwise_diversity(
        cls,
        preds_a: pd.Series | np.ndarray,
        preds_b: pd.Series | np.ndarray,
        rel_epsilon: float = 0.05,
        is_continuous: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Compute Cohen's Kappa / Pearson Correlation and Disagreement Rate between two prediction vectors."""
        a = np.asarray(preds_a)
        b = np.asarray(preds_b)

        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
        if len(a) == 0:
            raise ValueError("Prediction vectors must not be empty.")

        # Determine modality if not explicitly supplied
        if is_continuous is None:
            continuous_mode, a_num, b_num = cls._is_continuous_predictions(a, b)
        elif is_continuous:
            continuous_mode = True
            a_num = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
            b_num = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
        else:
            continuous_mode = False
            a_num = None
            b_num = None

        if continuous_mode:
            # Continuous mode: Pearson and Spearman correlation
            std_a = float(np.std(a_num))
            std_b = float(np.std(b_num))
            mean_a = float(np.mean(np.abs(a_num))) if len(a_num) > 0 else 1.0
            mean_b = float(np.mean(np.abs(b_num))) if len(b_num) > 0 else 1.0
            has_var_a = (std_a > 1e-8) or (std_a > 0 and (std_a / max(mean_a, 1e-12)) > 1e-7)
            has_var_b = (std_b > 1e-8) or (std_b > 0 and (std_b / max(mean_b, 1e-12)) > 1e-7)

            if len(a_num) > 1 and has_var_a and has_var_b:
                corr_p, _ = pearsonr(a_num, b_num)
                corr_s, _ = spearmanr(a_num, b_num)
                pearson_val = float(corr_p)
                spearman_val = float(corr_s)
                is_close = bool(np.allclose(a_num, b_num, atol=0.0, rtol=1e-5))
                if np.isnan(pearson_val):
                    pearson_val = 1.0 if is_close else 0.0
                if np.isnan(spearman_val):
                    spearman_val = 1.0 if is_close else 0.0
            else:
                is_close = bool(np.allclose(a_num, b_num, atol=0.0, rtol=1e-5))
                pearson_val = 1.0 if is_close else 0.0
                spearman_val = 1.0 if is_close else 0.0

            # Scale-invariant relative value disagreement rate:
            # |a - b| / max(std(a), std(b), 1e-8) >= rel_epsilon
            scale = max(float(np.std(a_num)), float(np.std(b_num)), 1e-8)
            diff_mask = (np.abs(a_num - b_num) / scale) >= rel_epsilon
            disagreement_count = int(diff_mask.sum())
            disagreement_rate = float(disagreement_count / len(a_num))

            return {
                "metric_type": "continuous",
                "cohen_kappa": round(pearson_val, 4),  # Preserved for backward compatibility
                "pearson_corr": round(pearson_val, 4),
                "spearman_corr": round(spearman_val, 4),
                "rel_epsilon": rel_epsilon,
                "disagreement_count": disagreement_count,
                "disagreement_rate": round(disagreement_rate, 4),
                "disagreement_pct": round(disagreement_rate * 100, 2),
            }

        # Discrete / Categorical mode: Cohen's Kappa and exact inequality
        kappa = float(cohen_kappa_score(a, b))
        diff_mask = (a != b)
        disagreement_count = int(diff_mask.sum())
        disagreement_rate = float(disagreement_count / len(a))

        return {
            "metric_type": "discrete",
            "cohen_kappa": round(kappa, 4),
            "disagreement_count": disagreement_count,
            "disagreement_rate": round(disagreement_rate, 4),
            "disagreement_pct": round(disagreement_rate * 100, 2),
        }

    @classmethod
    def audit_constituent_diversity(
        cls,
        constituent_dfs: List[pd.DataFrame],
        col: Optional[str] = None,
        max_kappa: float = 0.85,
        min_disagreement_pct: float = 5.0,
        rel_epsilon: float = 0.05,
        is_continuous: Optional[bool] = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Verify that all constituent models are sufficiently diverse and orthogonal."""
        if len(constituent_dfs) < 2:
            return (
                False,
                ["Ensemble must contain at least 2 distinct constituent prediction files."],
                {},
            )

        # Dynamically determine target column if not explicitly provided
        target_col = col
        task_is_regression = None
        if not target_col or is_continuous is None:
            first_df = constituent_dfs[0]
            try:
                from harness.task_spec import load_active_task_spec
                spec = load_active_task_spec()
                if spec.task_type == "regression":
                    task_is_regression = True
                if spec.primary_target:
                    pred_cand = f"pred_{spec.primary_target}"
                    if pred_cand in first_df.columns:
                        target_col = target_col or pred_cand
                    elif spec.primary_target in first_df.columns:
                        target_col = target_col or spec.primary_target
                    if spec.primary_target in spec.subtasks:
                        st = spec.subtasks[spec.primary_target]
                        if st.task_type == "regression":
                            task_is_regression = True
            except Exception:
                pass
            if not target_col:
                target_col = next(
                    (c for c in first_df.columns if c.startswith("pred_")),
                    first_df.columns[0],
                )

        if is_continuous is None and task_is_regression is not None:
            is_continuous = task_is_regression

        errors = []
        pairwise_stats = []

        for i in range(len(constituent_dfs)):
            for j in range(i + 1, len(constituent_dfs)):
                div = cls.compute_pairwise_diversity(
                    constituent_dfs[i][target_col],
                    constituent_dfs[j][target_col],
                    rel_epsilon=rel_epsilon,
                    is_continuous=is_continuous,
                )
                div["pair"] = f"model_{i}_vs_model_{j}"
                pairwise_stats.append(div)

                if div.get("metric_type") == "continuous":
                    effective_corr = max(div["pearson_corr"], div["spearman_corr"])
                    if effective_corr > max_kappa:
                        errors.append(
                            f"Excessive correlation between model {i} and model {j} on '{target_col}': "
                            f"Pearson Correlation = {div['pearson_corr']:.4f}, "
                            f"Spearman Correlation = {div['spearman_corr']:.4f} (ceiling: {max_kappa}). "
                            f"Models are too similar; pseudo-ensembling detected!"
                        )
                    if div["disagreement_pct"] < min_disagreement_pct:
                        errors.append(
                            f"Insufficient disagreement between model {i} and model {j} on '{target_col}' "
                            f"(rel_eps={rel_epsilon}): {div['disagreement_pct']:.2f}% "
                            f"(minimum required: {min_disagreement_pct}%)."
                        )
                else:
                    if div["cohen_kappa"] > max_kappa:
                        errors.append(
                            f"Excessive correlation between model {i} and model {j} on '{target_col}': "
                            f"Cohen's Kappa = {div['cohen_kappa']:.4f} (ceiling: {max_kappa}). "
                            f"Models are too similar; pseudo-ensembling detected!"
                        )
                    if div["disagreement_pct"] < min_disagreement_pct:
                        errors.append(
                            f"Insufficient disagreement between model {i} and model {j} on '{target_col}': "
                            f"{div['disagreement_pct']:.2f}% (minimum required: {min_disagreement_pct}%)."
                        )

        passed = len(errors) == 0
        details = {
            "column_checked": target_col,
            "pairwise_stats": pairwise_stats,
            "max_kappa_ceiling": max_kappa,
            "min_disagreement_floor": min_disagreement_pct,
            "rel_epsilon": rel_epsilon,
            "metric_type": pairwise_stats[0]["metric_type"] if pairwise_stats else "unknown",
        }
        return passed, errors, details

    @classmethod
    def verify_ensemble_superiority(
        cls,
        ensemble_score: float,
        constituent_scores: List[float],
        min_margin: float = 0.005,
        direction: str = "maximize",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Assert that the ensemble outperforms the single best constituent model."""
        if not constituent_scores:
            return False, "No constituent scores provided.", {}

        if direction == "minimize":
            best_constituent = min(constituent_scores)
            margin = best_constituent - ensemble_score
        else:
            best_constituent = max(constituent_scores)
            margin = ensemble_score - best_constituent

        passed = margin >= min_margin
        msg = (
            f"Ensemble score ({ensemble_score:.4f}) vs Best constituent ({best_constituent:.4f}) [{direction}]: "
            f"Margin = +{margin:.4f} (Required: +{min_margin:.4f})."
        )
        if not passed:
            msg = (
                f"REJECTED: Ensemble does not provide meaningful improvement over single best model! "
                f"Ensemble: {ensemble_score:.4f}, Best Single: {best_constituent:.4f}, Margin: {margin:.4f} < {min_margin:.4f}."
            )

        return passed, msg, {
            "ensemble_score": round(ensemble_score, 4),
            "best_constituent_score": round(best_constituent, 4),
            "margin": round(margin, 4),
            "min_margin_required": min_margin,
            "direction": direction,
        }

    @staticmethod
    def analyze_rescue_samples(
        y_true: pd.Series | np.ndarray,
        cand_ensemble: pd.Series | np.ndarray,
        base_model_a: pd.Series | np.ndarray,
        base_model_b: Optional[pd.Series | np.ndarray] = None,
        *other_models: pd.Series | np.ndarray,
        rel_epsilon: float = 0.05,
    ) -> Dict[str, Any]:
        """Analyze how many samples were genuinely 'rescued' by ensembling across N models.

        Supports both discrete classification and continuous regression.
        For regression, a sample is rescued from model m if ensemble error is lower
        by at least delta = rel_epsilon * std(y).
        """
        y_arr = np.asarray(y_true)
        ens_arr = np.asarray(cand_ensemble)
        models_arr = [np.asarray(base_model_a)]
        if base_model_b is not None:
            models_arr.append(np.asarray(base_model_b))
        for m in other_models:
            models_arr.append(np.asarray(m))

        # Check whether inputs are continuous or discrete
        is_continuous = False
        try:
            s_y = pd.to_numeric(pd.Series(y_arr), errors="raise")
            s_ens = pd.to_numeric(pd.Series(ens_arr), errors="raise")
            if y_arr.dtype == bool or ens_arr.dtype == bool:
                is_continuous = False
            elif np.issubdtype(s_y.dtype, np.floating) or np.issubdtype(s_ens.dtype, np.floating):
                if bool(np.any(np.mod(s_y, 1) != 0) or np.any(np.mod(s_ens, 1) != 0)):
                    is_continuous = True
                elif np.issubdtype(y_arr.dtype, np.floating) or np.issubdtype(ens_arr.dtype, np.floating):
                    is_continuous = True
                elif len(np.unique(s_y)) > 20:
                    is_continuous = True
            elif len(np.unique(s_y)) > 20:
                is_continuous = True
        except (ValueError, TypeError):
            is_continuous = False

        if is_continuous:
            y = s_y.to_numpy(dtype=float)
            ens = s_ens.to_numpy(dtype=float)
            models = [pd.to_numeric(pd.Series(m), errors="coerce").to_numpy(dtype=float) for m in models_arr]

            std_y = float(np.std(y))
            mean_abs_y = float(np.mean(np.abs(y))) if len(y) > 0 else 1.0
            scale_y = std_y if std_y > 1e-8 else (mean_abs_y if (not np.isnan(mean_abs_y) and mean_abs_y > 1e-8) else 1.0)
            delta = rel_epsilon * scale_y

            e_ens = np.abs(ens - y)
            e_models = [np.abs(m - y) for m in models]

            res: Dict[str, Any] = {}
            for idx, e_m in enumerate(e_models):
                rescued_mask = e_ens < (e_m - delta)
                res[f"rescued_from_model_{idx}"] = int(rescued_mask.sum())

            min_e_models = np.minimum.reduce(e_models)
            synergistic_mask = e_ens < (min_e_models - delta)
            res["synergistic_rescues"] = int(synergistic_mask.sum())

            max_e_models = np.maximum.reduce(e_models)
            degraded_mask = e_ens > (max_e_models + delta)
            res["degraded_samples"] = int(degraded_mask.sum())

            total_rescues = sum(res[f"rescued_from_model_{idx}"] for idx in range(len(models)))
            res["net_gain"] = total_rescues - res["degraded_samples"]
            res["mode"] = "continuous"
            res["rel_epsilon"] = rel_epsilon
            res["delta_margin"] = float(f"{delta:.6g}")
            return res

        # Discrete / Categorical mode (preserved exactly)
        y = y_arr
        ens = ens_arr
        models = models_arr

        ens_right = (ens == y)
        wrongs = [(m != y) for m in models]

        res = {}
        for idx, m_wrong in enumerate(wrongs):
            res[f"rescued_from_model_{idx}"] = int((m_wrong & ens_right).sum())

        all_wrong = wrongs[0].copy()
        for w in wrongs[1:]:
            all_wrong = all_wrong & w

        none_wrong = (~wrongs[0]).copy()
        for w in wrongs[1:]:
            none_wrong = none_wrong & (~w)

        res["synergistic_rescues"] = int((all_wrong & ens_right).sum())
        res["degraded_samples"] = int((none_wrong & (~ens_right)).sum())
        total_rescues = sum(res[f"rescued_from_model_{idx}"] for idx in range(len(models)))
        res["net_gain"] = total_rescues - res["degraded_samples"]
        res["mode"] = "discrete"
        return res
