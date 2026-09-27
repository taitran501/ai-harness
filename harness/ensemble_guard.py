"""Ensemble Diversity and Superiority Guard.

Prevents AI agents from submitting pseudo-ensembles (averaging or voting over nearly identical
models from the same family) by enforcing physical statistical orthogonality:
1. Pairwise Cohen's Kappa (kappa <= 0.85 ceiling to ensure diversity).
2. Prediction Disagreement Rate (disagreement >= 5.0% floor).
3. Statistical Superiority: Ensemble score must exceed the best constituent by at least delta.
4. Rescue Sample Analysis: Proves that the ensemble physically corrects constituent errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score


class EnsembleGuard:
    """Enforces diversity and genuine performance superiority for ensembles."""

    @staticmethod
    def compute_pairwise_diversity(
        preds_a: pd.Series | np.ndarray,
        preds_b: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """Compute Cohen's Kappa / Pearson Correlation and Disagreement Rate between two prediction vectors."""
        a = np.array(preds_a)
        b = np.array(preds_b)

        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")

        is_float = np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating)
        if is_float:
            from scipy.stats import pearsonr
            if len(a) > 1 and np.std(a) > 1e-8 and np.std(b) > 1e-8:
                corr, _ = pearsonr(a, b)
                metric_val = float(corr)
            else:
                metric_val = 1.0 if np.allclose(a, b) else 0.0

            # Disagreement for continuous values is diff > 0.05
            diff_mask = np.abs(a - b) > 0.05
            disagreement_count = int(diff_mask.sum())
            disagreement_rate = float(disagreement_count / len(a))

            return {
                "cohen_kappa": round(metric_val, 4),
                "pearson_corr": round(metric_val, 4),
                "disagreement_count": disagreement_count,
                "disagreement_rate": round(disagreement_rate, 4),
                "disagreement_pct": round(disagreement_rate * 100, 2),
            }

        kappa = float(cohen_kappa_score(a, b))
        disagreement_count = int((a != b).sum())
        disagreement_rate = float(disagreement_count / len(a))

        return {
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
        if not target_col:
            first_df = constituent_dfs[0]
            target_col = next(
                (c for c in first_df.columns if c.startswith("pred_")),
                "pred_stereotype" if "pred_stereotype" in first_df.columns else first_df.columns[0],
            )

        errors = []
        pairwise_stats = []

        for i in range(len(constituent_dfs)):
            for j in range(i + 1, len(constituent_dfs)):
                div = cls.compute_pairwise_diversity(
                    constituent_dfs[i][target_col], constituent_dfs[j][target_col]
                )
                div["pair"] = f"model_{i}_vs_model_{j}"
                pairwise_stats.append(div)

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
    ) -> Dict[str, Any]:
        """Analyze how many samples were genuinely 'rescued' by ensembling across N models."""
        y = np.array(y_true)
        ens = np.array(cand_ensemble)
        models = [np.array(base_model_a)]
        if base_model_b is not None:
            models.append(np.array(base_model_b))
        for m in other_models:
            models.append(np.array(m))

        ens_right = ens == y
        wrongs = [m != y for m in models]

        res: Dict[str, Any] = {}
        for idx, m_wrong in enumerate(wrongs):
            # Rescued from model idx: model idx was wrong, but ensemble got it right
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
        return res
