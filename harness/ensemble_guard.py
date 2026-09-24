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
        """Compute Cohen's Kappa and Disagreement Rate between two prediction vectors."""
        a = np.array(preds_a)
        b = np.array(preds_b)

        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")

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
        col: str = "pred_stereotype",
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

        errors = []
        pairwise_stats = []

        for i in range(len(constituent_dfs)):
            for j in range(i + 1, len(constituent_dfs)):
                div = cls.compute_pairwise_diversity(
                    constituent_dfs[i][col], constituent_dfs[j][col]
                )
                div["pair"] = f"model_{i}_vs_model_{j}"
                pairwise_stats.append(div)

                if div["cohen_kappa"] > max_kappa:
                    errors.append(
                        f"Excessive correlation between model {i} and model {j} on '{col}': "
                        f"Cohen's Kappa = {div['cohen_kappa']:.4f} (ceiling: {max_kappa}). "
                        f"Models are too similar; pseudo-ensembling detected!"
                    )
                if div["disagreement_pct"] < min_disagreement_pct:
                    errors.append(
                        f"Insufficient disagreement between model {i} and model {j} on '{col}': "
                        f"{div['disagreement_pct']:.2f}% (minimum required: {min_disagreement_pct}%)."
                    )

        passed = len(errors) == 0
        details = {
            "column_checked": col,
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
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Assert that the ensemble outperforms the single best constituent model."""
        if not constituent_scores:
            return False, "No constituent scores provided.", {}

        best_constituent = max(constituent_scores)
        margin = ensemble_score - best_constituent

        passed = margin >= min_margin
        msg = (
            f"Ensemble score ({ensemble_score:.4f}) vs Best constituent ({best_constituent:.4f}): "
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
        }

    @staticmethod
    def analyze_rescue_samples(
        y_true: pd.Series | np.ndarray,
        cand_ensemble: pd.Series | np.ndarray,
        base_model_a: pd.Series | np.ndarray,
        base_model_b: pd.Series | np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze how many samples were genuinely 'rescued' by ensembling."""
        y = np.array(y_true)
        ens = np.array(cand_ensemble)
        a = np.array(base_model_a)
        b = np.array(base_model_b)

        # Rescued: A was wrong, but B was right (or vice-versa), and ensemble got it right
        a_wrong = a != y
        b_wrong = b != y
        ens_right = ens == y

        rescued_from_a = int((a_wrong & (~b_wrong) & ens_right).sum())
        rescued_from_b = int((b_wrong & (~a_wrong) & ens_right).sum())
        both_wrong_ens_right = int((a_wrong & b_wrong & ens_right).sum())
        degraded_samples = int(((~a_wrong) & (~b_wrong) & (~ens_right)).sum())

        return {
            "rescued_from_model_0": rescued_from_a,
            "rescued_from_model_1": rescued_from_b,
            "synergistic_rescues": both_wrong_ens_right,
            "degraded_samples": degraded_samples,
            "net_gain": (rescued_from_a + rescued_from_b + both_wrong_ens_right) - degraded_samples,
        }
