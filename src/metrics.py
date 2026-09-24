"""Evaluation metrics computation for StereoQueerEval subtasks."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.data import IDENTITIES, TARGET_TYPES


def compute_single_label_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str],
) -> Dict[str, Any]:
    """Compute macro-F1, accuracy, and detailed per-class classification report."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def compute_subtask_c_identities(
    true_identity: np.ndarray,
    pred_identity: np.ndarray,
) -> Dict[str, Any]:
    """Compute multi-label metrics for Subtask C identities on hateful rows."""
    if len(true_identity) == 0:
        return {
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "exact_match": 0.0,
            "per_label_f1": {ident: 0.0 for ident in IDENTITIES},
            "dev_support": {ident: 0 for ident in IDENTITIES},
        }

    per_label_f1 = f1_score(true_identity, pred_identity, average=None, zero_division=0)
    supports = true_identity.sum(axis=0)

    return {
        "macro_f1": float(
            f1_score(true_identity, pred_identity, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(true_identity, pred_identity, average="micro", zero_division=0)
        ),
        "exact_match": float(accuracy_score(true_identity, pred_identity)),
        "per_label_f1": {
            identity: float(score) for identity, score in zip(IDENTITIES, per_label_f1)
        },
        "dev_support": {
            identity: int(support) for identity, support in zip(IDENTITIES, supports)
        },
    }


def evaluate_predictions(
    dev_df: pd.DataFrame,
    pred_stereotype: pd.Series | np.ndarray,
    pred_hate_speech: pd.Series | np.ndarray,
    pred_target_type: pd.Series | np.ndarray,
    pred_identities: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate predictions for a development set across Subtasks A, B, and C.

    Subtask C is strictly evaluated on gold hateful rows ('oracle_hate').
    """
    subtask_a = compute_single_label_metrics(
        dev_df["stereotype"], pred_stereotype, ["no", "yes"]
    )
    subtask_b = compute_single_label_metrics(
        dev_df["hate_speech"], pred_hate_speech, ["no", "yes_explicit", "yes_implicit"]
    )

    # Subtask C conditioning on gold hateful rows
    hateful_mask = (dev_df["hate_speech"] != "no").to_numpy()
    dev_c = dev_df[hateful_mask].reset_index(drop=True)

    if len(dev_c) > 0:
        pred_type_c = np.array(pred_target_type)[hateful_mask]
        subtask_c_type = compute_single_label_metrics(
            dev_c["target_type"], pred_type_c, TARGET_TYPES
        )

        true_identity_c = dev_c[[f"identity_{x}" for x in IDENTITIES]].to_numpy()
        pred_identity_c = np.array(pred_identities)[hateful_mask]
        subtask_c_id = compute_subtask_c_identities(true_identity_c, pred_identity_c)
    else:
        subtask_c_type = {"macro_f1": 0.0, "accuracy": 0.0, "per_class": {}}
        subtask_c_id = compute_subtask_c_identities(np.empty((0, len(IDENTITIES))), np.empty((0, len(IDENTITIES))))

    return {
        "subtask_a": subtask_a,
        "subtask_b": subtask_b,
        "subtask_c_target_type_oracle_hate": subtask_c_type,
        "subtask_c_identity_oracle_hate": subtask_c_id,
    }


def format_cv_summary_table(results_df: pd.DataFrame) -> str:
    """Format 5-fold cross validation summary table into Markdown."""
    metric_cols = [
        "a_macro_f1",
        "b_macro_f1",
        "c_target_type_macro_f1_oracle_hate",
        "c_identity_macro_f1_oracle_hate",
        "c_identity_micro_f1_oracle_hate",
    ]
    grouped = (
        results_df.groupby(["language", "input_variant"], sort=True)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    lines = [
        "# Cross-Validation Summary (Mean ± Std)",
        "",
        "> Subtask C is evaluated on gold hateful rows (`oracle_hate`).",
        "",
        "| Language | Input | A macro-F1 | B macro-F1 | C type macro-F1 | C identity macro-F1 | C identity micro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        def fmt(col: str) -> str:
            mean_val = row[(col, "mean")]
            std_val = row[(col, "std")]
            return f"{mean_val:.4f} ± {std_val:.4f}"

        lang = row[("language", "")]
        variant = row[("input_variant", "")]
        lines.append(
            f"| {lang} | {variant} | {fmt('a_macro_f1')} | {fmt('b_macro_f1')} | "
            f"{fmt('c_target_type_macro_f1_oracle_hate')} | {fmt('c_identity_macro_f1_oracle_hate')} | "
            f"{fmt('c_identity_micro_f1_oracle_hate')} |"
        )

    return "\n".join(lines) + "\n"
