"""Diagnostic and Error Slicing Engine for StereoQueerEval.

Extracts rich diagnostic signals (confusion matrices, minority identity recall,
per-language error slices, top confident failures) so AI agents have actionable,
ground-truth failure modes to target instead of guessing or pseudo-ensembling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data import IDENTITIES, TARGET_TYPES


class DiagnosticEngine:
    """Computes failure mode slices and generates diagnostic reports."""

    @staticmethod
    def compute_confusion_matrix(
        y_true: List[str] | pd.Series | np.ndarray,
        y_pred: List[str] | pd.Series | np.ndarray,
        labels: List[str],
    ) -> Dict[str, Any]:
        """Compute confusion matrix and per-class error distribution."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_dict = {}
        for i, true_lbl in enumerate(labels):
            matrix_dict[true_lbl] = {}
            for j, pred_lbl in enumerate(labels):
                matrix_dict[true_lbl][pred_lbl] = int(cm[i, j])
        return {"labels": labels, "matrix": matrix_dict}

    @staticmethod
    def analyze_minority_identities(
        df_hateful: pd.DataFrame,
        pred_identities: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze support, recall, false positive, and false negative counts per identity."""
        results = {}
        for idx, identity in enumerate(IDENTITIES):
            col = f"identity_{identity}"
            if col not in df_hateful.columns:
                continue
            y_true = df_hateful[col].astype(int).to_numpy()
            y_pred = pred_identities[:, idx].astype(int)

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            support = int((y_true == 1).sum())

            recall = round(tp / support, 4) if support > 0 else 0.0
            precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            f1 = (
                round(2 * precision * recall / (precision + recall), 4)
                if (precision + recall) > 0
                else 0.0
            )

            results[identity] = {
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "is_zero_recall": support > 0 and tp == 0,
            }
        return results

    @classmethod
    def generate_diagnostics(
        cls,
        eval_df: pd.DataFrame,
        oof_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Produce full structured diagnostic dictionary."""
        # Align dataframes by StereoQueerEval_id
        merged = pd.merge(
            eval_df,
            oof_df,
            on="StereoQueerEval_id",
            suffixes=("", "_pred_meta"),
        )

        # 1. Subtask A Confusion
        a_labels = ["no", "yes"]
        a_cm = cls.compute_confusion_matrix(
            merged["stereotype"], merged["pred_stereotype"], a_labels
        )

        # 2. Subtask B Confusion (focus on implicit vs explicit vs no)
        b_labels = ["no", "yes_explicit", "yes_implicit"]
        b_cm = cls.compute_confusion_matrix(
            merged["hate_speech"], merged["pred_hate_speech"], b_labels
        )

        # Calculate implicit hate confusion rate: what % of yes_implicit got called 'no'
        implicit_true = merged["hate_speech"] == "yes_implicit"
        implicit_total = int(implicit_true.sum())
        implicit_called_no = int((implicit_true & (merged["pred_hate_speech"] == "no")).sum())
        implicit_miss_rate = (
            round(implicit_called_no / implicit_total, 4) if implicit_total > 0 else 0.0
        )

        # 3. Subtask C Identities on hateful comments
        hateful_mask = merged["hate_speech"] != "no"
        hateful_df = merged[hateful_mask].reset_index(drop=True)

        identity_preds = np.zeros((len(hateful_df), len(IDENTITIES)), dtype=int)
        for i, val in enumerate(hateful_df["pred_identities"]):
            if val and val != "none":
                assigned = [x.strip() for x in str(val).split(",")]
                for j, ident in enumerate(IDENTITIES):
                    if ident in assigned:
                        identity_preds[i, j] = 1

        c_identity_diag = cls.analyze_minority_identities(hateful_df, identity_preds)

        # 4. Per-Language Breakdown
        lang_breakdown = {}
        for lang, lang_df in merged.groupby("language"):
            lang_hateful = lang_df[lang_df["hate_speech"] != "no"]
            lang_breakdown[lang] = {
                "total_samples": len(lang_df),
                "hateful_samples": len(lang_hateful),
                "a_accuracy": round(
                    float((lang_df["stereotype"] == lang_df["pred_stereotype"]).mean()), 4
                ),
                "b_accuracy": round(
                    float((lang_df["hate_speech"] == lang_df["pred_hate_speech"]).mean()), 4
                ),
            }

        # 5. Identify Critical Bottleneck
        zero_recall_identities = [
            k for k, v in c_identity_diag.items() if v["is_zero_recall"]
        ]

        bottlenecks = []
        if implicit_miss_rate >= 0.40:
            bottlenecks.append(
                f"Severe implicit hate speech miss rate: {implicit_miss_rate * 100:.1f}% "
                f"of 'yes_implicit' misclassified as 'no'."
            )
        if zero_recall_identities:
            bottlenecks.append(
                f"Zero recall on rare identity classes: {zero_recall_identities}."
            )

        return {
            "subtask_a_confusion": a_cm,
            "subtask_b_confusion": b_cm,
            "implicit_hate_analysis": {
                "total_implicit": implicit_total,
                "misclassified_as_no": implicit_called_no,
                "miss_rate": implicit_miss_rate,
            },
            "subtask_c_identities": c_identity_diag,
            "language_breakdown": lang_breakdown,
            "critical_bottlenecks": bottlenecks,
        }

    @classmethod
    def render_markdown_report(cls, diag: Dict[str, Any], exp_id: str) -> str:
        """Render markdown diagnostic report."""
        lines = [
            f"# Failure Diagnostics Report: `{exp_id}`",
            "",
            "This report details ground-truth failure modes, error distributions, and critical bottlenecks.",
            "Agents MUST use these findings to formulate hypotheses rather than doing pseudo-ensembling.",
            "",
            "## 1. Critical Bottlenecks (Action Items for Agent)",
            "",
        ]

        if diag.get("critical_bottlenecks"):
            for b in diag["critical_bottlenecks"]:
                lines.append(f"- 🔴 **{b}**")
        else:
            lines.append("- 🟢 No critical zero-recall or >40% miss rate bottlenecks detected.")

        lines.extend([
            "",
            "## 2. Subtask B (Hate Speech) Confusion Matrix",
            "",
            "| True Label \\ Predicted | `no` | `yes_explicit` | `yes_implicit` |",
            "|---|---:|---:|---:|",
        ])

        b_mat = diag["subtask_b_confusion"]["matrix"]
        for true_lbl in ["no", "yes_explicit", "yes_implicit"]:
            row = b_mat.get(true_lbl, {})
            lines.append(
                f"| **{true_lbl}** | {row.get('no', 0)} | {row.get('yes_explicit', 0)} | {row.get('yes_implicit', 0)} |"
            )

        imp_info = diag["implicit_hate_analysis"]
        lines.extend([
            "",
            f"> **Implicit Hate Miss Rate**: {imp_info['miss_rate'] * 100:.1f}% "
            f"({imp_info['misclassified_as_no']} / {imp_info['total_implicit']} misclassified as `no`).",
            "",
            "## 3. Subtask C Minority Identity Analysis",
            "",
            "| Identity | Support | TP | FP | FN | Precision | Recall | F1 | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])

        for ident, stats in diag["subtask_c_identities"].items():
            badge = "❌ ZERO RECALL" if stats["is_zero_recall"] else ("⚠️ LOW" if stats["recall"] < 0.3 else "✅ OK")
            lines.append(
                f"| `{ident}` | {stats['support']} | {stats['tp']} | {stats['fp']} | {stats['fn']} | "
                f"{stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {badge} |"
            )

        lines.extend([
            "",
            "## 4. Per-Language Accuracy Breakdown",
            "",
            "| Language | Total Samples | Hateful Samples | Subtask A Acc | Subtask B Acc |",
            "|---|---:|---:|---:|---:|",
        ])

        for lang, lstats in diag["language_breakdown"].items():
            lines.append(
                f"| **{lang}** | {lstats['total_samples']} | {lstats['hateful_samples']} | "
                f"{lstats['a_accuracy']:.4f} | {lstats['b_accuracy']:.4f} |"
            )

        return "\n".join(lines) + "\n"

    @classmethod
    def save_diagnostics_bundle(
        cls,
        eval_df: pd.DataFrame,
        oof_df: pd.DataFrame,
        output_dir: Path,
        exp_id: str,
    ) -> Dict[str, Any]:
        """Save both diagnostics.json and DIAGNOSTICS.md into output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        diag = cls.generate_diagnostics(eval_df, oof_df)

        with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)

        md_content = cls.render_markdown_report(diag, exp_id)
        (output_dir / "DIAGNOSTICS.md").write_text(md_content, encoding="utf-8")

        return diag
