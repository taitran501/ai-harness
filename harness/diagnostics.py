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
        """Produce full structured diagnostic dictionary dynamically adapted to task schema."""
        # 1. Dynamically resolve ID column
        id_col = None
        for candidate_id in ["StereoQueerEval_id", "id", "ID", "comment_id", "sample_id"]:
            if candidate_id in eval_df.columns and candidate_id in oof_df.columns:
                id_col = candidate_id
                break
        if not id_col:
            from harness.task_spec import load_active_task_spec
            spec = load_active_task_spec()
            if spec.id_column in eval_df.columns and spec.id_column in oof_df.columns:
                id_col = spec.id_column
            else:
                id_col = eval_df.columns[0]

        merged = pd.merge(
            eval_df,
            oof_df,
            on=id_col,
            suffixes=("", "_pred_meta"),
        )

        bottlenecks: List[str] = []
        target_confusions: Dict[str, Any] = {}

        # 2. Generic Target Confusion & Accuracy
        for col in eval_df.columns:
            pred_col = f"pred_{col}"
            if pred_col in merged.columns and merged[col].dtype == object and merged[col].nunique() <= 50:
                labels = sorted(list(merged[col].dropna().unique()))
                cm = cls.compute_confusion_matrix(merged[col], merged[pred_col], labels)
                target_confusions[col] = cm

        # Backward compatibility for Subtask A
        a_cm = target_confusions.get("stereotype")
        if not a_cm and "stereotype" in merged.columns and "pred_stereotype" in merged.columns:
            a_cm = cls.compute_confusion_matrix(merged["stereotype"], merged["pred_stereotype"], ["no", "yes"])

        # Backward compatibility for Subtask B
        b_cm = target_confusions.get("hate_speech")
        if not b_cm and "hate_speech" in merged.columns and "pred_hate_speech" in merged.columns:
            b_cm = cls.compute_confusion_matrix(
                merged["hate_speech"], merged["pred_hate_speech"], ["no", "yes_explicit", "yes_implicit"]
            )

        # 3. Domain or Generic Error Analysis (e.g. Implicit Hate if column exists)
        imp_info = None
        if "hate_speech" in merged.columns and "pred_hate_speech" in merged.columns:
            implicit_true = merged["hate_speech"] == "yes_implicit"
            implicit_total = int(implicit_true.sum())
            implicit_called_no = int((implicit_true & (merged["pred_hate_speech"] == "no")).sum())
            implicit_miss_rate = (
                round(implicit_called_no / implicit_total, 4) if implicit_total > 0 else 0.0
            )
            imp_info = {
                "total_implicit": implicit_total,
                "misclassified_as_no": implicit_called_no,
                "miss_rate": implicit_miss_rate,
            }
            if implicit_miss_rate >= 0.40:
                bottlenecks.append(
                    f"Severe implicit hate speech miss rate: {implicit_miss_rate * 100:.1f}% "
                    f"of 'yes_implicit' misclassified as 'no'."
                )

        # 4. Minority Identity Analysis (if conditional multi-label identities exist)
        c_identity_diag = {}
        if "pred_identities" in merged.columns:
            hateful_df = merged[merged["hate_speech"] != "no"].reset_index(drop=True) if "hate_speech" in merged.columns else merged
            identity_preds = np.zeros((len(hateful_df), len(IDENTITIES)), dtype=int)
            for i, val in enumerate(hateful_df["pred_identities"]):
                if val and val != "none":
                    assigned = [x.strip() for x in str(val).split(",")]
                    for j, ident in enumerate(IDENTITIES):
                        if ident in assigned:
                            identity_preds[i, j] = 1

            c_identity_diag = cls.analyze_minority_identities(hateful_df, identity_preds)
            zero_recall_identities = [k for k, v in c_identity_diag.items() if v["is_zero_recall"]]
            if zero_recall_identities:
                bottlenecks.append(f"Zero recall on rare identity classes: {zero_recall_identities}.")

        # 5. Generic Slice / Language Breakdown
        slice_col = None
        for cand_slice in ["language", "group", "category", "split", "fold"]:
            if cand_slice in merged.columns and 1 < merged[cand_slice].nunique() <= 30:
                slice_col = cand_slice
                break

        slice_breakdown = {}
        if slice_col:
            for s_val, s_df in merged.groupby(slice_col):
                stats = {"total_samples": len(s_df)}
                for t_name in list(target_confusions.keys())[:3]:
                    pred_t = f"pred_{t_name}"
                    stats[f"{t_name}_acc"] = round(float((s_df[t_name] == s_df[pred_t]).mean()), 4)
                if "stereotype" in s_df.columns and "pred_stereotype" in s_df.columns:
                    stats["a_accuracy"] = round(float((s_df["stereotype"] == s_df["pred_stereotype"]).mean()), 4)
                if "hate_speech" in s_df.columns and "pred_hate_speech" in s_df.columns:
                    stats["b_accuracy"] = round(float((s_df["hate_speech"] == s_df["pred_hate_speech"]).mean()), 4)
                if "hate_speech" in s_df.columns:
                    stats["hateful_samples"] = int((s_df["hate_speech"] != "no").sum())
                slice_breakdown[str(s_val)] = stats

        # Generic bottleneck check on low recall classes
        for t_name, cm in target_confusions.items():
            for lbl, row_stats in cm.get("per_class", {}).items():
                if row_stats.get("is_zero_recall"):
                    bottlenecks.append(f"Zero recall on class '{lbl}' in target '{t_name}'.")

        res = {
            "critical_bottlenecks": bottlenecks,
            "target_confusions": target_confusions,
            "slice_column": slice_col,
            "slice_breakdown": slice_breakdown,
            "language_breakdown": slice_breakdown,  # Backward compatible alias
        }
        if a_cm:
            res["subtask_a_confusion"] = a_cm
        if b_cm:
            res["subtask_b_confusion"] = b_cm
        if imp_info:
            res["implicit_hate_analysis"] = imp_info
        if c_identity_diag:
            res["subtask_c_identities"] = c_identity_diag

        return res

    @classmethod
    def render_markdown_report(cls, diag: Dict[str, Any], exp_id: str) -> str:
        """Render markdown diagnostic report dynamically for any competition task."""
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
            lines.append("- 🟢 No critical zero-recall or severe failure bottlenecks detected.")

        # 2. Confusion Matrices
        section_idx = 2
        if diag.get("subtask_b_confusion"):
            lines.extend([
                "",
                f"## {section_idx}. Subtask B (Hate Speech) Confusion Matrix",
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
            section_idx += 1

            if diag.get("implicit_hate_analysis"):
                imp_info = diag["implicit_hate_analysis"]
                lines.extend([
                    "",
                    f"> **Implicit Hate Miss Rate**: {imp_info['miss_rate'] * 100:.1f}% "
                    f"({imp_info['misclassified_as_no']} / {imp_info['total_implicit']} misclassified as `no`).",
                ])

        elif diag.get("target_confusions"):
            for t_name, cm in list(diag["target_confusions"].items())[:2]:
                labels = list(cm.get("matrix", {}).keys())
                lines.extend([
                    "",
                    f"## {section_idx}. Target '{t_name}' Confusion Matrix",
                    "",
                    f"| True Label \\ Predicted | " + " | ".join([f"`{l}`" for l in labels]) + " |",
                    "|---|" + ":---:|"*len(labels),
                ])
                for t_lbl in labels:
                    row = cm["matrix"].get(t_lbl, {})
                    lines.append(f"| **{t_lbl}** | " + " | ".join([str(row.get(l, 0)) for l in labels]) + " |")
                section_idx += 1

        # 3. Subtask C Minority Identity Analysis
        if diag.get("subtask_c_identities"):
            lines.extend([
                "",
                f"## {section_idx}. Subtask C Minority Identity Analysis",
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
            section_idx += 1

        # 4. Slice Breakdown (e.g. Language / Group)
        slice_data = diag.get("slice_breakdown") or diag.get("language_breakdown")
        if slice_data:
            slice_name = diag.get("slice_column", "Slice").capitalize()
            lines.extend([
                "",
                f"## {section_idx}. Per-{slice_name} Breakdown",
                "",
                f"| {slice_name} | Total Samples | Performance Details |",
                "|---|---:|---|",
            ])
            for s_val, sstats in slice_data.items():
                detail_parts = [f"{k}: {v}" for k, v in sstats.items() if k != "total_samples"]
                lines.append(f"| **{s_val}** | {sstats['total_samples']} | {', '.join(detail_parts)} |")

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
