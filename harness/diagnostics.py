"""Diagnostic and Error Slicing Engine for ML-Harness.

Extracts rich diagnostic signals (confusion matrices, minority class recall,
subgroup error slices, top confident failures) so AI agents have actionable,
ground-truth failure modes to target instead of guessing or pseudo-ensembling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


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
        per_class = {}
        for i, true_lbl in enumerate(labels):
            support = int(cm[i].sum())
            tp = int(cm[i, i])
            predicted = int(cm[:, i].sum())
            per_class[str(true_lbl)] = {"support": support, "tp": tp,
                "recall": tp / support if support else 0.0,
                "precision": tp / predicted if predicted else 0.0,
                "is_zero_recall": support > 0 and tp == 0}
            matrix_dict[true_lbl] = {}
            for j, pred_lbl in enumerate(labels):
                matrix_dict[true_lbl][pred_lbl] = int(cm[i, j])
        return {"labels": labels, "matrix": matrix_dict, "per_class": per_class}

    @staticmethod
    def analyze_minority_labels(
        df_target: pd.DataFrame,
        pred_labels: np.ndarray,
        label_names: List[str],
        column_prefix: str = "",
    ) -> Dict[str, Any]:
        """Generic analysis of support, recall, false positive, and false negative counts per label."""
        results = {}
        for idx, lbl in enumerate(label_names):
            col = f"{column_prefix}{lbl}" if column_prefix else lbl
            if col not in df_target.columns:
                continue
            y_true = df_target[col].astype(int).to_numpy()
            y_pred = pred_labels[:, idx].astype(int) if idx < pred_labels.shape[1] else np.zeros(len(df_target), dtype=int)

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

            results[lbl] = {
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
    def analyze_minority_label_matrix(
        cls,
        df_target: pd.DataFrame,
        pred_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        column_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze support, recall, false positives, and false negatives by label."""
        prefix = column_prefix or "label_"
        if labels:
            target_labels = labels
        elif any(c.startswith(prefix) for c in df_target.columns):
            target_labels = [c[len(prefix) :] for c in df_target.columns if c.startswith(prefix)]
        else:
            try:
                from harness.task_spec import load_active_task_spec
                spec = load_active_task_spec()
                target_labels = [
                    label
                    for subtask in spec.subtasks.values()
                    for label in subtask.labels
                ]
            except Exception:
                target_labels = []
        return cls.analyze_minority_labels(
            df_target=df_target,
            pred_labels=pred_matrix,
            label_names=target_labels,
            column_prefix=prefix,
        )

    @classmethod
    def generate_diagnostics(
        cls,
        eval_df: pd.DataFrame,
        oof_df: pd.DataFrame,
        task_spec=None,
    ) -> Dict[str, Any]:
        """Produce full structured diagnostic dictionary dynamically adapted to task schema."""
        # 1. Dynamically resolve ID column from TaskSpec first
        id_col = None
        spec = task_spec
        try:
            from harness.task_spec import load_active_task_spec
            spec = task_spec or load_active_task_spec()
            if spec.id_column and spec.id_column in eval_df.columns and spec.id_column in oof_df.columns:
                id_col = spec.id_column
        except Exception:
            pass

        if not id_col:
            for candidate_id in ["id", "ID", "sample_id", "row_id", "uid", "item_id"]:
                if candidate_id in eval_df.columns and candidate_id in oof_df.columns:
                    id_col = candidate_id
                    break
        if not id_col:
            common_id_cols = [c for c in eval_df.columns if c in oof_df.columns and c.lower().endswith("id")]
            if common_id_cols:
                id_col = common_id_cols[0]
        if not id_col:
            id_col = eval_df.columns[0]

        merged = pd.merge(
            eval_df,
            oof_df,
            on=id_col,
            suffixes=("", "_pred_meta"),
        )

        bottlenecks: List[str] = []
        target_confusions: Dict[str, Any] = {}
        classification_targets = {}
        for key, subtask in getattr(spec, "subtasks", {}).items():
            if subtask.task_type in {"binary_classification", "multiclass_classification"}:
                classification_targets[subtask.column or key] = f"pred_{key}"

        # 2. Generic Target Confusion & Accuracy for all categorical/classification targets
        for col in eval_df.columns:
            pred_col = classification_targets.get(col, f"pred_{col}")
            # Support both pandas 2.x and pandas 3.x categorical/string types
            col_is_categorical = (
                merged[col].dtype == object
                or merged[col].dtype.kind == "O"
                or str(merged[col].dtype).lower() in ("object", "string", "str", "category")
            )
            if pred_col in merged.columns and (col_is_categorical or col in classification_targets) and merged[col].nunique() <= 50:
                actual = merged[col].astype(str)
                predicted = merged[pred_col].astype(str)
                labels = sorted(set(actual) | set(predicted))
                cm = cls.compute_confusion_matrix(actual, predicted, labels)
                target_confusions[col] = cm

                # Identify zero-recall or severely failing classes (recall < 40%)
                for lbl, row_stats in cm.get("per_class", {}).items():
                    if row_stats.get("is_zero_recall"):
                        bottlenecks.append(f"Zero recall on class '{lbl}' in target '{col}'.")
                    elif row_stats.get("support", 0) >= 5 and row_stats.get("recall", 1.0) < 0.40:
                        bottlenecks.append(
                            f"Low recall ({row_stats['recall']*100:.1f}%) on class '{lbl}' in target '{col}' "
                            f"({row_stats['tp']}/{row_stats['support']} correct)."
                        )

        # 3. Generic Multi-Label / Minority Label Scan
        minority_labels: Dict[str, Any] = {}
        for key, subtask in getattr(spec, "subtasks", {}).items():
            if subtask.task_type != "multilabel_classification" or f"pred_{key}" not in merged:
                continue
            rows = [json.loads(v) if isinstance(v, str) else v for v in merged[f"pred_{key}"]]
            matrix = np.asarray(rows)
            if matrix.shape != (len(merged), len(subtask.labels)):
                raise ValueError(f"Invalid multilabel diagnostic prediction shape for {key}")
            stats = cls.analyze_minority_labels(merged, matrix, subtask.labels, subtask.column_prefix or "")
            for label, values in stats.items():
                minority_labels[f"{key}.{label}"] = values
                if values["is_zero_recall"]:
                    bottlenecks.append(f"Zero recall on label '{label}' in target '{key}'.")
        for col in eval_df.columns:
            if col.startswith("label_") or col.startswith("tag_"):
                pred_col = f"pred_{col}"
                if pred_col in merged.columns:
                    lbl_name = col.split("_", 1)[1]
                    y_true = merged[col].astype(int).to_numpy()
                    y_pred = merged[pred_col].astype(int).to_numpy()
                    support = int((y_true == 1).sum())
                    tp = int(((y_true == 1) & (y_pred == 1)).sum())
                    fp = int(((y_true == 0) & (y_pred == 1)).sum())
                    fn = int(((y_true == 1) & (y_pred == 0)).sum())
                    recall = round(tp / support, 4) if support > 0 else 0.0
                    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
                    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
                    is_zero = support > 0 and tp == 0
                    minority_labels[lbl_name] = {
                        "support": support,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "is_zero_recall": is_zero,
                    }
                    if is_zero:
                        bottlenecks.append(f"Zero recall on rare label '{lbl_name}' in '{col}'.")

        # 4. Generic Slice Breakdown
        declared_slices: List[str] = []
        try:
            from harness.task_spec import load_active_task_spec

            task_spec = task_spec or load_active_task_spec()
            data_cfg = task_spec.metadata.get("data_config") or task_spec.metadata.get("data", {})
            declared_slices = [
                value
                for value in (task_spec.group_column, data_cfg.get("partition_column"))
                if value
            ]
        except Exception:
            pass
        slice_col = None
        for cand_slice in [*declared_slices, "group", "category", "split", "fold", "segment", "cluster"]:
            if cand_slice in merged.columns and 1 < merged[cand_slice].nunique() <= 30:
                slice_col = cand_slice
                break

        slice_breakdown: Dict[str, Any] = {}
        if slice_col:
            for s_val, s_df in merged.groupby(slice_col):
                stats: Dict[str, Any] = {"total_samples": len(s_df)}
                for t_name in list(target_confusions.keys())[:3]:
                    pred_t = classification_targets.get(t_name, f"pred_{t_name}")
                    stats[f"{t_name}_acc"] = round(float((s_df[t_name] == s_df[pred_t]).mean()), 4)
                slice_breakdown[str(s_val)] = stats

        res: Dict[str, Any] = {
            "critical_bottlenecks": bottlenecks,
            "target_confusions": target_confusions,
            "minority_labels": minority_labels,
            "slice_column": slice_col,
            "slice_breakdown": slice_breakdown,
        }
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

        section_idx = 2

        # 2. Confusion Matrices
        if diag.get("target_confusions"):
            for t_name, cm in list(diag["target_confusions"].items()):
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

        # 3. Minority / Rare Label Analysis
        if diag.get("minority_labels"):
            lines.extend([
                "",
                f"## {section_idx}. Rare & Minority Label Analysis",
                "",
                "| Label | Support | TP | FP | FN | Precision | Recall | F1 | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ])
            for lbl, stats in diag["minority_labels"].items():
                badge = "❌ ZERO RECALL" if stats["is_zero_recall"] else ("⚠️ LOW" if stats["recall"] < 0.3 else "✅ OK")
                lines.append(
                    f"| `{lbl}` | {stats['support']} | {stats['tp']} | {stats['fp']} | {stats['fn']} | "
                    f"{stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {badge} |"
                )
            section_idx += 1

        # 4. Slice Breakdown
        slice_data = diag.get("slice_breakdown")
        if slice_data:
            slice_name = str(diag.get("slice_column", "Slice")).capitalize()
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
        task_spec=None,
    ) -> Dict[str, Any]:
        """Save both diagnostics.json and DIAGNOSTICS.md into output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        diag = cls.generate_diagnostics(eval_df, oof_df,task_spec=task_spec)

        from harness.io_utils import atomic_write_json, atomic_write_text
        atomic_write_json(output_dir / "diagnostics.json", diag, indent=2, ensure_ascii=False)

        md_content = cls.render_markdown_report(diag, exp_id)
        atomic_write_text(output_dir / "DIAGNOSTICS.md", md_content, encoding="utf-8")

        return diag


DiagnosticsEngine = DiagnosticEngine
