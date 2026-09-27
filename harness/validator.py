"""Hard Guardrails & Scientific Constraints Validator.

Enforces physical code assertions instead of relying on LLM self-policing:
1. Model parameter ceiling assertion.
2. Grouped Cross-Validation split leakage check (video title/id intersection).
3. Subtask C conditioning assertions (oracle_hate).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Set
import pandas as pd


class ExperimentValidator:
    """Deterministic constraint validator for ML models and pipelines."""

    @staticmethod
    def count_model_parameters(model: Any) -> int:
        """Count parameters of PyTorch model or estimate for sklearn/custom model."""
        if hasattr(model, "parameters") and callable(getattr(model, "parameters")):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        if hasattr(model, "num_parameters") and callable(getattr(model, "num_parameters")):
            return model.num_parameters()
        # Fallback for sklearn or sparse pipelines
        if hasattr(model, "named_steps"):
            total = 0
            for step_name, step in model.named_steps.items():
                if hasattr(step, "coef_"):
                    total += step.coef_.size
                if hasattr(step, "intercept_") and step.intercept_ is not None:
                    total += step.intercept_.size
            return total
        return 0

    @classmethod
    def assert_parameter_limit(cls, model: Any, max_params: Optional[int] = None) -> int:
        """Assert that model trainable parameters do not exceed maximum ceiling."""
        param_count = cls.count_model_parameters(model)
        if max_params is not None and param_count > max_params:
            raise ValueError(
                f"[HARD CONSTRAINT VIOLATION] Model has {param_count:,} parameters, "
                f"which exceeds the allowed limit of {max_params:,} parameters! "
                f"Agent must reduce hidden dimension, freeze backbone layers, or switch to a lighter model."
            )
        return param_count

    @classmethod
    def assert_no_group_data_leakage(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        group_column: Optional[str] = None,
    ) -> None:
        """Assert that no groups exist in both training and validation folds."""
        if group_column is None:
            from harness.task_spec import load_active_task_spec
            spec = load_active_task_spec()
            group_column = spec.group_column or ("yt_title" if "yt_title" in train_df.columns else "group")

        if group_column not in train_df.columns or group_column not in val_df.columns:
            return  # Column not available in this view

        train_groups: Set[Any] = set(train_df[group_column].dropna().unique())
        val_groups: Set[Any] = set(val_df[group_column].dropna().unique())

        intersection = train_groups.intersection(val_groups)
        if intersection:
            leaked_sample = list(intersection)[:3]
            raise AssertionError(
                f"[CRITICAL DATA LEAKAGE DETECTED] {len(intersection)} '{group_column}' groups leaked between "
                f"train and validation splits! Sample leaked: {leaked_sample}. "
                f"Pipelines MUST strictly use 'splits/fold_assignments.tsv'."
            )

    @classmethod
    def assert_no_video_data_leakage(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        group_column: Optional[str] = None,
    ) -> None:
        """Backward-compatible alias for assert_no_group_data_leakage."""
        return cls.assert_no_group_data_leakage(train_df, val_df, group_column=group_column or "yt_title")

    @classmethod
    def assert_conditional_integrity(
        cls,
        df: pd.DataFrame,
        parent_col: str,
        cond_val: Any,
        child_col: str,
        req_val: Any,
    ) -> None:
        """Assert that if parent_col == cond_val then child_col == req_val."""
        if parent_col not in df.columns or child_col not in df.columns:
            return
        cond_mask = df[parent_col] == cond_val
        invalid = df[cond_mask & (df[child_col] != req_val)]
        if len(invalid) > 0:
            raise AssertionError(
                f"[CONDITIONAL CONSTRAINT VIOLATION] Found {len(invalid)} rows where {parent_col} == '{cond_val}' "
                f"but {child_col} != '{req_val}'."
            )

    @classmethod
    def assert_subtask_c_oracle_conditioning(cls, df: pd.DataFrame) -> None:
        """Assert that Subtask C targets are strictly conditioned on hate_speech != 'no'."""
        if "hate_speech" not in df.columns or "target_type" not in df.columns:
            return
        cls.assert_conditional_integrity(
            df=df,
            parent_col="hate_speech",
            cond_val="no",
            child_col="target_type",
            req_val="none",
        )
