"""Hard Guardrails & Scientific Constraints Validator.

Enforces physical code assertions instead of relying on LLM self-policing:
1. Model parameter ceiling assertion.
    2. Grouped Cross-Validation split leakage check.
    3. Task-declared conditional target integrity.
"""

from __future__ import annotations

from typing import Any, Optional, Set

import pandas as pd


class ExperimentValidator:
    """Deterministic constraint validator for ML models and pipelines."""

    @staticmethod
    def count_trainable_parameters(model: Any) -> Optional[int]:
        """Count trainable parameters without confusing them with model size."""
        if hasattr(model, "parameters") and callable(getattr(model, "parameters")):
            return sum(
                int(parameter.numel())
                for parameter in model.parameters()
                if getattr(parameter, "requires_grad", True)
            )
        return ExperimentValidator._count_fitted_estimator_parameters(model)

    @staticmethod
    def count_logical_parameters(model: Any) -> Optional[int]:
        """Count all logical parameters, including frozen backbone weights.

        ``None`` means that the object does not expose a trustworthy count.
        A trainable-parameter count is never used as a substitute.
        """
        if hasattr(model, "parameters") and callable(getattr(model, "parameters")):
            return sum(int(parameter.numel()) for parameter in model.parameters())
        if hasattr(model, "num_parameters") and callable(getattr(model, "num_parameters")):
            try:
                count = model.num_parameters()
            except TypeError:
                return None
            return int(count) if isinstance(count, int) and not isinstance(count, bool) else None
        return ExperimentValidator._count_fitted_estimator_parameters(model)

    @staticmethod
    def _count_fitted_estimator_parameters(model: Any) -> Optional[int]:
        """Return the known fitted coefficient count for sklearn-style estimators."""
        if hasattr(model, "named_steps"):
            total = 0
            found = False
            for step in model.named_steps.values():
                if hasattr(step, "coef_"):
                    total += int(step.coef_.size)
                    found = True
                if hasattr(step, "intercept_") and step.intercept_ is not None:
                    total += int(step.intercept_.size)
                    found = True
            return total if found else None
        if hasattr(model, "coef_"):
            total = int(model.coef_.size)
            if hasattr(model, "intercept_") and model.intercept_ is not None:
                total += int(model.intercept_.size)
            return total
        return None

    @staticmethod
    def count_model_parameters(model: Any) -> int:
        """Backward-compatible alias for the legacy trainable budget count.

        Older callers expect an integer and use this method with the legacy
        ``max_params_limit`` field. New governance code should use explicit
        trainable or logical-total methods instead.
        """
        count = ExperimentValidator.count_trainable_parameters(model)
        if count is not None:
            return count
        # Preserve the historical estimator protocol for callers that expose
        # only ``num_parameters``. Governed logical totals use the explicit
        # method above and the registry's declared count basis.
        if hasattr(model, "num_parameters") and callable(getattr(model, "num_parameters")):
            try:
                value = model.num_parameters()
            except TypeError:
                value = None
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return 0

    @classmethod
    def assert_trainable_parameter_limit(
        cls, model: Any, max_params: Optional[int] = None
    ) -> int:
        """Enforce the legacy training budget against trainable parameters."""
        param_count = cls.count_model_parameters(model)
        if max_params is not None and param_count > max_params:
            raise ValueError(
                f"[HARD CONSTRAINT VIOLATION] Model has {param_count:,} trainable parameters, "
                f"which exceeds the allowed training budget of {max_params:,}."
            )
        return param_count

    @classmethod
    def assert_logical_parameter_limit(
        cls, model: Any, max_params: Optional[int] = None
    ) -> Optional[int]:
        """Enforce a logical model-size ceiling, failing closed when unknown."""
        param_count = cls.count_logical_parameters(model)
        if max_params is None:
            return param_count
        if param_count is None:
            raise ValueError(
                "MODEL_PARAMETER_COUNT_UNKNOWN: logical total parameter count is unavailable."
            )
        if param_count > max_params:
            raise ValueError(
                f"MODEL_PARAMETER_LIMIT_EXCEEDED: model has {param_count:,} logical parameters, "
                f"exceeding the competition limit of {max_params:,}."
            )
        return param_count

    @classmethod
    def assert_parameter_limit(cls, model: Any, max_params: Optional[int] = None) -> int:
        """Backward-compatible trainable-parameter budget assertion."""
        return cls.assert_trainable_parameter_limit(model, max_params)

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
            try:
                spec = load_active_task_spec()
                group_column = spec.group_column
            except Exception:
                group_column = None

        if not group_column or group_column not in train_df.columns or group_column not in val_df.columns:
            return  # Column not available in this view

        train_groups: Set[Any] = set(train_df[group_column].dropna().unique())
        val_groups: Set[Any] = set(val_df[group_column].dropna().unique())

        intersection = train_groups.intersection(val_groups)
        if intersection:
            leaked_sample = list(intersection)[:3]
            raise AssertionError(
                f"[CRITICAL DATA LEAKAGE DETECTED] {len(intersection)} '{group_column}' groups leaked between "
                f"train and validation splits! Sample leaked: {leaked_sample}. "
                f"Pipelines MUST strictly use valid grouped split assignments without cross-fold leakage."
            )

    @classmethod
    def assert_task_conditional_integrity(cls, df: pd.DataFrame, spec: Optional[Any] = None) -> None:
        """Assert all conditional rules declared in UniversalTaskSpec across subtasks."""
        if spec is None:
            from harness.task_spec import load_active_task_spec
            spec = load_active_task_spec()

        for subtask_id, st in getattr(spec, "subtasks", {}).items():
            if st.condition is not None:
                cond = st.condition
                parent_subtask = spec.subtasks.get(cond.parent_target)
                parent_col = parent_subtask.column if parent_subtask else cond.parent_target
                child_col = st.column or subtask_id

                if parent_col in df.columns and child_col in df.columns:
                    active_mask = cond.evaluate_condition(df[parent_col])
                    inactive_mask = ~active_mask
                    if cond.fallback_value is not None:
                        invalid = df[inactive_mask & (df[child_col] != cond.fallback_value)]
                        if len(invalid) > 0:
                            raise AssertionError(
                                f"[CONDITIONAL CONSTRAINT VIOLATION] Found {len(invalid)} rows where '{child_col}' "
                                f"violates conditional dependency on '{parent_col}'! Expected '{cond.fallback_value}'."
                            )
