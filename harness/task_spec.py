"""Universal Task Specification and Conditional Logic Engine for ML-Harness.

Enables declarative definition of multi-task, hierarchical, and conditional
relationships (if-else routing, oracle masks) between prediction targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


@dataclass
class InputVariantSpec:
    """Specification for an input representation variant."""

    name: str
    type: str  # 'text', 'text_template', 'tabular', 'multimodal'
    source_columns: List[str] = field(default_factory=list)
    template: Optional[str] = None

    def compose_text(self, df: pd.DataFrame) -> pd.Series:
        """Compose text series according to the specification."""
        if self.type == "text":
            col = self.source_columns[0]
            return df[col].fillna("").astype(str)

        if self.type == "text_template" and self.template:
            def format_row(row):
                return self.template.format(**{col: str(row.get(col, "")) for col in self.source_columns})

            return df.apply(format_row, axis=1)

        if len(self.source_columns) > 0:
            return df[self.source_columns[0]].fillna("").astype(str)

        raise ValueError(f"Cannot compose input for variant '{self.name}' with type '{self.type}'.")


@dataclass
class ConditionalRoutingRule:
    """Specifies an if-else dependency between a child target and parent target."""

    parent_target: str
    operator: str  # 'in', 'not_in', '==', '!=', 'gt', 'lt'
    values: Any
    fallback_value: Any
    eval_condition_only: bool = True  # If True, metric is evaluated only on rows where condition is True (e.g. oracle_hate)

    def evaluate_condition(self, parent_values: pd.Series | np.ndarray) -> np.ndarray:
        """Return boolean array where condition holds true."""
        series = pd.Series(parent_values) if not isinstance(parent_values, pd.Series) else parent_values

        if self.operator == "==":
            return (series == self.values).to_numpy()
        elif self.operator == "!=":
            return (series != self.values).to_numpy()
        elif self.operator == "in":
            val_list = self.values if isinstance(self.values, (list, tuple, set)) else [self.values]
            return series.isin(val_list).to_numpy()
        elif self.operator == "not_in":
            val_list = self.values if isinstance(self.values, (list, tuple, set)) else [self.values]
            return (~series.isin(val_list)).to_numpy()
        elif self.operator == "gt":
            return (series.astype(float) > float(self.values)).to_numpy()
        elif self.operator == "lt":
            return (series.astype(float) < float(self.values)).to_numpy()
        else:
            raise ValueError(f"Unsupported conditional operator: {self.operator}")


@dataclass
class SubtaskSpec:
    """Specification for an individual subtask."""

    id: str
    name: str
    task_type: str  # 'binary_classification', 'multiclass_classification', 'multilabel_classification', 'regression'
    column: Optional[str] = None
    column_prefix: Optional[str] = None  # For multi-label (e.g., 'identity_')
    classes: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)  # For multi-label
    primary_metric: str = "macro_f1"
    direction: str = "maximize"
    weight: float = 1.0
    condition: Optional[ConditionalRoutingRule] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubtaskSpec:
        cond_data = data.get("condition")
        cond_obj = None
        if cond_data:
            cond_obj = ConditionalRoutingRule(
                parent_target=cond_data["parent_target"],
                operator=cond_data.get("operator", "=="),
                values=cond_data["values"],
                fallback_value=cond_data.get("fallback_value"),
                eval_condition_only=cond_data.get("eval_condition_only", True),
            )

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            task_type=data.get("task_type", data.get("type", "binary_classification")),
            column=data.get("column"),
            column_prefix=data.get("column_prefix"),
            classes=data.get("classes", []),
            labels=data.get("labels", []),
            primary_metric=data.get("primary_metric", data.get("metric", "macro_f1")),
            direction=data.get("direction", "maximize"),
            weight=float(data.get("weight", 1.0)),
            condition=cond_obj,
        )


@dataclass
class UniversalTaskSpec:
    """Complete specification of a competition/experiment task."""

    name: str
    task_type: str
    primary_metric: str
    direction: str = "maximize"
    inputs: List[InputVariantSpec] = field(default_factory=list)
    subtasks: Dict[str, SubtaskSpec] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> UniversalTaskSpec:
        task_cfg = cfg.get("task", cfg)
        inputs_list = []
        for inp in task_cfg.get("inputs", []):
            inputs_list.append(
                InputVariantSpec(
                    name=inp["name"],
                    type=inp.get("type", "text"),
                    source_columns=inp.get("source_columns", []),
                    template=inp.get("template"),
                )
            )

        subtasks_dict = {}
        st_configs = task_cfg.get("subtasks", task_cfg.get("metrics", []))
        for st in st_configs:
            spec = SubtaskSpec.from_dict(st)
            subtasks_dict[spec.id] = spec

        return cls(
            name=task_cfg.get("name", task_cfg.get("task_name", "universal_task")),
            task_type=task_cfg.get("task_type", task_cfg.get("type", "classification")),
            primary_metric=task_cfg.get("primary_metric", "macro_f1"),
            direction=task_cfg.get("direction", "maximize"),
            inputs=inputs_list,
            subtasks=subtasks_dict,
            metadata=task_cfg,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> UniversalTaskSpec:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    def apply_routing(
        self, raw_predictions: Dict[str, Any], n_samples: int
    ) -> Dict[str, Any]:
        """Apply conditional fallback rules across subtasks during inference."""
        routed = dict(raw_predictions)

        for subtask_id, spec in self.subtasks.items():
            if spec.condition is None:
                continue

            cond = spec.condition
            parent_pred = routed.get(cond.parent_target)
            if parent_pred is None:
                continue

            active_mask = cond.evaluate_condition(parent_pred)
            inactive_mask = ~active_mask

            current_preds = routed.get(subtask_id)
            if current_preds is None:
                continue

            # Route fallback values for inactive instances
            if spec.task_type in ("binary_classification", "multiclass_classification"):
                preds_arr = np.array(current_preds, dtype=object)
                preds_arr[inactive_mask] = cond.fallback_value
                routed[subtask_id] = preds_arr

            elif spec.task_type == "multilabel_classification":
                preds_arr = np.array(current_preds)
                fallback = cond.fallback_value
                if fallback is None or fallback == 0:
                    preds_arr[inactive_mask] = 0
                elif isinstance(fallback, (list, tuple, np.ndarray)):
                    preds_arr[inactive_mask] = np.array(fallback)
                routed[subtask_id] = preds_arr

            elif spec.task_type == "regression":
                preds_arr = np.array(current_preds, dtype=float)
                preds_arr[inactive_mask] = float(cond.fallback_value if cond.fallback_value is not None else 0.0)
                routed[subtask_id] = preds_arr

        return routed

    def compute_evaluation_masks(self, ground_truth_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute evaluation mask per subtask (handles oracle condition masks)."""
        masks = {}
        n_samples = len(ground_truth_df)

        for subtask_id, spec in self.subtasks.items():
            if spec.condition is not None and spec.condition.eval_condition_only:
                parent_subtask = self.subtasks.get(spec.condition.parent_target)
                parent_col = parent_subtask.column if parent_subtask else None
                if parent_col and parent_col in ground_truth_df.columns:
                    parent_vals = ground_truth_df[parent_col]
                    masks[subtask_id] = spec.condition.evaluate_condition(parent_vals)
                else:
                    masks[subtask_id] = np.ones(n_samples, dtype=bool)
            else:
                masks[subtask_id] = np.ones(n_samples, dtype=bool)

        return masks
