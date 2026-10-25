"""Reference TaskPlugin adapter: the canonical, runnable example.

Copy this file into your own adapter package and edit the hooks. The ONLY
canonical interface is :class:`harness.plugins.TaskPlugin`; every hook below
exists on that class and is called by the harness runtime. Override only what
your competition needs -- anything not overridden falls back to the declarative
defaults driven by ``configs/task.yaml``.

Register it in your adapter package's ``pyproject.toml``::

    [project.entry-points."ai_harness.task_plugins"]
    example_adapter = "my_adapter_package.example_adapter:ExampleAdapterPlugin"

Hooks demonstrated (signature = what the runtime actually calls):

* ``load_training_data(context)``            -> DataFrame | None
* ``build_validation_splits(context, data, strategy, validation_config)``
                                             -> [(fold_id, train_df, val_df)] | None
* ``evaluate_predictions(context, ground_truth, predictions)`` -> metric dict
* ``build_submission(context, predictions, output_path)``      -> Path
* ``validate_submission(context, submission)``                 -> None (raise on error)
* ``extend_eda_report(context, reports)``                      -> dict | list | None

There is intentionally no ``load_test_data`` or ``run_eda`` hook: test-set
inference is driven by your experiment script, and EDA is extended through
``extend_eda_report``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from harness.plugins import TaskPlugin, TaskPluginContext, TaskPluginError


class ExampleAdapterPlugin(TaskPlugin):
    """Binary-label adapter with a custom metric and a strict submission format."""

    plugin_id = "example_adapter"
    version = "1"

    ID_COLUMN = "id"
    LABEL_COLUMN = "label"
    PREDICTION_COLUMN = "pred_label"

    def load_training_data(self, context: TaskPluginContext) -> Optional[pd.DataFrame]:
        path = context.project_root / "dataset" / "train.csv"
        frame = pd.read_csv(path)
        frame[self.ID_COLUMN] = frame[self.ID_COLUMN].astype(str)
        return frame

    def build_validation_splits(
        self,
        context: TaskPluginContext,
        data: pd.DataFrame,
        strategy: str,
        validation_config: Dict[str, Any],
    ) -> Optional[List[Tuple[Any, pd.DataFrame, pd.DataFrame]]]:
        if strategy not in ("kfold", "stratified_kfold"):
            # Defer custom strategies (e.g. group_kfold, time_series) to core harness defaults
            return None
        n_folds = int(validation_config.get("num_folds", 2))
        # Deterministic, class-stratified split ensuring each fold has balanced class distribution
        if self.LABEL_COLUMN in data:
            folds = pd.Series(index=data.index, dtype=int)
            for _, group in data.groupby(self.LABEL_COLUMN):
                group_folds = pd.Series(range(len(group)), index=group.index) % n_folds
                folds.loc[group.index] = group_folds
        else:
            folds = pd.Series(range(len(data)), index=data.index) % n_folds
        return [
            (fold, data[folds != fold].copy(), data[folds == fold].copy())
            for fold in range(n_folds)
        ]

    def evaluate_predictions(
        self,
        context: TaskPluginContext,
        ground_truth: pd.DataFrame,
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        # ``predictions`` maps ``pred_<target>`` -> list of values (row-aligned).
        predicted = pd.Series(predictions[self.PREDICTION_COLUMN]).astype(int).to_numpy()
        truth = ground_truth[self.LABEL_COLUMN].astype(int).to_numpy()
        recalls = [
            float((predicted[truth == cls] == cls).mean())
            for cls in sorted(set(truth.tolist()))
        ]
        # The harness reads ``primary_score``; extra keys are preserved as-is.
        return {"primary_score": sum(recalls) / len(recalls), "balanced_accuracy": sum(recalls) / len(recalls)}

    def build_submission(
        self, context: TaskPluginContext, predictions: pd.DataFrame, output_path: Path
    ) -> Path:
        submission = predictions[[self.ID_COLUMN, self.PREDICTION_COLUMN]].rename(
            columns={self.PREDICTION_COLUMN: self.LABEL_COLUMN}
        )
        submission.to_csv(output_path, index=False)
        return output_path

    def validate_submission(self, context: TaskPluginContext, submission: Path) -> None:
        frame = pd.read_csv(submission)
        if list(frame.columns) != [self.ID_COLUMN, self.LABEL_COLUMN]:
            raise TaskPluginError(
                f"Submission columns must be [{self.ID_COLUMN!r}, {self.LABEL_COLUMN!r}], "
                f"got {list(frame.columns)}"
            )
        if not set(frame[self.LABEL_COLUMN].unique()) <= {0, 1}:
            raise TaskPluginError("Submission labels must be 0 or 1.")

    def extend_eda_report(
        self, context: TaskPluginContext, reports: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
        return None  # keep generic EDA; return a dict/list to add task-specific checks
