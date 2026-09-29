"""StereoQueerEval adapter. Competition-specific imports live in this package."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import pandas as pd

from harness.plugins import TaskPlugin, TaskPluginContext, TaskPluginError


class StereoQueerEvalPlugin(TaskPlugin):
    plugin_id = "stereoqueereval"
    version = "0.1.0"

    def _task_module(self, context: TaskPluginContext, module_name: str):
        root = str(context.project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            raise TaskPluginError(
                f"StereoQueerEval adapter could not import '{module_name}' from {context.project_root}: {exc}"
            ) from exc

    def load_training_data(self, context: TaskPluginContext) -> Optional[pd.DataFrame]:
        data_module = self._task_module(context, "src.data")
        data_dir = context.project_root / context.task_spec.train_path
        loaded = data_module.load_dataset(data_dir)
        frame = loaded[0] if isinstance(loaded, tuple) else loaded

        data_config = context.task_spec.metadata.get("data_config", {})
        run_paths = (context.run_config or {}).get("paths", {})
        configured_split = (
            run_paths.get("split_path")
            or data_config.get("split_path")
            or data_config.get("fold_assignments")
            or context.task_spec.validation.split_file
        )
        candidates = [configured_split] if configured_split else [
            "splits/fold_assignments.tsv",
            "splits/folds.csv",
            "splits/splits.csv",
        ]
        split_path = next(
            (
                (context.project_root / candidate).resolve()
                for candidate in candidates
                if candidate and (context.project_root / candidate).is_file()
            ),
            None,
        )
        if split_path is not None:
            split_data = data_module.load_splits(split_path)
            frame = data_module.merge_data_and_splits(frame, split_data)
        else:
            from sklearn.model_selection import GroupKFold

            group_column = context.task_spec.group_column
            if not group_column or group_column not in frame.columns:
                raise TaskPluginError(
                    "No declared group column or split file is available for StereoQueerEval validation."
                )
            unique_groups = frame[group_column].nunique(dropna=True)
            validation = context.task_spec.metadata.get("validation", {})
            n_splits = int(
                data_config.get("n_folds")
                or validation.get("num_folds")
                or context.task_spec.validation.num_folds
            )
            n_splits = min(n_splits, unique_groups)
            if n_splits < 2:
                raise TaskPluginError("GroupKFold requires at least two distinct validation groups.")
            frame = frame.copy()
            frame["fold"] = -1
            splitter = GroupKFold(n_splits=n_splits)
            for fold_id, (_, validation_indices) in enumerate(
                splitter.split(frame, groups=frame[group_column])
            ):
                frame.iloc[validation_indices, frame.columns.get_loc("fold")] = fold_id
        return frame

    def build_model(
        self, context: TaskPluginContext, model_name: str, params: Dict[str, Any]
    ) -> Any:
        model_module = self._task_module(context, "src.models")
        model_params = dict(params)
        if "C" in model_params and "c_param" not in model_params:
            model_params["c_param"] = model_params.pop("C")
        return model_module.build_model(model_name, **model_params)

    def normalize_predictions(
        self, context: TaskPluginContext, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        normalized = dict(predictions)
        if "pred_identities" in normalized:
            normalized.setdefault("pred_subtask_c_identity", normalized["pred_identities"])
        return normalized

    def validate_training_data(self, context: TaskPluginContext, data: pd.DataFrame) -> None:
        data_module = self._task_module(context, "src.data")
        missing = sorted(set(data_module.REQUIRED_COLUMNS) - set(data.columns))
        if missing:
            raise TaskPluginError(f"StereoQueerEval training data is missing columns: {missing}")
        id_column = context.task_spec.id_column
        if id_column and data[id_column].duplicated().any():
            raise TaskPluginError(f"StereoQueerEval training IDs are not unique in '{id_column}'.")
        if data.empty:
            raise TaskPluginError("StereoQueerEval training data is empty.")

    def evaluate_predictions(
        self,
        context: TaskPluginContext,
        ground_truth: pd.DataFrame,
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = self.normalize_predictions(context, predictions)
        generic_metrics = context.task_spec.evaluate_predictions(ground_truth, normalized)
        metrics_module = self._task_module(context, "src.metrics")
        required = (
            "pred_stereotype",
            "pred_hate_speech",
            "pred_target_type",
            "pred_identities",
        )
        missing = [key for key in required if key not in normalized]
        if missing:
            return generic_metrics
        task_metrics = metrics_module.evaluate_predictions(
            ground_truth,
            normalized["pred_stereotype"],
            normalized["pred_hate_speech"],
            normalized["pred_target_type"],
            normalized["pred_identities"],
        )
        return {**generic_metrics, **task_metrics}
