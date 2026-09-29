"""StereoQueerEval adapter. Competition-specific imports live in this package."""

from __future__ import annotations

import importlib
import base64
import csv
import hashlib
import io
import re
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import zipfile

import pandas as pd
import yaml

from harness.plugins import TaskPlugin, TaskPluginContext, TaskPluginError


class StereoQueerEvalPlugin(TaskPlugin):
    plugin_id = "stereoqueereval"
    version = "0.1.0"

    @staticmethod
    def _remote_wheel(context: TaskPluginContext) -> Path:
        """Repackage the installed adapter module for an exact, dependency-free remote install."""
        from importlib import metadata

        module_path = Path(__file__).resolve()
        distribution = metadata.distribution("ml-harness-stereoqueereval")
        name = str(distribution.metadata["Name"])
        version = distribution.version
        normalized_name = re.sub(r"[-_.]+", "_", name)
        source = module_path.read_bytes()
        digest = hashlib.sha256(source + version.encode("utf-8")).hexdigest()[:16]
        wheel_dir = context.project_root / ".ml-harness" / "runtime-wheels" / digest
        wheel_dir.mkdir(parents=True, exist_ok=True)
        wheel_path = wheel_dir / f"{normalized_name}-{version}-py3-none-any.whl"
        if wheel_path.is_file():
            return wheel_path

        dist_info = f"{normalized_name}-{version}.dist-info"
        metadata_bytes = distribution.metadata.as_bytes()
        entry_points = distribution.read_text("entry_points.txt")
        if not entry_points:
            entry_points = (
                "[ml_harness.task_plugins]\n"
                "stereoqueereval = stereoqueereval_plugin:StereoQueerEvalPlugin\n"
            )
        entries = {
            module_path.name: source,
            f"{dist_info}/METADATA": metadata_bytes,
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: ml-harness\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ).encode("utf-8"),
            f"{dist_info}/entry_points.txt": entry_points.encode("utf-8"),
            f"{dist_info}/top_level.txt": b"stereoqueereval_plugin\n",
        }
        record_path = f"{dist_info}/RECORD"
        record_buffer = io.StringIO(newline="")
        writer = csv.writer(record_buffer, lineterminator="\n")
        for file_name, content in entries.items():
            encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
            writer.writerow((file_name, f"sha256={encoded}", str(len(content))))
        writer.writerow((record_path, "", ""))
        entries[record_path] = record_buffer.getvalue().encode("utf-8")
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_name, content in entries.items():
                archive.writestr(file_name, content)
        return wheel_path

    def remote_runtime_requirements(self, context: TaskPluginContext) -> list[str]:
        document = yaml.safe_load(context.task_config_path.read_text(encoding="utf-8")) or {}
        task = document.get("task", document)
        requirements = (task.get("runtime") or {}).get("requirements", [])
        if not isinstance(requirements, list) or any(not isinstance(item, str) for item in requirements):
            raise TaskPluginError("Task runtime requirements must be a list of pip requirement strings.")
        return requirements

    def remote_runtime_manifest(self, context: TaskPluginContext) -> Dict[str, Any]:
        from importlib import metadata

        distribution = metadata.distribution("ml-harness-stereoqueereval")
        return {
            "distribution": distribution.metadata["Name"],
            "version": distribution.version,
            "wheel": str(self._remote_wheel(context)),
            "requirements": self.remote_runtime_requirements(context),
        }

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
