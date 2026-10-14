"""Public task-plugin API and installed entry-point discovery."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TASK_PLUGIN_ENTRY_POINT_GROUP = "ai_harness.task_plugins"


class TaskPluginError(RuntimeError):
    """Raised when an installed task plugin is missing or invalid."""


class PluginCapabilityError(TaskPluginError):
    """Raised when an adapter does not implement a requested task capability."""


@dataclass(frozen=True)
class TaskPluginContext:
    project_root: Path
    task_config_path: Path
    task_spec: Any
    plugin_id: str
    run_config: Optional[Dict[str, Any]] = None


class TaskPlugin:
    """Base class for competition adapters installed through Python entry points.

    Plugins own competition-specific contracts. The default implementation uses
    UniversalTaskSpec and is deliberately free of column or platform assumptions.
    """

    plugin_id = "default"
    version = "1"

    def load_training_data(self, context: TaskPluginContext) -> Optional[pd.DataFrame]:
        """Return task-specific training data, or None for declarative loading."""
        return None

    def build_model(
        self, context: TaskPluginContext, model_name: str, params: Dict[str, Any]
    ) -> Any:
        """Return a task-specific model, or None for the generic model factory."""
        return None

    def normalize_predictions(
        self, context: TaskPluginContext, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        return predictions

    def evaluate_predictions(
        self,
        context: TaskPluginContext,
        ground_truth: pd.DataFrame,
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        return context.task_spec.evaluate_predictions(ground_truth, predictions)

    def validate_training_data(
        self, context: TaskPluginContext, data: pd.DataFrame
    ) -> None:
        """Raise TaskPluginError if the task's data contract is invalid."""

    def build_validation_splits(
        self,
        context: TaskPluginContext,
        data: pd.DataFrame,
        strategy: str,
        validation_config: Dict[str, Any],
    ) -> Optional[List[Tuple[Any, pd.DataFrame, pd.DataFrame]]]:
        """Return task-specific ``(fold_id, train, validation)`` splits, or None for core defaults."""
        return None

    def validate_split(
        self,
        context: TaskPluginContext,
        fold_id: Any,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> None:
        """Raise when a task-specific split violates its data contract."""

    def extend_eda_report(
        self, context: TaskPluginContext, reports: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
        """Optionally add task-specific, value-safe checks to generic EDA reports."""
        return None

    def remote_runtime_requirements(self, context: TaskPluginContext) -> List[str]:
        """Return pip requirements needed by this adapter in a remote runtime."""
        return []

    def remote_bundle_files(self, context: TaskPluginContext) -> List[Path]:
        """Return adapter source/config files that must accompany a remote job."""
        return []

    def remote_runtime_manifest(self, context: TaskPluginContext) -> Optional[Dict[str, Any]]:
        """Declare the installable adapter identity and remote dependencies, if supported.

        Custom adapters must opt into this hook so the dispatcher never assumes that
        a locally installed plugin also exists on Kaggle or Modal.
        """
        return None

    def validate_submission(self, context: TaskPluginContext, submission: Path) -> None:
        raise PluginCapabilityError(
            f"Task plugin '{self.plugin_id}' does not provide an official submission validator."
        )

    def build_submission(
        self, context: TaskPluginContext, predictions: pd.DataFrame, output_path: Path
    ) -> Path:
        raise PluginCapabilityError(
            f"Task plugin '{self.plugin_id}' does not provide an official submission builder."
        )

    def capabilities(self) -> Dict[str, bool]:
        return {
            "custom_data_loader": type(self).load_training_data is not TaskPlugin.load_training_data,
            "custom_model_factory": type(self).build_model is not TaskPlugin.build_model,
            "custom_metrics": type(self).evaluate_predictions is not TaskPlugin.evaluate_predictions,
            "custom_validation_split": (
                type(self).build_validation_splits is not TaskPlugin.build_validation_splits
            ),
            "custom_split_validation": type(self).validate_split is not TaskPlugin.validate_split,
            "submission_validator": type(self).validate_submission is not TaskPlugin.validate_submission,
            "submission_builder": type(self).build_submission is not TaskPlugin.build_submission,
            "custom_eda": type(self).extend_eda_report is not TaskPlugin.extend_eda_report,
            "remote_runtime": (
                self.plugin_id == "default"
                or type(self).remote_runtime_manifest is not TaskPlugin.remote_runtime_manifest
            ),
        }


class DeclarativeTaskPlugin(TaskPlugin):
    """Built-in adapter for tasks fully described by configs/task.yaml."""

    plugin_id = "default"

    def __init__(self, context: Optional[TaskPluginContext] = None):
        self.context = context

    def _get_submission_spec(self, context: Optional[TaskPluginContext] = None) -> Optional[Dict[str, Any]]:
        ctx = context or self.context
        task_spec = ctx.task_spec if ctx else None
        if task_spec is None:
            try:
                from harness.task_spec import load_active_task_spec
                task_spec = load_active_task_spec()
            except Exception:
                task_spec = None
        if task_spec is not None:
            sub_spec = getattr(task_spec, "submission_spec", None) or task_spec.metadata.get("submission_spec")
            if isinstance(sub_spec, dict) and sub_spec:
                return sub_spec
        return None

    def capabilities(self) -> Dict[str, bool]:
        caps = super().capabilities()
        sub_spec = self._get_submission_spec()
        has_sub_spec = bool(sub_spec)
        caps["submission_builder"] = has_sub_spec
        caps["submission_validator"] = has_sub_spec
        return caps

    def build_submission(
        self, context: TaskPluginContext, predictions: pd.DataFrame, output_path: Path
    ) -> Path:
        sub_spec = self._get_submission_spec(context)
        if not sub_spec:
            raise PluginCapabilityError(
                f"Task plugin '{self.plugin_id}' does not provide an official submission builder when submission_spec is not declared."
            )
        task_spec = context.task_spec
        id_col = task_spec.id_column or task_spec.data_config.get("id_column", "id")
        if id_col not in predictions.columns:
            matches = [c for c in predictions.columns if c.lower() == id_col.lower()]
            if matches:
                id_col = matches[0]
            else:
                raise ValueError(f"ID column '{id_col}' not found in predictions.")

        out_df = pd.DataFrame()
        req_cols = sub_spec.get("required_columns")
        if req_cols:
            for col in req_cols:
                if col == id_col:
                    if str(id_col).upper() == "GEOID":
                        out_df[col] = predictions[id_col].astype(str).str.split(".").str[0].str.zfill(11)
                    else:
                        out_df[col] = predictions[id_col]
                elif f"pred_{col}" in predictions.columns:
                    out_df[col] = predictions[f"pred_{col}"]
                elif col in predictions.columns:
                    out_df[col] = predictions[col]
                else:
                    found = False
                    for target, st in task_spec.subtasks.items():
                        if st.column == col or target == col:
                            for cand in (f"pred_{target}", f"pred_{st.column}", target, st.column):
                                if cand and cand in predictions.columns:
                                    out_df[col] = predictions[cand]
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        raise ValueError(f"Required submission column '{col}' could not be resolved from predictions.")
        else:
            if str(id_col).upper() == "GEOID":
                out_df[id_col] = predictions[id_col].astype(str).str.split(".").str[0].str.zfill(11)
            else:
                out_df[id_col] = predictions[id_col]
            target_col = task_spec.data_config.get("target_column") or task_spec.primary_target
            if target_col:
                if f"pred_{target_col}" in predictions.columns:
                    out_df[target_col] = predictions[f"pred_{target_col}"]
                elif target_col in predictions.columns:
                    out_df[target_col] = predictions[target_col]
                else:
                    for target, st in task_spec.subtasks.items():
                        st_col = st.column or target
                        for cand in (f"pred_{target}", f"pred_{st_col}", target, st_col):
                            if cand and cand in predictions.columns:
                                out_df[target_col] = predictions[cand]
                                break

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        archive_format = str(sub_spec.get("archive_format", "csv")).lower()
        if archive_format in ("zip",) or output_path.suffix.lower() == ".zip":
            import zipfile
            zip_path = output_path if output_path.suffix.lower() == ".zip" else output_path.with_suffix(".zip")
            inner_name = sub_spec.get("inner_file_name") or "submission.csv"
            csv_str = out_df.to_csv(index=False)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(inner_name, csv_str)
            return zip_path

        out_df.to_csv(output_path, index=False)
        return output_path

    def validate_submission(self, context: TaskPluginContext, submission: Path) -> None:
        sub_spec = self._get_submission_spec(context)
        if not sub_spec:
            raise PluginCapabilityError(
                f"Task plugin '{self.plugin_id}' does not provide an official submission validator when submission_spec is not declared."
            )
        submission = Path(submission)
        if not submission.is_file():
            raise FileNotFoundError(f"Submission file does not exist: {submission}")

        task_spec = context.task_spec
        import zipfile
        archive_format = str(sub_spec.get("archive_format", "csv")).lower()
        if submission.suffix.lower() == ".zip" or (zipfile.is_zipfile(submission) and archive_format == "zip"):
            with zipfile.ZipFile(submission, "r") as zf:
                csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
                if not csv_members:
                    raise ValueError(f"Zip submission '{submission.name}' contains no CSV file.")
                with zf.open(csv_members[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(submission)

        id_col = task_spec.id_column or task_spec.data_config.get("id_column", "id")
        req_cols = sub_spec.get("required_columns")
        if not req_cols:
            req_cols = [id_col, task_spec.data_config.get("target_column") or task_spec.primary_target]
            req_cols = [c for c in req_cols if c]

        for col in req_cols:
            if col not in df.columns:
                raise ValueError(f"Submission missing required column '{col}'. Present: {list(df.columns)}")
            if df[col].isna().any():
                raise ValueError(f"Submission column '{col}' contains null values.")

        if id_col in df.columns:
            if df[id_col].duplicated().any():
                raise ValueError(f"Submission column '{id_col}' contains duplicate values.")
            test_path_val = task_spec.test_path
            if test_path_val:
                tp = Path(test_path_val)
                if not tp.is_absolute():
                    tp = context.project_root / tp
                if tp.exists():
                    try:
                        test_df = task_spec.load_dataset_split("test", project_root=context.project_root)
                        if id_col in test_df:
                            if len(df) != len(test_df):
                                raise ValueError(f"Submission row count ({len(df)}) does not match test set ({len(test_df)}).")
                            if list(df[id_col]) != list(test_df[id_col]):
                                if set(df[id_col]) != set(test_df[id_col]):
                                    raise ValueError("Submission IDs do not match test set IDs.")
                    except Exception as e:
                        if isinstance(e, ValueError) and "Submission" in str(e):
                            raise

        for col in req_cols:
            if col != id_col:
                for st in task_spec.subtasks.values():
                    if (st.column == col or st.id == col) and st.task_type in ("regression", "ranking"):
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if not np.isfinite(vals).all():
                            raise ValueError(f"Submission column '{col}' contains non-finite values.")


def _plugin_entry_points() -> list[EntryPoint]:
    discovered = entry_points()
    groups = [TASK_PLUGIN_ENTRY_POINT_GROUP]
    points = []
    seen = set()
    for grp in groups:
        if hasattr(discovered, "select"):
            found = list(discovered.select(group=grp))
        else:
            found = list(discovered.get(grp, []))
        for pt in found:
            if pt.name not in seen:
                seen.add(pt.name)
                points.append(pt)
    return points


def list_task_plugins() -> list[Dict[str, str]]:
    plugins = [{"id": "default", "distribution": "ai-harness", "target": "DeclarativeTaskPlugin"}]
    plugins.extend(
        {
            "id": point.name,
            "distribution": point.dist.name if point.dist else "unknown",
            "target": point.value,
        }
        for point in _plugin_entry_points()
    )
    return plugins


def load_task_plugin(
    plugin_id: Optional[str], context: TaskPluginContext
) -> TaskPlugin:
    if not plugin_id or plugin_id == "default":
        return DeclarativeTaskPlugin(context)

    points = {point.name: point for point in _plugin_entry_points()}
    point = points.get(plugin_id)
    if point is None:
        installed = ", ".join(sorted(points)) or "none"
        raise TaskPluginError(
            f"Task plugin '{plugin_id}' is not installed. Installed plugins: {installed}. "
            f"Install the task adapter locally, then retry."
        )

    try:
        loaded = point.load()
        plugin = loaded() if isinstance(loaded, type) else loaded
    except Exception as exc:
        raise TaskPluginError(f"Could not load task plugin '{plugin_id}': {exc}") from exc

    if not isinstance(plugin, TaskPlugin):
        raise TaskPluginError(
            f"Entry point '{plugin_id}' must resolve to a TaskPlugin subclass or instance."
        )
    declared_id = getattr(plugin, "plugin_id", plugin_id)
    if declared_id not in (plugin_id, "default"):
        raise TaskPluginError(
            f"Plugin entry point '{plugin_id}' resolved to adapter declaring ID '{declared_id}'."
        )
    return plugin
