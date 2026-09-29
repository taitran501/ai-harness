"""Public task-plugin API and installed entry-point discovery."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


TASK_PLUGIN_ENTRY_POINT_GROUP = "ml_harness.task_plugins"


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
            "submission_validator": type(self).validate_submission is not TaskPlugin.validate_submission,
            "submission_builder": type(self).build_submission is not TaskPlugin.build_submission,
        }


class DeclarativeTaskPlugin(TaskPlugin):
    """Built-in adapter for tasks fully described by configs/task.yaml."""

    plugin_id = "default"


def _plugin_entry_points() -> list[EntryPoint]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=TASK_PLUGIN_ENTRY_POINT_GROUP))
    return list(discovered.get(TASK_PLUGIN_ENTRY_POINT_GROUP, []))


def list_task_plugins() -> list[Dict[str, str]]:
    plugins = [{"id": "default", "distribution": "ml-harness", "target": "DeclarativeTaskPlugin"}]
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
        return DeclarativeTaskPlugin()

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
