"""Resolve the exact core/adapter runtime contract before cloud packaging."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
from typing import Any, Dict

from harness.plugins import TaskPluginContext, load_task_plugin, list_task_plugins
from harness.project import HarnessProject
from harness.task_spec import load_active_task_spec


CORE_RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "PyYAML": "PyYAML",
    "scipy": "scipy",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
}


def resolve_remote_runtime(project: HarnessProject) -> Dict[str, Any]:
    """Return a pinned remote runtime manifest or fail closed for undeclared adapters."""
    task_spec = load_active_task_spec(project.task_config_path)
    plugin_id = project.plugin_id or "default"
    context = TaskPluginContext(
        project_root=project.root,
        task_config_path=project.task_config_path,
        task_spec=task_spec,
        plugin_id=plugin_id,
    )
    plugin = load_task_plugin(project.plugin_id, context)
    try:
        core_version = metadata.version("ai-harness")
    except metadata.PackageNotFoundError:
        try:
            core_version = metadata.version("ml-harness")
        except metadata.PackageNotFoundError:
            core_version = "0.2.0"
    core_requirements = []
    for requirement_name, distribution_name in CORE_RUNTIME_DISTRIBUTIONS.items():
        try:
            installed_version = metadata.version(distribution_name)
            core_requirements.append(f"{requirement_name}=={installed_version}")
        except metadata.PackageNotFoundError as exc:
            if requirement_name == "lightgbm":
                core_requirements.append("lightgbm>=4.0.0")
            elif requirement_name == "xgboost":
                core_requirements.append("xgboost>=2.0.0")
            else:
                raise ValueError(
                    f"Cannot pin remote runtime dependency {distribution_name} because it is not installed locally."
                ) from exc

    runtime: Dict[str, Any] = {
        "schema_version": 1,
        "core": {"distribution": "ai-harness", "version": core_version, "source": "bundled"},
        "plugin": {"id": plugin_id, "version": str(plugin.version), "distribution": None, "install": None},
        "requirements": core_requirements,
        "bundle_files": [],
    }
    if plugin_id == "default":
        return runtime

    declared = plugin.remote_runtime_manifest(context)
    if not isinstance(declared, dict):
        raise ValueError(
            f"Task plugin `{plugin_id}` must implement remote_runtime_manifest() before Kaggle or Modal execution."
        )
    distribution = declared.get("distribution")
    version = declared.get("version") or plugin.version
    if not distribution or not version:
        raise ValueError(
            f"Task plugin `{plugin_id}` remote runtime must pin its installable distribution and version."
        )
    installed = {item["id"]: item for item in list_task_plugins()}.get(plugin_id, {})
    expected_distribution = str(installed.get("distribution") or "").casefold().replace("_", "-")
    if expected_distribution and str(distribution).casefold().replace("_", "-") != expected_distribution:
        raise ValueError(
            f"Remote plugin distribution `{distribution}` does not match installed entry point distribution `{installed.get('distribution')}`."
        )
    requirements = declared.get("requirements", plugin.remote_runtime_requirements(context))
    if not isinstance(requirements, list) or any(not isinstance(item, str) or not item.strip() for item in requirements):
        raise ValueError("Remote runtime requirements must be a list of non-empty pip requirement strings.")
    invalid_requirements = [
        item for item in requirements
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9.+_-]*", item.strip())
    ]
    if invalid_requirements:
        raise ValueError(
            "Remote plugin dependencies must use exact package==version pins: "
            + ", ".join(invalid_requirements)
        )
    wheel = declared.get("wheel")
    if wheel:
        wheel_path = Path(wheel).expanduser().resolve()
        if not wheel_path.is_file() or wheel_path.suffix.lower() != ".whl":
            raise ValueError(f"Remote adapter wheel must be an existing .whl file: {wheel}")
        runtime["plugin"]["install"] = {"type": "wheel", "path": str(wheel_path)}
    else:
        runtime["plugin"]["install"] = {"type": "index", "requirement": f"{distribution}=={version}"}
    bundle_files = declared.get("bundle_files", plugin.remote_bundle_files(context))
    if not isinstance(bundle_files, list):
        raise ValueError("Remote runtime `bundle_files` must be a list of paths.")
    normalized_files = []
    for value in bundle_files:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Declared remote runtime file does not exist: {path}")
        try:
            path.relative_to(project.root)
        except ValueError as exc:
            raise ValueError(f"Declared adapter bundle file must stay inside the task workspace: {path}") from exc
        if any(part.casefold() in {".env", ".kaggle", ".modal", ".git", "artifacts"} for part in path.parts):
            raise ValueError(f"Declared adapter bundle file is in a protected path: {path}")
        normalized_files.append(str(path))
    runtime["plugin"].update(distribution=str(distribution), version=str(version))
    runtime["requirements"] = [*core_requirements, *[item.strip() for item in requirements]]
    runtime["bundle_files"] = normalized_files
    return runtime
