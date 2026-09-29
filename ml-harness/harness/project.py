"""Explicit workspace discovery and attachment for local ML-Harness installs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

import yaml


MANIFEST_RELATIVE_PATH = Path(".ml-harness") / "project.yaml"
DEFAULT_TASK_CONFIG = Path("configs") / "task.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def resolve_project_root(root: Optional[Path | str] = None) -> Path:
    """Resolve a task workspace without relying on the installed package path."""
    configured = root or os.environ.get("HARNESS_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / MANIFEST_RELATIVE_PATH).is_file():
        return cwd
    if (cwd / DEFAULT_TASK_CONFIG).is_file() or (cwd / "dataset").exists():
        return cwd

    package_root = Path(__file__).resolve().parent.parent
    if (package_root / DEFAULT_TASK_CONFIG).is_file():
        return package_root
    return cwd


@dataclass(frozen=True)
class HarnessProject:
    """Resolved task workspace settings shared by the CLI, runner, and agents."""

    root: Path
    manifest_path: Path
    task_config_path: Path
    plugin_id: Optional[str]
    artifacts_dir: Path
    manifest: Dict[str, Any]

    @classmethod
    def discover(
        cls,
        root: Optional[Path | str] = None,
        task_config: Optional[Path | str] = None,
    ) -> "HarnessProject":
        project_root = resolve_project_root(root)
        manifest_path = project_root / MANIFEST_RELATIVE_PATH
        manifest = _read_yaml(manifest_path)

        config_value = (
            task_config
            or os.environ.get("HARNESS_TASK_CONFIG")
            or manifest.get("task_config")
            or str(DEFAULT_TASK_CONFIG)
        )
        config_path = Path(config_value).expanduser()
        if not config_path.is_absolute():
            config_path = project_root / config_path
        config_path = config_path.resolve()

        task_document = _read_yaml(config_path)
        task = task_document.get("task", task_document)
        configured_plugin = manifest.get("plugin_id") or task.get("plugin_id") or task.get("plugin")
        if isinstance(configured_plugin, dict):
            configured_plugin = configured_plugin.get("id") or configured_plugin.get("name")

        artifacts_value = manifest.get("artifacts_dir", ".ml-harness/artifacts" if manifest else "artifacts")
        artifacts_dir = Path(artifacts_value).expanduser()
        if not artifacts_dir.is_absolute():
            artifacts_dir = project_root / artifacts_dir

        return cls(
            root=project_root,
            manifest_path=manifest_path,
            task_config_path=config_path,
            plugin_id=str(configured_plugin) if configured_plugin else None,
            artifacts_dir=artifacts_dir.resolve(),
            manifest=manifest,
        )


def attach_project(
    project_root: Path | str,
    task_config: Path | str = DEFAULT_TASK_CONFIG,
    plugin_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Attach an installed harness to a task project and add an agent skill."""
    root = Path(project_root).expanduser().resolve()
    task_path = Path(task_config).expanduser()
    if not task_path.is_absolute():
        task_path = root / task_path
    task_path = task_path.resolve()
    if not task_path.is_file():
        raise FileNotFoundError(f"Task config does not exist: {task_path}")

    task_document = _read_yaml(task_path)
    task_data = task_document.get("task", task_document)
    resolved_plugin = plugin_id or task_data.get("plugin_id") or task_data.get("plugin") or "default"
    if isinstance(resolved_plugin, dict):
        resolved_plugin = resolved_plugin.get("id") or resolved_plugin.get("name") or "default"
    task_name = task_data.get("name") or task_data.get("task_name") or "Competition task"
    task_id = re.sub(r"[^a-z0-9]+", "-", str(task_data.get("id") or task_name).lower()).strip("-") or "task"

    try:
        task_rel = task_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Task config must be inside the attached project root.") from exc

    manifest_path = root / MANIFEST_RELATIVE_PATH
    skill_path = root / ".agents" / "skills" / "ml-harness" / "SKILL.md"
    if manifest_path.exists() and not force:
        current = _read_yaml(manifest_path)
        same_attachment = (
            Path(str(current.get("task_config", ""))).as_posix() == task_rel
            and current.get("plugin_id", "default") == resolved_plugin
        )
        if same_attachment and skill_path.is_file():
            return {
                "status": "ALREADY_ATTACHED",
                "project_root": str(root),
                "manifest": str(manifest_path),
                "agent_skill": str(skill_path),
                "plugin_id": str(resolved_plugin),
            }
        if not same_attachment:
            raise FileExistsError(f"Harness manifest already exists: {manifest_path}; pass --force to replace it.")

    if skill_path.exists() and not force:
        raise FileExistsError(f"Agent skill already exists: {skill_path}; pass --force to replace it.")

    manifest = {
        "schema_version": 1,
        "task_config": task_rel,
        "plugin_id": str(resolved_plugin),
        "artifacts_dir": f".ml-harness/artifacts/{task_id}",
        "leaderboard_path": f".ml-harness/artifacts/{task_id}/EXPERIMENTS.md",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        "---\n"
        "name: ml-harness\n"
        "description: Work with this competition through the installed ML-Harness task plugin and Agent Bridge.\n"
        "---\n\n"
        f"# ML-Harness for {task_name}\n\n"
        "Use the task's declared plugin for data loading, validation, metrics, and submission checks. "
        "Do not infer or replace the official scorer or submission contract.\n\n"
        "## Agent workflow\n\n"
        f"- Workspace: `{root}`\n"
        f"- Task config: `{task_rel}`\n"
        f"- Task plugin: `{resolved_plugin}`\n"
        "- Read current state: `ml-harness-agent --project-root . context`\n"
        "- Review the task contract: `ml-harness-agent --project-root . act --action confirm_task_spec --user-approved`\n"
        "- Inspect data: `ml-harness-agent --project-root . act --action inspect_data_readiness`\n"
        "- Confirm reviewed data risks: `ml-harness-agent --project-root . act --action confirm_data_risks --user-approved`\n"
        "- Start a bounded session: `ml-harness-agent --project-root . start-session --goal \"...\" --max-experiments 3`\n"
        "- Run a smoke test before a full validation run.\n",
        encoding="utf-8",
    )
    return {
        "status": "ATTACHED",
        "project_root": str(root),
        "manifest": str(manifest_path),
        "agent_skill": str(skill_path),
        "plugin_id": str(resolved_plugin),
        "task_config": task_rel,
    }
