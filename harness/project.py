"""Explicit workspace discovery and attachment for local AI-Harness installs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

import yaml


MANIFEST_RELATIVE_PATH = Path(".ai-harness") / "project.yaml"
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

        # An explicit workspace root must not inherit another workspace's task-config
        # environment override. The attachment manifest is authoritative for that root.
        if task_config:
            config_value = task_config
        elif root is not None:
            config_value = manifest.get("task_config") or str(DEFAULT_TASK_CONFIG)
        else:
            config_value = (
                manifest.get("task_config")
                or os.environ.get("HARNESS_TASK_CONFIG")
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

        default_artifacts = ".ai-harness/artifacts"
        artifacts_value = manifest.get("artifacts_dir", default_artifacts if manifest else "artifacts")
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
) -> Dict[str, Any]:
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
    skill_path = root / ".agents" / "skills" / "ai-harness" / "SKILL.md"

    if manifest_path.exists() and not force:
        current = _read_yaml(manifest_path)
        same_attachment = (
            Path(str(current.get("task_config", ""))).as_posix() == Path(task_rel).as_posix()
            and current.get("plugin_id", "default") == resolved_plugin
        )
        if same_attachment:
            from harness.phase_workflow import initialize_prompt_pack
            prompt_pack = initialize_prompt_pack(root, force=False)
            return {
                "status": "ALREADY_ATTACHED",
                "project_root": str(root),
                "manifest": str(manifest_path),
                "agent_skill": str(skill_path),
                "plugin_id": str(resolved_plugin),
                "task_id": task_id,
                "prompt_pack": prompt_pack,
            }
        if not same_attachment:
            raise FileExistsError(f"Harness manifest already exists: {manifest_path}; pass --force to replace it.")

    from harness.phase_workflow import initialize_prompt_pack
    prompt_pack = initialize_prompt_pack(root, force=False)

    manifest = {
        "schema_version": 1,
        "task_id": task_id,
        "task_config": task_rel,
        "plugin_id": str(resolved_plugin),
        "artifacts_dir": f".ai-harness/artifacts/{task_id}",
        "leaderboard_path": f".ai-harness/artifacts/{task_id}/EXPERIMENTS.md",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    skill_content = (
        "---\n"
        "name: ai-harness\n"
        "description: Work with this competition through the installed AI-Harness task plugin and Agent Bridge.\n"
        "---\n\n"
        f"# AI-Harness for {task_name}\n\n"
        "Use the task's declared plugin for data loading, validation, metrics, and submission checks. "
        "Do not infer or replace the official scorer or submission contract.\n\n"
        "## Agent workflow\n\n"
        f"- Workspace: `{root}`\n"
        f"- Task config: `{task_rel}`\n"
        f"- Task plugin: `{resolved_plugin}`\n"
        "- Read current state: `ai-harness-agent --project-root . context`\n"
        "- After restart or context compaction, recover the latest research checkpoint before brainstorming.\n"
        "- Search the full history with `act --action search_research_memory --params '{\"query\":\"mechanism keywords\"}'`.\n"
        "- Before implementation, use `act --action review_hypothesis` with a config object or config_path.\n"
        "- Explain differences from related approaches; exact repeats require an explicit revisit_reason.\n"
        "- Record each insight with record_research_insight (title, observation, mechanism, next_action). "
        "Use parent_insight_id when a new direction follows an older lesson.\n"
        "- Before linked implementation, plan_research_probe: control, one intervention, expected_observation, "
        "alternative_explanation, decision_rule, budget, stop_rule. Link config.research.insight_id.\n"
        "- Use trace_research_insight to recover its plans, experiments, decisions and derived directions. "
        "Interpret research_frontier and unreviewed_results before rebrainstorming.\n"
        "- Read the generated RESEARCH_PROGRESS.md at context.research_memory.progress_path for a human-readable "
        "journal. Record meaningful observations/decisions through the Bridge after each iteration, not every keystroke.\n"
        "- After each result, use record_research_decision (outcome, conclusion, evidence run IDs, next_action) "
        "with insight_id, lesson and revisit_when for rejected/deferred directions; "
        "then record_research_checkpoint (summary, next_action, open_questions, insight_ids).\n"
        "- On stagnation, inspect verified diagnostics and old lessons, compare alternative mechanisms, "
        "then choose a bounded discriminating probe with a promotion/stop criterion. Plateau never authorizes freeze.\n"
        "- Review the task contract: `ai-harness-agent --project-root . act --action confirm_task_spec --user-approved`\n"
        "- Inspect data: `ai-harness-agent --project-root . act --action inspect_data_readiness`\n"
        "- Confirm reviewed data risks: `ai-harness-agent --project-root . act --action confirm_data_risks --user-approved`\n"
        "- Start a bounded session: `ai-harness-agent --project-root . start-session --goal \"...\" --max-experiments 3`\n"
        "- Run a smoke test before a full validation run.\n"
        "- Render the next phase prompt: `ai-harness phase render`\n"
        "- Validate the workspace prompt pack: `ai-harness phase validate`\n"
        "- Record response, inspected source URLs, and artifact paths with `ai-harness phase record`.\n"
        "- Run privacy-conscious EDA: `ai-harness eda run`\n"
        "- Use the IDE agent's native web search and cite only sources actually inspected.\n"
    )
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(skill_content, encoding="utf-8")

    return {
        "status": "ATTACHED",
        "project_root": str(root),
        "manifest": str(manifest_path),
        "agent_skill": str(skill_path),
        "plugin_id": str(resolved_plugin),
        "task_id": task_id,
        "task_config": task_rel,
        "prompt_pack": prompt_pack,
    }
