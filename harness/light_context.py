"""Minimal task context available without the numerical ML extra."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from harness.phase_workflow import PhaseWorkflow
from harness.project import HarnessProject


_CONTEXT_FIELDS = (
    "description",
    "problem_statement",
    "competition_context",
    "competition_rules",
    "evaluation",
    "data_config",
    "submission_spec",
    "constraints",
)


def build_light_context(
    project_root: Path | str = ".",
    task_config: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build workspace and phase context using only the base dependencies."""
    project = HarnessProject.discover(project_root, task_config)
    task_document = yaml.safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
    task = task_document.get("task", task_document)
    phase_status = PhaseWorkflow(project.root, project.task_config_path).phase_status()

    return {
        "profile": "workflow",
        "workspace": str(project.root),
        "task_config": project.task_config_path.relative_to(project.root).as_posix(),
        "task": {
            key: task[key]
            for key in ("name", "task_name", "primary_metric", "direction", "modality", *_CONTEXT_FIELDS)
            if key in task and task[key] not in (None, "", [], {})
        },
        "plugin_id": project.plugin_id or "default",
        "artifacts_dir": str(project.artifacts_dir),
        "phase_status": phase_status,
        "capabilities": {
            "available": ["attach", "prompt_phases", "basic_context", "plugin_discovery"],
            "requires_ml_extra": [
                "eda",
                "experiment_execution",
                "agent_bridge_actions",
                "diagnostics",
                "task_bootstrap",
                "submission_audit",
                "dashboard",
            ],
        },
        "note": "Install the ml extra to enable EDA, experiment execution, full Agent Bridge context and actions, task bootstrap, submission auditing, and the dashboard.",
    }


def render_light_context(context: Dict[str, Any]) -> str:
    """Render a concise Markdown context packet for a workflow-only install."""
    task = context.get("task", {})
    lines = [
        "# ML-Harness workspace context",
        "",
        f"- Workspace: {context.get('workspace')}",
        f"- Task config: {context.get('task_config')}",
        f"- Task: {task.get('name') or task.get('task_name') or 'Unnamed task'}",
        f"- Plugin: {context.get('plugin_id')}",
        f"- Primary metric: {task.get('primary_metric', 'not declared')} ({task.get('direction', 'not declared')})",
        f"- Next phase: {context.get('phase_status', {}).get('next_phase') or 'initial pass complete'}",
        f"- Artifacts: {context.get('artifacts_dir')}",
        "",
        "## Phase progress",
    ]
    for phase in context.get("phase_status", {}).get("phases", []):
        state = "complete" if phase.get("completed") else "upcoming"
        if phase.get("can_revisit"):
            state += "; new evidence allows revisit"
        lines.append(f"- {phase.get('id')}: {state} ({phase.get('runs', 0)} response(s))")

    extra_context = {key: task[key] for key in _CONTEXT_FIELDS if key in task}
    if extra_context:
        lines.extend(["", "## Competition context", yaml.safe_dump(extra_context, sort_keys=False, allow_unicode=True).strip()])

    lines.extend(["", context.get("note", "")])
    return "\n".join(lines) + "\n"
