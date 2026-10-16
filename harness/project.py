"""Explicit workspace discovery and attachment for local AI-Harness installs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Optional

import yaml


MANIFEST_RELATIVE_PATH = Path(".ai-harness") / "project.yaml"
LEGACY_MANIFEST_RELATIVE_PATH = Path(".ml-harness") / "project.yaml"
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
    if (cwd / MANIFEST_RELATIVE_PATH).is_file() or (cwd / LEGACY_MANIFEST_RELATIVE_PATH).is_file():
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

    root: Path = Path(".")
    manifest_path: Optional[Path] = None
    task_config_path: Optional[Path] = None
    plugin_id: Optional[str] = None
    artifacts_dir: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        project_root = resolve_project_root(self.root)
        object.__setattr__(self, "root", project_root)

        if self.manifest_path is None or self.manifest is None:
            m_path = self.manifest_path
            if m_path is None:
                m_path = project_root / MANIFEST_RELATIVE_PATH
                if not m_path.is_file() and (project_root / LEGACY_MANIFEST_RELATIVE_PATH).is_file():
                    m_path = project_root / LEGACY_MANIFEST_RELATIVE_PATH
            m_data = self.manifest if self.manifest is not None else _read_yaml(m_path)
            object.__setattr__(self, "manifest_path", m_path)
            object.__setattr__(self, "manifest", m_data)

        if self.task_config_path is None:
            config_val = (self.manifest or {}).get("task_config") or str(DEFAULT_TASK_CONFIG)
            cfg_path = Path(config_val).expanduser()
            if not cfg_path.is_absolute():
                cfg_path = (project_root / cfg_path).resolve()
            object.__setattr__(self, "task_config_path", cfg_path)
        else:
            cfg_path = Path(self.task_config_path).expanduser()
            if not cfg_path.is_absolute():
                cfg_path = (project_root / cfg_path).resolve()
            object.__setattr__(self, "task_config_path", cfg_path)

        if self.plugin_id is None:
            task_document = _read_yaml(self.task_config_path) if self.task_config_path.is_file() else {}
            task = task_document.get("task", task_document)
            configured_plugin = (
                (self.manifest or {}).get("plugin_id")
                or task.get("plugin_id")
                or task.get("plugin")
            )
            if isinstance(configured_plugin, dict):
                configured_plugin = configured_plugin.get("id") or configured_plugin.get("name")
            object.__setattr__(
                self, "plugin_id", str(configured_plugin) if configured_plugin else None
            )

        if self.artifacts_dir is None:
            default_artifacts = ".ai-harness/artifacts"
            artifacts_value = (self.manifest or {}).get(
                "artifacts_dir", default_artifacts if self.manifest else "artifacts"
            )
            art_dir = Path(artifacts_value).expanduser()
            if not art_dir.is_absolute():
                art_dir = (project_root / art_dir).resolve()
            object.__setattr__(self, "artifacts_dir", art_dir)
        else:
            art_dir = Path(self.artifacts_dir).expanduser()
            if not art_dir.is_absolute():
                art_dir = (project_root / art_dir).resolve()
            object.__setattr__(self, "artifacts_dir", art_dir)

    @classmethod
    def discover(
        cls,
        root: Optional[Path | str] = None,
        task_config: Optional[Path | str] = None,
    ) -> "HarnessProject":
        project_root = resolve_project_root(root)
        manifest_path = project_root / MANIFEST_RELATIVE_PATH
        if not manifest_path.is_file() and (project_root / LEGACY_MANIFEST_RELATIVE_PATH).is_file():
            manifest_path = project_root / LEGACY_MANIFEST_RELATIVE_PATH
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


AGENTS_MD_MARKER_START = "<!-- ai-harness:managed:start -->"
AGENTS_MD_MARKER_END = "<!-- ai-harness:managed:end -->"


def render_agents_md_body() -> str:
    """Universal entry point telling ANY coding agent about this harness.

    Skills under ``.agents/skills/`` are only loaded by agent harnesses that
    support them AND when the task matches; a plain "train a model" request
    to a generic agent otherwise finds no instruction at all and silently
    bypasses the framework (observed in the rsna-knee workspace: 22/22
    manual kernel pushes, zero harness experiments, zero warnings).
    """
    parts = [
        "## AI-Harness workspace (managed block - do not edit between markers)",
        "",
        "This workspace runs ML experiments through **ai-harness** (framework clone: `./ai-harness`). "
        "All experiment work goes through the harness - there is no supported alternative.",
        "",
        "1. **Read the SOP first**: `.agents/skills/ai-harness/SKILL.md` (generated per task, always current).",
        "2. **Current state** (~1k tokens): `ai-harness-agent --project-root . context`",
        "3. **Experiment lifecycle**: declarative config in `configs/` -> smoke probe "
        "(`ai-harness-run --config configs/<exp>.yaml --debug --fold 0`) -> full validation "
        "(`ai-harness-run --config configs/<exp>.yaml`). Every run is journaled under `.ai-harness/artifacts/`.",
        "",
        "**Non-negotiable rules:**",
        "- Cloud runs **only** via harness dispatch (`run_full_validation` / `ai-harness kaggle`). "
        "Never push kernels manually (`kaggle kernels push`) - manual runs have no evidence trail "
        "and the framework cannot see them.",
        "- If harness dispatch fails, **report the error** - do not silently fall back to a manual run.",
        "- Directories under `.ai-harness/artifacts/` are immutable: never mutate completed runs.",
        "- After each iteration, record research continuity through the Bridge "
        "(`record_research_insight` / `record_research_decision` / `record_research_checkpoint`; "
        "contracts are documented in the skill).",
        "- Framework upgrades: `ai-harness sync` (one command) - never hand-run git steps in `./ai-harness`.",
    ]
    return chr(10).join(parts)

def update_agents_md(root: Path) -> Dict[str, Any]:
    """Create or refresh the managed AGENTS.md block; never touch user content."""
    agents_md = root / "AGENTS.md"
    nl = chr(10)
    body = AGENTS_MD_MARKER_START + nl + render_agents_md_body() + AGENTS_MD_MARKER_END
    status = "created"
    if agents_md.exists():
        existing = agents_md.read_text(encoding="utf-8")
        if AGENTS_MD_MARKER_START in existing and AGENTS_MD_MARKER_END in existing:
            start, end = existing.index(AGENTS_MD_MARKER_START), existing.index(AGENTS_MD_MARKER_END)
            agents_md.write_text(
                existing[:start] + body + existing[end + len(AGENTS_MD_MARKER_END):],
                encoding="utf-8",
            )
            status = "refreshed"
        else:
            return {"path": str(agents_md), "status": "skipped_user_owned"}
    else:
        agents_md.write_text(body + nl, encoding="utf-8")
    return {"path": str(agents_md), "status": status}


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
            agents_md = update_agents_md(root)
            return {
                "status": "ALREADY_ATTACHED",
                "agents_md": agents_md["status"],
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

    # Single source of truth for the task skill: skill_injector. The legacy
    # inline block below it was a second, leaner generator that `ai-harness
    # attach` wrote and `ai-harness sync` silently overwrote — agents
    # attached directly (without sync) missed the param-file guidance,
    # script adapter, cloud-monitoring and research-contract sections.
    # Pin project discovery: the skill injector resolves the workspace from
    # cwd/env, not from this function's arguments (same pinning as sync).
    import os as _os
    prev_root_env = _os.environ.get("HARNESS_PROJECT_ROOT")
    _os.environ["HARNESS_PROJECT_ROOT"] = str(root)
    try:
        from harness.skill_injector import inject_skill_file
        skill_path = inject_skill_file(config_path=task_path)
    finally:
        if prev_root_env is None:
            _os.environ.pop("HARNESS_PROJECT_ROOT", None)
        else:
            _os.environ["HARNESS_PROJECT_ROOT"] = prev_root_env

    for legacy_name in ("ml-harness", "ml-experiment-harness"):
        legacy_dir = root / ".agents" / "skills" / legacy_name
        if legacy_dir.is_dir():
            shutil.rmtree(legacy_dir, ignore_errors=True)

    agents_md = update_agents_md(root)
    return {
        "status": "ATTACHED",
        "agents_md": agents_md["status"],
        "project_root": str(root),
        "manifest": str(manifest_path),
        "agent_skill": str(skill_path),
        "plugin_id": str(resolved_plugin),
        "task_id": task_id,
        "task_config": task_rel,
        "prompt_pack": prompt_pack,
    }
