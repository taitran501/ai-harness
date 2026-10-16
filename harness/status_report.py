"""`ai-harness status` — single-command attach health + workspace state.

Designed to be the first command an agent runs in a fresh conversation.
Returns JSON (default) or a short human-readable summary so the agent does
not have to re-discover install location, active project, session state,
artifact integrity, and cloud readiness via multiple probing tasks.

Failure mode: never raises. Every section is wrapped in its own try/except
so one bad section does not hide the rest (an agent diagnosing an install
conflict still needs the other sections).
"""

from __future__ import annotations

import importlib.metadata as _importlib_metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


HARNESS_PACKAGE_CANDIDATES = ("harness", "ai_harness", "ml_harness")


def _installed_info() -> Dict[str, Any]:
    """Locate the harness package actually in sys.path and its origin."""
    for name in HARNESS_PACKAGE_CANDIDATES:
        try:
            module = __import__(name, fromlist=["__path__"])
        except Exception as exc:  # pragma: no cover - defensive
            return {"package": name, "found": False, "error": str(exc)}
        path = getattr(module, "__file__", None)
        if not path:
            continue
        path = Path(path).resolve()
        top = path.parent.parent  # site-packages/harness/__init__.py -> site-packages
        in_site_packages = "site-packages" in str(top).replace("\\", "/")
        return {
            "package": name,
            "found": True,
            "file": str(path),
            "origin": "site-packages" if in_site_packages else "repo-editable-or-local",
            "version": _importlib_metadata.version("ai-harness")
            if _installed_dist("ai-harness")
            else None,
        }
    return {"found": False, "error": "no harness module importable"}


def _installed_dist(name: str) -> bool:
    try:
        return _importlib_metadata.version(name) is not None
    except Exception:
        return False


def _section(fn, *args) -> Dict[str, Any]:
    """Run a status section; isolate failures so one bad section does not hide the rest."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - status must never raise
        return {"error": f"{type(exc).__name__}: {exc}"}


def _project_section(project_root: Optional[str]) -> Dict[str, Any]:
    from harness.project import HarnessProject, MANIFEST_RELATIVE_PATH, LEGACY_MANIFEST_RELATIVE_PATH

    root = Path(project_root or ".").expanduser().resolve()
    attached = (root / MANIFEST_RELATIVE_PATH).is_file() or (root / LEGACY_MANIFEST_RELATIVE_PATH).is_file()
    if not attached:
        return {"attached": False, "error": f"no ai-harness manifest found under {root}", "root": str(root)}
    project = HarnessProject.discover(project_root)
    manifest = project.manifest or {}
    task = manifest.get("task") or {}
    artifacts_dir = project.artifacts_dir if hasattr(project, "artifacts_dir") else "artifacts"
    return {
        "attached": True,
        "root": str(Path(project.root).resolve()) if getattr(project, "root", None) else str(Path(project_root or ".").resolve()),
        "manifest": str(project.manifest_path) if getattr(project, "manifest_path", None) else None,
        "task_config": manifest.get("task_config") or task.get("task_config"),
        "plugin_id": manifest.get("plugin_id") or task.get("plugin_id") or task.get("plugin"),
        "artifacts_dir": str(artifacts_dir),
    }


def _plugins_section(project_root: Optional[str]) -> Dict[str, Any]:
    """Installed task plugins, the active plugin, and the union of model names.

    After the B16 refactor, which model names a workspace accepts depends on
    its active plugin; this section makes that discoverable from one health
    command instead of trial-and-error.
    """
    from harness.models import available_model_names
    from harness.plugins import list_task_plugins, load_task_plugin
    from harness.project import HarnessProject

    plugins = list_task_plugins()
    built_in = available_model_names()
    plugin_names: Dict[str, List[str]] = {}
    active_plugin_id = None
    project: Optional[HarnessProject] = None
    try:
        project = HarnessProject.discover(project_root or ".")
        active_plugin_id = project.plugin_id
    except Exception:  # noqa: BLE001 - not attached: report install only
        project = None
    active_names = built_in
    if project is not None and active_plugin_id and active_plugin_id != "default":
        try:
            from harness.plugins import TaskPluginContext
            from harness.task_spec import load_active_task_spec

            spec = load_active_task_spec(project.task_config_path)
            ctx = TaskPluginContext(
                project.root, project.task_config_path, spec, active_plugin_id
            )
            plugin = load_task_plugin(active_plugin_id, ctx)
            names = getattr(plugin, "model_names", None) or getattr(
                type(plugin), "model_names", None
            )
            if names:
                plugin_names[active_plugin_id] = sorted(str(n) for n in names)
                active_names = sorted(set(built_in) | set(plugin_names[active_plugin_id]))
        except Exception:  # noqa: BLE001 - plugin broken: surface the error
            plugin_names[active_plugin_id] = {
                "error": "active plugin failed to load; run `ai-harness plugins`"
            }
    return {
        "installed": plugins,
        "active_plugin_id": active_plugin_id,
        "model_names": active_names,
        "plugin_model_names": plugin_names,
    }


def _session_section(project_root: Optional[str]) -> Dict[str, Any]:
    """Latest agent-bridge session, if any. Cheap: only reads session JSON files."""
    from harness.project import HarnessProject, MANIFEST_RELATIVE_PATH, LEGACY_MANIFEST_RELATIVE_PATH

    root = Path(project_root or ".").expanduser().resolve()
    if not ((root / MANIFEST_RELATIVE_PATH).is_file() or (root / LEGACY_MANIFEST_RELATIVE_PATH).is_file()):
        return {"active_session": None, "sessions_total": 0, "error": "project not attached"}
    project = HarnessProject.discover(project_root)
    root = Path(project.root if getattr(project, "root", None) else (project_root or "."))
    artifacts_dir = root / (project.artifacts_dir if hasattr(project, "artifacts_dir") else "artifacts")
    sessions_dir = artifacts_dir / "agent_sessions"
    if not sessions_dir.is_dir():
        return {"active_session": None, "sessions_total": 0}
    session_files = sorted(sessions_dir.glob("session_*.json"))
    if not session_files:
        return {"active_session": None, "sessions_total": 0}
    latest = session_files[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    slim = {
        k: data.get(k)
        for k in (
            "session_id",
            "status",
            "goal",
            "consecutive_rejections",
            "compute_usage",
            "experiments_run_count",
        )
    }
    if slim.get("goal"):
        slim["goal"] = str(slim["goal"])[:300]
    runs = data.get("experiments_run") or []
    slim["experiments_run_last5"] = runs[-5:]
    return {"active_session": slim, "sessions_total": len(session_files)}


def _artifact_health_section(project_root: Optional[str]) -> Dict[str, Any]:
    """Count verified / drift-repairable / unverifiable runs under artifacts/.

    Distinguishes the three states so an agent knows when `ai-harness repair`
    will help (drift) versus when it won't (unverifiable)."""
    from harness.project import HarnessProject, MANIFEST_RELATIVE_PATH, LEGACY_MANIFEST_RELATIVE_PATH
    from harness.repair import scan_artifacts

    root = Path(project_root or ".").expanduser().resolve()
    if not ((root / MANIFEST_RELATIVE_PATH).is_file() or (root / LEGACY_MANIFEST_RELATIVE_PATH).is_file()):
        return {
            "runs_total": 0, "verified": 0, "drifted_repairable": 0, "unverifiable": 0,
            "broken": [], "repair_hint": None, "error": "project not attached",
        }
    project = HarnessProject.discover(project_root)
    root = Path(project.root if getattr(project, "root", None) else (project_root or "."))
    artifacts_dir = root / (project.artifacts_dir if hasattr(project, "artifacts_dir") else "artifacts")
    if not artifacts_dir.is_dir():
        return {
            "runs_total": 0, "verified": 0, "drifted_repairable": 0, "unverifiable": 0,
            "broken": [], "repair_hint": None, "error": "artifacts dir missing",
        }

    scan = scan_artifacts(artifacts_dir)
    verified = len(scan["verified"])
    drifted = scan["drifted"]
    unverifiable = scan["unverifiable"]
    unmanifested = _count_unverified(artifacts_dir)
    runs_total = verified + len(drifted) + len(unverifiable) + unmanifested

    # `broken` keeps the old shape (top-5) for backward compatibility, but now
    # separates drift-repairable from genuinely unverifiable runs.
    broken: List[Dict[str, Any]] = []
    for item in drifted:
        if len(broken) < 5:
            broken.append({"exp_dir": item["exp_dir"], "kind": "drifted", "reason": item["reason"]})
    for item in unverifiable:
        if len(broken) < 5:
            broken.append({"exp_dir": item["exp_dir"], "kind": "unverifiable", "reason": item.get("reason", "")})

    repair_hint = None
    if drifted:
        repair_hint = (
            f"{len(drifted)} run(s) have artifact hash drift — "
            f"`ai-harness repair --check` previews the fix, `ai-harness repair --all` re-seals them."
        )
    return {
        "runs_total": runs_total,
        "verified": verified,
        "drifted_repairable": len(drifted),
        "unverifiable": len(unverifiable) + unmanifested,
        "broken": broken,
        "repair_hint": repair_hint,
    }


def _count_unverified(artifacts_dir: Path) -> int:
    """Count run folders lacking a manifest (they are invisible to validate_run)."""
    total = sum(1 for p in artifacts_dir.iterdir() if p.is_dir())
    with_manifest = sum(1 for p in artifacts_dir.iterdir() if p.is_dir() and (p / "artifact_manifest.json").is_file())
    return max(0, total - with_manifest)


def _cloud_section(project_root: Optional[str]) -> Dict[str, Any]:
    from harness.cloud_policy import UploadConsentStore
    from harness.environment_detector import EnvironmentDetector
    from harness.project import HarnessProject

    env = EnvironmentDetector.get_full_status()
    # keep only presence flags, not credentials.
    # Each provider reports readiness under different key names; using the
    # wrong key made an authenticated kaggle CLI report as unavailable,
    # which silently pushed agents to manual `kaggle kernels push`
    # (observed in the rsna-knee workspace: 22/22 manual run-output dirs).
    availability_keys = {
        "modal": ("installed", "has_credentials"),
        "kaggle": ("installed", "authenticated"),
        "docker": ("installed", "running"),
        "gpu": ("detected",),
        "llm": ("total_configured",),
    }
    accounts = {}
    for k, v in env.items():
        if isinstance(v, dict):
            keys = availability_keys.get(k, ("available", "detected"))
            accounts[k] = {"available": all(bool(v.get(key)) for key in keys)}
        else:
            accounts[k] = {"available": bool(v)}
    try:
        project = HarnessProject.discover(project_root)
        consent = UploadConsentStore(project).status()
    except Exception:  # noqa: BLE001 - consent requires attachment
        consent = None
    return {"accounts": accounts, "upload_consent": consent}


def _adoption_section(project_root: Optional[str]) -> Dict[str, Any]:
    """T5: adoption health — the human's at-a-glance bypass audit.

    The framework can only measure what flows through it; bypassed runs
    leave no trace by design. This section makes the blind spot visible:
    harness experiments vs manual run-output dirs vs agent sessions.
    """
    root = Path(project_root) if project_root else Path.cwd()
    if not (root / ".ai-harness").is_dir():
        return {"status": "not_attached"}
    artifacts = root / ".ai-harness" / "artifacts"
    experiments = 0
    quarantined_experiments = 0
    recovered_dirs = 0
    if artifacts.is_dir():
        for task_dir in artifacts.iterdir():
            if not task_dir.is_dir():
                continue
            for run in task_dir.iterdir():
                if run.is_dir() and (run / "artifact_manifest.json").is_file():
                    experiments += 1
            quar_dir = task_dir / "quarantine"
            if quar_dir.is_dir():
                for run in quar_dir.iterdir():
                    if run.is_dir() and (run / "artifact_manifest.json").is_file():
                        quarantined_experiments += 1
            try:
                src_payload = json.loads((task_dir / "imported_sources.json").read_text(encoding="utf-8"))
                if isinstance(src_payload, list):
                    recovered_dirs += sum(1 for r in src_payload if isinstance(r, dict))
            except (OSError, json.JSONDecodeError):
                pass
    sessions_total = 0
    sessions_dir = root / ".ai-harness" / "agent_sessions"
    if not sessions_dir.is_dir() and artifacts.is_dir():
        for task_dir in artifacts.iterdir():
            cand = task_dir / "agent_sessions"
            if cand.is_dir():
                sessions_dir = cand
                break
    if sessions_dir.is_dir():
        sessions_total = len(list(sessions_dir.glob("session_*.json")))
    bypassed = 0
    bypassed_dirs = 0
    try:
        from harness.hygiene import scan_workspace_hygiene

        findings = scan_workspace_hygiene(root).get("findings", [])
        bypass_findings = [f for f in findings if f.get("kind") == "bypassed_run_output"]
        bypassed = len(bypass_findings)
        bypassed_dirs = sum(int(f.get("dir_count") or 0) for f in bypass_findings)
    except Exception:  # noqa: BLE001 - advisory only
        bypassed = -1
    if bypassed > 0:
        verdict = "bypass_detected"
    elif experiments > 0:
        verdict = "healthy"
    else:
        verdict = "no_activity"
    return {
        "status": "ok",
        "experiments": experiments,
        "quarantined_experiments": quarantined_experiments,
        "bypassed_run_outputs": bypassed,
        "bypassed_run_output_dirs": bypassed_dirs,
        "recovered_run_output_dirs": recovered_dirs,
        "sessions": sessions_total,
        "verdict": verdict,
    }


def collect_status(project_root: Optional[str] = None, include_cloud: bool = True) -> Dict[str, Any]:
    """Assemble the full status payload. Never raises."""
    status: Dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "install": _section(_installed_info),
        "project": _section(_project_section, project_root),
        "plugins": _section(_plugins_section, project_root),
        "adoption": _section(_adoption_section, project_root),
        "session": _section(_session_section, project_root),
        "artifacts": _section(_artifact_health_section, project_root),
    }
    if include_cloud:
        status["cloud"] = _section(_cloud_section, project_root)
    return status


def render_markdown(status: Dict[str, Any]) -> str:
    """Short human-readable summary for terminal use."""

    def mark(ok: bool) -> str:
        return "\u2705" if ok else "\u2741"

    lines = ["# ai-harness status", ""]
    inst = status.get("install", {})
    if inst.get("found"):
        lines.append(f"{mark(True)} install: {inst['package']} @ {inst['origin']}")
        if inst.get("version"):
            lines[-1] += f" (v{inst['version']})"
    else:
        lines.append(f"{mark(False)} install: {inst.get('error', 'not found')}")
    proj = status.get("project", {})
    if proj.get("attached"):
        lines.append(f"{mark(True)} project: {proj['root']}")
        lines.append(f"         task={proj.get('task_config')} plugin={proj.get('plugin_id')}")
    else:
        lines.append(f"{mark(False)} project: {proj.get('error', 'not attached')}")
    sess = status.get("session", {})
    active = sess.get("active_session")
    if active:
        lines.append(f"{mark(True)} session: {active.get('session_id')} ({active.get('status')})")
    else:
        lines.append(f"{mark(False)} session: none (sessions_total={sess.get('sessions_total', 0)})")
    art = status.get("artifacts", {})
    if art.get("error"):
        lines.append(f"{mark(False)} artifacts: {art['error']}")
    elif art.get("drifted_repairable") or art.get("unverifiable") or art.get("broken"):
        lines.append(
            f"{mark(False)} artifacts: {art.get('verified', 0)} verified, "
            f"{art.get('drifted_repairable', 0)} drift-repairable, "
            f"{art.get('unverifiable', 0)} unverifiable (of {art.get('runs_total')})"
        )
        for item in art.get("broken", [])[:3]:
            kind = item.get("kind", "")
            lines.append(f"         - {item['exp_dir']} ({kind}): {item['reason'][:80]}")
        if art.get("repair_hint"):
            lines.append(f"         → {art['repair_hint']}")
    else:
        lines.append(f"{mark(True)} artifacts: {art.get('verified', 0)} verified runs")
    cloud = status.get("cloud")
    if cloud:
        accounts = cloud.get("accounts", {})
        flags = ", ".join(f"{k}={'ok' if v.get('available') else 'off'}" for k, v in accounts.items())
        lines.append(f"{mark(any(v.get('available') for v in accounts.values()))} cloud: {flags}")
    adoption = status.get("adoption", {})
    if adoption.get("status") == "ok":
        verdict = adoption.get("verdict", "?")
        ok = verdict in ("healthy", "bypass_detected") and adoption.get("bypassed_run_outputs", 0) == 0
        detail = (
            f"{adoption.get('experiments', 0)} experiments"
            + (f", {adoption.get('quarantined_experiments', 0)} quarantined" if adoption.get("quarantined_experiments") else "")
            + f", {adoption.get('sessions', 0)} sessions"
        )
        if adoption.get("recovered_run_output_dirs", 0) > 0:
            detail += f", {adoption['recovered_run_output_dirs']} recovered"
        if adoption.get("bypassed_run_outputs", 0) > 0:
            dirs = adoption.get("bypassed_run_output_dirs") or 0
            detail += f", **{dirs} bypassed run-output dirs**" if dirs else ", **bypassed run-output dirs**"
        lines.append(f"{mark(ok)} adoption: {detail} ({verdict})")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    # Called as a subcommand of `ai-harness`, argv has no leading "status".
    parser = argparse.ArgumentParser(prog="ai-harness status", description="Single-command install + workspace health report.")
    parser.add_argument("--project-root", default=None, help="Task workspace root (default: current dir / HARNESS_PROJECT_ROOT)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON (default output)")
    parser.add_argument("--markdown", action="store_true", help="Print short human-readable summary")
    parser.add_argument("--no-cloud", action="store_true", help="Skip cloud/provider checks (faster, offline-friendly)")
    args = parser.parse_args(argv)
    status = collect_status(project_root=args.project_root, include_cloud=not args.no_cloud)
    if args.markdown:
        print(render_markdown(status))
    else:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    # exit 0 if install + project OK; 2 otherwise so scripts can gate on it
    ok = bool(status.get("install", {}).get("found")) and bool(status.get("project", {}).get("attached"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
