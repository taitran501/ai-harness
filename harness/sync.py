"""`ai-harness sync` — one-command framework update for a competition workspace.

Every competition workspace carries its own clone of the ai-harness repository
(e.g. `<workspace>/ai-harness`). Historically, updating after a framework
release required a hand-written 4-step brief pasted into each chat
(`git pull`, `pip install -e`, re-attach, verify entrypoints). This command
performs those steps deterministically:

1. update the framework clone (fast-forward only; a dirty tree is never touched),
2. reinstall the editable package with the current interpreter,
3. re-attach the workspace (regenerates `.ai-harness/project.yaml` and the
   task-specific `.agents/skills/ai-harness/SKILL.md` from the new code),
4. report before/after commit + installed version + entrypoint health.

The command never commits, never resets, and never deletes anything.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CANDIDATE_DIR_NAMES = ("ai-harness", "ai_harness")


def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 600) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "rc": proc.returncode,
            "out": (proc.stdout or "").strip(),
            "err": (proc.stderr or "").strip(),
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"rc": None, "out": "", "err": "", "error": f"timed out after {timeout}s"}
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        return {"rc": None, "out": "", "err": "", "error": str(exc)}


def find_framework_repo(
    project_root: Path, framework_dir: Optional[str] = None
) -> Optional[Path]:
    """Locate the ai-harness git clone that serves this workspace.

    Resolution order: explicit `--framework-dir` > `HARNESS_FRAMEWORK_DIR` env
    > conventional workspace subdirectories (`ai-harness/`, `ai_harness/`)
    that contain a git repo with a `harness/` package.
    """
    if framework_dir:
        p = Path(framework_dir).expanduser()
        if not p.is_absolute():
            p = (Path(project_root).expanduser() / p).resolve()
        return p if p.is_dir() else None
    env_dir = os.environ.get("HARNESS_FRAMEWORK_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.is_dir():
            return p
    root = Path(project_root).expanduser()
    for name in CANDIDATE_DIR_NAMES:
        p = root / name
        if p.is_dir() and (p / ".git").exists() and (p / "harness").is_dir():
            return p
    return None


def _git(repo: Path, *args: str) -> Dict[str, Any]:
    return _run(["git", *args], cwd=repo, timeout=600)


def git_state(repo: Path) -> Dict[str, Any]:
    head = _git(repo, "rev-parse", "--short", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    return {
        "commit": head["out"].splitlines()[0] if head["out"] else None,
        "branch": branch["out"].splitlines()[0] if branch["out"] else None,
        "dirty": bool((dirty["out"] or "").strip()),
        "error": head.get("error") or branch.get("error") or dirty.get("error"),
    }


def installed_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("ai-harness")
    except Exception:  # noqa: BLE001
        try:
            import harness

            return getattr(harness, "__version__", None)
        except Exception:  # noqa: BLE001
            return None


def entrypoint_health() -> Dict[str, Any]:
    import importlib

    def _importable(module: str) -> bool:
        try:
            importlib.import_module(module)
            return True
        except Exception:  # noqa: BLE001
            return False

    return {
        "ai-harness": shutil.which("ai-harness") is not None,
        "ai-harness-agent": shutil.which("ai-harness-agent") is not None,
        "harness importable": _importable("harness"),
        "tools.agent_bridge_cli importable": _importable("tools.agent_bridge_cli"),
    }


def run_sync(
    project_root: Optional[str] = None,
    framework_dir: Optional[str] = None,
    no_pull: bool = False,
    no_install: bool = False,
    no_attach: bool = False,
    as_json: bool = False,
) -> int:
    """Run the full sync pipeline; returns a process exit code (0 ok / 2 problem)."""
    from harness.project import HarnessProject

    root = Path(project_root or ".").expanduser().resolve()
    report: Dict[str, Any] = {
        "project_root": str(root),
        "framework_repo": None,
        "before": None,
        "after": None,
        "steps": {},
        "entrypoints": entrypoint_health(),
        "ok": False,
    }

    repo = find_framework_repo(root, framework_dir)
    if repo is None:
        report["error"] = (
            "No ai-harness git clone found in this workspace. Pass --framework-dir "
            "or set HARNESS_FRAMEWORK_DIR (expected a git repo containing a 'harness/' package)."
        )
        _emit(report, as_json)
        return 2
    report["framework_repo"] = str(repo)

    before = git_state(repo)
    report["before"] = before

    # 1. Update the framework clone — fast-forward only, never touch a dirty tree.
    pull: Dict[str, Any] = {"skipped": no_pull, "rc": None}
    if not no_pull:
        if before.get("dirty"):
            pull = {
                "skipped": False,
                "rc": None,
                "reason": "working tree is dirty; commit or stash local changes first",
            }
        else:
            remotes = _git(repo, "remote")
            if not (remotes["out"] or "").strip():
                pull = {
                    "skipped": False,
                    "rc": None,
                    "reason": (
                        "no git remote configured in the framework clone "
                        "(copied, not cloned) — copy a fresh checkout or "
                        "`git remote add origin <url>` to enable updates"
                    ),
                }
            else:
                fetch = _git(repo, "fetch", "--all", "--prune")
                pull = _git(repo, "pull", "--ff-only")
                if fetch.get("error"):
                    pull["fetch_error"] = fetch["error"]
    report["steps"]["pull"] = pull

    # 2. Reinstall the editable package with the current interpreter.
    install: Dict[str, Any] = {"skipped": no_install, "rc": None}
    if not no_install:
        install = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(repo),
                "--quiet",
                "--disable-pip-version-check",
            ],
            timeout=1800,
        )
    report["steps"]["install"] = install

    # 3. Re-attach: refresh manifest + regenerate the task skill from new code.
    attach: Dict[str, Any] = {"skipped": no_attach, "rc": None}
    if not no_attach:
        project = HarnessProject.discover(root)
        task_cfg = project.task_config_path
        if task_cfg is None or not Path(task_cfg).is_file():
            attach = {
                "skipped": False,
                "rc": None,
                "reason": f"task config not found ({task_cfg}); run attach manually",
            }
        else:
            # Pin project discovery for the (cwd-based) skill injector.
            # ATTACH MUST RUN IN A SUBPROCESS: the `ai-harness` console
            # script is a single global editable install whose target
            # changes per workspace, so this process may have imported
            # harness from ANOTHER workspace's clone (observed: sync
            # attached with stale code and silently skipped new
            # workspace-level features). PYTHONPATH pins resolution to
            # the clone just pulled/installed above.
            prev_root_env = os.environ.get("HARNESS_PROJECT_ROOT")
            os.environ["HARNESS_PROJECT_ROOT"] = str(root)
            try:
                code = (
                    "import harness, json;"
                    "from pathlib import Path;"
                    "from harness.project import attach_project;"
                    "from harness.skill_injector import inject_skill_file;"
                    "result = attach_project("
                    f"{str(root)!r}, task_config=Path({str(task_cfg)!r}), force=True);"
                    "result['skill_path'] = str("
                    f"inject_skill_file(config_path=Path({str(task_cfg)!r}))" ") ;"
                    "result['harness_origin'] = harness.__file__;"
                    "print(json.dumps(result))"
                )
                child_env = dict(os.environ)
                child_env["PYTHONPATH"] = str(repo) + os.pathsep + child_env.get("PYTHONPATH", "")
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=child_env,
                    cwd=str(root),
                )
                if proc.returncode == 0:
                    out = proc.stdout.strip().splitlines()
                    result = json.loads(out[-1]) if out else {}
                    attach = {
                        "skipped": False,
                        "rc": 0,
                        "status": result.get("status"),
                        "manifest": result.get("manifest"),
                        "skill_path": result.get("skill_path"),
                        "harness_origin": result.get("harness_origin"),
                    }
                else:
                    attach = {
                        "skipped": False,
                        "rc": proc.returncode,
                        "error": ((proc.stderr or proc.stdout)[-800:]).strip(),
                    }
            except subprocess.TimeoutExpired:
                attach = {"skipped": False, "rc": 3, "error": "attach subprocess timed out"}
            finally:
                if prev_root_env is None:
                    os.environ.pop("HARNESS_PROJECT_ROOT", None)
                else:
                    os.environ["HARNESS_PROJECT_ROOT"] = prev_root_env
    report["steps"]["attach"] = attach

    # Toolchain drift check: did this process start with harness code from
    # the workspace's own clone? (Informational — attach now always uses a
    # pinned subprocess, so drift no longer affects sync results.)
    try:
        import harness as _harness_mod

        origin = Path(_harness_mod.__file__).resolve()
        expected = (repo / "harness" / "__init__.py").resolve()
        report["toolchain"] = {
            "console_script_origin": str(origin),
            "expected_origin": str(expected),
            "pinned_subprocess": True,
        }
    except Exception:  # noqa: BLE001 - informational only
        report["toolchain"] = {"error": "unreadable"}

    report["after"] = git_state(repo)
    report["installed_version"] = installed_version()
    report["entrypoints"] = entrypoint_health()

    pull_ok = no_pull or pull.get("skipped") or pull.get("rc") == 0
    install_ok = no_install or install.get("skipped") or install.get("rc") == 0
    attach_ok = no_attach or attach.get("skipped") or attach.get("rc") == 0
    report["ok"] = bool(pull_ok and install_ok and attach_ok)
    warnings = []
    if pull.get("reason"):
        warnings.append(pull["reason"])
    if install.get("error"):
        warnings.append(f"install: {install['error']}")
    if warnings:
        report["warnings"] = warnings
    _emit(report, as_json)
    return 0 if report["ok"] else 2


def _emit(report: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    def mark(state: Optional[Dict[str, Any]], skip_label: str = "skipped") -> str:
        if state is None:
            return "?"
        if state.get("skipped"):
            return f"[{skip_label}]"
        rc = state.get("rc")
        if rc == 0:
            return "[done]"
        if rc is None and not state.get("error") and not state.get("reason"):
            return "[done]"
        reason = state.get("reason") or state.get("error") or state.get("err", "")
        return f"[FAIL rc={rc}] {reason}".strip()

    repo = report.get("framework_repo") or "<not found>"
    before = report.get("before") or {}
    after = report.get("after") or {}
    print(f"AI-Harness sync — workspace: {report['project_root']}")
    print(f"  framework repo : {repo}")
    print(
        f"  commit       : {before.get('commit') or '?'} -> {after.get('commit') or '?'}"
        f" (branch: {after.get('branch') or '?'})"
    )
    for step in ("pull", "install", "attach"):
        state = report.get("steps", {}).get(step)
        extra = ""
        if step == "install" and state and state.get("rc") not in (0, None):
            extra = f" — {state.get('err', '')[-300:]}"
        if step == "attach" and state and state.get("skill_path"):
            extra = f" — skill: {state['skill_path']}"
        print(f"  {step:<8} : {mark(state)}{extra}")
    print(f"  installed    : {report.get('installed_version') or 'unknown'}")
    for name, ok in (report.get("entrypoints") or {}).items():
        print(f"  entrypoint : {'ok  ' if ok else 'MISSING'} {name}")
    for warning in report.get("warnings") or []:
        print(f"  warning    : {warning}")
    if report.get("error"):
        print(f"  error      : {report['error']}")
    print(f"  result       : {'OK' if report.get('ok') else 'PROBLEMS (see above)'}")
