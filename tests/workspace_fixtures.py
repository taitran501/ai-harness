"""Helpers for tests that exercise project-root-aware dashboard APIs."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch


def prepare_dashboard_workspace(owner: type, prefix: str) -> Path:
    """Point dashboard globals and process context at a disposable workspace."""
    from harness.project import HarnessProject
    from tools import dashboard_server

    source_root = Path(__file__).resolve().parent.parent
    temporary = tempfile.TemporaryDirectory(prefix=prefix)
    workspace = Path(temporary.name)
    shutil.copytree(source_root / "configs", workspace / "configs")
    for relative in (Path("tools") / "web", Path("dashboard") / "dist"):
        source = source_root / relative
        if source.is_dir():
            shutil.copytree(source, workspace / relative)

    owner._workspace_temp = temporary
    owner.workspace_root = workspace
    owner._original_cwd = Path.cwd()
    owner._dashboard_globals = {
        name: getattr(dashboard_server, name)
        for name in ("PROJECT_ROOT", "HARNESS_PROJECT", "ARTIFACTS_DIR")
    }
    owner._environment_patch = patch.dict(
        os.environ,
        {
            "HARNESS_PROJECT_ROOT": str(workspace),
            "HARNESS_TASK_CONFIG": str(workspace / "configs" / "task.yaml"),
        },
    )
    owner._environment_patch.start()

    project = HarnessProject.discover(workspace)
    dashboard_server.PROJECT_ROOT = workspace
    dashboard_server.HARNESS_PROJECT = project
    dashboard_server.ARTIFACTS_DIR = project.artifacts_dir
    os.chdir(workspace)
    return workspace


def cleanup_dashboard_workspace(owner: type) -> None:
    """Restore process globals after the isolated dashboard server has stopped."""
    from tools import dashboard_server

    for name, value in owner._dashboard_globals.items():
        setattr(dashboard_server, name, value)
    os.chdir(owner._original_cwd)
    owner._environment_patch.stop()
    owner._workspace_temp.cleanup()
