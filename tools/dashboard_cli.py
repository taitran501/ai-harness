"""Command line entry point for the installed AI-Harness dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start the AI-Harness dashboard for a task workspace")
    parser.add_argument("--project-root", default=None, help="Attached task workspace (defaults to the current directory)")
    parser.add_argument("--task-config", default=None, help="Task YAML path relative to the workspace")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    args = parser.parse_args(argv)

    from harness.toolchain import ensure_pinned_toolchain

    _pinned = ensure_pinned_toolchain(args.project_root, "tools.dashboard_cli", sys.argv[1:])
    if _pinned is not None:
        sys.exit(int(_pinned.returncode))
    if args.project_root:
        os.environ["HARNESS_PROJECT_ROOT"] = args.project_root
    if args.task_config:
        os.environ["HARNESS_TASK_CONFIG"] = args.task_config

    # Import only after applying workspace overrides: the dashboard and its
    # managers resolve the project during module initialization.
    from tools.dashboard_server import run_server

    run_server(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
