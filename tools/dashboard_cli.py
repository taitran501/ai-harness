"""Command line entry point for the installed ML-Harness dashboard."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start the ML-Harness dashboard for a task workspace")
    parser.add_argument("--project-root", default=None, help="Attached task workspace (defaults to the current directory)")
    parser.add_argument("--task-config", default=None, help="Task YAML path relative to the workspace")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    args = parser.parse_args(argv)

    if args.project_root:
        os.environ["HARNESS_PROJECT_ROOT"] = args.project_root
    if args.task_config:
        os.environ["HARNESS_TASK_CONFIG"] = args.task_config

    from harness.optional_deps import require_extra

    if not require_extra("ml", "The dashboard"):
        return 2

    # Import only after applying workspace overrides: the dashboard and its
    # managers resolve the project during module initialization.
    from tools.dashboard_server import run_server

    run_server(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
