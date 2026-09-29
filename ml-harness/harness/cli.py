"""Command line for attaching and inspecting local ML-Harness installations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from harness.plugins import list_task_plugins
from harness.project import attach_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    attach = subparsers.add_parser("attach", help="Attach the installed harness to a task workspace")
    attach.add_argument("--project-root", default=".", help="Task repository root")
    attach.add_argument("--task-config", default="configs/task.yaml", help="Task specification YAML")
    attach.add_argument("--plugin", default=None, help="Installed task-plugin entry point ID")
    attach.add_argument("--force", action="store_true", help="Replace the ML-Harness manifest and skill")

    subparsers.add_parser("plugins", help="List installed task plugins")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plugins":
        print(json.dumps(list_task_plugins(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "attach":
        installed_ids = {plugin["id"] for plugin in list_task_plugins()}
        requested_plugin = args.plugin or "default"
        if requested_plugin not in installed_ids:
            print(
                f"Attach failed: task plugin '{requested_plugin}' is not installed. "
                f"Installed plugins: {', '.join(sorted(installed_ids))}."
            )
            return 2
        try:
            result = attach_project(
                project_root=Path(args.project_root),
                task_config=args.task_config,
                plugin_id=args.plugin,
                force=args.force,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            print(f"Attach failed: {exc}")
            return 2
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
