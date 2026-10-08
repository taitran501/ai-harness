"""Command line for attaching and inspecting local ML-Harness installations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from harness.plugins import list_task_plugins
from harness.phase_workflow import PhaseWorkflow, PromptPackError, initialize_prompt_pack
from harness.eda import run_eda, render_eda_markdown
from harness.project import HarnessProject, attach_project
from harness.cloud_policy import UploadConsentStore, dataset_fingerprint, provider_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    attach = subparsers.add_parser("attach", help="Attach the installed harness to a task workspace")
    attach.add_argument("--project-root", default=".", help="Task repository root")
    attach.add_argument("--task-config", default="configs/task.yaml", help="Task specification YAML")
    attach.add_argument("--plugin", default=None, help="Installed task-plugin entry point ID")
    attach.add_argument("--force", action="store_true", help="Replace the ML-Harness manifest and skill")

    subparsers.add_parser("plugins", help="List installed task plugins")

    phase = subparsers.add_parser("phase", help="Manage the workspace-owned 8-phase prompt workflow")
    phase_commands = phase.add_subparsers(dest="phase_command", required=True)
    phase_init = phase_commands.add_parser("init", help="Create editable starter prompts under prompts/ml-harness")
    phase_init.add_argument("--project-root", default=".", help="Attached task workspace root")
    phase_init.add_argument("--force", action="store_true", help="Replace starter pack files")
    phase_validate = phase_commands.add_parser("validate", help="Validate phase order, prompt files, and output contracts")
    phase_validate.add_argument("--project-root", default=".", help="Attached task workspace root")
    phase_validate.add_argument("--task-config", default=None, help="Task specification path relative to the workspace")
    phase_status = phase_commands.add_parser("status", help="Show completion and evidence-gated phase state")
    phase_status.add_argument("--project-root", default=".", help="Attached task workspace root")
    phase_status.add_argument("--task-config", default=None, help="Task specification path relative to the workspace")
    phase_status.add_argument("--format", choices=["json", "markdown"], default="markdown")
    phase_render = phase_commands.add_parser("render", help="Render a phase prompt with current task context")
    phase_render.add_argument("--project-root", default=".", help="Attached task workspace root")
    phase_render.add_argument("--task-config", default=None, help="Task specification path relative to the workspace")
    phase_render.add_argument("--phase", choices=["setup", "understand", "baseline", "experiment", "select", "public_lb", "freeze", "finalization"])
    phase_record = phase_commands.add_parser("record", help="Persist the IDE agent's phase response and evidence")
    phase_record.add_argument("--project-root", default=".", help="Attached task workspace root")
    phase_record.add_argument("--task-config", default=None, help="Task specification path relative to the workspace")
    phase_record.add_argument("--phase", required=True, choices=["setup", "understand", "baseline", "experiment", "select", "public_lb", "freeze", "finalization"])
    phase_record.add_argument("--response-file", required=True, help="Markdown or JSON response saved by the IDE agent")
    phase_record.add_argument("--sources-file", help="JSON list of inspected source objects with URL and citation metadata")
    phase_record.add_argument("--metadata-file", help="Optional JSON object with phase-specific metadata")
    phase_record.add_argument("--artifact", action="append", default=[], help="Workspace-relative artifact path; repeat as needed")

    eda = subparsers.add_parser("eda", help="Run general-purpose exploratory data analysis")
    eda_commands = eda.add_subparsers(dest="eda_command", required=True)
    eda_run = eda_commands.add_parser("run", help="Profile declared training data without copying raw rows")
    eda_run.add_argument("--project-root", default=".", help="Attached task workspace root")
    eda_run.add_argument("--data", default=None, help="Optional CSV/TSV/Parquet/JSON input file or directory")
    eda_run.add_argument("--output-dir", default=None, help="Workspace-relative report directory")

    cloud = subparsers.add_parser("cloud", help="Inspect cloud readiness and task-scoped upload consent")
    cloud_commands = cloud.add_subparsers(dest="cloud_command", required=True)
    cloud_preflight = cloud_commands.add_parser("preflight", help="Inspect provider account, CLI, data consent, and experiment config")
    cloud_preflight.add_argument("--project-root", default=".", help="Attached task workspace root")
    cloud_preflight.add_argument("--provider", choices=["kaggle", "modal"], required=True)
    cloud_preflight.add_argument("--config", default=None, help="Optional experiment YAML to validate")
    cloud_consent = cloud_commands.add_parser("consent", help="Record private dataset upload consent for this data fingerprint")
    cloud_consent.add_argument("--project-root", default=".", help="Attached task workspace root")
    cloud_consent.add_argument("--provider", choices=["kaggle", "modal"], required=True)
    cloud_consent.add_argument("--confirm-upload", action="store_true", required=True, help="Explicitly authorize this provider to receive the current task dataset")
    cloud_status = cloud_commands.add_parser("status", help="Show account and task-scoped upload-consent status")
    cloud_status.add_argument("--project-root", default=".", help="Attached task workspace root")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plugins":
        print(json.dumps(list_task_plugins(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "phase":
        root = Path(args.project_root or ".").expanduser().resolve()
        try:
            if args.phase_command == "init":
                result = initialize_prompt_pack(root, force=args.force)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return 0
            workflow = PhaseWorkflow(root, args.task_config)
            if args.phase_command == "validate":
                phases = workflow.load_pack()
                print(json.dumps({"valid": True, "phase_ids": [p["id"] for p in phases], "pack": str(workflow.pack_path)}, indent=2, ensure_ascii=False))
            elif args.phase_command == "status":
                status = workflow.phase_status()
                if args.format == "json":
                    print(json.dumps(status, indent=2, ensure_ascii=False))
                else:
                    print(f"Next phase: {status['next_phase'] or 'initial pass complete'}")
                    for item in status["phases"]:
                        state = "complete" if item["completed"] else "upcoming"
                        if item["can_revisit"]:
                            state += "; new evidence allows revisit"
                        print(f"{item['id']:<14} {state} ({item['runs']} response(s))")
            elif args.phase_command == "render":
                print(workflow.render(args.phase), end="")
            elif args.phase_command == "record":
                response_path = (root / args.response_file).resolve() if not Path(args.response_file).is_absolute() else Path(args.response_file).resolve()
                response_path.relative_to(root)
                response = response_path.read_text(encoding="utf-8")
                sources = json.loads(Path(args.sources_file).read_text(encoding="utf-8")) if args.sources_file else []
                metadata = json.loads(Path(args.metadata_file).read_text(encoding="utf-8")) if args.metadata_file else {}
                record = workflow.record(args.phase, response, sources=sources, artifacts=args.artifact, metadata=metadata)
                print(json.dumps(record, indent=2, ensure_ascii=False))
            return 0
        except (PromptPackError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Phase workflow error: {exc}")
            return 2

    if args.command == "eda":
        try:
            result = run_eda(args.project_root or ".", data_path=args.data, output_dir=args.output_dir)
            print(render_eda_markdown(result))
            print(f"Reports: {result['paths']['json']} and {result['paths']['markdown']}")
            return 0
        except Exception as exc:
            print(f"EDA failed: {exc}")
            return 2

    if args.command == "cloud":
        try:
            project = HarnessProject.discover(args.project_root)
            consent_store = UploadConsentStore(project)
            if args.cloud_command == "status":
                from harness.environment_detector import EnvironmentDetector
                result = {
                    "accounts": EnvironmentDetector.get_full_status(),
                    "upload_consent": consent_store.status(),
                }
            elif args.cloud_command == "consent":
                fingerprint = dataset_fingerprint(project)
                consent = consent_store.grant(args.provider, fingerprint)
                result = {
                    "status": "GRANTED",
                    "consent": consent,
                    "message": "Consent is scoped to this task, provider, and exact data fingerprint. Visibility remains private by default.",
                }
            else:
                result = provider_preflight(project, args.provider)
                if args.config:
                    from harness.cloud_inspector import CloudPreflightInspector
                    config_path = Path(args.config)
                    if not config_path.is_absolute():
                        config_path = project.root / config_path
                    passed, errors, _ = CloudPreflightInspector.run_full_preflight_check(
                        config_path, project_root=project.root
                    )
                    result["experiment_preflight"] = {"passed": passed, "errors": errors}
                    result["ready"] = bool(result["ready"] and passed)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if args.cloud_command != "preflight" or result.get("ready") else 2
        except Exception as exc:
            print(f"Cloud preflight failed: {exc}")
            return 2

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
