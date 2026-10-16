"""Command line for attaching and inspecting local AI-Harness installations."""

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
    parser = argparse.ArgumentParser(prog="ai-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    attach = subparsers.add_parser("attach", help="Attach the installed harness to a task workspace")
    attach.add_argument("--project-root", default=".", help="Task repository root")
    attach.add_argument("--task-config", default="configs/task.yaml", help="Task specification YAML")
    attach.add_argument("--plugin", default=None, help="Installed task-plugin entry point ID")
    attach.add_argument("--force", action="store_true", help="Replace the AI-Harness manifest and skill")

    subparsers.add_parser("plugins", help="List installed task plugins")

    status = subparsers.add_parser("status", help="Single-command install + workspace health report (default JSON)")
    status.add_argument("--project-root", default=None, help="Task workspace root (default: current dir / HARNESS_PROJECT_ROOT)")
    status.add_argument("--markdown", action="store_true", help="Print short human-readable summary instead of JSON")
    status.add_argument("--no-cloud", action="store_true", help="Skip cloud/provider checks (faster, offline-friendly)")

    kaggle = subparsers.add_parser("kaggle", help="Inspect Kaggle kernel state (status/log/output) without re-dispatching")
    kaggle.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: status|log|output + flags")

    repair = subparsers.add_parser("repair", help="Re-seal artifact manifest file hashes that drifted after a run (or --check to report)")
    repair.add_argument("--project-root", default=".", help="Attached task workspace root (default: current dir)")
    repair.add_argument("--exp", action="append", default=None, help="Experiment (run folder) name to repair; repeat for several")
    repair.add_argument("--all", action="store_true", help="Repair every drifted run in the workspace")
    repair.add_argument("--check", action="store_true", help="Only report drifted/unverifiable runs; do not write anything")
    repair.add_argument("--drop-missing", action="store_true", help="Also drop manifest file entries missing on disk")
    repair.add_argument("--json", action="store_true", help="Emit machine-readable result")

    phase = subparsers.add_parser("phase", help="Manage the workspace-owned 8-phase prompt workflow")
    phase_commands = phase.add_subparsers(dest="phase_command", required=True)
    phase_init = phase_commands.add_parser("init", help="Create editable starter prompts under prompts/ai-harness")
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
    cloud_pull = cloud_commands.add_parser("pull", help="Pull artifacts from completed remote Kaggle kernel")
    cloud_pull.add_argument("--project-root", default=".", help="Attached task workspace root")
    cloud_pull.add_argument("--exp-id", required=True, help="Experiment ID to pull")
    cloud_pull.add_argument("--kernel-slug", default=None, help="Explicit Kaggle kernel slug (owner/title)")
    cloud_pull.add_argument("--force-import", action="store_true", help="Import even if non-critical identity mismatches occur")

    nb_export = subparsers.add_parser("export-kaggle-notebook", help="Export standalone offline Kaggle submission notebook (.ipynb)")
    nb_export.add_argument("--exp-dir", required=True, help="Experiment artifact directory")
    nb_export.add_argument("--output", default="submission_notebook.ipynb", help="Target output .ipynb path")
    nb_export.add_argument("--id-column", default="id", help="Dataset identifier column name")
    nb_export.add_argument("--target-column", default="target", help="Dataset prediction target column name")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plugins":
        print(json.dumps(list_task_plugins(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "status":
        from harness.status_report import collect_status, render_markdown

        status = collect_status(project_root=args.project_root, include_cloud=not args.no_cloud)
        if args.markdown:
            print(render_markdown(status))
        else:
            print(json.dumps(status, indent=2, ensure_ascii=False))
        ok = bool(status.get("install", {}).get("found")) and bool(status.get("project", {}).get("attached"))
        return 0 if ok else 2

    if args.command == "kaggle":
        from harness.kaggle_inspect import main as kaggle_main
        return kaggle_main(list(args.args))

    if args.command == "repair":
        from harness.repair import run_repair
        return run_repair(
            project_root=args.project_root,
            exp=args.exp,
            all_drifted=args.all,
            check=args.check,
            drop_missing=args.drop_missing,
            as_json=args.json,
        )

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
            elif args.cloud_command == "pull":
                from harness.templates.kaggle_template import KaggleKernelManager
                result = KaggleKernelManager.pull_kernel_output(
                    exp_id=args.exp_id,
                    kernel_slug=args.kernel_slug,
                    project_root=project.root,
                    force_import=args.force_import,
                )
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

    if args.command == "export-kaggle-notebook":
        from harness.kaggle_notebook_builder import build_kaggle_submission_notebook
        try:
            res = build_kaggle_submission_notebook(
                exp_dir=args.exp_dir,
                output_path=args.output,
                id_column=args.id_column,
                target_column=args.target_column,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"Export notebook failed: {exc}")
            return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
