"""Command line for attaching and inspecting local AI-Harness installations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from harness.cloud_policy import (
    UploadConsentStore,
    dataset_fingerprint,
    provider_preflight,
)
from harness.eda import render_eda_markdown, run_eda
from harness.phase_workflow import (
    PhaseWorkflow,
    PromptPackError,
    initialize_prompt_pack,
)
from harness.plugins import list_task_plugins
from harness.project import HarnessProject, attach_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="Preview or remove explicitly selected framework clutter")
    clean.add_argument("--project-root", default=".", help="Workspace sandbox root")
    mode = clean.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    mode.add_argument("--apply", action="store_true", help="Delete after validation and confirmation")
    clean.add_argument("--temp", action="store_true", help="Old framework staging in system temp")
    clean.add_argument("--legacy", action="store_true", help="Inactive .ml-harness directory")
    clean.add_argument("--probes", action="store_true", help="Old individual probe runs")
    clean.add_argument("--older-than-hours", type=float, default=2, help="Minimum age; temp always at least 2 hours")
    clean.add_argument("--archive", action="store_true", help="Verify a legacy ZIP backup before deletion")
    clean.add_argument("--yes", "-y", action="store_true", help="Skip interactive confirmation")

    attach = subparsers.add_parser("attach", help="Attach the installed harness to a task workspace")
    attach.add_argument("--project-root", default=".", help="Task repository root")
    attach.add_argument("--task-config", default="configs/task.yaml", help="Task specification YAML")
    attach.add_argument("--plugin", default=None, help="Installed task-plugin entry point ID")
    attach.add_argument("--force", action="store_true", help="Replace the AI-Harness manifest and skill")

    subparsers.add_parser("plugins", help="List installed task plugins")

    install_skills = subparsers.add_parser(
        "install-skills",
        help="Install or update modular slash-command skills globally (~/.gemini/config/skills) and/or in workspace (.agents/skills)",
    )
    install_skills.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install into global Antigravity skills directory (~/.gemini/config/skills)",
    )
    install_skills.add_argument(
        "--project-root",
        default=None,
        help="Workspace root to install workspace-scoped skills into .agents/skills",
    )
    install_skills.add_argument(
        "--all",
        action="store_true",
        help="Install both globally and into the active workspace",
    )
    install_skills.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON result",
    )

    audit_scripts = subparsers.add_parser("audit-scripts", help="Scan workspace scripts for raw dispatch calls")
    audit_scripts.add_argument("--project-root", default=".", help="Workspace root")
    audit_scripts.add_argument("--json", action="store_true", help="Emit JSON report")
    audit_claims = subparsers.add_parser("audit-claims", help="Scan markdown documents for forum topic citations and audit them against raw files on disk")
    audit_claims.add_argument("--project-root", default=".", help="Workspace root")
    audit_claims.add_argument("--file", default=None, help="Specific markdown file to audit (defaults to scanning workspace .md files)")
    audit_claims.add_argument("--json", action="store_true", help="Emit JSON report")

    status = subparsers.add_parser("status", help="Single-command install + workspace health report (default JSON)")
    status.add_argument("--project-root", default=None, help="Task workspace root (default: current dir / HARNESS_PROJECT_ROOT)")
    status.add_argument("--markdown", action="store_true", help="Print short human-readable summary instead of JSON")
    status.add_argument("--no-cloud", action="store_true", help="Skip cloud/provider checks (faster, offline-friendly)")

    sync = subparsers.add_parser("sync", help="Update the workspace's ai-harness clone, reinstall, and re-attach (one-command framework upgrade)")
    sync.add_argument("--project-root", default=".", help="Competition workspace root (default: current dir)")
    sync.add_argument("--framework-dir", default=None, help="Path to the ai-harness git clone (default: auto-detect <workspace>/ai-harness or HARNESS_FRAMEWORK_DIR)")
    sync.add_argument("--no-pull", action="store_true", help="Skip git update of the framework clone")
    sync.add_argument("--no-install", action="store_true", help="Skip pip reinstall of the framework package")
    sync.add_argument("--no-attach", action="store_true", help="Skip manifest re-attach and skill regeneration")
    sync.add_argument("--json", action="store_true", help="Emit machine-readable result")

    kaggle = subparsers.add_parser("kaggle", help="Inspect/watch Kaggle kernel state without re-dispatching")
    kaggle.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: status|watch|log|output|list + flags")

    submission = subparsers.add_parser("submission", help="Kaggle submission dispatch, polling, and leaderboard (harness-native)")
    submission.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: dispatch|status|watch|latest|list|quota|leaderboard + flags")

    leaderboard = subparsers.add_parser("leaderboard", help="View Kaggle competition public leaderboard (shortcut to submission leaderboard)")
    leaderboard.add_argument("args", nargs=argparse.REMAINDER, help="competition [flags]")

    comp = subparsers.add_parser("competition", aliases=["comp"], help="Inspect Kaggle competition metadata, status, deadlines, and leaderboards")
    comp.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: info|list|leaderboard + flags")

    discussion = subparsers.add_parser("discussion", help="Inspect Kaggle competition discussions/topics and read messages (harness-native)")
    discussion.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: list|view + flags")

    codabench = subparsers.add_parser("codabench", help="Codabench platform integration (status, logs, artifact download)")
    codabench.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: status|download + flags")

    dispatch_kernel_cmd = subparsers.add_parser("dispatch-kernel", help="Wrap a local notebook into a Kaggle kernel and push it")
    dispatch_kernel_cmd.add_argument("args", nargs=argparse.REMAINDER, help="subcommand args: notebook + flags")

    repair = subparsers.add_parser("repair", help="Re-seal artifact manifest file hashes that drifted after a run (or --check to report)")
    repair.add_argument("--project-root", default=".", help="Attached task workspace root (default: current dir)")
    repair.add_argument("--exp", action="append", default=None, help="Experiment (run folder) name to repair; repeat for several")
    repair.add_argument("--all", action="store_true", help="Repair every drifted run in the workspace")
    repair.add_argument("--check", action="store_true", help="Only report drifted/unverifiable runs; do not write anything")
    repair.add_argument("--drop-missing", action="store_true", help="Also drop manifest file entries missing on disk")
    repair.add_argument("--json", action="store_true", help="Emit machine-readable result")

    import_artifacts = subparsers.add_parser(
        "import-artifacts",
        help="Import verified local run outputs (kernel_output_*, kaggle_results, ...) into the artifact tree; --scan lists candidates",
    )
    import_artifacts.add_argument("--project-root", default=".", help="Attached task workspace root (default: current dir)")
    import_artifacts.add_argument("--source", default=None, help="Run directory containing artifact_manifest.json to import")
    import_artifacts.add_argument("--scan", action="store_true", help="List importable/quarantine candidates instead of importing")
    import_artifacts.add_argument("--reject-stale", action="store_true", help="Refuse runs whose task/scorer identity drifts from the attached spec")

    adoption_scan = subparsers.add_parser(
        "adoption-scan",
        help="Cross-workspace adoption overview: per-workspace adoption verdict + unimported verified runs",
    )
    adoption_scan.add_argument("--root", default=None, help="Parent directory containing workspaces (default: parent of current dir)")
    adoption_scan.add_argument("--include", default=None, help="Comma-separated workspace dir names to include; default: every dir with .ai-harness")

    validate_task_cmd = subparsers.add_parser(
        "validate-task",
        help="Static task.yaml validation (schema, data files, column cross-checks, fold sanity) without running anything",
    )
    validate_task_cmd.add_argument("--project-root", default=".", help="Attached task workspace root (default: current dir)")
    validate_task_cmd.add_argument("--config", default=None, help="Task spec path (default: the attached task config)")

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

    preflight = subparsers.add_parser("preflight", help="Epistemic preflight verification of candidate predictions before costly runs or submissions")
    preflight.add_argument("candidate", help="Candidate predictions CSV file")
    preflight.add_argument("--incumbent", default=None, help="Incumbent baseline predictions CSV file (auto-resolved if omitted)")
    preflight.add_argument("--id-col", default=None, help="Explicit ID column name")
    preflight.add_argument("--pred-col", default=None, help="Explicit prediction column name")
    preflight.add_argument("--pitch", action="store_true", help="Validate candidate pitch governance (reject near-duplicate without paired gain)")
    preflight.add_argument("--claim-kind", default="submission_candidate", choices=("blend", "temperature_tuning", "submission_candidate", "candidate"))
    preflight.add_argument("--candidate-exp-id", default=None)
    preflight.add_argument("--incumbent-exp-id", default=None)
    preflight.add_argument("--paired-oof-eval-id", default=None)
    preflight.add_argument("--project-root", default=".", help="Task workspace root (default: current dir)")
    preflight.add_argument("--json", action="store_true", help="Emit JSON output")

    audit_memory = subparsers.add_parser("audit-memory", help="Classify research memory against live validated evidence")
    audit_memory.add_argument("--project-root", default=".", help="Task workspace root (default: current dir)")
    return parser


def _run_install_skills(args) -> int:
    from harness.skill_injector import get_global_skills_dir, install_global_skills, install_workspace_skills

    installed = {"global": [], "workspace": []}
    do_global = args.global_install or args.all
    do_workspace = (args.project_root is not None) or args.all

    if not args.global_install and args.project_root is None and not args.all:
        do_global = True
        do_workspace = True

    if do_workspace:
        root = Path(args.project_root or ".").expanduser().resolve()
        try:
            paths = install_workspace_skills(project_root=root)
            installed["workspace"] = [str(p) for p in paths]
        except Exception as exc:
            if args.project_root is not None:
                if args.json:
                    print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
                else:
                    print(f"Error installing workspace skills: {exc}", file=sys.stderr)
                return 2
            installed["workspace_error"] = str(exc)

    if do_global:
        try:
            paths = install_global_skills()
            installed["global"] = [str(p) for p in paths]
        except Exception as exc:
            if args.json:
                print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            else:
                print(f"Error installing global skills: {exc}", file=sys.stderr)
            return 2

    if args.json:
        print(json.dumps({"status": "ok", "installed": installed}, indent=2, ensure_ascii=False))
    else:
        if installed["global"]:
            print(f"✔ Installed {len(installed['global'])} global skills in {get_global_skills_dir()}:")
            for p in installed["global"]:
                print(f"  - {p}")
        if installed["workspace"]:
            print(f"✔ Installed {len(installed['workspace'])} workspace skills in .agents/skills:")
            for p in installed["workspace"]:
                print(f"  - {p}")
        if installed.get("workspace_error"):
            print("ℹ Workspace skills skipped (not an attached workspace).")
    return 0


def _run_import_artifacts(args) -> int:
    from harness.import_local import import_verified_local_run, scan_importable_runs

    root = Path(args.project_root or ".").expanduser().resolve()
    if args.scan:
        try:
            entries = scan_importable_runs(root)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"status": "error", "error": str(exc)}))
            return 2
        print(json.dumps({"status": "ok", "root": str(root), "candidates": entries}, indent=2, ensure_ascii=False))
        return 0
    if not args.source:
        print("import-artifacts: --source is required (or use --scan to list candidates)")
        return 2
    source = Path(args.source)
    if not source.is_absolute():
        source = root / source
    try:
        result = import_verified_local_run(root, source, reject_stale=args.reject_stale)
    except ValueError as exc:
        print(json.dumps({"status": "rejected", "source": str(source), "error": str(exc)}, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", **result}, indent=2, ensure_ascii=False))
    return 0


def _run_validate_task(args) -> int:
    """Static task.yaml validation; JSON report, exit 0 ok / 1 errors / 2 usage."""
    from harness.project import HarnessProject
    from harness.task_validate import validate_task

    root = Path(args.project_root or ".").expanduser().resolve()
    config = args.config
    if config is None:
        try:
            config = str(HarnessProject.discover(root).task_config_path)
        except Exception as exc:  # noqa: BLE001 - not attached: report it
            print(json.dumps({"status": "error",
                              "error": f"no task config found under {root} ({exc}); "
                                       f"pass --config explicitly"}))
            return 2
    result = validate_task(Path(config), root)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 1


def _run_adoption_scan(args) -> int:
    from harness.import_local import scan_importable_runs
    from harness.status_report import _adoption_section

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        cwd = Path.cwd().resolve()
        # Default: parent of the current workspace, so all siblings are visible.
        root = cwd.parent if (cwd / ".ai-harness").is_dir() else cwd
    if not root.is_dir():
        print(json.dumps({"status": "error", "error": f"root not found: {root}"}))
        return 2
    workspaces = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / ".ai-harness").is_dir()
    )
    if args.include:
        wanted = {w.strip() for w in args.include.split(",") if w.strip()}
        workspaces = [d for d in workspaces if d.name in wanted]
    rows = []
    for ws in workspaces:
        row: dict = {"workspace": ws.name}
        try:
            adoption = _adoption_section(str(ws))
        except Exception as exc:  # noqa: BLE001
            adoption = {"status": "error", "error": str(exc)}
        row.update(adoption)
        try:
            unimported = [
                e
                for e in scan_importable_runs(ws)
                if e.get("verdict") in ("importable", "quarantine")
            ]
        except Exception:  # noqa: BLE001 - scan is advisory
            unimported = None
        if unimported is not None:
            row["unimported_runs"] = unimported
            row["unimported_count"] = len(unimported)
        rows.append(row)
    summary = {
        "ok": sum(1 for r in rows if r.get("verdict") == "healthy"),
        "bypass": sum(1 for r in rows if r.get("verdict") == "bypass_detected"),
        "no_activity": sum(1 for r in rows if r.get("verdict") == "no_activity"),
        "other": sum(1 for r in rows if r.get("verdict") not in ("healthy", "bypass_detected", "no_activity")),
    }
    print(json.dumps({"status": "ok", "root": str(root), "workspaces": rows, "summary": summary}, indent=2, ensure_ascii=False))
    return 0


def _clean_command(args) -> int:
    from harness.cleaner import execute_cleanup, scan_system_temp, scan_workspace_clutter

    if not (args.temp or args.legacy or args.probes):
        print("Select --temp, --legacy and/or --probes. Preview is the default.", file=sys.stderr)
        return 2
    try:
        plan = []
        if args.legacy or args.probes:
            plan.extend(item for item in scan_workspace_clutter(args.project_root, args.older_than_hours)
                        if (item.kind == "legacy" and args.legacy) or (item.kind == "probe" and args.probes))
        if args.temp:
            plan.extend(scan_system_temp(args.older_than_hours, project_root=args.project_root))
    except (OSError, ValueError) as exc:
        print(f"Cleanup refused: {exc}", file=sys.stderr)
        return 2
    preview = execute_cleanup(plan)
    print(json.dumps({"dry_run": True, "candidate_bytes": sum(i.size_bytes for i in plan),
                      "items": preview}, indent=2))
    if not args.apply or not plan:
        return 2 if any(i["status"] == "refused" for i in preview) else 0
    if not args.yes:
        try:
            confirmed = input("Delete these candidates? Type yes to continue: ").strip().lower() == "yes"
        except (EOFError, KeyboardInterrupt):
            confirmed = False
        if not confirmed:
            print("Cleanup cancelled.")
            return 1
    results = execute_cleanup(plan, apply=True, archive=args.archive)
    print(json.dumps({"dry_run": False, "items": results,
                      "deleted_source_bytes": sum(i["size_bytes"] for i in results if i["status"] == "deleted")}, indent=2))
    return 2 if any(i["status"] == "refused" for i in results) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "clean":
        return _clean_command(args)
    if args.command != "plugins":
        from harness.toolchain import ensure_pinned_toolchain

        _pinned = ensure_pinned_toolchain(getattr(args, "project_root", None), "harness.cli", sys.argv[1:] if argv is None else list(argv))
        if _pinned is not None:
            return int(_pinned.returncode)
    if args.command == "plugins":
        print(json.dumps(list_task_plugins(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "audit-scripts":
        from harness.audit_gate import scan_dispatch_scripts
        report = scan_dispatch_scripts(args.project_root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("No raw dispatch scripts found." if not report["findings"] else "\n".join(
                f"{item['path']}:{item['line']} [{item['severity']}] {item['pattern']}" for item in report["findings"]
            ))
        return 1 if report["findings"] else 0
    if args.command == "audit-claims":
        from harness.audit_gate import scan_markdown_citations
        report = scan_markdown_citations(args.project_root, file_path=args.file)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            if report["status"] == "clean":
                print(f"✔ All {report['total_citations']} citations are verified against disk topics.")
            else:
                print(f"Found {len(report['findings'])} issue(s) across {report['total_citations']} citations:")
                for item in report["findings"]:
                    print(f"  {item['path']}:{item['line']} [{item['severity']}] Topic #{item['topic_id']}: {item['summary']}")
        has_blocking = any(item.get("severity") == "blocking" for item in report["findings"])
        return 1 if has_blocking else 0

    if args.command == "status":
        from harness.status_report import collect_status, render_markdown

        status = collect_status(project_root=args.project_root, include_cloud=not args.no_cloud)
        if args.markdown:
            print(render_markdown(status))
        else:
            print(json.dumps(status, indent=2, ensure_ascii=False))
        ok = bool(status.get("install", {}).get("found")) and bool(status.get("project", {}).get("attached"))
        return 0 if ok else 2

    if args.command == "sync":
        from harness.sync import run_sync

        return run_sync(
            project_root=args.project_root,
            framework_dir=args.framework_dir,
            no_pull=args.no_pull,
            no_install=args.no_install,
            no_attach=args.no_attach,
            as_json=args.json,
        )

    if args.command == "kaggle":
        from harness.kaggle_inspect import main as kaggle_main
        return kaggle_main(list(args.args))

    if args.command == "submission":
        from harness.submission_dispatch import main as submission_main
        return submission_main(list(args.args))

    if args.command == "leaderboard":
        from harness.submission_dispatch import main as submission_main
        return submission_main(["leaderboard"] + list(args.args))

    if args.command in ("competition", "comp"):
        from harness.competition_cli import main as competition_main
        return competition_main(list(args.args))

    if args.command == "discussion":
        from harness.discussion_cli import main as discussion_main
        return discussion_main(list(args.args))

    if args.command == "codabench":
        from harness.codabench_cli import main as codabench_main
        return codabench_main(list(args.args))

    if args.command == "dispatch-kernel":
        from harness.kernel_dispatch import main as kernel_dispatch_main
        return kernel_dispatch_main(list(args.args))

    if args.command == "preflight":
        return _run_preflight(args)

    if args.command == "audit-memory":
        from harness.research_memory import ResearchMemory
        try:
            memory = ResearchMemory.load(args.project_root)
            print(json.dumps(memory.audit_research_memory(), indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:  # noqa: BLE001 - audit is a read-only report
            print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
            return 2

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

    if args.command == "install-skills":
        return _run_install_skills(args)

    if args.command == "import-artifacts":
        return _run_import_artifacts(args)

    if args.command == "adoption-scan":
        return _run_adoption_scan(args)

    if args.command == "validate-task":
        return _run_validate_task(args)

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

                disk_dir = project.root / ".ai-harness"
                if not disk_dir.is_dir():
                    disk_dir = None
                result = {
                    "accounts": EnvironmentDetector.get_full_status(disk_cache_dir=disk_dir),
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


def _run_preflight(args) -> int:
    from harness.candidate_preflight import run_candidate_preflight, validate_candidate_pitch

    try:
        receipt = run_candidate_preflight(
            project_root=args.project_root,
            candidate_path=args.candidate,
            incumbent_path=args.incumbent,
            id_col=args.id_col,
            pred_col=args.pred_col,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, default=str))
        else:
            print(f"Preflight error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.pitch:
        pitch_res = validate_candidate_pitch(
            project_root=args.project_root,
            preflight_id=receipt["preflight_id"],
            candidate_path=args.candidate,
            claim_kind=args.claim_kind,
            candidate_exp_id=args.candidate_exp_id,
            incumbent_exp_id=args.incumbent_exp_id,
            paired_oof_evaluation_id=args.paired_oof_eval_id,
        )
        if args.json:
            print(json.dumps(pitch_res, indent=2, default=str))
        else:
            print(f"Preflight ID: {pitch_res['preflight_id']}")
            print(f"Claim kind:   {pitch_res['claim_kind']}")
            print(f"Allowed:      {pitch_res['allowed']}")
            print(f"Verdict:      {pitch_res['verdict']}")
            if pitch_res.get("spearman_rho") is not None:
                print(f"Spearman rho: {pitch_res['spearman_rho']:.6f}")
            if pitch_res.get("warnings"):
                for w in pitch_res["warnings"]:
                    print(f"Warning: {w}")
            if pitch_res.get("errors"):
                for e in pitch_res["errors"]:
                    print(f"Error: {e}", file=sys.stderr)
        return 0 if pitch_res["allowed"] else 1

    if args.json:
        print(json.dumps(receipt, indent=2, default=str))
    else:
        sim = receipt.get("similarity", {})
        print(f"Preflight ID:    {receipt['preflight_id']}")
        print(f"Candidate:       {receipt['candidate']['path']}")
        print(f"Incumbent:       {receipt['incumbent']['path']}")
        print(f"Spearman rho:    {sim.get('spearman_rho', 0.0):.6f}")
        print(f"Pearson r:       {sim.get('pearson_r', 0.0):.6f}")
        print(f"Rank swap ratio: {sim.get('rank_swap_ratio', 0.0):.4%}")
        print(f"Mean abs delta:  {sim.get('mean_abs_delta', 0.0):.6f}")
        print(f"Max abs delta:   {sim.get('max_abs_delta', 0.0):.6f}")
        print(f"Verdict:         {receipt['verdict']}")
        print(f"Classification:  {receipt['classification']}")
        if receipt.get("delusion_warning"):
            print(f"\n{receipt['delusion_warning']}")
        print(f"\nRecommendation:  {receipt['recommendation']}")

    return 0 if receipt["verdict"] == "PASS" else (1 if receipt["verdict"] == "FLAGGED_NEAR_DUPLICATE" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
