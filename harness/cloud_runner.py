"""Cloud Runner Bridge for Heavy ML Experiments.

Provides headless execution interfaces for:
1. Modal.com (Serverless Cloud GPU: A10G, A100, H100 with live log streaming).
2. Kaggle Kernels API (Free 30h GPU quota).

Features:
- Mandatory local Pre-flight Inspection (schema, datasets, syntax) to prevent costly cloud failures.
- Automatic artifact pulling and migration to artifacts/<exp_id>/.
- Optional --auto-submit to directly register verified submission package upon completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cloud_inspector import CloudPreflightInspector
from harness.project import HarnessProject
from harness.submission_manager import SubmissionManager
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script
from harness.cloud_policy import UploadConsentStore, dataset_fingerprint, private_upload_resource_names
from harness.remote_runtime import resolve_remote_runtime
from harness.tree_journal import TreeJournal
from harness.job_state import update_worker_state
from harness.data_bundle import declared_data_roots


def _prepare_remote_run(config_path: str, backend: str, project_root: Optional[str | Path] = None) -> Dict[str, Any]:
    project = HarnessProject.discover(project_root)
    os.environ["HARNESS_PROJECT_ROOT"] = str(project.root)
    os.environ["HARNESS_TASK_CONFIG"] = str(project.task_config_path)
    config = Path(config_path).expanduser()
    if not config.is_absolute():
        config = project.root / config
    config = config.resolve()
    try:
        config.relative_to(project.root)
    except ValueError as exc:
        raise ValueError("Experiment config must be inside the attached task workspace.") from exc
    passed, errors, cfg = CloudPreflightInspector.run_full_preflight_check(config, project_root=project.root)
    if not passed:
        raise ValueError("Preflight failed: " + "; ".join(errors))
    from harness.cloud_policy import provider_preflight
    provider = provider_preflight(project, backend)
    if not provider.get("cli_installed") or not provider.get("authenticated"):
        raise ValueError(f"{backend} CLI/account is not ready; authenticate through its official CLI and retry.")
    if backend == "modal" and not provider.get("version_supported"):
        raise ValueError("Modal 1.5.0 or newer is required for the workspace-safe remote runner.")
    exp_id = str(cfg.get("experiment", {}).get("id") or config.stem)
    declared_roots = declared_data_roots(project)
    for split_value in (
        cfg.get("paths", {}).get("split_path"),
        cfg.get("validation", {}).get("split_file"),
    ):
        if not isinstance(split_value, str) or not split_value.strip():
            continue
        split_path = Path(split_value).expanduser()
        split_path = (split_path if split_path.is_absolute() else project.root / split_path).resolve()
        if not any(split_path == path or path.is_dir() and split_path.is_relative_to(path) for path, _ in declared_roots):
            raise ValueError(
                f"Split file `{split_value}` is outside the task data upload fingerprint. Declare it under `data_config.split_path` in the task config and grant consent again."
            )
    artifacts_dir = project.artifacts_dir / exp_id
    if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        raise ValueError(f"Experiment artifacts already exist at `{artifacts_dir}`; use a new experiment ID.")
    runtime = resolve_remote_runtime(project)
    task_doc = __import__("yaml").safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
    task = task_doc.get("task", task_doc)
    data_config = task.get("data_config") or task.get("data") or {}
    fingerprint = dataset_fingerprint(project)
    consent_store = UploadConsentStore(project)
    external_kaggle_dataset = data_config.get("kaggle_dataset_slug") if backend == "kaggle" else None
    requires_upload = backend == "modal" or not external_kaggle_dataset
    if requires_upload and not consent_store.is_granted(backend, fingerprint["fingerprint"]):
        raise ValueError(
            f"Dataset upload is not approved for {backend} and this task/data fingerprint. "
            f"Review it, then run `ai-harness cloud consent --provider {backend} --confirm-upload`."
        )
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    return {
        "project": project,
        "config": config,
        "config_rel": config.relative_to(project.root).as_posix(),
        "cfg": cfg,
        "exp_id": exp_id,
        "task": task,
        "data_config": data_config,
        "runtime": runtime,
        "config_sha256": config_sha256,
        "task_id": project.manifest.get("task_id") or project.plugin_id or "task",
        "job_id": None,
        "fingerprint": fingerprint,
        "external_kaggle_dataset": external_kaggle_dataset,
    }


def _sync_completed_run(project: HarnessProject, exp_id: str) -> Dict[str, Any]:
    from harness.artifacts import validate_run
    from harness.task_spec import load_active_task_spec
    from harness.plugins import TaskPluginContext,load_task_plugin
    spec=load_active_task_spec(project.task_config_path)
    context=TaskPluginContext(project.root,project.task_config_path,spec,project.plugin_id or "default")
    plugin=load_task_plugin(project.plugin_id,context)
    validate_run(project.artifacts_dir/exp_id,task_spec=spec,plugin=plugin,expected={"exp_id":exp_id})
    journal = TreeJournal(
        journal_path=project.artifacts_dir / "journal.json",
        artifacts_dir=project.artifacts_dir,task_spec=spec,task_plugin=plugin,
    )
    synced = journal.sync_from_artifacts(project.artifacts_dir)
    metrics_path = project.artifacts_dir / exp_id / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metrics = None
    manifest_path = project.artifacts_dir / exp_id / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_paths = [
            str((project.artifacts_dir / exp_id / filename).relative_to(project.root)).replace("\\", "/")
            for filename in (manifest.get("files") or {})
        ]
    except (OSError, json.JSONDecodeError):
        artifact_paths = []
    return {
        "exp_id": exp_id,
        "artifacts_dir": str(project.artifacts_dir / exp_id),
        "metrics_path": str(metrics_path),
        "journal_nodes_synced": synced,
        "result_metrics": metrics if isinstance(metrics, dict) else {"folds": metrics} if metrics is not None else {},
        "artifact_paths": artifact_paths,
    }


def run_via_modal(
    config_path: str,
    gpu: str = "A10",
    auto_submit: bool = False,
    baseline_csv: Optional[str] = None,
    *,
    project_root: Optional[str | Path] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload consented task data, dispatch Modal, verify returned artifact identity, and sync the journal."""
    print(f"===========================================================")
    print(f"🚀 MODAL.COM CLOUD DISPATCH: {config_path}")
    print(f"===========================================================")

    context = _prepare_remote_run(config_path, "modal", project_root)
    project = context["project"]
    exp_id = context["exp_id"]
    managed_job_id = job_id
    job_id = job_id or f"modal_{exp_id}"
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="RUNNING", provider_status="PREPARING",
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )

    try:
        import modal
    except ImportError:
        print(
            "❌ Modal is not installed. Run `pip install modal` and `modal setup` to authenticate.",
            file=sys.stderr,
        )
        raise RuntimeError("Modal is not installed. Install the `cloud` extra and authenticate with the official Modal CLI.")

    # A cloud upload is explicit and scoped to the current dataset fingerprint.
    resources = private_upload_resource_names(project, fingerprint_data=context["fingerprint"])
    volume_name = resources["modal_volume_name"]
    upload = push_dataset_to_modal(volume_name=volume_name, project_root=project.root)
    if not upload.get("success"):
        raise RuntimeError(f"Modal dataset upload failed: {upload.get('error') or upload.get('logs')}")

    # Generate code in a unique temporary directory; never replace an older job bundle.
    stage_dir = Path(tempfile.mkdtemp(prefix=f"aih-modal-{job_id}-"))
    stage_dir.mkdir(parents=True, exist_ok=True)
    modal_script_path = stage_dir / f"app_{exp_id}.py"

    code_content = generate_modal_script(
        context["config_rel"], gpu=gpu, exp_id=exp_id, volume_name=volume_name,
        project_root=project.root, task_config_path=project.task_config_path,
        task_id=context["task_id"], job_id=job_id,
        runtime_manifest=context["runtime"], config_sha256=context["config_sha256"],
        artifacts_rel=project.artifacts_dir.relative_to(project.root).as_posix(),
    )
    modal_script_path.write_text(code_content, encoding="utf-8")

    # 3. Execute Modal Run
    cmd = ["modal", "run", str(modal_script_path)]
    print(f"Executing: {' '.join(cmd)}")
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="RUNNING", provider_status="RUNNING",
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )
    ret = subprocess.run(cmd, cwd=str(project.root))

    if ret.returncode != 0:
        raise RuntimeError(f"Modal execution failed with code {ret.returncode}")

    print(f"\n✔ Modal remote execution finished successfully.")
    artifact_dir = project.artifacts_dir / exp_id
    manifest_path = artifact_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Modal returned without artifact_manifest.json; result is not accepted.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"job_id": job_id, "task_id": context["task_id"], "exp_id": exp_id, "config_sha256": context["config_sha256"]}
    mismatch = {key: {"expected": val, "actual": manifest.get(key)} for key, val in expected.items() if val is not None and manifest.get(key) != val}
    if mismatch:
        raise RuntimeError(f"Modal artifact identity mismatch: {mismatch}")
    result = _sync_completed_run(project, exp_id)
    result.update({"backend": "modal", "job_id": job_id, "status": "COMPLETED", "identity_verified": True, "upload_fingerprint": context["fingerprint"]["fingerprint"]})
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="COMPLETED", provider_status="COMPLETED",
            result_metrics=result.get("result_metrics"), artifact_paths=result.get("artifact_paths"),
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _kaggle_dry_run_plan(
    context: Dict[str, Any],
    project,
    exp_id: str,
    dataset_slug: Optional[str],
    stage_dir: Path,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize what a Kaggle dispatch would push, without contacting the cloud."""
    metadata_path = stage_dir / "kernel-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        metadata = {}
    bundle_files = []
    for sub in ("code", "configs", "src"):
        d = stage_dir / sub
        if d.is_dir():
            bundle_files.extend(sorted(str(p.relative_to(stage_dir)).replace("\\", "/") for p in d.rglob("*") if p.is_file()))
    return {
        "exp_id": exp_id,
        "task_id": context.get("task_id"),
        "dataset_slug": dataset_slug,
        "config_rel": context.get("config_rel"),
        "config_sha256": context.get("config_sha256"),
        "runtime": runtime,
        "enable_internet": bool((context.get("task", {}).get("runtime") or {}).get("enable_internet", False)),
        "bundle_mode": (context.get("task", {}).get("runtime") or {}).get("kaggle_bundle_mode", "dataset"),
        "kernel_metadata": metadata,
        "bundle_files": bundle_files[:50],
        "stage_dir": str(stage_dir),
        "would_run": [
            "kaggle kernels push -p <stage_dir>",
            "kaggle kernels status <slug> (poll until terminal)",
            "kaggle kernels output <slug> (identity-checked import)",
        ],
    }


def run_via_kaggle(
    config_path: str,
    auto_submit: bool = False,
    baseline_csv: Optional[str] = None,
    *,
    project_root: Optional[str | Path] = None,
    job_id: Optional[str] = None,
    poll_interval: float = 10.0,
    timeout_seconds: float = 24 * 60 * 60,
    force_import: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Push, poll through a terminal Kaggle state, pull identity-checked artifacts, and sync the journal.

    With dry_run=True, build the full dispatch plan (pre-flight, dataset slug,
    bundle contents, dependency plan, kernel metadata) and return it without
    pushing anything to Kaggle or consuming cloud quota.
    """
    print(f"===========================================================")
    print(f"🚀 KAGGLE KERNELS API DISPATCH: {config_path}")
    print(f"===========================================================")

    context = _prepare_remote_run(config_path, "kaggle", project_root)
    project = context["project"]
    exp_id = context["exp_id"]
    managed_job_id = job_id
    job_id = job_id or f"kaggle_{exp_id}_{uuid.uuid4().hex[:10]}"
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="RUNNING", provider_status="PREPARING",
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )

    # Upload task data only after the task/provider/fingerprint consent check.
    dataset_slug = context["external_kaggle_dataset"]
    if not dataset_slug:
        username = KaggleKernelManager.get_authenticated_username()
        resources = private_upload_resource_names(
            project, fingerprint_data=context["fingerprint"], kaggle_username=username
        )
        dataset_slug = resources["kaggle_dataset_slug"]
        if not dataset_slug:
            raise RuntimeError("Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry.")
        data_cfg = context["data_config"]
        license_name = data_cfg.get("license") or data_cfg.get("dataset_license")
        uploaded = KaggleKernelManager.push_dataset_to_kaggle(
            dataset_slug=dataset_slug,
            license_name=license_name,
            project_root=project.root,
        )
        if not uploaded.get("success"):
            raise RuntimeError(f"Kaggle private dataset upload failed: {uploaded.get('error') or uploaded.get('logs')}")
        KaggleKernelManager.wait_for_dataset(dataset_slug)

    # 2. Package exact core/adapter runtime and task code without credentials or dataset bytes.
    stage_dir = Path(tempfile.mkdtemp(prefix=f"aih-kaggle-{job_id[:8]}-"))
    runtime = dict(context["runtime"])
    runtime["core"]["version"] = runtime["core"].get("version", "0.2.0")
    KaggleKernelManager.setup_kernel_package(
        stage_dir, context["config_rel"], exp_id, dataset_slug,
        project_root=project.root, task_config_path=project.task_config_path,
        task_id=context["task_id"], job_id=job_id, runtime_manifest=runtime,
        enable_internet=bool((context["task"].get("runtime") or {}).get("enable_internet", False)),
        bundle_mode=(context["task"].get("runtime") or {}).get("kaggle_bundle_mode", "dataset"),
    )

    print("Preparing private Kaggle code bundle input...")
    if dry_run:
        plan = _kaggle_dry_run_plan(context, project, exp_id, dataset_slug, stage_dir, runtime)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("\n\u2705 --dry-run: dispatch plan built. Nothing was pushed to Kaggle; no cloud quota consumed.")
        return {"dry_run": True, "plan": plan}
    KaggleKernelManager.push_code_bundle(stage_dir)

    # 3. Push to Kaggle
    cmd = ["kaggle", "kernels", "push", "-p", str(stage_dir)]
    print(f"Pushing kernel: {' '.join(cmd)}")
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="RUNNING", provider_status="PUSHING",
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )
    ret = subprocess.run(cmd, cwd=str(project.root))
    if ret.returncode != 0:
        raise RuntimeError(f"Kaggle push failed with code {ret.returncode}")

    # Use the ID actually submitted; title normalization and job suffixes may change it.
    kernel_slug = json.loads((stage_dir / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]
    print(f"Polling Kaggle kernel until a terminal state: {kernel_slug}")

    def persist_kaggle_status(state: Dict[str, Any]) -> None:
        if managed_job_id:
            update_worker_state(
                project.root, managed_job_id, status="RUNNING",
                provider_status=state.get("status") or "UNKNOWN", remote_id=kernel_slug,
                exp_id=exp_id, config_sha256=context["config_sha256"],
            )

    status = KaggleKernelManager.wait_for_kernel(
        exp_id, kernel_slug, poll_interval=poll_interval, timeout_seconds=timeout_seconds,
        status_callback=persist_kaggle_status if managed_job_id else None,
    )
    if status.get("status") != "COMPLETED":
        err_log = KaggleKernelManager.get_kernel_error_log(kernel_slug)
        err_msg = f"Kaggle kernel ended with status {status.get('status')}: {status.get('raw_output', '')}"
        if err_log:
            err_msg += f"\n\n--- Kaggle Remote Kernel Error Log ---\n{err_log}\n--------------------------------------"
        raise RuntimeError(err_msg)
    expected_identity = {
        "job_id": job_id,
        "task_id": context["task_id"],
        "exp_id": exp_id,
        "config_sha256": context["config_sha256"],
        "core_version": context["runtime"]["core"]["version"],
        "plugin_id": context["runtime"]["plugin"]["id"],
    }
    pulled = KaggleKernelManager.pull_kernel_output(
        exp_id, kernel_slug,
        artifact_dir=project.artifacts_dir / exp_id,
        expected_identity=expected_identity,
        project_root=project.root,
        force_import=force_import,
    )
    if not pulled.get("success"):
        raise RuntimeError(f"Kaggle output import failed: {pulled.get('raw_output') or pulled.get('identity_mismatches')}")
    result = _sync_completed_run(project, exp_id)
    result.update({"backend": "kaggle", "job_id": job_id, "kernel_slug": kernel_slug, "status": "COMPLETED", "identity_verified": True, "metrics": pulled.get("metrics"), "migrated_files": pulled.get("migrated_files"), "upload_fingerprint": context["fingerprint"]["fingerprint"]})
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="COMPLETED", provider_status="COMPLETED",
            remote_id=kernel_slug, result_metrics=result.get("result_metrics"),
            artifact_paths=result.get("artifact_paths"), exp_id=exp_id,
            config_sha256=context["config_sha256"],
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def handle_auto_submit(exp_id: str, baseline_csv: Optional[str] = None) -> None:
    """Register verified submission if predictions.csv or oof_predictions.csv exists."""
    pred_path = Path("artifacts") / exp_id / "predictions.csv"
    if not pred_path.exists():
        pred_path = Path("artifacts") / exp_id / "oof_predictions.csv"
    if not pred_path.exists():
        print(f"⚠ Auto-submit skipped: neither predictions.csv nor oof_predictions.csv found on local disk.", file=sys.stderr)
        return

    print(f"\n--- Initiating Auto-Submit for '{exp_id}' ---")
    mgr = SubmissionManager()
    try:
        entry = mgr.register_submission(
            exp_id=exp_id,
            candidate_csv=str(pred_path),
            baseline_csv=baseline_csv,
            notes="Auto-submitted after cloud execution",
        )
        print(f"🎉 Auto-Submit Succeeded! Submission ID: {entry.id}")
        print(f"Registered at: submissions/SUBMISSIONS.md")
    except Exception as e:
        print(f"❌ Auto-submit failed: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Harness Cloud GPU Dispatcher")
    parser.add_argument("--backend", choices=["modal", "kaggle"], default="modal")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--project-root", default=None, help="Attached task workspace")
    parser.add_argument("--task-config", default=None, help="Task YAML path inside the workspace")
    parser.add_argument("--job-id", default=None, help="Stable local/cloud job identity")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Kaggle status polling interval in seconds")
    parser.add_argument("--timeout-seconds", type=float, default=86400.0, help="Maximum Kaggle kernel wait")
    parser.add_argument("--gpu", default="A10", help="Modal GPU type (A10, A100, H100, T4, L4)")
    parser.add_argument(
        "--auto-submit",
        action="store_true",
        help="Automatically package and register submission upon successful run",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline predictions for auto-submit diff check",
    )

    args = parser.parse_args()
    if args.auto_submit:
        parser.error("Cloud execution does not submit automatically. Use the human-approved prepare_submission action after reviewing the result.")
    project = HarnessProject.discover(args.project_root, args.task_config)
    os.environ["HARNESS_PROJECT_ROOT"] = str(project.root)
    os.environ["HARNESS_TASK_CONFIG"] = str(project.task_config_path)
    try:
        if args.job_id:
            update_worker_state(project.root, args.job_id, status="RUNNING", provider_status="STARTING")
        if args.backend == "modal":
            run_via_modal(args.config, gpu=args.gpu, project_root=project.root, job_id=args.job_id)
        else:
            run_via_kaggle(
                args.config, project_root=project.root, job_id=args.job_id,
                poll_interval=args.poll_interval, timeout_seconds=args.timeout_seconds,
            )
    except Exception as exc:
        if args.job_id:
            try:
                update_worker_state(
                    project.root, args.job_id, status="FAILED", provider_status="FAILED",
                    error_message=str(exc),
                )
            except Exception:
                pass
        parser.exit(2, f"Cloud run failed: {exc}\n")


if __name__ == "__main__":
    main()
