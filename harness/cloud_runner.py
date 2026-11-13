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
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cloud_inspector import CloudPreflightInspector
from harness.cleaner import STAGING_MARKER, _bounded, _inventory, _no_links
from harness.cloud_policy import (
    UploadConsentStore,
    dataset_fingerprint,
    private_upload_resource_names,
)
from harness.data_bundle import declared_data_roots
from harness.job_state import update_worker_state
from harness.model_assets.resolution import (
    enforce_model_admission,
    prepare_or_validate_model_plan,
    resolve_effective_config,
)
from harness.model_assets.resolver import ModelResolver
from harness.model_assets.store import ModelStore
from harness.project import HarnessProject
from harness.remote_runtime import resolve_remote_runtime
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import (
    generate_modal_script,
    push_dataset_to_modal,
)
from harness.tree_journal import TreeJournal


def _release_staging(stage_dir: Path, keep_staging: bool) -> None:
    """Only release the allocated temp tree; refuse changed link boundaries."""
    try:
        _bounded(stage_dir, Path(tempfile.gettempdir()))
        if not stage_dir.name.startswith(("aih-kaggle-job_", "aih-modal-job_")):
            raise ValueError("Unrecognized staging directory prefix")
        if keep_staging:
            marker = stage_dir / STAGING_MARKER
            if marker.exists():
                _no_links(marker)
                marker.unlink()
            print(f"Staging retained: {stage_dir}")
        else:
            _inventory(stage_dir)  # refuse nested links/junctions before rmtree
            shutil.rmtree(stage_dir, ignore_errors=True)
            if stage_dir.exists():
                print(f"Warning: could not fully remove staging: {stage_dir}", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(f"Warning: staging cleanup refused for {stage_dir}: {exc}", file=sys.stderr)


def _warn_if_framework_checkout_dirty() -> None:
    """Warn when the canonical framework checkout is dirty at dispatch time.

    Remote bundles are built from the installed (canonical) framework
    source. Uncommitted edits there diverge from any committed identity
    the kernel records and will not survive ``ai-harness sync`` — most
    often they are an agent patching the checkout in place instead of
    fixing the framework repo.
    """
    import harness

    framework_dir = Path(harness.__file__).resolve().parent.parent
    if not (framework_dir / ".git").exists():
        return
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(framework_dir), capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return
    if proc.returncode != 0:
        return
    dirty = [line for line in proc.stdout.splitlines() if line.strip()]
    if dirty:
        print(
            "[dispatch preflight] WARNING: framework checkout is dirty "
            f"({len(dirty)} uncommitted change(s) at {framework_dir}). The "
            "remote bundle ships these uncommitted edits, and they will not "
            "match any committed framework identity. Commit or stash them "
            "before dispatching for reproducible remote runs.",
            flush=True,
        )


def validate_cloud_recipe(config, backend):
    """Honor explicit draft locks before any cloud preparation/provider checks."""
    offline = config.get('offline')
    if isinstance(offline, dict) and 'ready_for_execution' in offline:
        if offline['ready_for_execution'] is not True:
            raise ValueError('Offline recipe is a draft; cloud upload/dispatch is blocked')
        if offline.get('backend') != backend:
            raise ValueError('Offline recipe backend differs from requested cloud backend')


def _discover_dispatch_project(project_root):
    """Honor a task override only when it belongs to the selected workspace."""
    project = HarnessProject.discover(project_root)
    override = os.environ.get("HARNESS_TASK_CONFIG")
    if override:
        path = Path(override).expanduser()
        path = (path if path.is_absolute() else project.root / path).resolve()
        if path.is_relative_to(project.root) and path.is_file():
            return HarnessProject.discover(project.root, path)
    return project


def _prepare_remote_run(
    config_path: str,
    backend: str,
    project_root: Optional[str | Path] = None,
    *,
    effective_config: Optional[Dict[str, Any]] = None,
    model_plan: Any = None,
    base_config: Optional[str | Path] = None,
    probe_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if os.environ.get("AI_HARNESS_REMOTE_KERNEL") == "1":
        raise RuntimeError(
            "Refusing to prepare a remote dispatch: this process is already "
            "inside a dispatched remote kernel (AI_HARNESS_REMOTE_KERNEL=1). "
            "Run the experiment locally with the runner CLI instead."
        )
    project = _discover_dispatch_project(project_root)
    config = Path(config_path).expanduser()
    if not config.is_absolute():
        config = project.root / config
    config = config.resolve()
    try:
        config.relative_to(project.root)
    except ValueError as exc:
        raise ValueError("Experiment config must be inside the attached task workspace.") from exc

    if effective_config is None:
        effective_config = resolve_effective_config(
            project,
            config,
            {"base_config": base_config} if base_config is not None else None,
        )
    elif not isinstance(effective_config, dict):
        raise ValueError("Effective experiment config must be a mapping")
    validate_cloud_recipe(effective_config, backend)
    from harness.task_spec import load_active_task_spec

    task_spec = load_active_task_spec(project.task_config_path)
    contract_backend = getattr(task_spec, "execution_backend", None)
    if (
        bool(getattr(task_spec, "execution_locked", False))
        and contract_backend
        and backend != contract_backend
    ):
        raise ValueError(
            f"Backend lock violation: the task contract locks execution to "
            f"'{contract_backend}' (execution.locked: true) but '{backend}' was requested. "
            f"Run 'ai-harness set-backend {backend} --unlock' to switch the contract backend."
        )
    if probe_spec is None:
        from harness.run_data import resolve_execution_scope
        validation_scope = resolve_execution_scope(project, effective_config, task_spec=task_spec)
        if not validation_scope["complete_selection"]:
            raise ValueError("VALIDATION_SCOPE_INCOMPLETE: partial validation cannot dispatch as a full run")
    resolved_plan = prepare_or_validate_model_plan(
        task_spec, effective_config, ModelStore(), project=project
    )
    if model_plan is not None and model_plan.lock_digest != resolved_plan.lock_digest:
        from harness.model_assets.resolution import ModelAdmissionError

        raise ModelAdmissionError(
            "MODEL_PLAN_INVALID",
            "Supplied model plan does not match the effective config and active task policy.",
        )
    model_plan = resolved_plan
    enforce_model_admission(model_plan)

    base_config_rel = None
    if base_config is not None:
        base_path = Path(base_config).expanduser()
        if not base_path.is_absolute():
            base_path = project.root / base_path
        base_path = base_path.resolve()
        try:
            base_config_rel = base_path.relative_to(project.root).as_posix()
        except ValueError as exc:
            raise ValueError("Cloud base config must be inside the attached task workspace.") from exc
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config does not exist: {base_path}")

    _warn_if_framework_checkout_dirty()
    passed, errors, cfg = CloudPreflightInspector.run_full_preflight_check(
        config,
        project_root=project.root,
        full_data_check=False,
        effective_config=effective_config,
    )
    if not passed:
        raise ValueError("Preflight failed: " + "; ".join(errors))
    from harness.citation_audit import enforce_citation_admission
    citation_admission = enforce_citation_admission(project.root, cfg, action="cloud_dispatch")
    if citation_admission.get("admission") == "blocked":
        raise ValueError("CITATION_ADMISSION_BLOCKED: " + str(citation_admission.get("reason")))
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
    if probe_spec is None and artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        raise ValueError(f"Experiment artifacts already exist at `{artifacts_dir}`; use a new experiment ID.")
    runtime = resolve_remote_runtime(project)
    task_doc = __import__("yaml").safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
    task = task_doc.get("task", task_doc)
    data_config = task.get("data_config") or task.get("data") or {}
    task_runtime = task.get("runtime") or {}
    resource_scope = {
        "root": project.root,
        "task": task,
        "runtime": task_runtime,
        "offline": not bool(task_runtime.get("enable_internet", False)),
        "model_locations": cfg.get("model_locations") or task.get("model_locations") or (task.get("model_governance") or {}).get("model_locations") or {},
        "model_asset_paths": cfg.get("model_asset_paths") or task.get("model_asset_paths") or {},
        "model_manifests": cfg.get("model_manifests") or task.get("model_manifests") or {},
        "model_sources": cfg.get("model_sources") or task.get("model_sources") or [],
        "model_volume_name": cfg.get("model_volume_name") or task.get("model_volume_name"),
        "hf_cache_dir": cfg.get("hf_cache_dir") or task.get("hf_cache_dir"),
        "hf_hub_cache_dir": cfg.get("hf_hub_cache_dir") or task.get("hf_hub_cache_dir"),
    }
    model_resource_plan = ModelResolver().plan_backend_resources(
        model_plan, backend, resource_scope
    )
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
        "effective_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "base_config_rel": base_config_rel,
        "model_plan": model_plan,
        "model_resource_plan": model_resource_plan,
        "task_id": project.manifest.get("task_id") or project.plugin_id or "task",
        "job_id": None,
        "fingerprint": fingerprint,
        "external_kaggle_dataset": external_kaggle_dataset,
    }


def _sync_completed_run(project: HarnessProject, exp_id: str) -> Dict[str, Any]:
    from harness.artifacts import validate_run
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.task_spec import load_active_task_spec
    spec=load_active_task_spec(project.task_config_path)
    context=TaskPluginContext(project.root,project.task_config_path,spec,project.plugin_id or "default")
    plugin=load_task_plugin(project.plugin_id,context)
    validate_run(project.artifacts_dir/exp_id,task_spec=spec,plugin=plugin,expected={"exp_id":exp_id})
    import yaml
    cfg = yaml.safe_load((project.artifacts_dir / exp_id / "run_config.yaml").read_text(encoding="utf-8")) or {}
    rule_id = cfg.get("research", {}).get("decision_rule_id")
    if rule_id:
        from harness.decision_rules import evaluate_decision_rule, load_registration
        registration = load_registration(project, rule_id)
        evaluate_decision_rule(registration["rule"],
            {"kind": "full_run", "id": exp_id, "partition": cfg.get("validation", {}).get("partition", "oof")},
            project=project, task_spec=spec, plugin=plugin, registration=registration)
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


def _stage_modal_model_resources(context, receipt_path, *, detach=False):
    """Stage selected models in a new owned volume; retain a recovery receipt."""
    resource_plan = context.get('model_resource_plan')
    if not resource_plan or not resource_plan.required_sources:
        return None
    task = context['task']
    environment_name = (task.get('runtime') or {}).get('modal_environment')
    if not isinstance(environment_name, str) or not environment_name.strip():
        raise ValueError('Declare task.runtime.modal_environment before staging model resources')
    if detach:
        raise ValueError('Owned-model staging requires attached mode until detached cleanup is supported')
    from harness.model_assets.transfer import plan_model_transfers
    from harness.model_assets.modal_staging import stage_modal_models
    scope = {**task, **(task.get('model_governance') or {}), 'root': context['project'].root}
    local = ModelResolver().plan_backend_resources(context['model_plan'], 'local', scope)
    transfer = plan_model_transfers(local, resource_plan)
    receipt = stage_modal_models(transfer, receipt_path, exp_id=context['exp_id'],
                                 environment_name=environment_name)
    dispatched = copy.deepcopy(resource_plan.to_dict())
    for source in dispatched['required_sources']:
        source['volume_name'] = receipt['volume_name']
    return dispatched


def run_via_modal(
    config_path: str,
    gpu: str = "A10",
    auto_submit: bool = False,
    baseline_csv: Optional[str] = None,
    *,
    project_root: Optional[str | Path] = None,
    job_id: Optional[str] = None,
    keep_staging: bool = False,
    detach: bool = False,
    dry_run: bool = False,
    effective_config: Optional[Dict[str, Any]] = None,
    base_config: Optional[str | Path] = None,
    probe_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Upload consented task data, dispatch Modal, verify returned artifact identity, and sync the journal."""
    if probe_spec is not None:
        from harness.cloud_probe import probe_selection as normalize_probe
        probe_spec = normalize_probe(fold=probe_spec['fold'], probe_samples=probe_spec['sample_limit'])
        if detach:
            raise ValueError('Remote probes require attached execution')
    if dry_run:
        # Plan before account checks, consent, upload, job state or provider imports.
        from harness.task_spec import load_active_task_spec
        project = _discover_dispatch_project(project_root)
        config = Path(config_path).expanduser()
        config = (config if config.is_absolute() else project.root / config).resolve()
        config.relative_to(project.root)
        cfg = effective_config if effective_config is not None else resolve_effective_config(
            project, config, {"base_config": base_config} if base_config is not None else None
        )
        task_spec = load_active_task_spec(project.task_config_path)
        model_plan = prepare_or_validate_model_plan(task_spec, cfg, ModelStore(), project=project)
        enforce_model_admission(model_plan)
        scope = {**(getattr(task_spec, "metadata", {}) or {}),
                 **(getattr(task_spec, "model_governance_config", {}) or {}), "root": project.root}
        resources = ModelResolver().plan_backend_resources(model_plan, "modal", scope)
        from harness.model_assets.transfer import plan_model_transfers
        local_resources = ModelResolver().plan_backend_resources(model_plan, "local", scope)
        transfer_plan = plan_model_transfers(local_resources, resources)
        task_doc = __import__('yaml').safe_load(project.task_config_path.read_text(encoding='utf-8')) or {}
        task = task_doc.get('task', task_doc)
        execution_blockers = []
        if resources.required_sources and not (task.get('runtime') or {}).get('modal_environment'):
            execution_blockers.append('Declare task.runtime.modal_environment for owned model staging')
        if resources.required_sources and detach:
            execution_blockers.append('Owned model staging currently requires attached execution')
        offline_recipe = cfg.get('offline')
        if isinstance(offline_recipe, dict):
            if offline_recipe.get('ready_for_execution') is not True:
                execution_blockers.append('Offline recipe remains a draft')
            if offline_recipe.get('backend') != 'modal':
                execution_blockers.append('Offline recipe must select modal backend before remote execution')
            if any(not value for value in (offline_recipe.get('runtime_versions') or {}).values()):
                execution_blockers.append('Offline runtime dependency versions are not fully pinned')
        plan = {
            "backend": "modal", "gpu": gpu,
            "exp_id": cfg.get("experiment", {}).get("id", config.stem),
            "config": config.relative_to(project.root).as_posix(),
            "model_resource_plan": resources.to_dict(),
            "model_transfer_plan": transfer_plan,
            "blockers": [error for source in resources.required_sources
                         for error in source.get("missing_prerequisites", [])],
            "provider_dispatch": "not performed",
            "execution_blockers": execution_blockers,
            "probe_selection": probe_spec,
            "limitations": "Static plan only; account, uploads, volume contents, runtime and model inference not verified.",
        }
        result = {"dry_run": True, "plan": plan}
        print(json.dumps(result, indent=2, default=str))
        return result
    print("===========================================================")
    print(f"🚀 MODAL.COM CLOUD DISPATCH: {config_path}")
    print("===========================================================")

    context = _prepare_remote_run(
        config_path,
        "modal",
        project_root,
        effective_config=effective_config,
        base_config=base_config,
        probe_spec=probe_spec,
    )
    project = context["project"]
    exp_id = context["exp_id"]
    managed_job_id = job_id
    from harness.decision_rules import prepare_remote_research
    if probe_spec is None:
        prepare_remote_research(project, context["config_rel"], effective_config=context["cfg"])
    job_id = job_id or f"modal_{exp_id}_{uuid.uuid4().hex[:12]}"
    if managed_job_id:
        update_worker_state(
            project.root, managed_job_id, status="RUNNING", provider_status="PREPARING",
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )

    try:
        import modal  # noqa: F401  (import presence is the availability check)
    except ImportError:
        print(
            "❌ Modal is not installed. Run `pip install modal` and `modal setup` to authenticate.",
            file=sys.stderr,
        )
        raise RuntimeError("Modal is not installed. Install the `cloud` extra and authenticate with the official Modal CLI.")

    # A cloud upload is explicit and scoped to the current dataset fingerprint.
    resources = private_upload_resource_names(project, fingerprint_data=context["fingerprint"])
    volume_name = resources["modal_volume_name"]
    # Generate code in a unique temporary directory; never replace an older job bundle.
    stage_dir = Path(tempfile.mkdtemp(prefix="aih-modal-job_"))
    model_receipt_path = project.root / '.ai-harness' / 'cloud-model-staging' / f'{uuid.uuid4().hex}.json'
    try:
        (stage_dir / STAGING_MARKER).write_text(str(os.getpid()), encoding="ascii")
        stage_dir.mkdir(parents=True, exist_ok=True)
        modal_script_path = stage_dir / f"app_{exp_id}.py"
        dispatch_models = _stage_modal_model_resources(context, model_receipt_path, detach=detach)
        if probe_spec is not None and not dispatch_models:
            raise ValueError('Remote probe currently requires governed owned-model storage')
        model_environment = (context['task'].get('runtime') or {}).get('modal_environment') if dispatch_models else None
        if dispatch_models:
            from harness.model_assets.modal_staging import upload_owned_task_data
            receipt = upload_owned_task_data(project, model_receipt_path, context['fingerprint']['fingerprint'])
            volume_name = receipt['volume_name']
            upload = {'success': True}
        else:
            upload = push_dataset_to_modal(volume_name=volume_name, project_root=project.root,
                                           environment_name=model_environment)
        if not upload.get('success'):
            raise RuntimeError(f"Modal dataset upload failed: {upload.get('error') or upload.get('logs')}")

        code_content = generate_modal_script(
            context["config_rel"], gpu=gpu, exp_id=exp_id, volume_name=volume_name,
            project_root=project.root, task_config_path=project.task_config_path,
            task_id=context["task_id"], job_id=job_id,
            runtime_manifest=context["runtime"], config_sha256=context["config_sha256"],
            artifacts_rel=project.artifacts_dir.relative_to(project.root).as_posix(),
            base_config_path=context.get("base_config_rel"),
            model_resource_plan=dispatch_models,
            owned_job_volume=bool(dispatch_models),
            probe_selection=probe_spec,
        )
        modal_script_path.write_text(code_content, encoding="utf-8")

        # 3. Execute Modal Run
        cmd = ["modal", "run"]
        if detach:
            cmd.append("--detach")
        cmd.append(str(modal_script_path))
        print(f"Executing: {' '.join(cmd)}", flush=True)
        if managed_job_id:
            update_worker_state(
                project.root, managed_job_id, status="RUNNING", provider_status="RUNNING",
                exp_id=exp_id, config_sha256=context["config_sha256"],
            )

        provider_env = {**os.environ, 'MODAL_ENVIRONMENT': model_environment} if model_environment else None
        if model_receipt_path.is_file():
            from harness.model_assets.modal_staging import mark_modal_dispatch_started
            from harness.templates.modal_template import modal_app_name
            mark_modal_dispatch_started(model_receipt_path,
                identity={'exp_id': exp_id, 'task_id': context['task_id'], 'job_id': job_id,
                          'config_sha256': context['config_sha256']},
                app_name=modal_app_name(context['task_id'], exp_id, job_id))
            if probe_spec is not None:
                from harness.io_utils import atomic_write_text
                receipt = json.loads(model_receipt_path.read_text(encoding='utf-8'))
                receipt['probe_selection'] = probe_spec
                atomic_write_text(model_receipt_path, json.dumps(receipt, indent=2))
        ret = subprocess.run(cmd, cwd=str(project.root), env=provider_env)
        if ret.returncode != 0:
            raise RuntimeError(f"Modal execution failed with code {ret.returncode}")

        if detach:
            detected_app_id = None
            try:
                list_proc = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=15)
                if list_proc.returncode == 0:
                    apps = json.loads(list_proc.stdout)
                    for a in apps:
                        desc = a.get("description", "")
                        if exp_id in desc or (managed_job_id and managed_job_id in desc):
                            detected_app_id = a.get("app_id")
                            break
            except Exception:
                pass

            print("\n✔ Modal app successfully dispatched in detached cloud mode.")
            if detected_app_id:
                print(f"  App ID: {detected_app_id}")
                print(f"  Live logs: modal app logs -f {detected_app_id}")
                print(f"  Inspect: ai-harness-agent act --action inspect_cloud_job --job-id {managed_job_id or detected_app_id}")
            result = {
                "backend": "modal",
                "job_id": job_id,
                "remote_id": detected_app_id,
                "status": "RUNNING",
                "detached": True,
                "exp_id": exp_id,
                "upload_fingerprint": context["fingerprint"]["fingerprint"],
            }
            if managed_job_id:
                update_worker_state(
                    project.root, managed_job_id, status="RUNNING", provider_status="RUNNING",
                    remote_id=detected_app_id, exp_id=exp_id, config_sha256=context["config_sha256"],
                )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result

        print("\n✔ Modal remote execution finished successfully.")
        if probe_spec is not None:
            from harness.cloud_probe import probe_relative_path
            artifact_dir = project.artifacts_dir / probe_relative_path(exp_id, job_id)
        else:
            artifact_dir = project.artifacts_dir / exp_id
        manifest_path = artifact_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Modal returned without artifact_manifest.json; result is not accepted.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"job_id": job_id, "task_id": context["task_id"], "exp_id": exp_id, "config_sha256": context["config_sha256"]}
        mismatch = {key: {"expected": val, "actual": manifest.get(key)} for key, val in expected.items() if val is not None and manifest.get(key) != val}
        if mismatch:
            raise RuntimeError(f"Modal artifact identity mismatch: {mismatch}")
        if probe_spec is not None:
            from harness.cloud_probe import verify_probe_artifacts
            result = verify_probe_artifacts(artifact_dir,
                expected={**expected, 'run_id': job_id}, selection=probe_spec)
        else:
            result = _sync_completed_run(project, exp_id)
        result.update({"backend": "modal", "job_id": job_id, "status": "COMPLETED", "identity_verified": True, "upload_fingerprint": context["fingerprint"]["fingerprint"]})
        if model_receipt_path.is_file():
            from harness.model_assets.modal_staging import cleanup_modal_models
            cleaned = cleanup_modal_models(model_receipt_path, artifact_dir)
            result['model_cleanup'] = {'status': cleaned['status'], 'receipt_path': str(model_receipt_path)}
        if managed_job_id:
            update_worker_state(
                project.root, managed_job_id, status="COMPLETED", provider_status="COMPLETED",
                result_metrics=result.get("result_metrics"), artifact_paths=result.get("artifact_paths"),
                exp_id=exp_id, config_sha256=context["config_sha256"],
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except BaseException as exc:
        if model_receipt_path.is_file():
            print(f'Model staging recovery receipt retained: {model_receipt_path}', file=sys.stderr)
            from harness.io_utils import atomic_write_text
            failure_path = model_receipt_path.with_suffix('.failure.json')
            atomic_write_text(failure_path, json.dumps({'exp_id': exp_id, 'error_type': type(exc).__name__,
                'model_receipt': str(model_receipt_path)}, indent=2))
            receipt = json.loads(model_receipt_path.read_text(encoding='utf-8'))
            if receipt.get('created') is True and receipt.get('dispatch_started') is False:
                from harness.model_assets.modal_staging import cleanup_undispatched_models
                try:
                    cleanup_undispatched_models(model_receipt_path)
                except Exception as cleanup_error:
                    print(f'Model cleanup incomplete: {type(cleanup_error).__name__}; receipt retained', file=sys.stderr)
            elif receipt.get('created') is True and receipt.get('dispatch_started') is True and receipt.get('status') not in {'cleaned', 'cleaned_after_failure'}:
                from harness.model_assets.modal_staging import recover_failed_modal_models
                try:
                    recover_failed_modal_models(model_receipt_path,
                        model_receipt_path.with_suffix('') / uuid.uuid4().hex)
                except Exception as recovery_error:
                    print(f'Remote recovery incomplete: {type(recovery_error).__name__}; receipt retained', file=sys.stderr)
        raise
    finally:
        _release_staging(stage_dir, keep_staging)


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
        "effective_config_sha256": context.get("effective_config_sha256"),
        "model_lock_digest": getattr(context.get("model_plan"), "lock_digest", None),
        "model_policy_status": getattr(context.get("model_plan"), "status", None),
        "model_resource_plan": (
            context["model_resource_plan"].to_dict()
            if context.get("model_resource_plan") is not None
            else None
        ),
        "runtime": runtime,
        "enable_internet": bool((context.get("task", {}).get("runtime") or {}).get("enable_internet", False)),
        "requirement_check": (context.get("task", {}).get("runtime") or {}).get("requirement_check", "strict"),
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
    keep_staging: bool = False,
    poll_interval: float = 10.0,
    timeout_seconds: float = 24 * 60 * 60,
    force_import: bool = False,
    dry_run: bool = False,
    effective_config: Optional[Dict[str, Any]] = None,
    base_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Push, poll through a terminal Kaggle state, pull identity-checked artifacts, and sync the journal.

    With dry_run=True, build the full dispatch plan (pre-flight, dataset slug,
    bundle contents, dependency plan, kernel metadata) and return it without
    pushing anything to Kaggle or consuming cloud quota.
    """
    print("===========================================================")
    print(f"🚀 KAGGLE KERNELS API DISPATCH: {config_path}")
    print("===========================================================")

    context = _prepare_remote_run(
        config_path,
        "kaggle",
        project_root,
        effective_config=effective_config,
        base_config=base_config,
    )
    project = context["project"]
    exp_id = context["exp_id"]
    if not dry_run:
        from harness.decision_rules import prepare_remote_research
        prepare_remote_research(project, context["config_rel"], effective_config=context["cfg"])
    # Fail fast on a missing/broken kaggle CLI before any upload or push.
    KaggleKernelManager.ensure_cli_ready()
    managed_job_id = job_id
    job_id = job_id or f"kaggle_{exp_id}_{uuid.uuid4().hex[:10]}"
    if managed_job_id and not dry_run:
        update_worker_state(
            project.root, managed_job_id, status="RUNNING", provider_status="PREPARING",
            exp_id=exp_id, config_sha256=context["config_sha256"],
        )

    # Determine the dataset source before staging. Dry-run may resolve a slug,
    # but must return before uploading any dataset or dispatching any kernel.
    dataset_slug = context["external_kaggle_dataset"]
    if dataset_slug and str(dataset_slug).lower() == "none":
        dataset_slug = None
    elif not dataset_slug:
        username = KaggleKernelManager.get_authenticated_username()
        resources = private_upload_resource_names(
            project, fingerprint_data=context["fingerprint"], kaggle_username=username
        )
        dataset_slug = resources["kaggle_dataset_slug"]
        if not dataset_slug:
            raise RuntimeError("Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry.")
        data_cfg = context["data_config"]
        license_name = data_cfg.get("license") or data_cfg.get("dataset_license")
        if not dry_run:
            uploaded = KaggleKernelManager.push_dataset_to_kaggle(
                dataset_slug=dataset_slug,
                license_name=license_name,
                project_root=project.root,
            )
            if not uploaded.get("success"):
                raise RuntimeError(f"Kaggle private dataset upload failed: {uploaded.get('error') or uploaded.get('logs')}")
            KaggleKernelManager.wait_for_dataset(dataset_slug)

    # 2. Package exact core/adapter runtime and task code without credentials or dataset bytes.
    stage_dir = Path(tempfile.mkdtemp(prefix="aih-kaggle-job_"))
    try:
        (stage_dir / STAGING_MARKER).write_text(str(os.getpid()), encoding="ascii")
        runtime = dict(context["runtime"])
        runtime["core"]["version"] = runtime["core"].get("version", "0.2.0")
        model_resource_plan = context.get("model_resource_plan")
        if model_resource_plan and model_resource_plan.required_sources:
            runtime["model_resource_plan"] = model_resource_plan.to_dict()
        else:
            runtime.pop("model_resource_plan", None)
        model_sources = [
            item["dataset_slug"]
            for item in (model_resource_plan.required_sources if model_resource_plan else [])
            if item.get("dataset_slug")
        ]
        KaggleKernelManager.setup_kernel_package(
            stage_dir, context["config_rel"], exp_id, dataset_slug,
            project_root=project.root, task_config_path=project.task_config_path,
            task_id=context["task_id"], job_id=job_id, runtime_manifest=runtime,
            enable_internet=bool((context["task"].get("runtime") or {}).get("enable_internet", False)),
            bundle_mode=(context["task"].get("runtime") or {}).get("kaggle_bundle_mode", "dataset"),
            requirement_check=(context["task"].get("runtime") or {}).get("requirement_check", "strict"),
            base_config_path=context.get("base_config_rel"),
            model_sources=model_sources,
        )

        print("Preparing private Kaggle code bundle input...")
        if dry_run:
            plan = _kaggle_dry_run_plan(context, project, exp_id, dataset_slug, stage_dir, runtime)
            plan["staging_retained"] = keep_staging
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
        from harness.kaggle_compat import retry_with_backoff

        def _do_push():
            p = subprocess.run(cmd, cwd=str(project.root), capture_output=True, text=True)
            if p.returncode != 0:
                out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
                print(f"[Kaggle push retry notice]: {out}", flush=True)
                raise RuntimeError(f"Kaggle push failed with code {p.returncode}: {out}")
            return p

        retry_with_backoff(_do_push, max_attempts=4, base_delay=2.0, max_delay=15.0)

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
    finally:
        _release_staging(stage_dir, keep_staging)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Harness Cloud GPU Dispatcher")
    parser.add_argument("--backend", choices=["modal", "kaggle"], default="modal")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--base-config", default=None, help="Optional base config inherited by the experiment")
    parser.add_argument("--project-root", default=None, help="Attached task workspace")
    parser.add_argument("--task-config", default=None, help="Task YAML path inside the workspace")
    parser.add_argument("--job-id", default=None, help="Stable local/cloud job identity")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Kaggle status polling interval in seconds")
    parser.add_argument("--timeout-seconds", type=float, default=86400.0, help="Maximum Kaggle kernel wait")
    parser.add_argument("--keep-staging", action="store_true", help="Retain staging for debugging")
    parser.add_argument("--gpu", default="A10", help="Modal GPU type (A10, A100, H100, T4, L4)")
    parser.add_argument("--detach", action="store_true", help="Run Modal job in detached cloud mode without blocking local terminal")
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
            if not args.job_id.startswith("job_"):
                args.job_id = f"job_{args.job_id}"
            update_worker_state(project.root, args.job_id, status="RUNNING", provider_status="STARTING")
        if args.backend == "modal":
            run_via_modal(
                args.config,
                gpu=args.gpu,
                project_root=project.root,
                job_id=args.job_id,
                keep_staging=args.keep_staging,
                detach=args.detach,
                base_config=args.base_config,
            )
        else:
            run_via_kaggle(
                args.config, project_root=project.root, job_id=args.job_id,
                poll_interval=args.poll_interval, timeout_seconds=args.timeout_seconds,
                keep_staging=args.keep_staging, base_config=args.base_config,
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
