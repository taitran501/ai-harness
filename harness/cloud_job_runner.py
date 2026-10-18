"""Cloud Job Manager and Asynchronous Execution Engine with Live Log Streaming.

Supports:
1. Modal.com remote execution
2. Kaggle Kernels execution
3. Local/Docker GPU training execution
4. Real-time stdout/stderr log capture to disk & SSE streaming
5. Process cancellation / termination
6. Automated SWE-agent deterministic audit gate upon job completion
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

HARNESS_DIR = Path(__file__).resolve().parent.parent
from harness.io_utils import get_project_root
from harness.project import HarnessProject

PROJECT_ROOT = get_project_root()
HARNESS_PROJECT = HarnessProject.discover(PROJECT_ROOT)
ARTIFACTS_DIR = HARNESS_PROJECT.artifacts_dir
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _find_script(rel_path: str) -> Path:
    """Finds script path either at competition project root or in harness folder."""
    p_root = PROJECT_ROOT / rel_path
    if p_root.exists():
        return p_root
    p_harness = HARNESS_DIR / rel_path
    if p_harness.exists():
        return p_harness
    return p_root


from harness.tree_journal import TreeJournal


@dataclass
class CloudJob:
    """State record for a background execution job."""

    job_id: str
    backend: str  # 'modal', 'kaggle', 'local'
    config_path: str
    exp_id: str
    status: str = "PENDING"  # 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    return_code: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    log_file: str = ""
    pid: Optional[int] = None
    auto_audit: bool = True
    auto_sync: bool = True
    baseline_csv: Optional[str] = None
    audit_passed: Optional[bool] = None
    audit_receipt: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    artifact_dir: str = ""
    remote_id: Optional[str] = None
    task_id: Optional[str] = None
    config_sha256: Optional[str] = None
    artifact_identity_verified: bool = False
    provider_status: Optional[str] = None
    upload_fingerprint: Optional[str] = None
    gpu: Optional[str] = None
    result_metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CloudJobManager:
    """Persistent per-workspace manager for background training jobs and live streaming."""

    def __init__(self, project_root: Optional[Path | str] = None):
        self.project = HarnessProject.discover(project_root)
        self.project_root = self.project.root
        self.artifacts_dir = self.project.artifacts_dir
        os.environ["HARNESS_PROJECT_ROOT"] = str(self.project.root)
        os.environ["HARNESS_TASK_CONFIG"] = str(self.project.task_config_path)
        self._jobs: Dict[str, CloudJob] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._jobs_dir = self.artifacts_dir / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_saved_jobs()

    def _job_path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    def _persist_job(self, job: CloudJob) -> None:
        from harness.state_store import transaction

        with transaction(self._job_path(job.job_id)) as current:
            incoming = job.to_dict()
            if current.get("status") in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            } and incoming.get("status") not in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            current.update(incoming)
        ledger_path = self.project.artifacts_dir / "resource_ledger.json"
        if job.status in {"COMPLETED", "FAILED", "CANCELLED"} and ledger_path.is_file():
            from harness.resource_ledger import terminal

            ledger = json.loads(ledger_path.read_text())
            wall_seconds = max(0, (job.ended_at or time.time()) - job.started_at)
            for token, row in ledger.get("reservations", {}).items():
                if row.get("job_id") == job.job_id:
                    # Compute gpu_hours from the job's GPU allocation so that the
                    # session's `max_gpu_hours` budget is actually enforced for
                    # cloud runs. A row with gpu_hours=None would otherwise make
                    # `_reconcile_session_usage` set gpu_hours_used=None, which
                    # silently disables the GPU budget.
                    # A managed job allocates exactly one GPU; the model name
                    # ("A10G", "A100", "T4") must never be regex-parsed as a
                    # count — that inflated gpu_hours 4–100× and silently
                    # exhausted max_gpu_hours budgets.
                    gpu_count = None
                    allocation_seconds = None
                    if getattr(job, "gpu", None):
                        gpu_count = 1.0
                        allocation_seconds = wall_seconds
                    terminal(
                        ledger_path,
                        token,
                        job.status,
                        wall_seconds,
                        gpu_count=gpu_count,
                        allocation_seconds=allocation_seconds,
                    )

    def _load_saved_jobs(self) -> None:
        field_names = set(CloudJob.__dataclass_fields__)
        for path in self._jobs_dir.glob("job_*.json"):
            if path.name.endswith(".worker.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = CloudJob(
                    **{key: value for key, value in data.items() if key in field_names}
                )
                self._jobs[job.job_id] = job
            except Exception:
                continue

    @staticmethod
    def _pid_running(pid: Optional[int]) -> bool:
        if not pid:
            return False
        pid = int(pid)
        if sys.platform == "win32":
            # CRITICAL: on Windows `os.kill(pid, 0)` is NOT a no-op liveness
            # probe — CPython maps every signal to TerminateProcess(handle,
            # sig), so even signal 0 KILLS the process (verified empirically:
            # a live sleep process died with exit 3221225794 after
            # os.kill(pid, 0)). Status polling would silently terminate the
            # worker under inspection. Use OpenProcess + GetExitCodeProcess.
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Exists but owned by another user; treat as running.
            return True
        except (OSError, ValueError):
            return False

    def _refresh_saved_job(self, job: CloudJob) -> CloudJob:
        worker_state_path = self._jobs_dir / f"{job.job_id}.worker.json"
        worker_state: Dict[str, Any] = {}
        if worker_state_path.is_file():
            try:
                worker_state = json.loads(worker_state_path.read_text(encoding="utf-8"))
                job.provider_status = (
                    worker_state.get("provider_status") or job.provider_status
                )
                job.remote_id = worker_state.get("remote_id") or job.remote_id
                job.result_metrics = (
                    worker_state.get("result_metrics") or job.result_metrics
                )
                job.artifact_paths = (
                    worker_state.get("artifact_paths") or job.artifact_paths
                )
                identity_mismatch = (
                    worker_state.get("job_id") not in (None, job.job_id)
                    or worker_state.get("task_id") not in (None, job.task_id)
                    or worker_state.get("exp_id") not in (None, job.exp_id)
                    or worker_state.get("config_sha256")
                    not in (None, job.config_sha256)
                )
                if identity_mismatch:
                    job.status = "FAILED"
                    job.error_message = "Worker state identity does not match the persisted task, experiment, or config."
                    job.ended_at = job.ended_at or time.time()
                    self._persist_job(job)
                    return job
                if worker_state.get("status") in {"FAILED", "CANCELLED"}:
                    job.status = worker_state["status"]
                    job.error_message = worker_state.get("error_message")
                    job.ended_at = job.ended_at or time.time()
                    self._persist_job(job)
                    return job
            except (OSError, json.JSONDecodeError):
                worker_state = {}

        if job.status not in {"PENDING", "RUNNING"}:
            return job
        if job.job_id in self._processes or self._pid_running(job.pid):
            if worker_state.get("status") == "COMPLETED":
                # The child has durably completed; its process may still be exiting.
                pass
            else:
                return job

        if worker_state.get("status") == "COMPLETED":
            try:
                if job.backend in {"modal", "kaggle"}:
                    self._verify_remote_artifacts(job)
                else:
                    self._verify_local_artifacts(job, worker_state)
                job.status = "COMPLETED"
                job.ended_at = job.ended_at or time.time()
                job.error_message = None
                self._handle_post_run(job)
            except Exception as exc:
                job.status = "FAILED"
                job.error_message = f"Recovered job result verification failed: {exc}"
            self._persist_job(job)
            return job

        if job.backend == "kaggle" and job.remote_id:
            recovered = self._recover_kaggle_job(job)
            if recovered:
                return job

        artifact_dir = (
            Path(job.artifact_dir)
            if job.artifact_dir
            else self.artifacts_dir / job.exp_id
        )
        if not artifact_dir.is_absolute():
            artifact_dir = self.project_root / artifact_dir
        metrics_path = artifact_dir / "metrics.json"
        manifest_path = artifact_dir / "artifact_manifest.json"
        if (
            metrics_path.is_file()
            and job.backend in {"modal", "kaggle"}
            and manifest_path.is_file()
        ):
            try:
                self._verify_remote_artifacts(job)
                job.status = "COMPLETED"
                job.ended_at = job.ended_at or time.time()
                job.error_message = None
                self._handle_post_run(job)
            except Exception as exc:
                job.status = "FAILED"
                job.error_message = (
                    f"Could not recover persisted cloud artifacts: {exc}"
                )
        elif job.backend == "local" and metrics_path.is_file():
            try:
                self._verify_local_artifacts(job, worker_state)
                job.status = "COMPLETED"
                job.ended_at = job.ended_at or time.time()
                self._handle_post_run(job)
            except Exception as exc:
                job.status = "FAILED"
                job.error_message = f"Could not recover local run artifacts: {exc}"
        else:
            job.status = "FAILED"
            job.error_message = (
                "Worker process is no longer running and no terminal state or "
                "verified artifacts were recorded "
                f"(pid {job.pid or 'unknown'} died before completion). "
                "Check the job log, then re-dispatch the run."
            )
        self._persist_job(job)
        return job

    def _verify_local_artifacts(
        self, job: CloudJob, worker_state: Dict[str, Any]
    ) -> None:
        artifact_dir = self.project_root / job.artifact_dir
        metrics_path = artifact_dir / "metrics.json"
        config_path = self.project_root / job.config_path
        if not metrics_path.is_file():
            raise ValueError("Local run has no metrics.json artifact.")
        if worker_state.get("config_sha256") not in (None, job.config_sha256):
            raise ValueError(
                "Worker config fingerprint does not match the recorded job config."
            )
        if config_path.is_file():
            digest = __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()
            if digest != job.config_sha256:
                raise ValueError(
                    "Experiment config changed after this job was created."
                )
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Local run metrics are unreadable: {exc}") from exc
        records = metrics if isinstance(metrics, list) else [metrics]
        if not any(
            isinstance(row, dict)
            and str(row.get("experiment_id") or row.get("exp_id") or job.exp_id)
            == job.exp_id
            for row in records
        ):
            raise ValueError(
                "Local run metrics do not match the tracked experiment ID."
            )
        from harness.artifacts import validate_run
        from harness.task_spec import load_active_task_spec

        validate_run(
            artifact_dir,
            task_spec=load_active_task_spec(self.project.task_config_path),
            expected={"exp_id": job.exp_id, "run_id": job.job_id},
        )
        job.artifact_identity_verified = True
        job.result_metrics = (
            metrics if isinstance(metrics, dict) else {"folds": metrics}
        )
        job.artifact_paths = [
            path.relative_to(self.project_root).as_posix()
            for path in sorted(artifact_dir.iterdir())
            if path.is_file()
        ]

    def _recover_kaggle_job(self, job: CloudJob) -> bool:
        """Resume provider polling and output import after the dashboard worker was lost."""
        try:
            from harness.templates.kaggle_template import KaggleKernelManager

            status = KaggleKernelManager.get_kernel_status(job.exp_id, job.remote_id)
            provider_status = str(status.get("status") or "UNKNOWN").upper()
            job.provider_status = provider_status
            if provider_status in {"QUEUED", "RUNNING", "UNKNOWN", "ERROR"}:
                self._persist_job(job)
                return True
            if provider_status in {"FAILED", "CANCELLED"}:
                job.status = provider_status
                job.error_message = (
                    status.get("raw_output")
                    or f"Kaggle kernel {provider_status.lower()}."
                )
                job.ended_at = job.ended_at or time.time()
                self._persist_job(job)
                return True
            if provider_status != "COMPLETED":
                return False

            from harness.remote_runtime import resolve_remote_runtime

            runtime = resolve_remote_runtime(self.project)
            expected_identity = {
                "job_id": job.job_id,
                "task_id": job.task_id,
                "exp_id": job.exp_id,
                "config_sha256": job.config_sha256,
                "core_version": runtime["core"]["version"],
                "plugin_id": runtime["plugin"]["id"],
            }
            result = KaggleKernelManager.pull_kernel_output(
                job.exp_id,
                job.remote_id,
                artifact_dir=self.artifacts_dir / job.exp_id,
                expected_identity=expected_identity,
                project_root=self.project.root,
            )
            if not result.get("success"):
                raise RuntimeError(
                    result.get("raw_output") or result.get("identity_mismatches")
                )
            self._verify_remote_artifacts(job)
            job.status = "COMPLETED"
            job.ended_at = job.ended_at or time.time()
            job.error_message = None
            self._handle_post_run(job)
            self._persist_job(job)
            return True
        except Exception as exc:
            job.status = "FAILED"
            job.error_message = f"Could not resume Kaggle output collection: {exc}"
            job.ended_at = job.ended_at or time.time()
            self._persist_job(job)
            return True

    def start_job(
        self,
        backend: str,
        config_path: str,
        gpu: str = "A10G",
        auto_audit: bool = True,
        auto_sync: bool = True,
        baseline_csv: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> CloudJob:
        """Launch an asynchronous execution job."""
        state_path = self.artifacts_dir / "competition_state.json"
        if state_path.is_file() and json.loads(state_path.read_text()).get(
            "gates", {}
        ).get("freeze_locked"):
            raise ValueError("Competition is frozen")
        if backend not in {"local", "modal", "kaggle"}:
            raise ValueError(f"Unsupported execution backend: {backend}")
        is_custom_plugin = self.project.plugin_id not in (None, "default")
        upload_fingerprint = None
        if backend in {"modal", "kaggle"}:
            from harness.remote_runtime import resolve_remote_runtime

            resolve_remote_runtime(self.project)
            from harness.cloud_policy import (
                UploadConsentStore,
                dataset_fingerprint,
                provider_preflight,
            )

            provider = provider_preflight(self.project, backend)
            if not provider.get("cli_installed") or not provider.get("authenticated"):
                raise ValueError(
                    f"{backend} CLI/account is not ready; run the official CLI login, then retry preflight."
                )
            if backend == "modal" and not provider.get("version_supported"):
                raise ValueError(
                    "Modal 1.5.0 or newer is required for the workspace-safe remote runner."
                )
            fingerprint = dataset_fingerprint(self.project)
            task_doc = (
                __import__("yaml").safe_load(
                    self.project.task_config_path.read_text(encoding="utf-8")
                )
                or {}
            )
            task_data = task_doc.get("task", task_doc)
            data_cfg = task_data.get("data_config") or task_data.get("data") or {}
            requires_upload = backend == "modal" or not data_cfg.get(
                "kaggle_dataset_slug"
            )
            if requires_upload and not UploadConsentStore(self.project).is_granted(
                backend, fingerprint["fingerprint"]
            ):
                raise ValueError(
                    f"Dataset upload is not approved for {backend} and the current data fingerprint. "
                    f"Review and run `ai-harness cloud consent --provider {backend} --confirm-upload`."
                )
            upload_fingerprint = fingerprint["fingerprint"] if requires_upload else None
        if is_custom_plugin and not config_path.lower().endswith((".yaml", ".yml")):
            raise ValueError(
                "Task-plugin experiments must use a YAML config through the AI-Harness runner."
            )

        # Resolve experiment ID from config
        import yaml

        full_cfg_path = Path(config_path).expanduser()
        if not full_cfg_path.is_absolute():
            full_cfg_path = self.project_root / full_cfg_path
        full_cfg_path = full_cfg_path.resolve()
        try:
            full_cfg_path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(
                "Experiment config must be inside the attached task workspace."
            ) from exc
        exp_id = Path(config_path).stem
        if full_cfg_path.exists():
            try:
                with open(full_cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
                    exp_id = cfg_data.get("experiment", {}).get("id", exp_id)
            except Exception:
                pass
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", str(exp_id)):
            raise ValueError(
                "Experiment ID must use letters, numbers, dots, underscores, or hyphens and start with a letter/number."
            )

        # If baseline_csv not provided, try to find the best completed node's predictions/oof
        if not baseline_csv:
            from harness.task_spec import load_active_task_spec

            journal = TreeJournal(
                journal_path=self.artifacts_dir / "journal.json",
                artifacts_dir=self.artifacts_dir,
                task_spec=load_active_task_spec(self.project.task_config_path),
            )
            best = journal.get_best_node()
            if best:
                for candidate_name in ("predictions.csv", "oof_predictions.csv"):
                    candidate_file = self.artifacts_dir / best.id / candidate_name
                    if candidate_file.exists():
                        baseline_csv = str(candidate_file)
                        break

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{timestamp_str}_{exp_id}_{__import__('uuid').uuid4().hex[:12]}"
        log_file = self._jobs_dir / f"{job_id}.log"
        import hashlib

        config_sha256 = hashlib.sha256(full_cfg_path.read_bytes()).hexdigest()

        job = CloudJob(
            job_id=job_id,
            backend=backend,
            config_path=config_path,
            exp_id=exp_id,
            status="PENDING",
            started_at=time.time(),
            log_file=str(log_file.relative_to(self.project_root)).replace("\\", "/"),
            auto_audit=auto_audit and not is_custom_plugin,
            auto_sync=auto_sync,
            baseline_csv=baseline_csv,
            artifact_dir=str(
                (self.artifacts_dir / exp_id).relative_to(self.project_root)
            ).replace("\\", "/"),
            task_id=self.project.manifest.get("task_id")
            or self.project.plugin_id
            or "task",
            config_sha256=config_sha256,
            upload_fingerprint=upload_fingerprint,
            gpu=gpu if backend in ("modal", "kaggle") else None,
        )
        self._jobs[job_id] = job
        self._persist_job(job)

        # Construct execution command
        cmd: List[str] = [sys.executable, "-u"]
        if backend in ("modal", "kaggle"):
            cmd.extend(
                [
                    str(_find_script("harness/cloud_runner.py")),
                    "--backend",
                    backend,
                    "--config",
                    str(full_cfg_path),
                    "--project-root",
                    str(self.project_root),
                    "--task-config",
                    str(self.project.task_config_path),
                    "--job-id",
                    job_id,
                ]
            )
            if backend == "modal" and gpu:
                cmd.extend(["--gpu", gpu])
        elif backend == "local":
            if config_path.endswith(".py"):
                script_path = (
                    self.project_root / config_path
                    if not Path(config_path).is_absolute()
                    else Path(config_path)
                )
                cmd.append(str(script_path))
            else:
                cmd.extend(
                    [
                        "-m",
                        "harness.runner",
                        "--project-root",
                        str(self.project_root),
                        "--config",
                        str(full_cfg_path),
                        "--task-config",
                        str(self.project.task_config_path),
                        "--job-id",
                        job_id,
                    ]
                )
        else:
            raise ValueError(f"Unsupported execution backend: {backend}")

        if extra_args:
            cmd.extend(extra_args)

        # Write initial header to log file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== AI-Harness Cloud Job: {job_id} ===\n")
            f.write(
                f"Backend: {backend} | Config: {config_path} | Target ID: {exp_id}\n"
            )
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started at: {time.ctime(job.started_at)}\n")
            f.write(
                "=================================================================\n\n"
            )

        # Launch process with stdout/stderr appended to log file
        log_handle = open(log_file, "a", encoding="utf-8", buffering=1)
        child_env = os.environ.copy()
        python_bin = str(Path(sys.executable).parent)
        child_env["PATH"] = python_bin + os.pathsep + child_env.get("PATH", "")
        child_env["HARNESS_PROJECT_ROOT"] = str(self.project_root)
        child_env["HARNESS_TASK_CONFIG"] = str(self.project.task_config_path)
        child_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(HARNESS_DIR), child_env.get("PYTHONPATH", "")) if p
        )
        popen_kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            )
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(self.project_root),
            text=True,
            env=child_env,
            **popen_kwargs,
        )
        job.pid = proc.pid
        job.status = "RUNNING"
        self._persist_job(job)
        self._processes[job_id] = proc

        # Monitor thread
        monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(job_id, proc, log_handle),
            daemon=True,
        )
        monitor_thread.start()

        return job

    def _monitor_process(
        self, job_id: str, proc: subprocess.Popen, log_handle: Any
    ) -> None:
        """Background thread monitoring process termination and running post-processing."""
        try:
            retcode = proc.wait()
        except Exception:
            retcode = -1
        finally:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

        job = self._jobs.get(job_id)
        if not job:
            return

        job.ended_at = time.time()
        job.return_code = retcode

        if job.status == "CANCELLED":
            # Was manually stopped
            pass
        elif retcode == 0:
            try:
                if job.backend in {"modal", "kaggle"}:
                    self._verify_remote_artifacts(job)
                job.status = "COMPLETED"
                job.provider_status = "COMPLETED"
                self._handle_post_run(job)
            except Exception as exc:
                job.status = "FAILED"
                job.error_message = f"Post-run verification failed: {exc}"
        else:
            job.status = "FAILED"
            worker_path = self._jobs_dir / f"{job.job_id}.worker.json"
            try:
                worker_state = json.loads(worker_path.read_text(encoding="utf-8"))
                job.provider_status = (
                    worker_state.get("provider_status") or job.provider_status
                )
                job.remote_id = worker_state.get("remote_id") or job.remote_id
                job.error_message = (
                    worker_state.get("error_message")
                    or f"Process exited with non-zero exit code: {retcode}"
                )
            except (OSError, json.JSONDecodeError):
                job.error_message = f"Process exited with non-zero exit code: {retcode}"
        self._persist_job(job)

        # Clean up process handle
        self._processes.pop(job_id, None)

    def _verify_remote_artifacts(self, job: CloudJob) -> None:
        artifact_dir = self.project_root / job.artifact_dir
        manifest_path = artifact_dir / "artifact_manifest.json"
        metrics_path = artifact_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            raise ValueError(
                "Remote completion has no identity manifest and metrics file."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "job_id": job.job_id,
            "run_id": job.job_id,
            "task_id": job.task_id,
            "exp_id": job.exp_id,
            "config_sha256": job.config_sha256,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if value is not None and manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Artifact identity mismatch: {mismatches}")
        for filename, expected_hash in (manifest.get("files") or {}).items():
            path = artifact_dir / filename
            if not path.is_file():
                raise ValueError(f"Manifest artifact is missing: {filename}")
            digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Artifact SHA-256 mismatch: {filename}")
        from harness.artifacts import validate_run
        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.project import HarnessProject
        from harness.task_spec import load_active_task_spec

        project = HarnessProject.discover(self.project_root)
        spec = load_active_task_spec(project.task_config_path)
        plugin = load_task_plugin(
            project.plugin_id,
            TaskPluginContext(
                project.root,
                project.task_config_path,
                spec,
                project.plugin_id or "default",
            ),
        )
        validate_run(artifact_dir, task_spec=spec, plugin=plugin, expected=expected)
        job.artifact_identity_verified = True
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            job.result_metrics = (
                metrics if isinstance(metrics, dict) else {"folds": metrics}
            )
        except (OSError, json.JSONDecodeError):
            job.result_metrics = {}
        job.artifact_paths = [
            str((artifact_dir / filename).relative_to(self.project_root)).replace(
                "\\", "/"
            )
            for filename in (manifest.get("files") or {})
        ]

    def _handle_post_run(self, job: CloudJob) -> None:
        """Run deterministic audit gate and solution tree synchronization."""
        exp_dir = self.project_root / job.artifact_dir
        candidate_csv = exp_dir / "predictions.csv"
        if not candidate_csv.exists():
            candidate_csv = exp_dir / "oof_predictions.csv"
        log_path = self.project_root / job.log_file

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                "\n\n=================================================================\n"
            )
            f.write("🏁 Execution Finished. Initiating Post-Flight Verification...\n")

            if not candidate_csv.exists():
                f.write(
                    f"⚠️ Warning: Candidate predictions '{candidate_csv}' not found.\n"
                )
                f.write(
                    "Solution Tree sync will proceed using summary metrics if available.\n"
                )
            elif job.auto_audit:
                f.write("🛡️ Running SWE-Agent Deterministic Audit Gate...\n")
                try:
                    # Audit outputs live OUTSIDE the sealed run directory —
                    # CORRECTNESS_CONTRACT §3: completed run evidence folders
                    # are strictly immutable. The job record keeps the receipt.
                    audit_dir = (
                        HarnessProject.discover(self.project_root).artifacts_dir
                        .parent
                        / "audit"
                        / job.job_id
                    )
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    audit_zip = audit_dir / "candidate-audit.zip"
                    audit_receipt = audit_dir / "AUDIT_RECEIPT.json"
                    audit_cmd = [
                        sys.executable,
                        str(_find_script("tools/audit_gate.py")),
                        "--candidate",
                        str(candidate_csv),
                        "--output-zip",
                        str(audit_zip),
                        "--receipt",
                        str(audit_receipt),
                    ]
                    if job.baseline_csv and Path(job.baseline_csv).exists():
                        audit_cmd.extend(["--baseline", job.baseline_csv])

                    audit_proc = subprocess.run(
                        audit_cmd,
                        capture_output=True,
                        text=True,
                        cwd=str(self.project_root),
                        check=False,
                    )
                    f.write(audit_proc.stdout)
                    if audit_proc.stderr:
                        f.write(audit_proc.stderr)

                    job.audit_passed = audit_proc.returncode == 0

                    # Check for receipt
                    if audit_receipt.exists():
                        try:
                            with open(audit_receipt, "r", encoding="utf-8") as rf:
                                job.audit_receipt = json.load(rf)
                        except Exception:
                            pass

                    if job.audit_passed:
                        f.write(
                            "✅ Audit Gate PASSED! Meaningful effect size & label integrity verified.\n"
                        )
                    else:
                        f.write(
                            "❌ Audit Gate REJECTED the predictions (insufficient effect size or integrity error).\n"
                        )
                except Exception as audit_err:
                    f.write(f"⚠️ Audit Gate error: {audit_err}\n")
                    job.audit_passed = False
            elif self.project.plugin_id not in (None, "default"):
                f.write(
                    f"ℹ️ Generic submission audit skipped for task plugin '{self.project.plugin_id}'. "
                    "Use the task plugin's declared submission validator with test-set predictions.\n"
                )

            # Solution tree synchronization
            if job.auto_sync:
                f.write("🌳 Synchronizing Artifacts with Solution Tree DAG...\n")
                try:
                    from harness.task_spec import load_active_task_spec

                    journal = TreeJournal(
                        journal_path=self.artifacts_dir / "journal.json",
                        artifacts_dir=self.artifacts_dir,
                        task_spec=load_active_task_spec(self.project.task_config_path),
                    )
                    synced = journal.sync_from_artifacts(self.artifacts_dir)
                    f.write(f"✅ Solution Tree updated: {synced} new node(s) synced.\n")
                except Exception as sync_err:
                    f.write(f"⚠️ Solution tree sync failed: {sync_err}\n")

            f.write(
                "=================================================================\n"
            )

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Terminate a running job."""
        job = self._jobs.get(job_id)
        if not job:
            return {"success": False, "error": f"Job '{job_id}' not found."}

        proc = self._processes.get(job_id)
        process_running = bool(proc and proc.poll() is None) or self._pid_running(
            job.pid
        )
        if process_running:
            if job.backend in {"kaggle", "modal"}:
                return {
                    "success": False,
                    "error": (
                        "The monitor cannot prove remote cancellation. The harness is still monitoring it; "
                        "stop it in Kaggle if the provider offers that control."
                    ),
                    "job_id": job_id,
                    "status": job.status,
                }
            job.status = "CANCELLED"
            job.ended_at = time.time()
            job.provider_status = "CANCELLED"
            job.error_message = None
            try:
                pid = proc.pid if proc else job.pid
                if sys.platform == "win32" and pid:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)], check=False
                    )
                else:
                    os.killpg(int(pid), signal.SIGTERM)
            except Exception:
                pass
            from harness.job_state import update_worker_state

            update_worker_state(
                self.project_root,
                job_id,
                status="CANCELLED",
                provider_status=job.provider_status,
                remote_id=job.remote_id,
                error_message=job.error_message,
            )
            self._persist_job(job)
            return {"success": True, "job_id": job_id, "status": "CANCELLED"}

        return {"success": False, "error": "Job is not running."}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job:
            job = self._refresh_saved_job(job)
        return job.to_dict() if job else None

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        sorted_jobs = sorted(
            (self._refresh_saved_job(job) for job in self._jobs.values()),
            key=lambda j: j.started_at,
            reverse=True,
        )
        return [j.to_dict() for j in sorted_jobs[:limit]]

    def stream_logs(
        self, job_id: str, start_byte: int = 0
    ) -> Generator[str, None, None]:
        """Yield SSE events line by line as log file grows."""
        job = self._jobs.get(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        log_path = self.project_root / job.log_file
        # Wait until file exists
        waited = 0
        while not log_path.exists() and waited < 10:
            time.sleep(0.2)
            waited += 0.2

        if not log_path.exists():
            yield f"event: error\ndata: {json.dumps({'error': 'Log file not created'})}\n\n"
            return

        # Binary mode is required: f.seek() with a non-zero offset is only
        # defined for binary streams (text-mode seek is undefined behaviour
        # on CPython and corrupts resume positions).
        with open(log_path, "rb") as f:
            if start_byte > 0:
                f.seek(start_byte)

            while True:
                line = f.readline()
                if line:
                    event_data = {
                        "job_id": job_id,
                        "status": job.status,
                        "text": line.decode("utf-8", errors="replace"),
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                else:
                    # Check if process is still running
                    if job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                        # Read any final trailing lines
                        trailing = f.read()
                        if trailing:
                            for line in trailing.splitlines(keepends=True):
                                event_data = {
                                    "job_id": job_id,
                                    "status": job.status,
                                    "text": line,
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                        # Send completion event
                        final_data = {
                            "job_id": job_id,
                            "status": job.status,
                            "return_code": job.return_code,
                            "audit_passed": job.audit_passed,
                            "audit_receipt": job.audit_receipt,
                        }
                        yield f"event: end\ndata: {json.dumps(final_data)}\n\n"
                        break
                    time.sleep(0.3)
