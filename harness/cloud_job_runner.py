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
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CloudJobManager:
    """Singleton manager for background training jobs and live streaming."""

    _instance: Optional[CloudJobManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> CloudJobManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._jobs: Dict[str, CloudJob] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._jobs_dir = PROJECT_ROOT / "tmp" / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

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
        # Resolve experiment ID from config
        import yaml
        full_cfg_path = PROJECT_ROOT / config_path
        exp_id = Path(config_path).stem
        if full_cfg_path.exists():
            try:
                with open(full_cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
                    exp_id = cfg_data.get("experiment", {}).get("id", exp_id)
            except Exception:
                pass

        # If baseline_csv not provided, try to find the best completed node's predictions/oof
        if not baseline_csv:
            journal = TreeJournal()
            best = journal.get_best_node()
            if best:
                for candidate_name in ("predictions.csv", "oof_predictions.csv"):
                    candidate_file = PROJECT_ROOT / "artifacts" / best.id / candidate_name
                    if candidate_file.exists():
                        baseline_csv = str(candidate_file)
                        break

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{timestamp_str}_{exp_id}"
        log_file = self._jobs_dir / f"{job_id}.log"

        job = CloudJob(
            job_id=job_id,
            backend=backend,
            config_path=config_path,
            exp_id=exp_id,
            status="PENDING",
            started_at=time.time(),
            log_file=str(log_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            auto_audit=auto_audit,
            auto_sync=auto_sync,
            baseline_csv=baseline_csv,
        )
        self._jobs[job_id] = job

        # Construct execution command
        cmd: List[str] = [sys.executable]
        if backend in ("modal", "kaggle"):
            cmd.extend([
                str(PROJECT_ROOT / "harness" / "cloud_runner.py"),
                "--backend",
                backend,
                "--config",
                config_path,
            ])
            if backend == "modal" and gpu:
                cmd.extend(["--gpu", gpu])
        elif backend == "local":
            if config_path.endswith(".py"):
                script_path = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
                cmd.append(str(script_path))
            else:
                cmd.extend([
                    str(PROJECT_ROOT / "train.py"),
                    "--config",
                    config_path,
                ])
        else:
            raise ValueError(f"Unsupported execution backend: {backend}")

        if extra_args:
            cmd.extend(extra_args)

        # Write initial header to log file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== ML-Harness Cloud Job: {job_id} ===\n")
            f.write(f"Backend: {backend} | Config: {config_path} | Target ID: {exp_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started at: {time.ctime(job.started_at)}\n")
            f.write("=================================================================\n\n")

        # Launch process with stdout/stderr appended to log file
        log_handle = open(log_file, "a", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        job.pid = proc.pid
        job.status = "RUNNING"
        self._processes[job_id] = proc

        # Monitor thread
        monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(job_id, proc, log_handle),
            daemon=True,
        )
        monitor_thread.start()

        return job

    def _monitor_process(self, job_id: str, proc: subprocess.Popen, log_handle: Any) -> None:
        """Background thread monitoring process termination and running post-processing."""
        try:
            retcode = proc.wait()
        except Exception as e:
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
            job.status = "COMPLETED"
            # Execute post-run audit gate and tree sync
            self._handle_post_run(job)
        else:
            job.status = "FAILED"
            job.error_message = f"Process exited with non-zero exit code: {retcode}"

        # Clean up process handle
        self._processes.pop(job_id, None)

    def _handle_post_run(self, job: CloudJob) -> None:
        """Run deterministic audit gate and solution tree synchronization."""
        exp_dir = PROJECT_ROOT / "artifacts" / job.exp_id
        candidate_csv = exp_dir / "predictions.csv"
        if not candidate_csv.exists():
            candidate_csv = exp_dir / "oof_predictions.csv"
        log_path = PROJECT_ROOT / job.log_file

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n\n=================================================================\n")
            f.write("🏁 Execution Finished. Initiating Post-Flight Verification...\n")

            if not candidate_csv.exists():
                f.write(f"⚠️ Warning: Candidate predictions '{candidate_csv}' not found.\n")
                f.write("Solution Tree sync will proceed using summary metrics if available.\n")
            elif job.auto_audit:
                f.write(f"🛡️ Running SWE-Agent Deterministic Audit Gate...\n")
                try:
                    audit_cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "tools" / "audit_gate.py"),
                        "--candidate",
                        str(candidate_csv),
                    ]
                    if job.baseline_csv and Path(job.baseline_csv).exists():
                        audit_cmd.extend(["--baseline", job.baseline_csv])

                    audit_proc = subprocess.run(
                        audit_cmd,
                        capture_output=True,
                        text=True,
                        cwd=str(PROJECT_ROOT),
                        check=False,
                    )
                    f.write(audit_proc.stdout)
                    if audit_proc.stderr:
                        f.write(audit_proc.stderr)

                    job.audit_passed = (audit_proc.returncode == 0)

                    # Check for receipt
                    receipt_path = PROJECT_ROOT / "AUDIT_RECEIPT.json"
                    exp_receipt = exp_dir / "AUDIT_RECEIPT.json"
                    if receipt_path.exists():
                        try:
                            with open(receipt_path, "r", encoding="utf-8") as rf:
                                job.audit_receipt = json.load(rf)
                            # Copy receipt into experiment folder
                            import shutil
                            shutil.copy2(receipt_path, exp_receipt)
                        except Exception:
                            pass

                    if job.audit_passed:
                        f.write("✅ Audit Gate PASSED! Meaningful effect size & label integrity verified.\n")
                    else:
                        f.write("❌ Audit Gate REJECTED the predictions (insufficient effect size or integrity error).\n")
                except Exception as audit_err:
                    f.write(f"⚠️ Audit Gate error: {audit_err}\n")
                    job.audit_passed = False

            # Solution tree synchronization
            if job.auto_sync:
                f.write("🌳 Synchronizing Artifacts with Solution Tree DAG...\n")
                try:
                    journal = TreeJournal()
                    synced = journal.sync_from_artifacts()
                    f.write(f"✅ Solution Tree updated: {synced} new node(s) synced.\n")
                except Exception as sync_err:
                    f.write(f"⚠️ Solution tree sync failed: {sync_err}\n")

            f.write("=================================================================\n")

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Terminate a running job."""
        job = self._jobs.get(job_id)
        if not job:
            return {"success": False, "error": f"Job '{job_id}' not found."}

        proc = self._processes.get(job_id)
        if proc and proc.poll() is None:
            job.status = "CANCELLED"
            job.ended_at = time.time()
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], check=False)
                else:
                    proc.terminate()
            except Exception as e:
                pass
            return {"success": True, "job_id": job_id, "status": "CANCELLED"}

        return {"success": False, "error": "Job is not running."}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        sorted_jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
        return [j.to_dict() for j in sorted_jobs[:limit]]

    def stream_logs(self, job_id: str, start_byte: int = 0) -> Generator[str, None, None]:
        """Yield SSE events line by line as log file grows."""
        job = self._jobs.get(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        log_path = PROJECT_ROOT / job.log_file
        # Wait until file exists
        waited = 0
        while not log_path.exists() and waited < 10:
            time.sleep(0.2)
            waited += 0.2

        if not log_path.exists():
            yield f"event: error\ndata: {json.dumps({'error': 'Log file not created'})}\n\n"
            return

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            if start_byte > 0:
                f.seek(start_byte)

            while True:
                line = f.readline()
                if line:
                    event_data = {
                        "job_id": job_id,
                        "status": job.status,
                        "text": line,
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                else:
                    # Check if process is still running
                    if job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                        # Read any final trailing lines
                        trailing = f.read()
                        if trailing:
                            for l in trailing.splitlines(keepends=True):
                                event_data = {"job_id": job_id, "status": job.status, "text": l}
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
