"""Durable worker-side updates for jobs that outlive the dashboard process."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from harness.io_utils import atomic_write_json
from harness.project import HarnessProject


JOB_ID_PATTERN = re.compile(r"^job_[A-Za-z0-9_.-]{1,180}$")
WORKER_STATUS_VALUES = {"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"}


def worker_state_path(project: HarnessProject, job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Job ID contains unsupported characters.")
    return project.artifacts_dir / "jobs" / f"{job_id}.worker.json"


def update_worker_state(
    project_root: Path | str,
    job_id: str,
    *,
    status: str = "RUNNING",
    provider_status: Optional[str] = None,
    remote_id: Optional[str] = None,
    result_metrics: Optional[Dict[str, Any]] = None,
    artifact_paths: Optional[list[str]] = None,
    error_message: Optional[str] = None,
    exp_id: Optional[str] = None,
    config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically merge a worker update so dashboard restarts retain job progress."""
    normalized_status = status.upper()
    if normalized_status not in WORKER_STATUS_VALUES:
        raise ValueError(f"Unsupported worker job status: {status}")
    project = HarnessProject.discover(project_root)
    path = worker_state_path(project, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    from harness.state_store import transaction
    with transaction(path) as value:
        value.update({
            "job_id": job_id,
            "task_id": project.manifest.get("task_id") or project.plugin_id or "task",
            "status": normalized_status,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        if provider_status is not None:
            value["provider_status"] = str(provider_status).upper()
        if remote_id is not None:
            value["remote_id"] = remote_id
        if result_metrics is not None:
            value["result_metrics"] = result_metrics
        if artifact_paths is not None:
            value["artifact_paths"] = artifact_paths
        if error_message is not None:
            value["error_message"] = error_message
        if exp_id is not None:
            value["exp_id"] = exp_id
        if config_sha256 is not None:
            value["config_sha256"] = config_sha256
    return value
