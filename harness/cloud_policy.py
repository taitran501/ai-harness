"""Cloud preflight and per-task/provider dataset upload consent."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from harness.io_utils import atomic_write_json
from harness.project import HarnessProject
from harness.data_bundle import _files_under, declared_data_roots

_USAGE_CACHE_TTL_SECONDS = 120
_USAGE_CACHE: Dict[tuple[str, str], tuple[float, Dict[str, Any]]] = {}
_USAGE_CACHE_LOCK = threading.Lock()


def dataset_fingerprint(project: HarnessProject) -> Dict[str, Any]:
    """Hash the declared training/test data and split files for upload consent scoping."""
    files: Dict[str, str] = {}
    roots = declared_data_roots(project)
    for path, rel in _files_under(project.root, roots):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[rel] = digest.hexdigest()
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "file_count": len(files),
        "task_id": project.manifest.get("task_id"),
    }


def private_upload_resource_names(
    project: HarnessProject,
    *,
    fingerprint_data: Optional[Dict[str, Any]] = None,
    kaggle_username: Optional[str] = None,
) -> Dict[str, Any]:
    """Return collision-resistant, task-scoped names for private dataset resources."""
    fingerprint_data = fingerprint_data or dataset_fingerprint(project)
    task_id = str(project.manifest.get("task_id") or project.plugin_id or "task")
    task_slug = re.sub(r"[^a-z0-9-]+", "-", task_id.lower()).strip("-") or "task"
    fingerprint = str(fingerprint_data["fingerprint"])
    suffix = fingerprint[:12]
    dataset_name = f"mlh-{task_slug[:20]}-{suffix}-data"
    volume_name = f"mlh-{task_slug[:24]}-{suffix}-data-vol"
    username = str(kaggle_username or "").strip()
    if username and not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Kaggle CLI did not return a valid account username; authenticate through the official CLI first.")
    return {
        "task_slug": task_slug,
        "dataset_fingerprint": fingerprint,
        "kaggle_dataset_name": dataset_name,
        "kaggle_dataset_slug": f"{username}/{dataset_name}" if username else None,
        "modal_volume_name": volume_name,
    }


def _provider_usage(provider: str, account: str, authenticated: bool) -> Dict[str, Any]:
    """Read best-effort account usage using the provider's official CLI JSON output."""
    if not authenticated:
        return {"status": "unavailable", "reason": "Connect an account through the official CLI to read provider usage."}
    cache_key = (provider, account or "default")
    now = time.monotonic()
    with _USAGE_CACHE_LOCK:
        cached = _USAGE_CACHE.get(cache_key)
        if cached and now - cached[0] < _USAGE_CACHE_TTL_SECONDS:
            return dict(cached[1])

    if provider == "kaggle":
        command = ["kaggle", "quota", "--json"]
    else:
        command = ["modal", "billing", "summary", "--json"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Provider returned an error.").strip()[:500]
            fallback = ["kaggle", "quota"] if provider == "kaggle" else ["modal", "billing", "summary"]
            fallback_result = subprocess.run(fallback, capture_output=True, text=True, timeout=8, check=False)
            if fallback_result.returncode == 0 and (fallback_result.stdout or "").strip():
                value = {
                    "status": "available_text",
                    "command": " ".join(fallback),
                    "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "raw": (fallback_result.stdout or "").strip()[:4000],
                }
                if provider == "modal":
                    value.update({
                        "period": "current month",
                        "note": "This is a billing summary, not a remaining quota or spending limit.",
                    })
            else:
                fallback_detail = (fallback_result.stderr or fallback_result.stdout or detail).strip()[:500]
                value = {
                    "status": "unavailable",
                    "command": " ".join(command),
                    "reason": fallback_detail or detail,
                }
        else:
            try:
                payload = json.loads(result.stdout or "null")
            except json.JSONDecodeError:
                payload = None
            if payload is None:
                value = {
                    "status": "unavailable",
                    "command": " ".join(command),
                    "reason": "The CLI did not return valid JSON usage data.",
                }
            else:
                value = {
                    "status": "available",
                    "command": " ".join(command),
                    "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "data": payload,
                }
                if provider == "modal":
                    value.update({
                        "period": "current month",
                        "note": "This is a billing summary, not a remaining quota or spending limit.",
                    })
    except Exception as exc:
        value = {"status": "unavailable", "command": " ".join(command), "reason": str(exc)[:500]}

    with _USAGE_CACHE_LOCK:
        _USAGE_CACHE[cache_key] = (now, dict(value))
    return value


class UploadConsentStore:
    def __init__(self, project: HarnessProject):
        self.project = project
        self.path = project.artifacts_dir / "provider_upload_consents.json"

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "consents": []}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "consents": []}
        return value if isinstance(value, dict) else {"schema_version": 1, "consents": []}

    def is_granted(self, provider: str, fingerprint: str) -> bool:
        return any(
            item.get("provider") == provider and item.get("fingerprint") == fingerprint and item.get("granted") is True
            for item in self._load().get("consents", [])
        )

    def grant(self, provider: str, fingerprint_data: Dict[str, Any]) -> Dict[str, Any]:
        state = self._load()
        record = {
            "provider": provider,
            "task_id": fingerprint_data.get("task_id"),
            "fingerprint": fingerprint_data["fingerprint"],
            "file_count": fingerprint_data["file_count"],
            "granted": True,
            "granted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "visibility": "private",
        }
        consents = [
            item for item in state.get("consents", [])
            if not (item.get("provider") == provider and item.get("fingerprint") == record["fingerprint"])
        ]
        consents.append(record)
        state["consents"] = consents
        atomic_write_json(self.path, state, indent=2, ensure_ascii=False)
        return record

    def status(self) -> Dict[str, Any]:
        fingerprint = dataset_fingerprint(self.project)
        providers = {}
        for provider in ("kaggle", "modal"):
            providers[provider] = {
                "dataset_fingerprint": fingerprint["fingerprint"],
                "file_count": fingerprint["file_count"],
                "upload_allowed": self.is_granted(provider, fingerprint["fingerprint"]),
                "default_visibility": "private",
            }
        return {"providers": providers, "consents_path": str(self.path)}


def provider_preflight(project: HarnessProject, provider: str) -> Dict[str, Any]:
    """Report official CLI/account readiness and non-binding cost/network constraints."""
    if provider not in {"kaggle", "modal"}:
        raise ValueError(f"Unsupported execution provider: {provider}")
    from harness.environment_detector import EnvironmentDetector

    status = EnvironmentDetector.detect_kaggle() if provider == "kaggle" else EnvironmentDetector.detect_modal()
    command = [provider, "--version"]
    version = None
    version_error = None
    version_supported = True
    if status.get("installed"):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            version = (result.stdout or result.stderr).strip()[:500] if result.returncode == 0 else None
            if result.returncode != 0:
                version_error = (result.stderr or result.stdout).strip()[:500]
        except Exception as exc:
            version_error = str(exc)
    if provider == "modal":
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version or "")
        parsed_version = tuple(int(part) for part in match.groups()) if match else None
        version_supported = parsed_version is not None and parsed_version >= (1, 5, 0)
    authenticated = bool(status.get("authenticated") if provider == "kaggle" else status.get("has_credentials"))
    account = status.get("username") or status.get("active_profile") or "default"
    usage = _provider_usage(
        provider,
        str(account),
        authenticated=bool(status.get("installed")) and authenticated and version_supported,
    )
    fingerprint = dataset_fingerprint(project)
    consent = UploadConsentStore(project).is_granted(provider, fingerprint["fingerprint"])
    task_doc = yaml.safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
    task = task_doc.get("task", task_doc)
    internet = task.get("runtime", {}).get("enable_internet")
    warnings = ["Provider quotas, availability, and cost depend on the connected account and can change; check the provider before dispatch."]
    if internet is False and provider == "kaggle":
        warnings.append("Task runtime explicitly disables internet access; Kaggle kernels will also have internet disabled.")
    if provider == "modal" and not version_supported:
        warnings.append("ML-Harness cloud jobs require Modal CLI/SDK version 1.5.0 or newer; upgrade the official Modal CLI.")
    return {
        "provider": provider,
        "cli_installed": bool(status.get("installed")),
        "cli_executable": status.get("executable"),
        "authenticated": bool(status.get("authenticated") if provider == "kaggle" else status.get("has_credentials")),
        "account": status.get("username") or status.get("active_profile"),
        "cli_version": version,
        "version_error": version_error,
        "version_supported": version_supported,
        "quota": usage if provider == "kaggle" else {
            "status": "not_exposed",
            "reason": "The Modal CLI billing summary does not report a remaining execution quota.",
        },
        "billing": usage if provider == "modal" else None,
        "dataset": {
            "fingerprint": fingerprint["fingerprint"],
            "file_count": fingerprint["file_count"],
            "upload_consent": consent,
            "visibility": "private",
        },
        "internet": internet,
        "warnings": warnings,
        "ready": bool(status.get("installed")) and bool(
            status.get("authenticated") if provider == "kaggle" else status.get("has_credentials")
        ) and consent and version_supported,
    }
