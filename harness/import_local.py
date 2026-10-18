"""Import verified local run evidence into the attached artifact tree.

Covers the adoption gap where an agent (or operator) executed a harness
experiment on a remote backend, downloaded the outputs into a local
directory (``kernel_output_*``, ``kaggle_results``, ...) but never ran the
verified import. Without import the run is invisible to the context HUD,
research memory and the audit pipeline; the next agent conversation starts
from zero and may hallucinate or redo past work.

Import is evidence-safe:

- only manifest v2 ``COMPLETED`` ``full`` runs are accepted;
- every manifest file hash is re-verified before anything is copied
  (content integrity, tamper-safe);
- contract identity (task/scorer) is compared against the currently
  attached task spec. A match imports into the live artifact tree as
  first-class evidence; a mismatch quarantines the run under
  ``artifacts/<task>/quarantine/<exp_id>`` with an ``IMPORT_RECEIPT.json``
  recording the drift — visible to the next agent, never mixed into the
  current CV/audit pipeline;
- existing evidence directories are never overwritten.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.artifacts import contract_identity, sha256

#: Manifest identity fields compared against the attached task spec.
IDENTITY_FIELDS = ("task_sha256", "scorer_sha256")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_spec_and_plugin(project_root: Path):
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.project import HarnessProject
    from harness.task_spec import load_active_task_spec

    project = HarnessProject.discover(project_root)
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
    return project, spec, plugin


def _identity_drift(manifest: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, str]:
    """Fields present in both where the values differ."""
    drift: Dict[str, str] = {}
    for field in IDENTITY_FIELDS:
        if manifest.get(field) and current.get(field) and manifest[field] != current[field]:
            drift[field] = f"manifest={str(manifest[field])[:12]}… current={str(current[field])[:12]}…"
    return drift


def _content_verified(source: Path, manifest: Dict[str, Any]) -> Optional[str]:
    """Return an error string if any manifest file is missing or hash-mismatched."""
    for name, value in (manifest.get("files") or {}).items():
        name_str = str(name)
        parts = Path(name_str).parts
        # Safe relative paths only: nested keys (e.g. `stages/train.log`) are
        # allowed by the recursive sealing contract; traversal is not.
        if not parts or Path(name_str).is_absolute() or ".." in parts:
            return f"Manifest file name unsafe: {name}"
        path = source / name_str
        if not path.is_file():
            return f"Manifest file missing: {name}"
        if sha256(path) != value:
            return f"Manifest file hash mismatch: {name}"
    for required in ("run_config.yaml", "metrics.json", "artifact_manifest.json"):
        if not (source / required).is_file():
            return f"Required file missing: {required}"
    return None


def _parse_manifest_basic(source: Path) -> Dict[str, Any]:
    path = source / "artifact_manifest.json"
    if not path.is_file():
        return {"ok": False, "error": "missing artifact_manifest.json"}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"manifest not parseable: {exc}"}
    problems = []
    if manifest.get("schema_version") != 2:
        problems.append("schema_version != 2")
    if manifest.get("status") != "COMPLETED":
        problems.append(f"status={manifest.get('status')}")
    if manifest.get("run_kind") != "full":
        problems.append(f"run_kind={manifest.get('run_kind')}")
    for field in ("run_id", "exp_id", "task_sha256", "scorer_sha256"):
        if not manifest.get(field):
            problems.append(f"missing {field}")
    if problems:
        return {"ok": False, "error": ", ".join(problems), "manifest": manifest}
    return {"ok": True, "manifest": manifest}


def scan_importable_runs(project_root: Path) -> List[Dict[str, Any]]:
    """List top-level local run dirs with a manifest, classified for import.

    Each entry: ``dir``, ``exp_id``, ``verdict`` (``importable`` |
    ``quarantine`` | ``already_imported`` | ``invalid``), ``detail``.
    Never raises on individual directories — a broken dir must not blind
    the rest of the scan.
    """
    root = Path(project_root)
    entries: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "artifact_manifest.json").is_file():
            continue
        parsed = _parse_manifest_basic(child)
        entry: Dict[str, Any] = {"dir": child.name}
        if not parsed["ok"]:
            entry.update(verdict="invalid", detail=parsed["error"])
            entries.append(entry)
            continue
        manifest = parsed["manifest"]
        exp_id = manifest["exp_id"]
        entry["exp_id"] = exp_id
        project, spec, plugin = _load_spec_and_plugin(root)
        current = contract_identity(spec, plugin)
        drift = _identity_drift(manifest, current)
        if (project.artifacts_dir / exp_id).is_dir():
            entry.update(verdict="already_imported", detail=str(project.artifacts_dir / exp_id))
        elif (project.artifacts_dir / "quarantine" / exp_id).is_dir():
            entry.update(verdict="already_imported", detail=str(project.artifacts_dir / "quarantine" / exp_id))
        else:
            error = _content_verified(child, manifest)
            if error:
                entry.update(verdict="invalid", detail=error)
            elif drift:
                entry.update(
                    verdict="quarantine",
                    detail=f"identity drift vs current spec: {drift}",
                )
            else:
                entry.update(verdict="importable", detail="")
        entries.append(entry)
    return entries


def import_verified_local_run(
    project_root: Path,
    source: Path,
    *,
    reject_stale: bool = False,
) -> Dict[str, Any]:
    """Import one verified local run directory into the artifact tree.

    Strict match -> ``artifacts/<task>/<exp_id>``.
    Task/scorer identity drift -> ``artifacts/<task>/quarantine/<exp_id>``
    unless ``reject_stale``.
    Returns a result dict; raises ``ValueError`` for unsafe/tampered runs.
    """
    root = Path(project_root)
    source = Path(source)
    if not source.is_dir():
        raise ValueError(f"Source directory not found: {source}")
    parsed = _parse_manifest_basic(source)
    if not parsed["ok"]:
        raise ValueError(f"Rejected: {parsed['error']}")
    manifest = parsed["manifest"]
    exp_id = manifest["exp_id"]

    error = _content_verified(source, manifest)
    if error:
        raise ValueError(f"Rejected (content integrity): {error}")

    project, spec, plugin = _load_spec_and_plugin(root)
    current = contract_identity(spec, plugin)
    drift = _identity_drift(manifest, current)

    quarantine = bool(drift)
    if quarantine and reject_stale:
        raise ValueError(f"Rejected (identity drift): {drift}")
    target = project.artifacts_dir / ("quarantine" if quarantine else "") / exp_id
    if target.exists():
        raise ValueError(f"Target already exists; evidence is never overwritten: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.importing-{os.getpid()}")
    try:
        tmp.mkdir()
        for name in list(manifest.get("files") or {}):
            dest_file = tmp / name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_bytes((source / name).read_bytes())
        receipt = {
            "schema_version": 1,
            "imported_at": _utc_now(),
            "source_dir": str(source),
            "exp_id": exp_id,
            "run_id": manifest.get("run_id"),
            "content_verified": True,
            "identity": {
                field: (
                    "match"
                    if field not in drift
                    else f"drift: manifest={str(manifest.get(field))[:16]} current={str(current.get(field))[:16]}"
                )
                for field in IDENTITY_FIELDS
            },
            "quarantined": quarantine,
            "reason": "identity drift vs attached task spec" if quarantine else "verified strict import",
            "files": list(manifest.get("files") or {}),
        }
        (tmp / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (tmp / "IMPORT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        tmp.rename(target)
    except BaseException:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Post-copy integrity: hashes of the imported files must match the manifest.
    post_error = _content_verified(target, manifest)
    if post_error:
        raise ValueError(f"Post-copy verification failed: {post_error}")

    _append_journal(project, target, quarantine)
    _record_imported_source(project, source, target.name, quarantine)
    return {
        "status": "ok",
        "exp_id": exp_id,
        "target": str(target),
        "quarantined": quarantine,
        "drift": drift,
        "files_imported": len(list(manifest.get("files") or {})) + 2,
    }


def _append_journal(project, target: Path, quarantine: bool) -> None:
    """Append an audit-journal entry for the import (best effort)."""
    try:
        import os

        journal_path = project.artifacts_dir / "journal.json"
        entries: List[Dict[str, Any]] = []
        if journal_path.is_file():
            loaded = json.loads(journal_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                entries = loaded
        entries.append(
            {
                "event": "local_artifact_import",
                "at": _utc_now(),
                "exp_id": target.name,
                "quarantined": quarantine,
                "target": str(target),
                "pid": os.getpid(),
            }
        )
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - journal is advisory; import already verified
        pass


def _record_imported_source(project, source: Path, exp_id: str, quarantine: bool) -> None:
    """Track which workspace source dirs were recovered into the artifact tree.

    Hygiene consults this file so recovered dirs stop being reported as
    bypass violations — they stay visible, labelled as recovered, while the
    un-recovered manual pushes remain flagged."""
    try:
        path = project.artifacts_dir / "imported_sources.json"
        records: List[Dict[str, Any]] = []
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = [r for r in loaded if isinstance(r, dict)]
        if not any(r.get("source_dir") == source.name for r in records):
            records.append(
                {
                    "source_dir": source.name,
                    "source_path": str(source),
                    "exp_id": exp_id,
                    "quarantined": quarantine,
                    "imported_at": _utc_now(),
                }
            )
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - advisory bookkeeping; import already verified
        pass
