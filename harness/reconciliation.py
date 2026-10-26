"""Reconciliation auditor comparing provider (Kaggle) submission history against the local dispatch ledger.

Guards authority boundary:
- Pulls live submission history from provider.
- Compares against append-only submission_events.jsonl and acknowledged submissions.
- Flags UNEXPLAINED_EXTERNAL_DISPATCH if any submission occurred outside ai-harness.
- Unlocks only with explicit operator acknowledgment.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from harness.dispatch_audit import read_submission_events
from harness.io_utils import atomic_write_json
from harness.policy_schema import validate_document
from harness.project import HarnessProject


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_acknowledged_refs(project_root: Path) -> Set[str]:
    path = project_root / ".ai-harness" / "acknowledged_submissions.json"
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(r) for r in data.get("acknowledged_refs", [])}
    except Exception:
        return set()


def _save_acknowledged_refs(project_root: Path, refs: List[str], reason: str, operator: str) -> None:
    path = project_root / ".ai-harness" / "acknowledged_submissions.json"
    existing = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            existing = []

    current_refs = _load_acknowledged_refs(project_root)
    current_refs.update(str(r) for r in refs)

    record = {
        "acknowledged_refs": sorted(current_refs),
        "history": existing + [
            {
                "acknowledged_at": _utc_now_iso(),
                "refs": [str(r) for r in refs],
                "reason": reason,
                "operator": operator,
            }
        ],
    }
    atomic_write_json(path, record)


def collect_accounted_refs(project_root: Path, competition: str) -> Set[str]:
    """Gather all provider refs that were legitimately authorized through ai-harness."""
    accounted: Set[str] = set()

    # 1. Authoritative submission_events.jsonl
    try:
        events = read_submission_events(project_root)
        for e in events:
            if e.get("competition") == competition and e.get("provider_ref"):
                accounted.add(str(e["provider_ref"]))
    except Exception:
        pass

    # 2. Legacy dispatch_log.jsonl
    legacy_log = project_root / ".ai-harness" / "dispatch_log.jsonl"
    if legacy_log.is_file():
        try:
            for line in legacy_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("competition") == competition and entry.get("ref"):
                    accounted.add(str(entry["ref"]))
        except Exception:
            pass

    # 3. Artifacts submissions registry
    reg_path = project_root / ".ai-harness" / "artifacts"
    for reg in reg_path.glob("**/submissions/registry.json"):
        try:
            records = json.loads(reg.read_text(encoding="utf-8")).get("submissions", [])
            for r in records:
                if r.get("ref"):
                    accounted.add(str(r["ref"]))
        except Exception:
            pass

    # 4. Acknowledged refs
    accounted.update(_load_acknowledged_refs(project_root))

    return accounted


def reconcile_provider_submissions(
    project_root: str | Path,
    competition: str,
    provider: str = "kaggle",
    provider_client: Any = None,
) -> Dict[str, Any]:
    """Reconcile remote provider submissions against local dispatch ledger."""
    project = HarnessProject.discover(Path(project_root).resolve())
    accounted_refs = collect_accounted_refs(project.root, competition)

    remote_subs = None
    if provider == "kaggle":
        client = provider_client
        if client is None:
            try:
                from harness.kaggle_compat import KaggleClient
                client = KaggleClient(project=project)
            except Exception:
                client = None

        if client is not None:
            try:
                remote_subs = client.competition_submissions(competition)
            except Exception:
                # Offline or unauthenticated: a failed query is unknown, never clean.
                remote_subs = None

    unexplained: List[Dict[str, Any]] = []

    if remote_subs is not None:
        for sub in remote_subs:
            ref = getattr(sub, "ref", None) or (sub.get("ref") if isinstance(sub, dict) else None)
            date = getattr(sub, "date", None) or (sub.get("date") if isinstance(sub, dict) else None)
            desc = getattr(sub, "description", "") or (sub.get("description", "") if isinstance(sub, dict) else "")
            fname = getattr(sub, "fileName", "") or (sub.get("fileName", "") if isinstance(sub, dict) else "")

            if ref is not None and str(ref) not in accounted_refs:
                unexplained.append(
                    {
                        "ref": str(ref),
                        "date": str(date) if date else _utc_now_iso(),
                        "description": str(desc),
                        "file_name": str(fname),
                    }
                )

    if remote_subs is None:
        status = "UNKNOWN"
        reason_code = "PROVIDER_UNAVAILABLE"
        total_provider_count = None
    elif unexplained:
        status = "DISCREPANCY_DETECTED"
        reason_code = None
        total_provider_count = len(remote_subs)
    else:
        status = "CLEAN"
        reason_code = None
        total_provider_count = len(remote_subs)

    report = {
        "schema_version": 1,
        "reconciliation_id": "rec-" + uuid.uuid4().hex,
        "competition": competition,
        "provider": provider,
        "provider_submissions_count": total_provider_count,
        "accounted_submissions_count": len(accounted_refs),
        "unexplained_submissions": unexplained,
        "status": status,
        "checked_at": _utc_now_iso(),
    }
    if reason_code is not None:
        report["reason_code"] = reason_code

    validate_document("reconciliation_report", report)

    out_path = project.root / ".ai-harness" / "reconciliation_report.json"
    atomic_write_json(out_path, report)

    return report


def acknowledge_external_submissions(
    project_root: str | Path,
    refs: List[str | int],
    reason: str,
    operator: Optional[str] = None,
) -> Dict[str, Any]:
    """Operator sign-off acknowledging unrecorded external submissions to unblock workspace."""
    project = HarnessProject.discover(Path(project_root).resolve())
    if not str(reason).strip():
        raise ValueError("Operator acknowledgment requires an explicit reason.")

    import getpass

    resolved_op = operator or getpass.getuser() or "operator"
    str_refs = [str(r) for r in refs]

    _save_acknowledged_refs(project.root, str_refs, reason=reason, operator=resolved_op)

    # Re-run reconciliation to update status
    # Read task config to get default competition
    from harness.task_spec import load_active_task_spec

    comp = "competition"
    try:
        spec = load_active_task_spec(project.task_config_path)
        comp = spec.get("competition", {}).get("id") or comp
    except Exception:
        pass

    return reconcile_provider_submissions(project.root, competition=comp)
