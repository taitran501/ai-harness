"""Append-only dispatch audit + duplicate detection (dispatch governance).

Every harness-native Kaggle dispatch — a submission file
(``KaggleLeaderboardTracker.dispatch``) or a kernel push
(``kernel_dispatch.dispatch_kernel``) — appends one JSONL entry to
``<workspace>/.ai-harness/dispatch_log.jsonl`` carrying the *content hash*
of what was sent. Re-dispatching an identical file to the same competition
(or the same notebook content to the same kernel id) raises an advisory
**duplicate warning** (never a hard block in v1) so accidental
double-submissions — which waste a competition's submission quota — are
visible at the moment they happen, and after the fact, in the log.

Safety contract (mirrors the PR-C constraints):

- **Best-effort.** Resolving or writing the audit must never break a
  dispatch; callers wrap the call in try/except and the resolver itself
  degrades to ``None`` (no-op) when no workspace can be identified.
- **Never the checkout.** The log is written only into a workspace that is
  explicitly targeted (``project_root`` param or ``HARNESS_PROJECT_ROOT``)
  or *attached* (a ``.ai-harness/project.yaml`` manifest in the CWD). The
  installed framework checkout is never a fallback target, so audit code
  cannot pollute the repo (the T1 class of bug).
- **Additive.** The audit log sits beside the run folders, outside every
  sealed bundle; it is append-only and read-tolerant (malformed lines are
  skipped, never raised on).
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_DISPATCH_LOG_NAME = "dispatch_log.jsonl"
_MANIFEST_NAME = Path(".ai-harness") / "project.yaml"
_LEGACY_MANIFEST_NAME = Path(".ml-harness") / "project.yaml"

_KIND_SUBMISSION = "submission"
_KIND_KERNEL = "kernel"


def read_submission_events(project_root):
    """Strict authoritative history; corruption cannot establish an empty portfolio."""
    from harness.policy_schema import validate_document
    path = Path(project_root) / ".ai-harness" / "submission_events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("HISTORY_INCOMPLETE: empty dispatch event")
        record = json.loads(line)
        validate_document("dispatch_event", record)
        events.append(record)
    return events


def append_dispatch_event(project_root, record):
    """Durably append before sending; unlike the v0 advisory log this must succeed."""
    from harness.policy_schema import validate_document
    from harness.state_store import workspace_lock
    validate_document("dispatch_event", record)
    directory = Path(project_root) / ".ai-harness"
    with workspace_lock(directory):
        read_submission_events(project_root)
        with (directory / "submission_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return record


def sha256_file(path: Optional[str | Path]) -> Optional[str]:
    """Stream-hash a local file, or ``None`` when it is missing/unreadable.

    Dispatches are recorded even when the local file has already been
    cleaned up (e.g. a code-competition kernel output that only exists on
    Kaggle); the hash is then ``null`` and duplicate detection simply
    cannot match on it.
    """
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def resolve_log_path(project_root: Optional[str | Path] = None) -> Optional[Path]:
    """Resolve the dispatch audit log path, or ``None`` when there is no
    trustworthy workspace to write into (the no-op case).

    Resolution order:

    1. ``project_root`` (explicit param — programmatic/tests) — trusted.
    2. ``HARNESS_PROJECT_ROOT`` environment variable — trusted.
    3. The CWD, but only when it is an *attached* workspace (has a
       ``.ai-harness/project.yaml`` or legacy ``.ml-harness/project.yaml``
       manifest). A bare directory — and, critically, the installed
       framework checkout — is **not** a valid target.
    """
    root = project_root or os.environ.get("HARNESS_PROJECT_ROOT")
    if root:
        return Path(root).expanduser().resolve() / ".ai-harness" / _DISPATCH_LOG_NAME
    cwd = Path.cwd().resolve()
    if (cwd / _MANIFEST_NAME).is_file() or (cwd / _LEGACY_MANIFEST_NAME).is_file():
        return cwd / ".ai-harness" / _DISPATCH_LOG_NAME
    return None


def read_dispatch_log(log_path: Optional[str | Path]) -> List[Dict[str, Any]]:
    """Read all audit entries; malformed lines are skipped, not raised on."""
    if log_path is None:
        return []
    path = Path(log_path)
    if not path.is_file():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _ledger_file_hashes(root: Path) -> List[Dict[str, Any]]:
    """Hash the files referenced by a workspace's submission ledger (if any).

    Read-only: the registry is scanned at its known locations and each
    entry's ``archive_path`` / ``candidate_path`` is hashed when present.
    This lets the duplicate check also see submissions registered through
    the governed Bridge path (``SubmissionManager``), not just CLI
    dispatches.
    """
    candidates = [
        root / ".ai-harness" / "artifacts" / "submissions" / "registry.json",
        root / "artifacts" / "submissions" / "registry.json",
        root / "submissions" / "registry.json",
    ]
    matches: List[Dict[str, Any]] = []
    for registry in candidates:
        if not registry.is_file():
            continue
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items = data.get("submissions", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("archive_path", "candidate_path"):
                raw = item.get(key)
                if not raw:
                    continue
                sha = sha256_file(Path(raw))
                if sha is None:
                    continue
                matches.append({
                    "sha256": sha,
                    "sub_id": item.get("id", ""),
                    "exp_id": item.get("exp_id", ""),
                    "source": "ledger",
                })
        break  # first registry that exists wins (no double-reporting)
    return matches


def check_duplicate(
    kind: str,
    *,
    competition: Optional[str] = None,
    kernel_id: Optional[str] = None,
    content_sha: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Find a previous dispatch of identical content, if any.

    Returns the most recent matching entry (``source="dispatch_log"``) or
    a ledger match (``source="ledger"``), else ``None``. Matching is
    exact: same content hash AND same competition (submissions) / same
    kernel id (kernels). A different file content or a different
    competition is *not* a duplicate.
    """
    if content_sha is None:
        return None
    log_path = (project_root / ".ai-harness" / _DISPATCH_LOG_NAME) if project_root else None
    if log_path is None:
        log_path = resolve_log_path()
    entries = read_dispatch_log(log_path)
    if kind == _KIND_SUBMISSION:
        for entry in reversed(entries):
            if (
                entry.get("kind") == _KIND_SUBMISSION
                and entry.get("file_sha256") == content_sha
                and entry.get("competition") == competition
            ):
                return {
                    "ref": entry.get("ref"),
                    "competition": entry.get("competition"),
                    "ts": entry.get("ts"),
                    "source": "dispatch_log",
                }
        if project_root is not None:
            for match in _ledger_file_hashes(project_root):
                if match["sha256"] == content_sha:
                    return {
                        "sub_id": match["sub_id"],
                        "exp_id": match["exp_id"],
                        "source": "ledger",
                    }
        return None
    if kind == _KIND_KERNEL:
        for entry in reversed(entries):
            if (
                entry.get("kind") == _KIND_KERNEL
                and entry.get("notebook_sha256") == content_sha
                and entry.get("kernel_id") == kernel_id
            ):
                return {
                    "ref": entry.get("ref"),
                    "kernel_id": entry.get("kernel_id"),
                    "ts": entry.get("ts"),
                    "source": "dispatch_log",
                }
    return None


def record_dispatch(
    *,
    kind: str,
    competition: Optional[str] = None,
    kernel_id: Optional[str] = None,
    file_path: Optional[str | Path] = None,
    notebook_path: Optional[str | Path] = None,
    kernel: Optional[str] = None,
    kernel_version: Optional[str] = None,
    message: str = "",
    ref: Optional[Any] = None,
    project_root: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Append one audit entry for a completed dispatch; ``None`` when there
    is no workspace to write into (best-effort no-op, never raises here —
    write errors propagate to the caller's guard).

    The entry records the content hash of what was sent plus the duplicate
    match (if any) computed *before* this entry is appended, so the log
    stays a faithful, append-only history.
    """
    log_path = resolve_log_path(project_root)
    if log_path is None:
        return None
    root = project_root or os.environ.get("HARNESS_PROJECT_ROOT")
    root = Path(root).expanduser().resolve() if root else Path.cwd().resolve()
    content_sha = (
        sha256_file(file_path) if kind == _KIND_SUBMISSION else sha256_file(notebook_path)
    )
    duplicate = check_duplicate(
        kind,
        competition=competition,
        kernel_id=kernel_id,
        content_sha=content_sha,
        project_root=root,
    )
    entry: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,
        "ref": ref,
        "message": message or "",
        "operator": _safe_user(),
        "cwd": str(Path.cwd().resolve()),
        "duplicate_of": duplicate,
    }
    if kind == _KIND_SUBMISSION:
        entry.update(
            competition=competition,
            file_name=str(file_path) if file_path is not None else None,
            file_sha256=content_sha,
            kernel=kernel,
            kernel_version=kernel_version,
        )
    else:
        entry.update(
            kernel_id=kernel_id,
            notebook_sha256=content_sha,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - auditing must never fail on identity
        return "unknown"
