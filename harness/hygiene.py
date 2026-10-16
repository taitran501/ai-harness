"""Workspace hygiene scan — advisory clutter detection for the context HUD.

Motivation (agent friction audit, Oct 2026): agents repeatedly dumped scratch
files and data dumps into the competition workspace root, which the user
flagged as clutter ("đặt mọi thứ ở root folder làm rối quá"). This module
provides a deterministic, read-only scan that the Context HUD surfaces as a
compact advisory block. It never deletes anything and never blocks actions —
it only names what is out of place so an agent (or the user) can clean up.

Detection rules (all advisory):
- ``scratch_file``: top-level files matching scratch patterns (``_*``,
  ``~$*``, ``*.tmp``, ``*.bak``, ``*.swp``, ``*.orig``, ``.DS_Store``).
- ``stray_data_file``: top-level CSV/parquet files — datasets belong in
  ``data/`` (or the mounted data dir), not the workspace root.
- ``unknown_dir``: top-level non-hidden directories outside the allow-list.
  The allow-list is generous; anything genuinely new shows up here for a
  human review rather than silently accumulating.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

# Top-level directories considered legitimate workspace structure.
ALLOWED_TOP_DIRS = {
    "configs",
    "src",
    "scripts",
    "docs",
    "tests",
    "tools",
    "submissions",
    "data",
    "notebooks",
    "models",
    "outputs",
    "results",
    "artifacts",
    "kernels",
    "kaggle",
    "ai-harness",
    "research",
    "prompts",
    ".ai-harness",
    ".ml-harness",
    ".agents",
}

_SCRATCH_SUFFIXES = (".tmp", ".bak", ".swp", ".orig")
_SCRATCH_STEM_PREFIXES = ("_", "~$")


def _is_scratch_file(path: Path) -> bool:
    name = path.name
    if name.startswith("~$"):
        return True
    if name.lower().endswith(_SCRATCH_SUFFIXES):
        return True
    stem = path.stem
    if stem.startswith("_") and len(stem) > 1:
        return True
    return False


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}GB"


def scan_workspace_hygiene(
    project_root: Path,
    max_findings: int = 12,
) -> Dict[str, object]:
    """Scan a competition workspace root for clutter.

    Returns ``{"clean": bool, "findings": [{"kind", "path", "detail"}],
    "summary": str}``. Deterministic and read-only; failures to stat are
    skipped, never raised.
    """
    project_root = Path(project_root)
    findings: List[Dict[str, str]] = []
    try:
        entries = sorted(project_root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return {"clean": True, "findings": [], "summary": "unreadable"}

    for entry in entries:
        if entry.name.startswith("."):
            continue  # hidden state dirs are harness-managed
        try:
            if entry.is_dir():
                if entry.name.lower() not in ALLOWED_TOP_DIRS:
                    try:
                        n_files = sum(1 for _ in entry.rglob("*") if _.is_file())
                    except OSError:
                        n_files = -1
                    detail = f"top-level dir outside allow-list ({n_files} files)"
                    if n_files == -1:
                        detail = "top-level dir outside allow-list"
                    findings.append(
                        {"kind": "unknown_dir", "path": entry.name, "detail": detail}
                    )
            else:
                if _is_scratch_file(entry):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    findings.append(
                        {
                            "kind": "scratch_file",
                            "path": entry.name,
                            "detail": f"scratch/temp file ({_human_size(size)})",
                        }
                    )
                elif entry.suffix.lower() in {".csv", ".parquet", ".feather"}:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    findings.append(
                        {
                            "kind": "stray_data_file",
                            "path": entry.name,
                            "detail": (
                                f"data file in workspace root ({_human_size(size)}); "
                                "move to data/ or ignore it"
                            ),
                        }
                    )
        except OSError:
            continue
        if len(findings) >= max_findings:
            break

    clean = not findings
    summary = "workspace root clean" if clean else f"{len(findings)} hygiene issue(s)"
    return {"clean": clean, "findings": findings, "summary": summary}


def hygiene_hud_lines(hygiene: Optional[Dict[str, object]], limit: int = 6) -> List[str]:
    """Render the hygiene scan into compact HUD lines (empty list when clean)."""
    if not hygiene or not hygiene.get("findings"):
        return []
    lines = [f"- ⚠️ {hygiene.get('summary')} — move clutter out of the workspace root:"]
    for item in list(hygiene.get("findings") or [])[:limit]:
        lines.append(f"  - `{item['path']}` ({item['kind']}): {item['detail']}")
    more = len(hygiene.get("findings") or []) - limit
    if more > 0:
        lines.append(f"  - … and {more} more")
    return lines
