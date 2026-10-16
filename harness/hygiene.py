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
    "dataset",
    "splits",
    "notebooks",
    "models",
    "outputs",
    "results",
    "artifacts",
    "kernels",
    "kaggle",
    "ai-harness",
    "research",
    "research_datasets",
    "research_kernels",
    "external_kernels",
    "prompts",
    "scratch",
    "plugin",
    "__pycache__",
    ".ai-harness",
    ".ml-harness",
    ".agents",
}

# Run-output directories produced by Kaggle/cloud dispatches. Individually
# these are legitimate, but dozens of them pile up at the workspace root;
# the scanner aggregates them into ONE advisory instead of 30 lines of noise.
RUN_OUTPUT_DIR_PREFIXES = ("kaggle_sub", "kernel_output")
RUN_OUTPUT_DIR_EXACT = {"kaggle_results", "kaggle_err_log"}


def _is_run_output_dir(name: str) -> bool:
    low = name.lower()
    if low in RUN_OUTPUT_DIR_EXACT:
        return True
    return low.startswith(RUN_OUTPUT_DIR_PREFIXES)

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
    non_run_output = 0
    try:
        entries = sorted(project_root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return {"clean": True, "findings": [], "summary": "unreadable"}

    for entry in entries:
        if entry.name.startswith("."):
            continue  # hidden state dirs are harness-managed
        try:
            if entry.is_dir():
                name_low = entry.name.lower()
                if name_low in ALLOWED_TOP_DIRS:
                    continue
                if _is_run_output_dir(name_low):
                    # Run-output dirs collapse into ONE aggregated advisory
                    # later; count ALL of them (no file-count rglob needed)
                    # without being limited by max_findings.
                    findings.append(
                        {"kind": "unknown_dir", "path": entry.name, "detail": "run-output dir"}
                    )
                    continue
                if non_run_output >= max_findings:
                    break
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
                non_run_output += 1
            else:
                if non_run_output >= max_findings:
                    break
                finding = None
                if _is_scratch_file(entry):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    finding = {
                        "kind": "scratch_file",
                        "path": entry.name,
                        "detail": f"scratch/temp file ({_human_size(size)})",
                    }
                elif entry.suffix.lower() in {".csv", ".parquet", ".feather"}:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    finding = {
                        "kind": "stray_data_file",
                        "path": entry.name,
                        "detail": (
                            f"data file in workspace root ({_human_size(size)}); "
                            "move to data/ or ignore it"
                        ),
                    }
                if finding is not None:
                    findings.append(finding)
                    non_run_output += 1
        except OSError:
            continue

    findings = _aggregate_run_output_dirs(findings)
    clean = not findings
    summary = "workspace root clean" if clean else f"{len(findings)} hygiene issue(s)"
    return {"clean": clean, "findings": findings, "summary": summary}


def _aggregate_run_output_dirs(findings: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Collapse per-directory run-output findings into a single advisory.

    A workspace with 30 ``kaggle_sub_*`` / ``kernel_output*`` directories
    should produce ONE line (with examples + count), not 30 lines that
    dominate the context HUD.
    """
    run_output = [f for f in findings if f["kind"] == "unknown_dir" and _is_run_output_dir(f["path"].lower())]
    others = [f for f in findings if f not in run_output]
    if len(run_output) <= 2:
        return findings
    names = [f["path"] for f in run_output]
    examples = ", ".join(names[:3]) + (f", … +{len(names) - 3} more" if len(names) > 3 else "")
    return [
        {
            "kind": "stale_run_output",
            "path": f"{len(run_output)} dirs",
            "detail": f"Kaggle/kernel run-output dirs at workspace root — move to artifacts/ or delete (e.g. {examples})",
        }
    ] + others


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
