"""Opt-in, fail-closed cleanup of framework-owned disposable directories."""

from __future__ import annotations

import math
import os
import stat
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

TEMP_PREFIXES = ("aih-kaggle-job_", "aih-modal-job_", "harness_kernel_", "harness_audit_runs")
PROTECTED_DIRS = {"data", "dataset", "src", "scripts", "configs", "prompts", ".git", ".agents"}
STAGING_MARKER = ".aih-staging-pid"


@dataclass(frozen=True)
class CleanupItem:
    path: Path
    root: Path
    kind: str
    older_than_hours: float
    size_bytes: int
    snapshot: tuple
    workspace_root: Path | None = None


def _age(hours: float, *, temp: bool = False) -> float:
    hours = float(hours)
    if not math.isfinite(hours) or hours < 0:
        raise ValueError("older_than_hours must be finite and nonnegative")
    return max(2.0, hours) if temp else hours


def _no_links(path: Path) -> None:
    """lstat every ancestor; reject junctions and other reparse points too."""
    for entry in (path, *path.parents):
        info = entry.lstat()
        if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"Link/reparse point refused: {entry}")


def _bounded(path: Path, root: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("Path traversal refused")
    _no_links(path)
    actual, base = path.resolve(), root.resolve()
    if actual == base or not actual.is_relative_to(base):
        raise ValueError(f"Outside sandbox or sandbox root: {path}")
    if not actual.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return actual


def _inventory(path: Path) -> tuple:
    """No-follow traversal; reject the whole candidate if any entry is unsafe."""
    rows = []
    pending = [path]
    while pending:
        entry = pending.pop()
        _no_links(entry)
        info = entry.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"Special file refused: {entry}")
        rows.append((entry.relative_to(path).as_posix(), info.st_mode, info.st_dev,
                     info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        if stat.S_ISDIR(info.st_mode):
            pending.extend(entry.iterdir())
    return tuple(sorted(rows))


def _overlaps(a: Path, b: Path) -> bool:
    return a.is_relative_to(b) or b.is_relative_to(a)


def _workspace_protection(root: Path) -> tuple[list[Path], object]:
    try:
        return _read_workspace_protection(root)
    except (AttributeError, TypeError, yaml.YAMLError) as exc:
        raise ValueError("Invalid workspace metadata; cleanup refused") from exc


def _read_workspace_protection(root: Path) -> tuple[list[Path], object]:
    from harness.project import HarnessProject

    for manifest in (root / ".ai-harness/project.yaml", root / ".ml-harness/project.yaml"):
        if manifest.is_file():
            metadata = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Invalid workspace manifest; cleanup refused")
            break
    project = HarnessProject.discover(root)
    if project.manifest_path.is_file() and not project.task_config_path.is_file():
        raise ValueError("Attached task config missing; cleanup refused")
    protected = [root / name for name in PROTECTED_DIRS]
    protected.append(project.task_config_path.resolve())
    if project.manifest_path.is_file():
        protected.append(project.manifest_path.resolve())
    if project.task_config_path.is_file():
        document = yaml.safe_load(project.task_config_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Invalid task config; cleanup refused")
        task = document.get("task", document)
        if not isinstance(task, dict) or any(
                key in task and not isinstance(task[key], dict) for key in ("data", "data_config")):
            raise ValueError("Invalid task data config; cleanup refused")

        def collect(value, data=False):
            if isinstance(value, dict):
                for key, child in value.items():
                    collect(child, data or key in {"data", "data_config"} or
                            str(key).endswith(("_path", "_paths", "_dir", "_file")))
            elif isinstance(value, list):
                for child in value:
                    collect(child, data)
            elif data and isinstance(value, str) and value.strip():
                p = Path(value).expanduser()
                protected.append((p if p.is_absolute() else root / p).resolve())

        collect(document)
    return protected, project


def _live_staging(path: Path) -> bool:
    marker = path / STAGING_MARKER
    if not marker.exists():
        return False
    try:
        _no_links(marker)
        pid = int(marker.read_text(encoding="ascii"))
        if pid <= 0:
            return True
        # os.kill(pid, 0) can terminate a process on Windows; use query-only API.
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                kernel.CloseHandle(handle)
                return True
            return ctypes.get_last_error() != 87  # invalid PID = no process
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError):
        return True


def _validate(item: CleanupItem) -> tuple:
    root = item.root.resolve()
    path = _bounded(item.path, root)
    hours = _age(item.older_than_hours, temp=item.kind == "temp")
    rel = path.relative_to(root)
    if item.kind == "temp":
        if root != Path(tempfile.gettempdir()).resolve() or len(rel.parts) != 1:
            raise ValueError("Invalid system temp sandbox")
        if not path.name.startswith(TEMP_PREFIXES):
            raise ValueError("Unrecognized temp prefix")
        if _live_staging(path):
            raise ValueError("Active staging process")
        if any((path / p).exists() for p in (".git", ".ai-harness", ".ml-harness", "configs/task.yaml")):
            raise ValueError("Temp candidate appears to be a workspace")
        workspace = item.workspace_root or Path(os.environ.get("HARNESS_PROJECT_ROOT", Path.cwd()))
        _no_links(workspace)
        workspace = workspace.resolve()
        protected, _ = _workspace_protection(workspace)
        if any(_overlaps(path, p) for p in [workspace, *protected]):
            raise ValueError("Temp candidate overlaps workspace or declared task data")
    else:
        protected, project = _workspace_protection(root)
        if item.kind == "legacy":
            if rel.parts != (".ml-harness",):
                raise ValueError("Invalid legacy location")
            if project.artifacts_dir.resolve().is_relative_to(path):
                raise ValueError("Legacy artifacts are still active")
        elif item.kind == "probe":
            if (len(rel.parts) != 6 or rel.parts[:2] != (".ai-harness", "artifacts")
                    or rel.parts[3] != "probes"):
                raise ValueError("Invalid probe location")
            manifest_path = path / "artifact_manifest.json"
            if manifest_path.is_file():
                try:
                    import json

                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(manifest, dict) and manifest.get("status") == "RUNNING":
                        raise ValueError("Active probe run (status: RUNNING)")
                except (OSError, json.JSONDecodeError):
                    pass
            if (path / ".state.lock").exists():
                raise ValueError("Active probe run lock")
        else:
            raise ValueError("Unknown cleanup category")
        if any(_overlaps(path, p) for p in protected):
            raise ValueError("Protected workspace path")
    snapshot = _inventory(path)
    if item.kind != "temp":
        for row in snapshot:
            parts = Path(row[0]).parts
            if any(p.casefold() in PROTECTED_DIRS for p in parts):
                raise ValueError("Protected content inside candidate")
    if time.time() - max(row[5] / 1e9 for row in snapshot) <= hours * 3600:
        raise ValueError("Candidate is too recent")
    return snapshot


def _candidate(path: Path, root: Path, kind: str, hours: float, workspace_root=None) -> CleanupItem | None:
    try:
        item = CleanupItem(path, root, kind, hours, 0, (), workspace_root)
        snapshot = _validate(item)
        size = sum(row[4] for row in snapshot if stat.S_ISREG(row[1]))
        return CleanupItem(path, root, kind, hours, size, snapshot, workspace_root)
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _children(path: Path) -> list[Path]:
    try:
        _no_links(path)
        return sorted(path.iterdir())
    except (OSError, ValueError):
        return []


def scan_workspace_clutter(project_root, older_than_hours=2) -> list[CleanupItem]:
    """Inventory only inactive legacy and individual old probe-run directories."""
    hours = _age(older_than_hours)
    root = Path(project_root).absolute()
    _no_links(root)
    root = root.resolve()
    paths = [(root / ".ml-harness", "legacy")]
    for task in _children(root / ".ai-harness" / "artifacts"):
        for experiment in _children(task / "probes"):
            for run in _children(experiment):
                paths.append((run, "probe"))
    return [item for path, kind in paths if (item := _candidate(path, root, kind, hours))]


def scan_system_temp(older_than_hours=2, *, project_root=None) -> list[CleanupItem]:
    """Inventory exact framework prefixes in system temp, with a two-hour floor."""
    hours = _age(older_than_hours, temp=True)
    root = Path(tempfile.gettempdir()).resolve()
    workspace = Path(project_root or os.environ.get("HARNESS_PROJECT_ROOT", Path.cwd())).absolute()
    return [item for path in _children(root) if path.name.startswith(TEMP_PREFIXES)
            if (item := _candidate(path, root, "temp", hours, workspace))]


def _archive(item: CleanupItem) -> Path:
    dest = item.root / ".ai-harness" / "archives"
    protected, _ = _workspace_protection(item.root)
    if any(_overlaps(dest, p) for p in protected):
        raise ValueError("Archive destination overlaps protected workspace content")
    # Check existing ancestors before mkdir: never write through a junction.
    ancestor = dest
    while not ancestor.exists():
        ancestor = ancestor.parent
    _no_links(ancestor)
    dest.mkdir(parents=True, exist_ok=True)
    _bounded(dest, item.root)
    target = dest / f"legacy-{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in item.snapshot:
            source = item.path / row[0]
            _no_links(source)
            archive.write(source, (Path(item.path.name) / row[0]).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    return target


def _delete_tree(item: CleanupItem) -> None:
    # Recheck each entry and ancestors; never recursively follow a newly added link.
    for row in sorted(item.snapshot, key=lambda r: len(Path(r[0]).parts), reverse=True):
        path = item.path / row[0]
        _no_links(path)
        info = path.lstat()
        if (info.st_mode, info.st_dev, info.st_ino) != row[1:4]:
            raise ValueError("Entry identity changed during cleanup")
        if stat.S_ISREG(row[1]) and (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != row[4:]:
            raise ValueError("File changed during cleanup")
        if not path.resolve().is_relative_to(item.path.resolve()):
            raise ValueError("Candidate changed during cleanup")
        if stat.S_ISDIR(row[1]):
            path.rmdir()
        else:
            path.unlink()


def execute_cleanup(plan, apply=False, archive=False) -> list[dict]:
    """Revalidate each planned item; preview by default. Return individual outcomes."""
    results = []
    for item in plan:
        result = {"path": str(item.path), "kind": item.kind, "size_bytes": item.size_bytes}
        try:
            if _validate(item) != item.snapshot:
                raise ValueError("Candidate changed since scan; rescan required")
            result["status"] = "would_delete"
            if apply:
                if archive and item.kind == "legacy":
                    result["archive"] = str(_archive(item))
                    if _validate(item) != item.snapshot:
                        raise ValueError("Candidate changed during archive; source retained")
                _delete_tree(item)
                result["status"] = "deleted"
        except (OSError, ValueError, yaml.YAMLError, zipfile.BadZipFile) as exc:
            result.update(status="refused", error=str(exc))
        results.append(result)
    return results
