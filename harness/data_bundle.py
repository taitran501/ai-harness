"""Resolve and stage only task-declared competition data for a remote provider."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from harness.project import HarnessProject


DATA_PATH_KEYS = (
    "train_path",
    "training_path",
    "test_path",
    "validation_path",
    "dev_path",
    "sample_submission_path",
    "split_path",
    "split_file",
    "dataset_path",
    "data_path",
)
BLOCKED_PATH_PARTS = {".git", ".env", ".kaggle", ".modal", ".ml-harness", "artifacts", "submissions"}
DATA_MANIFEST_NAME = "mlh-data-bundle.json"


class DataBundleError(ValueError):
    """Raised when task data cannot be staged within the attached workspace safely."""


def _path_values(value: Any) -> Iterable[str]:
    if isinstance(value, str) and value.strip():
        yield value.strip()
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                yield item.strip()


def declared_data_roots(
    project: HarnessProject,
    *,
    additional_paths: Iterable[str] = (),
) -> List[Tuple[Path, str]]:
    """Return configured data roots as ``(absolute_path, workspace_relative_path)`` pairs."""
    document = yaml.safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
    task = document.get("task", document)
    data_config = task.get("data_config") or task.get("data") or {}
    values: List[str] = []
    for key in DATA_PATH_KEYS:
        values.extend(_path_values(data_config.get(key)))
    validation = task.get("validation") or {}
    values.extend(_path_values(validation.get("split_file")))
    values.extend(_path_values(task.get("split_path")))
    values.extend(_path_values(task.get("split_file")))
    values.extend(_path_values(list(additional_paths)))

    if not values:
        values = [
            path for path in ("dataset", "splits")
            if (project.root / path).exists()
        ]
        if not values:
            raise DataBundleError("The task config declares no local data paths and the workspace has no dataset/ or splits/ directory.")
    elif (project.root / "splits").is_dir():
        values.append("splits")

    resolved: List[Tuple[Path, str]] = []
    seen = set()
    for value in values:
        candidate = Path(value).expanduser()
        candidate = (candidate if candidate.is_absolute() else project.root / candidate).resolve()
        try:
            relative = candidate.relative_to(project.root).as_posix()
        except ValueError as exc:
            raise DataBundleError(
                f"Declared task data must be inside the attached workspace before cloud upload: {value}"
            ) from exc
        parts = {part.casefold() for part in Path(relative).parts}
        if parts.intersection(BLOCKED_PATH_PARTS):
            raise DataBundleError(f"Refusing to package a protected path as task data: {relative}")
        if not candidate.exists():
            raise DataBundleError(f"Declared task data path does not exist: {relative}")
        if not candidate.is_file() and not candidate.is_dir():
            raise DataBundleError(f"Declared task data path is not a regular file or directory: {relative}")
        key = candidate.as_posix().casefold()
        if key not in seen:
            resolved.append((candidate, relative))
            seen.add(key)

    # If a directory is declared, it already contains any child path declared separately.
    resolved.sort(key=lambda item: (len(Path(item[1]).parts), item[1].casefold()))
    roots: List[Tuple[Path, str]] = []
    for candidate, relative in resolved:
        if any(relative == parent_rel or relative.startswith(parent_rel.rstrip("/") + "/") for _, parent_rel in roots):
            continue
        roots.append((candidate, relative))
    if not roots:
        raise DataBundleError("No task data paths are available to upload.")
    return roots


def _files_under(project_root: Path, roots: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    files: List[Tuple[Path, str]] = []
    for source, relative in roots:
        source_files = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
        if not source_files:
            raise DataBundleError(f"Declared task data directory contains no files: {relative}")
        for path in source_files:
            actual = path.resolve()
            try:
                file_rel = actual.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise DataBundleError(f"Task data symlink points outside the workspace: {path}") from exc
            parts = {part.casefold() for part in Path(file_rel).parts}
            if parts.intersection(BLOCKED_PATH_PARTS):
                raise DataBundleError(f"Refusing to include a protected task data file: {file_rel}")
            files.append((actual, file_rel))
    deduplicated = {relative: path for path, relative in files}
    return [(deduplicated[key], key) for key in sorted(deduplicated)]


def stage_declared_data(
    project: HarnessProject,
    destination: Path | str,
    *,
    additional_paths: Iterable[str] = (),
) -> Dict[str, Any]:
    """Copy declared data preserving workspace paths and write a content-free file manifest."""
    target_root = Path(destination).expanduser().resolve()
    roots = declared_data_roots(project, additional_paths=additional_paths)
    files = _files_under(project.root, roots)
    target_root.mkdir(parents=True, exist_ok=True)
    for source, relative in files:
        target = target_root / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema_version": 1,
        "task_id": project.manifest.get("task_id") or project.plugin_id or "task",
        "paths": [relative for _, relative in roots],
        "files": [relative for _, relative in files],
    }
    (target_root / DATA_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"manifest": manifest, "file_count": len(files)}
