"""Import external datasets into a task workspace without changing its task config."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from harness.project import HarnessProject


_KAGGLE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CATALOG_RELATIVE = Path(".ml-harness") / "dataset_sources.json"


class DatasetImportError(ValueError):
    """Raised when an external dataset cannot be imported safely."""


def _catalog_path(project: HarnessProject) -> Path:
    return project.root / _CATALOG_RELATIVE


def list_dataset_sources(project: HarnessProject) -> List[Dict[str, Any]]:
    """Read the workspace-local catalog of imported dataset sources."""
    path = _catalog_path(project)
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetImportError(f"Could not read imported dataset list: {exc}") from exc
    sources = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(sources, list):
        raise DatasetImportError("The imported dataset list is malformed. Move .ml-harness/dataset_sources.json aside and retry.")
    clean_sources = [item for item in sources if isinstance(item, dict)]
    for item in clean_sources:
        value = item.get("local_path")
        if not isinstance(value, str) or not value.strip():
            raise DatasetImportError("An imported dataset is missing its workspace path.")
        candidate = Path(value).expanduser()
        candidate = (candidate if candidate.is_absolute() else project.root / candidate).resolve()
        try:
            candidate.relative_to(project.root.resolve())
        except ValueError as exc:
            raise DatasetImportError("An imported dataset path points outside this task workspace.") from exc
    return clean_sources


def _safe_extract(archive_path: Path, destination: Path) -> int:
    """Extract a Kaggle download archive while rejecting path traversal and links."""
    extracted = 0
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise DatasetImportError(f"Dataset archive contains an unsafe path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DatasetImportError(f"Dataset archive contains a symbolic link: {member.filename}")
            target = (destination / Path(*relative.parts)).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise DatasetImportError(f"Dataset archive path escapes the workspace: {member.filename}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    return extracted


def import_kaggle_dataset(project: HarnessProject, slug: str, *, timeout_seconds: int = 1800) -> Dict[str, Any]:
    """Download a Kaggle dataset page (not a competition bundle) into this workspace."""
    slug = str(slug or "").strip().strip("/")
    if not _KAGGLE_SLUG.fullmatch(slug):
        raise DatasetImportError("Enter a Kaggle dataset slug in the form `owner/dataset-name`.")

    sources = list_dataset_sources(project)
    existing = next((item for item in sources if item.get("platform") == "kaggle" and str(item.get("slug") or "").casefold() == slug.casefold()), None)
    if existing:
        local_path = Path(str(existing.get("local_path") or ""))
        local_path = (local_path if local_path.is_absolute() else project.root / local_path).resolve()
        try:
            local_path.relative_to(project.root.resolve())
        except ValueError:
            local_path = None
        if local_path is not None and local_path.is_dir():
            return {"status": "EXISTS", "source": existing, "sources": sources}

    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise DatasetImportError("Kaggle CLI is not installed. Install the lightweight Kaggle extra and connect your Kaggle account in this dashboard.")

    path_slug = slug.replace("/", "--")
    relative = Path("dataset") / "sources" / "kaggle" / path_slug
    destination = (project.root / relative).resolve()
    try:
        destination.relative_to(project.root.resolve())
    except ValueError as exc:
        raise DatasetImportError("The dataset destination must stay inside this task workspace.") from exc
    if destination.exists():
        if any(destination.iterdir()):
            raise DatasetImportError(
                f"{relative.as_posix()} already contains files. Choose a different dataset workspace or move that folder before importing again."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mlh-kaggle-download-", dir=destination.parent) as temp_name:
        temp_root = Path(temp_name)
        download_dir = temp_root / "download"
        extract_dir = temp_root / "extracted"
        download_dir.mkdir()
        extract_dir.mkdir()
        try:
            result = subprocess.run(
                [kaggle, "datasets", "download", slug, "--path", str(download_dir), "--quiet"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DatasetImportError("Kaggle dataset download took too long. Check the dataset size and try again.") from exc
        except OSError as exc:
            raise DatasetImportError(f"Could not start the Kaggle CLI: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Kaggle CLI returned an error.").strip()
            raise DatasetImportError(detail[-3000:])

        downloaded = sorted(path for path in download_dir.iterdir() if path.is_file())
        if not downloaded:
            raise DatasetImportError("Kaggle completed the download but returned no files.")
        extracted_count = 0
        for path in downloaded:
            if path.suffix.casefold() == ".zip":
                extracted_count += _safe_extract(path, extract_dir)
            else:
                shutil.copy2(path, extract_dir / path.name)
                extracted_count += 1
        if extracted_count == 0:
            raise DatasetImportError("The Kaggle download did not contain any files.")

        if destination.exists():
            destination.rmdir()
        os.replace(extract_dir, destination)

    imported_at = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "platform": "kaggle",
        "kind": "public_dataset",
        "slug": slug,
        "url": f"https://www.kaggle.com/datasets/{slug}",
        "local_path": relative.as_posix(),
        "imported_at": imported_at,
        "license_review_required": True,
        "file_count": sum(1 for path in destination.rglob("*") if path.is_file()),
    }
    sources = [item for item in sources if not (item.get("platform") == "kaggle" and str(item.get("slug") or "").casefold() == slug.casefold())]
    sources.append(record)
    catalog = _catalog_path(project)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    temporary_catalog = catalog.with_suffix(".tmp")
    temporary_catalog.write_text(json.dumps({"schema_version": 1, "sources": sources}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary_catalog, catalog)
    return {"status": "IMPORTED", "source": record, "sources": sources}
