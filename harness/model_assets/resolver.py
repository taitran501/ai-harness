"""Plan and verify model resources for local and remote execution backends."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import unquote, urlsplit

from harness.model_assets.resolution import ModelPlan


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATASET_SLUG = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?$")
_BACKENDS = {"local", "kaggle", "modal"}


@dataclass(frozen=True)
class BackendResourcePlan:
    """Backend-specific model resource requirements and expected mount roots."""

    backend: str
    required_sources: List[Dict[str, Any]] = field(default_factory=list)
    mount_expectations: Dict[str, Path] = field(default_factory=dict)
    offline: bool = False

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in _BACKENDS:
            raise ValueError(f"Unsupported model resource backend: {self.backend!r}")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "required_sources", deepcopy(list(self.required_sources)))
        object.__setattr__(
            self,
            "mount_expectations",
            {str(key): Path(value) for key, value in self.mount_expectations.items()},
        )
        if not isinstance(self.offline, bool):
            raise ValueError("offline must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        """Return a portable JSON-compatible representation for remote jobs."""
        return {
            "backend": self.backend,
            "required_sources": deepcopy(self.required_sources),
            "mount_expectations": {
                key: value.as_posix() for key, value in self.mount_expectations.items()
            },
            "offline": self.offline,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendResourcePlan":
        if not isinstance(value, Mapping):
            raise ValueError("Backend resource plan must be a mapping")
        unknown = set(value) - {
            "backend",
            "required_sources",
            "mount_expectations",
            "offline",
        }
        if unknown:
            raise ValueError("Unknown backend resource plan field(s): " + ", ".join(sorted(unknown)))
        sources = value.get("required_sources", [])
        mounts = value.get("mount_expectations", {})
        if not isinstance(sources, list) or any(not isinstance(item, Mapping) for item in sources):
            raise ValueError("required_sources must be a list of mappings")
        if not isinstance(mounts, Mapping):
            raise ValueError("mount_expectations must be a mapping")
        return cls(
            backend=str(value.get("backend", "")),
            required_sources=[dict(item) for item in sources],
            mount_expectations={str(key): Path(path) for key, path in mounts.items()},
            offline=value.get("offline", False),
        )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _scope_value(scope: Any, key: str, default: Any = None) -> Any:
    value = _get(scope, key, default)
    if value is not default:
        return value
    task = _get(scope, "task", {})
    return _get(task, key, default)


def _scope_root(scope: Any) -> Path:
    if isinstance(scope, (str, os.PathLike)):
        return Path(scope).expanduser().resolve()
    project = _get(scope, "project")
    root = _get(scope, "root") or _get(scope, "project_root") or _get(project, "root")
    if root is None:
        root = Path.cwd()
    return Path(root).expanduser().resolve()


def _lock_for(model_plan: Any) -> Dict[str, Any]:
    value = _get(model_plan, "model_lock")
    if value is None:
        to_lock = _get(model_plan, "to_lock_dict")
        value = to_lock() if callable(to_lock) else model_plan
    if not isinstance(value, Mapping):
        raise TypeError("model_plan must expose a model_lock mapping")
    lock = dict(value)
    roots = lock.get("roots", [])
    if not isinstance(roots, list) or any(not isinstance(root, Mapping) for root in roots):
        raise ValueError("model lock roots must be a list of mappings")
    return lock


def _lookup(mapping: Any, root: Mapping[str, Any]) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in (
        str(root.get("slot", "")),
        str(root.get("model_id", "")),
        str(root.get("definition_digest", "")),
        str(root.get("artifact_bundle_digest", "")),
    ):
        if key and key in mapping:
            return mapping[key]
    return None


def _location_for(scope: Any, backend: str, root: Mapping[str, Any], index: int) -> Any:
    locations = _scope_value(scope, "model_locations", _scope_value(scope, "locations", {}))
    selected = _lookup(locations, root)
    if selected is None and isinstance(locations, Mapping):
        backend_locations = locations.get(backend)
        selected = _lookup(backend_locations, root)
    if isinstance(selected, Mapping) and backend in selected and isinstance(selected[backend], Mapping):
        selected = selected[backend]

    if backend == "local" and selected is None:
        selected = _lookup(_scope_value(scope, "model_asset_paths", {}), root)
    if backend == "modal" and selected is None:
        selected = _lookup(_scope_value(scope, "modal_model_locations", {}), root)
    if backend == "kaggle" and selected is None:
        sources = _scope_value(scope, "model_sources", [])
        if isinstance(sources, Mapping):
            selected = _lookup(sources, root)
        elif isinstance(sources, list) and index < len(sources):
            selected = sources[index]
    return selected


def _definition(root: Mapping[str, Any]) -> Dict[str, Any]:
    value = root.get("definition", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _relative_asset_path(value: Any, label: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a relative path string")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"{label} must not be absolute or traverse parent directories")
    return "" if path == PurePosixPath(".") else path.as_posix().strip("/")


def _manifest_documents(scope: Any, root_path: Path, definition: Mapping[str, Any]) -> tuple[Any, Optional[Path]]:
    artifacts = definition.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    reference = artifacts.get("manifest_ref")
    if not reference:
        return None, None
    if not isinstance(reference, str):
        raise ValueError("model artifacts.manifest_ref must be a string")

    configured = _scope_value(scope, "model_manifests", {})
    configured_document = _lookup(configured, {"slot": "", "model_id": "", "definition_digest": reference})
    if configured_document is not None:
        if isinstance(configured_document, (str, os.PathLike)):
            path = Path(configured_document).expanduser()
            if not path.is_absolute():
                path = _scope_root(scope) / path
            return None, path.resolve()
        return configured_document, None

    workspace = _scope_root(scope)
    ref_path = Path(reference).expanduser()
    candidates: List[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.extend([workspace / ref_path, root_path / ref_path])
        if _SHA256.fullmatch(reference):
            candidates.extend(
                [
                    workspace / ".ai-harness" / "models" / "manifests" / f"{reference}.json",
                    workspace / ".ai-harness" / "models" / "manifests" / reference,
                ]
            )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError:
            # Absolute manifest references are explicit selections and are permitted.
            if not ref_path.is_absolute():
                continue
        if resolved.is_file():
            return None, resolved
    return None, None


def _parse_manifest(document: Any, path: Optional[Path]) -> List[Dict[str, Any]]:
    if document is None:
        if path is None or not path.is_file():
            return []
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read model artifact manifest {path}: {exc}") from exc
    if isinstance(document, list):
        files = document
    elif isinstance(document, Mapping):
        files = document.get("files", document.get("artifacts"))
    else:
        files = None
    if not isinstance(files, list):
        raise ValueError("Model artifact manifest must contain a files list")

    parsed: List[Dict[str, Any]] = []
    paths = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("Each model artifact manifest entry must be a mapping")
        relative = _relative_asset_path(entry.get("relative_path", entry.get("path")), "manifest path")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if not relative:
            raise ValueError("Model artifact manifest paths must not be empty")
        if relative.casefold() in paths:
            raise ValueError(f"Duplicate model artifact manifest path: {relative}")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"Model artifact manifest entry {relative!r} needs a lowercase SHA-256")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError(f"Model artifact manifest size for {relative!r} must be a non-negative integer")
        paths.add(relative.casefold())
        parsed.append({"relative_path": relative, "sha256": digest, "bytes": size})
    return parsed


def _manifest_for(scope: Any, root: Mapping[str, Any], model_root: Path, definition: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        document, path = _manifest_documents(scope, model_root, definition)
        files = _parse_manifest(document, path)
    except (OSError, TypeError, ValueError) as exc:
        return [], str(exc)
    if not files:
        artifacts = definition.get("artifacts")
        reference = artifacts.get("manifest_ref") if isinstance(artifacts, Mapping) else None
        return [], "Model artifact manifest is missing or contains no files" if reference else "Model artifact manifest is not declared"
    return files, None


def _local_root(scope: Any, root: Mapping[str, Any], location: Any, definition: Mapping[str, Any]) -> Path:
    project_root = _scope_root(scope)
    explicit = location.get("path", location.get("root", location.get("directory"))) if isinstance(location, Mapping) else location
    if explicit:
        path = Path(str(explicit)).expanduser()
        return (path if path.is_absolute() else project_root / path).resolve()

    source = definition.get("source")
    source = source if isinstance(source, Mapping) else {}
    provider = str(source.get("provider", "")).lower()
    repository = str(source.get("repository", "")).strip()
    revision = str(source.get("revision", "")).strip()
    if provider in {"local", "workspace"} and repository:
        repo_path = Path(repository).expanduser()
        if repo_path.is_absolute():
            return repo_path.resolve()
        selected = (project_root / repo_path).resolve()
        if selected.exists():
            return selected

    if provider in {"huggingface", "hf"} and repository and revision:
        hub_cache = _scope_value(scope, "hf_hub_cache_dir") or os.environ.get("HF_HUB_CACHE")
        cache_home = _scope_value(scope, "hf_cache_dir") or os.environ.get("HF_HOME")
        if hub_cache:
            hub = Path(hub_cache).expanduser()
        else:
            cache = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache" / "huggingface"
            hub = cache if cache.name == "hub" else cache / "hub"
        snapshot = hub / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
        if snapshot.exists():
            return snapshot.resolve()

    safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "--", repository or str(root.get("model_id", "model")))
    safe_revision = re.sub(r"[^A-Za-z0-9_.-]+", "--", revision or "unversioned")
    candidates = [
        project_root / ".ai-harness" / "models" / "cache" / provider / safe_repo / safe_revision,
        project_root / ".ai-harness" / "models" / "assets" / str(root.get("model_id", "model")) / str(root.get("artifact_bundle_digest") or safe_revision),
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def _kaggle_dataset(location: Any, definition: Mapping[str, Any]) -> Optional[str]:
    if isinstance(location, Mapping):
        value = location.get("dataset", location.get("dataset_slug", location.get("source_slug")))
        if value:
            return str(value).strip()
    elif isinstance(location, str) and location.strip():
        if "://" not in location:
            return location.strip()
    source = definition.get("source")
    source = source if isinstance(source, Mapping) else {}
    provider = str(source.get("provider", "")).lower()
    repository = str(source.get("repository", "")).strip()
    uri = str(source.get("source_uri", "")).strip()
    if provider in {"kaggle", "kaggle_dataset", "kaggle-dataset"} and repository:
        return repository
    if uri.startswith(("kaggle://", "kaggle-dataset://")):
        parsed = urlsplit(uri)
        dataset = unquote((parsed.netloc + parsed.path).strip("/"))
        return dataset.split("@", 1)[0] or None
    return None


def _provider_version(location: Any, definition: Mapping[str, Any]) -> Any:
    if isinstance(location, Mapping) and location.get("version") is not None:
        return location["version"]
    source = definition.get("source")
    uri = source.get("source_uri") if isinstance(source, Mapping) else None
    if isinstance(uri, str) and uri.startswith(("kaggle://", "kaggle-dataset://")):
        parsed = urlsplit(uri)
        dataset = unquote(parsed.netloc + parsed.path).strip("/")
        if "@" in dataset:
            return dataset.rsplit("@", 1)[1]
    return None


def _mount_path(backend: str, dataset: Optional[str], relative_path: str, slot: str, model_id: str) -> Path:
    if backend == "local":
        return Path(relative_path)
    if backend == "kaggle":
        folder = dataset.rsplit("/", 1)[-1] if dataset else re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
        return Path("/kaggle/input") / folder / relative_path
    return Path("/vol/models") / relative_path


def _is_valid_dataset_slug(value: Optional[str]) -> bool:
    if value is None or not _DATASET_SLUG.fullmatch(value):
        return False
    return all(part not in {".", ".."} for part in value.split("/"))


def _offline_for(scope: Any, backend: str) -> bool:
    explicit = _scope_value(scope, "offline")
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ValueError("task_scope.offline must be a boolean")
        return explicit
    runtime = _scope_value(scope, "runtime", {})
    enabled = _get(runtime, "enable_internet")
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("task_scope.runtime.enable_internet must be a boolean")
        return not enabled
    return backend in {"kaggle", "modal"}


class ModelResolver:
    """Resolve portable model lock entries to exact backend mount paths."""

    def __init__(self) -> None:
        self.verification_errors: List[str] = []

    def plan_backend_resources(
        self, model_plan: ModelPlan, backend: str, task_scope: Any
    ) -> BackendResourcePlan:
        backend = str(backend).strip().lower()
        if backend not in _BACKENDS:
            raise ValueError(f"Unsupported model resource backend: {backend!r}")
        lock = _lock_for(model_plan)
        roots = lock.get("roots", [])
        offline = _offline_for(task_scope, backend)
        required_sources: List[Dict[str, Any]] = []
        mount_expectations: Dict[str, Path] = {}
        modal_default_volume = str(
            _scope_value(task_scope, "model_volume_name")
            or "aih-" + re.sub(r"[^A-Za-z0-9-]+", "-", str(lock.get("task_id", "task")).lower()).strip("-") + "-models"
        )

        for index, raw_root in enumerate(roots):
            root = dict(raw_root)
            definition = _definition(root)
            model_id = str(root.get("model_id") or definition.get("model_id") or root.get("slot") or "model")
            slot = str(root.get("slot") or model_id)
            kind = str(definition.get("kind", "")).lower()
            if kind == "service":
                continue
            location = _location_for(task_scope, backend, root, index)
            relative_path = _relative_asset_path(
                location.get("relative_path", location.get("path_in_volume", ""))
                if isinstance(location, Mapping)
                else "",
                f"{backend} model location",
            )
            dataset = _kaggle_dataset(location, definition) if backend == "kaggle" else None
            volume_name = None
            if backend == "modal":
                volume_name = (
                    location.get("volume_name", location.get("volume"))
                    if isinstance(location, Mapping)
                    else None
                ) or modal_default_volume
                if not relative_path:
                    bundle = root.get("artifact_bundle_digest") or definition.get("source", {}).get("revision", "")
                    relative_path = "/".join(
                        part for part in (re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id), str(bundle)) if part
                    )

            model_root = (
                _local_root(task_scope, root, location, definition)
                if backend == "local"
                else _mount_path(backend, dataset, relative_path, slot, model_id)
            )
            files, manifest_error = _manifest_for(task_scope, root, model_root, definition)
            source_info: Dict[str, Any] = {
                "slot": slot,
                "model_id": model_id,
                "definition_digest": root.get("definition_digest"),
                "bundle_digest": root.get("artifact_bundle_digest"),
                "provider": (definition.get("source") or {}).get("provider") if isinstance(definition.get("source"), Mapping) else None,
                "repository": (definition.get("source") or {}).get("repository") if isinstance(definition.get("source"), Mapping) else None,
                "revision": (definition.get("source") or {}).get("revision") if isinstance(definition.get("source"), Mapping) else None,
                "source_uri": (definition.get("source") or {}).get("source_uri") if isinstance(definition.get("source"), Mapping) else None,
                "mount_path": model_root.as_posix(),
                "relative_path": relative_path,
                "files": files,
                "missing_prerequisites": [manifest_error] if manifest_error else [],
            }
            if backend == "kaggle":
                source_info["dataset_slug"] = dataset
                source_info["provider_version"] = _provider_version(location, definition)
                if dataset and not _is_valid_dataset_slug(dataset):
                    source_info["missing_prerequisites"].append("Kaggle model dataset slug is invalid")
                elif not dataset:
                    source_info["missing_prerequisites"].append("No Kaggle Dataset location is declared for this model")
            elif backend == "modal":
                source_info["volume_name"] = volume_name
                source_info["provider_version"] = location.get("version") if isinstance(location, Mapping) else None
            required_sources.append(source_info)
            mount_expectations[slot] = model_root

        return BackendResourcePlan(
            backend=backend,
            required_sources=required_sources,
            mount_expectations=mount_expectations,
            offline=offline,
        )

    def verify_runtime_resources(self, resource_plan: BackendResourcePlan) -> bool:
        """Check every declared asset file and uncached SHA-256 before model loading."""
        if not isinstance(resource_plan, BackendResourcePlan):
            raise TypeError("resource_plan must be a BackendResourcePlan")
        errors: List[str] = []
        for source in resource_plan.required_sources:
            slot = str(source.get("slot", "model"))
            root = resource_plan.mount_expectations.get(slot, Path(source.get("mount_path", "")))
            if not root.is_dir():
                mode = "Offline " if resource_plan.offline else ""
                errors.append(f"{mode}model resource mount is missing for {slot}: {root}")
                continue
            try:
                resolved_root = root.resolve(strict=True)
            except OSError as exc:
                errors.append(f"Model resource mount is unavailable for {slot}: {exc}")
                continue
            files = source.get("files") or []
            if not files:
                errors.append(
                    f"Model resource {slot} has no integrity manifest; refusing to load unverified weights"
                )
                continue
            for entry in files:
                relative = entry.get("relative_path")
                path = (root / Path(*PurePosixPath(relative).parts)).resolve()
                try:
                    path.relative_to(resolved_root)
                except ValueError:
                    errors.append(f"Model resource path escapes its mount for {slot}: {relative}")
                    continue
                if not path.is_file():
                    mode = "Offline " if resource_plan.offline else ""
                    errors.append(f"{mode}model weight file is missing for {slot}: {relative}")
                    continue
                expected_bytes = entry.get("bytes")
                if expected_bytes is not None and path.stat().st_size != expected_bytes:
                    errors.append(f"Model resource size mismatch for {slot}: {relative}")
                    continue
                digest = _sha256_file(path)
                if digest != entry.get("sha256"):
                    errors.append(f"Model resource SHA-256 mismatch for {slot}: {relative}")
        self.verification_errors = errors
        return not errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["BackendResourcePlan", "ModelResolver"]
