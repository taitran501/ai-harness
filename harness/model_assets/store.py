"""Bounded model-definition parsing and immutable workspace storage."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

import yaml

from harness.io_utils import atomic_write_text
from harness.model_assets.types import ModelDefinition


MAX_MODEL_DEFINITION_BYTES = 16 * 1024
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class _BoundedSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that disallows aliases and duplicate mapping keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not allowed in model definitions")
        return super().compose_node(parent, index)


def _construct_unique_mapping(loader: _BoundedSafeLoader, node: yaml.MappingNode) -> Dict[Any, Any]:
    loader.flatten_mapping(node)
    result: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"Duplicate YAML key in model definition: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_BoundedSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _read_bounded(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_MODEL_DEFINITION_BYTES:
        raise ValueError(
            f"Model definition file exceeds 16 KiB limit ({MAX_MODEL_DEFINITION_BYTES} bytes): {path}"
        )
    with path.open("rb") as stream:
        payload = stream.read(MAX_MODEL_DEFINITION_BYTES + 1)
    if len(payload) > MAX_MODEL_DEFINITION_BYTES:
        raise ValueError(
            f"Model definition file exceeds 16 KiB limit ({MAX_MODEL_DEFINITION_BYTES} bytes): {path}"
        )
    return payload


def _parse_document(path: Path, payload: bytes) -> Dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Model definition must be UTF-8: {path}") from exc
    try:
        if path.suffix.lower() == ".json":
            document = json.loads(text, object_pairs_hook=_unique_json_object)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            document = yaml.load(text, Loader=_BoundedSafeLoader)
        else:
            raise ValueError("Model definition extension must be .json, .yaml, or .yml")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not parse model definition {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Model definition document must be a mapping")
    return document


def _unique_json_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    document: Dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"Duplicate JSON key in model definition: {key!r}")
        document[key] = value
    return document


class ModelStore:
    """Content-addressed storage for small, normalized model definitions."""

    @staticmethod
    def _objects_dir(workspace_root: Path | str) -> Path:
        return Path(workspace_root) / ".ai-harness" / "models" / "objects"

    def register_definition(self, model_def: ModelDefinition, workspace_root: Path) -> str:
        """Atomically save a normalized definition and return its SHA-256 digest."""
        normalized_model = ModelDefinition.from_dict(model_def.to_dict())
        normalized = normalized_model.to_dict()
        digest = normalized_model.compute_digest()
        target_dir = self._objects_dir(workspace_root)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{digest}.json"
        serialized = json.dumps(
            normalized,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        atomic_write_text(target_path, serialized, encoding="utf-8")
        return digest

    def load_definition(
        self, digest: str, workspace_root: Path
    ) -> Optional[ModelDefinition]:
        """Load a content-addressed definition, returning ``None`` if absent."""
        if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
            raise ValueError("Model definition digest must be a lowercase SHA-256 hex string")
        path = self._objects_dir(workspace_root) / f"{digest}.json"
        if not path.exists():
            return None
        raw = _parse_document(path, _read_bounded(path))
        model_def = ModelDefinition.from_dict(raw)
        actual_digest = model_def.compute_digest()
        if actual_digest != digest:
            raise ValueError(
                f"Model definition digest mismatch: expected {digest}, got {actual_digest}"
            )
        return model_def

    def load_catalog_file(self, path: Path) -> ModelDefinition:
        """Load one strictly validated YAML or JSON definition below the 16 KiB cap."""
        path = Path(path)
        raw = _parse_document(path, _read_bounded(path))
        return ModelDefinition.from_dict(raw)
