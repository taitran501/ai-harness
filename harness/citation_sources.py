"""Content-addressed, explicitly adapted source registry for citation evidence.

The registry deliberately extracts only known discussion fields. It does not
walk arbitrary JSON strings, since metadata and error text are not evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from harness.io_utils import atomic_write_json
from harness.state_store import workspace_lock


class CitationSourceError(ValueError):
    """An input cannot be registered as citation evidence."""


def normalize_topic(value: Any) -> str:
    """Return a canonical numeric Kaggle discussion topic identifier."""
    topic = str(value).strip()
    match = re.fullmatch(r"(?:topic[_\s#-]*)?(\d{1,12})", topic, re.IGNORECASE)
    if not match:
        raise CitationSourceError(f"INVALID_TOPIC_ID: {value!r}")
    return str(int(match.group(1)))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CitationSourceError(f"INVALID_JSON_VALUE: {exc}") from exc


def _json_pointer(path: Iterable[Any]) -> str:
    def escape(part: Any) -> str:
        return str(part).replace("~", "~0").replace("/", "~1")
    return "/" + "/".join(escape(part) for part in path)


def _author(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    names = ("author", "author_name", "username", "user_name", "owner")
    result = {key: item[key] for key in names if isinstance(item.get(key), (str, int))}
    return result or None


def _text_value(value: Any, *, locator: str, kind: str, author=None, comment_id=None) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value
    return {
        "schema_version": 1,
        "kind": kind,
        "comment_id": str(comment_id) if comment_id is not None else None,
        "author": author,
        "locator": locator,
        "text": text,
        "segment_sha256": _sha256(text.encode("utf-8")),
    }


def _normalized_segments(topic: Any) -> list[Dict[str, Any]]:
    if isinstance(topic, list):
        segments = [_text_value(text, locator=_json_pointer([i]), kind="body") for i, text in enumerate(topic)]
        return [s for s in segments if s]
    if not isinstance(topic, dict):
        raise CitationSourceError("UNSUPPORTED_SCHEMA: normalized topic must be an object or string list")

    segments = []
    title = topic.get("title")
    segment = _text_value(title, locator="/title", kind="title")
    if segment:
        segments.append(segment)

    # Select one canonical body field; duplicate aliases are not separate evidence.
    body_key = next((key for key in ("content", "body", "text", "description") if isinstance(topic.get(key), str)), None)
    if body_key:
        segment = _text_value(topic[body_key], locator=f"/{body_key}", kind="body")
        if segment:
            segments.append(segment)

    comment_key = next((key for key in ("comments", "replies", "messages") if isinstance(topic.get(key), list)), None)
    if comment_key:
        for index, item in enumerate(topic[comment_key]):
            base = [comment_key, index]
            if isinstance(item, str):
                segment = _text_value(item, locator=_json_pointer(base), kind="comment")
            elif isinstance(item, dict):
                content_key = next((key for key in ("content", "body", "text", "message") if isinstance(item.get(key), str)), None)
                if content_key is None:
                    continue
                id_value = item.get("id", item.get("comment_id", item.get("commentId")))
                locator = _json_pointer(base + [content_key])
                segment = _text_value(item[content_key], locator=locator, kind="comment",
                                      author=_author(item), comment_id=id_value)
            else:
                continue
            if segment:
                segments.append(segment)
    return segments


class SourceRegistry:
    """Registry rooted at ``<workspace>/.ai-harness/evidence``."""

    def __init__(self, project_root: str | Path, evidence_dir: Optional[str | Path] = None):
        self.root = Path(project_root).expanduser().resolve()
        configured = evidence_dir
        if configured is None:
            try:
                from harness.project import HarnessProject
                configured = HarnessProject.discover(self.root).citation_evidence_dir
            except Exception:
                configured = self.root / ".ai-harness" / "evidence"
        self.directory = Path(configured).expanduser()
        if not self.directory.is_absolute():
            self.directory = self.root / self.directory
        self.directory = self.directory.resolve()
        if not self.directory.is_relative_to(self.root):
            raise CitationSourceError("EVIDENCE_PATH_ESCAPE")
        self.snapshots_dir = self.directory / "snapshots"
        self.registry_path = self.directory / "registry.json"
        self._registry_cache: Optional[Dict[str, Any]] = None
        self._source_cache: Dict[tuple[str, str], Dict[str, Any]] = {}

    def _load(self) -> Dict[str, Any]:
        if self._registry_cache is not None:
            return self._registry_cache
        if not self.registry_path.exists():
            self._registry_cache = {"schema_version": 1, "sources": [], "ambiguous_sources": []}
            return self._registry_cache
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            raise CitationSourceError(f"REGISTRY_READ_ERROR: {self.registry_path}: {exc}") from exc
        except UnicodeError as exc:
            raise CitationSourceError(f"INVALID_ENCODING: {self.registry_path}") from exc
        except json.JSONDecodeError as exc:
            raise CitationSourceError(f"MALFORMED_REGISTRY: {self.registry_path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("sources"), list):
            raise CitationSourceError("INVALID_REGISTRY_SCHEMA")
        try:
            from harness.policy_schema import validate_document
            for record in value["sources"]:
                validate_document("source_record", record, version=1)
        except (ValueError, TypeError, KeyError) as exc:
            raise CitationSourceError(f"INVALID_REGISTRY_SCHEMA: {exc}") from exc
        self._registry_cache = value
        return value

    def register_topic_source(
        self,
        topic_id: str | int,
        payload: Any,
        *,
        adapter: str,
        competition: Optional[str] = None,
        canonical_url: Optional[str] = None,
        retrieval_provenance: Optional[Dict[str, Any]] = None,
        complete: bool = True,
        snapshot_version: Optional[str] = None,
        source_path: Optional[str | Path] = None,
        provider: str = "kaggle",
    ) -> Dict[str, Any]:
        requested_id = normalize_topic(topic_id)
        source_path_obj = Path(source_path).expanduser().resolve() if source_path else None
        if source_path_obj is not None:
            if not source_path_obj.is_file():
                raise CitationSourceError(f"SOURCE_FILE_NOT_FOUND: {source_path_obj}")
            filename_match = re.search(r"topic[_-](\d{1,12})\.json$", source_path_obj.name, re.IGNORECASE)
            if filename_match and normalize_topic(filename_match.group(1)) != requested_id:
                raise CitationSourceError("TOPIC_ID_MISMATCH: requested and filename IDs differ")
            try:
                raw_bytes = source_path_obj.read_bytes()
            except OSError as exc:
                raise CitationSourceError(f"SOURCE_READ_ERROR: {source_path_obj}: {exc}") from exc
            try:
                payload = json.loads(raw_bytes.decode("utf-8-sig"))
            except UnicodeDecodeError as exc:
                raise CitationSourceError(f"INVALID_ENCODING: {source_path_obj}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise CitationSourceError(f"MALFORMED_JSON: {source_path_obj}: {exc}") from exc
        else:
            raw_bytes = _canonical_bytes(payload)

        id_holder = payload
        if adapter == "data_wrapper_v1":
            if not isinstance(payload, dict) or "data" not in payload:
                raise CitationSourceError("UNSUPPORTED_SCHEMA: data_wrapper_v1 requires a data field")
            id_holder = payload.get("data")
        payload_ids = []
        for holder in (payload, id_holder) if id_holder is not payload else (payload,):
            if isinstance(holder, dict):
                payload_ids.extend(holder[k] for k in ("topic_id", "topicId", "id") if holder.get(k) is not None)
        if any(normalize_topic(value) != requested_id for value in payload_ids):
            raise CitationSourceError("TOPIC_ID_MISMATCH: requested and payload IDs differ")

        if adapter == "normalized_topic_v1":
            segments = _normalized_segments(payload)
        elif adapter == "data_wrapper_v1":
            segments = _normalized_segments(payload["data"])
        elif adapter == "string_list_v1":
            if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
                raise CitationSourceError("UNSUPPORTED_SCHEMA: string_list_v1 requires a list of strings")
            segments = _normalized_segments(payload)
        else:
            raise CitationSourceError(f"UNSUPPORTED_ADAPTER: {adapter}")
        if not segments:
            raise CitationSourceError("EMPTY_EVIDENCE: no nonempty supported text fields")

        raw_hash = _sha256(raw_bytes)
        normalized_hash = _sha256(_canonical_bytes(segments))
        version = str(snapshot_version or raw_hash[:16])
        source_key = f"{provider}:topic:{requested_id}:v{version}"
        record = {
            "schema_version": 1,
            "source_id": source_key,
            "provider": provider,
            "topic_id": requested_id,
            "snapshot_version": version,
            "competition": competition,
            "canonical_url": canonical_url or f"https://www.kaggle.com/competitions/{competition}/discussion/{requested_id}" if competition else canonical_url,
            "snapshot_path": f"snapshots/{raw_hash}.json",
            "raw_sha256": raw_hash,
            "schema_adapter": adapter,
            "schema_adapter_version": 1,
            "normalized_content_sha256": normalized_hash,
            "retrieval_provenance": retrieval_provenance or {"method": "explicit_local_registration"},
            "completeness": {"complete": bool(complete), "comments_retrieved": bool(complete)},
            "segments": segments,
        }
        from harness.policy_schema import validate_document
        validate_document("source_record", record, version=1)

        self.directory.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        with workspace_lock(self.directory):
            registry = self._load()
            previous = [item for item in registry["sources"] if item.get("source_id") == source_key]
            if previous:
                if previous[0].get("raw_sha256") != raw_hash:
                    raise CitationSourceError("CONFLICTING_SOURCE_VERSION: explicit version already names different bytes")
                return previous[0]
            # Identical bytes for the same provider/topic are deduplicated independent of traversal order.
            identical = next((item for item in registry["sources"]
                              if item.get("provider") == provider and item.get("topic_id") == requested_id
                              and item.get("raw_sha256") == raw_hash), None)
            if identical and (snapshot_version is None or identical.get("source_id") == source_key):
                return identical
            conflicts = [item for item in registry["sources"]
                         if item.get("provider") == provider and item.get("topic_id") == requested_id]
            if conflicts and snapshot_version is None:
                ambiguous = registry.setdefault("ambiguous_sources", [])
                hashes = sorted({item.get("raw_sha256") for item in conflicts} | {raw_hash})
                marker = {"provider": provider, "topic_id": requested_id, "raw_sha256s": hashes}
                if marker not in ambiguous:
                    ambiguous.append(marker)
                    ambiguous.sort(key=lambda item: (item["provider"], item["topic_id"], item["raw_sha256s"]))
                    atomic_write_json(self.registry_path, registry, indent=2, ensure_ascii=False)
                    self._registry_cache = registry
                raise CitationSourceError("AMBIGUOUS_SOURCE_VERSION: provide an explicit snapshot_version")
            snapshot_path = self.directory / record["snapshot_path"]
            if not snapshot_path.exists():
                fd, temporary = tempfile.mkstemp(prefix=f".{snapshot_path.name}.", dir=str(snapshot_path.parent))
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(raw_bytes)
                    os.replace(temporary, snapshot_path)
                except Exception:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                    raise
            registry["sources"].append(record)
            registry["sources"].sort(key=lambda item: item["source_id"])
            atomic_write_json(self.registry_path, registry, indent=2, ensure_ascii=False)
            self._registry_cache = registry
        return record

    def resolve_source(self, topic_id: str | int, *, source_hash: Optional[str] = None,
                       snapshot_version: Optional[str] = None, provider: str = "kaggle") -> Dict[str, Any]:
        normalized_id = normalize_topic(topic_id)
        registry = self._load()
        ambiguous = any(item.get("provider") == provider and item.get("topic_id") == normalized_id
                        for item in registry.get("ambiguous_sources", []))
        if ambiguous and not source_hash and not snapshot_version:
            raise CitationSourceError(f"AMBIGUOUS_SOURCE_VERSION: topic #{normalized_id} requires a source hash/version")
        candidates = [item for item in registry["sources"] if item.get("provider") == provider and item.get("topic_id") == normalized_id]
        if source_hash:
            candidates = [item for item in candidates if item.get("raw_sha256") == source_hash]
        if snapshot_version:
            candidates = [item for item in candidates if item.get("source_id", "").endswith(f":v{snapshot_version}")]
        if not candidates:
            raise CitationSourceError(f"SOURCE_NOT_REGISTERED: topic #{normalized_id}")
        hashes = {item.get("raw_sha256") for item in candidates}
        if len(hashes) != 1:
            raise CitationSourceError(f"AMBIGUOUS_SOURCE_VERSION: topic #{normalized_id} requires a source hash/version")
        record = candidates[0]
        cache_key = (record["source_id"], record["raw_sha256"])
        if cache_key in self._source_cache:
            return self._source_cache[cache_key]
        snapshot = (self.directory / record["snapshot_path"]).resolve()
        if not snapshot.is_relative_to(self.directory.resolve()):
            raise CitationSourceError("SOURCE_PATH_ESCAPE")
        try:
            raw_bytes = snapshot.read_bytes()
        except OSError as exc:
            raise CitationSourceError(f"SOURCE_READ_ERROR: {snapshot}: {exc}") from exc
        if _sha256(raw_bytes) != record.get("raw_sha256"):
            raise CitationSourceError("SOURCE_HASH_MISMATCH")
        try:
            payload = json.loads(raw_bytes.decode("utf-8-sig"))
        except UnicodeError as exc:
            raise CitationSourceError(f"INVALID_ENCODING: {snapshot}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CitationSourceError(f"MALFORMED_JSON: {snapshot}: {exc}") from exc
        adapter = record.get("schema_adapter")
        if record.get("schema_adapter_version") != 1:
            raise CitationSourceError("UNSUPPORTED_SCHEMA_ADAPTER_VERSION")
        if adapter == "normalized_topic_v1":
            segments = _normalized_segments(payload)
        elif adapter == "data_wrapper_v1" and isinstance(payload, dict) and "data" in payload:
            segments = _normalized_segments(payload["data"])
        elif adapter == "string_list_v1" and isinstance(payload, list) and all(isinstance(item, str) for item in payload):
            segments = _normalized_segments(payload)
        else:
            raise CitationSourceError("UNSUPPORTED_SCHEMA_ADAPTER_IN_REGISTRY")
        if _sha256(_canonical_bytes(segments)) != record.get("normalized_content_sha256") or segments != record.get("segments"):
            raise CitationSourceError("NORMALIZED_SOURCE_CONTENT_MISMATCH")
        resolved = {**record, "segments": segments, "snapshot_bytes": raw_bytes}
        self._source_cache[cache_key] = resolved
        return resolved


def register_topic_source(project_root: str | Path, topic_id: str | int, payload: Any, **kwargs) -> Dict[str, Any]:
    return SourceRegistry(project_root).register_topic_source(topic_id, payload, **kwargs)


def resolve_source(project_root: str | Path, topic_id: str | int, **kwargs) -> Dict[str, Any]:
    return SourceRegistry(project_root).resolve_source(topic_id, **kwargs)
