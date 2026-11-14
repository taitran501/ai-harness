"""Short reentrant cross-process JSON transactions; never lock training work."""

import copy
import json
import math
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from harness.io_utils import atomic_write_json

_locks = {}
_guard = threading.Lock()
_local = threading.local()


class StateIntegrityError(ValueError):
    """Existing authoritative state is unreadable or violates its schema."""

    def __init__(self, code, path, field=None, detail=""):
        self.code, self.path, self.field = code, Path(path), field
        super().__init__(f"{code}: {path}" + (f" [{field}]" if field else "") + f": {detail}")


class StateConflictError(ValueError):
    def __init__(self, path):
        self.field_path = path
        super().__init__("Concurrent state conflict: " + ".".join(path))


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def validate_state(value):
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    revision = value.get("revision", 0)
    if type(revision) is not int or revision < 0:
        raise ValueError("revision must be a nonnegative integer")


def _validate(value, path, validate=None):
    try:
        validate_state(value)
        json.dumps(value, allow_nan=False)
        if validate:
            validate(value)
    except StateIntegrityError as exc:
        raise StateIntegrityError(exc.code, path, field=exc.field, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise StateIntegrityError("STATE_SCHEMA_INVALID", path, detail=str(exc)) from exc


def read_json_state(path, *, default_factory=None, validate=None, migrate=None):
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise StateIntegrityError("STATE_UNREADABLE", path, detail="dangling link") from exc
        if default_factory is None:
            raise
        value = default_factory()
    except UnicodeError as exc:
        raise StateIntegrityError("STATE_ENCODING_INVALID", path, detail=str(exc)) from exc
    except OSError as exc:
        raise StateIntegrityError("STATE_UNREADABLE", path, detail=str(exc)) from exc
    else:
        try:
            value = json.loads(raw, object_pairs_hook=_unique_object,
                               parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except ValueError as exc:
            raise StateIntegrityError("STATE_JSON_INVALID", path, detail=str(exc)) from exc
    if migrate:
        try:
            value = migrate(value)
        except (ValueError, TypeError, KeyError) as exc:
            raise StateIntegrityError("STATE_SCHEMA_INVALID", path, detail=str(exc)) from exc
    _validate(value, path, validate)
    return value


@contextmanager
def workspace_lock(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    key = str(directory)
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_local, "held", {})
        if key in held:
            yield
            return
        path = directory / ".state.lock"
        with path.open("a+b") as stream:
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            held[key] = True
            _local.held = held
            try:
                yield
            finally:
                held.pop(key)
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def transaction(path, default=None, *, validate=None, migrate=None):
    path = Path(path)
    with workspace_lock(path.parent):
        value = read_json_state(path, default_factory=lambda: copy.deepcopy(default or {}),
                                validate=validate, migrate=migrate)
        yield value
        _validate(value, path, validate)
        value["revision"] = value.get("revision", 0) + 1
        atomic_write_json(path, value, indent=2, ensure_ascii=False)


def merge_changed(current, before, after, *, additive_paths=frozenset(), path=()):
    """Apply a snapshot's changes, preserving concurrent unrelated updates and append-only events."""
    for key, value in after.items():
        field_path = path + (key,)
        if field_path in additive_paths:
            operands = [value] + ([before[key]] if key in before else []) + ([current[key]] if key in current else [])
            fractional = field_path == ("compute_usage", "wall_seconds_used")
            if any(not (type(x) is int or (fractional and type(x) is float and math.isfinite(x)))
                   for x in operands):
                raise ValueError("Invalid counter operands: " + ".".join(field_path))
        if key in before and value == before[key]:
            continue
        if (
            isinstance(value, dict)
            and isinstance(before.get(key), dict)
            and isinstance(current.get(key), dict)
        ):
            merge_changed(current[key], before[key], value,
                          additive_paths=additive_paths, path=field_path)
        elif isinstance(value, list) and isinstance(before.get(key), list):
            added = [x for x in value if x not in before[key]]
            removed = [x for x in before[key] if x not in value]
            current[key] = [x for x in current.get(key, []) if x not in removed]
            current[key].extend(x for x in added if x not in current[key])
        else:
            if field_path in additive_paths and key in before and key in current:
                # A reset is an assignment. It cannot be combined with an increment.
                if value > before[key] and current[key] >= before[key]:
                    current[key] += value - before[key]
                    continue
                if current[key] != before[key] and current[key] != value:
                    raise StateConflictError(field_path)
            if key in current and (key not in before or current[key] != before[key]) and current[key] != value:
                raise StateConflictError(field_path)
            if key in before and key not in current:
                raise StateConflictError(field_path)
            current[key] = copy.deepcopy(value)
    return current


def save_snapshot(path, before, after, *, validate=None, migrate=None, additive_paths=frozenset()):
    with transaction(path, after if not before else before, validate=validate, migrate=migrate) as current:
        merge_changed(current, before, after, additive_paths=additive_paths)
    return copy.deepcopy(current)


def save_registry(path, key, records, before=None, *, validate=None, migrate=None, additive_paths=frozenset()):
    def validate_registry(value):
        rows = value.get(key)
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
            raise ValueError(f"{key} must contain records with string IDs")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{key} contains duplicate IDs")
        if validate:
            validate(value)
    with transaction(path, {key: []}, validate=validate_registry, migrate=migrate) as value:
        existing = {r["id"]: r for r in value.get(key, [])}
        previous = {r["id"]: r for r in (before or [])}
        for record in records:
            if record == previous.get(record["id"]):
                continue
            if record["id"] in existing:
                merge_changed(existing[record["id"]], previous.get(record["id"], {}), record,
                              additive_paths=additive_paths)
            else:
                existing[record["id"]] = record
        value[key] = list(existing.values())
    return value[key]
