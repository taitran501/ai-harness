"""Lightweight built-in model-name registry (no heavy imports).

``harness.models`` imports numpy/pandas/sklearn at module level (~1.3 s on
first import, measured) — a cost no one should pay just to list the model
names accepted by ``create_model`` (the ``ai-harness status`` plugins
section does exactly that, on every one-shot CLI process).

This module serves the name list from a disk cache validated against
``models.py``'s mtime; only a genuine code change pays for the real
import. The name *logic* itself is never duplicated here — a cache miss
delegates to ``harness.models.available_model_names``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

_MODELS_PY = Path(__file__).with_name("models.py")
_CACHE_FILE = Path(__file__).parent / ".model_names_cache.json"


def _read_cache() -> List[str] | None:
    try:
        record: Dict[str, Any] = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if record.get("mtime") != _MODELS_PY.stat().st_mtime:
            return None
        names = record.get("names")
        if isinstance(names, list) and names and all(isinstance(n, str) for n in names):
            return sorted(names)
    except Exception:  # noqa: BLE001 - cache is advisory; fall through
        return None
    return None


def _write_cache(names: List[str]) -> None:
    try:
        tmp = _CACHE_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"mtime": _MODELS_PY.stat().st_mtime, "names": names}),
            encoding="utf-8",
        )
        os.replace(tmp, _CACHE_FILE)  # atomic on Windows + POSIX
    except Exception:  # noqa: BLE001 - cache write is advisory
        pass


def builtin_model_names() -> List[str]:
    """Built-in model names, served from a validated disk cache.

    Identical to ``harness.models.available_model_names()`` — the cache
    only skips the module import when ``models.py`` has not changed.
    """
    cached = _read_cache()
    if cached is not None:
        return cached
    from harness.models import available_model_names

    names = available_model_names()
    _write_cache(names)
    return names
