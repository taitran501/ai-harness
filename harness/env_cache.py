"""Process-wide TTL cache for environment detector results.

Motivation: ``EnvironmentDetector.get_full_status`` is called from
``ai-harness status`` and the agent context HUD. The individual probes are
expensive — on a machine with PyTorch installed, ``import torch`` alone
costs ~6.5 s, ``modal profile list`` ~3 s, and the local-LLM port checks
~2 s each. GPU/driver/cloud credentials do not change within a process
lifetime, so re-probing on every call was pure waste (measured: an
``ai-harness status`` command taking 18 s, most of it in the detectors).

The cache is **per-process** (no disk state, no cross-process locking)
with per-key TTLs. A missing/expired key re-runs its probe; a crash or a
changed environment is therefore self-correcting within one TTL window.

TTLs can be overridden per key via the ``AI_HARNESS_ENV_TTL_<NAME>``
environment variable (seconds); ``AI_HARNESS_ENV_CACHE=off`` disables
caching entirely (useful in tests and in ``ai-harness status --force``).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional

# name -> TTL seconds. gpu gets the longest: hardware state effectively
# never changes during a session, and the torch fallback import is the
# single most expensive probe in the whole detector.
DEFAULT_TTLS: Dict[str, float] = {
    "modal": 300.0,
    "kaggle": 300.0,
    "docker": 300.0,
    "gpu": 600.0,
    "llm": 120.0,
    "local_endpoint": 120.0,
}

_lock = threading.Lock()
_cache: Dict[str, tuple] = {}  # key -> (expires_at, value)


def _caching_disabled() -> bool:
    return os.environ.get("AI_HARNESS_ENV_CACHE", "").lower() in {"0", "off", "false"}


def _ttl_for(name: str) -> float:
    override = os.environ.get(f"AI_HARNESS_ENV_TTL_{name.upper()}")
    if override is not None:
        try:
            value = float(override)
            if value >= 0:
                return value
        except ValueError:
            pass
    return DEFAULT_TTLS.get(name, 60.0)


def get_cached(name: str) -> Optional[Any]:
    """Return the cached value for *name*, or ``None`` when absent/expired."""
    if _caching_disabled():
        return None
    with _lock:
        entry = _cache.get(name)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del _cache[name]
            return None
        return value


def set_cached(name: str, value: Any, ttl: Optional[float] = None) -> None:
    if _caching_disabled():
        return
    if ttl is None:
        ttl = _ttl_for(name)
    if ttl <= 0:
        return
    with _lock:
        _cache[name] = (time.monotonic() + ttl, value)


def cached_probe(name: str, probe: Callable[[], Any]) -> Any:
    """Run *probe* only when *name* is not fresh in the cache, then cache it.

    The probe is executed OUTSIDE the cache lock, so a slow probe
    (torch import, modal subprocess) never blocks concurrent readers
    from taking or storing their own cache entries.
    """
    value = get_cached(name)
    if value is not None:
        return value
    result = probe()
    set_cached(name, result)
    return result


def invalidate() -> None:
    """Drop every cached probe result (used by ``--force`` and tests)."""
    with _lock:
        _cache.clear()
