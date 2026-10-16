"""Kaggle SDK compatibility layer.

Agents frequently hit ``AttributeError`` when the same code path uses either
the legacy ``kaggle`` package (``KaggleApi`` with snake_case methods like
``kernels_output``, ``competition_topic_fields``) or the newer ``kagglesdk``
(Pydantic models with snake_case attributes like ``public_score``,
``total_votes``).

This module exposes a single, stable facade that:

1. Detects which SDK is available (``kagglesdk`` preferred, ``kaggle`` fallback).
2. Normalises attribute / method names so callers can use one spelling.
3. Wraps the underlying client so agents don't need to know which SDK is in
   use.

Usage::

    from harness.kaggle_compat import KaggleClient

    client = KaggleClient()
    kernels = client.list_kernels(user="me", limit=20)
    for k in kernels:
        print(k.total_votes, k.title)

    submissions = client.list_submissions(competition="foo")
    for s in submissions:
        print(s.public_score)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Attribute-name mapping: legacy (kaggle) name -> canonical name.
# Callers always use the canonical (snake_case, lower) name; the wrapper
# translates to whichever SDK is installed.
_LEGACY_ALIASES = {
    # KaggleApi (legacy) method names are already snake_case, but the SDK
    # sometimes exposes camelCase on response objects. Normalise both.
    "totalVotes": "total_votes",
    "publicScore": "public_score",
    "privateScore": "private_score",
    "createdAt": "created_at",
    "kernelOutput": "kernels_output",
    "kernelLogs": "kernels_logs",
    "kernelStatus": "kernels_status",
    "competitionTopicsList": "competition_topics_list",
    "listCompetitionFiles": "list_competition_files",
    "listKernels": "list_kernels",
    "listSubmissions": "list_submissions",
}


def _try_import_kagglesdk():
    try:
        import kagglesdk  # type: ignore
        return kagglesdk
    except ImportError:
        return None


def _try_import_kaggle_legacy():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        return KaggleApi
    except ImportError:
        return None


def detect_sdk() -> str:
    """Return ``'kagglesdk'``, ``'kaggle'`` or ``'none'``."""
    if _try_import_kagglesdk() is not None:
        return "kagglesdk"
    if _try_import_kaggle_legacy() is not None:
        return "kaggle"
    return "none"


class _NormalisedObject:
    """Proxy that maps both legacy camelCase and canonical snake_case attrs."""

    def __init__(self, raw: Any, sdk: str):
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_sdk", sdk)

    def __getattr__(self, name: str) -> Any:
        raw = object.__getattribute__(self, "_raw")
        # 1. Direct attribute
        if hasattr(raw, name):
            value = getattr(raw, name)
            return self._wrap(value)
        # 2. Aliased attribute (e.g. totalVotes -> total_votes)
        alias = _LEGACY_ALIASES.get(name)
        if alias and hasattr(raw, alias):
            return self._wrap(getattr(raw, alias))
        # 3. Reverse alias (caller asked for legacy name, SDK has canonical)
        reverse = {v: k for k, v in _LEGACY_ALIASES.items()}.get(name)
        if reverse and hasattr(raw, reverse):
            return self._wrap(getattr(raw, reverse))
        # 4. Pydantic-style: try to_dict then lookup
        if hasattr(raw, "model_dump"):
            data = raw.model_dump()
            if name in data:
                return self._wrap(data[name])
            alias = _LEGACY_ALIASES.get(name)
            if alias and alias in data:
                return self._wrap(data[alias])
        if hasattr(raw, "dict"):
            data = raw.dict()
            if name in data:
                return self._wrap(data[name])
        raise AttributeError(
            f"'{type(raw).__name__}' object has no attribute '{name}'"
        )

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, bytes)) or value is None:
            return value
        return _NormalisedObject(value, object.__getattribute__(self, "_sdk"))

    def __repr__(self) -> str:
        return f"<KaggleCompat {type(object.__getattribute__(self, '_raw')).__name__}>"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _NormalisedObject):
            return object.__getattribute__(self, "_raw") == object.__getattribute__(other, "_raw")
        return object.__getattribute__(self, "_raw") == other

    def __hash__(self) -> int:
        return hash(object.__getattribute__(self, "_raw"))


class KaggleClient:
    """Unified facade over whichever Kaggle SDK is installed.

    Falls back to ``none`` when no SDK is available; callers should check
    :func:`detect_sdk` before instantiating.
    """

    def __init__(self):
        self.sdk: str = detect_sdk()
        if self.sdk == "kagglesdk":
            kagglesdk = _try_import_kagglesdk()
            self._sdk_module = kagglesdk
            self._legacy_api = None
            # kagglesdk exposes a top-level client factory; fall back to attribute
            # discovery if the constructor name differs across versions.
            self._client = self._make_kagglesdk_client(kagglesdk)
        elif self.sdk == "kaggle":
            KaggleApi = _try_import_kaggle_legacy()
            self._sdk_module = None
            self._legacy_api = KaggleApi()
            self._legacy_api.authenticate()
            self._client = self._legacy_api
        else:
            raise ImportError(
                "No Kaggle SDK found. Install either `kagglesdk` or `kaggle` "
                "(pip install kaggle) to use KaggleClient."
            )

    def _make_kagglesdk_client(self, kagglesdk: Any) -> Any:
        # kagglesdk exposes a top-level `KaggleClient` or `client()` factory.
        for attr in ("KaggleClient", "client", "Client"):
            factory = getattr(kagglesdk, attr, None)
            if factory is None:
                continue
            try:
                if callable(factory) and not isinstance(factory, type):
                    return factory()
                if isinstance(factory, type):
                    return factory()
            except TypeError:
                # Some factories need explicit credentials; try no-arg then
                # raise a helpful error.
                pass
        raise ImportError(
            "kagglesdk is installed but no client factory was found. "
            "Try upgrading to the latest kagglesdk release."
        )

    # ------------------------------------------------------------------
    # Normalised accessors
    # ------------------------------------------------------------------

    def list_kernels(self, user: Optional[str] = None, **kwargs: Any) -> List[Any]:
        """List kernels; returns normalised objects with snake_case attrs."""
        if self.sdk == "kaggle":
            api = self._legacy_api
            if user is None:
                kernels = api.kernels_list()
            else:
                kernels = api.kernels_list(user=user)
        else:
            kernels = self._client.kernels_list(user=user, **kwargs) if user else self._client.kernels_list(**kwargs)
        return [_NormalisedObject(k, self.sdk) for k in kernels]

    def list_submissions(self, competition: Optional[str] = None, **kwargs: Any) -> List[Any]:
        """List submissions for a competition; returns normalised objects."""
        if self.sdk == "kaggle":
            api = self._legacy_api
            if competition:
                subs = api.competitions_submissions(competition)
            else:
                subs = api.competitions_submissions()
        else:
            if competition:
                subs = self._client.competitions_submissions(competition, **kwargs)
            else:
                subs = self._client.competitions_submissions(**kwargs)
        return [_NormalisedObject(s, self.sdk) for s in subs]

    def kernel_output(self, slug: str, path: Optional[str] = None, **kwargs: Any) -> Any:
        """Download kernel output. Canonical name for both SDKs."""
        if self.sdk == "kaggle":
            return self._legacy_api.kernels_output(slug, path=path, **kwargs)
        # kagglesdk uses the same snake_case name
        return self._client.kernels_output(slug, path=path, **kwargs)

    def kernels_output(self, slug: str, path: Optional[str] = None, **kwargs: Any) -> Any:
        """Alias for :meth:`kernel_output`."""
        return self.kernel_output(slug, path=path, **kwargs)

    def kernels_status(self, slug: str, **kwargs: Any) -> Any:
        """Query kernel status."""
        if self.sdk == "kaggle":
            return self._legacy_api.kernels_status(slug, **kwargs)
        return self._client.kernels_status(slug, **kwargs)

    def kernels_logs(self, slug: str, **kwargs: Any) -> Any:
        """Fetch kernel logs."""
        if self.sdk == "kaggle":
            return self._legacy_api.kernels_logs(slug, **kwargs)
        return self._client.kernels_logs(slug, **kwargs)

    def competition_topics(self, competition: str, **kwargs: Any) -> Any:
        """List discussion topics for a competition (canonical name)."""
        if self.sdk == "kaggle":
            # Legacy KaggleApi uses `competition_topic_fields` or similar;
            # fall back to raw method lookup.
            for attr in ("competition_topic_fields", "competition_topics_list",
                         "competition_topics", "list_competition_topics"):
                if hasattr(self._legacy_api, attr):
                    method = getattr(self._legacy_api, attr)
                    return method(competition, **kwargs) if competition else method(**kwargs)
            raise AttributeError(
                "KaggleApi has no competition topic method; upgrade the `kaggle` package."
            )
        for attr in ("competition_topic_fields", "competition_topics_list",
                     "competition_topics", "list_competition_topics"):
            if hasattr(self._client, attr):
                method = getattr(self._client, attr)
                return method(competition, **kwargs) if competition else method(**kwargs)
        raise AttributeError(
            "kagglesdk client has no competition topic method; upgrade kagglesdk."
        )

    def get(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Generic accessor: look up a method by canonical name on the raw SDK.

        Useful for SDK-specific methods that are not part of the stable facade.
        """
        target = self._legacy_api if self.sdk == "kaggle" else self._client
        # Try canonical name first, then legacy alias
        if hasattr(target, method_name):
            return getattr(target, method_name)(*args, **kwargs)
        alias = _LEGACY_ALIASES.get(method_name)
        if alias and hasattr(target, alias):
            return getattr(target, alias)(*args, **kwargs)
        reverse = {v: k for k, v in _LEGACY_ALIASES.items()}.get(method_name)
        if reverse and hasattr(target, reverse):
            return getattr(target, reverse)(*args, **kwargs)
        raise AttributeError(f"No method '{method_name}' on {type(target).__name__}")


# Backwards compatibility: agents sometimes import `KaggleApi` directly.
KaggleApi = KaggleClient


__all__ = [
    "KaggleClient",
    "KaggleApi",
    "detect_sdk",
]
