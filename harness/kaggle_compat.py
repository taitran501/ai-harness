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

import re
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


# Attribute-name mapping: canonical (snake_case) name -> candidate raw
# attribute names, tried in order. Callers use ONLY the canonical name; the
# wrapper translates to whichever SDK shape is installed.
_LEGACY_ALIASES = {
    # Submission / kernel fields. MUST cover BOTH object shapes that appear
    # in the wild (pinned by tests/test_api_reference.py):
    #   * legacy kaggle 1.x camelCase objects (publicScore, createdAt, ...)
    #   * kaggle 2.x / kagglesdk `ApiSubmission` snake_case objects
    #     (public_score, date, total_bytes)
    "public_score": ("publicScore",),
    "private_score": ("privateScore",),
    "total_votes": ("totalVotes", "votes", "vote_count", "score"),
    "votes": ("total_votes", "totalVotes", "vote_count", "score"),
    "score": ("votes", "total_votes", "totalVotes", "vote_count"),
    "created_at": ("createdAt", "submittedDate", "date", "creationDate", "post_date", "postDate", "submitted_date"),
    "date": ("submittedDate", "createdAt", "post_date", "postDate", "submitted_date", "created_at"),
    "submitted_date": ("submittedDate", "date", "createdAt", "post_date"),
    "post_date": ("date", "created_at", "createdAt", "submittedDate"),
    "author": ("author_name", "authorName", "user", "userName", "username"),
    "author_name": ("author", "authorName", "user", "userName", "username"),
    "content": ("message", "body", "raw_markdown", "rawMarkdown"),
    "message": ("content", "body", "raw_markdown", "rawMarkdown"),
    "body": ("content", "message", "raw_markdown", "rawMarkdown"),
    "comment_count": ("commentCount", "comments", "total_comments"),
    "comments": ("comment_count", "commentCount", "total_comments"),
    "total_bytes": ("totalBytes",),
    "file_name": ("fileName",),
    "team_name": ("teamName",),
    "failure_message": ("failureMessage",),
    "error_message": ("errorMessage",),
    # Method-name aliases (used by KaggleClient.get): the SDK sometimes
    # exposes camelCase where the facade is snake_case.
    "kernels_output": ("kernelOutput",),
    "kernels_logs": ("kernelLogs",),
    "kernels_status": ("kernelStatus",),
    "competition_topics_list": ("competitionTopicsList",),
    "list_competition_files": ("listCompetitionFiles",),
    "list_kernels": ("listKernels",),
    "list_submissions": ("listSubmissions",),
}


def _describe_api_error(exc: BaseException) -> str:
    """Build an actionable one-line description for a failed Kaggle API call.

    The raw ``requests.HTTPError`` message (``400 Client Error: Bad Request
    for url: ...``) hides the payload where Kaggle explains *why* (daily
    submission limit, bad kernel slug, ...). When the exception carries a
    ``response`` (requests-shaped), surface its body (truncated) and URL so
    the caller — usually an agent — can act on the error without guessing.
    """
    response = getattr(exc, "response", None)
    body = ""
    url = ""
    if response is not None:
        try:
            body = (response.text or "")[:2000]
        except Exception:  # noqa: BLE001 - body is diagnostic, never fatal
            body = ""
        try:
            url = str(getattr(response, "url", "") or "")
        except Exception:  # noqa: BLE001 - url is diagnostic, never fatal
            url = ""
    parts = [f"Kaggle submission dispatch failed ({exc})"]
    if url:
        parts.append(f"url={url}")
    parts.append(f"response_body={body!r}" if body else "response_body=<none>")
    return " ".join(parts)


def _alias_candidates(name: str) -> Tuple[str, ...]:
    """Raw attribute names that may back a requested (normalised) name.

    Forward: the normalised name maps to its ordered raw candidates.
    Reverse: a legacy name requested on an object that holds the normalised
    attribute resolves back to the normalised key.
    Also dynamically generates camelCase and snake_case candidates.
    """
    cands = list(_LEGACY_ALIASES.get(name, ()))
    reverse = [
        canonical
        for canonical, raws in _LEGACY_ALIASES.items()
        if canonical != name and name in raws
    ]
    # Dynamic camelCase candidate if name is snake_case
    if "_" in name:
        parts = name.split("_")
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        if camel not in cands:
            cands.append(camel)
    # Dynamic snake_case candidate if name is camelCase
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    if snake != name and snake not in cands and snake not in reverse:
        reverse.append(snake)
    return tuple(cands + reverse)


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
        # 0. Dict lookup if raw is dict
        if isinstance(raw, dict):
            if name in raw:
                return self._wrap(raw[name])
            for alias in _alias_candidates(name):
                if alias in raw:
                    return self._wrap(raw[alias])
        # 1. Direct attribute
        if hasattr(raw, name):
            value = getattr(raw, name)
            return self._wrap(value)
        # 2/3. Forward + reverse alias candidates (see _alias_candidates)
        for alias in _alias_candidates(name):
            if hasattr(raw, alias):
                return self._wrap(getattr(raw, alias))
        # 4. Pydantic-style: try to_dict then lookup
        if hasattr(raw, "model_dump"):
            data = raw.model_dump()
            if name in data:
                return self._wrap(data[name])
            for alias in _alias_candidates(name):
                if alias in data:
                    return self._wrap(data[alias])
        if hasattr(raw, "dict"):
            data = raw.dict()
            if name in data:
                return self._wrap(data[name])
            for alias in _alias_candidates(name):
                if alias in data:
                    return self._wrap(data[alias])
        raise AttributeError(
            f"'{type(raw).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, name: str) -> Any:
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    def __contains__(self, name: str) -> bool:
        try:
            getattr(self, name)
            return True
        except AttributeError:
            return False

    def keys(self) -> List[str]:
        raw = object.__getattribute__(self, "_raw")
        if isinstance(raw, dict):
            return list(raw.keys())
        if hasattr(raw, "model_dump"):
            return list(raw.model_dump().keys())
        if hasattr(raw, "dict"):
            return list(raw.dict().keys())
        if hasattr(raw, "__dict__"):
            return [k for k in raw.__dict__.keys() if not k.startswith("_")]
        return []

    def values(self) -> List[Any]:
        return [self.get(k) for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, self.get(k)) for k in self.keys()]

    def _wrap(self, value: Any) -> Any:
        import datetime
        if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
            return value
        if hasattr(value, "name") and not isinstance(value, type):
            name_val = getattr(value, "name")
            if isinstance(name_val, str):
                return name_val
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
        self._legacy_api_cache: Optional[Any] = None
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
        """List kernels; returns normalised objects with snake_case attrs.

        Uses the self-contained legacy ``KaggleApi.kernels_list`` for BOTH SDKs
        (the kagglesdk client exposes only namespaced services —
        ``client.kernels`` — with no flat ``kernels_list``, so the legacy entry
        point is the reliable cross-SDK path).
        """
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        if user is None:
            kernels = api.kernels_list(**kwargs)
        else:
            kernels = api.kernels_list(user=user, **kwargs)
        return [_NormalisedObject(k, self.sdk) for k in kernels]

    kernels_list = list_kernels

    def list_submissions(self, competition: Optional[str] = None, **kwargs: Any) -> List[Any]:
        """List submissions for a competition; returns normalised objects.

        Uses the self-contained legacy ``KaggleApi.competition_submissions`` for
        BOTH SDKs (the kagglesdk client exposes only namespaced low-level
        services, not this flat call, so the legacy entry point is the
        reliable cross-SDK path).
        """
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        if competition:
            subs = api.competition_submissions(competition)
        else:
            subs = api.competition_submissions()
        return [_NormalisedObject(s, self.sdk) for s in subs]

    def submit_submission(
        self,
        file_name: str,
        message: str,
        competition: str,
        kernel: Optional[str] = None,
        kernel_version: Optional[str] = None,
    ) -> Any:
        """Submit to a competition (standard file, or code-competition kernel output).

        Uses the self-contained legacy ``KaggleApi`` (``competition_submit`` /
        ``competition_submit_code``) which encapsulates the upload + create
        round-trip that the kagglesdk low-level client does not expose as a
        single call. Returns a normalised response exposing ``ref`` and
        ``message``.
        """
        api = self._legacy_api_instance()
        try:
            if kernel:
                resp = api.competition_submit_code(
                    file_name,
                    message,
                    competition,
                    kernel=kernel,
                    kernel_version=kernel_version,
                )
            else:
                resp = api.competition_submit(file_name, message, competition)
        except Exception as exc:  # noqa: BLE001 - surface Kaggle API errors with body
            raise RuntimeError(_describe_api_error(exc)) from exc
        return _NormalisedObject(resp, self.sdk)

    def kernels_push(
        self, folder: str, timeout: Optional[int] = None, acc: Optional[str] = None
    ) -> Any:
        """Push a kernel folder (``kernel-metadata.json`` + code) to Kaggle.

        Self-contained legacy call: reads the metadata + code, builds the save
        request, and pushes. Returns a normalised response with ``ref``,
        ``url``, ``kernel_id``, ``version_number`` and the ``invalid_*`` source
        warnings.
        """
        api = self._legacy_api_instance()
        resp = api.kernels_push(folder, timeout=timeout, acc=acc)
        return _NormalisedObject(resp, self.sdk)

    def _legacy_api_instance(self) -> Any:
        """Return a self-contained legacy ``KaggleApi`` instance (cached).

        The legacy ``kaggle`` package is always present alongside the
        ``kaggle`` CLI, and its high-level methods encapsulate the multi-step
        SDK plumbing. We reuse it for dispatch operations even when the
        detected SDK is ``kagglesdk``.
        """
        if self.sdk == "kaggle":
            return self._legacy_api
        if self._legacy_api_cache is None:
            KaggleApi = _try_import_kaggle_legacy()
            if KaggleApi is None:
                raise ImportError(
                    "Legacy 'kaggle' package is required for submission/kernel "
                    "dispatch but is not installed (pip install kaggle)."
                )
            api = KaggleApi()
            api.authenticate()
            self._legacy_api_cache = api
        return self._legacy_api_cache

    def kernel_output(self, slug: str, path: Optional[str] = None, **kwargs: Any) -> Any:
        """Download kernel output. Canonical name for both SDKs."""
        if self.sdk == "kaggle":
            return self._legacy_api.kernels_output(slug, path=path, **kwargs)
        # kagglesdk clients are namespaced (client.kernels.*), so the legacy
        # flat API is the stable cross-SDK path (same as list_submissions).
        return self._legacy_api_instance().kernels_output(slug, path=path, **kwargs)

    def kernels_output(self, slug: str, path: Optional[str] = None, **kwargs: Any) -> Any:
        """Alias for :meth:`kernel_output`."""
        return self.kernel_output(slug, path=path, **kwargs)

    def kernels_status(self, slug: str, **kwargs: Any) -> Any:
        """Query kernel status, returning a normalised object with dict-like access."""
        try:
            if self.sdk == "kaggle":
                raw = self._legacy_api.kernels_status(slug, **kwargs)
            else:
                raw = self._legacy_api_instance().kernels_status(slug, **kwargs)
        except Exception as exc:
            # Handle transient 404 when kernel was newly pushed and session status is initializing
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code == 404 or "404" in str(exc):
                return _NormalisedObject({"status": "QUEUED", "failureMessage": None}, self.sdk)
            raise
        return _NormalisedObject(raw, self.sdk)

    def kernels_logs(self, slug: str, **kwargs: Any) -> Any:
        """Fetch kernel logs."""
        if self.sdk == "kaggle":
            return self._legacy_api.kernels_logs(slug, **kwargs)
        return self._legacy_api_instance().kernels_logs(slug, **kwargs)

    def competition_topics(self, competition: str, **kwargs: Any) -> List[Any]:
        """List discussion topics for a competition; returns normalised objects.

        Routed through ``KaggleApi.competition_list_topics`` for BOTH SDKs:
        the kagglesdk client is namespaced (no flat topic method), and the
        legacy ``competition_topic_fields`` attribute is a class-level constant
        (field-projection list), not a callable — attribute probing against it
        used to raise ``TypeError: 'list' object is not callable``.
        """
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        if not hasattr(api, "competition_list_topics"):
            raise AttributeError(
                "KaggleApi has no competition_list_topics method; upgrade the `kaggle` package (>= 2.x)."
            )
        page_size = kwargs.pop("page_size", None)
        valid_kwargs: Dict[str, Any] = {}
        if "sort_by" in kwargs and kwargs["sort_by"] is not None:
            valid_kwargs["sort_by"] = kwargs["sort_by"]
        if "page" in kwargs and kwargs["page"] is not None:
            valid_kwargs["page"] = kwargs["page"]
        resp = api.competition_list_topics(competition, **valid_kwargs)
        raw_topics = getattr(resp, "topics", None)
        if raw_topics is None:
            raw_topics = resp if isinstance(resp, list) else []
        if page_size is not None and isinstance(raw_topics, list):
            raw_topics = raw_topics[:page_size]
        return [_NormalisedObject(t, self.sdk) for t in raw_topics]

    def competition_topic_messages(
        self, topic_id: int, competition: Optional[str] = None, **kwargs: Any
    ) -> List[Any]:
        """Fetch messages/comments for a topic; returns normalised objects."""
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        page_size = kwargs.get("page_size")
        resp = None
        if competition and hasattr(api, "competition_list_topic_messages"):
            try:
                c_kwargs: Dict[str, Any] = {}
                if "sort_by" in kwargs:
                    c_kwargs["sort_by"] = kwargs["sort_by"]
                if "page_size" in kwargs:
                    c_kwargs["page_size"] = kwargs["page_size"]
                resp = api.competition_list_topic_messages(competition, topic_id, **c_kwargs)
            except Exception:
                resp = None
        if resp is None and hasattr(api, "forums_topic_show"):
            try:
                f_kwargs: Dict[str, Any] = {}
                if "page_size" in kwargs:
                    f_kwargs["page_size"] = kwargs["page_size"]
                if "page_token" in kwargs:
                    f_kwargs["page_token"] = kwargs["page_token"]
                resp = api.forums_topic_show(topic_id, **f_kwargs)
            except Exception:
                resp = None
        if resp is None and hasattr(api, "competition_list_topic_messages") and not competition:
            try:
                resp = api.competition_list_topic_messages(topic_id)
            except Exception:
                pass
        raw_msgs = getattr(resp, "messages", None)
        if raw_msgs is None:
            raw_msgs = getattr(resp, "comments", None)
        if raw_msgs is None:
            raw_msgs = resp if isinstance(resp, list) else []
        if page_size is not None and isinstance(raw_msgs, list):
            raw_msgs = raw_msgs[:page_size]
        return [_NormalisedObject(m, self.sdk) for m in raw_msgs]

    def get(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Generic accessor: look up a method by canonical name on the raw SDK.

        Useful for SDK-specific methods that are not part of the stable facade.
        """
        target = self._legacy_api if self.sdk == "kaggle" else self._client
        # Try canonical name first, then legacy aliases
        if hasattr(target, method_name):
            return getattr(target, method_name)(*args, **kwargs)
        for alias in _alias_candidates(method_name):
            if hasattr(target, alias):
                return getattr(target, alias)(*args, **kwargs)
        raise AttributeError(f"No method '{method_name}' on {type(target).__name__}")


# Backwards compatibility: agents sometimes import `KaggleApi` directly.
KaggleApi = KaggleClient


# ----------------------------------------------------------------------
# Retry with exponential backoff for transient Kaggle HTTP 429 / 5xx
# ----------------------------------------------------------------------

_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_transient_error(exc: BaseException) -> bool:
    """Heuristic: treat 429 / 5xx / timeout / connection errors as transient."""
    name = type(exc).__name__
    if name in ("Timeout", "TimeoutError", "ConnectionError", "ConnectionResetError",
                "ConnectionAbortedError", "ChunkedEncodingError", "RemoteDisconnected",
                "ProxyError", "SSLError", "CertificateError"):
        return True
    msg = str(exc).lower()
    if any(token in msg for token in ("429", "too many requests", "rate limit",
                                      "500", "502", "503", "504",
                                      "timeout", "timed out", "connection reset",
                                      "temporarily unavailable", "server error")):
        return True
    return False


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    transient_check: Callable[[BaseException], bool] = _is_transient_error,
) -> T:
    """Call ``fn`` with exponential backoff on transient errors.

    Defaults are tuned for the Kaggle API's public rate limits (~429 on
    burst, 5xx on upstream hiccups). Callers can override ``max_attempts``
    or supply a custom ``transient_check`` for provider-specific heuristics.

    Raises the last exception if all attempts are exhausted.
    """
    attempt = 0
    delay = base_delay
    last_exc: Optional[BaseException] = None
    while attempt < max_attempts:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            if attempt >= max_attempts or not transient_check(exc):
                raise
            sleep = min(delay, max_delay)
            time.sleep(sleep)
            delay *= multiplier
    # Unreachable, but satisfies type checkers.
    assert last_exc is not None
    raise last_exc


__all__ = [
    "KaggleClient",
    "KaggleApi",
    "detect_sdk",
    "retry_with_backoff",
    "_is_transient_error",
]
