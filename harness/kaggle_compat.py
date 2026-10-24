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
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from harness.endgame_submission import KaggleLeaderboardTracker

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
    "comment_count": ("commentCount", "comments", "total_comments", "total_replies"),
    "comments": ("comment_count", "commentCount", "total_comments", "total_replies"),
    "total_replies": ("comment_count", "commentCount", "comments", "total_comments"),
    "total_bytes": ("totalBytes", "size", "bytes"),
    "size": ("total_bytes", "totalBytes", "bytes"),
    "file_name": ("fileName",),
    "team_id": ("teamId",),
    "team_name": ("teamName",),
    "is_code_competition": ("isKernelsSubmissionsOnly", "is_kernels_submissions_only"),
    "submissions_disabled": ("submissionsDisabled", "submissions_disabled"),
    "max_daily_submissions": ("maxDailySubmissions", "max_daily_submissions"),
    "evaluation_metric": ("evaluationMetric", "evaluation_metric"),
    "team_count": ("teamCount", "team_count"),
    "user_rank": ("userRank", "user_rank"),
    "user_has_entered": ("userHasEntered", "user_has_entered"),
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
    Also dynamically generates camelCase and snake_case candidates for all.
    """
    cands = list(_LEGACY_ALIASES.get(name, ()))
    reverse = [
        canonical
        for canonical, raws in _LEGACY_ALIASES.items()
        if canonical != name and name in raws
    ]
    expanded: List[str] = []
    for cand in cands + reverse + [name]:
        if cand not in expanded and cand != name:
            expanded.append(cand)
        if "_" in cand:
            parts = cand.split("_")
            camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
            if camel not in expanded and camel != name:
                expanded.append(camel)
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", cand).lower()
        if snake not in expanded and snake != name:
            expanded.append(snake)
    return tuple(expanded)


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

    def __getitem__(self, name: Any) -> Any:
        if isinstance(name, int):
            raw = object.__getattribute__(self, "_raw")
            for attr in ("competitions", "topics", "messages", "submissions", "kernels", "dataset_files", "files"):
                if hasattr(raw, attr):
                    sublist = getattr(raw, attr)
                    if isinstance(sublist, (list, tuple)):
                        return self._wrap(sublist[name])
            if isinstance(raw, (list, tuple)):
                return self._wrap(raw[name])
            raise IndexError(f"'{type(raw).__name__}' object is not indexable")
        try:
            return getattr(self, str(name))
        except AttributeError as exc:
            raise KeyError(name) from exc

    def __iter__(self):
        raw = object.__getattribute__(self, "_raw")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                yield self._wrap(item)
            return
        for attr in ("competitions", "topics", "messages", "submissions", "kernels", "dataset_files", "files"):
            if hasattr(raw, attr):
                sublist = getattr(raw, attr)
                if isinstance(sublist, (list, tuple)):
                    for item in sublist:
                        yield self._wrap(item)
                    return
        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes, dict)):
            for item in raw:
                yield self._wrap(item)
            return
        raise TypeError(f"'{type(raw).__name__}' object is not iterable")

    def __len__(self) -> int:
        raw = object.__getattribute__(self, "_raw")
        if hasattr(raw, "__len__"):
            return len(raw)
        for attr in ("competitions", "topics", "messages", "submissions", "kernels", "dataset_files", "files"):
            if hasattr(raw, attr):
                sublist = getattr(raw, attr)
                if hasattr(sublist, "__len__"):
                    return len(sublist)
        raise TypeError(f"object of type '{type(raw).__name__}' has no len()")

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

    competition_submissions = list_submissions

    def get_competition(self, competition: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive metadata for a competition.

        Returns a dictionary with canonical keys:
        - ref: competition slug
        - slug: normalized slug
        - title: human-readable title
        - url: competition URL
        - description: short description
        - deadline: submission deadline
        - is_ongoing: boolean whether competition is still active
        - category: competition category
        - reward: prize / reward description
        - team_count: total participating teams
        - user_rank: user's current rank (or None)
        - user_has_entered: boolean
        - max_daily_submissions: allowed daily submissions
        - evaluation_metric: official competition evaluation metric
        - is_code_competition: boolean (notebook submission only)
        - submissions_disabled: boolean
        """
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        slug = competition.strip().split("/")[-1]
        try:
            resp = api.competitions_list(search=slug)
            comps = getattr(resp, "competitions", resp) if resp else []
            target = None
            for c in (comps or []):
                c_ref = getattr(c, "ref", "") or getattr(c, "url", "")
                if slug.lower() in str(c_ref).lower() or getattr(c, "id", None) == slug:
                    target = c
                    break
            if target is None and comps:
                target = comps[0]
            if target is None:
                return None
            norm = _NormalisedObject(target, self.sdk)
            deadline_str = str(norm.get("deadline", ""))
            return {
                "id": norm.get("id"),
                "ref": slug,
                "slug": slug,
                "title": norm.get("title"),
                "url": norm.get("url") or f"https://www.kaggle.com/competitions/{slug}",
                "description": norm.get("description"),
                "deadline": deadline_str,
                "is_ongoing": not bool(norm.get("submissions_disabled")),
                "category": norm.get("category"),
                "reward": norm.get("reward"),
                "team_count": norm.get("team_count") or norm.get("teamCount"),
                "user_rank": norm.get("user_rank") or norm.get("userRank"),
                "user_has_entered": norm.get("user_has_entered") or norm.get("userHasEntered"),
                "max_daily_submissions": norm.get("max_daily_submissions") or norm.get("maxDailySubmissions"),
                "evaluation_metric": norm.get("evaluation_metric") or norm.get("evaluationMetric"),
                "is_code_competition": bool(norm.get("is_kernels_submissions_only") or norm.get("isKernelsSubmissionsOnly")),
                "submissions_disabled": bool(norm.get("submissions_disabled") or norm.get("submissionsDisabled")),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ref": slug, "slug": slug, "error": str(exc)}

    competition_view = get_competition

    def list_competitions(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        sort_by: Optional[str] = None,
        page_size: int = 20,
        **kwargs: Any,
    ) -> List[Any]:
        """List competitions; returns normalised objects with snake_case attrs."""
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        resp = api.competitions_list(
            search=search, category=category, sort_by=sort_by, page_size=page_size, **kwargs
        )
        comps = getattr(resp, "competitions", resp) if resp else []
        return [_NormalisedObject(c, self.sdk) for c in (comps or [])]

    competitions_list = list_competitions

    def get_leaderboard(
        self, competition: str, page_size: int = 20, **kwargs: Any
    ) -> List[Any]:
        """Fetch competition public leaderboard entries.

        Returns normalised objects with snake_case attributes:
        team_id, team_name, score, submission_date.
        """
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        lb = api.competition_leaderboard_view(competition, page_size=page_size, **kwargs)
        return [_NormalisedObject(item, self.sdk) for item in (lb or [])]

    def get_submission_limits(self, competition: str) -> Dict[str, Any]:
        """Fetch live daily submission limits for a competition.

        Returns a dictionary with keys:
        - num_today: number of submissions made today (UTC)
        - num_allowed_now: number of submissions remaining today
        - num_total: total lifetime submissions made
        - limited_by_total: boolean whether blocked by lifetime limit
        - has_remaining_quota: boolean (num_allowed_now > 0)
        - daily_quota: total daily quota (num_today + num_allowed_now)
        """
        api = self._legacy_api_instance()
        try:
            raw = api.competition_get_submission_limits(competition)
            if isinstance(raw, str):
                import json
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            elif hasattr(raw, "to_dict"):
                data = raw.to_dict()
            else:
                data = {}
            today = int(data.get("numToday") or 0)
            allowed = int(data.get("numAllowedNow") or 0)
            total = int(data.get("numTotal") or 0)
            limited = bool(data.get("limitedByTotal", False))
            return {
                "competition": competition,
                "num_today": today,
                "num_allowed_now": allowed,
                "num_total": total,
                "limited_by_total": limited,
                "has_remaining_quota": allowed > 0 and not limited,
                "daily_quota": today + allowed,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "competition": competition,
                "num_today": None,
                "num_allowed_now": None,
                "num_total": None,
                "limited_by_total": False,
                "has_remaining_quota": None,
                "error": str(exc),
            }


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

    kernel_status = kernels_status
    kernel_logs = kernels_logs

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
            s_val = str(kwargs["sort_by"]).lower()
            if s_val == "hotness":
                s_val = "hot"
            valid_kwargs["sort_by"] = s_val
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

    def get_topic(self, topic_id: int, competition: Optional[str] = None) -> Dict[str, Any]:
        """Fetch complete topic details including title, author, content, and all comments."""
        def _clean_text(html: str) -> str:
            if not html:
                return ""
            return re.sub(r"<[^>]+>", "", html).strip()

        # 1. Try modern kagglesdk discussions API client if available
        try:
            from kagglesdk.discussions.types.discussions_api_service import (
                ApiGetTopicRequest,
                ApiListCommentsRequest,
            )

            disc_client = getattr(
                getattr(self._client, "discussions", None),
                "discussion_api_client",
                None,
            )
            if disc_client is not None:
                req = ApiGetTopicRequest()
                req.id = topic_id
                res = disc_client.get_topic(req)
                t_obj = res.topic

                comments_list = []
                try:
                    req_c = ApiListCommentsRequest()
                    req_c.topic_id = topic_id
                    comms = disc_client.list_comments(req_c)
                    comments_list = getattr(comms, "comments", []) or []
                except Exception:
                    pass

                return {
                    "id": topic_id,
                    "title": getattr(t_obj, "title", ""),
                    "author": getattr(t_obj, "author_name", "") or "",
                    "date": str(getattr(t_obj, "post_date", "") or ""),
                    "votes": getattr(t_obj, "votes", 0) or 0,
                    "content": _clean_text(getattr(t_obj, "content", "") or ""),
                    "content_raw": getattr(t_obj, "content", "") or "",
                    "comments_count": len(comments_list),
                    "comments": [
                        {
                            "id": getattr(c, "id", None),
                            "author": getattr(c, "author_name", "") or "",
                            "date": str(getattr(c, "post_date", "") or ""),
                            "votes": getattr(c, "votes", 0) or 0,
                            "content": _clean_text(getattr(c, "content", "") or ""),
                            "content_raw": getattr(c, "content", "") or "",
                        }
                        for c in comments_list
                    ],
                }
        except Exception:
            pass

        # 2. Fallback to competition_topic_messages
        msgs = self.competition_topic_messages(topic_id, competition=competition)
        first_content = msgs[0].content if msgs else ""
        return {
            "id": topic_id,
            "title": f"Topic {topic_id}",
            "author": getattr(msgs[0], "author", "") if msgs else "",
            "date": str(getattr(msgs[0], "date", "")) if msgs else "",
            "votes": getattr(msgs[0], "votes", 0) if msgs else 0,
            "content": _clean_text(first_content),
            "content_raw": first_content,
            "comments_count": max(0, len(msgs) - 1),
            "comments": [
                {
                    "id": getattr(m, "id", None),
                    "author": getattr(m, "author", "") or getattr(m, "author_name", ""),
                    "date": str(getattr(m, "date", "") or getattr(m, "post_date", "")),
                    "votes": getattr(m, "votes", 0) or getattr(m, "total_votes", 0),
                    "content": _clean_text(getattr(m, "content", "") or getattr(m, "message", "")),
                    "content_raw": getattr(m, "content", "") or getattr(m, "message", ""),
                }
                for m in msgs[1:]
            ],
        }

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

    @property
    def api(self) -> Any:
        """Underlying legacy KaggleApi instance for direct operations."""
        return self._legacy_api_instance()

    @property
    def _api(self) -> Any:
        """Alias for :attr:`api`."""
        return self._legacy_api_instance()

    def kernels_pull(
        self,
        kernel: str,
        path: Optional[str] = None,
        metadata: bool = False,
        quiet: bool = True,
    ) -> Any:
        """Pull notebook/script and metadata for a Kaggle kernel."""
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        return api.kernels_pull(kernel, path=path or ".", metadata=metadata, quiet=quiet)

    kernel_pull = kernels_pull
    pull_kernel = kernels_pull

    def get_kernel(self, kernel: str) -> Optional[Any]:
        """Get metadata for a specific kernel (by 'username/slug' or 'slug')."""
        api = self._legacy_api if self.sdk == "kaggle" else self._legacy_api_instance()
        search_term = kernel.split("/", 1)[-1] if "/" in kernel else kernel
        user_name = kernel.split("/", 1)[0] if "/" in kernel else None
        try:
            results = api.kernels_list(search=search_term, user=user_name, page_size=20)
            for item in (results or []):
                item_ref = getattr(item, "ref", "") or getattr(item, "slug", "")
                if item_ref == kernel or item_ref.endswith(f"/{search_term}"):
                    return _NormalisedObject(item, self.sdk)
            if results:
                return _NormalisedObject(results[0], self.sdk)
        except Exception:
            pass
        return None

    # Discussion & Leaderboard aliases
    leaderboard = get_leaderboard
    list_topics = competition_topics
    topics = competition_topics
    discussions = competition_topics
    list_discussions = competition_topics
    topic_messages = competition_topic_messages
    list_topic_messages = competition_topic_messages


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
    "KaggleLeaderboardTracker",
    "detect_sdk",
    "retry_with_backoff",
    "_is_transient_error",
]
