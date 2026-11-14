from __future__ import annotations
# Consolidated Kaggle CLI & Compat Test Suite
# Tier: integration
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Imports from tests/test_kaggle_compat.py ---
import unittest
from unittest import mock
from types import SimpleNamespace
from harness.kaggle_compat import (
    KaggleClient,
    _is_transient_error,
    _NormalisedObject,
    detect_sdk,
    retry_with_backoff,
)
# --- Tests from tests/test_kaggle_compat.py ---
"""Tests for ``harness.kaggle_compat`` — the Kaggle SDK compatibility facade.

This module runs inside the *remote* Kaggle kernel (via the generated
``kaggle_template``), where a normalisation bug means a full kernel loss with
no local traceback. The pure, network-free surface is exercised here:

- ``detect_sdk`` precedence (kagglesdk > kaggle > none)
- ``_NormalisedObject`` attribute normalisation (direct / legacy alias /
  reverse alias / pydantic ``model_dump`` / nested wrap / miss)
- ``_is_transient_error`` classification
- ``retry_with_backoff`` success / retry / permanent-fail / exhaustion

Client dispatch paths are unit-tested below with faked legacy instances.
The upstream live API surface is not re-pinned here; it is verified by the
live smoke path (see ``docs/kaggle_execution.md``).
"""




class _FakeLegacy:
    """Object exposing camelCase attributes (legacy ``kaggle`` shape)."""

    def __init__(self):
        self.totalVotes = 5
        self.publicScore = 0.91
        self.createdAt = "2026-01-01T00:00:00Z"


class _FakePydantic:
    """Object exposing canonical attributes + a model_dump() (new SDK shape)."""

    def __init__(self):
        self.total_votes = 7
        self.title = "kernel-1"

    def model_dump(self):
        return {"total_votes": 7, "title": "kernel-1"}

class _FakeCanonicalOnly:
    """Canonical snake attributes, NO model_dump/dict -> isolates step-2 alias."""

    def __init__(self):
        self.total_votes = 42


class _FakeKagglesdkShape:
    """Live kaggle 2.x / kagglesdk ``ApiSubmission`` shape (snake_case, ``date``)."""

    def __init__(self):
        self.public_score = 0.958
        self.private_score = 0.957
        self.date = "2026-10-17T21:14:27Z"
        self.file_name = "submission.csv"
        self.total_bytes = 12345


class _FakeLegacyDateShape:
    """Legacy kaggle 1.x shape with the ``submittedDate`` timestamp field."""

    def __init__(self):
        self.publicScore = 0.958
        self.submittedDate = "2026-10-17T21:14:27Z"
        self.fileName = "submission.csv"


class DetectSdkTests(unittest.TestCase):
    def test_detect_sdk_returns_a_known_value(self):
        # Both SDKs are installed in this environment, so kagglesdk wins.
        self.assertIn(detect_sdk(), {"kagglesdk", "kaggle", "none"})

    def test_kagglesdk_is_preferred_when_both_present(self):
        import importlib.util

        if importlib.util.find_spec("kagglesdk") and importlib.util.find_spec("kaggle"):
            self.assertEqual(detect_sdk(), "kagglesdk")


class NormalisedObjectTests(unittest.TestCase):
    def test_direct_attribute(self):
        obj = _NormalisedObject(_FakeLegacy(), "kaggle")
        self.assertEqual(obj.totalVotes, 5)
        self.assertEqual(obj.publicScore, 0.91)

    def test_legacy_alias_to_canonical(self):
        # Caller asks the canonical snake name; object only has camel.
        obj = _NormalisedObject(_FakeLegacy(), "kaggle")
        self.assertEqual(obj.total_votes, 5)
        self.assertEqual(obj.created_at, "2026-01-01T00:00:00Z")

    def test_reverse_alias_canonical_to_legacy(self):
        # Object has canonical snake attr, NO model_dump; caller asks legacy
        # camel name -> must resolve through the forward-alias step.
        obj = _NormalisedObject(_FakeCanonicalOnly(), "kagglesdk")
        self.assertEqual(obj.totalVotes, 42)

    def test_created_at_resolves_from_kagglesdk_date_field(self):
        # Live shape: ApiSubmission exposes `date`, not `created_at`.
        obj = _NormalisedObject(_FakeKagglesdkShape(), "kagglesdk")
        self.assertEqual(obj.created_at, "2026-10-17T21:14:27Z")
        self.assertEqual(obj.date, "2026-10-17T21:14:27Z")

    def test_date_resolves_from_legacy_submitted_date(self):
        # Legacy 1.x shape: timestamp lives in `submittedDate`.
        obj = _NormalisedObject(_FakeLegacyDateShape(), "kaggle")
        self.assertEqual(obj.date, "2026-10-17T21:14:27Z")
        self.assertEqual(obj.created_at, "2026-10-17T21:14:27Z")

    def test_pydantic_model_dump_lookup(self):
        obj = _NormalisedObject(_FakePydantic(), "kagglesdk")
        self.assertEqual(obj.title, "kernel-1")

    def test_missing_attribute_raises_attributeerror(self):
        obj = _NormalisedObject(_FakeLegacy(), "kaggle")
        with self.assertRaises(AttributeError):
            _ = obj.nonExistentField

    def test_wrap_leaves_primitives_untouched(self):
        obj = _NormalisedObject(_FakeLegacy(), "kaggle")
        self.assertIsInstance(obj.totalVotes, int)
        self.assertIsInstance(obj.publicScore, float)
        self.assertIsInstance(obj.createdAt, str)

    def test_eq_and_hash_by_underlying_raw(self):
        a = _FakeLegacy()
        o1 = _NormalisedObject(a, "kaggle")
        o2 = _NormalisedObject(a, "kaggle")
        self.assertEqual(o1, o2)
        self.assertEqual(hash(o1), hash(o2))
        self.assertEqual(o1, a)  # compares through to raw

    def test_repr_names_the_raw_type(self):
        obj = _NormalisedObject(_FakeLegacy(), "kaggle")
        self.assertIn("_FakeLegacy", repr(obj))


class TransientErrorTests(unittest.TestCase):
    def test_known_transient_names(self):
        self.assertTrue(_is_transient_error(ConnectionResetError()))
        self.assertTrue(_is_transient_error(TimeoutError()))

    def test_message_based_transient(self):
        self.assertTrue(_is_transient_error(Exception("HTTP 429 too many requests")))
        self.assertTrue(_is_transient_error(Exception("upstream 503")))
        self.assertTrue(_is_transient_error(Exception("read timed out")))

    def test_permanent_errors_are_not_transient(self):
        self.assertFalse(_is_transient_error(ValueError("bad slug")))
        self.assertFalse(_is_transient_error(Exception("HTTP 404 not found")))
        self.assertFalse(_is_transient_error(KeyError("x")))


class RetryWithBackoffTests(unittest.TestCase):
    def test_success_first_try_no_retry(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        self.assertEqual(retry_with_backoff(fn, base_delay=0.0), "ok")
        self.assertEqual(calls["n"], 1)

    def test_retries_transient_then_succeeds(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionResetError("reset")
            return "recovered"

        out = retry_with_backoff(fn, max_attempts=5, base_delay=0.0)
        self.assertEqual(out, "recovered")
        self.assertEqual(calls["n"], 3)

    def test_permanent_error_raises_immediately(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            retry_with_backoff(fn, max_attempts=5, base_delay=0.0)
        self.assertEqual(calls["n"], 1)

    def test_exhaustion_raises_last_exception(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ConnectionAbortedError("flaky")

        with self.assertRaises(ConnectionAbortedError):
            retry_with_backoff(fn, max_attempts=3, base_delay=0.0)
        self.assertEqual(calls["n"], 3)

    def test_custom_transient_check_is_respected(self):
        class AlwaysTransient(BaseException):
            pass

        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise AlwaysTransient()
            return "done"

        out = retry_with_backoff(
            fn, max_attempts=3, base_delay=0.0,
            transient_check=lambda e: isinstance(e, AlwaysTransient),
        )
        self.assertEqual(out, "done")


class _FakeLegacyApi:
    """Records the high-level dispatch calls (legacy ``kaggle`` surface)."""

    def __init__(self):
        self.calls = []

    def competition_submit(self, file, message, competition):
        self.calls.append(("competition_submit", file, message, competition))
        return SimpleNamespace(ref=111)

    def competition_submit_code(self, file, message, competition, kernel=None, kernel_version=None):
        self.calls.append(("competition_submit_code", file, message, competition, kernel, kernel_version))
        return SimpleNamespace(ref=222)

    def kernels_push(self, folder, timeout=None, acc=None):
        self.calls.append(("kernels_push", folder, timeout, acc))
        return SimpleNamespace(ref=333, url="https://www.kaggle.com/kernels/own/k")

    def kernels_status(self, slug, **kwargs):
        self.calls.append(("kernels_status", slug))
        return SimpleNamespace(status="COMPLETE")

    def kernels_logs(self, slug, **kwargs):
        self.calls.append(("kernels_logs", slug))
        return "logs for " + slug

    def kernels_output(self, slug, path=None, **kwargs):
        self.calls.append(("kernels_output", slug, path))
        return "output of " + slug

    def kernels_list(self, user=None, **kwargs):
        self.calls.append(("kernels_list", user))
        return [SimpleNamespace(slug="own/k1", title="Kernel One")]

    def competition_list_topics(self, competition, **kwargs):
        self.calls.append(("competition_list_topics", competition))
        return SimpleNamespace(
            topics=[SimpleNamespace(id=7, title="Topic", author_name="a", comment_count=3)],
            total_count=1,
        )


def _make_client(sdk="kaggle"):
    """Build a ``KaggleClient`` without a live SDK (dispatch-path unit tests)."""
    c = KaggleClient.__new__(KaggleClient)
    c.sdk = sdk
    c._legacy_api = _FakeLegacyApi()
    c._legacy_api_cache = None
    c._client = c._legacy_api if sdk == "kaggle" else object()
    c._sdk_module = None
    return c


class ClientDispatchPathTests(unittest.TestCase):
    """The dispatch routing an agent would otherwise guess at by bypassing the
    facade (file vs code-comp submit, kernels push). Pinning these prevents a
    silent regression on the exact high-level calls the reference advertises.
    """

    def setUp(self):
        # These tests isolate SDK routing/error normalization. The actual permit
        # boundary is exercised without mocks by SubmissionGateTests.
        patcher = mock.patch('harness.submission_gate.consume_dispatch_permit')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_submit_file_uses_competition_submit(self):
        c = _make_client("kaggle")
        resp = c.submit_submission("sub.csv", "msg", "comp")
        self.assertEqual(c._legacy_api.calls[0][0], "competition_submit")
        self.assertEqual(c._legacy_api.calls[0][3], "comp")
        self.assertEqual(resp.ref, 111)

    def test_submit_code_uses_competition_submit_code(self):
        c = _make_client("kaggle")
        resp = c.submit_submission("out.csv", "msg", "comp", kernel="owner/k", kernel_version="2")
        self.assertEqual(c._legacy_api.calls[0][0], "competition_submit_code")
        self.assertEqual(c._legacy_api.calls[0][4], "owner/k")
        self.assertEqual(c._legacy_api.calls[0][5], "2")
        self.assertEqual(resp.ref, 222)

    def test_kernels_push_routes_to_legacy(self):
        c = _make_client("kaggle")
        resp = c.kernels_push("folder", acc="KaggleGPU")
        self.assertEqual(c._legacy_api.calls[0][0], "kernels_push")
        self.assertEqual(c._legacy_api.calls[0][3], "KaggleGPU")
        self.assertEqual(resp.ref, 333)

    def test_submit_error_surfaces_response_body_and_url(self):
        """A 400 (e.g. daily submission limit) must expose Kaggle's JSON body.

        Regression: arc-prize dispatch failed with a bare
        '400 Client Error: Bad Request for url: ...' — the agent could not tell
        daily-limit from bad-slug from bad-file without the payload.
        """
        c = _make_client("kaggle")
        exc = Exception(
            "400 Client Error: Bad Request for url: "
            "https://www.kaggle.com/competitions/arc-prize/submissions/create"
        )
        exc.response = SimpleNamespace(
            text='{"code":400,"message":"You have reached the daily submission limit"}',
            url="https://www.kaggle.com/competitions/arc-prize/submissions/create",
        )

        def boom(*args, **kwargs):
            raise exc

        c._legacy_api.competition_submit_code = boom
        with self.assertRaises(RuntimeError) as ctx:
            c.submit_submission("out.csv", "msg", "comp", kernel="owner/k", kernel_version="2")
        message = str(ctx.exception)
        self.assertIn("You have reached the daily submission limit", message)
        self.assertIn("https://www.kaggle.com/competitions/arc-prize", message)
        self.assertIsInstance(ctx.exception.__cause__, Exception)

    def test_submit_error_without_response_keeps_fallback(self):
        c = _make_client("kaggle")

        def boom(*args, **kwargs):
            raise ConnectionError("kapow")

        c._legacy_api.competition_submit = boom
        with self.assertRaises(RuntimeError) as ctx:
            c.submit_submission("sub.csv", "msg", "comp")
        message = str(ctx.exception)
        self.assertIn("kapow", message)
        self.assertIn("response_body=<none>", message)

    def test_describe_api_error_truncates_long_body(self):
        from harness.kaggle_compat import _describe_api_error

        exc = Exception("400")
        exc.response = SimpleNamespace(text="z" * 5000, url="https://kaggle.com/a")
        out = _describe_api_error(exc)
        self.assertLessEqual(out.count("z"), 2000)
        self.assertIn("response_body=", out)

    def test_legacy_api_instance_returns_configured_api_for_kaggle(self):
        c = _make_client("kaggle")
        self.assertIs(c._legacy_api_instance(), c._legacy_api)

    def test_get_resolves_registered_method(self):
        c = _make_client("kaggle")
        c.get("kernels_push", "f")
        self.assertEqual(c._legacy_api.calls[-1][0], "kernels_push")


class ClientKagglesdkKernelRoutingTests(unittest.TestCase):
    """On kagglesdk the raw client is namespaced (client.kernels.*) and has no
    flat kernels_status / kernels_logs / kernels_output — the facade must route
    through the legacy flat API instead (regression: live RSNA run raised
    ``AttributeError: 'KaggleClient' object has no attribute 'kernels_status'``).
    """

    def _make_kagglesdk_client(self):
        c = _make_client("kagglesdk")
        # Pre-seed the legacy cache so no real KaggleApi is constructed.
        c._legacy_api_cache = _FakeLegacyApi()
        return c

    def test_kernels_status_routes_to_legacy_api(self):
        c = self._make_kagglesdk_client()
        resp = c.kernels_status("owner/kernel-1")
        self.assertEqual(c._legacy_api_cache.calls[0][:2], ("kernels_status", "owner/kernel-1"))
        self.assertEqual(resp.status, "COMPLETE")

    def test_kernels_logs_routes_to_legacy_api(self):
        c = self._make_kagglesdk_client()
        out = c.kernels_logs("owner/kernel-1")
        self.assertEqual(c._legacy_api_cache.calls[0][0], "kernels_logs")
        self.assertEqual(out, "logs for owner/kernel-1")

    def test_kernels_output_routes_to_legacy_api(self):
        c = self._make_kagglesdk_client()
        out = c.kernel_output("owner/kernel-1", path="out.csv")
        self.assertEqual(c._legacy_api_cache.calls[0][:3], ("kernels_output", "owner/kernel-1", "out.csv"))
        self.assertEqual(out, "output of owner/kernel-1")

    def test_list_kernels_routes_to_legacy_api(self):
        c = self._make_kagglesdk_client()
        kernels = c.list_kernels(user="owner")
        self.assertEqual(c._legacy_api_cache.calls[0], ("kernels_list", "owner"))
        self.assertEqual(kernels[0].slug, "own/k1")
        self.assertEqual(kernels[0].title, "Kernel One")

    def test_list_kernels_uses_legacy_api_on_kaggle_sdk(self):
        c = _make_client("kaggle")
        kernels = c.list_kernels()
        self.assertEqual(c._legacy_api.calls[0], ("kernels_list", None))
        self.assertEqual(kernels[0].slug, "own/k1")

    def test_competition_topics_routes_to_legacy_api(self):
        c = self._make_kagglesdk_client()
        topics = c.competition_topics("rsna-knee")
        self.assertEqual(c._legacy_api_cache.calls[0], ("competition_list_topics", "rsna-knee"))
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0].id, 7)
        self.assertEqual(topics[0].title, "Topic")
        self.assertEqual(topics[0].comment_count, 3)

    def test_competition_topics_uses_legacy_api_on_kaggle_sdk(self):
        c = _make_client("kaggle")
        topics = c.competition_topics("rsna-knee")
        self.assertEqual(c._legacy_api.calls[0], ("competition_list_topics", "rsna-knee"))
        self.assertEqual(topics[0].id, 7)


class ConstantFieldsRegressionTests(unittest.TestCase):
    """``KaggleApi.competition_topic_fields`` is a class-level CONSTANT (a
    field-projection list in kaggle >= 2.x), not a method. The old
    attribute-probing loop found it via ``hasattr`` and called it, raising
    ``TypeError: 'list' object is not callable`` — on BOTH SDKs. The facade
    must never treat ``*_fields`` constants as callables.
    """

    def _make_client_with_constant(self):
        c = _make_client("kagglesdk")
        legacy = _FakeLegacyApi()
        # Mirror the real KaggleApi: constant list field-projection attribute.
        legacy.competition_topic_fields = ["id", "title", "authorName"]
        c._legacy_api_cache = legacy
        return c

    def test_constant_fields_not_treated_as_callable(self):
        c = self._make_client_with_constant()
        # Old code: hasattr(legacy, "competition_topic_fields") -> True -> call
        # -> TypeError. Fixed code routes to competition_list_topics instead.
        topics = c.competition_topics("comp-x")
        self.assertEqual(topics[0].id, 7)
        self.assertEqual(c._legacy_api_cache.calls[0], ("competition_list_topics", "comp-x"))


# --- Imports from tests/test_kaggle_competition_discovery.py ---
import unittest
from unittest.mock import MagicMock, patch
# --- Tests from tests/test_kaggle_competition_discovery.py ---
"""Unit tests for Kaggle competition discovery, container indexing, and compat stubs."""




class TestNormalisedObjectContainersAndAliases(unittest.TestCase):
    def test_container_indexing_and_iteration(self):
        comp1 = SimpleNamespace(title="Comp 1", ref="comp-1")
        comp2 = SimpleNamespace(title="Comp 2", ref="comp-2")
        raw = SimpleNamespace(competitions=[comp1, comp2])
        norm = _NormalisedObject(raw, "kagglesdk")

        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0].title, "Comp 1")
        self.assertEqual(norm[1].title, "Comp 2")

        titles = [item.title for item in norm]
        self.assertEqual(titles, ["Comp 1", "Comp 2"])

    def test_aliases_size_total_bytes_and_replies(self):
        raw = {
            "totalBytes": 1024,
            "totalReplies": 15,
            "teamId": 42,
            "teamName": "WinningTeam",
            "isKernelsSubmissionsOnly": True,
        }
        norm = _NormalisedObject(raw, "kagglesdk")
        self.assertEqual(norm.total_bytes, 1024)
        self.assertEqual(norm.size, 1024)
        self.assertEqual(norm.total_replies, 15)
        self.assertEqual(norm.comment_count, 15)
        self.assertEqual(norm.team_id, 42)
        self.assertEqual(norm.team_name, "WinningTeam")
        self.assertTrue(norm.is_code_competition)


class TestKaggleClientCompetitionDiscovery(unittest.TestCase):
    def test_get_competition_extracts_metadata(self):
        client = KaggleClient()
        fake_comp = SimpleNamespace(
            id=123,
            ref="https://www.kaggle.com/competitions/test-comp",
            title="Test Challenge",
            url="https://www.kaggle.com/competitions/test-comp",
            description="A test competition",
            deadline="2026-11-01 23:59:00",
            category="Featured",
            reward="$50,000",
            teamCount=500,
            userRank=12,
            userHasEntered=True,
            maxDailySubmissions=5,
            evaluationMetric="Log Loss",
            isKernelsSubmissionsOnly=True,
            submissionsDisabled=False,
        )
        fake_resp = SimpleNamespace(competitions=[fake_comp])

        mock_api = MagicMock()
        mock_api.competitions_list.return_value = fake_resp

        with patch.object(client, "_legacy_api_instance", return_value=mock_api):
            info = client.get_competition("test-comp")

        self.assertIsNotNone(info)
        self.assertEqual(info["ref"], "test-comp")
        self.assertEqual(info["title"], "Test Challenge")
        self.assertEqual(info["deadline"], "2026-11-01 23:59:00")
        self.assertEqual(info["evaluation_metric"], "Log Loss")
        self.assertTrue(info["is_code_competition"])
        self.assertTrue(info["is_ongoing"])
        self.assertEqual(info["user_rank"], 12)

    def test_list_competitions_wraps_objects(self):
        client = KaggleClient()
        fake_comp = SimpleNamespace(id=1, title="Alpha Comp")
        fake_resp = SimpleNamespace(competitions=[fake_comp])

        mock_api = MagicMock()
        mock_api.competitions_list.return_value = fake_resp

        with patch.object(client, "_legacy_api_instance", return_value=mock_api):
            comps = client.list_competitions(search="alpha")

        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0].title, "Alpha Comp")

    def test_competition_aliases(self):
        client = KaggleClient()
        self.assertEqual(client.competition_submissions, client.list_submissions)
        self.assertEqual(client.competition_view, client.get_competition)
        self.assertEqual(client.competitions_list, client.list_competitions)
        self.assertEqual(client.kernel_status, client.kernels_status)
        self.assertEqual(client.kernel_logs, client.kernels_logs)


class TestCompetitionCliAndCompatStubs(unittest.TestCase):
    def test_competition_cli_info(self):
        from harness.competition_cli import main

        fake_info = {
            "ref": "test-comp",
            "title": "Test Comp",
            "deadline": "2026-12-01",
            "is_ongoing": True,
        }
        with patch("harness.competition_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_competition.return_value = fake_info
            mock_cls.return_value = mock_inst

            ret = main(["info", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.get_competition.assert_called_once_with("test-comp")

    def test_submission_cli_info(self):
        from harness.submission_dispatch import main

        fake_info = {
            "ref": "test-comp",
            "title": "Test Comp",
            "deadline": "2026-12-01",
        }
        with patch("harness.kaggle_compat.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_competition.return_value = fake_info
            mock_cls.return_value = mock_inst

            ret = main(["info", "test-comp"])
            self.assertEqual(ret, 0)

    def test_compat_stubs_importable(self):
        from harness.agent_bridge import typed_actions
        from harness.endgame_submission import SubmissionValidator

        self.assertIsInstance(typed_actions, list)
        self.assertIn("create_experiment", typed_actions)
        self.assertIsNotNone(SubmissionValidator)


# --- Imports from tests/test_kaggle_discussions_and_submissions.py ---
import unittest
import datetime
from harness.endgame_submission import KaggleLeaderboardTracker
# --- Tests from tests/test_kaggle_discussions_and_submissions.py ---


class TestKaggleCompatAliasesAndWrapping(unittest.TestCase):
    def test_datetime_objects_not_wrapped(self):
        dt = datetime.datetime(2026, 10, 21, 10, 30, 0)
        norm = _NormalisedObject({"dt": dt}, "kagglesdk")
        self.assertIsInstance(norm.dt, datetime.datetime)
        self.assertEqual(type(norm.dt), datetime.datetime)

    def test_topic_and_message_aliases(self):
        # Raw topic-like dict or object
        raw_topic = {
            "id": 12345,
            "title": "Discussion Topic",
            "votes": 15,
            "author_name": "researcher1",
            "post_date": datetime.datetime(2026, 10, 20, 12, 0, 0),
            "comment_count": 8,
        }
        topic = _NormalisedObject(raw_topic, "kagglesdk")
        self.assertEqual(topic.votes, 15)
        self.assertEqual(topic.total_votes, 15)
        self.assertEqual(topic.totalVotes, 15)
        self.assertEqual(topic.score, 15)
        self.assertEqual(topic.author, "researcher1")
        self.assertEqual(topic.author_name, "researcher1")
        self.assertEqual(topic.post_date, raw_topic["post_date"])
        self.assertEqual(topic.date, raw_topic["post_date"])
        self.assertEqual(topic.created_at, raw_topic["post_date"])
        self.assertEqual(topic.comments, 8)
        self.assertEqual(topic.comment_count, 8)

    def test_message_content_aliases(self):
        raw_msg = {
            "id": 999,
            "author_name": "expert",
            "content": "Here is the key insight",
            "post_date": datetime.datetime(2026, 10, 20, 14, 0, 0),
            "votes": 5,
        }
        msg = _NormalisedObject(raw_msg, "kagglesdk")
        self.assertEqual(msg.author, "expert")
        self.assertEqual(msg.author_name, "expert")
        self.assertEqual(msg.content, "Here is the key insight")
        self.assertEqual(msg.message, "Here is the key insight")
        self.assertEqual(msg.body, "Here is the key insight")
        self.assertEqual(msg.date, raw_msg["post_date"])
        self.assertEqual(msg.created_at, raw_msg["post_date"])
        self.assertEqual(msg.votes, 5)
        self.assertEqual(msg.total_votes, 5)


class TestSubmissionListSubcommand(unittest.TestCase):
    def test_tracker_list_submissions(self):
        mock_client = MagicMock()
        mock_sub = MagicMock()
        mock_sub.ref = 123456
        mock_sub.status = "COMPLETE"
        mock_sub.public_score = "0.950"
        mock_sub.private_score = "0.940"
        mock_sub.date = datetime.datetime(2026, 10, 21, 8, 0, 0)
        mock_sub.description = "Test sub"
        mock_sub.file_name = "sub.csv"
        mock_sub.error_description = ""
        mock_client.list_submissions.return_value = [mock_sub]

        tracker = KaggleLeaderboardTracker(client=mock_client)
        subs = tracker.list_submissions("test-comp", limit=10)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["ref"], 123456)
        self.assertEqual(subs[0]["status"], "COMPLETE")
        self.assertEqual(subs[0]["public_score"], "0.950")
        self.assertEqual(subs[0]["public_score_f"], 0.950)
        self.assertEqual(subs[0]["description"], "Test sub")

    def test_submission_dict_metadata_parity(self):
        mock_client = MagicMock()
        mock_sub = MagicMock()
        mock_sub.ref = 57510673
        mock_sub.status = "ERROR"
        mock_sub.public_score = None
        mock_sub.private_score = None
        mock_sub.date = datetime.datetime(2026, 10, 21, 12, 0, 0)
        mock_sub.description = "V7 Hearth Idle Timeout"
        mock_sub.file_name = "submission.csv"
        mock_sub.error_description = "Hearth Gateway connection timeout after 15m idle"
        mock_sub.submitted_by = "test_user"
        mock_sub.team_name = "AlphaTeam"
        mock_sub.url = "https://www.kaggle.com/code/test_user/hearth-v7?scriptVersionId=123"

        tracker = KaggleLeaderboardTracker(client=mock_client)
        info = tracker._submission_dict(mock_sub)
        self.assertEqual(info["ref"], 57510673)
        self.assertEqual(info["submitted_by"], "test_user")
        self.assertEqual(info["team_name"], "AlphaTeam")
        self.assertEqual(info["url"], "https://www.kaggle.com/code/test_user/hearth-v7?scriptVersionId=123")
        self.assertIn("Hearth Gateway", info["error_description"])

    def test_tracker_wait_for_scores_multi(self):
        mock_client = MagicMock()
        sub1 = MagicMock()
        sub1.ref = 111
        sub1.status = "COMPLETE"
        sub1.public_score = "0.85"
        sub1.private_score = None
        sub1.date = None
        sub1.description = "Sub 1"
        sub1.file_name = "sub1.csv"
        sub1.error_description = ""
        sub1.submitted_by = "alice"
        sub1.team_name = "team"
        sub1.url = ""

        sub2 = MagicMock()
        sub2.ref = 222
        sub2.status = "COMPLETE"
        sub2.public_score = "0.88"
        sub2.private_score = None
        sub2.date = None
        sub2.description = "Sub 2"
        sub2.file_name = "sub2.csv"
        sub2.error_description = ""
        sub2.submitted_by = "bob"
        sub2.team_name = "team"
        sub2.url = ""

        mock_client.list_submissions.return_value = [sub1, sub2]
        tracker = KaggleLeaderboardTracker(client=mock_client)
        res = tracker.wait_for_scores("test-comp", [111, 222], timeout_s=10.0, poll_interval_s=0.01)

        self.assertTrue(res["all_final"])
        self.assertEqual(res["count"], 2)
        self.assertEqual(len(res["submissions"]), 2)
        scores = {s["ref"]: s["public_score_f"] for s in res["submissions"]}
        self.assertEqual(scores[111], 0.85)
        self.assertEqual(scores[222], 0.88)




class TestDiscussionCLI(unittest.TestCase):
    def test_discussion_cli_list(self):
        from unittest.mock import patch
        from harness.discussion_cli import main

        mock_topic = MagicMock()
        mock_topic.id = 101
        mock_topic.title = "Important Topic"
        mock_topic.author = "alice"
        mock_topic.votes = 10
        mock_topic.comments = 4
        mock_topic.date = "2026-10-21"
        mock_topic.topic_url = "/discussion/101"

        with patch("harness.discussion_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.competition_topics.return_value = [mock_topic]
            mock_cls.return_value = mock_inst
            ret = main(["list", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.competition_topics.assert_called_once()

    def test_discussion_cli_view(self):
        from unittest.mock import patch
        from harness.discussion_cli import main

        mock_topic_payload = {
            "id": 101,
            "title": "Topic Title",
            "author": "bob",
            "date": "2026-10-21",
            "votes": 3,
            "content": "Insightful response",
            "comments_count": 0,
            "comments": [],
        }

        with patch("harness.discussion_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_topic.return_value = mock_topic_payload
            mock_cls.return_value = mock_inst
            ret = main(["view", "101", "--competition", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.get_topic.assert_called_once_with(101, competition="test-comp")


class TestSubmissionDispatchCLI(unittest.TestCase):
    def test_submission_dispatch_list(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.list_submissions.return_value = [{"ref": 111, "status": "COMPLETE", "public_score": "0.9"}]
            mock_cls.return_value = mock_inst
            ret = main(["list", "test-comp", "--limit", "5"])
            self.assertEqual(ret, 0)
            mock_inst.list_submissions.assert_called_once_with("test-comp", limit=5)

    def test_submission_dispatch_watch_single(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.wait_for_score.return_value = {"ref": "111", "status": "COMPLETE", "final": True}
            mock_cls.return_value = mock_inst
            ret = main(["watch", "test-comp", "111"])
            self.assertEqual(ret, 0)
            mock_inst.wait_for_score.assert_called_once()
            self.assertEqual(mock_inst.wait_for_score.call_args[0][0], "test-comp")
            self.assertEqual(mock_inst.wait_for_score.call_args[0][1], "111")

    def test_submission_dispatch_watch_multi_args(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.wait_for_scores.return_value = {"all_final": True, "count": 2, "submissions": []}
            mock_cls.return_value = mock_inst
            ret = main(["watch", "test-comp", "111", "222"])
            self.assertEqual(ret, 0)
            mock_inst.wait_for_scores.assert_called_once()
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][0], "test-comp")
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][1], ["111", "222"])

    def test_submission_dispatch_watch_multi_comma(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.wait_for_scores.return_value = {"all_final": True, "count": 2, "submissions": []}
            mock_cls.return_value = mock_inst
            ret = main(["watch", "test-comp", "111,222"])
            self.assertEqual(ret, 0)
            mock_inst.wait_for_scores.assert_called_once()
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][0], "test-comp")
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][1], ["111", "222"])

    def test_submission_dispatch_quota_cli(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_quota.return_value = {
                "competition": "test-comp",
                "num_today": 2,
                "num_allowed_now": 3,
                "num_total": 33,
                "has_remaining_quota": True,
                "daily_quota": 5,
            }
            mock_cls.return_value = mock_inst
            ret = main(["quota", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.get_quota.assert_called_once_with("test-comp")

            # Also verify alias limits
            ret2 = main(["limits", "test-comp"])
            self.assertEqual(ret2, 0)
            self.assertEqual(mock_inst.get_quota.call_count, 2)

    def test_tracker_get_quota_and_caching(self):
        import tempfile
        import shutil
        import json
        from pathlib import Path
        from harness.endgame_submission import KaggleLeaderboardTracker

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            dot_ai = tmp_dir / ".ai-harness"
            dot_ai.mkdir()

            mock_client = MagicMock()
            mock_client.get_submission_limits.return_value = {
                "competition": "test-comp",
                "num_today": 1,
                "num_allowed_now": 4,
                "num_total": 10,
                "has_remaining_quota": True,
                "daily_quota": 5,
            }

            tracker = KaggleLeaderboardTracker(client=mock_client)
            quota = tracker.get_quota("test-comp", project_root=tmp_dir)

            self.assertEqual(quota["num_today"], 1)
            self.assertEqual(quota["num_allowed_now"], 4)
            self.assertTrue(quota["has_remaining_quota"])

            # Verify cached file written
            cached_file = dot_ai / "submission_limits.json"
            self.assertTrue(cached_file.is_file())
            cached_data = json.loads(cached_file.read_text(encoding="utf-8"))
            self.assertEqual(cached_data["num_allowed_now"], 4)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tracker_get_leaderboard(self):
        from types import SimpleNamespace
        from harness.endgame_submission import KaggleLeaderboardTracker

        mock_client = MagicMock()
        mock_client.get_leaderboard.return_value = [
            SimpleNamespace(team_id=101, team_name="WinnerTeam", score="0.9950", submission_date="2026-10-24 10:00:00"),
            SimpleNamespace(team_id=102, team_name="RunnerUp", score="0.9850", submission_date="2026-10-23 12:00:00"),
        ]

        tracker = KaggleLeaderboardTracker(client=mock_client)
        entries = tracker.get_leaderboard("test-comp", top=2)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["rank"], 1)
        self.assertEqual(entries[0]["team_name"], "WinnerTeam")
        self.assertEqual(entries[0]["score"], "0.9950")
        self.assertEqual(entries[1]["rank"], 2)
        self.assertEqual(entries[1]["team_name"], "RunnerUp")
        mock_client.get_leaderboard.assert_called_once_with("test-comp", page_size=2)

    def test_submission_leaderboard_cli(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_leaderboard.return_value = [
                {"rank": 1, "team_id": 1, "team_name": "T1", "score": "0.99", "submission_date": "2026-10-24"},
            ]
            mock_cls.return_value = mock_inst
            ret = main(["leaderboard", "test-comp", "--top", "5"])
            self.assertEqual(ret, 0)
            mock_inst.get_leaderboard.assert_called_once_with("test-comp", top=5)

            # Test alias lb
            ret_alias = main(["lb", "test-comp"])
            self.assertEqual(ret_alias, 0)


# --- Imports from tests/test_kaggle_inspect.py ---
import json
import unittest
from pathlib import Path
from harness import cloud_runner, kaggle_inspect
from harness.cli import build_parser
# --- Tests from tests/test_kaggle_inspect.py ---
"""Tests for the Kaggle kernel-inspect commands and the dispatch --dry-run plan."""




class TestKaggleInspectCli(unittest.TestCase):
    def test_parser_registers_kaggle(self):
        parser = build_parser()
        args = parser.parse_args(["kaggle", "status", "user/title"])
        self.assertEqual(args.command, "kaggle")
        self.assertEqual(args.args, ["status", "user/title"])

    def test_kernel_status_delegates_to_manager(self):
        fake = {"kernel_slug": "user/title", "status": "FAILED", "raw_output": "has status 'failed'"}
        with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status", return_value=fake) as m:
            result = kaggle_inspect.kernel_status("user/title")
        self.assertEqual(result["status"], "FAILED")
        m.assert_called_once_with("", "user/title")

    def test_kernel_log_returns_error_log(self):
        status_fake = {"kernel_slug": "user/title", "status": "FAILED", "raw_output": ""}
        log_fake = "Traceback ... ModuleNotFoundError: No module named 'lightgbm'"
        with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status", return_value=status_fake), \
             mock.patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_error_log", return_value=log_fake) as m:
            result = kaggle_inspect.kernel_log("user/title")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("lightgbm", result["error_log"])
        m.assert_called_once_with("user/title", max_lines=60)

    def test_kernel_watch_delegates_to_manager(self):
        wait_fake = {"kernel_slug": "user/title", "status": "COMPLETED", "success": True, "raw_output": "done"}
        with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.wait_for_kernel", return_value=wait_fake) as m:
            result = kaggle_inspect.kernel_watch("user/title", poll_interval=5.0, timeout_seconds=120.0)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["success"])
        m.assert_called_once_with(exp_id="", kernel_slug="user/title", poll_interval=5.0, timeout_seconds=120.0)

    def test_kernel_list_delegates_to_client(self):
        from types import SimpleNamespace
        k1 = SimpleNamespace(ref="user/k1", title="Kernel 1", author="alice", total_votes=10, last_run_time="2026-10-24", url=None)
        k2 = SimpleNamespace(ref="user/k2", title="Kernel 2", author="bob", votes=5, last_run_time=None, url=None)
        mock_client = mock.MagicMock()
        mock_client.list_kernels.return_value = [k1, k2]
        with mock.patch("harness.kaggle_compat.KaggleClient", return_value=mock_client):
            res = kaggle_inspect.kernel_list(competition="comp-x", sort_by="hot", page_size=10)
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["kernels"][0]["ref"], "user/k1")
        self.assertEqual(res["kernels"][0]["total_votes"], 10)
        self.assertEqual(res["kernels"][1]["total_votes"], 5)
        mock_client.list_kernels.assert_called_once_with(user=None, page_size=10, sort_by="hotness", competition="comp-x")

    def test_cli_watch_and_list_parsers(self):
        parser = build_parser()
        args_w = parser.parse_args(["kaggle", "watch", "user/title", "--poll", "10", "--timeout", "500"])
        self.assertEqual(args_w.args, ["watch", "user/title", "--poll", "10", "--timeout", "500"])

        args_l = parser.parse_args(["kaggle", "list", "--competition", "my-comp", "--sort-by", "votes"])
        self.assertEqual(args_l.args, ["list", "--competition", "my-comp", "--sort-by", "votes"])


class TestDryRunPlan(unittest.TestCase):
    def test_dry_run_plan_captures_metadata_and_bundle(self):
        stage = Path(__file__).parent / "tmp_dryrun_stage"
        stage.mkdir(exist_ok=True)
        (stage / "kernel-metadata.json").write_text(json.dumps({"id": "user/title", "enable_internet": "true"}), encoding="utf-8")
        (stage / "code" / "train.py").parent.mkdir(parents=True, exist_ok=True)
        (stage / "code" / "train.py").write_text("print('hi')", encoding="utf-8")
        context = {
            "exp_id": "exp001",
            "task_id": "task-x",
            "config_rel": "configs/exp001.yaml",
            "config_sha256": "abc123",
            "task": {"runtime": {"enable_internet": True, "kaggle_bundle_mode": "dataset"}},
        }
        runtime = {"core": {"version": "0.2.0"}, "plugin": {"id": "default"}, "requirements": ["lightgbm>=4"]}
        plan = cloud_runner._kaggle_dry_run_plan(context, mock.Mock(root=stage), "exp001", "user/ds", stage, runtime)
        self.assertEqual(plan["exp_id"], "exp001")
        self.assertEqual(plan["dataset_slug"], "user/ds")
        self.assertTrue(plan["enable_internet"])
        self.assertEqual(plan["bundle_mode"], "dataset")
        self.assertEqual(plan["kernel_metadata"]["id"], "user/title")
        self.assertIn("code/train.py", plan["bundle_files"])
        self.assertTrue(any("kaggle kernels push" in s for s in plan["would_run"]))
        # cleanup
        import shutil
        shutil.rmtree(stage, ignore_errors=True)


# --- Imports from tests/test_kaggle_leaderboard_tracker.py ---
import unittest
# --- Tests from tests/test_kaggle_leaderboard_tracker.py ---
"""Tests for the ``KaggleLeaderboardTracker`` (Gap A: dispatch + polling).

All network calls are mocked with a lightweight in-memory client, so the suite
runs offline and fast. Covers dispatch (ref + inputs), get_submission (found /
not-found), wait_for_score (PENDING->COMPLETE scored, ERROR, timeout) and
latest (best scored).
"""




def _sub(ref, status, public="", private="", error=""):
    return SimpleNamespace(
        ref=ref,
        status=status,
        public_score=public,
        private_score=private,
        error_description=error,
        description="exp",
        file_name="submission.csv",
    )


def _no_sleep(_seconds):
    return None


class _MockClient:
    """In-memory client exposing the two KaggleClient methods the tracker uses."""

    def __init__(self, submissions, submit_ref=1234):
        self._subs = submissions
        self._ref = submit_ref
        self.submit_calls = []

    def submit_submission(self, file_name, message, competition, kernel=None, kernel_version=None):
        self.submit_calls.append((file_name, message, competition, kernel, kernel_version))
        return SimpleNamespace(ref=self._ref, message=message)

    def list_submissions(self, competition, **kwargs):
        return self._subs


class _SequenceClient:
    """Returns a different submissions list on each poll (for wait tests)."""

    def __init__(self, sequences):
        self._seq = sequences
        self._n = 0

    def list_submissions(self, competition, **kwargs):
        idx = min(self._n, len(self._seq) - 1)
        self._n += 1
        return self._seq[idx]


class KaggleLeaderboardTrackerTests(unittest.TestCase):
    def test_dispatch_requires_workspace_even_with_mock_client(self):
        client = _MockClient([], submit_ref=999)
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        with self.assertRaisesRegex(ValueError, "WORKSPACE_REQUIRED"):
            tracker.dispatch("comp", "sub.csv", "Exp 1", project_root="missing-workspace")
        self.assertEqual(client.submit_calls, [])

    def test_dispatch_code_requires_a_verified_workspace(self):
        client = _MockClient([], submit_ref=55)
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        with self.assertRaisesRegex(ValueError, "WORKSPACE_REQUIRED"):
            tracker.dispatch("comp", "sub.csv", "label", kernel="own/k", kernel_version="3", project_root="missing-workspace")
        self.assertEqual(client.submit_calls, [])

    def test_get_submission_finds_ref(self):
        client = _MockClient([_sub(1234, "COMPLETE", public="0.76", private="0.77")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        info = tracker.get_submission("comp", 1234)
        self.assertTrue(info["found"])
        self.assertIn("COMPLETE", info["status"])
        self.assertAlmostEqual(info["public_score_f"], 0.76)
        self.assertAlmostEqual(info["private_score_f"], 0.77)
        self.assertEqual(info["file_name"], "submission.csv")

    def test_get_submission_matches_enum_style_status(self):
        # Real kaggle enums stringify to e.g. "SUBMISSIONSTATUS.COMPLETE".
        client = _MockClient([_sub(1, "SUBMISSIONSTATUS.COMPLETE", public="0.5")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        info = tracker.get_submission("comp", 1)
        self.assertTrue(info["found"])
        self.assertIn("COMPLETE", info["status"])

    def test_get_submission_not_found(self):
        client = _MockClient([_sub(1, "COMPLETE", public="0.5")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        info = tracker.get_submission("comp", 42)
        self.assertFalse(info["found"])
        self.assertEqual(info["status"], "NOT_FOUND")

    def test_wait_for_score_pending_to_complete(self):
        seq = [[_sub(9, "PENDING")], [_sub(9, "COMPLETE", public="0.80")]]
        tracker = KaggleLeaderboardTracker(client=_SequenceClient(seq), sleep_fn=_no_sleep)
        result = tracker.wait_for_score("comp", 9, timeout_s=100, poll_interval_s=1)
        self.assertEqual(result["outcome"], "scored")
        self.assertTrue(result["final"])
        self.assertEqual(result["attempts"], 2)
        self.assertAlmostEqual(result["public_score_f"], 0.80)

    def test_wait_for_score_error(self):
        client = _MockClient([_sub(9, "ERROR", error="row count mismatch")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        result = tracker.wait_for_score("comp", 9, timeout_s=100, poll_interval_s=1)
        self.assertEqual(result["outcome"], "error")
        self.assertTrue(result["final"])
        self.assertEqual(result["error_description"], "row count mismatch")

    def test_wait_for_score_timeout(self):
        client = _MockClient([_sub(9, "PENDING")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        result = tracker.wait_for_score("comp", 9, timeout_s=0, poll_interval_s=1)
        self.assertEqual(result["outcome"], "timeout")
        self.assertFalse(result["final"])
        self.assertEqual(result["attempts"], 1)

    def test_wait_for_score_grows_interval_within_cap(self):
        # PENDING forever; assert the adaptive interval is capped (no crash).
        client = _MockClient([_sub(9, "PENDING")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        # A small positive timeout lets several polls run before timing out.
        result = tracker.wait_for_score(
            "comp", 9, timeout_s=0.05, poll_interval_s=0.01, max_poll_interval_s=0.02
        )
        self.assertEqual(result["outcome"], "timeout")
        self.assertGreater(result["attempts"], 1)

    def test_latest_picks_best_scored(self):
        client = _MockClient([
            _sub(1, "COMPLETE", public="0.70"),
            _sub(2, "PENDING"),
            _sub(3, "COMPLETE", public="0.95"),
        ])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        result = tracker.latest("comp")
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["scored"], 2)
        self.assertEqual(result["best"]["ref"], 3)
        self.assertAlmostEqual(result["best"]["public_score_f"], 0.95)

    def test_latest_no_scored(self):
        client = _MockClient([_sub(1, "PENDING")])
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        result = tracker.latest("comp")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["scored"], 0)
        self.assertIsNone(result["best"])


# --- Imports from tests/test_kaggle_preflight.py ---
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import harness
from harness.cloud_runner import _warn_if_framework_checkout_dirty
from harness.templates.kaggle_template import KaggleKernelManager
# --- Tests from tests/test_kaggle_preflight.py ---
"""Regression: Kaggle CLI preflight + wait_for_dataset error surface.

Live origin: job_20261019_095919 (arc-prize-2026) died in
``wait_for_dataset`` with an empty error surface — the CLI failed
silently (no stdout/stderr) and the old code raised immediately with
``Kaggle dataset status failed: `` and nothing after the colon. Empty CLI
failures are now retried like other transient errors, and the raised
message always carries exit code, attempt count, and a manual check
command. ``ensure_cli_ready`` fails fast (with install guidance) when the
CLI is missing or broken, before any upload or push.
"""





class WaitForKeywordSurfaceTests(unittest.TestCase):
    def setUp(self):
        self._saved_cache = KaggleKernelManager._cli_version_cache
        KaggleKernelManager._cli_version_cache = None

    def tearDown(self):
        KaggleKernelManager._cli_version_cache = self._saved_cache

    def test_empty_failure_is_retried_then_surfaces_with_detail(self):
        failed = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[failed, failed]) as mock_run:
            with self.assertRaisesRegex(
                RuntimeError,
                r"Kaggle dataset status failed \(exit 1\) after 2 attempt\(s\): <no CLI output>",
            ):
                KaggleKernelManager.wait_for_dataset(
                    "owner/test-dataset",
                    poll_interval=0.01,
                    timeout_seconds=30.0,
                    max_consecutive_errors=2,
                )
        self.assertEqual(mock_run.call_count, 2)

    def test_empty_failure_then_ready_succeeds(self):
        failed = MagicMock(returncode=1, stdout="", stderr="")
        ready = MagicMock(returncode=0, stdout='"READY"', stderr="")
        with patch("subprocess.run", side_effect=[failed, ready]) as mock_run:
            KaggleKernelManager.wait_for_dataset(
                "owner/test-dataset", poll_interval=0.01, timeout_seconds=30.0
            )
        self.assertEqual(mock_run.call_count, 2)

    def test_timeout_surfaces_last_error_and_manual_command(self):
        processing = MagicMock(returncode=0, stdout='"PROCESSING"', stderr="")
        with patch("subprocess.run", return_value=processing):
            with self.assertRaisesRegex(
                TimeoutError, r"Check manually: kaggle datasets status owner/test-dataset"
            ):
                KaggleKernelManager.wait_for_dataset(
                    "owner/test-dataset",
                    poll_interval=0.01,
                    timeout_seconds=0.05,
                )


class EnsureCliReadyTests(unittest.TestCase):
    def setUp(self):
        self._saved_cache = KaggleKernelManager._cli_version_cache
        KaggleKernelManager._cli_version_cache = None

    def tearDown(self):
        KaggleKernelManager._cli_version_cache = self._saved_cache

    def test_missing_cli_raises_with_install_guidance(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("kaggle")):
            with self.assertRaisesRegex(
                RuntimeError, r"Kaggle CLI not found on PATH"
            ):
                KaggleKernelManager.ensure_cli_ready()

    def test_broken_cli_raises_with_output(self):
        broken = MagicMock(returncode=2, stdout="", stderr="internal cli crash")
        with patch("subprocess.run", return_value=broken):
            with self.assertRaisesRegex(
                RuntimeError, r"failed on startup \(exit 2\): internal cli crash"
            ):
                KaggleKernelManager.ensure_cli_ready()

    def test_version_is_reported_and_cached(self):
        good = MagicMock(returncode=0, stdout="Kaggle CLI 2.2.4", stderr="")
        with patch("subprocess.run", return_value=good) as mock_run:
            version = KaggleKernelManager.ensure_cli_ready()
            self.assertEqual(version, "Kaggle CLI 2.2.4")
            KaggleKernelManager.ensure_cli_ready()  # served from cache
        # version probe + one auth probe on the first call only
        self.assertEqual(mock_run.call_count, 2)


class CliAuthProbeTests(unittest.TestCase):
    """G4: advisory CLI auth probe in ``ensure_cli_ready``.

    Live origin: an unauthenticated CLI (no ``~/.kaggle/kaggle.json``)
    failed only mid-push with "Authentication required to call the Kaggle
    API" (arc-prize-2026 conversation, exp005 push stage). The preflight
    now detects it up front and warns — advisory only, never fatal, because
    the SDK-based dispatch path may be authenticated while the CLI is not.
    """

    def setUp(self):
        self._saved_version = KaggleKernelManager._cli_version_cache
        KaggleKernelManager._cli_version_cache = None
        self.stderr = io.StringIO()

    def tearDown(self):
        KaggleKernelManager._cli_version_cache = self._saved_version

    @staticmethod
    def _good_version():
        return MagicMock(returncode=0, stdout="Kaggle CLI 2.2.4", stderr="")

    def test_unauthenticated_cli_warns_without_raising(self):
        auth_fail = MagicMock(
            returncode=1,
            stdout="",
            stderr="Authentication required to call the Kaggle API",
        )
        with patch(
            "subprocess.run", side_effect=[self._good_version(), auth_fail]
        ) as mock_run:
            with redirect_stderr(self.stderr):
                version = KaggleKernelManager.ensure_cli_ready()
        self.assertEqual(version, "Kaggle CLI 2.2.4")
        self.assertIn(
            "WARNING: Kaggle CLI reports no usable", self.stderr.getvalue()
        )
        self.assertIn("kaggle auth login", self.stderr.getvalue())
        self.assertEqual(mock_run.call_count, 2)

    def test_authenticated_cli_prints_no_auth_warning(self):
        auth_ok = MagicMock(returncode=0, stdout='{"kernels": []}', stderr="")
        with patch(
            "subprocess.run", side_effect=[self._good_version(), auth_ok]
        ):
            with redirect_stderr(self.stderr):
                KaggleKernelManager.ensure_cli_ready()
        self.assertNotIn("WARNING", self.stderr.getvalue())

    def test_auth_probe_timeout_is_skipped_safely(self):
        with patch(
            "subprocess.run",
            side_effect=[
                self._good_version(),
                subprocess.TimeoutExpired("kaggle", 60),
            ],
        ):
            with redirect_stderr(self.stderr):
                version = KaggleKernelManager.ensure_cli_ready()
        self.assertEqual(version, "Kaggle CLI 2.2.4")
        self.assertNotIn("WARNING", self.stderr.getvalue())

    def test_non_auth_probe_failure_does_not_warn(self):
        other_fail = MagicMock(returncode=1, stdout="", stderr="quota exceeded")
        with patch(
            "subprocess.run", side_effect=[self._good_version(), other_fail]
        ):
            with redirect_stderr(self.stderr):
                KaggleKernelManager.ensure_cli_ready()
        self.assertNotIn("WARNING", self.stderr.getvalue())

    def test_auth_probe_can_be_disabled(self):
        with patch("subprocess.run", return_value=self._good_version()) as mock_run:
            KaggleKernelManager.ensure_cli_ready(check_auth=False)
        self.assertEqual(mock_run.call_count, 1)

    def test_auth_probe_runs_once_per_process(self):
        auth_fail = MagicMock(
            returncode=1, stdout="", stderr="Authentication required"
        )
        with patch(
            "subprocess.run", side_effect=[self._good_version(), auth_fail]
        ) as mock_run:
            with redirect_stderr(self.stderr):
                KaggleKernelManager.ensure_cli_ready()
                KaggleKernelManager.ensure_cli_ready()  # served from cache
        self.assertEqual(mock_run.call_count, 2)  # no second auth probe


class DirtyCheckoutWarningTests(unittest.TestCase):
    """Dispatch must warn when the canonical framework checkout is dirty.

    Live origin: an agent patched the canonical checkout in place during a
    session; the remote bundle then shipped uncommitted edits with no
    visible signal. The dispatch preflight now surfaces this.
    """

    def _run_warning(self, tmp, porcelain_lines, rc=0):
        repo = Path(tmp)
        (repo / ".git").mkdir()
        (repo / "harness").mkdir()
        (repo / "harness" / "__init__.py").write_text("", encoding="utf-8")
        fake_file = str(repo / "harness" / "__init__.py")
        proc = MagicMock(
            returncode=rc,
            stdout="\n".join(porcelain_lines) + ("\n" if porcelain_lines else ""),
            stderr="",
        )
        captured = io.StringIO()
        with patch.object(harness, "__file__", fake_file), \
             patch("subprocess.run", return_value=proc) as mock_run, \
             redirect_stdout(captured):
            _warn_if_framework_checkout_dirty()
        return captured.getvalue(), mock_run

    def test_clean_checkout_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run_warning(tmp, [])
        self.assertNotIn("WARNING", out)

    def test_dirty_checkout_warns_with_change_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run_warning(tmp, [" M harness/runner.py", "?? notes.md"])
        self.assertIn("WARNING: framework checkout is dirty", out)
        self.assertIn("2 uncommitted change(s)", out)

    def test_non_git_directory_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "harness").mkdir()
            (repo / "harness" / "__init__.py").write_text("", encoding="utf-8")
            captured = io.StringIO()
            with patch.object(harness, "__file__", str(repo / "harness" / "__init__.py")), \
                 patch("subprocess.run") as mock_run, redirect_stdout(captured):
                _warn_if_framework_checkout_dirty()
            mock_run.assert_not_called()
        self.assertEqual(captured.getvalue(), "")

    def test_git_failure_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run_warning(tmp, [], rc=128)
        self.assertNotIn("WARNING", out)


# --- Imports from tests/test_kernel_dispatch.py ---
import contextlib
import unittest
from pathlib import Path
from harness.kernel_dispatch import build_kernel_metadata, dispatch_kernel
# --- Tests from tests/test_kernel_dispatch.py ---
"""Tests for ``harness.kernel_dispatch`` (Gap B: kernel wrap + dispatch).

Covers build_kernel_metadata (schema + defaults), dispatch_kernel (staging +
push + warnings + missing notebook), the main() CLI entry point (arg
parsing, JSON-on-stdout contract, exit codes), and top-level CLI
registration of ``dispatch-kernel`` and ``submission``. All push calls are
mocked, so the suite runs offline.
"""




class _MockKernelClient:
    """In-memory client exposing ``kernels_push`` (the only method used)."""

    def __init__(self, warnings=None):
        self._warnings = warnings or {}
        self.push_calls = []

    def kernels_push(self, folder, timeout=None, acc=None):
        self.push_calls.append({"folder": folder, "timeout": timeout, "acc": acc})
        resp = SimpleNamespace(
            ref=777,
            url="https://www.kaggle.com/kernels/own/kernel",
            kernel_id=555,
            version_number=2,
            error=None,
            invalid_dataset_sources=None,
            invalid_competition_sources=None,
            invalid_kernel_sources=None,
            invalid_model_sources=None,
        )
        for key, value in self._warnings.items():
            setattr(resp, key, value)
        return resp


class BuildKernelMetadataTests(unittest.TestCase):
    def test_default_schema(self):
        meta = build_kernel_metadata("own/kernel", "My Kernel")
        self.assertEqual(meta["id"], "own/kernel")
        self.assertEqual(meta["title"], "My Kernel")
        self.assertEqual(meta["code_file"], "code.py")
        self.assertEqual(meta["language"], "python")
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertTrue(meta["is_private"])
        self.assertFalse(meta["enable_internet"])
        self.assertFalse(meta["enable_gpu"])
        self.assertFalse(meta["enable_tpu"])
        self.assertEqual(meta["keywords"], [])
        self.assertEqual(meta["dataset_sources"], [])
        self.assertEqual(meta["competition_sources"], [])
        self.assertEqual(meta["kernel_sources"], [])
        self.assertEqual(meta["model_sources"], [])
        self.assertNotIn("machine_shape", meta)

    def test_machine_shape_included_when_set(self):
        meta = build_kernel_metadata(
            "own/k", "T", machine_shape="KaggleGPU", enable_internet=True
        )
        self.assertEqual(meta["machine_shape"], "KaggleGPU")
        self.assertTrue(meta["enable_internet"])

    def test_sources_and_keywords(self):
        meta = build_kernel_metadata(
            "own/k",
            "T",
            dataset_sources=["own/data"],
            competition_sources=["comp"],
            kernel_sources=["own/base"],
            model_sources=["own/model"],
            keywords=["ml", "ensemble"],
        )
        self.assertEqual(meta["dataset_sources"], ["own/data"])
        self.assertEqual(meta["competition_sources"], ["comp"])
        self.assertEqual(meta["kernel_sources"], ["own/base"])
        self.assertEqual(meta["model_sources"], ["own/model"])
        self.assertEqual(meta["keywords"], ["ml", "ensemble"])


class DispatchKernelTests(unittest.TestCase):
    def _notebook(self, folder: Path):
        nb = folder / "train.ipynb"
        nb.write_text(json.dumps({"cells": []}))
        return nb

    def test_dispatch_stages_and_pushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            result = dispatch_kernel(
                str(nb), "own/kernel", "My Kernel", machine_shape="KaggleGPU", client=client
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["ref"], 777)
            self.assertEqual(result["kernel_id"], 555)
            self.assertEqual(result["version_number"], 2)
            self.assertEqual(len(client.push_calls), 1)
            self.assertEqual(client.push_calls[0]["acc"], "KaggleGPU")
            self.assertEqual(result["metadata"]["id"], "own/kernel")
            self.assertEqual(result["metadata"]["machine_shape"], "KaggleGPU")

    def test_dispatch_staging_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            result = dispatch_kernel(str(nb), "own/k", "T", keep_staging=True, client=client)
            staging = Path(result["staging_dir"])
            self.assertTrue(staging.exists())
            self.assertTrue((staging / "train.ipynb").exists())
            meta_file = staging / "kernel-metadata.json"
            self.assertTrue(meta_file.exists())
            meta = json.loads(meta_file.read_text())
            self.assertEqual(meta["id"], "own/k")
            self.assertEqual(meta["code_file"], "train.ipynb")

    def test_dispatch_surfaces_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient(warnings={"invalid_dataset_sources": ["bad/source"]})
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertTrue(result["ok"])
            self.assertEqual(result["invalid_dataset_sources"], ["bad/source"])

    def test_dispatch_error_field_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient(warnings={"error": "invalid slug"})
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertEqual(result.get("error"), "invalid slug")

    def test_missing_notebook_raises(self):
        with self.assertRaises(FileNotFoundError):
            dispatch_kernel("/nonexistent/train.ipynb", "own/k", "T", client=_MockKernelClient())

    def test_default_code_file_is_notebook_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = Path(tmp) / "my_model.ipynb"
            nb.write_text(json.dumps({"cells": []}))
            client = _MockKernelClient()
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertEqual(result["metadata"]["code_file"], "my_model.ipynb")

    def test_staging_cleaned_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertNotIn("staging_dir", result)

    def test_dispatch_with_watch_waits_for_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            watch_fake = {"kernel_slug": "own/k", "status": "COMPLETED", "success": True}
            with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.wait_for_kernel", return_value=watch_fake) as m:
                result = dispatch_kernel(str(nb), "own/k", "T", client=client, watch=True, poll_interval=5.0, watch_timeout=100.0)
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["watch"]["status"], "COMPLETED")
            m.assert_called_once_with(exp_id="", kernel_slug=777, poll_interval=5.0, timeout_seconds=100.0)


class KernelDispatchMainCliTests(unittest.TestCase):
    """``main()`` CLI entry point: arg parsing, JSON-on-stdout contract, exit
    codes. ``dispatch_kernel`` is mocked — no staging, client, or network.
    The JSON printed to stdout is what the agent reads back, so it is pinned.
    """

    def _run_main(self, argv, dispatch_result=None, dispatch_exc=None):
        from harness.kernel_dispatch import main

        with mock.patch("harness.kernel_dispatch.dispatch_kernel") as dk:
            if dispatch_exc is not None:
                dk.side_effect = dispatch_exc
            else:
                dk.return_value = dispatch_result or {
                    "ok": True, "ref": 9, "url": "https://k/own/k",
                }
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(argv)
        return rc, json.loads(buf.getvalue()), dk

    def test_happy_path_exit_zero_and_json(self):
        rc, out, dk = self._run_main(["nb.ipynb", "-i", "owner/k", "-t", "Title"])
        self.assertEqual(rc, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["ref"], 9)
        dk.assert_called_once()

    def test_failure_exit_one_and_error_json(self):
        rc, out, _ = self._run_main(
            ["nb.ipynb", "-i", "owner/k", "-t", "T"],
            dispatch_exc=RuntimeError("boom"),
        )
        self.assertEqual(rc, 1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "boom")

    def test_all_flags_forwarded_to_dispatch(self):
        rc, _, dk = self._run_main([
            "nb.ipynb", "-i", "o/k", "-t", "T",
            "--code-file", "x.py",
            "--machine-shape", "T4",
            "--dataset", "d1", "--dataset", "d2",
            "--competition", "c1",
            "--kernel-source", "ks1",
            "--model", "m1",
            "--keyword", "kw1",
            "--enable-internet",
            "--timeout", "1200",
            "--keep-staging",
            "--watch",
            "--poll", "5.0",
            "--watch-timeout", "600.0",
        ])
        self.assertEqual(rc, 0)
        args, kwargs = dk.call_args
        self.assertEqual(args[:3], ("nb.ipynb", "o/k", "T"))
        self.assertEqual(kwargs["code_file"], "x.py")
        self.assertEqual(kwargs["machine_shape"], "T4")
        self.assertEqual(kwargs["dataset_sources"], ["d1", "d2"])
        self.assertEqual(kwargs["competition_sources"], ["c1"])
        self.assertEqual(kwargs["kernel_sources"], ["ks1"])
        self.assertEqual(kwargs["model_sources"], ["m1"])
        self.assertEqual(kwargs["keywords"], ["kw1"])
        self.assertTrue(kwargs["enable_internet"])
        self.assertEqual(kwargs["timeout"], 1200)
        self.assertTrue(kwargs["keep_staging"])
        self.assertTrue(kwargs["watch"])
        self.assertEqual(kwargs["poll_interval"], 5.0)
        self.assertEqual(kwargs["watch_timeout"], 600.0)

    def test_missing_required_args_exits_two(self):
        from harness.kernel_dispatch import main

        with mock.patch("harness.kernel_dispatch.dispatch_kernel") as dk, \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as cm:
            main(["nb.ipynb", "-t", "T"])  # --kernel-id missing
        self.assertEqual(cm.exception.code, 2)
        dk.assert_not_called()




class PushKernelDirectoryTests(unittest.TestCase):
    def test_push_kernel_directory_success(self):
        from harness.kernel_dispatch import push_kernel_directory
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            meta = {
                "id": "owner/arc-solution",
                "title": "ARC Solution",
                "code_file": "main.py",
                "machine_shape": "KaggleGPU",
            }
            (folder / "kernel-metadata.json").write_text(json.dumps(meta))
            (folder / "main.py").write_text("print('ARC')")

            client = _MockKernelClient()
            result = push_kernel_directory(folder, client=client)
            self.assertTrue(result["ok"])
            self.assertEqual(result["kernel_id"], 555)
            self.assertEqual(len(client.push_calls), 1)
            self.assertEqual(client.push_calls[0]["folder"], str(folder.resolve()))
            self.assertEqual(client.push_calls[0]["acc"], "KaggleGPU")

    def test_push_kernel_directory_missing_metadata(self):
        from harness.kernel_dispatch import push_kernel_directory
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "main.py").write_text("print('hello')")
            with self.assertRaises(FileNotFoundError):
                push_kernel_directory(folder)
class CliRegistrationTests(unittest.TestCase):
    def test_new_subcommands_registered(self):
        from harness.cli import build_parser

        parser = build_parser()
        choices = list(parser._subparsers._group_actions[0].choices.keys())
        self.assertIn("dispatch-kernel", choices)
        self.assertIn("submission", choices)
        self.assertIn("leaderboard", choices)
        self.assertIn("preflight", choices)


# --- Imports from tests/test_codabench_compat.py ---
import unittest
from pathlib import Path
import shutil
from harness.codabench_compat import CodabenchClient
from harness.codabench_cli import main as codabench_cli_main
# --- Tests from tests/test_codabench_compat.py ---



class TestCodabenchCompat(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("requests.get")
    def test_get_submission(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": 1007196,
            "status": "Finished",
            "scores": [
                {"column_key": "accuracy", "score": "0.714285", "is_primary": True},
                {"column_key": "coverage", "score": "1.0", "is_primary": False},
            ],
            "description": "Test V30",
        }
        mock_get.return_value = mock_resp

        client = CodabenchClient(session_id="dummy_session_123")
        sub = client.get_submission(1007196)
        self.assertEqual(sub["id"], 1007196)
        self.assertEqual(sub["status"], "Finished")
        self.assertEqual(sub["primary_score"], 0.714285)
        self.assertEqual(sub["description"], "Test V30")

    @patch("requests.get")
    def test_download_submission(self, mock_get):
        import io
        import zipfile

        # Prepare a mock zip in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("predictions.jsonl", '{"id": "case-1", "is_robust": true}\n')
        zip_bytes = zip_buf.getvalue()

        def side_effect(url, headers=None):
            resp = MagicMock()
            if "api/submissions/1007196/get_details/" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "logs": [{"name": "ingestion_log", "data_file": "https://example.com/log.txt"}],
                    "prediction_result": "https://example.com/predictions.zip",
                }
                resp.text = json.dumps(resp.json.return_value)
            elif "api/submissions/1007196/" in url:
                resp.status_code = 200
                resp.json.return_value = {"id": 1007196, "status": "Finished", "scores": []}
                resp.text = json.dumps(resp.json.return_value)
            elif "log.txt" in url:
                resp.status_code = 200
                resp.content = b"All models evaluated successfully."
            elif "predictions.zip" in url:
                resp.status_code = 200
                resp.content = zip_bytes
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        client = CodabenchClient(session_id="dummy_session")
        out_dir = self.tmp / "audit_1007196"
        res = client.download_submission(1007196, output_dir=out_dir)

        self.assertEqual(res["sub_id"], 1007196)

        self.assertTrue((out_dir / "meta.json").is_file())
        self.assertTrue((out_dir / "details.json").is_file())
        self.assertTrue((out_dir / "ingestion_log.txt").is_file())
        self.assertTrue((out_dir / "predictions.jsonl").is_file())
        self.assertIn("case-1", (out_dir / "predictions.jsonl").read_text())

    @patch("requests.get")
    def test_codabench_cli(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": 1007196, "status": "Finished", "scores": []}
        mock_get.return_value = mock_resp

        ret = codabench_cli_main(["status", "1007196", "--session", "test_sess"])
        self.assertEqual(ret, 0)

    def test_resolve_session_id_scratch_fallback(self):
        scratch_dir = self.tmp / "scratch"
        scratch_dir.mkdir(parents=True)
        py_file = scratch_dir / "probe.py"
        py_file.write_text('session_id = "test_scanned_token_abc1234567890"\nprint("ok")', encoding="utf-8")

        resolved = CodabenchClient._resolve_session_id(project_root=self.tmp)
        self.assertEqual(resolved, "test_scanned_token_abc1234567890")

    def test_resolve_session_id_home_dir(self):
        home_dir = self.tmp / "fake_home"
        home_dir.mkdir(parents=True)
        token_file = home_dir / ".codabench_session"
        token_file.write_text("home_token_xyz987654321", encoding="utf-8")

        with patch("pathlib.Path.home", return_value=home_dir):
            resolved = CodabenchClient._resolve_session_id(project_root=self.tmp)
            self.assertEqual(resolved, "home_token_xyz987654321")

    def test_codabench_cli_auth(self):
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmp)
            ret = codabench_cli_main(["auth", "my_secret_token_12345"])
            self.assertEqual(ret, 0)
            self.assertTrue((self.tmp / ".codabench_session").is_file())
            self.assertEqual((self.tmp / ".codabench_session").read_text(encoding="utf-8"), "my_secret_token_12345")
        finally:
            os.chdir(orig_cwd)


class TestKaggleLicenseFallback(unittest.TestCase):
    def test_kaggle_license_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "task.yaml").write_text(
                "task:\n  name: TestKaggle\n  id: test_kaggle\n  data:\n    train: train.csv\n",
                encoding="utf-8"
            )
            (root / "dataset").mkdir()
            (root / "dataset" / "train.csv").write_text("id,val\n1,10\n", encoding="utf-8")
            from harness.project import attach_project
            attach_project(project_root=root, task_config=cfg_dir / "task.yaml", force=True)

            res = KaggleKernelManager.push_dataset_to_kaggle(project_root=root)
            self.assertEqual(res["success"], False)
            self.assertIn("consent", res["error"].lower())
            self.assertNotIn("license", res["error"].lower())


if __name__ == "__main__":
    import unittest
    unittest.main()