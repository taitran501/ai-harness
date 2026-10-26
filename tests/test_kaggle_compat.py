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
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
