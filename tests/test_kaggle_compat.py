"""Tests for ``harness.kaggle_compat`` — the Kaggle SDK compatibility facade.

This module runs inside the *remote* Kaggle kernel (via the generated
``kaggle_template``), where a normalisation bug means a full kernel loss with
no local traceback. The pure, network-free surface is exercised here:

- ``detect_sdk`` precedence (kagglesdk > kaggle > none)
- ``_NormalisedObject`` attribute normalisation (direct / legacy alias /
  reverse alias / pydantic ``model_dump`` / nested wrap / miss)
- ``_is_transient_error`` classification
- ``retry_with_backoff`` success / retry / permanent-fail / exhaustion

Client list_* methods are not unit-tested (they require a live or mocked
SDK session); their contracts are covered by ``test_kaggle_runtime``.
"""
from __future__ import annotations

import unittest
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

    def test_legacy_api_instance_returns_configured_api_for_kaggle(self):
        c = _make_client("kaggle")
        self.assertIs(c._legacy_api_instance(), c._legacy_api)

    def test_get_resolves_registered_method(self):
        c = _make_client("kaggle")
        c.get("kernels_push", "f")
        self.assertEqual(c._legacy_api.calls[-1][0], "kernels_push")


if __name__ == "__main__":
    unittest.main()
