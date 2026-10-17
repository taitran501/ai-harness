"""Tests for ``harness.submission_dispatch`` (the ``ai-harness submission`` CLI).

Covers all four subcommands (dispatch / status / watch / latest) plus the JSON
error path, with the leaderboard tracker fully mocked so the suite runs offline
(no network, no real Kaggle ref). This module was previously at 0% coverage —
the exact dispatch surface agents hit when the tracker is reached.
"""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

from harness import submission_dispatch


class _FakeTracker:
    """In-memory stand-in for ``KaggleLeaderboardTracker`` (records calls)."""

    def __init__(self):
        self.dispatch_calls = []
        self.status_calls = []
        self.watch_calls = []
        self.latest_calls = []

    def dispatch(self, competition, file, message, kernel=None, kernel_version=None):
        self.dispatch_calls.append(dict(
            competition=competition, file=file, message=message,
            kernel=kernel, kernel_version=kernel_version,
        ))
        return {"ref": 123456, "message": "submitted"}

    def get_submission(self, competition, ref):
        self.status_calls.append((competition, ref))
        return {"ref": ref, "status": "complete", "public_score": 0.9}

    def wait_for_score(self, competition, ref, timeout_s=0, poll_interval_s=0, max_poll_interval_s=0):
        self.watch_calls.append(dict(
            competition=competition, ref=ref, timeout_s=timeout_s,
            poll_interval_s=poll_interval_s, max_poll_interval_s=max_poll_interval_s,
        ))
        return {"ref": ref, "status": "complete", "public_score": 0.87, "attempts": 3}

    def latest(self, competition):
        self.latest_calls.append(competition)
        return {"ref": 999, "public_score": 0.93, "message": "best"}


def _run(argv, tracker):
    """Run ``submission_dispatch.main`` with the tracker mocked; return (code, payload)."""
    buf = io.StringIO()
    with mock.patch.object(
        submission_dispatch, "KaggleLeaderboardTracker", return_value=tracker
    ):
        with contextlib.redirect_stdout(buf):
            code = submission_dispatch.main(argv)
    return code, json.loads(buf.getvalue())


class SubmissionDispatchTests(unittest.TestCase):
    def test_dispatch_file_submits_and_reports_ref(self):
        t = _FakeTracker()
        code, payload = _run(["dispatch", "comp", "sub.csv", "-m", "Exp 22"], t)
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["action"], "dispatch")
        self.assertEqual(payload["ref"], 123456)
        self.assertEqual(t.dispatch_calls[0]["message"], "Exp 22")
        self.assertIsNone(t.dispatch_calls[0]["kernel"])

    def test_dispatch_code_competition_passes_kernel(self):
        t = _FakeTracker()
        code, payload = _run(
            ["dispatch", "comp", "out.csv", "-k", "owner/k", "--kernel-version", "3"], t
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "dispatch")
        self.assertEqual(t.dispatch_calls[0]["kernel"], "owner/k")
        self.assertEqual(t.dispatch_calls[0]["kernel_version"], "3")

    def test_status_returns_tracker_payload(self):
        t = _FakeTracker()
        code, payload = _run(["status", "comp", "57314333"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "status")
        self.assertEqual(payload["public_score"], 0.9)
        self.assertEqual(t.status_calls[0], ("comp", "57314333"))

    def test_watch_passes_poll_args(self):
        t = _FakeTracker()
        code, payload = _run(["watch", "comp", "57314333", "--timeout", "30", "--poll", "5"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "watch")
        self.assertEqual(payload["attempts"], 3)
        self.assertEqual(t.watch_calls[0]["timeout_s"], 30)
        self.assertEqual(t.watch_calls[0]["poll_interval_s"], 5)

    def test_latest_returns_best(self):
        t = _FakeTracker()
        code, payload = _run(["latest", "comp"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "latest")
        self.assertEqual(payload["ref"], 999)
        self.assertEqual(t.latest_calls[0], "comp")

    def test_error_path_returns_nonzero_and_json_error(self):
        t = _FakeTracker()

        def _boom(*args, **kwargs):
            raise RuntimeError("no such ref")

        t.get_submission = _boom
        code, payload = _run(["status", "comp", "0"], t)
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["action"], "status")
        self.assertIn("no such ref", payload["error"])


if __name__ == "__main__":
    unittest.main()
