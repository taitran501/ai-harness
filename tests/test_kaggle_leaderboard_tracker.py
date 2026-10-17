"""Tests for the ``KaggleLeaderboardTracker`` (Gap A: dispatch + polling).

All network calls are mocked with a lightweight in-memory client, so the suite
runs offline and fast. Covers dispatch (ref + inputs), get_submission (found /
not-found), wait_for_score (PENDING->COMPLETE scored, ERROR, timeout) and
latest (best scored).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from harness.endgame_submission import KaggleLeaderboardTracker


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
    def test_dispatch_returns_ref_and_inputs(self):
        client = _MockClient([], submit_ref=999)
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        result = tracker.dispatch(
            "comp", "sub.csv", "Exp 1", kernel="own/k", kernel_version="3"
        )
        self.assertEqual(result["ref"], 999)
        self.assertEqual(result["competition"], "comp")
        self.assertEqual(result["file_name"], "sub.csv")
        self.assertEqual(result["kernel"], "own/k")
        self.assertEqual(result["kernel_version"], "3")
        self.assertEqual(
            client.submit_calls, [("sub.csv", "Exp 1", "comp", "own/k", "3")]
        )

    def test_dispatch_standard(self):
        client = _MockClient([], submit_ref=55)
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=_no_sleep)
        result = tracker.dispatch("comp", "sub.csv", "label")
        self.assertEqual(result["ref"], 55)
        self.assertIsNone(result["kernel"])
        self.assertEqual(client.submit_calls, [("sub.csv", "label", "comp", None, None)])

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


if __name__ == "__main__":
    unittest.main()
