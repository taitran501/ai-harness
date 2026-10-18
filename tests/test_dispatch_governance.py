"""Dispatch governance (PR-C): audit trail + duplicate warning.

The reviewer finding (C2): ``ai-harness submission dispatch`` and
``ai-harness dispatch-kernel`` drove the Kaggle API in-process with no
session, no ledger, and no trace — a double-submission (or a pushed
kernel that was never tracked) left no evidence behind, unlike the
governed Bridge ``prepare_submission`` path.

These tests pin the additive governance contract:

1. Every harness-native dispatch appends one JSONL entry to
   ``<workspace>/.ai-harness/dispatch_log.jsonl`` (append-only, outside
   any sealed bundle) with the *content hash* of what was sent.
2. Re-dispatching an identical file to the same competition (or the same
   notebook to the same kernel id) raises a duplicate warning — but does
   NOT block (v1 is advisory, never a hard block).
3. No project context (no explicit root, no attached workspace) → the
   dispatch still succeeds and NO log is written anywhere — in
   particular never into the installed framework checkout.
4. Audit failure never breaks the dispatch.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness import dispatch_audit, kernel_dispatch, submission_dispatch
from harness.endgame_submission import KaggleLeaderboardTracker


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeSubmitClient:
    """In-memory Kaggle client for the tracker (records every submit)."""

    def __init__(self):
        self.calls = []

    def submit_submission(self, file_name, message, competition, kernel=None, kernel_version=None):
        self.calls.append(
            dict(file_name=file_name, message=message, competition=competition,
                 kernel=kernel, kernel_version=kernel_version)
        )
        return SimpleNamespace(ref=1000 + len(self.calls), message="accepted")

    def list_submissions(self, competition):
        return []


class _FakeKernelClient:
    """In-memory Kaggle client for kernel push (records every push)."""

    def __init__(self):
        self.calls = []

    def kernels_push(self, folder, timeout=None, acc=None):
        self.calls.append(dict(folder=folder, timeout=timeout, acc=acc))
        return SimpleNamespace(
            ref=700 + len(self.calls),
            url=f"https://www.kaggle.com/kernels/own/k/{len(self.calls)}",
            kernel_id=555,
            version_number=len(self.calls),
            error=None,
            invalid_dataset_sources=None,
            invalid_competition_sources=None,
            invalid_kernel_sources=None,
            invalid_model_sources=None,
        )


def _attached_workspace(root: Path) -> Path:
    """Minimal attached workspace: manifest present so the audit resolves it."""
    (root / ".ai-harness").mkdir(parents=True, exist_ok=True)
    (root / ".ai-harness" / "project.yaml").write_text(
        "task_config: configs/task.yaml\nplugin_id: default\n", encoding="utf-8"
    )
    return root


def _log_entries(root: Path) -> list:
    path = root / ".ai-harness" / "dispatch_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SubmissionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aih-dispatch-gov-")
        self.root = _attached_workspace(Path(self.tmp.name))
        self.client = _FakeSubmitClient()
        self.tracker = KaggleLeaderboardTracker(client=self.client, sleep_fn=lambda s: None)
        self.sub = self.root / "submission.csv"
        self.sub.write_text("id,target\n1,0.1\n2,0.9\n", encoding="utf-8")
        self.env = {"HARNESS_PROJECT_ROOT": str(self.root)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatch_appends_audit_entry_with_content_hash(self):
        with mock.patch.dict(os.environ, self.env):
            result = self.tracker.dispatch("comp-a", str(self.sub), "Exp 1")
        self.assertEqual(result["ref"], 1001)
        entries = _log_entries(self.root)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["kind"], "submission")
        self.assertEqual(e["competition"], "comp-a")
        self.assertEqual(e["ref"], 1001)
        self.assertEqual(e["file_sha256"], _sha256(self.sub.read_bytes()))
        self.assertTrue(e["ts"])
        self.assertIn("operator", e)
        self.assertIn("cwd", e)

    def test_identical_resubmission_warns_but_still_dispatches(self):
        with mock.patch.dict(os.environ, self.env):
            r1 = self.tracker.dispatch("comp-a", str(self.sub), "Exp 1")
            r2 = self.tracker.dispatch("comp-a", str(self.sub), "Exp 1 (retry)")
        self.assertNotIn("duplicate_warning", r1)
        self.assertIn("duplicate_warning", r2)
        self.assertEqual(r2["duplicate_of"]["ref"], r1["ref"])
        self.assertEqual(len(self.client.calls), 2, "v1 must not hard-block")
        self.assertEqual(_log_entries(self.root)[1]["duplicate_of"]["ref"], 1001)

    def test_same_file_different_competition_no_warning(self):
        with mock.patch.dict(os.environ, self.env):
            _ = self.tracker.dispatch("comp-a", str(self.sub), "")
            r2 = self.tracker.dispatch("comp-b", str(self.sub), "")
        self.assertNotIn("duplicate_warning", r2)

    def test_changed_file_no_warning(self):
        with mock.patch.dict(os.environ, self.env):
            self.tracker.dispatch("comp-a", str(self.sub), "")
        self.sub.write_text("id,target\n1,0.5\n2,0.5\n", encoding="utf-8")
        with mock.patch.dict(os.environ, self.env):
            r2 = self.tracker.dispatch("comp-a", str(self.sub), "")
        self.assertNotIn("duplicate_warning", r2)

    def test_missing_file_dispatches_with_null_sha_no_crash(self):
        with mock.patch.dict(os.environ, self.env):
            result = self.tracker.dispatch("comp-a", str(self.root / "ghost.csv"), "")
        self.assertEqual(result["ref"], 1001)
        entries = _log_entries(self.root)
        self.assertEqual(entries[0]["file_sha256"], None)

    def test_ledger_duplicate_detected(self):
        # A previously ledger-registered submission with identical content.
        sub_dir = self.root / "submissions"
        sub_dir.mkdir()
        archive = sub_dir / "sub_001_archive.csv"
        archive.write_text(self.sub.read_text(encoding="utf-8"), encoding="utf-8")
        (sub_dir / "registry.json").write_text(
            json.dumps({"submissions": [{
                "id": "sub_001", "exp_id": "exp_001", "created_at": "2026-01-01",
                "status": "SUBMITTED_PENDING", "local_cv_scores": {},
                "official_scores": {}, "score_deltas": {},
                "candidate_path": str(self.sub), "baseline_path": None,
                "archive_path": str(archive), "receipt_path": "",
                "notes": "", "divergence_warning": False,
            }]}),
            encoding="utf-8",
        )
        with mock.patch.dict(os.environ, self.env):
            result = self.tracker.dispatch("comp-a", str(self.sub), "")
        self.assertIn("duplicate_warning", result)
        self.assertEqual(result["duplicate_of"]["source"], "ledger")
        self.assertEqual(result["duplicate_of"]["sub_id"], "sub_001")

    def test_audit_failure_never_breaks_dispatch(self):
        with mock.patch.dict(os.environ, self.env), mock.patch.object(
            dispatch_audit, "record_dispatch", side_effect=OSError("disk full")
        ), mock.patch(
            "harness.endgame_submission.record_dispatch", side_effect=OSError("disk full")
        ):
            result = self.tracker.dispatch("comp-a", str(self.sub), "")
        self.assertEqual(result["ref"], 1001)
        self.assertEqual(len(self.client.calls), 1)

    def test_no_project_context_dispatches_without_any_log(self):
        bare = Path(self.tmp.name) / "bare"
        bare.mkdir()
        ghost = bare / "s.csv"
        ghost.write_text("id,target\n1,0.2\n", encoding="utf-8")
        checkout_log = Path(__file__).resolve().parent.parent / ".ai-harness" / "dispatch_log.jsonl"
        pre_checkout = checkout_log.is_file()
        saved_cwd = os.getcwd()
        # strip the env var so resolution cannot see the temp workspace
        env = {k: v for k, v in os.environ.items() if k != "HARNESS_PROJECT_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True):
            os.chdir(bare)
            try:
                result = self.tracker.dispatch("comp-a", str(ghost), "")
            finally:
                os.chdir(saved_cwd)
        self.assertEqual(result["ref"], 1001)
        self.assertFalse((bare / ".ai-harness").exists(), "no log in bare dir")
        self.assertEqual(checkout_log.is_file(), pre_checkout,
                         "must never write into the framework checkout")


class KernelAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aih-kernel-gov-")
        self.root = _attached_workspace(Path(self.tmp.name))
        self.client = _FakeKernelClient()
        self.nb = self.root / "exp.ipynb"
        self.nb.write_text("{'cells': []}", encoding="utf-8")
        self.env = {"HARNESS_PROJECT_ROOT": str(self.root)}

    def tearDown(self):
        self.tmp.cleanup()

    def _push(self, kernel_id="own/exp048"):
        with mock.patch.dict(os.environ, self.env):
            return kernel_dispatch.dispatch_kernel(
                str(self.nb), kernel_id, "Exp048", client=self.client,
            )

    def test_kernel_push_appends_audit_with_notebook_sha(self):
        result = self._push()
        self.assertTrue(result["ok"])
        entries = _log_entries(self.root)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["kind"], "kernel")
        self.assertEqual(e["kernel_id"], "own/exp048")
        self.assertEqual(e["notebook_sha256"], _sha256(self.nb.read_bytes()))
        self.assertEqual(e["ref"], 701)

    def test_same_kernel_same_content_warns(self):
        self._push()
        r2 = self._push()
        self.assertIn("duplicate_warning", r2)
        self.assertEqual(r2["duplicate_of"]["kernel_id"], "own/exp048")
        self.assertEqual(len(self.client.calls), 2, "v1 must not hard-block")

    def test_version_bump_different_content_no_warning(self):
        self._push()
        self.nb.write_text("{'cells': [{'v': 2}]}", encoding="utf-8")
        r2 = self._push()
        self.assertNotIn("duplicate_warning", r2)


class CliWarningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aih-cli-gov-")
        self.root = _attached_workspace(Path(self.tmp.name))
        self.client = _FakeSubmitClient()
        self.tracker = KaggleLeaderboardTracker(client=self.client, sleep_fn=lambda s: None)
        self.sub = self.root / "submission.csv"
        self.sub.write_text("id,target\n1,0.3\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_dispatch_duplicate_warning_on_stderr_not_stdout(self):
        def _run():
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(
                submission_dispatch, "KaggleLeaderboardTracker", return_value=self.tracker
            ), mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(self.root)}), \
                 contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = submission_dispatch.main(["dispatch", "comp-a", str(self.sub)])
            return code, json.loads(out.getvalue()), err.getvalue()

        c1, p1, e1 = _run()
        self.assertEqual(c1, 0)
        self.assertTrue(p1["ok"])
        self.assertNotIn("duplicate_warning", p1)
        self.assertEqual(e1, "")

        c2, p2, e2 = _run()
        self.assertEqual(c2, 0, "warning is advisory: exit code stays 0")
        self.assertTrue(p2["ok"])
        self.assertIn("duplicate_warning", p2, "payload carries the warning")
        self.assertIn("duplicate", e2.lower(), "human-readable warning on stderr")


if __name__ == "__main__":
    unittest.main()
