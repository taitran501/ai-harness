"""Submission sends fail closed; execution kernel pushes retain advisory history."""
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

    def get_submission_limits(self, competition):
        return dict(daily_quota=5, num_today=0, has_remaining_quota=True)

    def submit_submission(self, file_name, message, competition, kernel=None, kernel_version=None, *, _permit=None):
        from harness.submission_gate import consume_dispatch_permit
        consume_dispatch_permit(_permit, file_name=file_name, competition=competition, kernel=kernel, kernel_version=kernel_version)
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
    """Real shared gate with a fake transport; kernel pushes retain v0 logging."""
    def setUp(self):
        import shutil
        import pandas as pd
        from tests.fixtures import workspace_with_verified_pair
        from harness.submission_gate import SubmissionGate
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        self.gate = SubmissionGate(self.root)
        self.client = _FakeSubmitClient()
        self.tracker = KaggleLeaderboardTracker(client=self.client, sleep_fn=lambda _: None)
        self.sub = self.gate.plugin.build_submission(self.gate.context,
            pd.read_csv(self.gate.project.artifacts_dir / 'control/test_predictions.csv'), self.root / 'submission.csv')

    def send(self, **options):
        return self.tracker.dispatch('comp-a', str(self.sub), 'Exp 1', project_root=self.root,
            exp_id='control', user_approved=True, **options)

    def test_dispatch_records_durable_intent_before_provider_call(self):
        result = self.send()
        events = dispatch_audit.read_submission_events(self.root)
        self.assertEqual([e['event'] for e in events], ['prepared', 'sending', 'submitted'])
        self.assertEqual(result['ref'], 1001)
        self.assertEqual(events[-1]['payload_sha256'], _sha256(self.sub.read_bytes()))

    def test_identical_resubmission_is_blocked_before_provider_call(self):
        from harness.submission_gate import SubmissionGateError
        self.send()
        with self.assertRaisesRegex(SubmissionGateError, 'DUPLICATE_PAYLOAD'):
            self.send()
        self.assertEqual(len(self.client.calls), 1)

    def test_changed_file_requires_new_sealed_evidence(self):
        from harness.submission_gate import SubmissionGateError
        self.send()
        self.sub.write_text('id,label\n40,1\n')
        with self.assertRaisesRegex(SubmissionGateError, 'PAYLOAD_EVIDENCE_MISMATCH'):
            self.send()
        self.assertEqual(len(self.client.calls), 1)

    def test_missing_file_is_blocked(self):
        self.sub.unlink()
        with self.assertRaisesRegex(ValueError, 'PAYLOAD_MISSING'):
            self.send()
        self.assertEqual(self.client.calls, [])

    def test_durable_audit_failure_blocks_provider(self):
        with mock.patch('harness.submission_gate.append_dispatch_event', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.send()
        self.assertEqual(self.client.calls, [])

    def test_missing_workspace_cannot_dispatch_through_fake_client(self):
        with self.assertRaisesRegex(ValueError, 'WORKSPACE_REQUIRED'):
            self.tracker.dispatch('comp-a', str(self.sub), project_root=self.root / 'bare')
        self.assertEqual(self.client.calls, [])


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
    def test_cli_reports_blocked_unaudited_dispatch_as_json(self):
        client = _FakeSubmitClient()
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=lambda _: None)
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(submission_dispatch, 'KaggleLeaderboardTracker', return_value=tracker), contextlib.redirect_stdout(out):
                code = submission_dispatch.main(['dispatch', 'comp-a', 'missing.csv', '--project-root', tmp])
        payload = json.loads(out.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(payload['ok'])
        self.assertEqual(payload['error_code'], 'WORKSPACE_REQUIRED')
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
