import shutil
import unittest
from types import SimpleNamespace

from harness.dispatch_audit import append_dispatch_event
from harness.project import attach_project
from harness.reconciliation import (
    acknowledge_external_submissions,
    reconcile_provider_submissions,
)
from tests.fixtures import make_workspace


class MockKaggleClient:
    def __init__(self, submissions):
        self._subs = submissions

    def competition_submissions(self, competition):
        return self._subs


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.root, self.spec_path = make_workspace(prefix="aih-test-rec-")
        self.addCleanup(shutil.rmtree, self.root)
        self.project = attach_project(self.root, task_config=self.spec_path)

    def test_reconciliation_detects_unrecorded_external_submission(self):
        # Local has 1 legitimate submission with ref 1001
        append_dispatch_event(
            self.root,
            {
                "schema_version": 1,
                "event_id": "event-1",
                "intent_id": "intent-1",
                "event": "submitted",
                "competition": "test-comp",
                "payload_sha256": "a" * 64,
                "prediction_sha256": "b" * 64,
                "history_snapshot_sha256": "c" * 64,
                "session_id": "session-1",
                "receipt_id": "receipt-1",
                "error_code": None,
                "provider_ref": "1001",
                "timestamp": "2026-10-26T12:00:00Z",
            },
        )

        # Provider has 1001 AND unrecorded 1002 (e.g. from scratch script)
        mock_subs = [
            SimpleNamespace(ref="1001", date="2026-10-26", description="legit", fileName="sub1.csv"),
            SimpleNamespace(ref="1002", date="2026-10-26", description="backdoor", fileName="sub2.csv"),
        ]
        client = MockKaggleClient(mock_subs)

        report = reconcile_provider_submissions(
            self.root, competition="test-comp", provider="kaggle", provider_client=client
        )
        self.assertEqual(report["status"], "DISCREPANCY_DETECTED")
        self.assertEqual(len(report["unexplained_submissions"]), 1)
        self.assertEqual(str(report["unexplained_submissions"][0]["ref"]), "1002")

        # Now operator acknowledges ref 1002
        ack_report = acknowledge_external_submissions(
            self.root, refs=["1002"], reason="Operator emergency submission", operator="admin"
        )
        self.assertIsNotNone(ack_report)
        # Should now be clean
        rec_after = reconcile_provider_submissions(
            self.root, competition="test-comp", provider="kaggle", provider_client=client
        )
        self.assertEqual(rec_after["status"], "CLEAN")
        self.assertEqual(len(rec_after["unexplained_submissions"]), 0)


if __name__ == "__main__":
    unittest.main()
