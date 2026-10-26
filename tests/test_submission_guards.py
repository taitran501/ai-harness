"""Preparation delegates to runtime validation; dispatch owns promotion policy."""
import shutil
import unittest

from harness.agent_bridge import AgentBridge
from harness.submission_manager import SubmissionEntry
from tests.fixtures import make_submission_evidence, workspace_with_verified_pair


class TestSubmissionGuards(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        self.bridge = AgentBridge(self.root)
        self.art = self.bridge.artifacts_dir

    def test_preparation_validates_declared_predictions(self):
        evidence = self.bridge._guard_submission(self.art / "control/test_predictions.csv", "control", self.bridge._new_submission_manager())
        self.assertEqual(evidence["scope"], "validation")
        self.assertEqual(evidence["checks"]["integrity"], "PASS")
        self.assertNotIn("mmes", evidence["checks"])

    def test_semantic_duplicates_are_blocked_across_prepared_models(self):
        path = make_submission_evidence(self.root, "candidate", labels=[0] * 40)
        manager = self.bridge._new_submission_manager()
        manager.submissions["prior"] = SubmissionEntry(id="prior", exp_id="control", created_at="t", status="LOCAL_VERIFIED",
            candidate_path=str(self.art / "control/test_predictions.csv"))
        with self.assertRaisesRegex(ValueError, "SEMANTIC_DUPLICATE"):
            self.bridge._guard_submission(path / "test_predictions.csv", "candidate", manager)

    def test_empty_prediction_files_fail_integrity(self):
        path = self.root / "empty.csv"
        path.write_text("id,pred_label\n")
        with self.assertRaisesRegex(ValueError, "PREDICTION_INTEGRITY"):
            self.bridge._guard_submission(path, "candidate", self.bridge._new_submission_manager())

    def test_missing_prior_snapshots_are_not_silently_ignored(self):
        manager = self.bridge._new_submission_manager()
        manager.submissions["prior"] = SubmissionEntry(id="prior", exp_id="prior", created_at="t", status="LOCAL_VERIFIED",
            candidate_path=str(self.root / "missing.csv"))
        with self.assertRaisesRegex(ValueError, "HISTORY_INCOMPLETE"):
            self.bridge._guard_submission(self.art / "control/test_predictions.csv", "control", manager)

    def test_preparation_never_claims_metric_promotion(self):
        path = make_submission_evidence(self.root, "candidate")
        evidence = self.bridge._guard_submission(path / "test_predictions.csv", "candidate", self.bridge._new_submission_manager())
        self.assertEqual(evidence["scope"], "validation")
        self.assertNotIn("outcome", evidence)
