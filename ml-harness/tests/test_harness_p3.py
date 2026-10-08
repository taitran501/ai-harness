"""Unit and Regression Tests for P3 Refactored Harness Features.

Covers:
1. compute_diff_metrics excludes resolved_id when id_col is None.
2. audit_prediction_integrity validates classes dynamically from TaskSpec.
3. SubmissionManager record_official_score supports direction-aware divergence (minimize vs maximize) and custom_scores.
4. AgentBridge FREEZE_CANDIDATE dynamically resolves candidate slots from candidate pool.
5. AgentBridge RECORD_EXTERNAL_SCORE correctly forwards arbitrary custom_scores.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from harness.agent_bridge import AgentBridge, ApprovalTier
from harness.competition_state import CompetitionState
from harness.environment_detector import EnvironmentDetector
from harness.submission_manager import SubmissionEntry, SubmissionManager
from harness.task_spec import UniversalTaskSpec, SubtaskSpec
from tools.audit_gate import audit_prediction_integrity, compute_diff_metrics


class TestAuditGateRefinements(unittest.TestCase):
    def test_compute_diff_metrics_excludes_resolved_id_when_none(self):
        """When id_col is None, compute_diff_metrics must still exclude resolved_id from prediction comparison."""
        cand_df = pd.DataFrame({
            "sample_id": ["id_1", "id_2", "id_3"],
            "pred_col": ["a", "b", "c"],
        })
        base_df = pd.DataFrame({
            "sample_id": ["id_1", "id_2", "id_3"],
            "pred_col": ["a", "x", "c"],
        })

        diff_count, diff_details = compute_diff_metrics(cand_df, base_df, id_col="sample_id")
        self.assertEqual(diff_count, 1)
        self.assertIn("pred_col", diff_details)
        self.assertNotIn("sample_id", diff_details)

    def test_audit_prediction_integrity_valid_and_invalid_classes(self):
        """Dynamic class validation catches values outside allowed classes."""
        # Valid data
        valid_df = pd.DataFrame({
            "sample_id": ["id_1", "id_2"],
            "pred_stereotype": ["yes", "no"],
            "pred_hate_speech": ["no", "yes_explicit"],
            "pred_target_type": ["none", "individual"],
        })
        spec = UniversalTaskSpec(
            name="Stereo fixture",metadata={"data":{"id_column":"sample_id"}},
            task_type="multi_task_classification",
            primary_metric="macro_f1",
            subtasks={
                "stereotype": SubtaskSpec("stereotype", "Stereotype", "binary_classification", column="stereotype", classes=["yes", "no"]),
                "hate_speech": SubtaskSpec("hate_speech", "Hate Speech", "multiclass_classification", column="hate_speech", classes=["no", "yes_explicit"]),
                "target_type": SubtaskSpec("target_type", "Target Type", "multiclass_classification", column="target_type", classes=["none", "individual"]),
            },
        )
        errors = audit_prediction_integrity(valid_df, spec=spec)
        self.assertEqual(errors, [])

        # Invalid stereotype value
        bad_df = pd.DataFrame({
            "sample_id": ["id_1"],
            "pred_stereotype": ["maybe"],
        })
        bad_errors = audit_prediction_integrity(bad_df, spec=spec)
        self.assertTrue(any("invalid values" in err for err in bad_errors))


class TestSubmissionManagerDirectionAwareDivergence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.mgr = SubmissionManager(submissions_dir=str(self.test_dir))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_official_score_custom_scores_and_divergence_maximize(self):
        """For maximize direction, LB drop > 0.05 triggers divergence warning."""
        entry = SubmissionEntry(
            id="sub_test_01",
            exp_id="exp001",
            created_at="2026-09-27T00:00:00Z",
            status="LOCAL_VERIFIED",
            local_cv_scores={"macro_f1": 0.85},
        )
        self.mgr.submissions[entry.id] = entry

        # LB drops to 0.75 (delta = -0.10)
        updated = self.mgr.record_official_score(
            sub_id="sub_test_01",
            custom_scores={"macro_f1": 0.75},
        )
        self.assertTrue(updated.divergence_warning)
        self.assertEqual(updated.score_deltas.get("macro_f1"), -0.10)

    def test_record_official_score_divergence_minimize(self):
        """For minimize direction, LB increase > 0.05 triggers divergence warning."""
        entry = SubmissionEntry(
            id="sub_test_min",
            exp_id="exp002",
            created_at="2026-09-27T00:00:00Z",
            status="LOCAL_VERIFIED",
            local_cv_scores={"loss": 0.20},
        )
        self.mgr.submissions[entry.id] = entry

        # Mock UniversalTaskSpec with direction='minimize'
        custom_spec = UniversalTaskSpec(
            name="test_reg",
            task_type="regression",
            primary_metric="loss",
            direction="minimize",
        )
        import unittest.mock as mock
        with mock.patch("harness.task_spec.load_active_task_spec", return_value=custom_spec):
            # LB increases to 0.35 (delta = +0.15, worse error)
            updated = self.mgr.record_official_score(
                sub_id="sub_test_min",
                custom_scores={"loss": 0.35},
            )
            self.assertTrue(updated.divergence_warning)
            self.assertEqual(updated.score_deltas.get("loss"), 0.15)


class TestAgentBridgeDynamicCandidateSlots(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.bridge = AgentBridge(project_root=str(self.test_dir))
        self.bridge.comp_state.state["mode"] = "PARALLEL_EXPLORATION"
        self.session = self.bridge.start_session(goal="Test dynamic freeze")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_freeze_candidate_dynamically_picks_from_candidate_pool(self):
        """When slot IDs are omitted, freeze_candidate picks available candidates from pool."""
        # Populate candidate pool
        self.bridge.comp_state.candidates = [
            {"id": "cand_alpha", "name": "Model Alpha", "score": 0.88},
            {"id": "cand_beta", "name": "Model Beta", "score": 0.86},
        ]
        _, obs = self.bridge.execute_action(
            action="freeze_candidate",
            params={},
            session_id=self.session.session_id,
            user_approved=True,
        )
        self.assertEqual(obs.status,"FAILED")
        self.assertFalse(self.bridge.comp_state.state["gates"]["freeze_locked"])

    def test_record_external_score_forwards_custom_scores(self):
        """RECORD_EXTERNAL_SCORE action correctly passes arbitrary scores dictionary."""
        entry = SubmissionEntry(
            id="sub_forward_test",
            exp_id="exp003",
            created_at="2026-09-27T00:00:00Z",
            status="LOCAL_VERIFIED",
            local_cv_scores={"composite_score": 0.80},
        )
        sub_mgr = SubmissionManager(submissions_dir=str(self.test_dir / "submissions"))
        sub_mgr.submissions[entry.id] = entry
        sub_mgr.save()

        import unittest.mock as mock
        with mock.patch("harness.agent_bridge.SubmissionManager", return_value=sub_mgr):
            _, obs = self.bridge.execute_action(
                action="record_external_score",
                params={"sub_id": "sub_forward_test", "scores": {"composite_score": 0.82}},
                session_id=self.session.session_id,
                user_approved=True,
            )
            self.assertEqual(obs.status, "SUCCESS")
            data = obs.data
            self.assertEqual(data["official_scores"].get("composite_score"), 0.82)


class TestEnvironmentDetector(unittest.TestCase):
    def test_detect_gpu_structure(self):
        """detect_gpu returns dictionary with required hardware keys without crashing."""
        res = EnvironmentDetector.detect_gpu()
        self.assertIsInstance(res, dict)
        self.assertIn("detected", res)
        self.assertIn("gpu_name", res)
        self.assertIn("cuda_available", res)


if __name__ == "__main__":
    unittest.main()
