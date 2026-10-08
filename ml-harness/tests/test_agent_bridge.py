"""Unit tests for the Antigravity <-> ML Harness Bidirectional AgentBridge (`harness/agent_bridge.py`)."""

from __future__ import annotations

import unittest
from pathlib import Path
import yaml

from harness.agent_bridge import AgentBridge, TypedActionName
from harness.agent_runtime import AgentRuntime
from harness.grounding_gate import GroundingGate


class TestAgentBridge(unittest.TestCase):
    def setUp(self) -> None:
        from harness.io_utils import get_project_root
        self.project_root = get_project_root()
        self.bridge = AgentBridge(self.project_root)

    def test_context_packet_and_markdown(self) -> None:
        pkt = self.bridge.get_context_packet()
        self.assertEqual(pkt["runtime_owner"], "Antigravity (IDE)")
        self.assertIn("competition_hud", pkt)
        self.assertIn("ucb_recommendation", pkt)
        self.assertIn("available_typed_actions", pkt)

        md = self.bridge.render_context_markdown()
        self.assertIn("ANTIGRAVITY <-> HARNESS CONTEXT PACKET", md)
        self.assertIn("Typed Bridge Actions", md)

    def test_create_experiment_passes_grounding_gate(self) -> None:
        sess = self.bridge.start_session(goal="Test GroundingGate schema compliance")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp999_feat_bridge_test",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "bridge_test",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.project_root / "configs" / "exp999_feat_bridge_test.yaml"
        self.assertTrue(cfg_file.exists())

        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            passed, errors, _ = GroundingGate.verify_config(cfg_data, raise_on_error=False)
            self.assertTrue(passed, f"GroundingGate failed with errors: {errors}")
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_session_budget_and_approval_guards(self) -> None:
        sess = self.bridge.start_session(
            goal="Test session budget & USER_APPROVAL enforcement",
            budget={"max_experiments": 1, "max_consecutive_rejections": 2},
        )

        # 1. USER_APPROVAL action without user_approved=True must pause
        sess, obs_pause = self.bridge.execute_action(
            action=TypedActionName.FREEZE_CANDIDATE.value,
            params={"slot_1_id": "baseline_v1_comment"},
            session_id=sess.session_id,
            user_approved=False,
        )
        self.assertEqual(obs_pause.status, "PAUSED_FOR_APPROVAL")
        self.assertEqual(sess.status, "PAUSED_FOR_APPROVAL")
        self.assertIsNotNone(sess.pending_approval_action)

        # 2. Approve pending action -> executes and unlocks
        sess, obs_approved = self.bridge.approve_pending_action(sess.session_id)
        self.assertIsNotNone(obs_approved)
        self.assertEqual(obs_approved.status,"FAILED")  # Approval cannot certify a legacy source.
        sess,obs=self.bridge.execute_action("synthesize_candidate",params={"name":"missing","mode":"specialist_router"},session_id=sess.session_id)
        self.assertEqual(obs.status,"FAILED")
        self.assertEqual(sess.compute_usage["experiments_used"],0)
        from harness.resource_ledger import reserve

        reserve(
            self.bridge.artifacts_dir / "resource_ledger.json",
            sess.session_id,
            "run_full_validation",
            "local",
            {"max_experiments": 1},
        )
        _,obs=self.bridge.execute_action("run_full_validation",params={"config_path":"missing.yaml"},session_id=sess.session_id)
        self.assertEqual(obs.status,"BLOCKED_BY_BUDGET")

    def test_backward_compatible_agent_runtime_adapter(self) -> None:
        rt = AgentRuntime(self.project_root)
        run = rt.start_run(goal="Adapter compatibility test", max_experiments=2, auto_approve_all=False)
        self.assertEqual(run.status, "PAUSED_FOR_APPROVAL")
        self.assertIsNotNone(run.pending_approval_action)

    def test_soft_stop_reassess_recommended_and_spec_min_improvement(self) -> None:
        # 1. Verify spec-driven min_meaningful_improvement resolution from configs/task.yaml
        spec_min = self.bridge._get_spec_min_improvement()
        self.assertAlmostEqual(spec_min, 0.002, places=4)

        # 2. Verify consecutive rejections trigger REASSESS_RECOMMENDED (soft stop) without blocking execution
        sess = self.bridge.start_session(
            goal="Test soft stop REASSESS_RECOMMENDED vs hard stop",
            budget={
                "max_consecutive_rejections": 1,
                "min_meaningful_improvement": 99.0,  # Force rejection on any finite delta
                "hard_stop_on_consecutive_rejections": False,
            },
        )
        sess, obs1 = self.bridge.execute_action(
            action=TypedActionName.SYNTHESIZE_CANDIDATE.value,
            params={"name": "soft_stop_probe_1", "mode": "specialist_router"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs1.status,"FAILED")
        sess.consecutive_rejections=1; self.bridge.save_session(sess)
        self.assertIsNone(self.bridge._check_budget_and_stop_guards(sess,"run_full_validation",{}))
        sess.budget["hard_stop_on_consecutive_rejections"]=True
        self.assertEqual(self.bridge._check_budget_and_stop_guards(sess,"run_full_validation",{})[0],"STOP_POLICY_TRIGGERED")


if __name__ == "__main__":
    unittest.main()
