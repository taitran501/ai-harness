"""Unit tests for the Antigravity <-> ML Harness Bidirectional AgentBridge (`harness/agent_bridge.py`)."""

from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import tempfile
import yaml

from harness.agent_bridge import AgentBridge, TypedActionName
from harness.grounding_gate import GroundingGate

SOURCE_ROOT = Path(__file__).resolve().parent.parent


class TestAgentBridge(unittest.TestCase):
    def setUp(self) -> None:
        self._workspace = tempfile.TemporaryDirectory(prefix="test_agent_bridge_")
        self.project_root = Path(self._workspace.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.project_root / "configs")
        self.bridge = AgentBridge(self.project_root)

    def tearDown(self) -> None:
        self._workspace.cleanup()

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
        # B1-a regression: an agent that *declares* a verified grounding source
        # (real source_id + verified_via) creates a config that passes the gate
        # and is runnable. The anti-hallucination guarantee is that a *missing*
        # grounding is flagged UNVERIFIED (see test below) and the runner refuses
        # it, so this path must actually verify the source.
        sess = self.bridge.start_session(goal="Test GroundingGate schema compliance")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp999_feat_bridge_test",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "bridge_test",
                "source_type": "verified_baseline",
                "source_id": "sklearn_tfidf_logreg_canonical",
                "retrieval_summary": "Canonical TFIDF/logreg baseline; parent diagnostics show rare-slice underfitting.",
                "verified_via": "local_registry",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["grounding_status"], "PASSED")
        cfg_file = self.project_root / "configs" / "exp999_feat_bridge_test.yaml"
        self.assertTrue(cfg_file.exists())

        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            passed, errors, _ = GroundingGate.verify_config(
                cfg_data, raise_on_error=False
            )
            self.assertTrue(passed, f"GroundingGate failed with errors: {errors}")
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_create_experiment_unverified_grounding_flagged(self) -> None:
        # B1-a anti-hallucination regression: an agent that omits grounding must
        # NOT get a fabricated 'verified_via'. The config is still written (workflow
        # intact) but is flagged UNVERIFIED so the runner refuses to train it.
        sess = self.bridge.start_session(goal="Test unverified grounding is flagged")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp997_unverified_grounding",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "no_source",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["grounding_status"], "UNVERIFIED")
        self.assertEqual(obs.data["grounding_verification"], "unverified_placeholder")
        self.assertTrue(obs.data["grounding_errors"])
        cfg_file = self.project_root / "configs" / "exp997_unverified_grounding.yaml"
        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_data["grounding"]["verified_via"], "not_verified")
            # The grounding gate must reject this non-baseline proposal.
            passed, errors, _ = GroundingGate.verify_config(cfg_data, raise_on_error=False)
            self.assertFalse(passed)
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_create_experiment_flexible_model_config(self) -> None:
        sess = self.bridge.start_session(goal="Test flexible model config in create_experiment")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp998_feat_custom_model",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "custom_model_test",
                "model": {
                    "name": "xgboost",
                    "params": {"n_estimators": 50, "max_depth": 4},
                },
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.project_root / "configs" / "exp998_feat_custom_model.yaml"
        self.assertTrue(cfg_file.exists())
        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_data["model"]["name"], "xgboost")
            self.assertEqual(cfg_data["model"]["params"]["n_estimators"], 50)
            self.assertEqual(cfg_data["model"]["params"]["max_depth"], 4)
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_create_experiment_preserves_custom_factory(self) -> None:
        sess = self.bridge.start_session(goal="Test custom factory model config in create_experiment")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp997_feat_custom_factory",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "custom_factory_test",
                "model": {
                    "factory": "custom_module.models.build_custom_model",
                    "params": {"learning_rate": 1e-4, "hidden_dim": 256},
                },
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.project_root / "configs" / "exp997_feat_custom_factory.yaml"
        self.assertTrue(cfg_file.exists())
        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_data["model"]["factory"], "custom_module.models.build_custom_model")
            self.assertEqual(cfg_data["model"]["params"]["learning_rate"], 1e-4)
            self.assertEqual(cfg_data["model"]["params"]["hidden_dim"], 256)
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
        self.assertEqual(
            obs_approved.status, "FAILED"
        )  # Approval cannot certify a legacy source.
        sess, obs = self.bridge.execute_action(
            "synthesize_candidate",
            params={"name": "missing", "mode": "specialist_router"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "FAILED")
        self.assertEqual(sess.compute_usage["experiments_used"], 0)
        from harness.resource_ledger import reserve

        reserve(
            self.bridge.artifacts_dir / "resource_ledger.json",
            sess.session_id,
            "run_full_validation",
            "local",
            {"max_experiments": 1},
        )
        _, obs = self.bridge.execute_action(
            "run_full_validation",
            params={"config_path": "missing.yaml"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "BLOCKED_BY_BUDGET")

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
        self.assertEqual(obs1.status, "FAILED")
        sess.consecutive_rejections = 1
        self.bridge.save_session(sess)
        self.assertIsNone(
            self.bridge._check_budget_and_stop_guards(sess, "run_full_validation", {})
        )
        sess.budget["hard_stop_on_consecutive_rejections"] = True
        self.assertEqual(
            self.bridge._check_budget_and_stop_guards(sess, "run_full_validation", {})[
                0
            ],
            "STOP_POLICY_TRIGGERED",
        )

    def test_bootstrap_task_typed_action(self) -> None:
        import tempfile, shutil
        import pandas as pd

        t_dir = Path(tempfile.mkdtemp(prefix="test_bridge_bootstrap_"))
        try:
            d_dir = t_dir / "dataset"
            d_dir.mkdir(parents=True)
            df = pd.DataFrame(
                {"id": [1, 2, 3], "feat": ["a", "b", "c"], "target": [0, 1, 0]}
            )
            df.to_csv(d_dir / "train.csv", index=False)
            doc_file = t_dir / "TASK.md"
            doc_file.write_text(
                "- **Competition Name**: Bridge Boot Test\n- **Target Columns**: target\n",
                encoding="utf-8",
            )

            sess = self.bridge.start_session(goal="Test bootstrap_task typed action")
            sess, obs = self.bridge.execute_action(
                action=TypedActionName.BOOTSTRAP_TASK.value,
                params={
                    "dataset_dir": str(d_dir),
                    "context_file": str(doc_file),
                    "output_config": str(t_dir / "task.yaml"),
                    "project_root": str(t_dir),
                },
                session_id=sess.session_id,
            )
            self.assertEqual(obs.status, "SUCCESS", f"Failed with: {obs.summary}")
            self.assertIn("Bootstrapped task", obs.summary)
        finally:
            shutil.rmtree(t_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
