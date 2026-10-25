"""Tests for agent session resilience, parameter aliases, and scorer tolerance.

Verifies fixes for real agent session issues:
1. Canonical generic scorer hash tolerance in validate_run
2. load_active_task_spec directory path resolution
3. Action aliases (checkpoint, checkpoint_research_progress, record_insight, etc.)
4. record_research_insight parameter aliases (hypothesis, observation, title fallback)
5. plan_research_probe defensive parameter defaults
6. record_research_decision outcome aliases (accepted, verified_and_ready) and conclusion fallback
7. Descriptive prose evidence handling in research memory
8. harness.agent shim package (harness.agent, harness.agent.actions, harness.agent.cli)
"""
import json
import shutil
import unittest

from harness.artifacts import sha256, validate_run
from harness.bridge_common import (
    ACTION_ALIASES,
    _TYPED_ACTION_NAMES,
    get_action_schema,
    resolve_action_name,
)
from harness.research_memory import (
    normalize_decision_outcome,
)
from harness.task_spec import load_active_task_spec
from tests.fixtures import make_workspace


class TestAgentSessionResilience(unittest.TestCase):
    def setUp(self):
        self.root, self.task_config = make_workspace()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_canonical_generic_scorer_hashes_in_validate_run(self):
        """Historical canonical generic scorer hashes must be accepted under generic scorer."""
        exp_dir = self.root / ".ai-harness" / "artifacts" / "test_task" / "exp000_baseline"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "run_config.yaml").write_text("model: {}\n", encoding="utf-8")
        (exp_dir / "metrics.json").write_text(
            json.dumps({
                "experiment_id": "exp000_baseline",
                "primary_score": 0.92,
                "direction": "maximize",
                "primary_metric": "roc_auc",
            }),
            encoding="utf-8",
        )
        (exp_dir / "oof_predictions.csv").write_text("id,pred\n1,0.5\n", encoding="utf-8")
        (exp_dir / "evaluation_truth.csv").write_text("id,truth\n1,1\n", encoding="utf-8")

        cfg_hash = sha256(exp_dir / "run_config.yaml")
        manifest = {
            "schema_version": 2,
            "status": "COMPLETED",
            "run_kind": "full",
            "run_id": "run_001",
            "exp_id": "exp000_baseline",
            "task_sha256": "dummy_task_hash",
            "scorer_sha256": "aca79278b34ca0000639c10298c1a0c789281062d3e4b8e03ae2cc46a9ffb87c",
            "custom_scorer_sha256": None,
            "data_sha256": "dummy_data",
            "split_sha256": "dummy_split",
            "resolved_config_sha256": cfg_hash,
            "files": {
                "run_config.yaml": cfg_hash,
                "metrics.json": sha256(exp_dir / "metrics.json"),
                "oof_predictions.csv": sha256(exp_dir / "oof_predictions.csv"),
                "evaluation_truth.csv": sha256(exp_dir / "evaluation_truth.csv"),
            },
        }
        (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        res = validate_run(exp_dir, expected={"task_sha256": "dummy_task_hash"})
        self.assertEqual(res["exp_id"], "exp000_baseline")

    def test_load_active_task_spec_with_directory_path(self):
        """load_active_task_spec on a directory must resolve configs/task.yaml."""
        spec = load_active_task_spec(self.root)
        self.assertTrue(bool(spec.name))
        self.assertNotEqual(spec.name, "universal_task")

    def test_action_aliases_resolution(self):
        """Common action name aliases must resolve to canonical typed action names."""
        self.assertEqual(resolve_action_name("checkpoint"), "record_research_checkpoint")
        self.assertEqual(resolve_action_name("save_checkpoint"), "record_research_checkpoint")
        self.assertEqual(resolve_action_name("checkpoint_research_progress"), "record_research_checkpoint")
        self.assertEqual(resolve_action_name("record_insight"), "record_research_insight")
        self.assertEqual(resolve_action_name("record_decision"), "record_research_decision")
        self.assertEqual(resolve_action_name("plan_probe"), "plan_research_probe")
        self.assertEqual(resolve_action_name("establish_baseline"), "establish_baseline_anchor")
        self.assertEqual(resolve_action_name("get_next_exp_id"), "get_next_experiment_id")

        # All aliases must be members of _TYPED_ACTION_NAMES
        for alias in ACTION_ALIASES:
            self.assertIn(alias, _TYPED_ACTION_NAMES)

        # get_action_schema must resolve aliases
        schema = get_action_schema("checkpoint")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["summary"], get_action_schema("record_research_checkpoint")["summary"])

    def test_decision_outcome_aliases(self):
        """Outcome aliases like accepted, verified_and_ready, failed must map properly."""
        self.assertEqual(normalize_decision_outcome("accepted"), "supported")
        self.assertEqual(normalize_decision_outcome("Verified_And_Ready"), "supported")
        self.assertEqual(normalize_decision_outcome("keep"), "supported")
        self.assertEqual(normalize_decision_outcome("failed"), "rejected")
        self.assertEqual(normalize_decision_outcome("discard"), "rejected")
        self.assertEqual(normalize_decision_outcome("postponed"), "deferred")
        self.assertEqual(normalize_decision_outcome("uncertain"), "inconclusive")

    def test_bridge_research_action_parameter_tolerance(self):
        """Bridge must tolerate missing or aliased parameters in research memory actions."""
        from harness.agent_bridge import AgentBridge
        bridge = AgentBridge(project_root=self.root)

        # 1. record_research_insight with hypothesis instead of observation/title
        sess, obs = bridge.execute_action(
            action="record_research_insight",
            params={
                "hypothesis": "Feature interactions between route and delays boost tree performance.",
                "mechanism": "Enables trees to identify high-congestion flight segments.",
                "next_action": "Run probe with interaction features",
            },
        )
        self.assertEqual(obs.status, "SUCCESS")
        insight_id = obs.data["id"]
        self.assertTrue(bool(insight_id))

        # 2. plan_research_probe with minimal parameters
        sess, obs = bridge.execute_action(
            action="plan_research_probe",
            params={
                "intervention": "LightGBM with route delay cross features",
                "expected_observation": "AUC increases by +0.0005",
            },
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["insight_id"], insight_id)

        # 3. record_research_decision with outcome alias and lesson instead of conclusion
        sess, obs = bridge.execute_action(
            action="record_research_decision",
            params={
                "decision": "accepted",
                "lesson": "Route delay interactions reliably improve validation AUC.",
                "next_action": "Promote candidate to baseline anchor",
                "evidence": "Kaggle LB scored 0.96213 with global rank 19",
            },
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["outcome"], "supported")

        # 4. record_research_checkpoint via alias 'checkpoint'
        sess, obs = bridge.execute_action(
            action="checkpoint",
            params={
                "summary": "Rehearsal and probe phases completed successfully.",
                "next_action": "Proceed to ensemble optimization",
            },
        )
        self.assertEqual(obs.status, "SUCCESS")

    def test_harness_agent_shim_imports(self):
        """harness.agent shims must import cleanly."""
        from harness.agent import AgentBridge as AB
        from harness.agent.actions import ActionHandlerMixin as AHM
        from harness.agent.cli import main as cli_main

        self.assertIsNotNone(AB)
        self.assertIsNotNone(AHM)
        self.assertTrue(callable(cli_main))


if __name__ == "__main__":
    unittest.main()
