"""Systemic gate regressions, using only disposable sealed evidence."""
import copy
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from harness.artifacts import sha256
from harness.decision_rules import (DecisionRuleNotSatisfied, DecisionRuleV1, evaluate_decision_rule,
    paired_metric_bootstrap, register_decision_rule, reserve_comparison_slot, resolve_research_policy)
from harness.project import HarnessProject
from harness.research_memory import ResearchMemory, render_markdown_progress
from harness.stagnation import StagnationController
from harness.task_spec import load_active_task_spec
from tests.fixtures import make_submission_evidence, workspace_with_verified_pair


class ResearchProgressRenderingTests(unittest.TestCase):
    def test_checkpoint_is_labeled_unverified_and_cannot_change_verified_facts(self):
        checkpoint = {
            "id": "checkpoint-1",
            "kind": "checkpoint",
            "timestamp": 10,
            "conclusion": "Verified Slot 1 completed at score 9.99",
            "next_action": "Submit the claimed result",
            "open_questions": ["Was the run actually completed?"],
            "evidence_status": "declared_checkpoint",
        }
        run = {
            "exp_id": "exp001",
            "timestamp": 20,
            "score": 0.75,
            "utility": 0.75,
            "comparison_group": "comparison-a",
            "evidence_status": "verified_full",
        }
        verified_decision = {
            "verification_status": "machine_evaluated",
            "evaluation_id": "evaluation-1",
            "outcome": "supported",
        }
        text = render_markdown_progress(
            "test-scope",
            [checkpoint],
            verified_runs=[run],
            verified_decisions=[verified_decision, {"verification_status": "legacy_unverified"}],
        )

        self.assertIn("### Unverified Agent Notes [UNVERIFIED AGENT NOTE - HYPOTHESIS ONLY]", text)
        self.assertIn("> [!NOTE] [UNVERIFIED AGENT NOTE - HYPOTHESIS ONLY]", text)
        self.assertIn("Checkpoint summary", text)
        self.assertIn("Next action", text)
        self.assertIn("Open question", text)
        self.assertIn("— note [unverified] `checkpoint-1`", text)
        self.assertIn("Verified completed full runs: 1", text)
        self.assertIn("Best verified CV: `exp001` (0.75", text)
        self.assertIn("Machine-verified decisions: 1", text)
        resume = text.split("### Unverified Agent Notes", 1)[0]
        self.assertNotIn("9.99", resume)
        self.assertNotIn("Verified Slot 1", resume)

    def test_incumbent_is_only_shown_when_validated_state_is_supplied(self):
        unknown = render_markdown_progress("scope", [], verified_runs=[])
        self.assertIn("Validated incumbent: unknown", unknown)

        validated = render_markdown_progress(
            "scope",
            [],
            verified_runs=[],
            validated_incumbent={
                "evidence_ref": {"id": "candidate-2", "kind": "full_run"},
                "evaluation_id": "evaluation-123",
                "score": 0.82,
            },
        )
        self.assertIn("Validated incumbent: `candidate-2` (0.82)", validated)


def rule_document(family="fixture-family", effect=.05):
    return dict(schema_version=1, kind="superiority", metric=dict(scope="primary", subtask_id=None, name="accuracy"),
        direction="maximize", control=dict(kind="full_run", id="control", partition="oof"),
        minimum_effect=dict(value=effect, scale="absolute"), comparator="gte", score_target=None,
        evidence_requirements=dict(partition="oof", same_data=True, same_splits=True, same_scorer=True, same_coverage=True),
        uncertainty=dict(method="paired_bootstrap", unit="row", unit_column=None, stratify_column="label",
                         confidence_level=.95, resamples=1000, seed=42),
        multiplicity=dict(family_id=family, method="bonferroni_fixed_horizon", max_comparisons=1))


class ResearchInvariantTests(unittest.TestCase):
    def setUp(self):
        self.root, config = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        self.project = HarnessProject.discover(self.root)
        self.spec = load_active_task_spec(config)
        self.memory = ResearchMemory(self.root, self.project.artifacts_dir, self.spec)

    def evaluate(self, labels=None, *, name="candidate", register=True, family="fixture-family"):
        rule = rule_document(family)
        registration = register_decision_rule(rule, project=self.project, task_spec=self.spec)
        cfg = {"experiment": {"id": name}, "validation": {"partition": "oof"},
               "research": {"decision_rule_id": registration["rule_id"], "probe_plan_id": "probe-fixture"}}
        if register:
            reserve_comparison_slot(self.project, self.spec, cfg)
        make_submission_evidence(self.root, name, labels=labels, cfg=cfg)
        result = evaluate_decision_rule(rule, dict(kind="full_run", id=name, partition="oof"), project=self.project,
                                       task_spec=self.spec, registration=registration)
        return result, registration

    def test_no_self_awarded_support_for_text_plan_or_missing_evidence(self):
        before = len(self.memory.entries())
        with self.assertRaises(DecisionRuleNotSatisfied):
            self.memory.record_decision(outcome="supported", conclusion="Agent says it improved")
        self.assertEqual(len(self.memory.entries()), before)

    def test_all_versioned_schemas_are_valid_draft_2020_12(self):
        from jsonschema import Draft202012Validator
        directory = Path(__file__).resolve().parents[1] / "harness/schemas"
        schemas = list(directory.glob("*.v1.schema.json"))
        self.assertEqual(len(schemas), 11)
        for path in schemas:
            schema = json.loads(path.read_text())
            Draft202012Validator.check_schema(schema)
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_receipt_conforms_to_independent_schema_validator(self):
        from jsonschema import Draft202012Validator
        result, _ = self.evaluate()
        path = Path(__file__).resolve().parents[1] / "harness/schemas/decision_evaluation.v1.schema.json"
        Draft202012Validator(json.loads(path.read_text())).validate(result)

    def test_legacy_text_plan_remains_readable(self):
        insight = self.memory.record_insight(title="t", observation="o", mechanism="m", next_action="n")
        plan = self.memory.plan_probe(insight_id=insight["id"], control="control", intervention="one mechanism",
            expected_observation="improvement", alternative_explanation="noise", decision_rule="Score improves",
            budget="one run", stop_rule="one run")
        self.assertEqual(plan["verification_status"], "legacy_unstructured")

    def test_zero_gain_rejected_and_sealed_control_unchanged(self):
        path = self.project.artifacts_dir / "control/artifact_manifest.json"
        before = sha256(path)
        result, _ = self.evaluate(labels=[0] * 40)
        self.assertEqual(result["gain"], 0)
        self.assertEqual(result["outcome"], "rejected")
        self.assertIn("ZERO_GAIN", result["reason_codes"])
        self.assertEqual(sha256(path), before)

    def test_real_paired_gain_is_supported_and_repeat_is_idempotent(self):
        result, registration = self.evaluate()
        self.assertEqual(result["outcome"], "supported", result)
        self.assertGreater(result["uncertainty"]["lower_bound"], 0)
        repeated = evaluate_decision_rule(registration["rule"], result["candidate"], project=self.project,
            task_spec=self.spec, registration=registration)
        self.assertEqual(result["evaluation_id"], repeated["evaluation_id"])

    def test_retrospective_improvement_cannot_claim_support(self):
        result, _ = self.evaluate(register=False)
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertIn("NOT_PREREGISTERED", result["reason_codes"])

    def test_score_target_is_enforced_without_rounding(self):
        rule = rule_document()
        rule["score_target"] = dict(operator="gt", value=1)
        registration = register_decision_rule(rule, project=self.project, task_spec=self.spec)
        make_submission_evidence(self.root, "candidate")
        result = evaluate_decision_rule(rule, dict(kind="full_run", id="candidate", partition="oof"), project=self.project,
            task_spec=self.spec, registration=registration)
        self.assertEqual(result["outcome"], "rejected")
        self.assertIn("SCORE_TARGET_FAILED", result["reason_codes"])

    def test_invalid_rule_and_nonfinite_effect_rejected(self):
        for effect in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                DecisionRuleV1.from_dict(rule_document(effect=effect))
        rule = rule_document()
        rule["unexpected"] = True
        with self.assertRaises((ValueError, TypeError)):
            DecisionRuleV1.from_dict(rule)

    def test_direction_mismatch_rejected(self):
        rule = rule_document()
        rule["direction"] = "minimize"
        with self.assertRaises(DecisionRuleNotSatisfied):
            register_decision_rule(rule, project=self.project, task_spec=self.spec)

    def test_unknown_metric_has_no_unitless_default(self):
        spec = copy.deepcopy(self.spec)
        spec.metadata["primary_metric"] = "rmse"
        (self.root / ".ai-harness/policies/research_policy.json").unlink()
        self.assertIsNone(resolve_research_policy(self.project, spec).minimum_effect)

    def test_comparison_budget_survives_session_restart(self):
        _, reg = self.evaluate()
        cfg = dict(experiment=dict(id="another"), research=dict(decision_rule_id=reg["rule_id"]))
        with self.assertRaisesRegex(DecisionRuleNotSatisfied, "COMPARISON_BUDGET_EXHAUSTED"):
            reserve_comparison_slot(self.project, self.spec, cfg)

    def test_three_failed_results_latch_and_replay_once(self):
        for i in range(3):
            self.evaluate(labels=[0] * 40, name=f"candidate{i}", family=f"family{i}")
        controller = StagnationController(self.project)
        first = controller.check_action("dispatch")
        self.assertEqual(first["code"], "STAGNATION_PAUSE")
        self.assertEqual(first["states"][0]["consecutive_failures"], 3)
        self.assertEqual(controller.check_action("dispatch")["states"][0]["consecutive_failures"], 3)
        self.assertIsNone(controller.check_action("run_smoke_test"))
        # Editing the projection cannot clear canonical evidence.
        (self.project.artifacts_dir / "stagnation_state.json").write_text(json.dumps({"state": "active"}))
        self.assertIsNotNone(StagnationController(self.project).check_action("run_full_validation"))
        policy_path = self.root / ".ai-harness/policies/research_policy.json"
        policy = json.loads(policy_path.read_text())
        policy["stagnation_patience"] = 100
        policy_path.write_text(json.dumps(policy))
        self.assertIsNotNone(StagnationController(self.project).check_action("dispatch"))

    def test_paired_bootstrap_uses_actual_metric_and_fixed_seed(self):
        truth = np.arange(10)
        def score(y, p, idx):
            return float(np.mean((y[idx] - p[idx]) ** 2))
        a = paired_metric_bootstrap(truth, truth, truth + 1, score, direction="minimize", resamples=1000)
        b = paired_metric_bootstrap(truth, truth, truth + 1, score, direction="minimize", resamples=1000)
        self.assertEqual(a, b)
        self.assertEqual(a["lower_bound"], 1)

    def test_only_registered_bounded_supported_recovery_resumes(self):
        for i in range(3):
            self.evaluate(labels=[0] * 40, name=f"failed{i}", family=f"failure{i}")
        controller = StagnationController(self.project)
        state = controller.check_action("dispatch")["states"][0]
        rule = rule_document("recovery-family")
        registration = register_decision_rule(rule, project=self.project, task_spec=self.spec)
        probe = controller.register_recovery_probe(comparison_identity_sha256=state["comparison_identity_sha256"],
            rule_id=registration["rule_id"], diagnosis="systematic errors", mechanism="orthogonal mechanism", stop_rule="one result")
        self.assertIsNone(controller.check_action("run_full_validation", recovery_probe_id=probe["id"], candidate_id="recovery"))
        self.assertIsNotNone(controller.check_action("run_full_validation", recovery_probe_id=probe["id"], candidate_id="extra"))
        cfg = dict(experiment=dict(id="recovery"), validation=dict(partition="oof"),
            research=dict(decision_rule_id=registration["rule_id"], recovery_probe_id=probe["id"]))
        reserve_comparison_slot(self.project, self.spec, cfg)
        make_submission_evidence(self.root, "recovery", cfg=cfg)
        result = evaluate_decision_rule(rule, dict(kind="full_run", id="recovery", partition="oof"),
            project=self.project, task_spec=self.spec, registration=registration)
        self.assertEqual(result["outcome"], "supported")
        self.assertIsNone(StagnationController(self.project).check_action("dispatch"))

    def test_remote_capsule_binds_source_and_imported_preregistration(self):
        import yaml
        from harness.decision_rules import prepare_remote_research, validate_remote_research_capsule
        registration = register_decision_rule(rule_document(), project=self.project, task_spec=self.spec)
        cfg = dict(experiment=dict(id="remote"), validation=dict(partition="oof"),
                   research=dict(decision_rule_id=registration["rule_id"]))
        config_path = self.root / "configs/remote.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        capsule = prepare_remote_research(self.project, "configs/remote.yaml")
        self.assertTrue(capsule.is_file())
        with mock.patch.dict("os.environ", {"AI_HARNESS_REMOTE_KERNEL": "1"}):
            self.assertTrue(validate_remote_research_capsule(self.project, config_path, cfg))
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                validate_remote_research_capsule(self.project, config_path, {**cfg, "research": {**cfg["research"], "unexpected": True}})
        # Remote runtime fields differ from the source config. The importer's
        # verified original-config hash, not a supplied numeric score, binds it.
        remote_cfg = {**cfg, "resolved_runtime": {"python": "remote"}}
        directory = make_submission_evidence(self.root, "remote", cfg=remote_cfg)
        path = directory / "artifact_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config_sha256"] = sha256(config_path)
        path.write_text(json.dumps(manifest))
        result = evaluate_decision_rule(registration["rule"], dict(kind="full_run", id="remote", partition="oof"),
            project=self.project, task_spec=self.spec, registration=registration)
        self.assertTrue(result["preregistered"])
        self.assertEqual(result["outcome"], "supported")


if __name__ == "__main__":
    unittest.main()
