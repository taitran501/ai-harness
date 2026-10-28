"""Macro projections and review claims against disposable sealed evidence."""
import copy
import json
import shutil
import unittest

from harness.macro_research import (FAMILIES, compute_macro_hypothesis_map, load_memory,
    record_macro_review, retrospective_status)
from harness.decision_rules import reserve_comparison_slot
from harness.policy_schema import validate_document
from harness.project import HarnessProject
from tests.fixtures import make_submission_evidence, workspace_with_verified_pair
from tests.test_research_memory import rule_document


def review_payload(memory, **overrides):
    return dict(dict(session_id="session_test", map_digest=compute_macro_hypothesis_map(memory, memory.sync())["map_digest"],
        changes_since_last_review="Reviewed broad coverage and latest evidence.",
        supported_mechanisms=[], rejected_mechanisms=[], highest_priority_untouched=dict(
            family_id="target", proposed_direction="Compare target formulations", rationale="Unexplored mechanism",
            source_ref="diagnostics:target-distribution"),
        next_probe_rationale="Target formulation changes the learning problem rather than a nearby parameter.",
        next_action="Register a one-fold target probe with a fixed control.", budget_limit=1), **overrides)


class MacroResearchTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        self.project = HarnessProject.discover(self.root)
        self.memory = load_memory(self.project)

    def insight(self, family="representation"):
        return self.memory.record_insight(title="Mechanism", observation="Observed errors",
            mechanism="Alternative representation", next_action="Plan a discriminating test",
            family_id=family, source_refs=["eda:errors"])

    def decision(self, outcome="supported", stop_rule="Stop after one run", revisit_when="New orthogonal evidence"):
        insight = self.insight()
        plan = self.memory.plan_probe(insight_id=insight["id"], control="control", intervention="Change representation",
            expected_observation="Fewer errors", alternative_explanation="Noise", decision_rule=rule_document(),
            budget="One run", stop_rule=stop_rule)
        cfg = dict(experiment=dict(id="candidate"), validation=dict(partition="oof"), research=dict(
            insight_id=insight["id"], probe_plan_id=plan["id"], decision_rule_id=plan["rule_id"]))
        reserve_comparison_slot(self.project, self.memory.spec, cfg)
        make_submission_evidence(self.root, "candidate", cfg=cfg, labels=None if outcome == "supported" else [0]*40)
        decision = self.memory.record_decision(insight_id=insight["id"], rule_id=plan["rule_id"],
            candidate_evidence=dict(kind="full_run", id="candidate", partition="oof"),
            conclusion="Interpreted the registered test", lesson="Mechanism has evidence", next_action="Explore gaps",
            revisit_when=revisit_when)
        return insight, plan, decision

    def map(self):
        return compute_macro_hypothesis_map(self.memory, self.memory.sync())

    def claim(self, insight, decision):
        return dict(family_id="representation", insight_id=insight["id"], decision_id=decision["id"],
                    run_id="candidate", mechanism_summary="Alternative representation")

    def test_taxonomy_and_legacy_unclassified(self):
        insight = self.insight("not-a-family")
        self.assertEqual(insight["family_id"], "unclassified")
        self.assertEqual(insight["source_refs"], ["eda:errors"])
        macro = self.map()
        self.assertEqual(set(macro["families"]), set(FAMILIES))
        self.assertEqual(macro["families"]["unclassified"]["state"], "untried")
        self.assertEqual(macro["families"]["unclassified"]["active"], 0)
        self.assertEqual(macro["families"]["target"]["state"], "untried")
        for value in ("bad", [""], [4]):
            with self.assertRaises(ValueError):
                self.memory.record_insight(title="t", observation="o", mechanism="m", next_action="n", source_refs=value)

    def test_unverified_insight_does_not_activate_family_but_registered_plan_does(self):
        insight = self.memory.record_insight(title="Legacy", observation="A possible pattern", mechanism="M",
            next_action="Probe", family_id="representation")
        row = self.map()["families"]["representation"]
        self.assertEqual((row["active"], row["state"], row["insight_count"]), (0, "untried", 1))
        self.memory.plan_probe(insight_id=insight["id"], control="fixed control", intervention="one mechanism",
            expected_observation="fewer errors", alternative_explanation="noise", decision_rule=rule_document(),
            budget="one run", stop_rule="stop after one run")
        row = self.map()["families"]["representation"]
        self.assertEqual((row["active"], row["state"]), (1, "active"))

    def test_legacy_entries_without_family_are_visible(self):
        self.memory._append("insight", title="Legacy", observation="o", mechanism="m", next_action="n")
        self.assertEqual(self.map()["families"]["unclassified"]["insight_count"], 1)

    def test_supported_requires_machine_receipt_and_live_evidence(self):
        insight, _, decision = self.decision()
        row = self.map()["families"]["representation"]
        self.assertEqual((row["state"], row["supported"], row["run_count"]), ("supported", 1, 1))
        (self.project.artifacts_dir / "candidate/oof_predictions.csv").write_text("tampered")
        row = self.map()["families"]["representation"]
        self.assertEqual(row["supported"], 0)
        self.assertEqual(row["state"], "active")
        with self.assertRaisesRegex(ValueError, "machine-evaluated"):
            record_macro_review(self.memory, review_payload(self.memory, supported_mechanisms=[self.claim(insight, decision)]))

    def test_declared_rejection_does_not_exhaust_family(self):
        insight = self.insight()
        self.memory.record_decision(insight_id=insight["id"], outcome="rejected", evidence=["control"],
            conclusion="Manual opinion", lesson="Unknown", next_action="Test", revisit_when="New evidence")
        row = self.map()["families"]["representation"]
        self.assertEqual((row["state"], row["rejected"]), ("untried", 0))

    def test_exhausted_requires_terminal_evidence_scope_stop_and_revisit(self):
        self.decision("rejected")
        row = self.map()["families"]["representation"]
        self.assertEqual((row["state"], row["rejected"]), ("exhausted", 1))
        self.insight()
        row = self.map()["families"]["representation"]
        self.assertEqual((row["state"], row["active"]), ("exhausted", 0))

    def test_missing_stop_or_revisit_prevents_exhaustion(self):
        self.decision("rejected")
        for kind, field in (("probe_plan", "stop_rule"), ("decision", "revisit_when")):
            data = json.loads(self.memory.path.read_text())
            original = copy.deepcopy(data)
            next(e for e in data["entries"] if e["kind"] == kind)[field] = ""
            self.memory.path.write_text(json.dumps(data))
            self.assertEqual(self.map()["families"]["representation"]["state"], "active")
            self.memory.path.write_text(json.dumps(original))

    def test_supported_family_with_open_direction_stays_active(self):
        self.decision()
        self.insight()
        row = self.map()["families"]["representation"]
        self.assertEqual((row["state"], row["supported"], row["active"]), ("supported", 1, 0))

    def test_new_plan_reopens_supported_insight(self):
        insight, _, _ = self.decision()
        self.memory.plan_probe(insight_id=insight["id"], control="control", intervention="Orthogonal extension",
            expected_observation="Fewer errors", alternative_explanation="Noise", decision_rule="Diagnostic only",
            budget="One probe", stop_rule="One probe")
        self.assertEqual(self.map()["families"]["representation"]["state"], "active")

    def test_new_proposal_reopens_supported_insight(self):
        insight, _, _ = self.decision()
        self.memory._append("proposal", insight_id=insight["id"], exp_id="followup", evidence_status="unrun_proposal")
        self.assertEqual(self.map()["families"]["representation"]["state"], "active")

    def test_comparison_groups_remain_separate(self):
        insight = self.insight()
        runs = [dict(exp_id=f"r{i}", evidence_status="verified_full", insight_id=insight["id"], comparison_group=f"split-{i}")
                for i in range(2)]
        row = compute_macro_hypothesis_map(self.memory, runs)["families"]["representation"]
        self.assertEqual(row["comparison_groups"], ["split-0", "split-1"])
        self.assertEqual(row["supported"], 0)

    def test_digest_and_both_schemas(self):
        from jsonschema import Draft202012Validator
        from pathlib import Path
        macro = self.map()
        self.assertEqual(macro, self.map())
        review = record_macro_review(self.memory, review_payload(self.memory))
        for name, value in (("macro_hypothesis_map", macro), ("macro_review", {
            k: v for k, v in review.items() if k not in {"id", "scope", "kind", "timestamp", "verified_run_ids", "stagnation_events", "evidence_status"}})):
            schema = json.loads((Path(__file__).resolve().parents[1] / "harness/schemas" / (name + ".v1.schema.json")).read_text())
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(value)
            validate_document(name, value)
        self.assertEqual(self.map()["map_digest"], macro["map_digest"])
        self.insight()
        self.assertNotEqual(self.map()["map_digest"], macro["map_digest"])

    def test_valid_review_claim_records_milestone_and_markdown(self):
        insight, _, decision = self.decision()
        record = record_macro_review(self.memory, review_payload(self.memory,
            supported_mechanisms=[self.claim(insight, decision)]))
        self.assertEqual(record["verified_run_ids"], ["candidate", "control"])
        self.assertFalse(retrospective_status(self.memory, self.memory.sync(), {})["due"])
        self.assertIn("highest_priority_untouched", self.memory.progress_path.read_text(encoding="utf-8"))

    def test_invalid_claim_links_and_stale_reviews_do_not_advance_milestone(self):
        insight, _, decision = self.decision()
        claim = self.claim(insight, decision)
        for key, value in (("decision_id", "invented"), ("insight_id", "invented"),
                           ("run_id", "control"), ("family_id", "target")):
            with self.assertRaises(ValueError):
                record_macro_review(self.memory, review_payload(self.memory, supported_mechanisms=[{**claim, key:value}]))
        with self.assertRaises(ValueError):
            record_macro_review(self.memory, review_payload(self.memory, rejected_mechanisms=[claim]))
        with self.assertRaisesRegex(ValueError, "Stale"):
            record_macro_review(self.memory, review_payload(self.memory, map_digest="old"))
        for override in (dict(task_id="other"), dict(run_count_at_review=0), dict(next_probe_rationale=" "),
                         dict(budget_limit=-1), dict(unrecognized="field")):
            with self.assertRaises(ValueError):
                record_macro_review(self.memory, review_payload(self.memory, **override))
        self.assertFalse(any(e["kind"] == "macro_review" for e in self.memory.entries()))

    def test_tampered_evaluation_receipt_cannot_support(self):
        _, _, decision = self.decision()
        from harness.decision_rules import policy_directory
        receipt = policy_directory(self.root) / (decision["evaluation_id"] + ".json")
        doc = json.loads(receipt.read_text())
        doc["gain"] = 999
        receipt.write_text(json.dumps(doc))
        # Avoid sync's independent evaluation integrity guard for this projection probe.
        runs = [dict(exp_id=rid, evidence_status="verified_full") for rid in ("candidate", "control")]
        self.assertEqual(compute_macro_hypothesis_map(self.memory, runs)["families"]["representation"]["supported"], 0)


if __name__ == "__main__":
    unittest.main()
