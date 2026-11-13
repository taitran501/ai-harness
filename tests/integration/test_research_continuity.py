"""Research continuity integration test suite consolidating restart, history, novelty, traces, and retrospective gate."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge, _research_continuity_hud_lines
from harness.artifacts import digest, validate_run
from harness.decision_rules import evaluate_decision_rule, register_decision_rule, reserve_comparison_slot
from harness.diagnostics import DiagnosticEngine
from harness.macro_research import (
    GateViolationError,
    enforce_retrospective,
    is_retrospective_due,
    load_memory,
    record_macro_review,
    retrospective_status,
)
from harness.phase_workflow import PhaseWorkflow, initialize_prompt_pack
from harness.project import HarnessProject, attach_project
from harness.research_memory import ResearchMemory
from harness.runner import UniversalRunner, build_parser
from harness.stagnation import StagnationController
from harness.submission_gate import SubmissionGate
from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import TreeJournal
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import (
    config,
    make_submission_evidence,
    review_payload,
    rule_document,
    text_task_document as task_document,
    workspace_with_verified_pair,
)


class ResearchContinuity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()))
        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.memory = ResearchMemory(self.root, self.artifacts, self.spec)

    def tearDown(self):
        self.temp.cleanup()

    def full(self, exp_id, score, c=1, **identity):
        folder = self.artifacts / exp_id
        folder.mkdir()
        (folder / "run_config.yaml").write_text(yaml.safe_dump(config(exp_id, c)))
        (folder / "diagnostics.json").write_text(json.dumps({"critical_bottlenecks": ["Zero minority recall"]}))
        seal_fixture(folder, self.spec, score, **identity)
        return folder

    def test_restart_search_reaches_history_outside_recent_hud(self):
        self.memory.record_decision(outcome="rejected", conclusion="Rare token augmentation amplified label noise")
        for i in range(22):
            self.memory.propose(config(f"e{i}", i + 1))
        self.memory.checkpoint(summary="Loss comparison remains inconclusive", next_action="Inspect minority slices",
                               open_questions=["Which errors are annotation noise?"])
        restarted = ResearchMemory(self.root, self.artifacts, self.spec)
        self.assertIn("label noise", restarted.search("augmentation")[0]["conclusion"])
        self.assertEqual(restarted.context()["latest_checkpoint"]["next_action"], "Inspect minority slices")
        self.assertEqual(restarted.context()["entry_count"], 24)
        self.assertLess(len(json.dumps(restarted.context())), 6500)

    def test_exact_repeat_changes_names_but_neighbors_remain_advisory(self):
        self.memory.propose(config("old"))
        repeat = config("renamed")
        repeat["experiment"]["hypothesis"] = "A completely different description of the same declared execution."
        self.assertFalse(self.memory.review(repeat)["allowed"])
        self.assertFalse(self.memory.review(config("numeric_alias", 1.0))["allowed"])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.memory.propose(repeat)
        neighbor = self.memory.review(config("new", 2))
        self.assertTrue(neighbor["allowed"])
        self.assertTrue(neighbor["related_approaches"])
        self.memory.propose(repeat, "Intentional replication to measure stochastic variance")
        self.assertTrue(self.memory.review(config("old"), for_execution=True)["allowed"])
        self.assertTrue(self.memory.review(config("old"), "Replication")["allowed"])

    def test_resolved_runtime_metadata_does_not_hide_completed_repeat(self):
        folder = self.full("old", .8)
        cfg = config("old")
        cfg["resolved_runtime"] = {"git_commit_hash": "example", "python_version": "3"}
        (folder / "run_config.yaml").write_text(yaml.safe_dump(cfg))
        seal_fixture(folder, self.spec, .8)
        self.assertFalse(self.memory.review(config("renamed"))["allowed"])

    def test_scope_isolation_and_live_evidence_revalidation(self):
        folder = self.full("full", .8)
        self.memory.record_decision(outcome="inconclusive", conclusion="Measured full validation gain", evidence=["full"])
        self.assertTrue(self.memory.context()["recent_decisions"][0]["evidence_currently_verified"])
        with self.assertRaises(ValueError):
            self.memory.record_decision(outcome="rejected", conclusion="No evidence", evidence=["missing"])
        (folder / "metrics.json").write_text("{}")
        self.assertEqual(self.memory.context()["verified_run_count"], 0)
        self.assertFalse(self.memory.context()["recent_decisions"][0]["evidence_currently_verified"])
        changed = task_document()
        changed["task"]["name"] = "other-task"
        other = ResearchMemory(self.root, self.artifacts, UniversalTaskSpec.from_dict(changed))
        self.assertEqual(other.context()["entry_count"], 0)

    def test_comparable_incumbent_plateau_ignores_failed_and_probe_runs(self):
        for i, score in enumerate([.7, .9, .85, .88, .9]):
            self.full(f"full{i}", score, i + 1)
        self.memory.record_execution("run_probe", "FAILED", "Implementation crashed")
        self.memory.record_decision(outcome="inconclusive", conclusion="Probe is insufficient evidence")
        ctx = self.memory.context(min_improvement=.002)
        self.assertEqual(ctx["consecutive_non_improvements"], 3)
        self.assertTrue(ctx["reassess_recommended"])
        self.full("different_split", .1, 8, split_sha256=digest("new split"))
        self.assertEqual(self.memory.context()["consecutive_non_improvements"], 0)

    def test_minimize_plateau_and_atomic_concurrent_proposals(self):
        doc = task_document()
        doc["task"].update(primary_metric="rmse", direction="minimize")
        doc["task"]["subtasks"][0].update(task_type="regression", primary_metric="rmse", direction="minimize")
        self.spec = UniversalTaskSpec.from_dict(doc)
        self.memory = ResearchMemory(self.root, self.artifacts, self.spec)
        for i, score in enumerate([1.0, .8, .85, .83, .8]):
            self.full(f"min{i}", score, i + 1)
        self.assertTrue(self.memory.context()["reassess_recommended"])
        def propose(i):
            try:
                self.memory.propose(config(f"parallel{i}", 100))
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sum(pool.map(propose, [1, 2])), 1)

    def test_bridge_checkpoint_recovery_duplicate_guard_and_no_auto_freeze(self):
        bridge = AgentBridge(self.root)
        session = bridge.start_session("Research continuity")
        _, obs = bridge.execute_action("record_research_checkpoint", params={
            "summary": "Control is fixed", "next_action": "Inspect minority slice"}, session_id=session.session_id)
        self.assertEqual(obs.status, "SUCCESS")
        restored = AgentBridge(self.root)
        self.assertEqual(restored.get_context_packet()["research_memory"]["latest_checkpoint"]["next_action"],
                         "Inspect minority slice")
        for _ in (False, True):
            run_bridge = AgentBridge(self.root)
            session2 = run_bridge.start_session("Scientific planning")
            self.assertEqual(session2.status, "ACTIVE")
            self.assertEqual(session2.actions, [])
            self.assertIsNone(session2.pending_approval_action)
        _, created = bridge.execute_action("create_experiment", params={"exp_id": "one"})
        _, blocked = bridge.execute_action("create_experiment", params={"exp_id": "two"})
        self.assertEqual(created.status, "SUCCESS")
        self.assertEqual(blocked.status, "FAILED")
        self.assertFalse((self.root / "configs/two.yaml").exists())
        _, repeated = bridge.execute_action("create_experiment", params={"exp_id": "two",
            "revisit_reason": "Independent replication"})
        self.assertEqual(repeated.status, "SUCCESS")
        self.assertEqual(yaml.safe_load((self.root / "configs/two.yaml").read_text())["research"]["revisit_reason"],
                         "Independent replication")

    def test_verified_diagnostics_reach_journal_and_numeric_aliases(self):
        folder = self.full("diagnostic", .8)
        journal = TreeJournal(artifacts_dir=self.artifacts, journal_path=self.artifacts / "journal.json",
                              task_spec=self.spec, auto_sync=True)
        self.assertEqual(journal.nodes[folder.name].diagnostics_summary["critical_bottlenecks"], ["Zero minority recall"])
        stamp = journal.nodes[folder.name].timestamp
        journal.sync_from_artifacts()
        self.assertEqual(journal.nodes[folder.name].timestamp, stamp)
        truth = pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 1]})
        preds = pd.DataFrame({"id": [1, 2, 3], "pred_sentiment": [0, 0, 0]})
        diag = DiagnosticEngine.generate_diagnostics(truth, preds, self.spec)
        self.assertTrue(diag["target_confusions"]["target"]["per_class"]["1"]["is_zero_recall"])
        self.assertTrue(diag["critical_bottlenecks"])

    def test_full_action_compares_incumbent_and_duplicate_never_spends_budget(self):
        self.full("base", .7)
        self.full("champion", .9, 2)
        bridge = AgentBridge(self.root)
        bridge.comp_state.state["gates"].update(baseline_exp_id="base", baseline_locked=True)
        bridge.comp_state.save()
        cfg = config("new_result", 3)
        (self.root / "configs/new.yaml").write_text(yaml.safe_dump(cfg))
        session = bridge.start_session("Improve incumbent")
        def completed(*args, **kwargs):
            self.full("new_result", .85, 3)
            return True, 0, "Completed fixture", .01
        with patch.object(bridge, "_run_subprocess_safe", side_effect=completed):
            session, obs = bridge.execute_action("run_full_validation", params={"config_path": "configs/new.yaml"},
                                                 session_id=session.session_id)
        self.assertEqual(obs.status, "SUCCESS")
        self.assertAlmostEqual(obs.data["incumbent_utility_gain"], -.05)
        self.assertEqual(session.consecutive_rejections, 1)
        duplicate = config("duplicate", 3)
        (self.root / "configs/duplicate.yaml").write_text(yaml.safe_dump(duplicate))
        used = session.compute_usage["experiments_used"]
        with patch.object(bridge, "_run_subprocess_safe") as runner:
            session, obs = bridge.execute_action("run_full_validation", params={"config_path": "configs/duplicate.yaml"},
                                                 session_id=session.session_id)
            runner.assert_not_called()
        self.assertEqual(obs.status, "FAILED")
        self.assertEqual(session.compute_usage["experiments_used"], used)
        unnamed = config("unused", 100)
        unnamed["experiment"].pop("id")
        _, written = bridge.execute_action("write_experiment_config", params={
            "path": "configs/implicit_id.yaml", "content": yaml.safe_dump(unnamed)})
        self.assertEqual(written.status, "SUCCESS")
        self.assertEqual(yaml.safe_load((self.root / "configs/implicit_id.yaml").read_text())["experiment"]["id"],
                         "implicit_id")

    def test_checkpoint_still_writable_after_freeze_and_legacy_notes_are_recovered(self):
        bridge = AgentBridge(self.root)
        bridge.comp_state.state["gates"]["freeze_locked"] = True
        bridge.comp_state.save()
        _, obs = bridge.execute_action("record_research_checkpoint", params={
            "summary": "Final candidate approved", "next_action": "Inspect final audit"})
        self.assertEqual(obs.status, "SUCCESS")
        (self.root / ".ai-harness").mkdir(exist_ok=True)
        (self.root / ".ai-harness/EXPERIMENTS.md").write_text(
            "## Idea Graveyard\n| exp1 | approach | noise | measure | deferred augmentation |\n")
        self.assertIn("deferred augmentation", bridge.get_context_packet()["idea_graveyard"][0])

    def test_nlp_generated_config_executes_single_fold_and_full(self):
        (self.root / "dataset").mkdir()
        pd.DataFrame({"id": range(12), "text": ["bad negative awful", "good positive great"] * 6,
                      "target": [0, 1] * 6, "fold": [0] * 6 + [1] * 6}).to_csv(
                          self.root / "dataset/train.csv", index=False)
        bridge = AgentBridge(self.root)
        _, obs = bridge.execute_action("create_experiment", params={"exp_id": "nlp",
            "source_type": "verified_baseline",
            "source_id": "sklearn_tfidf_logreg_canonical",
            "retrieval_summary": "Canonical baseline for the NLP classification task; anchors downstream iteration.",
            "verified_via": "local_registry"})
        self.assertEqual(obs.status, "SUCCESS")
        cfg = yaml.safe_load((self.root / "configs/nlp.yaml").read_text())
        args = SimpleNamespace(project_root=str(self.root), task_config=str(self.root / "configs/task.yaml"),
                               debug=True, fold=0, probe_samples=None, job_id=None)
        probe = UniversalRunner(copy.deepcopy(cfg), "configs/nlp.yaml", args)
        probe.run()
        self.assertEqual(probe.manifest["run_kind"], "probe")
        self.assertEqual(self.memory.context()["verified_run_count"], 0)
        args.debug, args.fold = False, None
        runner = UniversalRunner(copy.deepcopy(cfg), "configs/nlp.yaml", args)
        runner.run()
        validate_run(runner.exp_dir, task_spec=self.spec)
        self.assertEqual(self.memory.context()["verified_run_count"], 1)
        repeat = copy.deepcopy(cfg)
        repeat["experiment"]["id"] = "copy"
        self.assertFalse(self.memory.review(repeat)["allowed"])

    def test_multilabel_diagnostics_and_llm_label_prefix_is_not_input(self):
        doc = task_document()
        doc["task"]["subtasks"] = [{"id": "tags", "task_type": "multilabel_classification",
                                   "labels": ["a", "b"], "column_prefix": "gold_"}]
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(doc))
        spec = UniversalTaskSpec.from_dict(doc)
        frame = pd.DataFrame({"id": [1, 2], "text": ["first", "second"], "fold": [0, 1],
                              "gold_a": [1, 0], "gold_b": [0, 1]})
        preds = pd.DataFrame({"id": [1, 2], "pred_tags": ["[0, 0]", "[0, 0]"]})
        diag = DiagnosticEngine.generate_diagnostics(frame, preds, spec)
        self.assertTrue(diag["minority_labels"]["tags.b"]["is_zero_recall"])
        cfg = config("leaky")
        cfg["llm"] = {"prompt_template": "{text}; gold_a={gold_a}"}
        args = SimpleNamespace(project_root=str(self.root), task_config=str(self.root / "configs/task.yaml"),
                               debug=True, fold=0, probe_samples=None, job_id=None)
        runner = UniversalRunner(cfg, "config", args)
        with patch("harness.llm_engine.LLMEngine.generate_completion") as provider:
            with self.assertRaisesRegex(ValueError, "ground-truth"):
                runner._run_llm_prompting(frame, "test")
            provider.assert_not_called()


class ResearchTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-insight-trace-")
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()))
        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.memory = ResearchMemory(self.root, self.artifacts, self.spec)

    def tearDown(self):
        self.temp.cleanup()

    def insight(self, **kwargs):
        return self.memory.record_insight(title="Minority recall", observation="Minority errors persist",
            mechanism="Loss weighting could improve minority recall", next_action="Compare a fixed control",
            **kwargs)

    def plan(self, insight_id):
        return self.memory.plan_probe(insight_id=insight_id, control="Fixed incumbent and validation folds",
            intervention="Change only class weighting", expected_observation="Minority recall rises consistently",
            alternative_explanation="Annotation noise rather than class imbalance",
            decision_rule="Promote only if full metric improves and minority gain is stable across folds",
            budget="One fold smoke, then one authorized full validation", stop_rule="Stop on execution error or exhausted session budget")

    def full(self, exp_id, insight_id, score=.8, c=1):
        cfg = config(exp_id, c)
        if insight_id:
            cfg["research"] = {"insight_id": insight_id}
        folder = self.artifacts / exp_id
        folder.mkdir()
        (folder / "run_config.yaml").write_text(yaml.safe_dump(cfg))
        (folder / "diagnostics.json").write_text(json.dumps({"critical_bottlenecks": ["Minority recall"]}))
        seal_fixture(folder, self.spec, score)
        return folder

    def test_restart_recovers_parent_lesson_plan_experiment_and_derived_direction(self):
        insight = self.insight()
        plan = self.plan(insight["id"])
        cfg = config("weighted")
        cfg["research"] = {"insight_id": insight["id"]}
        self.memory.propose(cfg)
        self.full("weighted", insight["id"])
        self.assertEqual(self.memory.context()["research_frontier"][0]["status"], "needs_interpretation")
        self.memory.record_decision(insight_id=insight["id"], outcome="rejected", evidence=["weighted"],
            conclusion="No consistent gain under this weighting configuration", lesson="Weighting did not address this error slice",
            next_action="Audit annotation noise", revisit_when="A corrected label audit changes the error slice")
        child = self.insight(parent_insight_id=insight["id"])
        self.memory.checkpoint(summary="Weighting failed to address the slice; audit noise next",
                               next_action="Plan a discriminating annotation audit", insight_ids=[child["id"]])
        restarted = ResearchMemory(self.root, self.artifacts, self.spec)
        trace = restarted.trace(insight["id"])
        self.assertEqual(trace["state"]["status"], "rejected")
        self.assertEqual(trace["derived_insights"][0]["insight_id"], child["id"])
        proposal = next(event for event in trace["events"] if event["kind"] == "proposal")
        self.assertEqual(proposal["probe_plan_id"], plan["id"])
        self.assertEqual(trace["results"][0]["exp_id"], "weighted")
        ctx = restarted.context()
        self.assertEqual(ctx["latest_checkpoint"]["insight_ids"], [child["id"]])
        self.assertEqual(ctx["research_frontier"][0]["status"], "needs_probe_plan")
        matches = restarted.search("annotation noise")
        self.assertTrue(any(item.get("lesson") == "Weighting did not address this error slice" for item in matches))

    def test_linked_configuration_needs_plan_and_cannot_bypass_novelty_with_new_insight(self):
        insight = self.insight()
        cfg = config("first")
        cfg["research"] = {"insight_id": insight["id"]}
        with self.assertRaisesRegex(ValueError, "plan_research_probe"):
            self.memory.propose(cfg)
        self.plan(insight["id"])
        self.memory.propose(cfg)
        another = self.insight()
        self.plan(another["id"])
        repeated = config("renamed")
        repeated["research"] = {"insight_id": another["id"]}
        self.assertFalse(self.memory.review(repeated)["allowed"])
        self.assertTrue(self.memory.review(repeated, "Measure replicate variance")["allowed"])

    def test_multiple_decisions_do_not_resurface_old_runs_and_tampering_requires_review(self):
        insight = self.insight()
        self.plan(insight["id"])
        first = self.full("first", insight["id"])
        self.memory.record_decision(insight_id=insight["id"], outcome="inconclusive", evidence=["first"],
            conclusion="Recall improved without full gain", lesson="Need fold stability", next_action="Measure another seed")
        second = self.full("second", insight["id"], c=2)
        self.memory.record_decision(insight_id=insight["id"], outcome="inconclusive", evidence=["second"],
            conclusion="Second configuration measured a gain", lesson="Separate seed variance from intervention", next_action="Review candidate evidence")
        self.assertEqual(self.memory.context()["unreviewed_result_count"], 0)
        self.assertEqual(self.memory.trace(insight["id"])["state"]["unreviewed_result_count"], 0)
        (second / "metrics.json").write_text("{}")
        state = self.memory.context()["research_frontier"][0]
        self.assertEqual(state["status"], "needs_evidence_review")
        self.assertEqual(state["missing_evidence"], ["second"])
        self.assertTrue(first.is_dir())

    def test_history_pagination_and_unreviewed_direct_runs_survive_recent_window(self):
        insight = self.insight()
        for _ in range(12):
            self.plan(insight["id"])
        page = self.memory.trace(insight["id"], limit=4)
        self.assertEqual(page["event_count"], 13)
        self.assertEqual(page["next_offset"], 4)
        page2 = self.memory.trace(insight["id"], offset=4, limit=30)
        self.assertEqual(len(page2["events"]), 9)
        self.assertIsNone(page2["next_offset"])
        for i in range(9):
            self.full(f"direct{i}", None, c=i + 1)
        self.memory.record_decision(outcome="inconclusive", conclusion="Reviewed recent results", evidence=[f"direct{i}" for i in range(1, 9)])
        ctx = self.memory.context()
        self.assertEqual(ctx["unreviewed_results"][0]["exp_id"], "direct0")
        self.assertEqual(ctx["unreviewed_result_count"], 1)

    def test_declared_notes_require_scope_and_do_not_fabricate_verified_evidence(self):
        insight = self.insight()
        decision = self.memory.record_decision(insight_id=insight["id"], outcome="deferred", conclusion="Need evidence", next_action="Audit", lesson="No verified data")
        self.assertTrue(decision["revisit_when"])
        with self.assertRaises(ValueError):
            self.insight(evidence=["missing-run"])
        changed = task_document()
        changed["task"]["name"] = "another task"
        other = ResearchMemory(self.root, self.artifacts, UniversalTaskSpec.from_dict(changed))
        with self.assertRaisesRegex(ValueError, "current task/scorer scope"):
            other.trace(insight["id"])
        with self.assertRaises(ValueError):
            self.memory.checkpoint(summary="Resume", next_action="Audit", insight_ids=["unknown"])
        self.assertFalse(self.memory.trace(insight["id"])["insight"]["evidence_currently_verified"])
        self.assertEqual(self.memory.context()["verified_run_count"], 0)

    def test_bridge_preserves_links_on_repeat_and_records_failed_smoke_with_experiment(self):
        bridge = AgentBridge(self.root)
        _, recorded = bridge.execute_action("record_research_insight", params={"title": "Slice", "observation": "Recall gap",
            "mechanism": "Weighting may help", "next_action": "Plan a probe"})
        self.assertEqual(recorded.status, "SUCCESS")
        insight_id = recorded.data["id"]
        plan = self.plan(insight_id)
        params = {key: value for key, value in plan.items() if key in {
            "insight_id", "control", "intervention", "expected_observation", "alternative_explanation", "decision_rule", "budget", "stop_rule"}}
        _, planned = bridge.execute_action("plan_research_probe", params=params)
        self.assertEqual(planned.status, "SUCCESS")
        _, created = bridge.execute_action("create_experiment", params={"exp_id": "first", "insight_id": insight_id})
        self.assertEqual(created.status, "SUCCESS", created.summary)
        _, repeated = bridge.execute_action("create_experiment", params={"exp_id": "replicate", "insight_id": insight_id,
                                             "revisit_reason": "Measure variance on a repeated declared configuration"})
        self.assertEqual(repeated.status, "SUCCESS", repeated.summary)
        cfg = yaml.safe_load((self.root / "configs/replicate.yaml").read_text())
        self.assertEqual(cfg["research"]["insight_id"], insight_id)
        self.assertTrue(cfg["research"]["revisit_reason"])
        with patch.object(bridge, "_run_subprocess_safe", return_value=(False, 1, "Implementation crashed", .01)):
            _, failed = bridge.execute_action("run_smoke_test", params={"config_path": "configs/replicate.yaml"})
        self.assertEqual(failed.status, "FAILED")
        _, traced = bridge.execute_action("trace_research_insight", params={"insight_id": insight_id})
        self.assertEqual(traced.status, "SUCCESS")
        self.assertEqual(traced.data["state"]["status"], "needs_diagnosis")
        self.assertEqual(next(event for event in traced.data["events"] if event["kind"] == "execution")["exp_id"], "replicate")
        self.assertFalse(bridge.comp_state.state["gates"].get("freeze_locked"))
        hud = bridge.render_context_markdown()
        self.assertIn("frontier:", hud)
        self.assertIn(insight_id[:8], hud)
        self.assertNotIn('"research_frontier"', hud)

    def test_phase_revision_and_attached_skill_include_research_updates(self):
        attach_project(self.root, self.root / "configs/task.yaml")
        initialize_prompt_pack(self.root)
        workflow = PhaseWorkflow(self.root)
        self.memory = ResearchMemory(self.root, workflow.project.artifacts_dir, self.spec)
        revision = workflow.evidence_revision()
        self.insight()
        self.assertNotEqual(revision, workflow.evidence_revision())
        self.assertIn("research_frontier", workflow.render())
        skill = (self.root / ".agents/skills/harness/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("trace_research_insight", skill)
        self.assertIn("plan_research_probe", skill)

    def test_rewritten_unrun_config_does_not_transfer_execution_or_sealed_result(self):
        first, second = self.insight(), self.insight()
        for insight in (first, second):
            self.plan(insight["id"])
        cfg = config("shared")
        cfg["research"] = {"insight_id": first["id"]}
        self.memory.propose(cfg)
        self.memory.record_execution("run_smoke_test", "FAILED", "First implementation crashed",
                                     "shared", insight_id=first["id"])
        cfg["research"]["insight_id"] = second["id"]
        self.memory.propose(cfg, "Reassign an unrun configuration to the refined insight")
        self.full("shared", second["id"])
        old, new = self.memory.trace(first["id"]), self.memory.trace(second["id"])
        self.assertEqual(old["results"], [])
        self.assertEqual(old["state"]["status"], "needs_diagnosis")
        self.assertEqual(new["state"]["status"], "needs_interpretation")
        self.assertFalse(any(event["kind"] == "execution" for event in new["events"]))

    def test_frontier_is_bounded_when_history_contains_many_long_insights(self):
        for i in range(15):
            self.memory.record_insight(title=f"Direction {i}", observation="o" * 4000,
                mechanism="m" * 4000, next_action="a" * 4000)
        ctx = self.memory.context()
        self.assertEqual(ctx["open_insight_count"], 15)
        self.assertEqual(len(ctx["research_frontier"]), 3)
        self.assertLess(len(json.dumps(ctx)), 6500)
        self.assertEqual(len(self.memory.search("Direction", limit=30)), 15)

    def test_markdown_progress_tracks_lessons_and_live_evidence_without_duplicate_events(self):
        insight = self.insight()
        self.plan(insight["id"])
        run = self.full("written", insight["id"])
        self.memory.record_decision(insight_id=insight["id"], outcome="inconclusive", evidence=["written"],
            conclusion="Needs a realistic task", lesson="Synthetic validation cannot measure reasoning quality",
            next_action="Define a bounded actual-agent evaluation")
        self.memory.checkpoint(summary="Notes linked", next_action="Plan actual-agent evaluation", insight_ids=[insight["id"]])
        report = self.memory.progress_path.read_text(encoding="utf-8")
        self.assertIn("Synthetic validation cannot measure reasoning quality", report)
        self.assertIn("Plan actual-agent evaluation", report)
        self.assertIn("evidence currently verified=true", report)
        original = self.memory.path.read_bytes()
        ctx = self.memory.context()
        self.assertEqual(ctx["progress_path"], str(self.memory.progress_path))
        self.assertEqual(self.memory.path.read_bytes(), original)
        self.assertEqual(self.memory.progress_path.read_text(encoding="utf-8"), report)
        (run / "metrics.json").write_text("{}")
        self.memory.context()
        invalid = self.memory.progress_path.read_text(encoding="utf-8")
        self.assertIn("needs\\_evidence\\_review", invalid)
        self.assertIn("evidence currently verified=false", invalid)
        self.assertNotIn("verified_full_result", invalid)

    def test_progress_preserves_user_owned_file_and_notes_survive_report_write_failure(self):
        self.memory.progress_path.write_text("# My manual notes\nDo not replace\n")
        self.insight()
        self.assertEqual(self.memory.progress_path.read_text(), "# My manual notes\nDo not replace\n")
        self.assertIn("user-owned", self.memory.context()["progress_warning"])
        self.memory.progress_path.unlink()
        with patch("harness.research_memory.atomic_write_text", side_effect=OSError("File locked")):
            self.insight()
            ctx = self.memory.context()
            self.assertIsNone(ctx["progress_path"])
            self.assertIn("File locked", ctx["progress_warning"])
        self.assertEqual(len(self.memory.entries()), 2)
        self.assertIsNotNone(self.memory.context()["progress_path"])

    def test_concurrent_research_notes_all_appear_in_generated_journal(self):
        with ThreadPoolExecutor(3) as pool:
            insights = list(pool.map(lambda _: self.insight(), range(9)))
        report = self.memory.progress_path.read_text(encoding="utf-8")
        for insight in insights:
            self.assertIn(insight["id"], report)
        self.assertEqual(report.count("— insight `"), 9)
        self.assertEqual(len(self.memory.entries()), 9)


class RetrospectiveGateTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.project = HarnessProject.discover(self.root)
        self.memory = load_memory(self.project)

    def add_runs(self, count=4):
        for i in range(count):
            make_submission_evidence(self.root, f"extra{i}")

    def cfg(self, name="next", **research):
        cfg = dict(experiment=dict(id=name), validation=dict(partition="oof"), research=research)
        path = self.root / "configs" / (name + ".yaml")
        path.write_text(yaml.safe_dump(cfg))
        return cfg, path

    def pause(self):
        for i in range(3):
            registration = register_decision_rule(rule_document(f"fail-family-{i}"), project=self.project, task_spec=self.memory.spec)
            cfg, _ = self.cfg(f"failed{i}", decision_rule_id=registration["rule_id"])
            reserve_comparison_slot(self.project, self.memory.spec, cfg)
            make_submission_evidence(self.root, f"failed{i}", labels=[0]*40, cfg=cfg)
            evaluate_decision_rule(registration["rule"], dict(kind="full_run", id=f"failed{i}", partition="oof"),
                project=self.project, task_spec=self.memory.spec, registration=registration)
        states = StagnationController(self.project).reconcile_results()
        return states, next(iter(states.values()))

    def test_exact_five_results_count_not_actions(self):
        self.add_runs(3)
        for _ in range(6):
            self.memory.record_execution("run_eda", "SUCCESS", "EDA findings")
        self.assertFalse(is_retrospective_due(self.root, self.memory.sync(), {})[0])
        make_submission_evidence(self.root, "fifth")
        self.assertTrue(is_retrospective_due(self.root, self.memory.sync(), {})[0])
        with self.assertRaisesRegex(GateViolationError, "RETROSPECTIVE_REQUIRED"):
            enforce_retrospective(self.root)

    def test_probe_failed_and_duplicate_records_do_not_advance(self):
        self.add_runs(3)
        runs = self.memory.sync()
        runs.extend([dict(exp_id="smoke", evidence_status="verified_full", run_kind="probe"),
                     dict(exp_id="failed", evidence_status="verified_full", status="FAILED"),
                     dict(exp_id="eda", evidence_status="declared"), runs[0]])
        status = retrospective_status(self.memory, runs, {})
        self.assertFalse(status["due"])
        self.assertEqual(status["runs_since_review"], 4)
        for rid, key, value in (("smoke", "run_kind", "probe"), ("failed", "status", "FAILED")):
            path = make_submission_evidence(self.root, rid) / "artifact_manifest.json"
            manifest = json.loads(path.read_text())
            manifest[key] = value
            path.write_text(json.dumps(manifest))
        self.assertEqual(len(self.memory.sync()), 4)

    def test_review_persists_across_restart_and_resets_only_covered_ids(self):
        self.add_runs()
        record = record_macro_review(self.memory, review_payload(self.memory))
        fresh = load_memory(self.root)
        self.assertFalse(is_retrospective_due(self.root, fresh.sync(), {})[0])
        self.assertEqual(retrospective_status(fresh, fresh.sync(), {})["last_review_id"], record["review_id"])
        (self.project.artifacts_dir / "control/oof_predictions.csv").write_text("tamper")
        for i in range(5):
            make_submission_evidence(self.root, f"new{i}")
        self.assertTrue(is_retrospective_due(self.root, fresh.sync(), {})[0])
        self.assertEqual(retrospective_status(fresh, fresh.sync(), {})["runs_since_review"], 5)

    def test_stagnation_requires_review_before_cadence_and_review_does_not_resume(self):
        states, _ = self.pause()
        self.assertEqual(len(self.memory.sync()), 4)
        due, reason = is_retrospective_due(self.root, self.memory.sync(), states)
        self.assertTrue(due)
        self.assertIn("stagnation", reason)
        record_macro_review(self.memory, review_payload(self.memory))
        self.assertFalse(is_retrospective_due(self.root, self.memory.sync(), states)[0])
        self.assertIsNotNone(StagnationController(self.project).check_action("dispatch"))
        changed = {key: {**state, "generation":state["generation"] + 1} for key, state in states.items()}
        self.assertTrue(is_retrospective_due(self.root, self.memory.sync(), changed)[0])

    def test_unregistered_recovery_flag_cannot_bypass(self):
        self.add_runs()
        for research in ({}, dict(recovery_probe_id="invented", recovery_probe=True)):
            cfg, _ = self.cfg(**research)
            with self.assertRaises(GateViolationError):
                enforce_retrospective(self.root, cfg=cfg, recovery_probe=True)

    def test_registered_recovery_is_bounded_and_never_authorizes_submission(self):
        _, state = self.pause()
        registration = register_decision_rule(rule_document("recovery"), project=self.project, task_spec=self.memory.spec)
        controller = StagnationController(self.project)
        probe = controller.register_recovery_probe(comparison_identity_sha256=state["comparison_identity_sha256"],
            rule_id=registration["rule_id"], diagnosis="Systematic error slice", mechanism="Orthogonal signal",
            stop_rule="One result", max_runs=1)
        cfg, _ = self.cfg("recovery", recovery_probe_id=probe["id"], decision_rule_id=registration["rule_id"])
        enforce_retrospective(self.root, cfg=cfg, recovery_probe=True)
        enforce_retrospective(self.root, cfg=cfg, recovery_probe=True)
        with self.assertRaises(GateViolationError):
            enforce_retrospective(self.root, cfg={**cfg, "experiment":{"id":"extra"}}, recovery_probe=True)
        with self.assertRaises(GateViolationError):
            enforce_retrospective(self.root, cfg=cfg, recovery_probe=False)
        with self.assertRaises(GateViolationError):
            enforce_retrospective(self.root, cfg={**cfg, "research":{**cfg["research"], "decision_rule_id":"wrong"}}, recovery_probe=True)

    def test_direct_runner_blocks_before_training_but_smoke_remains_available(self):
        self.add_runs()
        cfg, path = self.cfg()
        args = build_parser().parse_args(["--config", str(path), "--project-root", str(self.root)])
        runner = UniversalRunner(cfg, str(path), args)
        with mock.patch("harness.runner.GroundingGate.verify_config", return_value=(None, None, {})), \
             mock.patch("harness.runner.GroundingGate.write_receipt"), \
             mock.patch.object(runner, "_get_model_factory", side_effect=RuntimeError("reached training")) as load:
            with self.assertRaises(GateViolationError):
                runner.run()
            load.assert_not_called()
        cfg, path = self.cfg("smoke")
        args = build_parser().parse_args(["--config", str(path), "--project-root", str(self.root), "--debug", "--fold", "0"])
        runner = UniversalRunner(cfg, str(path), args)
        with mock.patch("harness.runner.GroundingGate.verify_config", return_value=(None, None, {})), \
             mock.patch("harness.runner.GroundingGate.write_receipt"), \
             mock.patch.object(runner, "_load_training_data", side_effect=RuntimeError("reached training")):
            with self.assertRaisesRegex(RuntimeError, "reached training"):
                runner.run()

    def test_cloud_jobs_block_before_provider_or_subprocess(self):
        self.add_runs()
        from harness.cloud_job_runner import CloudJobManager
        _, path = self.cfg()
        manager = CloudJobManager(self.root)
        with mock.patch("harness.cloud_policy.provider_preflight") as provider, \
             mock.patch("harness.cloud_job_runner.subprocess.Popen") as popen:
            for backend in ("local", "kaggle", "modal"):
                with self.assertRaises(GateViolationError):
                    manager.start_job(backend, str(path))
            provider.assert_not_called()
            popen.assert_not_called()
        self.assertFalse(manager._jobs)
        script = self.root / "script.py"
        script.write_text("print('ordinary script')")
        with self.assertRaises(GateViolationError):
            manager.start_job("local", str(script), extra_args=["--debug"])
        with self.assertRaisesRegex(ValueError, "locally"):
            manager.start_job("kaggle", str(path), extra_args=["--debug"])

    def test_remote_capsule_dispatch_cannot_bypass(self):
        self.add_runs()
        from harness.decision_rules import prepare_remote_research
        _, path = self.cfg()
        with self.assertRaises(GateViolationError):
            prepare_remote_research(self.project, path)

    def test_submission_gate_blocks_before_payload_or_outbound_work(self):
        self.add_runs()
        gate = SubmissionGate(self.root)
        with mock.patch.object(gate, "_resolve_payload") as payload:
            with self.assertRaises(GateViolationError):
                gate.evaluate("comp", "unused.csv", user_approved=True)
            payload.assert_not_called()

    def test_bridge_map_status_review_and_context_after_restart(self):
        self.add_runs()
        bridge = AgentBridge(self.root)
        _, obs = bridge.execute_action("get_macro_hypothesis_map", {})
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        _, status = bridge.execute_action("check_retrospective_status", {})
        self.assertTrue(status.data["due"])
        context = bridge._research_memory().context()
        self.assertIn("RETROSPECTIVE REVIEW DUE", "\n".join(_research_continuity_hud_lines(context)))
        _, obs = bridge.execute_action("record_macro_review", review_payload(self.memory))
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertFalse(AgentBridge(self.root).execute_action("check_retrospective_status", {})[1].data["due"])

    def test_cli_flat_flags_and_json_payload_record_reviews(self):
        from tools.agent_bridge_cli import main
        for payload_as_json in (False, True):
            payload = review_payload(self.memory)
            payload.pop("session_id")
            argv = ["--project-root", str(self.root), "act", "--action", "record_macro_review"]
            if payload_as_json:
                argv += ["--params", json.dumps(payload)]
            else:
                for key, value in payload.items():
                    argv += ["--" + key.replace("_", "-"), json.dumps(value) if isinstance(value, (list, dict)) else str(value)]
            out = io.StringIO()
            with mock.patch("harness.toolchain.ensure_pinned_toolchain", return_value=None), contextlib.redirect_stdout(out):
                code = main(argv)
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(json.loads(out.getvalue())["observation"]["status"], "SUCCESS", out.getvalue())

    def test_session_start_cli_elevates_due_macro_map(self):
        from tools.agent_bridge_cli import main
        self.add_runs()
        out = io.StringIO()
        with mock.patch("harness.toolchain.ensure_pinned_toolchain", return_value=None), contextlib.redirect_stdout(out):
            main(["--project-root", str(self.root), "start-session", "--goal", "Review campaign"])
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["retrospective_status"]["due"])
        self.assertIn("target", payload["macro_hypothesis_map"]["families"])

    def test_bridge_blocks_costly_actions_and_allows_research_actions(self):
        self.add_runs()
        bridge = AgentBridge(self.root)
        _, path = self.cfg()
        session = bridge.start_session(goal="Review campaign")
        for action in ("run_full_validation", "prepare_submission"):
            with self.assertRaises(GateViolationError):
                bridge._check_budget_and_stop_guards(session, action, {"config_path":str(path)})
        for action in ("record_macro_review", "get_macro_hypothesis_map", "check_retrospective_status", "run_smoke_test", "run_eda"):
            self.assertIsNone(bridge._check_budget_and_stop_guards(session, action, {}))


if __name__ == "__main__":
    unittest.main()
