"""Insight-to-experiment trace, restart triage, and scientific interpretation contracts."""
import json
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from harness.agent_bridge import AgentBridge
from harness.phase_workflow import PhaseWorkflow, initialize_prompt_pack
from harness.project import attach_project
from harness.research_memory import ResearchMemory
from harness.task_spec import UniversalTaskSpec
from tests.evidence_fixtures import seal_fixture
from tests.test_research_continuity import config, task_document


class ResearchTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mlh-insight-trace-")
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
        self.memory.record_decision(insight_id=insight["id"], outcome="supported", evidence=["second"],
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
        with self.assertRaises(ValueError):
            self.memory.record_decision(insight_id=insight["id"], outcome="deferred", conclusion="Need evidence", next_action="Audit", lesson="No verified data")
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
        self.assertIn("research_frontier", bridge.render_context_markdown())

    def test_phase_revision_and_attached_skill_include_research_updates(self):
        attach_project(self.root, self.root / "configs/task.yaml")
        initialize_prompt_pack(self.root)
        workflow = PhaseWorkflow(self.root)
        self.memory = ResearchMemory(self.root, workflow.project.artifacts_dir, self.spec)
        revision = workflow.evidence_revision()
        self.insight()
        self.assertNotEqual(revision, workflow.evidence_revision())
        self.assertIn("research_frontier", workflow.render())
        skill = (self.root / ".agents/skills/ai-harness/SKILL.md").read_text()
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


if __name__ == "__main__":
    unittest.main()
