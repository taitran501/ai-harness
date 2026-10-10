"""Continuity contracts: restart, old history, novelty, evidence, and plateau."""
import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.agent_runtime import AgentRuntime
from harness.artifacts import digest, validate_run
from harness.diagnostics import DiagnosticEngine
from harness.research_memory import ResearchMemory
from harness.runner import UniversalRunner
from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import TreeJournal
from tests.evidence_fixtures import seal_fixture


def task_document():
    return {"task": {"name": "continuity-text", "modality": "text",
        "primary_metric": "macro_f1", "direction": "maximize",
        "data": {"id_column": "id", "train_path": "dataset/train.csv", "fold_column": "fold"},
        "validation": {"strategy": "kfold", "num_folds": 2},
        "inputs": [{"name": "text", "type": "text", "source_columns": ["text"]}],
        "subtasks": [{"id": "sentiment", "column": "target", "task_type": "binary_classification",
                      "classes": [0, 1], "primary_metric": "macro_f1"}]}}


def config(exp_id, c=1):
    return {"experiment": {"id": exp_id, "branch": "baseline", "parent_id": None,
        "problem_addressed": "Minority recall needs measurement on fixed validation folds.",
        "hypothesis": "Changing the loss penalty may improve minority recall on fixed folds.",
        "intervention": "Set the logistic regression penalty strength under fixed inputs."},
        "model": {"name": "tfidf_logistic_regression", "params": {"C": c}},
        "data": {"input_variants": ["text"], "selection_variant": "text"}}


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
        self.memory.record_decision(outcome="supported", conclusion="Measured full validation gain", evidence=["full"])
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
        for approved in (False, True):
            run = AgentRuntime(self.root).start_run("Scientific planning", auto_approve_all=approved)
            self.assertEqual(run.status, "ACTIVE")
            self.assertEqual(run.actions, [])
            self.assertIsNone(run.pending_approval_action)
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
        (self.root / ".ml-harness").mkdir(exist_ok=True)
        (self.root / ".ml-harness/EXPERIMENTS.md").write_text(
            "## Idea Graveyard\n| exp1 | approach | noise | measure | deferred augmentation |\n")
        self.assertIn("deferred augmentation", bridge.get_context_packet()["idea_graveyard"][0])

    def test_nlp_generated_config_executes_single_fold_and_full(self):
        (self.root / "dataset").mkdir()
        pd.DataFrame({"id": range(12), "text": ["bad negative awful", "good positive great"] * 6,
                      "target": [0, 1] * 6, "fold": [0] * 6 + [1] * 6}).to_csv(
                          self.root / "dataset/train.csv", index=False)
        bridge = AgentBridge(self.root)
        _, obs = bridge.execute_action("create_experiment", params={"exp_id": "nlp"})
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


if __name__ == "__main__":
    unittest.main()
