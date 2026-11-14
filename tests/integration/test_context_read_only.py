"""Context is an observation boundary, including cold construction and CLI routing."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from harness.agent_bridge import AgentBridge
from harness.io_utils import atomic_write_json
from harness.project import HarnessProject
from harness.state_store import StateIntegrityError
from harness.tree_journal import Node
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import SOURCE_ROOT, make_workspace, workspace_with_verified_pair, make_submission_evidence, rule_document


def inventory(root):
    return {p.relative_to(root).as_posix(): (p.is_dir(), p.stat().st_mtime_ns,
            None if p.is_dir() else p.read_bytes()) for p in [root, *root.rglob("*")]}


class ContextReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, _ = make_workspace(root_path=Path(self.tmp.name))
        self.artifacts = HarnessProject.discover(self.root).artifacts_dir

    def assert_python_reads(self, *, writable=False):
        before = inventory(self.root)
        with ExitStack() as stack:
            forbidden = []
            for target in ("harness.competition_state.CompetitionStateMachine.run_data_readiness_audit",
                           "harness.decision_rules.evaluate_decision_rule",
                           "harness.stagnation.StagnationController.reconcile_results",
                           "harness.research_memory.ResearchMemory._publish_progress",
                           "harness.state_store.workspace_lock", "subprocess.Popen", "threading.Thread.start",
                           "harness.kaggle_compat.KaggleClient"):
                forbidden.append(stack.enter_context(patch(target, side_effect=AssertionError(f"Context called {target}"))))
            bridge = writable or AgentBridge(self.root, read_only=True)
            packet = bridge.get_context_packet()
            markdown = bridge.render_context_markdown()
            for operation in forbidden:
                operation.assert_not_called()
        self.assertEqual(before, inventory(self.root))
        self.assertIn("refresh_required", packet)
        self.assertLessEqual(len(markdown), 4000)
        return packet

    def cli(self, command="context", fmt="json", *, env_extra=None):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONPATH": str(SOURCE_ROOT), "HARNESS_FRAMEWORK_DIR": str(SOURCE_ROOT)}
        env.pop("HARNESS_TASK_CONFIG", None)
        env.pop("HARNESS_PROJECT_ROOT", None)
        env.pop("AI_HARNESS_PIN_DEPTH", None)
        if env_extra:
            env.update(env_extra)
        args = [sys.executable, "-B", "-m", "tools.agent_bridge_cli", "--project-root", str(self.root), command]
        if command == "context":
            args += ["--format", fmt]
        return subprocess.run(args, cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)

    def test_cold_construction_python_and_cli_are_read_only(self):
        packet = self.assert_python_reads()
        self.assertTrue(packet["refresh_required"])
        for fmt in ("json", "markdown"):
            before = inventory(self.root)
            result = self.cli(fmt=fmt)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, inventory(self.root))
            self.assertIn("refresh-context", result.stdout)

    def test_writable_bridge_context_uses_fresh_read_only_dependencies(self):
        bridge = AgentBridge(self.root)
        self.assert_python_reads(writable=bridge)

    def test_unindexed_runs_are_unpromoted_until_explicit_refresh(self):
        bridge = AgentBridge(self.root)
        bridge._new_journal(auto_sync=False).save()
        seal_fixture(self.artifacts / "new_run", bridge.task_spec)
        packet = self.assert_python_reads()
        self.assertIn({"component": "journal", "reason": "unindexed_artifacts",
                       "command": "ai-harness-agent --project-root . refresh-context"}, packet["refresh_required"])
        self.assertIsNone(packet["candidate_pool_summary"]["overall_best"])
        result = bridge.refresh_context_state()
        self.assertTrue(result["success"], result)
        for path in ("journal.json", "DATA_READINESS_REPORT.json", "RESEARCH_PROGRESS.md"):
            self.assertTrue((self.artifacts / path).is_file())
        packet = self.assert_python_reads()
        self.assertEqual(packet["candidate_pool_summary"]["overall_best"]["id"], "new_run")

    def test_orphans_and_tampered_artifacts_are_demoted_in_memory_only(self):
        bridge = AgentBridge(self.root)
        folder = self.artifacts / "tampered"
        seal_fixture(folder, bridge.task_spec)
        (folder / "metrics.json").write_text("{}")
        atomic_write_json(self.artifacts / "journal.json", {"nodes": [
            asdict(Node(id="tampered", verified=True, eligible=True, artifact_dir=str(folder))),
            asdict(Node(id="orphan", verified=True, eligible=True))]})
        packet = self.assert_python_reads()
        self.assertIsNone(packet["candidate_pool_summary"]["overall_best"])
        self.assertIn("invalid_or_orphan_nodes", [r["reason"] for r in packet["refresh_required"]])

    def test_missing_corrupt_reports_and_user_owned_progress_are_preserved(self):
        AgentBridge(self.root)
        report = self.artifacts / "DATA_READINESS_REPORT.json"
        report.write_text("{")
        progress = self.artifacts / "RESEARCH_PROGRESS.md"
        progress.write_text("My own research notes")
        packet = self.assert_python_reads()
        self.assertEqual(packet["research_memory"]["progress_status"], "user_owned")
        self.assertIn("missing_or_corrupt_report", [r["reason"] for r in packet["refresh_required"]])

    def test_generated_stale_progress_is_reported_without_replacement(self):
        bridge = AgentBridge(self.root)
        self.assertTrue(bridge.refresh_context_state()["success"])
        progress = self.artifacts / "RESEARCH_PROGRESS.md"
        progress.write_text(progress.read_text() + "\nStale generated text")
        packet = self.assert_python_reads()
        self.assertEqual(packet["research_memory"]["progress_status"], "stale")

    def test_corrupt_authoritative_state_is_visible_and_unchanged(self):
        AgentBridge(self.root)
        for filename, raw in (("competition_state.json", "{"), ("journal.json", "null"),
                              ("research_memory.json", '{"entries":"bad"}'),
                              ("submissions/registry.json", "{")):
            target = self.artifacts / filename
            previous = target.read_bytes() if target.exists() else None
            target.write_text(raw)
            before = inventory(self.root)
            with self.assertRaises(StateIntegrityError):
                AgentBridge(self.root, read_only=True).get_context_packet()
            self.assertEqual(before, inventory(self.root))
            result = self.cli()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, inventory(self.root))
            if previous is None:
                target.unlink()
            else:
                target.write_bytes(previous)

    def test_pending_decision_evaluations_wait_for_refresh(self):
        from harness.decision_rules import register_decision_rule, reserve_comparison_slot
        root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, root)
        self.root = root
        bridge = AgentBridge(root)
        self.artifacts = bridge.artifacts_dir
        registration = register_decision_rule(rule_document(), project=root, task_spec=bridge.task_spec, plugin=bridge.task_plugin)
        cfg = {"experiment": {"id": "candidate"}, "research": {"decision_rule_id": registration["rule_id"]}}
        reserve_comparison_slot(root, bridge.task_spec, cfg, plugin=bridge.task_plugin)
        make_submission_evidence(root, "candidate", cfg=cfg)
        packet = self.assert_python_reads()
        self.assertIn("pending_decision_evaluation", [r["reason"] for r in packet["refresh_required"]])
        self.assertFalse((self.artifacts / "policy_evidence/evaluations.json").exists())
        result = bridge.refresh_context_state()
        self.assertTrue(result["success"], result)
        self.assertTrue((self.artifacts / "policy_evidence/evaluations.json").is_file())
        self.assert_python_reads()

    def test_cli_toolchain_reexecution_remains_read_only(self):
        framework = self.root / "framework"
        for package in ("harness", "tools"):
            shutil.copytree(SOURCE_ROOT / package, framework / package,
                            ignore=shutil.ignore_patterns("__pycache__", "web"))
        before = inventory(self.root)
        result = self.cli(env_extra={"HARNESS_FRAMEWORK_DIR": str(framework)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("re-executing", result.stderr)
        self.assertEqual(before, inventory(self.root))
        self.assertIn("refresh_required", json.loads(result.stdout))

    def test_cli_refresh_produces_state_then_context_reads_it(self):
        result = self.cli(command="refresh-context")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["success"])
        before = inventory(self.root)
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, inventory(self.root))
