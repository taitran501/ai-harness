"""Consolidated integration tests for HUD ergonomics, developer ergonomics, and strategy cards.

Consolidates:
- tests/agent_bridge/test_hud_ergonomics.py (all 3 classes)
- tests/test_ergonomics_and_resilience.py (ErgonomicsAndResilienceTests)
- tests/test_strategy_cards.py (TestStrategyCardsAndSpawner)
"""

from __future__ import annotations

import json
import shutil
import socketserver
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from harness.agent_bridge import (
    HANDLERS,
    _ACTION_REGISTRY,
    AgentBridge,
    _research_continuity_hud_lines,
    typed_actions,
)
from harness.artifacts import contract_identity, repair_run, repair_run_dir, run_repair
from harness.competition_state import CompetitionStateMachine
from harness.evaluation import validate_run
from harness.hygiene import scan_workspace_hygiene
from harness.io_utils import atomic_write_json
from harness.kaggle_compat import KaggleClient, KaggleLeaderboardTracker
from harness.kaggle_runtime import find_dataset_mount
from harness.models import available_model_names
from harness.project import attach_project
from harness.repair import repair_folder
from harness.repair import repair_run as repair_run_alias
from harness.research_memory import ResearchMemory
from harness.status_report import collect_status
from harness.task_spec import load_task_spec
from tests.fixtures import make_workspace
from tools.dashboard_server import (
    DEFAULT_STRATEGY_CARDS,
    DashboardHandler,
    get_next_experiment_id,
)


# ===========================================================================
# 1. Hygiene Allow-List + Run-Output Aggregation
# ===========================================================================

class TestHygieneAllowListAndAggregation(unittest.TestCase):
    def test_standard_dirs_not_flagged(self):
        root, _ = make_workspace()
        try:
            for name in ("dataset", "scratch", "splits", "__pycache__", "plugin"):
                (root / name).mkdir(exist_ok=True)
                (root / name / "a.txt").write_text("x", encoding="utf-8")
            res = scan_workspace_hygiene(root)
            flagged = {f["path"] for f in res["findings"]}
            self.assertFalse(
                flagged,
                f"standard workspace dirs must not be hygiene noise: {flagged}",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_output_dirs_aggregate_into_one_advisory(self):
        root, _ = make_workspace()
        try:
            names = [
                "kaggle_sub_exp010_apex",
                "kaggle_sub_exp012_fov_apex",
                "kaggle_sub_output",
                "kaggle_submission_exp009",
                "kernel_output_exp001",
                "kernel_output_07f1",
                "kaggle_results",
            ]
            for name in names:
                (root / name).mkdir()
                (root / name / "a.txt").write_text("x", encoding="utf-8")
            (root / "kaggle_sub_exp020_sealed").mkdir()
            (root / "kaggle_sub_exp020_sealed" / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            res = scan_workspace_hygiene(root)
            by_kind = {f["kind"]: f for f in res["findings"]}
            self.assertEqual(set(by_kind), {"bypassed_run_output", "stale_run_output"})
            self.assertIn("7 dirs", by_kind["bypassed_run_output"]["path"])
            self.assertIn("artifact manifest", by_kind["bypassed_run_output"]["detail"])
            self.assertIn("1 dirs", by_kind["stale_run_output"]["path"])
            self.assertIn("kaggle_sub_exp020_sealed", by_kind["stale_run_output"]["detail"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_few_run_output_dirs_stay_individual(self):
        root, _ = make_workspace()
        try:
            for name in ("kaggle_sub_exp001", "kernel_output"):
                (root / name).mkdir()
                (root / name / "a.txt").write_text("x", encoding="utf-8")
            res = scan_workspace_hygiene(root)
            kinds = [f["kind"] for f in res["findings"]]
            self.assertEqual(kinds.count("unknown_dir"), 2)
            self.assertNotIn("stale_run_output", kinds)
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ===========================================================================
# 2. HUD Research Continuity Rendering
# ===========================================================================

class TestResearchContinuityHudRendering(unittest.TestCase):
    def test_rendering_is_compact_markdown(self):
        memory = {
            "entry_count": 3,
            "verified_run_count": 0,
            "insight_count": 1,
            "open_insight_count": 0,
            "consecutive_non_improvements": 2,
            "reassess_recommended": False,
            "latest_checkpoint": {
                "conclusion": "V11 scored 0.50 due to removing rank partitioning",
                "next_action": "Implement V12 by restoring rank partitioning",
                "insight_ids": ["704da07f87cd474cb5c6123f0eae1732"],
                "open_questions": ["How to break past 10/14?"],
            },
            "recent_decisions": [
                {
                    "conclusion": "Bypassing rank partitioning collapses to All-False",
                    "outcome": "rejected",
                    "lesson": "Rank partitioning is mathematically essential",
                }
            ],
            "research_frontier": [],
            "unreviewed_result_count": 1,
            "progress_path": "C:\\ws\\.ai-harness\\artifacts\\task\\RESEARCH_PROGRESS.md",
            "progress_warning": None,
        }
        lines = _research_continuity_hud_lines(memory)
        text = "\n".join(lines)
        self.assertNotIn("{", text)
        self.assertNotIn('"entry_count"', text)
        self.assertIn("EPISTEMIC DISCONNECT WARNING", text)
        self.assertIn("[QUARANTINED NOTE] Latest checkpoint", text)
        self.assertNotIn("V11 scored 0.50", text)
        self.assertNotIn("Implement V12", text)
        self.assertIn("Recent decision (unverified interpretation)", text)
        self.assertNotIn("Recent decision (rejected)", text)
        self.assertIn("RESEARCH_PROGRESS.md", text)
        self.assertIn("Unreviewed results: 1", text)
        self.assertLess(len(text), 1200)

    def test_truncation_bounded_for_long_fields(self):
        memory = {
            "entry_count": 1,
            "verified_run_count": 0,
            "insight_count": 0,
            "open_insight_count": 0,
            "consecutive_non_improvements": 0,
            "reassess_recommended": True,
            "latest_checkpoint": {
                "conclusion": "x" * 2000,
                "next_action": "y" * 2000,
                "insight_ids": [],
                "open_questions": [],
            },
            "recent_decisions": [],
            "research_frontier": [],
            "unreviewed_result_count": 0,
            "progress_path": None,
            "progress_warning": "progress file missing",
        }
        lines = _research_continuity_hud_lines(memory)
        text = "\n".join(lines)
        self.assertIn("REASSESS PLATEAU", text)
        self.assertIn("\u26a0\ufe0f", text)
        self.assertLess(len(text), 1500)

    def test_stagnation_advisory_in_hud(self):
        memory = {
            "entry_count": 5,
            "verified_run_count": 4,
            "insight_count": 2,
            "open_insight_count": 1,
            "consecutive_non_improvements": 3,
            "reassess_recommended": True,
            "latest_checkpoint": None,
            "recent_decisions": [],
            "research_frontier": [],
            "unreviewed_result_count": 0,
            "progress_path": None,
            "progress_warning": None,
        }
        lines = _research_continuity_hud_lines(memory)
        text = "\n".join(lines)
        self.assertIn("STAGNATION / PLATEAU ADVISORY", text)
        self.assertIn("Honor the task-scoped recovery gate", text)
        self.assertIn("Continue diagnostics, research notes and bounded recovery probes", text)

    def test_empty_memory_renders_without_error(self):
        lines = _research_continuity_hud_lines(
            {
                "entry_count": 0,
                "verified_run_count": 0,
                "insight_count": 0,
                "open_insight_count": 0,
                "consecutive_non_improvements": 0,
                "reassess_recommended": False,
                "latest_checkpoint": None,
                "recent_decisions": [],
                "research_frontier": [],
                "unreviewed_result_count": 0,
                "progress_path": "C:\\ws\\RESEARCH_PROGRESS.md",
                "progress_warning": None,
            }
        )
        self.assertTrue(lines)
        self.assertIn("Entries: 0", lines[0])

    def test_hud_suppresses_checkpoint_and_marks_unverified_decisions(self):
        text = "\n".join(_research_continuity_hud_lines({
            "entry_count": 3, "verified_run_count": 0, "insight_count": 2,
            "open_insight_count": 2, "latest_checkpoint": {"conclusion": "SCORE 0.99 FINAL", "next_action": "Promote"},
            "recent_decisions": [{"conclusion": "Model is best", "outcome": "supported", "verification_status": "legacy_unstructured"}],
            "research_frontier": [], "unreviewed_result_count": 0,
        }))
        self.assertIn("EPISTEMIC DISCONNECT WARNING", text)
        self.assertIn("[QUARANTINED NOTE]", text)
        self.assertNotIn("SCORE 0.99", text)
        self.assertIn("[UNVERIFIED]", text)

    def test_context_hud_shows_cached_provider_reconciliation_truthfully(self):
        root, spec_path = make_workspace(prefix="aih-test-rec-hud-")
        try:
            attach_project(root, task_config=spec_path)
            bridge = AgentBridge(root, task_spec_path=spec_path)
            report_path = root / ".ai-harness" / "reconciliation_report.json"
            reports = [
                ("CLEAN", None, 0),
                ("DISCREPANCY_DETECTED", None, 3),
                ("UNKNOWN", "PROVIDER_UNAVAILABLE", None),
            ]
            for status, reason, count in reports:
                report = {
                    "schema_version": 1,
                    "reconciliation_id": "rec-fixture",
                    "competition": "fixture",
                    "provider": "kaggle",
                    "provider_submissions_count": count,
                    "accounted_submissions_count": 0,
                    "unexplained_submissions": [],
                    "status": status,
                    "checked_at": "2026-10-26T12:00:00+00:00",
                }
                if reason:
                    report["reason_code"] = reason
                atomic_write_json(report_path, report)
                text = bridge.render_context_markdown()
                display = f"{status} / {reason}" if reason else status
                self.assertIn(f"Status: `{display}`", text)
                self.assertIn("Last checked: `2026-10-26T12:00:00+00:00`", text)
                if count is None:
                    self.assertNotIn("Provider submissions: `0`", text)
                else:
                    self.assertIn(f"Provider submissions: `{count}`", text)
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ===========================================================================
# 3. Status Reports Plugins and Models
# ===========================================================================

class TestStatusReportsPluginsAndModels(unittest.TestCase):
    def test_status_plugins_section_default_workspace(self):
        root, _ = make_workspace()
        try:
            status = collect_status(project_root=root, include_cloud=False)
            plugins = status["plugins"]
            self.assertIsInstance(plugins.get("installed"), list)
            self.assertIn("default", {p["id"] for p in plugins["installed"]})
            self.assertEqual(plugins["active_plugin_id"], "default")
            names = plugins["model_names"]
            for name in ("lightgbm", "xgboost", "catboost", "baseline"):
                self.assertIn(name, names)
            self.assertNotIn("decomposed_mapping_equity", names)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_available_model_names_excludes_plugin_names(self):
        names = available_model_names()
        self.assertIn("lightgbm", names)
        for zindi_name in (
            "decomposed_mapping_equity",
            "analytic_decomposed",
            "dynamic_divisor_equity",
        ):
            self.assertNotIn(zindi_name, names)


# ===========================================================================
# 4. Ergonomics and Resilience (from test_ergonomics_and_resilience.py)
# ===========================================================================

class ErgonomicsAndResilienceTests(unittest.TestCase):
    def test_research_memory_defaults_and_load(self):
        rm = ResearchMemory(".")
        self.assertIsNotNone(rm.root)
        self.assertIsNotNone(rm.artifacts)

        rm_loaded = ResearchMemory.load(".")
        self.assertEqual(rm.root, rm_loaded.root)

    def test_research_memory_frontier_access_styles(self):
        rm = ResearchMemory(".")
        frontier_prop = rm.frontier
        self.assertIsInstance(frontier_prop, list)

        frontier_call = rm.frontier()
        self.assertEqual(frontier_prop, frontier_call)

        self.assertEqual(rm.research_frontier, frontier_prop)
        self.assertEqual(rm.research_frontier(), frontier_prop)

        class_frontier = ResearchMemory.frontier()
        self.assertIsInstance(class_frontier, list)
        class_rf = ResearchMemory.research_frontier()
        self.assertEqual(class_frontier, class_rf)

    def test_competition_state_machine_load(self):
        sm = CompetitionStateMachine.load(".")
        self.assertIsInstance(sm, CompetitionStateMachine)
        self.assertEqual(sm.project_root, Path(".").resolve())

    def test_artifacts_and_repair_cross_exports(self):
        self.assertIs(repair_run_dir, repair_run)
        self.assertIs(repair_run_alias, repair_folder)
        self.assertTrue(callable(run_repair))
        self.assertTrue(callable(validate_run))

    def test_agent_bridge_action_registry_and_dynamic_handlers(self):
        self.assertIn("record_research_insight", typed_actions)
        self.assertEqual(HANDLERS, typed_actions)
        self.assertEqual(_ACTION_REGISTRY, typed_actions)
        self.assertEqual(AgentBridge._ACTION_REGISTRY, typed_actions)

        bridge = AgentBridge(".")
        self.assertTrue(hasattr(bridge, "_action_record_research_insight"))
        self.assertTrue(hasattr(bridge, "record_research_insight"))

    def test_contract_identity_with_dict_task_spec(self):
        dict_spec = {"metadata": {"id": "test_comp", "name": "Test"}}
        ident = contract_identity(dict_spec)
        self.assertIn("task_sha256", ident)
        self.assertIn("scorer_sha256", ident)

        raw_dict = {"id": "raw_task", "name": "Raw"}
        ident_raw = contract_identity(raw_dict)
        self.assertIn("task_sha256", ident_raw)

    def test_load_task_spec_export(self):
        self.assertTrue(callable(load_task_spec))
        spec = load_task_spec(".")
        self.assertIsNotNone(spec)

    @mock.patch.object(KaggleClient, "_legacy_api_instance")
    def test_kaggle_client_facade_methods_and_aliases(self, legacy_api):
        self.assertIsNotNone(KaggleLeaderboardTracker)

        client = KaggleClient()
        self.assertTrue(hasattr(client, "api"))
        self.assertTrue(hasattr(client, "_api"))
        self.assertTrue(hasattr(client, "leaderboard"))
        self.assertTrue(hasattr(client, "get_leaderboard"))
        self.assertTrue(hasattr(client, "list_topics"))
        self.assertTrue(hasattr(client, "topics"))
        self.assertTrue(hasattr(client, "discussions"))
        self.assertTrue(hasattr(client, "list_discussions"))
        self.assertTrue(hasattr(client, "topic_messages"))
        self.assertTrue(hasattr(client, "list_topic_messages"))
        self.assertTrue(hasattr(client, "kernels_pull"))
        self.assertTrue(hasattr(client, "kernel_pull"))
        self.assertTrue(hasattr(client, "pull_kernel"))
        self.assertTrue(hasattr(client, "get_kernel"))

    def test_find_dataset_mount_casing_and_resilience(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ds_dir = base / "aih-kaggle-test-dataset"
            ds_dir.mkdir(parents=True)

            found = find_dataset_mount(base, "user/aih-kaggle-test-dataset")
            self.assertEqual(found, ds_dir)

            found_ci = find_dataset_mount(base, "user/AIH-KAGGLE-TEST-DATASET")
            self.assertEqual(found_ci, ds_dir)

            nested = base / "datasets" / "some_hash" / "nested-ds"
            nested.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                find_dataset_mount(base, "user/nested-ds")


# ===========================================================================
# 5. Strategy Cards and Spawner (from test_strategy_cards.py)
# ===========================================================================

class TestStrategyCardsAndSpawner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 0
        cls.server = socketserver.TCPServer(("127.0.0.1", cls.port), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_default_strategy_cards_completeness(self):
        self.assertEqual(len(DEFAULT_STRATEGY_CARDS), 6)
        branches = {c["branch"] for c in DEFAULT_STRATEGY_CARDS}
        self.assertIn("feat", branches)
        self.assertIn("model", branches)
        self.assertIn("loss", branches)
        self.assertIn("ens", branches)

        approaches = {c["approach_type"] for c in DEFAULT_STRATEGY_CARDS}
        self.assertIn("ml", approaches)
        self.assertIn("dl", approaches)
        self.assertIn("llm", approaches)

        for card in DEFAULT_STRATEGY_CARDS:
            self.assertIn("id", card)
            self.assertIn("title", card)
            self.assertIn("hypothesis", card)
            self.assertIn("intervention", card)
            self.assertIn("problem_addressed", card)
            self.assertIn("model_family", card)
            self.assertIn("model_name", card)
            self.assertIn("approach_type", card)
            self.assertNotIn("estimated_delta", card)
            self.assertNotIn("badge", card)

    def test_api_strategy_cards_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/api/strategy_cards"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode("utf-8"))
            self.assertIn("strategies", data)
            self.assertEqual(len(data["strategies"]), len([c for c in DEFAULT_STRATEGY_CARDS if c.get("model_name") != "threshold_optimizer"]))

    def test_get_next_experiment_id(self):
        res = get_next_experiment_id("feat", "video_title_ctx")
        self.assertIn("next_id", res)
        self.assertTrue(res["next_id"].startswith("exp"))
        self.assertIn("feat", res["next_id"])
        self.assertIn("video_title_ctx", res["next_id"])


class TestHygieneAndCliErgonomics(unittest.TestCase):
    def test_hygiene_allow_dirs_and_ignore_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copytree(Path("configs"), root / "configs")
            (root / "storage").mkdir()
            (root / "storage" / "file.txt").write_text("data")
            (root / "prototypes").mkdir()
            (root / "prototypes" / "proto.py").write_text("code")

            # Without allow_dirs: flagged
            res = scan_workspace_hygiene(root)
            paths = [f["path"] for f in res["findings"]]
            self.assertIn("storage", paths)
            self.assertIn("prototypes", paths)

            # With allow_dirs in task.yaml: clean
            task_cfg = (root / "configs" / "task.yaml")
            content = task_cfg.read_text(encoding="utf-8")
            content += "\n  hygiene:\n    allow_dirs: [storage, prototypes]\n"
            task_cfg.write_text(content, encoding="utf-8")

            res_clean = scan_workspace_hygiene(root)
            paths_clean = [f["path"] for f in res_clean["findings"]]
            self.assertNotIn("storage", paths_clean)
            self.assertNotIn("prototypes", paths_clean)

    def test_cli_flat_flags_and_input_alias(self):
        from tools.agent_bridge_cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copytree(Path("configs"), root / "configs")
            # 1. Test flat flags for record_research_insight
            rc = cli_main([
                "--project-root", str(root),
                "act",
                "--action", "record_research_insight",
                "--title", "Flat flag insight",
                "--observation", "Observed gain",
                "--mechanism", "Feature scaling",
                "--next-action", "Run full validation",
            ])
            self.assertEqual(rc, 0)

            # 2. Test --input JSON alias for record_research_decision
            rc = cli_main([
                "--project-root", str(root),
                "act",
                "--action", "record_research_decision",
                "--input", '{"conclusion": "Promising result", "outcome": "keep"}',
            ])
            self.assertEqual(rc, 0)

            # 3. Test checkpoint flat flags
            rc = cli_main([
                "--project-root", str(root),
                "act",
                "--action", "record_research_checkpoint",
                "--summary", "Checkpoint 1",
                "--next-action", "Next step",
            ])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
