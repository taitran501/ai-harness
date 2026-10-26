"""Regression tests for the agent-ergonomics update.

- Hygiene: standard workspace dirs (dataset/scratch/splits/__pycache__) are
  no longer flagged; dozens of Kaggle/kernel run-output dirs collapse into
  ONE advisory instead of 30 lines of HUD noise.
- HUD: research continuity renders as compact Markdown instead of a raw
  ``json.dumps`` blob (context-saturation was the framework's own rule).
- Status: ``ai-harness status`` now reports installed plugins, the active
  plugin, and the union of valid model names (post-B16, model names are
  plugin-dependent and were otherwise only discoverable by trial and error).
"""
from __future__ import annotations

import shutil
import unittest

from tests.fixtures import make_workspace


# ---------------------------------------------------------------------------
# Hygiene allow-list + run-output aggregation
# ---------------------------------------------------------------------------
class TestHygieneAllowListAndAggregation(unittest.TestCase):
    def test_standard_dirs_not_flagged(self):
        from harness.hygiene import scan_workspace_hygiene

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
        from harness.hygiene import scan_workspace_hygiene

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
            # one harness-sealed dir (artifact manifest present) stays in the
            # stale group; the seven unsealed dirs become a bypass finding.
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
        """<=2 run-output dirs are not worth an aggregation line."""
        from harness.hygiene import scan_workspace_hygiene

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


# ---------------------------------------------------------------------------
# HUD research continuity: compact Markdown, not a raw JSON blob
# ---------------------------------------------------------------------------
class TestResearchContinuityHudRendering(unittest.TestCase):
    def test_rendering_is_compact_markdown(self):
        from harness.agent_bridge import _research_continuity_hud_lines

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
        # No raw JSON dump in the HUD.
        self.assertNotIn("{", text)
        self.assertNotIn('"entry_count"', text)
        # The resume-critical facts survive.
        self.assertIn("V11 scored 0.50", text)
        self.assertIn("[UNVERIFIED NOTE] Latest checkpoint:", text)
        self.assertIn("[UNVERIFIED NOTE] Open questions:", text)
        self.assertIn("Implement V12", text)
        self.assertIn("[704da07f]", text)
        self.assertIn("rejected", text)
        self.assertIn("RESEARCH_PROGRESS.md", text)
        self.assertIn("Unreviewed results: 1", text)
        # Bounded: a HUD block should be far smaller than a json dump of the
        # same payload (old code embedded the full payload on one line).
        self.assertLess(len(text), 1200)

    def test_truncation_bounded_for_long_fields(self):
        from harness.agent_bridge import _research_continuity_hud_lines

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
        # Both long fields must be truncated, not embedded at full length.
        self.assertLess(len(text), 1500)

    def test_stagnation_advisory_in_hud(self):
        from harness.agent_bridge import _research_continuity_hud_lines

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
        self.assertIn("Honor the task-scoped stagnation pause", text)
        self.assertIn("ai-harness discussion list", text)

    def test_empty_memory_renders_without_error(self):
        from harness.agent_bridge import _research_continuity_hud_lines

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

    def test_context_hud_shows_cached_provider_reconciliation_truthfully(self):
        from harness.agent_bridge import AgentBridge
        from harness.io_utils import atomic_write_json
        from harness.project import attach_project

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


# ---------------------------------------------------------------------------
# ai-harness status: plugins + model names discoverable
# ---------------------------------------------------------------------------
class TestStatusReportsPluginsAndModels(unittest.TestCase):
    def test_status_plugins_section_default_workspace(self):
        from harness.status_report import collect_status

        root, _ = make_workspace()
        try:
            status = collect_status(project_root=root, include_cloud=False)
            plugins = status["plugins"]
            self.assertIsInstance(plugins.get("installed"), list)
            self.assertIn("default", {p["id"] for p in plugins["installed"]})
            self.assertEqual(plugins["active_plugin_id"], "default")
            names = plugins["model_names"]
            # Built-in registry names are discoverable without trial and error.
            for name in ("lightgbm", "xgboost", "catboost", "baseline"):
                self.assertIn(name, names)
            # Competition-specific names must NOT appear for a default plugin.
            self.assertNotIn("decomposed_mapping_equity", names)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_available_model_names_excludes_plugin_names(self):
        from harness.models import available_model_names

        names = available_model_names()
        self.assertIn("lightgbm", names)
        for zindi_name in (
            "decomposed_mapping_equity",
            "analytic_decomposed",
            "dynamic_divisor_equity",
        ):
            self.assertNotIn(zindi_name, names)


if __name__ == "__main__":
    unittest.main()
