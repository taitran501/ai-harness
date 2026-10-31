"""Regression tests for the agent-bypass audit (normal-user adoption).

Source: the rsna-knee workspace (2026-10-16): attached to the harness, yet
22/22 run-output dirs were manual kernel pushes (no ``artifact_manifest.json``),
zero harness experiments, zero agent sessions, zero warnings. Root causes
found and fixed here:

1. ``ai-harness status`` reported kaggle/modal as *unavailable* even when
   the CLI was installed + authenticated (key mismatch: status looked for
   ``available``/``detected`` while the detector reports
   ``installed``/``authenticated``/``has_credentials``). An agent reading
   the status concluded harness dispatch was impossible and pushed kernels
   manually.
2. Workspaces had no universal agent entry point (no AGENTS.md) — skills
   under ``.agents/skills/`` only load for harnesses that support them and
   when the task matches, so a generic agent never learned the harness
   exists.
3. Bypassed run-output dirs were silently aggregated into "stale" noise
   instead of being flagged as harness bypass.
4. Two different SKILL.md generators (inline in ``project.py`` vs
   ``skill_injector``) could disagree; attach must now produce the same
   rich skill as sync.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from harness.project import (
    AGENTS_MD_MARKER_END,
    AGENTS_MD_MARKER_START,
    attach_project,
    update_agents_md,
)
from harness.skill_injector import generate_skill_content
from tests.test_research_continuity import task_document

_TASK = task_document()


class CloudAvailabilityMappingTests(unittest.TestCase):
    def test_kaggle_authenticated_reports_available(self):
        from harness import status_report

        env = {
            "kaggle": {"installed": True, "authenticated": True, "username": "u"},
            "modal": {"installed": True, "has_credentials": True},
            "docker": {"installed": True, "running": False},
            "gpu": {"detected": True},
            "llm": {"total_configured": 0},
        }
        with patch("harness.environment_detector.EnvironmentDetector.get_full_status",
                   return_value=env):
            section = status_report._cloud_section(None)
        accounts = section["accounts"]
        self.assertTrue(accounts["kaggle"]["available"])
        self.assertTrue(accounts["modal"]["available"])
        self.assertFalse(accounts["docker"]["available"])
        self.assertTrue(accounts["gpu"]["available"])
        self.assertFalse(accounts["llm"]["available"])

    def test_kaggle_not_authenticated_reports_unavailable(self):
        from harness import status_report

        env = {
            "kaggle": {"installed": True, "authenticated": False, "username": None},
            "modal": {"installed": False, "has_credentials": False},
            "docker": {"installed": False},
            "gpu": {"detected": False},
            "llm": {"total_configured": 0},
        }
        with patch("harness.environment_detector.EnvironmentDetector.get_full_status",
                   return_value=env):
            section = status_report._cloud_section(None)
        self.assertFalse(section["accounts"]["kaggle"]["available"])
        self.assertFalse(section["accounts"]["modal"]["available"])


class HygieneBypassDetectionTests(unittest.TestCase):
    def _workspace(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-bypass-hygiene-")
        root = Path(temp.name)
        (root / "configs").mkdir()
        (root / "configs/task.yaml").write_text(yaml.safe_dump(_TASK))
        manual = root / "kaggle_sub_exp010_manual"
        manual.mkdir()
        (manual / "submission.csv").write_text("id,label\n")
        sealed = root / "kaggle_sub_exp011_harness"
        sealed.mkdir()
        (sealed / "submission.csv").write_text("id,label\n")
        (sealed / "artifact_manifest.json").write_text("{}")
        return temp, root

    def test_manual_and_sealed_dirs_are_reported_separately(self):
        from harness.hygiene import scan_workspace_hygiene

        temp, root = self._workspace()
        try:
            result = scan_workspace_hygiene(root)
        finally:
            temp.cleanup()
        # 2 run-output dirs is below the aggregation threshold (<=2), so the
        # per-dir findings remain; verify the sealed flag distinguishes them.
        details = {f["path"]: f for f in result["findings"]}
        self.assertFalse(details["kaggle_sub_exp010_manual"].get("sealed"))
        self.assertTrue(details["kaggle_sub_exp011_harness"].get("sealed"))

    def test_aggregated_bypass_finding_when_many_manual_dirs(self):
        from harness.hygiene import scan_workspace_hygiene

        temp, root = self._workspace()
        try:
            for i in range(3, 10):
                d = root / f"kaggle_sub_exp{i:03d}_manual"
                d.mkdir()
                (d / "submission.csv").write_text("id,label\n")
            result = scan_workspace_hygiene(root)
        finally:
            temp.cleanup()
        bypass = [f for f in result["findings"] if f["kind"] == "bypassed_run_output"]
        stale = [f for f in result["findings"] if f["kind"] == "stale_run_output"]
        self.assertEqual(len(bypass), 1)
        self.assertIn("artifact manifest", bypass[0]["detail"])
        self.assertIn("manual kaggle pushes", bypass[0]["detail"])
        self.assertTrue(any("exp011_harness" in f["detail"] for f in stale),
                        "sealed dir should stay in the stale group")


class AgentsMdTests(unittest.TestCase):
    def _attach(self, root):
        (root / "configs").mkdir(exist_ok=True)
        (root / "configs/task.yaml").write_text(yaml.safe_dump(_TASK))
        return attach_project(root, task_config=root / "configs/task.yaml", force=True)

    def test_attach_creates_managed_agents_md(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-agents-md-")
        root = Path(temp.name)
        try:
            result = self._attach(root)
            self.assertEqual(result["agents_md"], "created")
            content = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn(AGENTS_MD_MARKER_START, content)
            self.assertIn(AGENTS_MD_MARKER_END, content)
            self.assertIn("Never push kernels manually", content)
            self.assertIn(".agents/skills/ai-harness/SKILL.md", content)
        finally:
            temp.cleanup()

    def test_user_owned_agents_md_is_not_touched(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-agents-md-user-")
        root = Path(temp.name)
        (root / "AGENTS.md").write_text("# My own rules\n- do the thing\n", encoding="utf-8")
        try:
            result = self._attach(root)
            self.assertEqual(result["agents_md"], "skipped_user_owned")
            self.assertEqual(
                (root / "AGENTS.md").read_text(encoding="utf-8"),
                "# My own rules\n- do the thing\n",
            )
        finally:
            temp.cleanup()

    def test_managed_block_refresh_preserves_user_content(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-agents-md-refresh-")
        root = Path(temp.name)
        try:
            self._attach(root)
            agents = root / "AGENTS.md"
            original = agents.read_text(encoding="utf-8")
            with_marker = original.replace(
                AGENTS_MD_MARKER_END,
                AGENTS_MD_MARKER_END + "\n\n## User section\n- custom rule kept\n",
            )
            agents.write_text(with_marker, encoding="utf-8")
            updated = update_agents_md(root)
            self.assertEqual(updated["status"], "refreshed")
            content = agents.read_text(encoding="utf-8")
            self.assertIn("custom rule kept", content)
            self.assertEqual(content.count(AGENTS_MD_MARKER_START), 1)
        finally:
            temp.cleanup()


class SkillConsolidationTests(unittest.TestCase):
    def test_attach_skill_matches_sync_skill_content(self):
        """attach must write the skill_injector content (rich SOP), not the
        legacy inline block — otherwise attached workspaces miss the
        param-file guidance, research contracts and gate steps."""
        temp = tempfile.TemporaryDirectory(prefix="aih-skill-consolidation-")
        root = Path(temp.name)
        try:
            (root / "configs").mkdir()
            (root / "configs/task.yaml").write_text(yaml.safe_dump(_TASK))
            attach_project(root, task_config=root / "configs/task.yaml", force=True)
            written = (root / ".agents/skills/harness/SKILL.md").read_text(encoding="utf-8")
            alias = (root / ".agents/skills/ai-harness/SKILL.md").read_text(encoding="utf-8")
            self.assertIn("`/harness`", alias)
            generated = generate_skill_content(_TASK, artifact_path=".ai-harness/artifacts/<task-id>")
            for probe in (
                "Research continuity action contracts",
                "trace_research_insight",
                "confirm_task_spec",
                "phase render",
                "script adapter",
            ):
                self.assertIn(probe, written, f"attached skill missing: {probe}")
                self.assertIn(probe, generated)
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
