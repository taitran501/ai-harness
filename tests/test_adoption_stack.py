"""T3-T5 of the adoption stack (external-repo research-backed).

Findings under test:
- T3: session-start adoption gate in the Context HUD (aexp pattern,
  implemented host-agnostically in the layer the agent must read).
- T4: per-host guardrail hooks (Claude Code + Cursor) generated at attach —
  block manual kaggle pushes, fail-open, merge-safe, human override.
- T5: adoption health in `ai-harness status` (the human's bypass audit).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from harness.agent_bridge import AgentBridge
from harness.host_hooks import (
    HOOK_SCRIPT_REL,
    write_host_hooks,
)
from harness.project import attach_project
from harness.status_report import _adoption_section, render_markdown, collect_status
from tests.test_research_continuity import task_document


def _workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="aih-adoption-"))
    (root / "configs").mkdir()
    (root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()), encoding="utf-8")
    return root


class HookScriptTests(unittest.TestCase):
    """The generated hook script: block, allow, fail-open, human override."""

    def setUp(self):
        self.root = _workspace()
        info = write_host_hooks(self.root)
        self.script = Path(info["script"])
        self.assertTrue(self.script.is_file())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, stdin: str, env_extra=None):
        env = dict(os.environ)
        env.pop("AI_HARNESS_ALLOW_MANUAL", None)
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, str(self.script)],
            input=stdin, capture_output=True, text=True, env=env, cwd=self.root,
            timeout=30,
        )

    def test_blocks_manual_kaggle_push_claude_payload(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "kaggle kernels push -p kaggle_sub_exp001_manual"}})
        r = self._run(payload)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("ai-harness", r.stderr)
        self.assertIn("run_full_validation", r.stderr)

    def test_blocks_competitions_submit_cursor_payload(self):
        payload = json.dumps({"shell_command": "kaggle competitions submit -c x -f sub.csv"})
        r = self._run(payload)
        self.assertEqual(r.returncode, 2, r.stderr)

    def test_allows_benign_command(self):
        r = self._run(json.dumps({"tool_input": {"command": "python train.py"}}))
        self.assertEqual(r.returncode, 0)

    def test_allows_kaggle_non_push_commands(self):
        for cmd in ("kaggle kernels list", "kaggle datasets download x"):
            r = self._run(json.dumps({"tool_input": {"command": cmd}}))
            self.assertEqual(r.returncode, 0, cmd)

    def test_human_override_allows(self):
        payload = json.dumps({"tool_input": {"command": "kaggle kernels push -p x"}})
        r = self._run(payload, env_extra={"AI_HARNESS_ALLOW_MANUAL": "1"})
        self.assertEqual(r.returncode, 0)

    def test_fail_open_on_garbage_stdin(self):
        r = self._run("this is not json{{{")
        self.assertEqual(r.returncode, 0)

    def test_fail_open_on_empty_stdin(self):
        r = self._run("")
        self.assertEqual(r.returncode, 0)


class HostConfigTests(unittest.TestCase):
    """write_host_hooks: creates, merges, never destroys user config."""

    def tearDown(self):
        pass

    def test_creates_claude_and_cursor_configs(self):
        root = _workspace()
        try:
            write_host_hooks(root)
            claude = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            cursor = json.loads((root / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
            entries = claude["hooks"]["PreToolUse"][0]["hooks"]
            self.assertEqual(entries[0]["command"], "python " + HOOK_SCRIPT_REL.as_posix())
            self.assertEqual(entries[0]["managed_by"], "ai-harness")
            self.assertEqual(cursor["version"], 1)
            self.assertEqual(cursor["hooks"]["beforeShellExecution"][0]["managed_by"], "ai-harness")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_merge_preserves_user_hooks(self):
        root = _workspace()
        try:
            (root / ".claude").mkdir()
            user_cfg = {"hooks": {"PreToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "user-lint.sh"}]}]}}
            (root / ".claude" / "settings.json").write_text(json.dumps(user_cfg), encoding="utf-8")
            write_host_hooks(root)
            merged = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            groups = merged["hooks"]["PreToolUse"]
            self.assertEqual(len(groups), 2, "user hook group must survive merge")
            self.assertIn("Edit", [g["matcher"] for g in groups if isinstance(g, dict) and "matcher" in g])
            managed = [g for g in groups if any(h.get("managed_by") == "ai-harness" for h in g.get("hooks", []))]
            self.assertEqual(len(managed), 1)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_idempotent_refresh(self):
        root = _workspace()
        try:
            write_host_hooks(root)
            write_host_hooks(root)
            claude = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            groups = claude["hooks"]["PreToolUse"]
            managed_entries = [
                h for g in groups for h in g.get("hooks", []) if h.get("managed_by") == "ai-harness"
            ]
            self.assertEqual(len(managed_entries), 1, "refresh must not duplicate the managed entry")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_skips_unreadable_user_config(self):
        root = _workspace()
        try:
            (root / ".cursor").mkdir()
            bad = root / ".cursor" / "hooks.json"
            bad.write_text("not json at all", encoding="utf-8")
            write_host_hooks(root)
            self.assertEqual(bad.read_text(encoding="utf-8"), "not json at all", "unreadable user file must be untouched")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


class AdoptionStatusTests(unittest.TestCase):
    """T5: adoption health in status."""

    def _ws(self):
        root = _workspace()
        attach_project(root, root / "configs/task.yaml")
        return root

    def tearDown(self):
        pass

    def test_not_attached(self):
        root = _workspace()
        self.assertEqual(_adoption_section(str(root))["status"], "not_attached")

    def test_fresh_workspace_no_activity(self):
        root = self._ws()
        try:
            res = _adoption_section(str(root))
            self.assertEqual(res["verdict"], "no_activity")
            self.assertEqual(res["experiments"], 0)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_sealed_run_healthy(self):
        root = self._ws()
        try:
            run = root / ".ai-harness" / "artifacts" / "adhoc" / "run_0001"
            run.mkdir(parents=True)
            (run / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            res = _adoption_section(str(root))
            self.assertEqual(res["verdict"], "healthy")
            self.assertEqual(res["experiments"], 1)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_bypass_detected(self):
        root = self._ws()
        try:
            (root / "kaggle_sub_exp001_manual").mkdir()
            (root / "kaggle_sub_exp002_manual").mkdir()
            (root / "kaggle_sub_exp003_manual").mkdir()
            run = root / ".ai-harness" / "artifacts" / "adhoc" / "run_0001"
            run.mkdir(parents=True)
            (run / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            res = _adoption_section(str(root))
            self.assertEqual(res["verdict"], "bypass_detected")
            self.assertEqual(res["bypassed_run_outputs"], 1)  # aggregated finding
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_markdown_renders_adoption_line(self):
        root = self._ws()
        try:
            (root / "kaggle_sub_exp001_manual").mkdir()
            (root / "kaggle_sub_exp002_manual").mkdir()
            (root / "kaggle_sub_exp003_manual").mkdir()
            md = render_markdown(collect_status(str(root), include_cloud=False))
            self.assertIn("adoption:", md)
            self.assertIn("bypass_detected", md)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


class AdoptionGateTests(unittest.TestCase):
    """T3: session-start adoption gate in the Context HUD."""

    def setUp(self):
        self.root = _workspace()
        attach_project(self.root, self.root / "configs/task.yaml")
        self.bridge = AgentBridge(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_gate_shows_on_fresh_workspace_zero_experiments(self):
        md = self.bridge.render_context_markdown()
        self.assertIn("⛔ Adoption protocol", md)
        self.assertIn("0 harness experiments", md)

    def test_gate_shows_bypass_evidence(self):
        for i in (1, 2, 3):
            (self.root / f"kaggle_sub_exp{i:03d}_manual").mkdir()
        md = self.bridge.render_context_markdown()
        self.assertIn("⛔ Adoption protocol", md)
        self.assertIn("lack harness evidence", md)

    def test_gate_mentions_override_is_human_only(self):
        md = self.bridge.render_context_markdown()
        self.assertIn("AI_HARNESS_ALLOW_MANUAL=1", md)


if __name__ == "__main__":
    unittest.main()
