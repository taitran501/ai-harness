"""Unit tests for `ai-harness sync` (harness/sync.py + CLI wiring).

The tests never run `pip install` (no_install=True) and use a throwaway git
clone with no remote, so nothing touches the real environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness.cli import build_parser, main
from harness.sync import find_framework_repo, run_sync


def _make_workspace(tmp: Path, dirty: bool = False, with_remote: bool = False) -> Path:
    ws = tmp / "competition"
    fw = ws / "ai-harness"
    (ws / "configs").mkdir(parents=True)
    (fw / "harness").mkdir(parents=True)
    (fw / "harness" / "__init__.py").write_text("VERSION = 'test'\n", encoding="utf-8")
    (fw / "pyproject.toml").write_text('[project]\nname = "ai-harness"\n', encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def git(*args: str, cwd: Path = fw) -> None:
        subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=env)

    git("init", "-b", "main")
    git("add", "-A")
    git("commit", "-m", "init")
    if dirty:
        (fw / "stray.txt").write_text("local change\n", encoding="utf-8")
    if with_remote:
        # A second bare repo acts as the origin so pull --ff-only can succeed.
        origin = tmp / "origin.git"
        origin.mkdir()
        subprocess.run(
            ["git", "init", "--bare", str(origin)], capture_output=True, text=True, check=True, env=env
        )
        subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=fw, capture_output=True, text=True, check=True, env=env)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=fw, capture_output=True, text=True, check=True, env=env)
    (ws / "configs" / "task.yaml").write_text(
        "task:\n  name: SyncTask\n  primary_metric: rmsle\n"
        "  execution:\n    backend: kaggle\n    locked: true\n",
        encoding="utf-8",
    )
    return ws


class TestFindFrameworkRepo(unittest.TestCase):
    def test_auto_detect_conventional_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_detect_") as tmp:
            ws = _make_workspace(Path(tmp))
            self.assertEqual(find_framework_repo(ws), ws / "ai-harness")

    def test_explicit_dir_wins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_detect_") as tmp:
            ws = _make_workspace(Path(tmp))
            custom = ws / "custom-harness"
            custom.mkdir()
            (custom / ".git").mkdir()
            (custom / "harness").mkdir()
            self.assertEqual(find_framework_repo(ws, framework_dir="custom-harness"), custom)

    def test_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_detect_") as tmp:
            ws = Path(tmp) / "empty"
            ws.mkdir()
            self.assertIsNone(find_framework_repo(ws))


class TestRunSync(unittest.TestCase):
    def test_full_sync_no_remote_reports_problem_but_attaches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_run_") as tmp:
            ws = _make_workspace(Path(tmp))
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}):
                rc = run_sync(project_root=ws, no_install=True)
            self.assertEqual(rc, 2)  # no remote -> update impossible -> problem flagged
            manifest = ws / ".ai-harness" / "project.yaml"
            self.assertTrue(manifest.is_file())
            skill = ws / ".agents" / "skills" / "ai-harness" / "SKILL.md"
            self.assertTrue(skill.is_file())
            self.assertIn("Execution Backend LOCKED", skill.read_text(encoding="utf-8"))
            self.assertIn("## 5. Execution Backends", skill.read_text(encoding="utf-8"))

    def test_sync_skips_cleanly_and_reports_ok(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_run_") as tmp:
            ws = _make_workspace(Path(tmp))
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}):
                rc = run_sync(project_root=ws, no_pull=True, no_install=True, no_attach=True)
            self.assertEqual(rc, 0)

    def test_dirty_tree_skips_pull_with_reason(self) -> None:
        import io
        import json

        with tempfile.TemporaryDirectory(prefix="test_sync_dirty_") as tmp:
            ws = _make_workspace(Path(tmp), dirty=True)
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}), mock.patch.object(
                sys, "stdout", new=io.StringIO()
            ) as out:
                rc = run_sync(project_root=ws, no_install=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 2)
            self.assertIn("dirty", report["steps"]["pull"].get("reason", ""))

    def test_sync_with_remote_updates_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_remote_") as tmp:
            ws = _make_workspace(Path(tmp), with_remote=True)
            fw = ws / "ai-harness"
            env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
            (fw / "NEW.md").write_text("change\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=fw, capture_output=True, text=True, env=env)
            subprocess.run(["git", "commit", "-m", "second"], cwd=fw, capture_output=True, text=True, env=env)
            # publish the second commit so the remote actually has it
            subprocess.run(["git", "push", "origin", "main"], cwd=fw, capture_output=True, text=True, env=env)
            # The second commit is the target; local is then reset behind it.
            target = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fw, capture_output=True, text=True, env=env).stdout.strip()
            # reset local branch behind origin so pull --ff-only advances it
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=fw, capture_output=True, text=True, env=env)
            before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fw, capture_output=True, text=True, env=env).stdout.strip()
            self.assertNotEqual(before, target)
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}):
                rc = run_sync(project_root=ws, no_install=True)
            after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fw, capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(rc, 0, "clean clone with reachable origin should sync ok")
            self.assertEqual(after, target, "pull --ff-only should fast-forward to the newer commit")


class TestSyncCli(unittest.TestCase):
    def test_parser_exposes_sync_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["sync", "--framework-dir", "ai-harness", "--no-pull", "--no-install", "--json"]
        )
        self.assertEqual(args.command, "sync")
        self.assertTrue(args.no_pull)
        self.assertTrue(args.no_install)
        self.assertTrue(args.json)

    def test_missing_repo_exits_2(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_cli_") as tmp:
            ws = Path(tmp) / "empty"
            ws.mkdir()
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}), mock.patch.object(
                sys, "stdout", new=__import__("io").StringIO()
            ) as out:
                rc = main(["sync", "--project-root", str(ws), "--no-install"])
            self.assertEqual(rc, 2)
            self.assertIn("No ai-harness git clone found", out.getvalue())


if __name__ == "__main__":
    unittest.main()
