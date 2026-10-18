"""Unit tests for `ai-harness sync` (harness/sync.py + CLI wiring).

The tests never run `pip install` (no_install=True) and use a throwaway git
clone with no remote, so nothing touches the real environment.
"""

from __future__ import annotations

import io
import json
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
    """Throwaway competition workspace whose ai-harness dir is a REAL (local,
    hardlinked) git clone of this repository. sync's attach step runs in a
    subprocess pinned to the clone, so a stub package cannot stand in for
    the framework code."""
    import harness as _harness_pkg

    tmp = Path(tmp)
    ws = tmp / "competition"
    fw = ws / "ai-harness"
    (ws / "configs").mkdir(parents=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(
        ["git", "clone", "--quiet", str(Path(_harness_pkg.__file__).resolve().parents[1]), str(fw)],
        capture_output=True, text=True, check=True, env=env,
    )
    if dirty:
        (fw / "stray.txt").write_text("local change" + chr(10), encoding="utf-8")
    if with_remote:
        # A second bare repo acts as the origin so pull --ff-only can succeed.
        origin = tmp / "origin.git"
        origin.mkdir()
        subprocess.run(
            ["git", "init", "--bare", str(origin)], capture_output=True, text=True, check=True, env=env
        )
        subprocess.run(["git", "remote", "set-url", "origin", str(origin)], cwd=fw, capture_output=True, text=True, check=True)
        subprocess.run(["git", "push", "-u", "origin", "HEAD"], cwd=fw, capture_output=True, text=True, check=True)
    (ws / "configs" / "task.yaml").write_text(
        "task:" + chr(10) + "  name: SyncTask" + chr(10) + "  primary_metric: rmsle" + chr(10) +
        "  execution:" + chr(10) + "    backend: kaggle" + chr(10) + "    locked: true" + chr(10),
        encoding="utf-8",
    )
    return ws


def _make_stub_workspace(tmp: Path, framework_dir: str = "ai-harness") -> Path:
    """Minimal workspace stub for detection-logic tests.

    ``find_framework_repo()`` only checks directory shape (a ``.git``
    marker plus a ``harness/`` package dir), so a real git clone (~2.7 s
    each) is not needed. Attach/sync tests keep real clones because their
    subprocesses genuinely execute the cloned code.
    """
    ws = Path(tmp) / "competition"
    fw = ws / framework_dir
    (fw / ".git").mkdir(parents=True)
    (fw / "harness").mkdir()
    (ws / "configs").mkdir(parents=True)
    return ws


class TestFindFrameworkRepo(unittest.TestCase):
    def test_auto_detect_conventional_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_detect_") as tmp:
            ws = _make_stub_workspace(Path(tmp))
            self.assertEqual(find_framework_repo(ws), ws / "ai-harness")

    def test_explicit_dir_wins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_detect_") as tmp:
            ws = _make_stub_workspace(Path(tmp))
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


class TestAttachPinning(unittest.TestCase):
    """Regression: attach must run with the workspace clone's code, not the
    code this process happened to import (global editable install drift)."""

    def test_attach_uses_workspace_clone_code(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_pin_") as tmp:
            ws = _make_workspace(Path(tmp))
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}), mock.patch.object(
                sys, "stdout", new=io.StringIO()
            ) as out:
                rc = run_sync(project_root=ws, no_pull=True, no_install=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0)
            attach = report["steps"]["attach"]
            self.assertEqual(attach["rc"], 0)
            origin = attach.get("harness_origin", "")
            self.assertIn(
                (ws / "ai-harness" / "harness").as_posix(),
                origin.replace("\\", "/"),
                "attach must execute the workspace clone, got: " + origin,
            )
            # new workspace-level features exist end to end
            self.assertTrue((ws / "AGENTS.md").is_file())
            self.assertTrue((ws / ".ai-harness" / "hooks" / "block_manual_kaggle.py").is_file())

    def test_toolchain_report_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_pin2_") as tmp:
            ws = _make_workspace(Path(tmp))
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}), mock.patch.object(
                sys, "stdout", new=io.StringIO()
            ) as out:
                run_sync(project_root=ws, no_pull=True, no_install=True, no_attach=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertTrue(report.get("toolchain", {}).get("pinned_subprocess"))


class TestRunSync(unittest.TestCase):
    def test_full_sync_clean_local_origin_reports_ok(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sync_run_") as tmp:
            ws = _make_workspace(Path(tmp))
            with mock.patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(ws)}):
                rc = run_sync(project_root=ws, no_install=True)
            # clone source acts as a reachable origin -> pull is a no-op -> ok
            self.assertEqual(rc, 0)
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
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=fw, capture_output=True, text=True, env=env)
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
