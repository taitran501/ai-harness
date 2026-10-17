"""Regression tests for the single-install architecture.

Covers the pieces that replace the per-workspace clone + retargeted editable
install with one shared canonical checkout:

- ``find_framework_root`` resolution order (``HARNESS_FRAMEWORK_DIR`` >
  workspace clone > None)
- the toolchain pin targeting the framework root, and NOT re-execing when the
  loaded origin already IS the framework root (the re-exec tax disappears)
- ``run_sync`` canonical mode: pull the shared checkout but do NOT retarget
  the global editable install (no per-sync ``pip install -e``), and record a
  provenance pointer in the manifest
- ``status`` framework section: single-install flag + stale-install advisory
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

from harness.project import read_framework_pointer
from harness.status_report import _framework_section
from harness.sync import run_sync
from harness.toolchain import (
    FRAMEWORK_DIR_ENV,
    _needs_pin,
    ensure_pinned_toolchain,
    find_framework_root,
)


def _make_ws_clone(ws: Path) -> Path:
    """A per-workspace clone (legacy layout) that contains a harness package.

    ``run_sync`` auto-detection additionally requires a ``.git`` marker, so it
    is created here (an empty dir is enough for the resolution check)."""
    clone = ws / "ai-harness"
    (clone / "harness").mkdir(parents=True)
    (clone / ".git").mkdir()
    (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
    return clone


def _make_canonical(tmp: Path, name: str = "canonical") -> Path:
    """A canonical checkout OUTSIDE the workspace that contains a harness pkg."""
    canonical = tmp / name
    (canonical / "harness").mkdir(parents=True)
    (canonical / "harness" / "__init__.py").write_text("", encoding="utf-8")
    return canonical


class TestFindFrameworkRoot(unittest.TestCase):
    def test_env_var_precedence_over_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_root_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            canonical = _make_canonical(tmp)
            with mock.patch.dict(os.environ, {FRAMEWORK_DIR_ENV: str(canonical)}):
                self.assertEqual(find_framework_root(ws), canonical)

    def test_fallback_to_clone_when_no_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_fallback_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            clone = _make_ws_clone(ws)
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(find_framework_root(ws), clone)

    def test_none_when_nothing_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_none_") as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertIsNone(find_framework_root(ws))

    def test_invalid_env_falls_back_to_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_invalid_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            clone = _make_ws_clone(ws)
            bad = tmp / "not-a-framework"
            bad.mkdir()  # no harness/ package
            with mock.patch.dict(os.environ, {FRAMEWORK_DIR_ENV: str(bad)}):
                self.assertEqual(find_framework_root(ws), clone)


class TestCanonicalPin(unittest.TestCase):
    def test_pin_targets_canonical_when_env_set(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sip_pin_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            canonical = _make_canonical(tmp)
            fake_origin = tmp / "elsewhere" / "harness" / "__init__.py"
            captured: dict = {}

            def fake_runner(cmd, env=None, cwd=None):
                captured["env"] = env
                return subprocess.CompletedProcess(cmd, 0)

            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin), \
                    mock.patch.dict(os.environ, env, clear=False):
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=fake_runner)
            self.assertIsNotNone(result)
            self.assertTrue(captured["env"]["PYTHONPATH"].startswith(str(canonical)))

    def test_no_pin_when_origin_is_already_canonical(self) -> None:
        """The re-exec tax disappears: origin == framework root -> no re-exec."""
        with tempfile.TemporaryDirectory(prefix="test_sip_nopinx_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            canonical = _make_canonical(tmp)
            # origin is exactly the canonical harness __init__
            canonical_origin = (canonical / "harness" / "__init__.py").resolve()
            calls = {"n": 0}

            def counting_runner(cmd, env=None, cwd=None):  # pragma: no cover
                calls["n"] += 1
                return subprocess.CompletedProcess(cmd, 0)

            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch("harness.toolchain.current_origin", return_value=canonical_origin), \
                    mock.patch.dict(os.environ, env, clear=False):
                needs, _o, _e = _needs_pin(ws)
                self.assertFalse(needs)
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=counting_runner)
            self.assertIsNone(result)
            self.assertEqual(calls["n"], 0)


class TestCanonicalSync(unittest.TestCase):
    def test_canonical_mode_skips_pip(self) -> None:
        """Canonical repo outside the workspace -> mode=canonical, no pip."""
        with tempfile.TemporaryDirectory(prefix="test_sis_nopip_") as tmp:
            tmp = Path(tmp)
            canonical = _make_canonical(tmp)
            ws = tmp / "workspace"
            (ws / "configs").mkdir(parents=True)
            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                rc = run_sync(project_root=ws, no_pull=True, no_attach=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(report["mode"], "canonical")
            install = report["steps"]["install"]
            self.assertTrue(install.get("skipped"))
            self.assertIn("canonical", install.get("reason", ""))

    def test_workspace_clone_mode_when_no_env(self) -> None:
        """No canonical env -> legacy workspace-clone mode is detected."""
        with tempfile.TemporaryDirectory(prefix="test_sis_clone_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "workspace"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                rc = run_sync(project_root=ws, no_pull=True, no_install=True,
                              no_attach=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(report["mode"], "workspace-clone")

    def test_canonical_sync_writes_provenance_pointer(self) -> None:
        """End-to-end: a real (local) canonical checkout is pulled, NOT
        reinstalled, and the manifest records a path+commit pointer."""
        import harness as _pkg

        with tempfile.TemporaryDirectory(prefix="test_sis_ptr_") as tmp:
            tmp = Path(tmp)
            canonical = tmp / "canonical"
            git_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            }
            subprocess.run(
                ["git", "clone", "--quiet",
                 str(Path(_pkg.__file__).resolve().parents[1]), str(canonical)],
                capture_output=True, text=True, check=True, env=git_env,
            )
            ws = tmp / "workspace"
            (ws / "configs").mkdir(parents=True)
            (ws / "configs" / "task.yaml").write_text(
                "task:\n  name: SingleInstall\n  primary_metric: accuracy\n",
                encoding="utf-8",
            )
            env = {
                FRAMEWORK_DIR_ENV: str(canonical),
                "HARNESS_PROJECT_ROOT": str(ws),
            }
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                rc = run_sync(project_root=ws, no_pull=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0, "canonical sync should succeed")
            self.assertEqual(report["mode"], "canonical")
            self.assertTrue(report["steps"]["install"].get("skipped"),
                            "canonical sync must not retarget the editable install")
            ptr = report.get("framework_pointer")
            self.assertIsNotNone(ptr)
            self.assertEqual(Path(ptr["path"]).resolve(), canonical.resolve())
            self.assertTrue(ptr.get("commit"), "pointer should record a commit")
            manifest = ws / ".ai-harness" / "project.yaml"
            self.assertTrue(manifest.is_file())
            got = read_framework_pointer(manifest)
            self.assertEqual(Path(got["path"]).resolve(), canonical.resolve())
            self.assertEqual(got.get("commit"), ptr.get("commit"))


class TestStatusFrameworkSection(unittest.TestCase):
    def test_single_install_true_for_canonical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_status_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            canonical = _make_canonical(tmp)
            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch.dict(os.environ, env):
                section = _framework_section(str(ws))
            self.assertTrue(section["single_install"])
            self.assertEqual(Path(section["resolved_framework"]).resolve(), canonical.resolve())

    def test_single_install_false_for_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_status2_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                section = _framework_section(str(ws))
            self.assertFalse(section["single_install"])
            self.assertEqual(Path(section["resolved_framework"]).resolve(),
                             (ws / "ai-harness").resolve())

    def test_advisory_when_import_origin_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_adv_") as tmp:
            tmp = Path(tmp)
            ws = tmp / "ws"
            (ws / "configs").mkdir(parents=True)
            canonical = _make_canonical(tmp)
            stale = tmp / "stale" / "harness" / "__init__.py"
            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch.dict(os.environ, env), \
                    mock.patch("harness.toolchain.current_origin", return_value=stale):
                section = _framework_section(str(ws))
            self.assertIn("pip install -e", section["advisory"])


if __name__ == "__main__":
    unittest.main()
