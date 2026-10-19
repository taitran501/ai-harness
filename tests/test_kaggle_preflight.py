"""Regression: Kaggle CLI preflight + wait_for_dataset error surface.

Live origin: job_20261019_095919 (arc-prize-2026) died in
``wait_for_dataset`` with an empty error surface — the CLI failed
silently (no stdout/stderr) and the old code raised immediately with
``Kaggle dataset status failed: `` and nothing after the colon. Empty CLI
failures are now retried like other transient errors, and the raised
message always carries exit code, attempt count, and a manual check
command. ``ensure_cli_ready`` fails fast (with install guidance) when the
CLI is missing or broken, before any upload or push.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import harness

from harness.cloud_runner import _warn_if_framework_checkout_dirty
from harness.templates.kaggle_template import KaggleKernelManager


class WaitForKeywordSurfaceTests(unittest.TestCase):
    def setUp(self):
        self._saved_cache = KaggleKernelManager._cli_version_cache
        KaggleKernelManager._cli_version_cache = None

    def tearDown(self):
        KaggleKernelManager._cli_version_cache = self._saved_cache

    def test_empty_failure_is_retried_then_surfaces_with_detail(self):
        failed = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[failed, failed]) as mock_run:
            with self.assertRaisesRegex(
                RuntimeError,
                r"Kaggle dataset status failed \(exit 1\) after 2 attempt\(s\): <no CLI output>",
            ):
                KaggleKernelManager.wait_for_dataset(
                    "owner/test-dataset",
                    poll_interval=0.01,
                    timeout_seconds=30.0,
                    max_consecutive_errors=2,
                )
        self.assertEqual(mock_run.call_count, 2)

    def test_empty_failure_then_ready_succeeds(self):
        failed = MagicMock(returncode=1, stdout="", stderr="")
        ready = MagicMock(returncode=0, stdout='"READY"', stderr="")
        with patch("subprocess.run", side_effect=[failed, ready]) as mock_run:
            KaggleKernelManager.wait_for_dataset(
                "owner/test-dataset", poll_interval=0.01, timeout_seconds=30.0
            )
        self.assertEqual(mock_run.call_count, 2)

    def test_timeout_surfaces_last_error_and_manual_command(self):
        processing = MagicMock(returncode=0, stdout='"PROCESSING"', stderr="")
        with patch("subprocess.run", return_value=processing):
            with self.assertRaisesRegex(
                TimeoutError, r"Check manually: kaggle datasets status owner/test-dataset"
            ):
                KaggleKernelManager.wait_for_dataset(
                    "owner/test-dataset",
                    poll_interval=0.01,
                    timeout_seconds=0.05,
                )


class EnsureCliReadyTests(unittest.TestCase):
    def setUp(self):
        self._saved_cache = KaggleKernelManager._cli_version_cache
        KaggleKernelManager._cli_version_cache = None

    def tearDown(self):
        KaggleKernelManager._cli_version_cache = self._saved_cache

    def test_missing_cli_raises_with_install_guidance(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("kaggle")):
            with self.assertRaisesRegex(
                RuntimeError, r"Kaggle CLI not found on PATH"
            ):
                KaggleKernelManager.ensure_cli_ready()

    def test_broken_cli_raises_with_output(self):
        broken = MagicMock(returncode=2, stdout="", stderr="internal cli crash")
        with patch("subprocess.run", return_value=broken):
            with self.assertRaisesRegex(
                RuntimeError, r"failed on startup \(exit 2\): internal cli crash"
            ):
                KaggleKernelManager.ensure_cli_ready()

    def test_version_is_reported_and_cached(self):
        good = MagicMock(returncode=0, stdout="Kaggle CLI 2.2.4", stderr="")
        with patch("subprocess.run", return_value=good) as mock_run:
            version = KaggleKernelManager.ensure_cli_ready()
            self.assertEqual(version, "Kaggle CLI 2.2.4")
            KaggleKernelManager.ensure_cli_ready()  # served from cache
        self.assertEqual(mock_run.call_count, 1)


class DirtyCheckoutWarningTests(unittest.TestCase):
    """Dispatch must warn when the canonical framework checkout is dirty.

    Live origin: an agent patched the canonical checkout in place during a
    session; the remote bundle then shipped uncommitted edits with no
    visible signal. The dispatch preflight now surfaces this.
    """

    def _run_warning(self, tmp, porcelain_lines, rc=0):
        repo = Path(tmp)
        (repo / ".git").mkdir()
        (repo / "harness").mkdir()
        (repo / "harness" / "__init__.py").write_text("", encoding="utf-8")
        fake_file = str(repo / "harness" / "__init__.py")
        proc = MagicMock(
            returncode=rc,
            stdout="\n".join(porcelain_lines) + ("\n" if porcelain_lines else ""),
            stderr="",
        )
        captured = io.StringIO()
        with patch.object(harness, "__file__", fake_file), \
             patch("subprocess.run", return_value=proc) as mock_run, \
             redirect_stdout(captured):
            _warn_if_framework_checkout_dirty()
        return captured.getvalue(), mock_run

    def test_clean_checkout_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run_warning(tmp, [])
        self.assertNotIn("WARNING", out)

    def test_dirty_checkout_warns_with_change_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run_warning(tmp, [" M harness/runner.py", "?? notes.md"])
        self.assertIn("WARNING: framework checkout is dirty", out)
        self.assertIn("2 uncommitted change(s)", out)

    def test_non_git_directory_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "harness").mkdir()
            (repo / "harness" / "__init__.py").write_text("", encoding="utf-8")
            captured = io.StringIO()
            with patch.object(harness, "__file__", str(repo / "harness" / "__init__.py")), \
                 patch("subprocess.run") as mock_run, redirect_stdout(captured):
                _warn_if_framework_checkout_dirty()
            mock_run.assert_not_called()
        self.assertEqual(captured.getvalue(), "")

    def test_git_failure_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run_warning(tmp, [], rc=128)
        self.assertNotIn("WARNING", out)


if __name__ == "__main__":
    unittest.main()
