import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness.toolchain import FrameworkConfigurationError, ensure_pinned_toolchain, resolve_framework_dir
from harness.sync import find_framework_repo


class FrameworkResolutionTests(unittest.TestCase):
    def test_invalid_explicit_sources_fail_before_pin_or_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clone = root / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness/__init__.py").touch()
            (root / "file").touch()
            for value in ("", "  ", "absent", "file", "."):
                with self.subTest(value=value), patch.dict(os.environ, {"HARNESS_FRAMEWORK_DIR": value, "AI_HARNESS_TOOLCHAIN_NO_PIN": "1"}):
                    runner = Mock()
                    with self.assertRaises(FrameworkConfigurationError):
                        ensure_pinned_toolchain(root, "harness.cli", runner=runner)
                    runner.assert_not_called()
            with self.assertRaises(FrameworkConfigurationError):
                resolve_framework_dir(root, explicit_dir="bad\x00path")

    def test_relative_override_and_worktree_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "framework/harness").mkdir(parents=True)
            (root / "framework/harness/__init__.py").touch()
            (root / "framework/.git").write_text("gitdir: elsewhere", encoding="utf-8")
            with patch.dict(os.environ, {"HARNESS_FRAMEWORK_DIR": "bad"}):
                expected = (root / "framework").resolve()
                self.assertEqual(resolve_framework_dir(root, explicit_dir="framework"), expected)
                self.assertEqual(find_framework_repo(root, "framework"), expected)
