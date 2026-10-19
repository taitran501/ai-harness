"""Regression: custom metrics_evaluator hook references (colon + dot formats).

Live origin: the arc-prize-2026 remote kernel (job_20261019_095919) died in
``validate_run`` with ``AttributeError: module 'src' has no attribute
'evaluator:evaluate_rhae'`` because the integrity gate split the hook on the
last dot while the runner split on the colon. The agent then hand-patched
the canonical checkout in place; those patches were reverted, so this
regression lives in the framework repo instead.
"""

import importlib
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from harness.artifacts import digest, validate_run
from harness.hook_ref import resolve_metrics_hook
from harness.task_spec import UniversalTaskSpec
from tests.evidence_fixtures import seal_fixture
from tests.test_research_continuity import task_document


class ResolveMetricsHookTests(unittest.TestCase):
    def test_colon_format(self):
        self.assertEqual(
            resolve_metrics_hook("src.evaluator:evaluate_rhae"),
            ("src.evaluator", "evaluate_rhae"),
        )

    def test_dot_format(self):
        self.assertEqual(
            resolve_metrics_hook("src.evaluator.evaluate_rhae"),
            ("src.evaluator", "evaluate_rhae"),
        )

    def test_deep_module_with_colon(self):
        self.assertEqual(
            resolve_metrics_hook("a.b.c.fn:score"), ("a.b.c.fn", "score")
        )

    def test_whitespace_tolerated(self):
        self.assertEqual(resolve_metrics_hook("  a.b : c "), ("a.b", "c"))

    def test_invalid_references_raise(self):
        for bad in ("bare", "module:", ":name", "", "   "):
            with self.assertRaises(ValueError, msg=repr(bad)):
                resolve_metrics_hook(bad)


class ValidateRunCustomHookTests(unittest.TestCase):
    """validate_run must accept the hook in both reference formats and
    import the scorer from a project root that is not on sys.path."""

    def setUp(self):
        # Drop a cached test scorer module from an earlier test in this class.
        self._drop_cached_scorer()
        self.temp = tempfile.TemporaryDirectory(prefix="aih-hook-ref-")
        self.root = Path(self.temp.name)
        # Custom scorer module, importable only via this project root.
        (self.root / "src").mkdir()
        (self.root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "src" / "evaluator.py").write_text(
            "def evaluate_rhae(truth, predictions):\n"
            "    return {'rhae': 0.5}\n",
            encoding="utf-8",
        )
        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.run_dir = self.root / "artifacts" / "exp001_hook"
        self.run_dir.mkdir(parents=True)
        # Remember sys.path so the SUT's insertion cannot pollute other tests.
        self._sys_path = list(sys.path)

    def _drop_cached_scorer(self):
        """Remove only ``src.evaluator`` modules whose file is under a
        temp directory owned by this test (never unrelated packages)."""
        prefix = "aih-hook-ref-"
        for mod in list(sys.modules):
            if not (mod == "src" or mod.startswith("src.")):
                continue
            file_ = getattr(sys.modules[mod], "__file__", "") or ""
            if prefix in file_ or not file_:
                del sys.modules[mod]

    def tearDown(self):
        for p in list(sys.path):
            if str(self.root) == str(p):
                sys.path.remove(p)
        sys.path[:] = self._sys_path
        self._drop_cached_scorer()
        self.temp.cleanup()

    def _seal(self, hook_ref, tamper_score=False):
        """Write a sealed run whose resolved config declares ``hook_ref``."""
        cfg = {
            "experiment": {"id": "exp001_hook"},
            "paths": {"metrics_evaluator": hook_ref},
        }
        (self.run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(cfg), encoding="utf-8"
        )
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            fn = importlib.import_module("src.evaluator").evaluate_rhae
            scorer_digest = digest(inspect.getsource(fn))
        finally:
            sys.path.remove(str(self.root))
        seal_fixture(
            self.run_dir,
            self.spec,
            custom_scorer_sha256=scorer_digest if not tamper_score else "deadbeef",
        )

    def _run_validate(self):
        """Run validate_run with the project root NOT already on sys.path,
        resolvable only through HARNESS_PROJECT_ROOT (remote-kernel shape)."""
        env = dict(os.environ)
        env["HARNESS_PROJECT_ROOT"] = str(self.root)
        with patch.dict(os.environ, env, clear=True):
            return validate_run(self.run_dir)

    def test_colon_hook_passes_when_root_not_on_sys_path(self):
        self._seal("src.evaluator:evaluate_rhae")
        self.assertNotIn(str(self.root), sys.path)
        manifest = self._run_validate()
        # The SUT had to insert the project root to import the hook.
        self.assertIn(str(self.root), sys.path)
        self.assertTrue(manifest["custom_scorer_sha256"])

    def test_dot_hook_passes(self):
        self._seal("src.evaluator.evaluate_rhae")
        self.assertNotIn(str(self.root), sys.path)
        self._run_validate()

    def test_tampered_scorer_digest_rejected(self):
        self._seal("src.evaluator:evaluate_rhae", tamper_score=True)
        with self.assertRaisesRegex(ValueError, "Custom scorer"):
            self._run_validate()


if __name__ == "__main__":
    unittest.main()
