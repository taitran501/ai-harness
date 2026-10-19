"""Scorer hook imports must be hermetic on artifact-import paths (gap G1).

Regression (arc-prize-2026 audit): two remote kernels finished training
COMPLETE, but the local output import died with a bare ``No module named
'src'``. The import happens in ``validate_run`` inside a *detached* process
(agent CLI, journal sync, repair) that has no ambient workspace context —
``resolve_project_root()`` then falls back to the framework checkout (which
has ``configs/task.yaml`` but no workspace ``src/``), so the scorer hook
cannot resolve.

Defenses pinned here:

1. ``migrate_downloaded_artifacts(..., project_root=ws)`` puts the workspace
   on ``sys.path`` deterministically (explicit root takes precedence over
   env/resolved fallbacks) — imports succeed from any cwd.
2. When the scorer still cannot be imported, ``validate_run`` raises a
   ``RuntimeError`` naming the hook reference, the ``HARNESS_PROJECT_ROOT``
   env value, and a ``sys.path`` excerpt — no bare ModuleNotFoundError.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from harness.artifacts import digest, validate_run
from harness.hook_ref import ensure_hook_project_roots_on_sys_path
from harness.templates.kaggle_template import KaggleKernelManager
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task

PKG = "scorer_pkg"  # unique name: no collision with ambient packages


def _write_scorer(ws: Path) -> str:
    """Create ``ws/scorer_pkg/evaluator.py``; return the sealed scorer digest."""
    (ws / PKG).mkdir()
    (ws / PKG / "__init__.py").write_text("", encoding="utf-8")
    (ws / PKG / "evaluator.py").write_text(
        "def evaluate_rhae(y_true, y_pred, **kwargs):\n"
        "    pairs = list(zip(y_true, y_pred))\n"
        "    return sum(a == b for a, b in pairs) / max(len(pairs), 1)\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        f"aih_test_{PKG}_evaluator", ws / PKG / "evaluator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return digest(inspect.getsource(module.evaluate_rhae))


def _sealed_run_with_hook(download_dir: Path, scorer_sha: str) -> Path:
    """A valid sealed run whose run_config declares a custom scorer hook."""
    run_dir = download_dir / "exp001"
    run_dir.mkdir()
    (run_dir / "run_config.yaml").write_text(
        "experiment: {}\n"
        f"paths:\n  metrics_evaluator: '{PKG}.evaluator:evaluate_rhae'\n",
        encoding="utf-8",
    )
    seal_fixture(run_dir, task(), score=0.75, custom_scorer_sha256=scorer_sha)
    return run_dir


class _DetachedContext:
    """cwd in a bare temp dir, HARNESS_PROJECT_ROOT cleared, sys.modules clean."""

    def __init__(self):
        self.cwd = os.getcwd()
        self.env_value = os.environ.pop("HARNESS_PROJECT_ROOT", None)
        self.detached = Path(tempfile.mkdtemp(prefix="aih-detached-"))
        self.popped = {
            name: sys.modules.pop(name)
            for name in (PKG, f"{PKG}.evaluator")
            if name in sys.modules
        }
        os.chdir(self.detached)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        if self.env_value is not None:
            os.environ["HARNESS_PROJECT_ROOT"] = self.env_value
        sys.modules.update(self.popped)
        shutil.rmtree(self.detached, ignore_errors=True)
        return False


class ScoringImportHermeticTests(unittest.TestCase):
    def test_migrate_with_project_root_imports_scorer_from_detached_cwd(self):
        ws, _ = make_workspace()
        try:
            scorer_sha = _write_scorer(ws)
            with _DetachedContext() as ctx:
                # Precondition: the scorer is NOT importable from here.
                self.assertIsNone(importlib.util.find_spec(PKG))
                self.assertNotIn(str(ws), sys.path)
                download = ctx.detached / "download"
                download.mkdir()
                _sealed_run_with_hook(download, scorer_sha)
                target = ws / ".ai-harness" / "artifacts" / "exp001"
                migrated = KaggleKernelManager.migrate_downloaded_artifacts(
                    download, target, project_root=ws
                )
                self.assertIn("metrics.json", migrated)
                self.assertIn("run_config.yaml", migrated)
                self.assertTrue((target / "artifact_manifest.json").is_file())
                self.assertIn(str(ws), sys.path)
        finally:
            sys.modules.pop(PKG, None)
            sys.modules.pop(f"{PKG}.evaluator", None)
            if str(ws) in sys.path:
                sys.path.remove(str(ws))
            shutil.rmtree(ws, ignore_errors=True)

    def test_validate_run_failure_message_names_hook_and_context(self):
        ws, _ = make_workspace()  # workspace WITHOUT the scorer package
        try:
            run_dir = None
            with _DetachedContext() as ctx:
                download = ctx.detached / "download"
                download.mkdir()
                # Seal against a scorer digest computed from a temp module —
                # the hash value is irrelevant; the import must fail first.
                run_dir = _sealed_run_with_hook(download, digest("any-source"))
                with self.assertRaises(RuntimeError) as ctx_err:
                    validate_run(run_dir)
                message = str(ctx_err.exception)
                self.assertIn(f"{PKG}.evaluator:evaluate_rhae", message)
                self.assertIn(f"module='{PKG}.evaluator'", message)
                self.assertIn("project_root=", message)
                self.assertIn("sys.path", message)
                self.assertIsInstance(ctx_err.exception.__cause__, ModuleNotFoundError)
        finally:
            sys.modules.pop(PKG, None)
            sys.modules.pop(f"{PKG}.evaluator", None)
            shutil.rmtree(ws, ignore_errors=True)

    def test_ensure_hook_project_roots_prefers_explicit_root(self):
        explicit = Path(tempfile.mkdtemp(prefix="aih-explicit-"))
        fallback = Path(tempfile.mkdtemp(prefix="aih-fallback-"))
        try:
            ensure_hook_project_roots_on_sys_path(project_root=explicit)
            self.assertIn(str(explicit), sys.path)
            # Explicit root must sit ahead of any ambient candidate: it was
            # inserted last with insert(0), so it is the first sys.path entry.
            self.assertEqual(sys.path[0], str(explicit))
        finally:
            for p in (explicit, fallback):
                shutil.rmtree(p, ignore_errors=True)
            if str(explicit) in sys.path:
                sys.path.remove(str(explicit))


if __name__ == "__main__":
    unittest.main()
