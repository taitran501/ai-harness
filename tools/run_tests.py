"""Tiered test runner for the ai-harness suite.

Usage:
    python tools/run_tests.py                 # full suite (all modules)
    python tools/run_tests.py --fast          # fast tier: skip slow modules
    python tools/run_tests.py --list-slow     # show which modules are slow
    python tools/run_tests.py test_x test_y   # run specific modules

The slow tier holds the modules that exercise full pipelines (Kaggle
packaging, dashboard server, git audit gate, legacy end-to-end harness).
Run the full suite before pushing; use --fast during iteration.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Modules whose tests run full pipelines / heavy subprocess work.
#: Measured in-process: ~41s of the ~99s total suite time.
SLOW_MODULES = (
    "test_kaggle_runtime",   # Kaggle kernel packaging + dispatch contracts
    "test_dashboard",        # live dashboard server + HTTP endpoints
    "test_audit_gate",       # git audit gate subprocesses
    "test_harness",          # legacy end-to-end harness pipeline
)


def discover_modules() -> list[str]:
    test_dir = REPO_ROOT / "tests"
    return sorted(p.stem for p in test_dir.glob("test_*.py"))


def build_suite(modules: list[str]) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in modules:
        suite.addTests(loader.loadTestsFromName(f"tests.{name}"))
    return suite


def main(argv: list[str]) -> int:
    args = [a for a in argv if a]
    fast = "--fast" in args
    args = [a for a in args if a != "--fast"]

    if "--list-slow" in args:
        for name in SLOW_MODULES:
            print(name)
        return 0

    if args:
        # Accept "test_x", "tests.test_x", or "tests/test_x.py"
        modules = []
        for a in args:
            stem = Path(a).stem if a.endswith(".py") else a.split(".")[-1]
            modules.append(stem)
    else:
        modules = discover_modules()
        if fast:
            modules = [m for m in modules if m not in SLOW_MODULES]

    label = "fast tier" if fast and not args else "full suite" if not args else "selected modules"
    excluded = f" ({len(SLOW_MODULES)} slow excluded)" if fast and not args else ""
    print(f"Running {label}: {len(modules)} test modules{excluded}")

    t0 = time.time()
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(build_suite(modules))
    elapsed = time.time() - t0
    print(f"\n{result.testsRun} tests in {elapsed:.1f}s — "
          f"{'OK' if result.wasSuccessful() else 'FAILED'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    sys.exit(main(sys.argv[1:]))
