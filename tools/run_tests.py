"""Tiered test runner for the ai-harness suite.

Usage:
    python tools/run_tests.py                 # full suite (all modules)
    python tools/run_tests.py --smoke         # smoke tier: <10s critical invariants
    python tools/run_tests.py --fast          # fast tier: skip slow modules
    python tools/run_tests.py --list-slow     # show which modules are slow
    python tools/run_tests.py --list-smoke    # show which modules are smoke
    python tools/run_tests.py --no-lock       # disable the concurrency guard
    python tools/run_tests.py test_x test_y   # run specific modules

Concurrency guard: a suite acquires an advisory lock scoped to this checkout
so two ``run_tests.py`` invocations can never run at once on the same tree.
Concurrent suites contend for ``__pycache__`` / cache files and produce
spurious ``PermissionError`` / flaky failures that are not real regressions.
The guard fails fast (exit 3) when another suite holds the lock; pass
``--no-lock`` to bypass (not recommended).

The slow tier holds the modules that exercise full pipelines / heavy
subprocess work (Kaggle packaging, sync CLI against throwaway git clones,
dashboard server, cloud-job runners, audit-gate CLI).
Run the full suite before pushing; use --fast during iteration.

The smoke tier is the <10 s critical-invariants set for rapid feedback.
It covers the framework core (correctness contract + metric reference,
label-integrity audit, anti-hallucination grounding gate, research-memory
scope stability, backend-lock policy) in a single process, so the heavy
pandas/sklearn/pyarrow import cost is paid exactly once. Module membership
is chosen from per-module timing measurement, not by feel.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Modules whose tests run full pipelines / heavy subprocess work.
#: Membership from per-module timing measurement (2026-10-18, standalone
#: runs incl. cold start): kaggle_runtime ~27s, sync_command ~15s (after
#: detection-test clone stubbing), dashboard ~12s, cloud_jobs_and_env
#: ~12s, harness ~7s, audit_gate ~7s.
SLOW_MODULES = (
    "test_kaggle_runtime",      # Kaggle kernel packaging + dispatch contracts
    "test_dashboard",           # live dashboard server + HTTP endpoints
    "test_audit_gate",          # audit-gate numerics (diff metrics, t-test, integrity) + CLI receipt tests
    "test_harness",             # TreeJournal/Validator/AuditGate unit tests + audit-gate CLI subprocesses
    "test_sync_command",        # sync CLI against throwaway git clones (heavy subprocess work)
    "test_cloud_jobs_and_env",  # cloud job runner + environment cache (subprocess work)
)

#: Critical-invariants subset for the <10 s smoke tier, chosen by
#: per-module timing measurement (tools/timing/_measure_modules.py).
#: In-process total is below the per-module pytest sum because the
#: heavy shared imports (pandas/sklearn/pyarrow) are paid once.
SMOKE_MODULES = (
    "test_correctness_contracts",    # correctness contract + metric reference
    "test_correctness_candidates",   # label-integrity audit (missing/duplicate)
    "test_grounding_gate",           # anti-hallucination verified_via gate
    "test_research_scope_stability", # research-memory scope stability
    "test_execution_backend_policy", # kaggle backend lock policy
)


def discover_modules() -> list[str]:
    """All test modules as dotted names (``tests.test_x``).

    Recursive: modules may live in subpackages
    (``tests/agent_bridge/test_y.py`` -> ``tests.agent_bridge.test_y``).
    """
    test_dir = REPO_ROOT / "tests"
    return sorted(
        p.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", ".")
        for p in test_dir.rglob("test_*.py")
    )


def build_suite(modules: list[str]) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in modules:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


#: Slow/smoke membership is keyed by module stem (top-level test files).
def _stem(module: str) -> str:
    return module.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Concurrency guard: prevent two run_tests.py suites on the same checkout.
# ---------------------------------------------------------------------------


class _ConcurrentSuiteError(RuntimeError):
    """Raised when another suite already holds the per-checkout lock."""


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check. Fails safe (returns True) when it cannot
    determine, so a live lock is never stolen by mistake."""
    if pid is None or pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    # Windows
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    except Exception:  # noqa: BLE001 - fail safe on any ctypes/platform quirk
        return True


class _SuiteLock:
    """Advisory lock preventing concurrent suites on the same checkout.

    The lock lives in the OS temp dir (never the working tree) and is scoped
    to the checkout by a hash of the repo root. It is claimed with an atomic
    exclusive create; a crashed/finished holder is detected via its recorded
    PID + age and its lock is stolen, so a stuck run can never wedge the
    suite permanently.
    """

    MAX_AGE_SECONDS = 60 * 60  # any legitimate run is far shorter than 1 h
    RETRIES = 5

    def __init__(self, path: Path) -> None:
        self.path = path

    def acquire(self) -> None:
        for _ in range(self.RETRIES):
            if self._try_claim():
                return
            time.sleep(0.2)
        self._raise_if_live()

    def _try_claim(self) -> bool:
        try:
            with self.path.open("x", encoding="utf-8") as fd:  # atomic create
                fd.write(json.dumps({
                    "pid": os.getpid(),
                    "ts": time.time(),
                    "host": platform.node(),
                }))
            atexit.register(self.release)
            return True
        except FileExistsError:
            self._maybe_steal()  # remove a dead/stale lock so we can re-claim
            return False

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - unreadable/corrupt lock is treated as stale
            return {}

    def _maybe_steal(self) -> None:
        data = self._read()
        pid = data.get("pid")
        age = time.time() - float(data.get("ts", 0.0))
        if (not _pid_alive(pid)) or age > self.MAX_AGE_SECONDS:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _raise_if_live(self) -> None:
        data = self._read()
        pid = data.get("pid")
        age = int(time.time() - float(data.get("ts", 0.0)))
        raise _ConcurrentSuiteError(
            f"Another run_tests.py suite is already running on this checkout "
            f"(pid {pid}, ~{age}s ago, host {data.get('host', '?')}; lock: {self.path}). "
            f"Concurrent suites cause file-lock contention and corrupt results. "
            f"Wait for it to finish, or pass --no-lock to bypass this guard."
        )

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _suite_lock_path() -> Path:
    sig = hashlib.sha1(str(REPO_ROOT).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"ai-harness-run-tests-{sig}.lock"


def main(argv: list[str]) -> int:
    # Isolate the test environment: drop ANY globally-set HARNESS_* / AI_HARNESS_*
    # var so the suite runs in a clean environment. The suite is written to run in
    # a clean environment — tests that need a specific var set it explicitly per test
    # (via mock.patch.dict or direct os.environ assignment). Without this, a machine
    # with harness env vars set globally (e.g. the one-time single-install migration
    # sets HARNESS_FRAMEWORK_DIR) contaminates the suite with false failures
    # unrelated to the code under test. The prefix sweep is systemic: a future new
    # env var is isolated automatically without touching each test module.
    for k in [k for k in os.environ if k.startswith(("HARNESS_", "AI_HARNESS_"))]:
        del os.environ[k]
    args = [a for a in argv if a]
    fast = "--fast" in args
    args = [a for a in args if a != "--fast"]
    smoke = "--smoke" in args
    args = [a for a in args if a != "--smoke"]
    no_lock = "--no-lock" in args
    args = [a for a in args if a != "--no-lock"]

    if "--list-slow" in args:
        for name in SLOW_MODULES:
            print(name)
        return 0

    if "--list-smoke" in args:
        for name in SMOKE_MODULES:
            print(name)
        return 0

    # Guard against concurrent suites on this checkout (file-lock contention
    # corrupts results). --no-lock disables the guard.
    if not no_lock:
        try:
            _SuiteLock(_suite_lock_path()).acquire()
        except _ConcurrentSuiteError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3

    if args:
        # Accept "test_x", "tests.test_x", "tests/test_x.py",
        # or nested paths like "tests/subpkg/test_y.py" / "tests.subpkg.test_y".
        modules = []
        for a in args:
            s = a.replace("/", ".")
            if s.endswith(".py"):
                s = s[: -len(".py")]
            if not s.startswith("tests."):
                s = f"tests.{s}"
            modules.append(s)
    elif smoke:
        modules = [f"tests.{m}" for m in SMOKE_MODULES]
    else:
        modules = discover_modules()
        if fast:
            modules = [m for m in modules if _stem(m) not in SLOW_MODULES]

    if smoke and not args:
        label = "smoke tier (<10s invariants)"
    elif fast and not args:
        label = "fast tier"
    elif not args:
        label = "full suite"
    else:
        label = "selected modules"
    excluded = f" ({len(SLOW_MODULES)} slow excluded)" if fast and not smoke and not args else ""
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
