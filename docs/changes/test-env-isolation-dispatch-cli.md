# Test env isolation (conftest sweep) + `kernel_dispatch.main()` CLI tests

**PR:** #93 · **Branch:** `fix/test-env-isolation-and-dispatch-cli` · **Date:** 2026-10-18
**Status:** Implemented

## Why

Two audit findings (post-PR #92 codebase audit), both low-risk hardening:

- **Plain `pytest tests/` was not equivalent to `tools/run_tests.py`.** The
  canonical runner sweeps every `HARNESS_*` / `AI_HARNESS_*` env var before
  running (the suite is written for a clean environment; tests that need a var
  set it per-test). A bare `pytest` invocation inherits the ambient shell —
  on this machine `HARNESS_FRAMEWORK_DIR` (single-install migration) is set
  globally, so 12 env-sensitive tests (framework discovery, sync, adoption,
  status) false-failed under bare pytest while the canonical run passed
  888/888. An agent running bare `pytest` would see "12 failed" and risk a
  wrong diagnosis/fix.
- **`kernel_dispatch.main()` (the `ai-harness dispatch-kernel` entry point)
  had zero direct tests** — the CLI arg-parsing, JSON-on-stdout contract, and
  exit-code behaviour (the part an agent consumes) were uncovered, leaving
  `kernel_dispatch.py` at 60% coverage.

## What

- **`conftest.py`** (root): after the existing torch-seed bootstrap, sweep all
  `HARNESS_*` / `AI_HARNESS_*` vars (same prefix rule as `run_tests.py`), then
  re-apply the one var the suite itself relies on
  (`HARNESS_DISABLE_TORCH_SEED`). `run_tests.py` uses unittest, so this file
  only affects pytest — the two runners are now behaviourally identical.
- **`tests/test_kernel_dispatch.py`**: new `KernelDispatchMainCliTests` —
  `main()` happy path (exit 0 + valid JSON with `ok`/`ref`), failure path
  (dispatch exception → exit 1 + `{"ok": false, "error": ...}`), full flag
  passthrough (code-file/machine-shape/dataset/competition/kernel-source/
  model/keyword/enable-internet/timeout/keep-staging forwarded to
  `dispatch_kernel`), and missing-required-arg → argparse exit 2 with no
  dispatch call.
- Module docstring updated to reflect the new coverage.

## Approach

Defense-in-depth rather than per-test patching: the conftest sweep is
systemic (a future env var is isolated automatically, exactly mirroring the
runner's documented contract). The `main()` tests mock only
`dispatch_kernel` (no staging/client/network) and pin the JSON contract the
agent reads, matching the file's existing offline-test convention.

## Evidence

- Reproduced the failure mode before the fix: bare `pytest` with
  `HARNESS_FRAMEWORK_DIR` set → 12 failed across `test_status_report` /
  `test_sync_command` / `test_toolchain_adoption`; after the conftest sweep
  the same three files → 35/35 passed with the var still set.
- New `main()` tests: 4 pass (targeted `test_kernel_dispatch.py` → 15 passed).
- Gates: ruff F,E7 clean; fast + full suite OK (see commit / CI on this
  branch).

## Impact

- Any `pytest` invocation (agent or human) now runs in the same clean
  environment as the canonical runner — false failures from ambient harness
  env vars are eliminated at the suite level.
- `kernel_dispatch.py` CLI contract is regression-pinned: a broken
  argparse/JSON/exit-code path in `dispatch-kernel` will fail a test instead
  of surfacing as a silent agent-visible error.
