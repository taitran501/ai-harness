# Test-infrastructure: correct slow tier, subpackage discovery, pytest config

- **Date**: 2026-10-18
- **PR**: [#101](https://github.com/taitran501/ai-harness/pull/101) (open)
- **Branch**: `fix/test-infra-tiers`
- **Status**: open
- **Commits**: — (recorded at merge)

## Why

Tech-debt audit of the test suite (2026-10-18) found that the `--fast`
tier of `tools/run_tests.py` was mis-tiered and mis-documented:

- `test_sync_command` (measured 27.9 s standalone, 9 real `git clone`
  subprocesses) and `test_cloud_jobs_and_env` (11.5 s, subprocess work)
  sat in the **fast** tier even though they are slower than two of the
  four declared SLOW modules. Every `--fast` run paid ~39 s for heavy
  subprocess tests it was supposed to skip.
- The SLOW-module comment labels were stale/wrong: `test_audit_gate`
  was labeled "git audit gate subprocesses" (it is 20/23 pure numeric
  tests + 3 CLI receipt tests) and `test_harness` was labeled "legacy
  end-to-end harness pipeline" (it is TreeJournal/Validator/AuditGate
  unit tests + 2 audit-gate CLI subprocesses).
- The docstring's timing claim ("~41 s of the ~99 s total suite time")
  pre-dated suite growth (full suite now measures ~210 s in-process).
- `discover_modules()` used a non-recursive glob and the CLI arg parser
  only understood top-level modules — a blocker for the planned
  `tests/agent_bridge/` package consolidation (separate PR).
- Every plain `pytest` run printed a `pytest-asyncio` deprecation
  warning (`asyncio_default_fixture_loop_scope` unset) even though the
  suite contains no async tests.

## What changed

**`tools/run_tests.py`**
- `SLOW_MODULES` now has 6 members: added `test_sync_command` and
  `test_cloud_jobs_and_env`; labels corrected to describe what each
  module actually does; membership comment now cites the 2026-10-18
  standalone measurements (sync ~28 s, kaggle_runtime ~27 s, dashboard
  ~12 s, cloud_jobs ~12 s, harness ~7 s, audit_gate ~7 s).
- Module names are now dotted (`tests.test_x`): `discover_modules()`
  recurses with `rglob` so test modules in subpackages are discovered
  (preparation for `tests/agent_bridge/`), and `build_suite()` loads
  dotted names directly.
- CLI arg parsing accepts `test_x`, `tests.test_x`, `tests/test_x.py`,
  and nested `tests/subpkg/test_y.py` / `tests.subpkg.test_y`.
- Slow/smoke membership is compared by module stem, so top-level tier
  lists keep working with dotted discovery.

**`pyproject.toml`**
- New `[tool.pytest.ini_options]` with
  `asyncio_default_fixture_loop_scope = "function"` — silences the
  session-wide pytest-asyncio deprecation warning and pins loop-scope
  behavior for any future async fixture.

**`tests/test_sync_command.py`**
- The two `TestFindFrameworkRepo` tests that only exercise
  `find_framework_repo()` path detection no longer build a real throwaway
  `git clone` (2.7 s each): `find_framework_repo` only checks directory
  shape (`.git` marker + `harness/` package), so a stub workspace with
  those markers is a faithful input. The 7 attach/sync tests keep real
  clones because their subprocesses genuinely execute the cloned code.

## Approach

- **Re-tier from measurement, not feel** — the measurement script
  (`tools/timing/_measure_modules.py`) produced per-module standalone
  timings; the two slowest non-SLOW modules were promoted. This is the
  first time `test_sync_command`'s subprocess cost is accounted for in
  the tier.
- **Discovery change is inert for a flat suite** — dotted names resolve
  to the exact same 87 modules as before; verified by full-suite run.
  It lands in this PR (not the consolidation PR) so the consolidation
  stays a pure mechanical rename.
- **Stub only where the code under test ignores git internals** —
  verified in `harness/sync.py:find_framework_repo` (existence checks
  only). Where the subprocess runs the clone's code, real clones are
  kept; no semantic weakening.
- Alternatives considered: (a) uninstalling pytest-asyncio — rejected,
  the config line is one line and preserves compatibility if async
  tests ever appear; (b) cloning from a cached module-level source clone
  for the attach/sync tests — rejected, local-clone cost is dominated by
  checkout and the hardlink-sharing risk outweighs the saving.

## Evidence

- Full suite: `python tools/run_tests.py` → 908 tests OK (module count
  unchanged: 87 discovered, all dotted names).
- `--fast` tier: 81 modules (6 slow excluded) vs 83 before.
  Before: 832 tests in 178.3 s. After: **813 tests in 139.9 s**
  (−38.4 s per fast run; the 95 excluded tests are the sync +
  cloud-jobs modules).
- Full suite in the same verification run: 908 tests in 161.2 s — OK
  (1 pre-existing skip, unchanged).
- `tests/test_sync_command.py` standalone: 27.9 s → 14.7 s (11 tests OK)
  after stubbing the two detection clones.
- `python -m pytest tests/test_dashboard.py -q` — pytest-asyncio
  deprecation warning no longer emitted.
- `python tools/run_tests.py tests/test_sync_command.py` and
  `tests.test_sync_command` both resolve to the same module (arg parsing).
- `ruff check tools/ harness/` clean.

## Impact & follow-ups

- `--fast` iteration loop is materially faster and actually skips the
  heavy subprocess tests it was documented to skip.
- `docs/WORK_LOG.md` (archived) still references the old flat names;
  historical, intentionally untouched.
- Follow-up: `tests/agent_bridge/` consolidation PR (B) can now move
  13 agent-domain files into a subpackage with no further runner changes.
