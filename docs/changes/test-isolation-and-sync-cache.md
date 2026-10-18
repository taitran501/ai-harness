# Test isolation + research sync cache integrity

- **Date**: 2026-10-18
- **PR**: #95 (open)
- **Branch**: `fix/test-isolation-and-sync-cache`
- **Status**: open
- **Commits**: `<sha>` (squash)

## Why

An independent review of the framework (verified against `4ad6c1f`) found two
integrity bugs the internal performance audit had missed:

1. **T1 — test suite leaked artifact state into the real checkout.**
   `TestPerSubtaskDirection` in `tests/test_p2_fixes.py` constructed
   `TreeJournal(auto_sync=False)` without explicit paths. `TreeJournal`'s
   default path resolution falls back to `HarnessProject.discover()` → package
   root when nothing is attached to the CWD, so every full-suite run wrote
   `artifacts/journal.json` (and locks) into the framework checkout. The leak
   was invisible to `git status` because `artifacts/` is gitignored — it can
   only be detected by snapshotting artifact state before/after the suite.
2. **C3 — `ResearchMemory.sync()` served stale `verified_full` claims.**
   The module-level `_SYNC_CACHE` keyed on `(folder, mtime of
   manifest/config/metrics)`. Tampering any *other* sealed artifact (e.g. OOF
   predictions) between two `sync()` calls left the cached
   `verified_full` record in place even though a direct `validate_run`
   rejected the run — the HUD disagreed with the gate. The key also omitted
   the validating contract, so a second task spec in the same process reused
   the first contract's cached record.

## What changed

- `tests/test_p2_fixes.py` — `TestPerSubtaskDirection` now passes explicit
  `journal_path`/`artifacts_dir` (under the temp workspace) and the test's
  own `task_spec`, so no project discovery can reach the checkout.
- `conftest.py` (repo root) — new session-scoped autouse guard: snapshots the
  checkout's `artifacts/` + `.ai-harness/` file sets at session start and
  fails the suite at session end if the suite itself created any files there.
  Pre-existing files (a dev who attached this checkout) are the baseline, not
  a failure. This is the systemic version of the T1 regression test: any
  future test that leaks artifact state fails loudly.
- `harness/research_memory.py` — `sync()` cache key is now
  `(absolute folder path, contract identity digest, per-file
  (name, mtime_ns, size) tuple over every top-level artifact file)`:
  - any artifact file change (not just manifest/config/metrics) invalidates
    the cached claim and forces a fresh `validate_run`;
  - the identity digest (`digest(contract_identity(spec, plugin))`,
    precomputed in `__init__`) makes cached records per-contract, so a second
    spec/plugin in the same process can never inherit the first contract's
    claim;
  - mtime+size mirrors the existing `artifacts._HASH_CACHE` idiom; the
    docstrings now state the actual cache semantics (the old docstring
    claimed "revalidate on every read").
- `tests/test_research_sync_cache.py` — 4 regression tests: tampered-OOF
  scenario (reviewer C3), cross-contract staleness, cache-hit preservation
  (no perf regression: one `validate_run` per unchanged folder), and
  metrics.json invalidation.

## Approach

- **T1 fix in the test, not in `TreeJournal`'s default resolution.** Changing
  the default path would alter real-agent behavior (discovery fallback is
  intentional for attached-less workspaces); the anti-pattern is
  construction without explicit paths *in tests*. The conftest guard makes
  the invariant enforceable suite-wide without touching runtime code.
  Swept `tests/` for `TreeJournal(` calls: this was the only instance
  (others already pass explicit `journal_path`/`artifacts_dir`).
- **C3 kept the cache and strengthened the key** instead of deleting it:
  one `iterdir` + per-file `stat` per run folder is microseconds-to-low-ms
  versus the cost of `validate_run` (file hashes + yaml load). Alternatives
  considered and rejected: (a) content-hash key — pays exactly the cost the
  cache exists to avoid; (b) `id(spec)` — Python id is reused after GC,
  which would recreate the false-hit bug; (c) unbounded key from the
  manifest's own file hashes — still reads the files. The mtime+size trade
  (a tamper that preserves both bypasses the cache *and* the hash cache;
  `validate_run`'s file-hash check is the same strength) is documented in
  the helper.
- Fingerprint `OSError` propagates to the existing per-folder
  `except ...: continue` — a run that cannot be fingerprinted is skipped for
  that sync, never cached under a shared fallback key.

## Evidence

- Red→green: with the old `research_memory.py` (git-stashed), the two
  reviewer-scenario tests fail (`test_tampered_oof_not_served_from_stale_cache`,
  `test_second_contract_cannot_reuse_first_cache`); with the fix all 4 pass.
- Guard red-proof: a canary test writing `artifacts/leak_canary.json` fails
  the session with the conftest assertion; after removing the T1 fix's
  explicit paths the same mechanism would fire on `journal.json`.
- `tests/test_research_sync_cache.py` + `tests/test_p2_fixes.py`: 8/8 pass.
- Full suite via `python tools/run_tests.py`: **896 tests OK** (skipped=1,
  pre-existing) with the conftest guard active — no false positives, no
  leaks.
- A/B HUD timing (`python -m tools.agent_bridge_cli --project-root . context`,
  warm-up + 3 runs, `AI_HARNESS_TOOLCHAIN_NO_PIN=1`): old median 1827 ms vs
  new median 1781 ms — no regression from the folder-wide fingerprint.
- `ruff check harness/ tests/ conftest.py`: clean.

## Impact & follow-ups

- No manifest/contract schema change; no impact on sealed runs;
  `ResearchMemory` public API unchanged (only internal key semantics).
- Known limitation (shared with the artifacts hash cache, documented): a
  tamper preserving both mtime and size of every artifact file bypasses the
  cache key — `validate_run`'s own hash cache has the same bound, so no new
  gap is introduced. Content-addressed invalidation is a deliberate
  cost/robustness trade, revisit if tamper-by-mtime-preserve becomes a real
  threat model.
- Follow-ups (separate PRs, from the same review): C1 repair integrity
  (re-sealing without re-deriving metrics + no audit trail) and M2
  (non-recursive artifact sealing — nested `stages/*.log` outside the
  manifest hash map), C2 dispatch governance (CLI dispatch bypasses the
  Bridge ledger).
