# Journal sync dedup: halve per-action run validation + fix orphan resurrection

- **Date**: 2026-10-18
- **PR**: [#90](https://github.com/taitran501/ai-harness/pull/90)
- **Branch**: `perf/journal-sync-dedupe-and-dead-code-trim`
- **Status**: open
- **Commits**: `00ec8eb`

## Why

A "heaviness" audit of the framework (driven by real Antigravity IDE usage
data: ~630 `ai-harness-agent act` invocations and ~175 `context` invocations
across 299 conversation DBs) showed each agent action pays the full
solution-tree validation cost **twice**, and every `TreeJournal` construction
paid it twice more:

1. `TreeJournal.__init__` fully validated every journal node with
   `validate_run(task_spec=..., plugin=...)`, then `sync_from_artifacts()`
   re-validated every on-disk run directory and **overwrote the same
   nodes**. For a workspace with N runs that was 2N full validations per
   construction.
2. `AgentBridge.execute_action` built a **second** fully-synced
   `TreeJournal` after the action (for the post-action HUD snapshot), on
   top of the one built before the action → 4N full validations, two
   `journal.json` loads, two project re-discoveries, and a second
   `SubmissionManager` construction per action.

Both walks are intentional per the correctness contract (consumers must
reject mismatched manifest hashes), so the fix had to reduce *frequency*
without weakening the per-run gate.

**Explicitly rejected alternative**: the audit's top-ranked candidate —
lazy-loading `sklearn.metrics` out of `harness/evaluation.py` (≈1.3 s of
every agent/runner process start) — is **not safe**: `scorer_sha256` in
`artifact_manifest.json` is a hash of the *live source* of
`harness/evaluation.py` + `harness/task_spec.py` (+ the plugin's
`evaluate_predictions`) computed by `artifacts.contract_identity()`.
Empirically verified on the `aimo` workspace: appending a single comment
line to `evaluation.py` changed the digest (`aca79278…` → `59e34e39…`) and
every sealed run would raise `Run identity mismatch: scorer_sha256` in
`validate_run`, orphaning all past runs (baseline anchors, insight
scopes, candidate pools) as `LEGACY_UNVERIFIED`. Any such import
optimization requires a contract change (narrowing the hash scope) which
is a design decision, not a mechanical fix.

## What changed

- `harness/tree_journal.py` — `TreeJournal.__init__`:
  - When `auto_sync=True`, the per-node `validate_run` loop now skips
    nodes whose `artifact_dir` is a live on-disk run directory (the
    subsequent `sync_from_artifacts()` pass re-validates it and
    overwrites the node). Only orphan nodes (artifact dir gone from
    disk) keep the full check there — they fail fast on the missing
    manifest. Every on-disk run is still fully validated exactly once
    per auto-synced construction.
  - When `auto_sync=False` the pre-optimization behavior is preserved
    exactly (full per-node validation, no artifacts walk).
  - `self._snapshot` (the merge baseline for `save()`) is now captured
    **immediately after `load()`, before** the validation marking loop.
    See "Approach" — this fixes a pre-existing orphan-resurrection bug.
- `harness/agent_bridge.py` — `execute_action`: the post-action HUD
  snapshot reuses the action's journal instance (`journal.sync_from_artifacts()`
  once, idempotent) and the action's `SubmissionManager` instead of
  constructing a fresh `TreeJournal(auto_sync=True)` +
  `SubmissionManager()`.
- `tests/test_journal_sync_dedupe.py` — new regression tests (5):
  healthy on-disk run verified; **tampered on-disk run still rejected**
  (gate intact); orphan journal node marked `LEGACY_UNVERIFIED` and
  stays marked across `save()`; `auto_sync=False` unchanged;
  post-action snapshot equality with a fresh-journal reference.

## Approach

**Equivalence argument for the dedup.** For a node whose artifact dir is
on disk, old code = full `validate_run` in the loop, then
`sync_from_artifacts` does the same `validate_run` and replaces the node
with an identical fresh one; new code = only the sync pass. Final node
state is identical in both, and the sync pass is the authoritative one
(it rebuilds the node from disk). For an orphan node, both old and new
code run the full check (it fails fast on the missing manifest).

**Pre-existing bug found while testing: orphan resurrection.** The first
regression test (orphan marked unverified) failed even though the loop
correctly marked the node: `TreeJournal.save()` rebuilds the in-memory
`self.nodes` from the *merged on-disk registry* returned by
`state_store.save_registry`, and `save_registry` keeps the on-disk record
whenever the record equals the `_snapshot` baseline. The baseline was
captured **after** the marking loop, so the marked record looked
"unchanged" and the stale on-disk record (`verified=True`) was written
back and rebuilt into memory. Verified present in the original code
(`git show HEAD:harness/tree_journal.py` reproduces it whenever any
on-disk run triggers `save()` during sync). Fix: capture the baseline
right after `load()` so the merge sees the marking as a local change and
persists it (a true concurrent write to the same node still raises
`Concurrent state conflict`, as designed).

**Rejected: aggressive single-journal-per-process caching.** Reusing the
journal across separate CLI invocations is impossible (each `act` is a
new process) and caching inside `execute_action` beyond the snap_after
reuse would couple unrelated call sites (`get_context_packet`,
`start_session`, dashboard) to stale-state assumptions; the two
contained changes above capture the measurable gain with no cross-call
state.

**Not changed** (deliberately, contract-safe): the per-run validation
gate itself, `validate_run`'s mtime/size hash cache, `repair.py`'s
deliberate cache-bust, and everything in the `scorer_sha256` hash scope
(`evaluation.py`, `task_spec.py`, plugin scoring source).

## Evidence

- New: `python -m pytest tests/test_journal_sync_dedupe.py -q` →
  **5 passed** (including the tampered-run rejection and the
  orphan-resurrection regression).
- `python tools/run_tests.py --fast` → 799 tests OK (skipped=1);
  full suite `python tools/run_tests.py` → **875 tests OK (skipped=1)**
  (one first-run failure in a slow module was a flake: it passed in
  isolation and on full re-run).
- `ruff check harness/ tools/ tests/ --select F,E7` → clean.
- **A/B benchmark** (aimo workspace, identical interpreter/cwd, the
  loaded code differing only by this change; the toolchain-pin re-exec
  must be disabled with `AI_HARNESS_TOOLCHAIN_NO_PIN=1` on the dev run
  or the measurement is contaminated by a second process):
  - `context`: median 1.747s → 1.704s (~40 ms, ~2.4%)
  - `status`:  median 1.621s → 1.591s (~30 ms, ~1.9%)
  - Probe (`sitecustomize` counter): `validate_run` calls in one
    `context` run dropped **7 → 5** (aimo has 2 on-disk runs: the
    init-pass check of both on-disk nodes is the removed work; every
    on-disk run is still validated exactly once by the sync pass).
  - The absolute gain scales with the number of **local** artifact
    runs (each `validate_run(task_spec=...)` recomputes the contract
    source hash + per-file SHA256): aimo is a Kaggle-backend workspace
    with N=2, where the win is ~40 ms; a local workspace with N=30
    saves ~30 full validations per journal construction (~0.3–0.5 s).
- Contract invariance: aimo `exp000` `scorer_sha256` digest unchanged
  (`aca79278…`), `context` still reports the sealed baseline.
- Original-code bug repro: orphan node returned `verified=True` on the
  pre-change source; `verified=False` after the baseline fix.

## Impact & follow-ups

- Per agent action: `validate_run` calls drop from 4N to 2N (plus one
  cheap orphan check), one fewer `journal.json` load, one fewer project
  re-discovery, one fewer `SubmissionManager` construction. For the
  `context` command the per-construction cost drops from 2N to N+1.
  Measured on the live `aimo` workspace (N=2 local runs): `context`
  median 1.747s → 1.704s, `status` 1.621s → 1.591s. The gain scales
  linearly with local run count; the dominant per-process cost remains
  the ~1.3 s `harness.evaluation` import (see below).
- `journal.json` now durably records `LEGACY_UNVERIFIED` for orphaned
  nodes (previously resurrected to `verified=True` on the next save when
  any run was on disk) — strictly more truthful state.
- Follow-up (requires a contract decision, not done here): the
  `scorer_sha256` source-hash is what blocks lazy-loading `sklearn` out
  of `evaluation.py` (~1.3 s per process start). If the project wants
  that win, the hash scope must narrow to the scoring-relevant source
  with a documented migration of existing manifests.
- Follow-up (low priority): `_refresh_task_context` rebuilds five
  objects per action (measured ~30 ms warm); a fingerprint skip is
  possible but the gain is marginal today.
