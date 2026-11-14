# Deep module hardening and read-only context

- **Date**: 2026-11-14
- **PR**: #137
- **Branch**: `fix/harness-invariants-and-audit-hardening`
- **Status**: open
- **Commits**: Uncommitted working tree; host review and delivery pending.

## Why

Concurrent numeric assignments could fabricate scores, provider observation
provenance could be lost when registry hydration detached the returned object,
and payload filenames could collide with harness metadata. Corrupt sibling runs
could interrupt journal synchronization, alternate journal loading paths could
retain stale verification claims, threshold CSV inference collapsed lexical IDs,
and cancellation could report success before a process exited. Reading the
Context HUD also performed hidden writes and audits.

## What changed

- `state_store.py` defaults to conflict-checked assignments, with record-relative
  additive counter paths. Session persistence recomputes authoritative ledger
  totals under the save lock and uses explicit timestamp assignment semantics.
- `submission_manager.py` commits official scores and provider provenance in one
  authoritative transaction, preserves float precision and active entry identity,
  validates provider identity, and publishes atomic mirrors from committed data.
  One case-insensitive filename validator protects both packaging entry points
  and rejects plugin-created metadata collisions. Ignored arguments warn when
  supplied with nondefault values; CLI help marks them deprecated.
- `tree_journal.py` decodes and validates all loading paths without writes,
  preserves legacy disk shapes, isolates corrupt artifact siblings with bounded
  diagnostics, and persists demotions even when no run can be materialized.
- `threshold_data.py` reads truth IDs as strings before using the shared prediction
  reader and aligning selected OOF predictions. Synthetic sealing fixtures can
  explicitly preserve lexical IDs.
- `cloud_job_runner.py` uses a shared per-job finalization lock, bounded terminate /
  kill escalation, and confirmed process exit before cancellation and accounting.
  PID inspection distinguishes unknown liveness from confirmed absence.
- Removed the shadowed `harness/agent.py`, three obsolete skill generators, and
  dead bridge/research helpers. Consolidated legacy skill cleanup and fitted
  CatBoost/RandomForest prediction logic, including actual probability class order.
  Timing discovery is recursive, testable, and retains repository-relative paths.
- Context construction and dependencies have read-only modes. HUD reads validate
  journal artifacts, registry/state schemas, and persisted stagnation receipts
  without synchronization, evaluation receipts, data audits, locks, or progress
  publication. `refresh-context` performs explicit synchronous refresh and reports
  component outcomes; bounded `refresh_required` entries identify missing/stale state.

## Approach

Implemented the five phases from `scratch/astra_plan_step2.md` in order. Generic
numeric fields remain assignments; only declared counter paths admit positive
event deltas. Ledger totals are freshly recomputed assignments, so stale saves
cannot double-count them. Registry transactions are authoritative; derived
metadata and root mirrors use atomic replacement after a successful commit.
Separate files remain separate filesystem transactions.

Journal decoding captures persisted flags before live evidence checks demote
invalid nodes. Artifact discovery is the explicit promotion path. The HUD uses
fresh read-only dependency instances even when invoked on a writable bridge.
Research scanning is separated from decision evaluation, stagnation snapshot
reads from reconciliation, and progress status reads from publication.

Sparse legacy records retain their exact persisted merge baseline separately
from decoded defaults. Journal synchronization rebases local edits onto the
latest registry, preserving records published by another instance and preventing
default-generated timestamps from becoming false concurrent assignments.
Cancellation and monitor finalization share both thread and per-job filesystem
locks; a recovered manager's persisted cancellation is checked before completion
hooks run in the original monitor.

### File inventory

| File | Change |
| --- | --- |
| `harness/state_store.py` | Explicit record-relative counter policies, assignment conflicts, atomic failed saves, committed revisions. |
| `harness/submission_manager.py` | Single score/provenance commit, identity-preserving hydration, precision, committed mirrors, shared filename validation, deprecation warnings, read-only construction. |
| `harness/tree_journal.py` | Unified validated loading, raw/decoded snapshots, concurrent refresh rebasing, isolated parsing and bounded corrupt-run diagnostics. |
| `harness/threshold_data.py` | Lexical truth IDs and shared prediction reader before alignment. |
| `harness/cloud_job_runner.py` | Confirmed exit, termination escalation, liveness uncertainty, cross-instance finalization and cancellation/monitor race protection. |
| `harness/agent.py` | Deleted shadowed forwarding module; package retained. |
| `harness/skill_injector.py` | Removed three obsolete generators; shared legacy cleanup helper. |
| `harness/models.py` | Shared fitted-pipeline prediction with actual estimator probability class order. |
| `harness/agent_bridge.py` | Removed dead helpers; explicit session counter/ledger persistence; fresh read-only HUD dependencies; refresh command implementation and advisories. |
| `harness/competition_state.py` | Read-only defaults, explicit audit refresh, sparse-state snapshots and explicit timestamp policy. |
| `harness/bridge_actions.py` | Data-risk confirmation explicitly requests an audit refresh. |
| `harness/research_memory.py` | Removed unused Markdown helper; scanning/evaluation split, snapshot incumbent reads with live evidence checks, pure progress status. |
| `harness/stagnation.py` | Schema- and receipt-validated snapshot reader without replay, writes, or locks. |
| `tools/agent_bridge_cli.py` | Selects read-only mode before context construction; adds `refresh-context`. |
| `tools/submission_tool.py` | Deprecated ignored option descriptions and conditional forwarding. |
| `tools/timing/_measure_modules.py` | Recursive relative-path discovery, zero-module error, import-safe `main()`. |
| `tests/unit/test_state_integrity.py` | Five new counter/conflict/ledger regressions, including atomic failure and fractional time. |
| `tests/unit/test_cloud_job_cancellation.py` | New module with ten cancellation, liveness, escalation, accounting and race regressions. |
| `tests/unit/test_models.py` | One parameterized contract test over both wrappers, inputs, encoded/raw/reversed/subset classes, regression and malformed metadata. |
| `tests/integration/test_submission_manager.py` | Eight new transactional observation, precision, sparse registry, race, filename and deprecation tests; old rounded-delta assertions compare numeric value without demanding storage rounding. |
| `tests/integration/test_tree_journal.py` | Six new corruption, load-path, validation-count, sparse-state, demotion and concurrent-refresh regressions. |
| `tests/integration/test_evidence_sealing.py` | Sealed lexical-ID, large-ID, reversed declared class order, duplicate/missing-ID and immutable-evidence regression. |
| `tests/integration/test_context_read_only.py` | New module with ten filesystem-inventory, dependency, CLI, stale-state and explicit-refresh tests. |
| `tests/integration/test_framework_adoption.py` | Three legacy cleanup, package forwarding and recursive timing-discovery regressions. |
| `tests/integration/test_research_continuity.py` | Existing progress tests now distinguish read-only status from explicitly requested publication. |
| `tests/evidence_fixtures.py` | Optional lexical-ID preservation in synthetic sealing helpers. |
| `docs/changes/deep-module-hardening.md` | This implementation and validation record. |
| `docs/changes/README.md` | Added the change index entry. |

## Evidence

- Added **44 unittest regression methods**, with additional parameterized cases.
- New context read-only integration module: ten tests, including real CLI JSON /
  Markdown, toolchain re-execution, cold construction, pending decision evaluation,
  corrupt authoritative state, tampered/orphan evidence, and explicit refresh.
  Inventories include files/directories, contents, and modification timestamps.
- `ruff check harness/ tools/ tests/`: clean.
- Final `python tools/run_tests.py --smoke`: **84 tests OK**, exit 0
  (20.979 s test time; 24.3 s including loading).
- Full journal module plus cancellation and affected score class: **45 tests OK**.
- Latest focused journal/provider/cancellation regressions: **24 tests OK**, exit 0
  (9.126 s test time; 12.9 s including loading).
- Final `python tools/run_tests.py --fast`: **1,122 tests OK** (1,120 passed,
  two skipped), zero failures/errors, exit 0 (577.605 s test time; 582.3 s including loading).
- `git diff --check`: clean; branch and HEAD remain
  `fix/harness-invariants-and-audit-hardening` / `9b28406f6f70d292e15eb13eecef47a6a6853904`.
- Final smoke/fast output is retained in
  `scratch/deep-module-hardening-final-smoke.log` and
  `scratch/deep-module-hardening-final-fast.log` (local QA logs, ignored by Git).

## Impact & follow-ups

Context consumers should invoke `ai-harness-agent --project-root . refresh-context`
when derived state needs refreshing. Unindexed runs do not enter the candidate
pool during a context read. Deprecated submission arguments remain accepted for
one compatibility cycle and do not enforce submission gates.

Native Windows `taskkill` returned access denied for a disposable test child in
this restricted environment. The manager correctly preserved nonterminal state.
The real-child success test injects termination through the owned `Popen` handle
on Windows while retaining real `wait()`/`poll()` checks; denial, timeout,
survivor and escalation paths also have deterministic tests. No provider
cancellation or live cloud execution was attempted.

No commit, push, provider dispatch, or production competition run is part of this
change. The full suite remains required before a future push under repository policy.
