# Dispatch governance: audit trail + duplicate warning for Kaggle dispatches

- **Date**: 2026-10-18
- **PR**: [#97](https://github.com/taitran501/ai-harness/pull/97)
- **Branch**: `fix/dispatch-governance-audit`
- **Status**: open
- **Commits**: `8332bb3`

## Why

External integrity review finding **C2**: the harness-native dispatch CLIs
(`ai-harness submission dispatch`, `ai-harness dispatch-kernel`) drive the
Kaggle API in-process — authorized, since the host hook blocks the raw
`kaggle` shell commands — but they bypass every governance surface: no
Bridge session, no ledger registration, no audit trail. A
double-submission (same file submitted twice, which burns a competition
submission slot) or a kernel push that was never tracked left **no evidence
behind**, unlike the governed Bridge `prepare_submission` path (session,
journal, `SubmissionManager` ledger, dedup/MMES/significance guards).

Design constraints honored: additive/backward-compatible (dispatch still
succeeds exactly as before; new fields are optional in the JSON payload),
**advisory-only in v1** (duplicate detection warns, never blocks — a hard
block would break legitimate re-dispatches such as a file re-read after
regeneration with identical content), and fail-safe auditing (audit failure
never breaks the dispatch, and the audit layer can never write into the
installed framework checkout).

## What changed

**New: `harness/dispatch_audit.py`** — the audit layer:
- `record_dispatch(...)`: appends one JSONL entry per dispatch to
  `<workspace>/.ai-harness/dispatch_log.jsonl` (append-only, beside the run
  folders, outside every sealed bundle): timestamp, kind (`submission` /
  `kernel`), competition or kernel id, the **content hash** of what was
  sent (`file_sha256` / `notebook_sha256`, streamed; `null` when the local
  file is gone), ref, operator, cwd, and the duplicate match (if any)
  computed *before* the append so the log stays a faithful history.
- `check_duplicate(...)`: exact-content match — same hash AND same
  competition (submissions) / same kernel id (kernels) — against (a) the
  dispatch log and (b) the governed `SubmissionManager` ledger
  (`registry.json`, read-only; hashes `archive_path`/`candidate_path`), so
  duplicates are detected even when the "first" submission went through
  the Bridge path.
- `resolve_log_path(...)`: writes only into an explicitly targeted
  workspace (param / `HARNESS_PROJECT_ROOT`) or an *attached* workspace
  (CWD has `.ai-harness/project.yaml` or the legacy manifest). A bare
  directory — and the installed framework checkout — is never a target
  (the T1 class of repo pollution cannot happen from the audit layer).
- Read-tolerant: malformed log lines are skipped; registry/manifest read
  errors degrade to "no match", never raise into the dispatch path.

**`harness/endgame_submission.py`** — `KaggleLeaderboardTracker.dispatch`
(the single choke point for both the CLI and programmatic use) now
records the audit entry after a successful submit and adds optional
`duplicate_of` / `duplicate_warning` fields to the returned payload. The
audit call is wrapped: any failure prints a stderr note and the dispatch
result is returned unchanged.

**`harness/kernel_dispatch.py`** — `dispatch_kernel` records the audit
entry (notebook content hash, kernel id) and adds the same optional
duplicate fields when the identical notebook content was already pushed to
the same kernel id (a genuine version bump — different content — is *not*
flagged).

**`harness/submission_dispatch.py` / `harness/kernel_dispatch.py` CLI** —
a human-readable `WARNING:` line on **stderr** when the payload carries a
duplicate warning; stdout stays pure JSON for agents, and the exit code
stays 0 (advisory, never a block).

**Tests** — `tests/test_dispatch_governance.py` (new, 12): audit entry
shape with exact content hash, identical-resubmission warning +
no-hard-block, no warning across competitions or after content change,
missing-file dispatch records `null` hash, ledger-based duplicate
detection, audit-failure-never-breaks-dispatch, no-project-context
dispatches without writing any log (and provably not into the framework
checkout), kernel push audit + same-content re-push warning + clean
version bump, and the CLI stderr/stdout split.

## Approach

- **Audit at the tracker/choke point, not the CLI.** The CLI's existing
  tests mock the whole tracker; auditing at `KaggleLeaderboardTracker.dispatch`
  and `dispatch_kernel` covers every caller (CLI *and* programmatic)
  without touching argument parsing.
- **Content hashes, not names.** "Was this file already submitted?" is a
  byte question. File names can differ between regenerations; names in
  the log are for context, the SHA is the matcher.
- **Advisory, not blocking (v1).** Double-submitting *identical* content
  is almost always operator error — but the tracker cannot know the
  operator's intent (a retry after a network timeout on the raw API is a
  legitimate re-dispatch). v1 surfaces the warning at the moment it
  happens and in the durable log; a `--strict` hard-block mode is a
  natural follow-up once the log proves the false-positive rate is ~0.
- **Ledger scan is read-only.** `SubmissionManager` construction has
  side effects (mkdir, markdown render); the audit only reads
  `registry.json` at its known locations, so it never mutates state.
- **Checkout-safe resolution.** The resolver's CWD branch requires an
  attached manifest. The framework checkout itself has no
  `.ai-harness/project.yaml`, so tests and ad-hoc runs inside the repo
  cannot create a log there (the PR-C tests assert exactly this).

## Evidence

- Red→green: the new tests fail on pre-change code (no audit log exists,
  no duplicate fields) and pass after wiring.
- `tests/test_dispatch_governance.py`: **12/12 pass**.
- Regression: `tests/test_submission_dispatch.py`,
  `tests/test_kernel_dispatch.py`, `tests/test_remote_dispatch_hardening.py`,
  `tests/test_kaggle_leaderboard_tracker.py`, `tests/test_competitive_endgame.py`,
  `tests/test_endgame_tools.py`, agent-friction/usability suites — all pass
  unchanged (payload fields are additive; mocked-CLI paths unaffected).
- Full suite: `python tools/run_tests.py` → **904 tests OK** (skipped=1,
  pre-existing). `ruff check harness/ tests/ tools/` clean.
- No-context safety: with `HARNESS_PROJECT_ROOT` unset and no manifest in
  CWD, dispatch succeeds with zero files written (test asserts the bare
  dir gains no `.ai-harness/` and the framework checkout's log state is
  unchanged).

## Impact & follow-ups

- **New file**: `<workspace>/.ai-harness/dispatch_log.jsonl` appears in
  attached workspaces after the first harness-native dispatch (additive,
  git-ignored territory alongside other `.ai-harness/` state).
- **New payload fields** (`duplicate_of`, `duplicate_warning`) are
  optional; agents that ignore them see byte-identical behavior for
  non-duplicate dispatches.
- **v1 is advisory**: no hard block. Follow-up candidates (separate PRs):
  `--strict` mode that blocks identical-content resubmits, wiring the
  dispatch log into `ai-harness-agent context` HUD, and a `record_dispatch`
  hook for the Bridge path so *all* submission paths share one audit
  surface (the Bridge already has its own ledger; this would be
  cross-linking, not replacement).
