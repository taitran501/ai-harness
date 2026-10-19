# wait_for_dataset: surface CLI failures with detail (no more empty errors)

- **Date**: 2026-10-19
- **PR**: [#109](https://github.com/taitran501/ai-harness/pull/109)
- **Branch**: `fix/arc-prize-audit-remediation`
- **Status**: open
- **Commits**: `09ccfe6`

## Why

The arc-prize-2026 dispatch (job_20261019_095919) died in
`KaggleKernelManager.wait_for_dataset` with an **empty** error surface —
the CLI failed with no stdout/stderr, and the old code raised
`RuntimeError("Kaggle dataset status failed: ")` with nothing after the
colon, giving no diagnostic value. The retry predicate also required a
recognised HTTP keyword in the output, so a silent CLI failure was not
treated as transient and escalated immediately on the first attempt.

## What changed

- **`harness/templates/kaggle_template.py`** — `wait_for_dataset`:
  - An **empty** CLI failure (no stdout/stderr) is now treated as
    transient like HTTP 403/404/5xx/connection errors and retried up to
    `max_consecutive_errors`; only after the budget is spent does it
    raise.
  - The raised `RuntimeError` now includes the exit code, the attempt
    count, and an explicit `<no CLI output>` marker when the CLI stayed
    silent — never an empty message.
  - The `TimeoutError` message now includes the manual check command
    (`kaggle datasets status <slug>`) and degrades gracefully when no
    prior error was captured (`<none>`).
- **`tests/test_kaggle_preflight.py` (new, part 1)** —
  `WaitForKeywordSurfaceTests`: empty-failure retry-then-raise with the
  detailed message; empty-then-READY succeeds; timeout surfaces the
  manual command.

## Approach

Minimal, targeted hardening of the existing retry loop — no new
dependency or polling strategy. The key semantic decision: an empty
failure is *unknown*, not *fatal*; unknowns get the same bounded retry
budget as known-transient HTTP errors, after which the error carries
everything we observed (exit code, attempts, explicit no-output
marker). Alternative considered — parsing the CLI's JSON output —
rejected: the status command's output shape varies across CLI releases,
and the state keyword (READY/ERROR/FAILED) parsing already works.

## Evidence

- `tests/test_kaggle_preflight.py::WaitForKeywordSurfaceTests`: 3/3 pass.
- Existing `test_wait_for_dataset_transient_error_retry`
  (HTTP 403 retry behaviour): still passes — the new branch is a
  superset, old keyword path unchanged.

## Impact & follow-ups

- Operators now always get an actionable failure (exit code + output or
  explicit no-output marker + manual check command) instead of a blank
  `status failed:` error.
- Latency on persistent silent CLI failures increases by up to
  `(max_consecutive_errors - 1) × max(2s, poll_interval)` before the
  raise — bounded and intentional.
- Follow-up (none blocking): nothing.
