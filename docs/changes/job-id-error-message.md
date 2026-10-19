# Job-ID validation error names the offending id, pattern, and example

- **Date**: 2026-10-19
- **PR**: [#110](https://github.com/taitran501/ai-harness/pull/110)
- **Branch**: `feat/kaggle-error-surface`
- **Status**: open
- **Commits**: `d5509fa`

## Why

The arc-prize-2026 audit showed the agent dispatching test jobs with ids like
`test_job`, `job_test`, and `job_test_run_foreground` and receiving only
`Job ID contains unsupported characters.` — no offending value, no expected
shape, no example. It retried the same mistake three times because the error
did not tell it what *was* wrong with its id or what *would* be accepted.
The validation itself (`JOB_ID_PATTERN = ^job_[A-Za-z0-9_.-]{1,180}$` in
`harness/job_state.py`) is correct; only the message was unactionable.

## What changed

- **`harness/job_state.py`** — `worker_state_path` now raises
  `ValueError(f"Job ID contains unsupported characters: {job_id!r}. Expected
  pattern: {JOB_ID_PATTERN.pattern} (e.g. 'job_exp001_full',
  'job_exp002-smoke-0').")` — the offending id, the live regex (so the
  message can never drift from the constant), and two valid examples.
- **`tests/test_job_state.py` (new)** —
  - `test_valid_job_id_resolves_inside_artifacts_dir` (happy path + path
    shape),
  - `test_invalid_job_id_error_names_id_pattern_and_example` (message
    carries the bad id, the `JOB_ID_PATTERN.pattern` string, and the
    example),
  - `test_invalid_job_id_rejects_space_and_wrong_prefix` (subTest over
    `"job "`, `"JOB_x"`, `"x_job_1"`).

## Approach

One-line message fix at the single validation site (every job-id path —
dispatch, worker-state updates, dashboard lookups — flows through
`worker_state_path`). The pattern is interpolated from
`JOB_ID_PATTERN.pattern` rather than re-typed, so the documented contract
and the enforced contract cannot diverge. No change to the pattern itself
(`^job_[A-Za-z0-9_.-]{1,180}$`) — it was never wrong; the agent's
`test_job`/`job_test` ids genuinely violate it. The sibling
`submission_manager.py` filename check was reviewed and left as-is: its
message already includes the offending value, and its charset
(`[A-Za-z0-9_.-]+`) is self-evident from the value.

## Evidence

- `tests/test_job_state.py`: 3/3 pass (4 assertions incl. subTest).
- `python tools/run_tests.py --smoke`: 47 tests OK.
- `ruff check harness/ tools/ tests/`: clean.

## Impact & follow-ups

- Future invalid job ids are self-explanatory in one read — no retry loop.
- No behavioural change for valid ids; error type and raise site unchanged.
- Follow-up (none blocking): if a CLI-level `--job-id` pre-check is ever
  added, it should reuse `JOB_ID_PATTERN` rather than a second regex.
