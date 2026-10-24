# Kaggle kernel watch & list, dispatch --watch, Agent Bridge slug fallback, and live public leaderboard discovery

- **Date**: 2026-10-24
- **PR**: [#118](https://github.com/taitran501/ai-harness/pull/118)
- **Branch**: `feat/kaggle-kernel-watch-dispatch-and-leaderboard`
- **Status**: in-review
- **Commits**: pending

## Why

During active competition sessions on October 24, 2026 (`arc-prize-2026-arc-agi-3` `ce5361e6` and `airline_prediction` `8d9fcfe9`), real agents encountered several friction points that blocked autonomous workflows and prompted user frustration:

1. **Missing Kaggle kernel watch/wait CLI and agent hack attempt (`ce5361e6`)**:
   - In ARC Prize session `ce5361e6`, remote GPU kernel `taitran501/test-taaf-m2` ran for ~26 minutes.
   - The user asked: *"Sao đợi lâu vậy, có lệnh harness để waiting mà?"*.
   - The agent called `ai-harness-agent act --action wait_cloud_job --job_id taitran501/test-taaf-m2`, but was blocked with `ValueError: Cloud job taitran501/test-taaf-m2 was not found in this task workspace` because `_handle_inspect_cloud_job_or_wait_cloud_job` only checked local `.ai-harness/artifacts/jobs/*.json` state.
   - When inspecting `ai-harness kaggle --help`, only `status`, `log`, and `output` were available; `watch` and `list` were missing.
   - The agent wrote a temporary wait script and even modified `C:\Users\trant\ai-harness-framework\harness\kaggle_inspect.py` directly to add a `watch` action.
2. **Missing `--watch` option in `ai-harness dispatch-kernel`**:
   - When using `ai-harness dispatch-kernel`, agents had to run a separate custom script or loop to wait for execution to finish.
3. **Missing Kaggle Public Leaderboard & Kernel Listing CLI (`8d9fcfe9`)**:
   - In `airline_prediction`, the agent reached Global Rank #19 (LB 0.96213) and needed to monitor competitor standings and public notebooks.
   - Without native harness commands, the agent resorted to executing raw Python scripts calling `subprocess.run(['kaggle', 'competitions', 'leaderboard', ...])` and `subprocess.run(['kaggle', 'kernels', 'list', ...])`.

## What changed

- **`harness/kaggle_inspect.py`**:
  - Implemented `kernel_watch(kernel_slug, exp_id, poll_interval, timeout_seconds)` delegating to `KaggleKernelManager.wait_for_kernel`.
  - Implemented `kernel_list(competition, user, page_size, sort_by, search)` with automatic sort option mapping (`hot` $\to$ `hotness`, `votes` $\to$ `voteCount`, `updated` $\to$ `dateRun`, etc.).
  - Added CLI subcommands `ai-harness kaggle watch` and `ai-harness kaggle list`.
- **`harness/kernel_dispatch.py`**:
  - Added `--watch`, `--poll`, and `--watch-timeout` CLI flags to `ai-harness dispatch-kernel`.
  - `dispatch_kernel` now optionally waits for remote execution upon successful push and attaches the resulting status and watch record to the output JSON.
- **`harness/bridge_actions.py`**:
  - Enhanced `_handle_inspect_cloud_job_or_wait_cloud_job` with a fallback for Kaggle kernel slugs (when `job_id` contains `/` or starts with `kernel-` and no local job artifact matches).
  - Routes directly to `KaggleKernelManager.wait_for_kernel` or `get_kernel_status`, returning a clean `ActionObservation` instead of raising a `ValueError`.
- **`harness/kaggle_compat.py` & `harness/endgame_submission.py` & `harness/submission_dispatch.py`**:
  - Added `KaggleClient.get_leaderboard(competition, page_size, **kwargs)`.
  - Added `KaggleLeaderboardTracker.get_leaderboard(competition, top)`.
  - Added CLI subcommand `ai-harness submission leaderboard <competition> [--top N]` (with alias `lb`).
- **`harness/cli.py`**:
  - Added top-level shortcut `ai-harness leaderboard <competition> [--top N]`.
  - Updated help text for `kaggle` and `submission` parsers.
- **`tests/`**:
  - Added `test_kernel_watch_delegates_to_manager`, `test_kernel_list_delegates_to_client`, and `test_cli_watch_and_list_parsers` in `tests/test_kaggle_inspect.py`.
  - Added `test_dispatch_with_watch_waits_for_completion` and CLI flag tests in `tests/test_kernel_dispatch.py`.
  - Added `test_tracker_get_leaderboard` and `test_submission_leaderboard_cli` in `tests/test_kaggle_discussions_and_submissions.py`.
  - Added `TestWaitCloudJobKernelSlugFallback` in `tests/agent_bridge/test_friction_fixes.py`.

## Approach

- **Unified Remote Execution & Inspection**: Agents can push kernels with immediate monitoring via `--watch` or inspect existing running kernels via `ai-harness kaggle watch` without re-pushing or writing throwaway scripts.
- **Agent Bridge Ergonomics**: Bridge action `wait_cloud_job` seamlessly handles both local job hashes and direct Kaggle kernel slugs (`owner/slug`).
- **Native Platform Ground Truth**: Public leaderboards and competitor kernels can be fetched in standard JSON format using `ai-harness leaderboard` and `ai-harness kaggle list`, removing the need for raw subprocess calls or ad-hoc scrapers.

## Evidence

- **Live platform check**:
  - `ai-harness leaderboard playground-series-s6e10 --top 2` returns structured JSON with ranks, team names, and scores.
  - `ai-harness kaggle list --competition playground-series-s6e10 --page-size 2` returns public community kernels.
- **Unit test suites**:
  - `tests/test_kaggle_inspect.py`: 7/7 tests passed.
  - `tests/test_kernel_dispatch.py`: 16/16 tests passed.
  - `tests/test_kaggle_discussions_and_submissions.py`: 16/16 tests passed.
  - `tests/agent_bridge/test_friction_fixes.py`: 23/23 tests passed.
- **Smoke tier & Fast tier**:
  - `python tools/run_tests.py --smoke`: 47/47 passed.
- **Static gate**:
  - `ruff check harness/ tools/ tests/`: clean (0 errors).

## Impact & follow-ups

Eliminates the friction of manual kernel polling scripts and raw Kaggle CLI subprocess calls across competition workspaces.
