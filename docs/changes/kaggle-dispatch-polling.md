# Harness-Native Kaggle Submission + Kernel Dispatch

- **Date**: 2026-10-17
- **PR**: [#84](https://github.com/taitran501/ai-harness/pull/84)
- **Branch**: `feat/kaggle-dispatch-polling`
- **Status**: merged
- **Commits**: `63e4718`

## Why

An Antigravity meta-loop scan (scanning the latest agent conversations) surfaced two endgame gaps that agents kept working around instead of using:

- **Gap A** — no harness-native *submission dispatch + score polling*: the `KaggleLeaderboardTracker` was an empty stub, and the facade's `list_submissions` was broken, so agents either hand-polled the raw `kaggle` CLI (blocked by the host hook → friction loop) or gave up.
- **Gap B** — no *kernel wrap + dispatch*: agents hand-wrote `build_expXXX.py` wrappers + 45s polling loops to push kernels.

Adoption-over-prevention: instead of only blocking the raw CLI, give the agent a first-class, attributable, JSON-printed in-process dispatch path — removing the *reason* to bypass.

## What changed

- **Facade (`harness/kaggle_compat.py`)**:
  - **Fixed `list_submissions`** — it called `api.competitions_submissions` (a method that does **not** exist in either SDK), so it was broken cross-SDK. Now uses the shared `competition_submissions` path.
  - Added `submit_submission(file, message, competition, kernel, kernel_version)` (wraps `competition_submit` / `competition_submit_code`) and `kernels_push(folder, timeout, acc)` (wraps `api.kernels_push`), both via a lazily-instantiated cached `KaggleApi` — so the high-level dispatch methods work regardless of which SDK is installed.
- **Tracker (`harness/endgame_submission.py`)**: replaced the `KaggleLeaderboardTracker` stub with `dispatch`, `get_submission`, `wait_for_score` (adaptive 15s→60s backoff, mirrors the native `--wait`; returns `scored`/`error`/`timeout` + attempts), and `latest` (best scored). Status is read via `getattr(status, '_raw', status)` so both the kagglesdk `_NormalisedObject` path and the legacy path work.
- **CLI** (`harness/submission_dispatch.py`, `harness/kernel_dispatch.py`): `ai-harness submission {dispatch|status|watch|latest}` and `ai-harness dispatch-kernel` (stages notebook + generated `kernel-metadata.json`, pushes via the mockable client, returns ref/URL/version + `invalid_*` warnings). Both registered in `cli.py`; exported from `harness/__init__.py`.
- **Docs**: `docs/kaggle_execution.md` (new section), `docs/WORK_LOG.md` (entry), and the SKILL.md source in `harness/skill_injector.py` (endgame dispatch block).

## Approach

- **Cross-SDK dispatch strategy**: use the legacy `kaggle` `KaggleApi` package for all high-level dispatch (submit/push/list) even when `kagglesdk` is detected. Rationale: `kagglesdk` exposes only low-level namespaced services (e.g. `client.competitions.competition_api_client.create_submission`), while `KaggleApi` provides self-contained high-level methods that handle upload/auth internally. This simplifies the facade and guarantees compatibility. A cached `_legacy_api_instance()` lazily imports/authenticates `KaggleApi` only when a dispatch actually runs (construction stays credential-free).
- **Status extraction**: `getattr(obj, '_raw', obj)` unwraps the `_NormalisedObject` wrapper, then string-compare (`"COMPLETE" in str(...)`) to avoid an SDK-specific enum dependency — works for both the wrapped kagglesdk path and the legacy `SubmissionStatus` enum.
- **Polling**: custom `wait_for_score` with exponential backoff (15s start, 1.5x growth, 60s cap) since `kaggle` v2.2.4 lacks native `--wait`.
- **Anti-bypass alignment**: the host hook still blocks raw `kaggle kernels push` / `kaggle competitions submit` shell commands; the harness-native Python API is now the authorized path.

## Evidence

- 22 new tests (`tests/test_kaggle_leaderboard_tracker.py`, `tests/test_kernel_dispatch.py`) — all pass offline (mocked client, injectable sleep).
- Ruff `F` gate clean.
- Full suite: **848 tests, OK (skipped=1)**.
- Live: `list_submissions` returns 17 real RSNA submissions through the fixed facade.
- CLI registration verified (`submission`, `dispatch-kernel` present); generated SKILL.md renders both dispatch blocks.

## Impact & follow-ups

Agents can now dispatch submissions + kernels in-process with `ai-harness submission` / `dispatch-kernel` — no hand-written wrappers, no blocked-CLI friction. Both are JSON-printed for programmatic reading. No follow-up PRs required; monitor endgame adoption via `ai-harness status`.
