# Live Kaggle submission limits & quota discovery, Context HUD quota integration, interactive gateway keepalive daemon, and autonomous discussion mining mandate

- **Date**: 2026-10-22
- **PR**: [#117](https://github.com/taitran501/ai-harness/pull/117)
- **Branch**: `feat/submission-quota-and-gateway-keepalive`
- **Status**: merged
- **Commits**: `aa29970` (squash)

## Why

During real-world competition triage across active sessions (`arc-prize-2026-arc-agi-3` `ce5361e6`, `airline_prediction` `8d9fcfe9`, `rsna-knee` `95c2aed5`), three critical friction patterns emerged:
1. **Blindness to Kaggle daily submission limits & false quota assumptions**:
   - `competition_state.py` hardcoded `daily_quota = 5` and estimated `subs_today` via local date string matching on `.ai-harness/competition_state.json`.
   - Real competition daily limits vary drastically: ARC Prize allows only 1 sub/day (`numAllowedNow: 0`), Playground Series allows 10 subs/day (`numAllowedNow: 6`), and RSNA allows 5 subs/day (`numAllowedNow: 3`).
   - The agent was completely unaware of its actual remaining quota, leading to either premature stopping while slots remained or wasting the sole daily submission on unverified code.
2. **Platform idle watchdog termination on interactive evaluation gateways**:
   - In code competitions with streaming gateways (e.g. ARC Prize Hearth gateway), startup latency during SGLang/LLM shard loading or weights download exceeding 15 minutes causes an immediate idle watchdog disconnect with a `0.00` score.
   - Ref `57510673` burned a full daily quota because model loading took 16 minutes without sending an intermediate heartbeat action.
3. **Stagnation perseveration & human handover paralysis ("cố chấp")**:
   - When metric progress plateaued, agents repeatedly fell into micro-tuning loops ($\alpha = 0.18$ vs $0.25$) or generated handover briefs asking the human operator to make model choices ("Bạn tự làm đi chứ sao đưa cho tôi ?").
   - Agents failed to autonomously query Kaggle community discussions for breakthrough representations.

## What changed

- **`harness/kaggle_compat.py`**:
  - Implemented `KaggleClient.get_submission_limits(competition: str)`: queries live Kaggle API limits (`numToday`, `numAllowedNow`, `numTotal`, `limitedByTotal`, `daily_quota`).
- **`harness/endgame_submission.py` & `harness/submission_dispatch.py`**:
  - Implemented `KaggleLeaderboardTracker.get_quota(competition, project_root)` with automatic local caching to `.ai-harness/submission_limits.json`.
  - Added CLI subcommands `ai-harness submission quota <competition>` and `ai-harness submission limits <competition>`.
- **`harness/gateway_keepalive.py`**:
  - Created standalone, thread-safe `GatewayKeepaliveDaemon`:
    - Automatically discovers reachable gateway endpoints (`/api/games`) and emits periodic heartbeat actions (default `RESET` every 300s).
    - Usable via context manager `with GatewayKeepaliveDaemon(): ...` or explicit `start()` / `stop()` during heavy model initialization.
    - CLI executable via `python -m harness.gateway_keepalive`.
- **`harness/competition_state.py` & `harness/agent_bridge.py`**:
  - `derive_competition_state` now reads cached live submission limits, dynamically aligning `subs_today`, `daily_quota`, and `remaining_quota_today`.
  - `render_context_markdown` (Context HUD) displays exact daily quota (`X/Y used (Z remaining today)`).
  - Added **Autonomous Discussion Mining Mandate** alert when consecutive rejections ($\ge 2$) or stagnation occur, strictly barring micro-tuning and commanding immediate discussion inspection.
- **`harness/skill_injector.py`**:
  - Documented `GatewayKeepaliveDaemon` usage and `ai-harness submission quota` in the competition skill reference.
- **`tests/`**:
  - Added unit test suite `tests/test_gateway_keepalive.py` covering discovery, heartbeat pulses, and lifecycle.
  - Added `test_submission_dispatch_quota_cli` and `test_tracker_get_quota_and_caching` in `tests/test_kaggle_discussions_and_submissions.py`.

## Approach

- **Live platform ground truth with zero-latency fallback**: Quota queries ping the official Kaggle endpoint and persist results in `.ai-harness/submission_limits.json`. The Context HUD reads this cache without imposing synchronous HTTP overhead on every local state derivation.
- **Resilient background heartbeat**: `GatewayKeepaliveDaemon` runs entirely in a daemon thread, catching and logging transient network blips without crashing the main training/inference process, and cleanly shutting down upon solver entry.

## Evidence

- **Live platform check**:
  - `rsna-knee-abnormality-detection`: `{"num_today": 2, "num_allowed_now": 3, "num_total": 33, "daily_quota": 5}`.
  - `arc-prize-2026-arc-agi-3`: `{"num_today": 1, "num_allowed_now": 0, "num_total": 5, "daily_quota": 1}`.
  - `playground-series-s6e10`: `{"num_today": 4, "num_allowed_now": 6, "num_total": 104, "daily_quota": 10}`.
- **Unit test suite**:
  - `python -m unittest tests/test_gateway_keepalive.py tests/test_kaggle_discussions_and_submissions.py` passes 17/17 tests.
- **Smoke tier**:
  - `python tools/run_tests.py --smoke` passes 47/47 tests in 3.8s.
- **Static gate**:
  - `ruff check harness/ tools/ tests/` clean (0 errors).

## Impact & follow-ups

None.
