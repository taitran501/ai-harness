# Kaggle competition discovery & metadata CLI, response container iteration safety, and client facade parity

- **Date**: 2026-10-24
- **PR**: [#129](https://github.com/taitran501/ai-harness/pull/129)
- **Branch**: `feat/kaggle-competition-discovery-and-compat-stubs`
- **Status**: merged
- **Commits**: e499dd8

## Why

Auditing historical competition conversations across active workspaces (`rsna-knee` `95c2aed5`, `airline_prediction` `8d9fcfe9`, `arc-prize` `ce5361e6`, `cfaad263`, `e0aaeefd`) revealed persistent friction patterns where agents failed or resorted to ad-hoc workarounds:

1. **Competition metadata blindness and complex SDK probing**:
   - In RSNA Knee session `95c2aed5`, the agent spent multiple turns attempting to determine whether the competition had ended, the exact deadline, whether submissions were disabled, official metric, and current team rank.
   - The agent ran into repeated crashes (`AttributeError: 'KaggleApi' object has no attribute 'competition_view'`, `TypeError: ApiGetCompetitionRequest.__init__() got an unexpected keyword argument 'competition_name'`).
   - The framework lacked a native `ai-harness competition info <comp>` command.
2. **KaggleClient alias omissions and response container iteration crashes**:
   - Multiple agents called `client.competition_submissions(...)` expecting it to match the legacy SDK method, crashing with `AttributeError: 'KaggleClient' object has no attribute 'competition_submissions'`.
   - Callers iterating over raw response containers or requesting `len()` / index `[0]` encountered `TypeError: 'ApiListCompetitionsResponse' object is not iterable`, `TypeError: object of type 'ApiListCompetitionTopicsResponse' has no len()`, and `AttributeError: 'ApiKernelMetadata' object has no attribute 'totalVotes'`.
3. **Missing symbol exports on canonical modules**:
   - Agents attempted to import `typed_actions` from `harness.agent_bridge` or `SubmissionValidator` from `harness.endgame_submission`, triggering `ImportError`.

## What changed

- **`harness/kaggle_compat.py`**:
  - Implemented `KaggleClient.get_competition(competition: str) -> Optional[Dict[str, Any]]` and `KaggleClient.list_competitions(...)`.
  - Added method aliases on `KaggleClient`:
    - `competition_submissions = list_submissions`
    - `competition_view = get_competition`
    - `competitions_list = list_competitions`
    - `kernel_status = kernels_status`
    - `kernel_logs = kernels_logs`
  - Extended `_LEGACY_ALIASES` with bidirectional alias generation (camelCase + snake_case) across all candidates (`totalReplies`, `size`, `total_bytes`, `team_id`, `team_name`, `is_code_competition`, `max_daily_submissions`, `evaluation_metric`, etc.).
  - Enhanced `_NormalisedObject` with container safety:
    - `__getitem__`: supports integer indexing `[i]` across child collections (`competitions`, `topics`, `messages`, `submissions`, `kernels`, `files`).
    - `__iter__`: allows direct iteration over response containers.
    - `__len__`: enables `len(response)` delegation.
- **`harness/competition_cli.py` & `harness/cli.py`**:
  - Created standalone `harness/competition_cli.py` and top-level CLI command `ai-harness competition [info|list|leaderboard]`.
  - Added `ai-harness submission info <comp>` subcommand.
- **`harness/agent_bridge.py` & `harness/endgame_submission.py`**:
  - Exported `typed_actions = [a.value for a in TypedActionName]` in `harness/agent_bridge.py`.
  - Re-exported `SubmissionValidator = SubmissionPreflightValidator` in `harness/endgame_submission.py`.
- **`tests/`**:
  - Added test suite `tests/test_kaggle_competition_discovery.py` covering metadata extraction, container iteration/indexing, aliases, and CLI commands.

## Approach

- **Non-breaking Facade Parity**: All aliases and container delegations preserve backward compatibility while transparently absorbing naming differences between legacy Kaggle API and modern `kagglesdk`.
- **Self-Contained Discovery**: Agents can introspect competition rules, deadlines, and rank without raw SDK hacks.

## Evidence

- **Unit tests**: `tests/test_kaggle_competition_discovery.py` passed 8/8 tests; all 70 kaggle/bridge tests passed 100%.
- **Live verification**:
  - `ai-harness competition info playground-series-s6e10`: returns accurate JSON metadata (title, deadline, is_code_competition, user_rank, max_daily_submissions).
  - `from harness.agent_bridge import typed_actions`: resolves cleanly (39 actions).
- **Static gate & Smoke gate**:
  - `ruff check harness/ tools/ tests/`: clean (0 errors).
  - `python tools/run_tests.py --smoke`: 47/47 passed in 2.8s.

## Impact & follow-ups

None.
