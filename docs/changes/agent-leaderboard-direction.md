# Select the best leaderboard score in the task's metric direction

- **Date**: 2026-10-23
- **PR**: [#127](https://github.com/taitran501/ai-harness/pull/127)
- **Branch**: `fix/agent-leaderboard-direction`
- **Status**: open

## Why

An agent querying `ai-harness submission latest` or `KaggleLeaderboardTracker.latest` was told the largest score was best, even in a workspace declaring a minimize metric such as RMSE. With submitted scores 0.10 and 0.50, the tracker selected 0.50, directing the agent toward the worse submission.

## What changed

The tracker resolves direction from the active task and ranks finite scores accordingly. Both Python and CLI accept an explicit direction override for another competition. Results identify the direction and whether it came from the task, an explicit argument, or the legacy no-task default. Zero remains a valid best score; nonfinite values are excluded from ranking.

## Approach

Resolve the direction once at the tracker boundary so Python and CLI callers share the same behavior. Preserve maximize outside configured workspaces for compatibility and expose that default rather than presenting it as task knowledge. No submissions are dispatched or modified by this read-only lookup.

## Evidence

The previous implementation selected 0.50 over 0.10 in a minimizing competition. Regression tests exercise actual tracker selection and the CLI in an attached RMSE workspace, explicit overrides, no-task compatibility, zero/NaN/infinite scores and invalid directions. Full suite: 1,075 tests, 2 skipped, no failures/errors; Ruff and diff whitespace checks pass. All Kaggle responses are local fixtures.

## Impact & follow-ups

Agents receive the correct best submission for minimizing and maximizing tasks. When querying another competition or working without a task contract, pass its metric direction explicitly. Existing evidence and submission registries are unchanged.
