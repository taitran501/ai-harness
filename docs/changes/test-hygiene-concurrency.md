# Change: Test Hygiene — Concurrency Guard + Extended Drift Guards

- **Date**: 2026-10-19
- **PR**: [#108](https://github.com/taitran501/ai-harness/pull/108)
- **Branch**: `main` (squash commit `479503c`)
- **Status**: merged
- **Commits**: `479503c` (squash)

## Why

1. **Test Flakiness**: Concurrent `run_tests.py` invocations on the same
   checkout cause `PermissionError: [WinError 32]` on `__pycache__` and
   system-scoped cache files (e.g. `.model_names_cache.json`), leading to
   spurious failures.
2. **Maintenance Drift**: The `GENERIC_ALLOWED_METRICS` table for classification
   (binary/multilabel) needed the same drift-guard coverage as regression to
   ensure the allow-list stays in sync with `evaluate()`'s actual scoring
   branches.

## What changed

### `tools/run_tests.py`: Concurrency Guard
- Added a `_SuiteLock` advisory guard that acquires an exclusive lock in the
  OS temp directory (scoped to the checkout via a repo-root hash).
- **Behavior**: A second suite invocation fails fast (exit 3) with a detailed
  error (PID, age, host) instead of producing corrupt/flaky results.
- **Robustness**: Stale locks (dead PID or age > 1 h) are automatically stolen.
- **Bypass**: Added `--no-lock` for edge cases where overlapping runs are
  explicitly desired.

### `tests/test_task_validate.py`: Extended Drift Guards
- `test_allowed_table_matches_evaluate_for_binary_classification`: exercises
  all 11 binary metrics (including probability-dependent ones like `roc_auc`
  and `brier_score`) against `evaluate()` to prove table alignment.
- `test_multilabel_table_differs_from_binary_only_by_ndcg`: enforces the exact
  multilabel contract (binary set minus `qwk`/`brier_score` plus `ndcg`).

## Evidence

- `tools/run_tests.py` syntax/compile check: OK.
- Concurrency test: started a background suite; a second foreground suite
  failed correctly with exit 3.
- Bypass test: `--no-lock` allowed a concurrent smoke run.
- Lock release: after holder suite finished, subsequent runs pass normally.
- Classification drift guards: `tests/test_task_validate.py` 20/20 pass.

## Impact & follow-ups

- Eliminates false-negative "failures" caused by overlapping agent turns or
  automation jobs.
- Protects the generic aggregate's scoring contract against silent code drift
  in `evaluate()`.
