# Align GENERIC_ALLOWED_METRICS with evaluate() capability

- **Date**: 2026-10-19
- **PR**: [#107](https://github.com/taitran501/ai-harness/pull/107)
- **Branch**: `main` (squash commit `cf63ff5`)
- **Status**: merged
- **Commits**: `cf63ff5` (squash)

## Why

The `GENERIC_ALLOWED_METRICS` table (introduced in #106 as the single source of
truth for which metrics the generic aggregate can score per stock task type) was
carried over verbatim from the old inline allow-list in `runner.py`. That legacy
allow-list was a **conservative subset** of what `evaluate()` actually computes:

- `regression` listed 7 metrics, but the regression scoring branch in
  `evaluate()` computes **14** (missing `rmsle`, `smape`, `wape`,
  `pinball_loss`, `medae`, `pearson`, `mrr`).
- `binary`/`multiclass_classification` listed 6, but the classification branch
  computes **11** (missing `weighted_f1`, `balanced_accuracy`, `mcc`,
  `pr_auc`, `brier_score`).
- `multilabel_classification` listed 4 but computes **10**.
- `ranking` listed 4 though it shares the regression branch (14 scorable).
- The regression-family aliases (`time_series_forecasting`, `forecasting`,
  `time_series`, `multi_output_regression`) were absent entirely, so those
  task types were treated as "require a custom scorer" even though the
  generic path scores them.

Because #106 added a spec-time validation check in `task_validate` that uses
this table, the narrow table produced **live false positives**: a valid spec
declaring `rmsle` (regression) or `weighted_f1` (binary) was flagged as "not a
generic metric" and — at the runner preflight — rejected with "Unsupported task
metric at preflight", even though `evaluate()` can score it. The framework's
own tests already used `rmsle` as a regression metric (`default_metric_direction`
returns "minimize" for it), confirming the narrow table was a bug, not intent.

## What changed

- `harness/evaluation.py`: rebuilt `GENERIC_ALLOWED_METRICS` from two shared
  sets that mirror the `available` dicts computed in `evaluate()`:
  - `_REGRESSION_METRICS` (14) mapped to `regression`, `ranking`,
    `time_series_forecasting`, `forecasting`, `time_series`,
    `multi_output_regression`.
  - `_CLASSIFICATION_METRICS` (11) mapped to `binary_classification` and
    `multiclass_classification`; `multilabel_classification` = that set minus
    `{qwk, brier_score}` plus `{ndcg}` (10).
- `tests/test_task_validate.py`:
  - `test_scorable_metrics_not_flagged_as_unsupported` — regression `rmsle` and
    binary `weighted_f1` specs pass `validate_task` with no "not a generic
    metric" error/warning.
  - `test_allowed_table_matches_evaluate_for_regression` — drift guard: every
    metric in `GENERIC_ALLOWED_METRICS["regression"]` is actually scored by
    `evaluate()` (no "Unsupported metric"), and `accuracy` (not scorable for
    regression) is both rejected and absent from the table.

## Approach

Rather than hand-patching individual entries, I enumerated the exact `available`
dicts built by each branch of `evaluate()` and made the table mirror them, so the
table is a faithful description of generic scoring capability. The change is
**purely permissive** at the preflight (it only un-blocks specs that were
incorrectly rejected); it breaks no previously-valid spec. I deliberately did
**not** add the non-stock task types (segmentation, ASR, detection, NER,
interactive/agent) to the table — those remain "custom scorer required", matching
the prior design. A regression-family drift guard test locks the table to
`evaluate()` behavior so future metric additions to `evaluate()` surface as a
test failure.

## Evidence

- `tests/test_task_validate.py`: 18/18 pass (16 prior + 2 new).
- `GENERIC_ALLOWED_METRICS` now reports: regression family = 14 metrics each;
  binary/multiclass = 11; multilabel = 10 (with `ndcg`, without
  `qwk`/`brier_score`).
- Live check: `validate_task` on a regression spec with `primary_metric: rmsle`
  returns `ok: True` with no "not a generic metric" message.
- `ruff check harness/evaluation.py tests/test_task_validate.py` → All checks passed.
- `python tools/run_tests.py --fast` (isolation) → 837 OK.
- `python tools/run_tests.py` (full, isolation) → 932 OK.

## Note on a transient full-suite failure (not a regression)

A concurrent full-suite run (two suites executing in parallel) surfaced
`PermissionError: [WinError 32] ... being used by another process` on
`harness/__pycache__` and `.model_names_cache.json`, plus flaky
`test_env_cache` / `test_workspace_workflow` / dashboard errors. These were
**file-lock contention from parallel suite runs**, not a code regression: the
same tests pass in isolation on clean HEAD (33/33) and the full suite is
932 OK when run alone. Suggested hygiene: never run two `run_tests.py` suites
concurrently on the same checkout.

## Impact & follow-ups

- The runner preflight is now accurate for the stock task types: it rejects
  exactly what `evaluate()` would reject and nothing more.
- Specs using previously-misrejected metrics (`rmsle`, `smape`, `wape`,
  `pinball_loss`, `medae`, `pearson`, `mrr`, `weighted_f1`,
  `balanced_accuracy`, `mcc`, `pr_auc`, `brier_score`) now pass preflight and
  are scored by the generic aggregate.
- Non-stock task types (segmentation, ASR, detection, NER, interactive/agent)
  are unchanged: they still require a custom `paths.metrics_evaluator`.
- Follow-up (optional): extend the drift guard to the classification task types
  (needs probability-column fixtures); the regression family is fully guarded.
