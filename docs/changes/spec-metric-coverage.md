# Spec-time metric/direction coverage for custom evaluators

- **Date**: 2026-10-19
- **PR**: [#106](https://github.com/taitran501/ai-harness/pull/106)
- **Branch**: `main` (squash commit `dd0e09f`)
- **Status**: merged
- **Commits**: `dd0e09f` (squash)

## Why

The harness only detected spec-level metric/direction inconsistencies at
**scoring time**, after compute had already run. Observed live on the
ARC-AGI-3 Kaggle workspace: the task spec declared `primary_metric: rhae,
direction: maximize` while the subtask declared `primary_metric: rmse,
direction: maximize` (RMSE is canonically minimized). Because a custom
`paths.metrics_evaluator` was configured, every official plugin evaluation
failed with `Primary metric/direction must match the generic aggregate`,
polluting every fold with `official evaluation failed` warnings and making
the run look broken even though the custom scorer produced the official
score. Secondary coverage gaps found while fixing:

- `validate_task` (static spec validation) had no notion of canonical
  metric directions, so `rmse` + `maximize` was never flagged before
  dispatch.
- `interactive_agent` — a task type used by live competition specs — was
  absent from `KNOWN_TASK_TYPES` (validator error) and from the runner
  preflight metric allow-list (empty set → "Unsupported task metric at
  preflight" for any interactive spec without a custom hook).
- A subtask that omits its metric silently defaulted to `macro_f1`,
  diverging from the spec's `primary_metric` and hitting the same
  aggregate gate at scoring time for perfectly ordinary single-target
  specs.

## What changed

- `harness/evaluation.py`
  - Extracted canonical direction sets `MINIMIZE_METRICS` /
    `MAXIMIZE_METRICS` from `default_metric_direction`.
  - New `GENERIC_ALLOWED_METRICS` table — the single source of truth for
    which metrics the generic aggregate can score per stock task type
    (previously an inline literal in the runner preflight only).
  - New `canonical_direction_conflicts(metric, direction)` → human-readable
    conflict note (or `None`); unknown/custom metrics never conflict
    (declared direction is trusted).
  - The aggregate gates in `evaluate()` now raise **actionable** messages:
    mixed subtask pairs list the actual pairs; the spec-pair mismatch lists
    both pairs and names the two fixes (align the subtask, or declare
    `paths.metrics_evaluator`).
- `harness/task_validate.py`
  - `interactive_agent` added to `KNOWN_TASK_TYPES`.
  - New spec-time checks in `validate_task`:
    - direction-consistency warnings for the spec and each subtask
      (canonical-metric/opposite-direction class, e.g. `rmse`+`maximize`);
    - generic-aggregate feasibility: subtask (metric, direction) pair(s) vs
      the spec pair — **error** when the generic aggregate would refuse and
      no `paths.metrics_evaluator` is declared; **warning** (spec is still
      `ok`) when a custom evaluator is declared, telling the agent that the
      official score comes from the hook and that it must return the spec's
      primary metric;
    - non-generic metric under a stock task type (e.g. `rhae` under
      `regression`) — error without a custom scorer, warning with one.
- `harness/runner.py`
  - Preflight (run construction) now evaluates the same aggregate
    feasibility before dispatch — pair mismatch, mixed pairs, **and**
    non-generic metrics under stock task types —: hard `ValueError` on the
    pure generic path (fail before compute instead of after), warning on
    the custom-hook / custom-plugin path.
  - When a mismatch is known and a custom hook is configured, the generic
    base evaluation is **skipped** per fold (`_skip_generic_base_eval`)
    instead of raising-and-catching, so `_evaluation_warnings` stays clean
    and the run log carries the condition once (preflight) rather than per
    fold. Runtime fallback: if the base evaluation fails on the first fold
    for any reason the preflight could not predict, the failure is logged
    once and the base evaluation is skipped on later folds.
  - The metric allow-list preflight is bypassed for
    `interactive_agent` task types (generic allow-lists don't apply), and
    now reads the shared `GENERIC_ALLOWED_METRICS` table.
- `harness/task_spec.py`
  - `UniversalTaskSpec.from_dict`: a subtask that declares neither
    `primary_metric` nor the `metric` alias now **inherits the spec's
    primary metric and direction**, so single-target specs score through
    the generic aggregate instead of silently defaulting to `macro_f1`.
- `tests/test_task_validate.py`
  - 5 new tests: canonical direction conflict warns (and is
    warnings-only); aggregate mismatch errors without a custom scorer;
    aggregate mismatch warns (ok) with a custom scorer; mixed subtask
    metrics error without / warn with a custom scorer; custom metric under
    a stock task type errors without / warns with a custom scorer.

## Approach

Fail early + degrade by configuration, in three coordinated layers:

1. **Spec time** (`validate_task`): structural checks with an
   error/warning split — errors block, warnings are advisory. The
   aggregate-feasibility check is an *error* only when the run would
   provably die at scoring (no custom scorer); with a custom scorer it is
   a *warning* because the hook is authoritative by design.
2. **Run preflight** (runner `__init__`): the same decision before
   dispatch, so the generic path fails before compute instead of after.
3. **Scoring time** (`evaluate()`): unchanged gate, defense in depth, with
   actionable messages; when the preflight already proved the aggregate
   unscoreable and a hook is configured, the base call is skipped rather
   than raised-and-caught (no behavior loss: the pair gate raises before
   any metric is computed, so `base_result` was always empty in that case).

Alternatives considered and rejected:

- *Silently relaxing the `evaluate()` gate when a hook is configured* —
  rejected: the gate also protects specs with no hook, and silently
  accepting mismatched pairs would hide the spec bug instead of teaching
  the agent about it.
- *Making the subtask-metric mismatch an error even with a custom scorer* —
  rejected: custom evaluators are a supported, documented configuration
  (`paths.metrics_evaluator`); a spec where the local training metric
  differs from the official metric is legitimate (e.g. proxy metric
  locally, official score remotely).

## Evidence

- `python -m pytest tests/test_task_validate.py -q` → **15 passed**
  (11 pre-existing + 4 new).
- `python tools/run_tests.py --fast` → **835 tests**, 0 errors, 1
  failure (`test_generated_cloud_bundles_compile_and_exclude_data`) which
  is **pre-existing** in the working tree (reproduces with this change
  stashed — caused by unrelated uncommitted cloud/kaggle-template edits).
- `ruff check` on the five touched files → clean.
- Live check on the real arc-prize-2026-arc-agi-3 spec:
  `validate_task` now returns `ok: True` with exactly the two actionable
  warnings (direction conflict + aggregate mismatch naming the custom
  evaluator and the required returned metric) instead of the previous
  silence followed by per-fold "official evaluation failed" at scoring.
## Impact & follow-ups

- The arc-prize workspace spec should still be fixed by its agent: subtask
  `primary_metric: rmse, direction: maximize` → `rhae, maximize` (the
  direction-conflict warning now says so explicitly). Once the subtask
  metric is `rhae`, the aggregate pair matches but `rhae` is not a generic
  metric for `regression`, so the run preflight warns once and the generic
  base evaluation is skipped — the custom hook is the sole scorer, with no
  per-fold noise.
- The pre-existing dirty files in the framework working tree
  (`artifacts.py`, `cloud_job_runner.py`, `cloud_runner.py`,
  `remote_runtime.py`, `kaggle_template.py`) predate this change and own
  the one failing fast-suite test; they should be committed or reverted
  separately.
- Follow-up candidate: surface `validate_task` warnings in the
  `ai-harness attach` / `context` HUD so agents see them without running
  the CLI explicitly.
