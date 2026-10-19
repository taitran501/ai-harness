# Unified metrics_evaluator hook reference parsing + sys.path robustness in validate_run

- **Date**: 2026-10-19
- **PR**: [#109](https://github.com/taitran501/ai-harness/pull/109)
- **Branch**: `fix/arc-prize-audit-remediation`
- **Status**: open
- **Commits**: `09ccfe6`

## Why

The arc-prize-2026 remote kernel (job_20261019_095919) died in
`validate_run` with
`AttributeError: module 'src' has no attribute 'evaluator:evaluate_rhae'`.

Root cause: the `paths.metrics_evaluator` hook reference is written in
**colon** format (`src.evaluator:evaluate_rhae`), but the artifact
integrity gate parsed it with `hook.rsplit(".", 1)` — treating
`evaluator:evaluate_rhae` as the *function name* on module `src`. The
runner (`UniversalRunner.__init__` / `_evaluate`) and
`oof_revalidation._custom_hook` both parse the same reference with a
colon-aware split, so the runner scored fine while the integrity gate
crashed on any run that used a custom scorer. Three independent call
sites had drifted apart on the same reference grammar.

Second defect in the same function: `validate_run` runs outside the
runner on journal sync, repair, and artifact-import paths. There the
runner's `sys.path.insert(0, project_root)` never happened, so hook
modules importable only from the workspace root failed even with a
correct parse (remote kernels set `HARNESS_PROJECT_ROOT`, attached
workspaces resolve via `harness.project`).

## What changed

- **`harness/hook_ref.py` (new module)** — single source of truth:
  - `resolve_metrics_hook(hook) -> (module, function)`: colon takes
    precedence; otherwise the last dot splits; whitespace tolerated;
    invalid references (`bare`, `module:`, `:name`, empty) raise a clear
    `ValueError`.
  - `ensure_hook_project_roots_on_sys_path()`: puts the task workspace on
    `sys.path` (from `HARNESS_PROJECT_ROOT`, then `resolve_project_root()`).
  - Deliberately **not** part of the scorer identity hash scope
    (`scorer_sha256` seals `harness.evaluation` + `harness.task_spec`
    sources only), so hardening the parser can never invalidate sealed
    runs.
- **`harness/artifacts.py`** — `validate_run` now uses
  `ensure_hook_project_roots_on_sys_path()` + `resolve_metrics_hook()`
  for the custom-scorer identity check (was: bare `rsplit(".", 1)`).
- **`harness/runner.py`** — both inline splits
  (`__init__` digest capture, `_evaluate` call site) replaced with
  `resolve_metrics_hook()`; the existing `sys.path` insertion is kept.
- **`harness/oof_revalidation.py`** — `_custom_hook` delegates to
  `resolve_metrics_hook()`.
- **`tests/test_custom_scorer_validation.py` (new)** — 8 tests: parser
  unit coverage (colon/dot/deep-module/whitespace/invalid) +
  `validate_run` integration with a real on-disk hook module that is
  only importable via `HARNESS_PROJECT_ROOT` (both formats pass;
  tampered `custom_scorer_sha256` still rejected).

## Approach

A shared helper in a new, hash-out-of-scope module instead of editing
`harness/evaluation.py`: the latter's source is part of
`scorer_sha256` for default-scorer runs, so adding code there would
re-validate every existing sealed run with a mismatch and demote them
to `LEGACY_UNVERIFIED` on journal sync. The new module keeps the
integrity gate strict (both formats accepted, tamper still detected)
while making the three call sites structurally incapable of drifting
again. The `sys.path` helper is defensive (env var first, resolver
second, both optional) and only ever *adds* import roots.

## Evidence

- `tests/test_custom_scorer_validation.py`: 8/8 pass.
- `python tools/run_tests.py --smoke`: 47 tests OK.
- Targeted regression set (recursive sealing, repair, journal sync
  dedupe, task validate, artifact dedup): 72 passed, 1 skipped.

## Impact & follow-ups

- Runs using a custom `metrics_evaluator` in colon format are no longer
  rejected by the integrity gate on sync/import; dot format was already
  working and continues to work (identical parse result).
- No sealed-run identity change (helper is outside the hash scope).
- Follow-up (none blocking): the parser could additionally verify the
  imported callable is a function — deferred, out of scope.
