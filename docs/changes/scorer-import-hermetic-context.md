# Hermetic scorer import on artifact-import paths + actionable diagnostic

- **Date**: 2026-10-19
- **PR**: [#111](https://github.com/taitran501/ai-harness/pull/111)
- **Branch**: `feat/scorer-import-hermetic` (squash-merged, deleted)
- **Status**: merged
- **Commits**: `10e663e` (squash, on `main`)

## Why

Two remote kernels finished training COMPLETE, but the local output import
died with a bare `No module named 'src'` (arc-prize-2026 audit, gaps G1/C,
steps 726 and 1715). The scorer hook is imported by `validate_run`, which
runs in *detached* processes — journal sync, repair, the `pull` CLI, and the
Kaggle output import — that have no ambient workspace context. There,
`resolve_project_root()` falls back to the **framework checkout** (it has
`configs/task.yaml` but no workspace `src/`), so `import src.evaluator`
cannot resolve and the run's sealed output is lost. The error also gave the
agent no indication that the missing piece was the workspace on `sys.path`.

## What changed

- **`harness/hook_ref.py`** — `ensure_hook_project_roots_on_sys_path()`
  accepts an optional `project_root`. When given, it is inserted on
  `sys.path` with **precedence** over the `HARNESS_PROJECT_ROOT` env var and
  the `resolve_project_root()` fallback (inserted last via `insert(0)`, so it
  lands first). `os`/`sys`/`Path` moved to module-level imports.
- **`harness/templates/kaggle_template.py`** —
  `migrate_downloaded_artifacts(..., *, project_root=None)` calls
  `ensure_hook_project_roots_on_sys_path(project_root)` before both
  `validate_run` calls, so an artifact import from any cwd resolves the
  run's `src.*` scorer. `pull_kernel_output` (which already receives
  `project_root`) now forwards it to `migrate_downloaded_artifacts`.
- **`harness/artifacts.py`** — `validate_run` wraps the scorer import. A
  `ModuleNotFoundError`/`AttributeError` now becomes a `RuntimeError` that
  names the hook reference, the offending module + function, the current
  `HARNESS_PROJECT_ROOT` env value, a `sys.path[:5]` excerpt, and explicit
  guidance to pass the workspace root explicitly. Integrity checks (all the
  SHA-256 digest / hash comparisons) are untouched — this only changes the
  *diagnostic* on a missing module, not what is verified.
- **`harness/templates/kaggle_template.py`** — the generated
  `run_kaggle.py` now sets `os.environ["HARNESS_PROJECT_ROOT"] =
  str(workspace)` right after `os.chdir(workspace)`. Defense-in-depth for
  the parent-process artifact export (`export_run -> validate_run`) in the
  remote kernel, which imports the scorer in the same process that ran the
  runner.
- **`tests/test_scoring_import_hermetic.py` (new)** —
  - `test_migrate_with_project_root_imports_scorer_from_detached_cwd`: seals a
    run with a real `scorer_pkg.evaluator:evaluate_rhae` hook, chdirs to a
    bare temp dir with the env cleared, and asserts
    `migrate_downloaded_artifacts(..., project_root=ws)` imports + migrates.
  - `test_validate_run_failure_message_names_hook_and_context`: asserts the
    `RuntimeError` message carries the hook, module, `project_root=`, and
    `sys.path`, with the original `ModuleNotFoundError` as `__cause__`.
  - `test_ensure_hook_project_roots_prefers_explicit_root`: asserts the
    explicit root becomes `sys.path[0]`.
- **`tests/test_kaggle_runtime.py`** —
  `test_generated_script_pins_project_root_env_for_parent_process`: asserts
  the generated `run_kaggle.py` pins `HARNESS_PROJECT_ROOT`.

## Approach

Two independent defenses because a single one is not sufficient:

1. **Deterministic context propagation.** The import path
   (`pull_kernel_output -> migrate_downloaded_artifacts -> validate_run`)
   already carries the correct workspace (`project_root`); it simply stopped
   passing it to the `sys.path` helper. Explicit root > env > resolved keeps
   the detached import correct even when `resolve_project_root()` would
   otherwise pick the framework checkout.
2. **Actionable failure.** If the module still cannot be found, the error
   must tell the agent *which* hook, *where* it looked (`project_root`,
   `sys.path` excerpt), and *what to do* (pass the workspace root). The
   previous bare `ModuleNotFoundError` forced blind retry.

The integrity contract is deliberately **not** relaxed: the digest check on
`inspect.getsource(...)` still runs exactly as before; only the import step
gains a try/except that surfaces a better message. `export_run` in the
generated script gets the env var so the parent-process re-validation has
the same hermetic root the child runner used.

## Evidence

- `tests/test_scoring_import_hermetic.py`: 3/3 pass.
- `tests/test_kaggle_runtime.py`: all pass incl. new
  `test_generated_script_pins_project_root_env_for_parent_process` and the
  live `test_generated_script_runs_full_validation_and_exports_one_manifest`
  (executes the generated program + real runner end-to-end).
- `tests/test_submission_and_cloud.py` `migrate_downloaded_artifacts`: pass
  (signature change is backward-compatible — `project_root` is keyword-only,
  default `None`).
- **Full suite `python tools/run_tests.py`: 966 tests, OK (skipped=1), exit 0.**
- `python tools/run_tests.py --smoke`: 47 tests OK.
- `ruff check harness/ tools/ tests/`: clean (after moving `os`/`sys`/`Path`
  to module-level to satisfy the F821 undefined-name gate on the annotation).

## Impact & follow-ups

- Detached artifact imports (journal sync, repair, `pull`, Kaggle output
  import) now resolve workspace scorers hermetically from any cwd; a missing
  scorer yields a diagnostic, not a silent loss of a completed run.
- No behaviour change for runs without a custom scorer, or for the runner's
  own in-process scoring (which already sets the project root).
- Follow-up (none blocking): `cloud_job_runner` / `cli.py` / `cloud_runner`
  all already pass `project_root=project.root` to `pull_kernel_output`, so
  they inherit the fix without change.
