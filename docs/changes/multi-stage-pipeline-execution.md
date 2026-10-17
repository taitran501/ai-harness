# Multi-stage pipeline execution (PR #81)

- **Date**: 2026-10-17
- **PR**: [#81](https://github.com/taitran501/ai-harness/pull/81)
- **Status**: merged (squash)
- **Area**: runner / execution

## 1. Why (motivation + thought)

Closes the single-entrypoint gap. Historical Antigravity audits showed agents
writing dozens of bypass orchestration scripts (82 in `rsna-knee`, 38 in
`airline_prediction`) because `UniversalRunner` executed exactly one
entrypoint. Multi-stage workflows (train N models → blend → stack) had no
native support, so the agent hand-wrote one big orchestration script outside
the harness, losing the per-stage audit trail. The single-entrypoint design was
a *structural* gap, not a missing feature.

## 2. What (concrete changes)

- **`harness/runner.py`** — `execution.mode: pipeline` dispatch in
  `_execute_run` (validates `approach ∈ {script, custom_script}`, skips the
  model-factory gate) + new `_run_pipeline` / `_resolve_pipeline_stages`.
  Stages run **sequentially** in config order as audited subprocesses sharing
  the experiment directory (later stages read earlier stages' artifacts). Two
  config forms: `execution.pipeline` (named stages with per-stage `args`) and
  `execution.entrypoints` (simple list, names from the script stem).
  Per-stage stdout/stderr → `stages/<name>.log`; per-stage exit
  code/duration/log → `stage_manifest.json` (persisted even on failure). New
  env vars: `HARNESS_STAGE_NAME/INDEX/TOTAL/DIR`. Fail-fast: the first non-zero
  stage stops the pipeline and raises a `RuntimeError` naming the stage. The
  final stage must produce `oof_predictions.csv`, verified exactly as
  `mode: script` before `_save_and_finalize`.
- **Back-compat**: `execution.mode: script` (single `entrypoint`) unchanged.
- **Docs**: `docs/agent_workflow.md` §5 gained "Multi-Stage Pipeline
  Execution"; the `AGENTS.md` doc-map entry for `agent_workflow.md` mentions
  the pipeline mode.
- **`tools/run_tests.py` env isolation**: `main()` now pops
  `HARNESS_FRAMEWORK_DIR` before running, so the gate is reliable regardless of
  the machine's global state (the one-time single-install migration had set it
  globally, which contaminated the sync / toolchain-pin / status / runtime
  suites and produced false `--fast` failures).

## 3. Approach (decisions + rejected alternatives)

A first-class sequential stage runner (with per-stage logs + a manifest)
restores the audit guarantee without changing the single-script contract.
Chose sequential (not parallel) for determinism + the audit trail; each stage
is a real audited subprocess, so the immutable-artifact rule still holds per
stage.

## 4. Evidence

- 9 new tests in `tests/test_pipeline_execution.py` (sequential success,
  simple-entrypoints form, fail-fast, approach validation,
  `_resolve_pipeline_stages` named/simple/empty/collision/missing-entrypoint);
  `test_script_adapter.py` (back-compat) 2/2; smoke gate 47 tests OK;
  ruff F,E7 clean.

## 5. Impact & follow-ups

- **Follow-up**: PR #82 (pipeline adoption stack — audit gate, config
  hardening, discoverability, scaffolds) makes `mode: pipeline` trustworthy +
  discoverable. See `pipeline-adoption-stack.md`.
- **Downstream**: agents declare `execution.mode: pipeline` instead of writing
  a monolithic orchestration script.
