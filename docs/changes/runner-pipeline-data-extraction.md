# M4 refactoring (PR-F): UniversalRunner pipeline/data extraction

- **Date**: 2026-10-18
- **PR**: — (branch `refactor/runner-pipeline-data-extraction`)
- **Branch**: `refactor/runner-pipeline-data-extraction`
- **Status**: open
- **Commits**: —

## Why

External review finding **M4** (final slice): `harness/runner.py` (1971
lines, `UniversalRunner` with 17 methods) mixed execution orchestration
with two self-contained clusters — data loading/split resolution and
multi-stage pipeline execution. This completes the M4 refactor: all three
oversized modules (`agent_bridge`, `dashboard_server`, `runner`) are now
split along single-responsibility seams.

## What changed

**New: `harness/run_data.py`** (323 lines) — `RunDataMixin` holding
`_load_training_data` and `_resolve_validation_splits`, moved
**verbatim** (302-line block, byte-identical, verified by diff).

**New: `harness/run_pipeline.py`** (332 lines) — `RunPipelineMixin`
holding `_resolve_pipeline_stages`, `_run_pipeline`, and
`_verify_pipeline_manifest`, moved **verbatim** (306-line block,
byte-identical, verified by diff).

**`harness/runner.py`** (1971 → 1363 lines) — now orchestration only:
`__init__`, `run`, `_execute_run`, model factory, custom-script, training,
LLM-prompting, and save/finalize paths, plus module functions
(`merge_configs`, `build_parser`, `resolve_cli_backend`,
`is_local_only_run`, `main`). `UniversalRunner(RunDataMixin,
RunPipelineMixin)`; the `Runner` alias and CLI entry are unchanged.

## Approach

- **Pure mechanical move** — no logic line changed; both blocks
  byte-identical to `HEAD`.
- **Dependency audit before moving**: data cluster needs
  `ExperimentValidator`, `digest`, `Path`, `np`, `pd`; pipeline cluster
  needs `save_json`, stdlib, `pd`. Both satisfied inside the mixin
  modules; runner keeps everything it still uses (verified by ruff).
- **Facade unchanged**: `UniversalRunner`, `Runner`, `build_parser`,
  `merge_configs`, `main` all still importable from `harness.runner`
  (all 10+ external import sites are tests/CLI, none touched).
- Mixin methods call sibling methods only through `self` — resolution
  happens on the merged class, so no cross-mixin static references are
  needed.

## Evidence

- Byte-equivalence: data block (302 lines) and pipeline block (306 lines)
  are byte-identical to `harness/runner.py` at `HEAD`.
- `tests/test_correctness_runtime.py`, `tests/test_e2e_pipeline.py`,
  `tests/test_artifact_dedup.py`: pass.
- Full suite: `python tools/run_tests.py` → **908 tests OK**.
- `ruff check harness/` clean.
- Smoke import: `Runner is UniversalRunner`; both mixins applied; all
  moved + remaining methods present.

## Impact & follow-ups

- No behavior change; module layout only.
- M4 complete: `agent_bridge` 4098 → 2052 (PR #99),
  `dashboard_server` 2592 → 1829 (PR #98), `runner` 1971 → 1363 (this PR).
  No module in the framework exceeds ~2100 lines except the dashboard
  HTTP handler (single cohesive request-handler class, next candidate
  only if it grows).
