# M4 refactoring (PR-D): dashboard data layer extraction

- **Date**: 2026-10-18
- **PR**: [#98](https://github.com/taitran501/ai-harness/pull/98)
- **Branch**: `refactor/dashboard-config-extraction`
- **Status**: open
- **Commits**: `c7bdab7`

## Why

External review finding **M4**: `tools/dashboard_server.py` (2592 lines) mixed
two distinct layers in one module — the *data layer* (task discovery/switching,
task-config load/save, experiment-ID sequencing, priming-prompt generation,
`TASK_PRESETS`, `DEFAULT_STRATEGY_CARDS`) and the *HTTP layer*
(`DashboardHandler`, 43 methods, `run_server`). This is the first of three
slices in the M4 refactor (PR-E: agent_bridge action handlers; PR-F: runner
pipeline/data clusters); the slice order is lowest-risk-first, so this PR
proves the extraction pattern on the most isolated module.

## What changed

**New: `tools/dashboard_config.py`** (812 lines) — the dashboard data layer,
moved **verbatim** (763-line block, byte-identical, verified by diff):
- `get_next_experiment_id`, `list_available_tasks`, `switch_task`,
  `load_task_config`, `save_task_config`, `generate_prompt_by_template`
- module-level data: `TASK_PRESETS`, `DEFAULT_STRATEGY_CARDS`
- project discovery + the `HARNESS_PROJECT_ROOT` / `HARNESS_TASK_CONFIG`
  environment side-effects, now the **single source of truth** for dashboard
  project initialization (importing the module initializes the context).

**`tools/dashboard_server.py`** (2592 → 1829 lines) — now the HTTP layer only:
- `DashboardHandler`, `run_server`, `main` unchanged.
- Imports the data layer from `tools.dashboard_config` (including
  `TASK_PRESETS` / `DEFAULT_STRATEGY_CARDS`, which the handler consumes).
- `switch_task` re-exported (facade) for existing importers
  (`tests/test_cloud_jobs_and_env.py`); `PROJECT_ROOT` / `HARNESS_PROJECT` /
  `ARTIFACTS_DIR` remain module attributes as before.

## Approach

- **Pure mechanical move** — no logic line changed. The block was extracted
  by script and verified byte-identical against the original.
- **Import order preserved**: the server imports `tools.dashboard_config`
  (which runs discovery + env side-effects) *before* the harness manager
  imports, exactly mirroring the original module order.
- **Facade compatibility**: every name the old module exposed is still
  importable from `tools.dashboard_server`; no call site outside the two
  files was touched.

## Evidence

- Byte-equivalence: moved block (763 lines) is byte-identical to
  `git show HEAD:tools/dashboard_server.py` L61–L825.
- `tests/test_dashboard.py` (full HTTP layer, slow tier) +
  `tests/test_cloud_jobs_and_env.py` (task switch/list): pass.
- Full suite: `python tools/run_tests.py` → **908 tests OK**.
- `ruff check tools/` clean.
- Smoke import: `run_server`, `DashboardHandler`, all data-layer functions
  resolve from both modules against an attached workspace.

## Impact & follow-ups

- No behavior change; module layout only. New file is `tools/`-internal
  (not part of the `harness` package).
- PR-E: extract `AgentBridge`'s 30 `_handle_*` methods (~1860 lines) into a
  mixin module behind `harness/bridge_common.py`.
- PR-F: extract `UniversalRunner` pipeline/data clusters into mixin modules.
