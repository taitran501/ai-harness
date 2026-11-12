# Agent Ergonomics, Backend Switching & Hygiene Ignore Lists

- **Date**: 2026-11-12
- **PR**: —
- **Branch**: `feat/agent-ergonomics-and-backend-override`
- **Status**: open
- **Commits**: pending

## Why

Auditing real AI coding agent sessions over the past 7 days across competition workspaces (`zalo_ai_challenge`, `neurips`, `ai-harness`) identified three recurring operational friction points:
1. **Backend Lock Deadlock**: When a task contract specifies `execution: {backend: modal, locked: true}` and the human operator requests switching to Kaggle or local training, agents faced fatal `Backend lock violation` errors without a safe CLI command to mutate the contract.
2. **PowerShell JSON & Schema Friction**: Calling `ai-harness-agent act --action record_research_insight --input '{"title": "..."}'` in PowerShell failed frequently due to quote stripping, while missing explicit schema typing caused string fields to be coerced to bool/int. Agents were forced to write ad-hoc Python scripts in scratch folders.
3. **Workspace Hygiene False Alarms**: Legitimate project directories like `storage/`, `prototypes/`, or `data_raw/` continuously triggered clutter warnings in the Context HUD because only `task.hygiene.allowed_top_dirs` was checked, while agents and users expected aliases like `allow_dirs` or `ignore_dirs`.

## What changed

- **`harness/cli.py`**:
  - Added `ai-harness set-backend <backend> [--project-root <path>] [--unlock] [--lock]` command to safely update `execution.backend` and lock state in `task.yaml`.
- **`harness/runner.py` & `harness/agent_bridge.py` & `harness/cloud_runner.py`**:
  - Enhanced lock violation error messages to explicitly instruct running `ai-harness set-backend <backend> --unlock`.
  - Added task execution backend lock enforcement to `harness/cloud_runner.py` before remote preparation.
- **`harness/bridge_common.py` & `harness/bridge_actions.py`**:
  - Extended `update_task_contract` to accept partial execution patches (`backend`, `locked`).
  - Added strict parameter typing (`parameter_types`) for research actions (`record_research_insight`, `record_research_decision`, `record_research_checkpoint`) so flat CLI arguments preserve string types.
  - Added defensive alias support (`summary`, `title`, `conclusion`, `note`, `decision`) so agents don't fail when varying argument names.
- **`tools/agent_bridge_cli.py`**:
  - Added `--input` as a direct alias for `--params`.
  - Fixed `ensure_pinned_toolchain` to forward `argv` rather than hardcoded `sys.argv[1:]`.
- **`harness/hygiene.py` & `harness/task_validate.py`**:
  - Added support for `allow_dirs`, `ignore_dirs`, and `ignore` as aliases for `allowed_top_dirs` under `hygiene` in `task.yaml`.
  - Allowed both lists and single strings.

## Approach

1. Followed Codex's architectural recommendations: avoid ad-hoc bypass flags like `--force-backend` on the runner that would undermine contract immutability; instead provide an explicit, authorized contract mutation command (`ai-harness set-backend`) that validates against `UniversalTaskSpec` and atomically updates `task.yaml`.
2. Leveraged existing dynamic KV arg parsing in `tools/agent_bridge_cli.py` by declaring explicit string parameter types in `TYPED_ACTION_SCHEMAS`, allowing PowerShell agents to invoke flat flags cleanly (e.g. `ai-harness-agent act --action record_research_insight --title "..." --observation "..."`).
3. Made hygiene allow-list loading permissive and robust against common configuration aliases.

## Evidence

- `ruff check harness/ tools/ tests/`: 100% clean (0 errors).
- `python tools/run_tests.py --smoke`: 53 tests pass in 7.2s.
- `python -m unittest tests/unit/test_execution_backend_policy.py tests/integration/test_hud_ergonomics.py`: 45 tests pass in 17.1s.

## Impact & follow-ups

None. Completely backwards-compatible with existing task contracts, artifacts, and CLI workflows.
