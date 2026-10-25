# Plugin Contract Validation, Unified Capability Discovery, and Structured Failure Classification

- **Date**: 2026-10-25
- **PR**: — (local)
- **Branch**: `feat/plugin-validation-capability-discovery-failures`
- **Status**: open
- **Commits**: —

## Why

1. **Plugin Contract Validation (P0-02)**: Plugins could override hooks with invalid signatures or non-callable attributes, failing only midway through an expensive experiment rather than at attachment/load time.
2. **Host-Neutral Skills (P1-04B)**: The generated `SKILL.md` hardcoded "Antigravity in IDE" throughout the SOP and architecture sections, conflicting with use under Codex CLI, Claude Code, or Cursor.
3. **Unified Capability Discovery (P1-01)**: Agents had to piece together attachment, task validation, plugin capabilities, and cloud authentication from multiple CLI tools to know what a task could run.
4. **Structured Failure Classification & Recovery (P1-02 & P1-03)**: Bridge action failures previously only returned the last 500 characters of stdout/stderr with no machine-readable failure taxonomy or actionable next steps, causing agents to blindly retry or get stuck.

## What changed

- **`harness/plugins.py`**:
  - Implemented `validate_plugin_contract(plugin: TaskPlugin, raise_on_error: bool = True) -> List[str]`.
  - Added signature checking for all 13 canonical `TaskPlugin` hooks using `inspect.signature` against base hook requirements.
  - Validated that `capabilities()` returns a compliant dictionary of boolean flags.
  - Wired contract validation into `load_task_plugin()` to fail closed early on signature mismatches.
- **`harness/skill_injector.py`**:
  - Neutralized agent-host references (supporting Antigravity, Codex CLI, and Claude Code interchangeably) across architecture diagrams, HUD descriptors, and SOP workflows.
- **`harness/status_report.py`**:
  - Added `task_capabilities` section aggregating task data availability, custom scoring support, local execution, cloud authentication readiness, and submission requirements.
  - Rendered `Task Capabilities` in Markdown and structured JSON.
- **`harness/failure_classifier.py`**:
  - Created standardized error taxonomy (`CONFIG_INVALID`, `DEPENDENCY_MISSING`, `RESOURCE_OOM`, `AUTH_REQUIRED`, `QUOTA_EXCEEDED`, `DATA_CONTRACT_INVALID`, `ARTIFACT_UNVERIFIABLE`, `UNKNOWN`).
  - Added `classify_failure()` with regex/exception mapping and actionable `recovery` advice (`suggested_action`, `suggested_command`, `retryable`).
- **`harness/bridge_common.py` & `harness/bridge_actions.py`**:
  - Added optional `failure` dictionary to `ActionObservation`.
  - Integrated `classify_failure()` into `run_smoke_test`, `run_probe`, and `run_full_validation` failure paths.
- **Tests**:
  - Added `tests/test_plugin_contract_validation.py` (6 tests).
  - Added `tests/test_failure_classifier.py` (9 tests).
  - Extended `tests/test_status_report.py` with `TestTaskCapabilitiesDiscovery` (2 tests).

## Approach

All additions preserve backwards compatibility. Existing fields in `ActionObservation`, `collect_status`, and `TaskPlugin` remain completely intact. Error recovery guidance suggests specific commands (e.g. `pip install <pkg>`, `ai-harness validate-task`, `ai-harness repair --check`) without automatically running destructive or quota-consuming actions.

## Evidence

- `python -m pytest tests/test_plugin_contract_validation.py` -> 6 passed.
- `python -m pytest tests/test_failure_classifier.py` -> 9 passed.
- `python -m pytest tests/test_plugin_docs_and_example.py` -> 6 passed.
- `python -m pytest tests/test_status_report.py` -> 13 passed.
- `python -m pytest tests/test_skill_injector.py` -> 7 passed.
- `ruff check harness/ tools/ tests/ examples/` -> clean (all checks passed).
