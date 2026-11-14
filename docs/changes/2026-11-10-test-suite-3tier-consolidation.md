# Test Suite 3-Tier Consolidation and Canonical Domain Restructuring

- **Date**: 2026-11-10
- **PR**: Working tree
- **Branch**: `feat/test-suite-consolidation`
- **Status**: merged
- **Commits**: Working tree

## Why

Over iterative milestones, the test suite expanded to more than 135 ad-hoc test files scattered across the root `tests/` directory, `tests/agent_bridge/`, and partial subdirectories. This sprawl caused several issues:
1. High discovery and runtime overhead with no structured tier separation.
2. Cross-test environment leakage where lingering environment variables (such as `HARNESS_TASK_CONFIG`) contaminated downstream tests.
3. Lack of clear test classification into fast unit tests, intermediate integration tests, and long-running end-to-end (e2e) tests.
4. Slow feedback loops for developers needing fast iteration on core framework invariants.

## What changed

- **Canonical 3-Tier Architecture**:
  Consolidated all 135 legacy test files into exactly 38 canonical domain modules distributed across three dedicated tiers:
  - **Unit Tier** (`tests/unit/`, 15 canonical modules):
    `test_citation_audit.py`, `test_data_strategy.py`, `test_ensemble_guard.py`, `test_evaluation_contracts.py`, `test_evidence_manifest.py`, `test_execution_backend_policy.py`, `test_feature_pipeline.py`, `test_grounding_gate.py`, `test_io_utils.py`, `test_label_integrity.py`, `test_macro_research.py`, `test_model_governance.py`, `test_models.py`, `test_research_memory.py`, `test_task_spec.py`.
    Also includes 3 backward-compatible smoke shims (`test_correctness_contracts.py`, `test_correctness_candidates.py`, `test_research_scope_stability.py`) re-exporting canonical domain test suites.
  - **Integration Tier** (`tests/integration/`, 17 canonical modules):
    `test_agent_bridge.py`, `test_audit_gate_integration.py`, `test_citation_lifecycle.py`, `test_cloud_dispatch.py`, `test_custom_scorer.py`, `test_framework_adoption.py`, `test_interactive_prediction.py`, `test_kaggle_cli.py`, `test_reconciliation_integration.py`, `test_repair_and_cleaner.py`, `test_runner.py`, `test_script_adapter_integration.py`, `test_single_install_integration.py`, `test_submission_manager.py`, `test_toolchain_and_plugins.py`, `test_tree_journal.py`, `test_web_dashboard.py`.
  - **E2E Tier** (`tests/e2e/`, 6 canonical modules):
    `test_cloud_runners.py`, `test_dashboard_e2e.py`, `test_kaggle_runtime.py`, `test_pipeline_e2e.py`, `test_sync_e2e.py`, `test_workspace_workflow.py`.

- **Legacy Cleanup**:
  - Deleted all consolidated legacy test files from `tests/` root.
  - Completely removed `tests/agent_bridge/` directory.
  - Preserved root test fixtures: `tests/fixtures.py`, `tests/evidence_fixtures.py`, `tests/workspace_fixtures.py`, `tests/__init__.py`.

- **Test Isolation & Environment Hygiene**:
  - Scoped `HARNESS_TASK_CONFIG` removal within `tests/integration/test_submission_manager.py::LeaderboardDirectionTests.test_without_a_task_the_legacy_maximize_default_is_explicit` to guarantee clean hermetic execution without cross-test leakage.

- **Test Runner Ergonomics (`tools/run_tests.py`)**:
  - Implemented multi-tier command line options:
    - `--smoke`: Runs critical invariant modules in <10s (51 tests in ~1.4s).
    - `--fast`: Runs fast tier (35 modules, excluding 6 slow pipeline/e2e suites).
    - `--unit`, `--integration`, `--e2e`: Direct single-tier execution filters.
    - Default (no flags): Executes the full test suite across all 38 modules.
  - Added temporary file-based advisory locking to prevent concurrent test execution collisions.

## Approach

- Grouped related tests into unified domain modules based on subsystem boundaries and architectural contracts while preserving 100% of assertions and test coverage.
- Wrapped un-parameterized standalone pytest-style functions into `unittest.TestCase` classes to ensure uniform discovery and execution by `tools/run_tests.py`.
- Maintained exact API contracts and invariant checks for correctness contracts, candidate labels, research scope stability, and execution policies.
- Kept root test fixtures untouched and shared across all tiers to eliminate duplication.

## Evidence

- **Pre-Push Quality Gates**:
  1. `python tools/run_tests.py --smoke`: 51 tests passed in ~1.4s (<10s invariant satisfied).
  2. `python tools/run_tests.py --fast`: 35 fast modules passed 100%.
  3. `python tools/run_tests.py`: Full suite passed 100% across all 38 domain modules.
  4. `ruff check harness/ tools/ tests/`: 0 errors / all checks passed.
- **Module Count Verification**:
  - `tests/unit/`: 15 canonical modules + 3 smoke shims
  - `tests/integration/`: 17 canonical modules
  - `tests/e2e/`: 6 canonical modules
  - Total canonical domain modules: 38

## Impact & follow-ups

- Eliminates clutter in `tests/` root and provides clear organization for future unit, integration, and e2e tests.
- Significantly accelerates feedback loops during development with `--smoke` and `--fast`.
- No downstream regressions or breaking changes to public framework APIs.
