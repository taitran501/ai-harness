# Canonical AI-Harness Rebranding & Decoupled Fast Testing

- **Date**: 2026-10-13
- **PR**: rebrand batch (followed by the `mlh-` prefix eradication)
- **Status**: merged
- **Area**: naming / test infrastructure

## 1. Why (motivation + thought)

Audit the repository for legacy `ml-harness` bias and rebrand to `ai-harness`;
resolve unit tests that ran heavy local models and stalled test cycles. Harness
unit tests validating lifecycle DAG transitions, anti-leakage gates, and
orchestration contracts should never depend on heavy numerical convergence or
live network sockets.

## 2. What (concrete changes)

- **Comprehensive rebranding**: canonical name `ai-harness` across runtime
  (`project.py`, `runner.py`, `skill_injector.py`, `phase_workflow.py`,
  `cli.py`, `plugins.py`, `environment_detector.py`, `data_strategy.py`,
  `llm_engine.py`, `templates/`), dev tools (`bootstrap_task.py`,
  `dashboard_server.py`, `agent_bridge_cli.py`, `agent_runtime_cli.py`), the web
  dashboard, the skill, and the docs.
- **Dual-path back-compat**: preserved automatic fallback for legacy
  `.ml-harness` workspaces, manifests (`ml-harness-runtime.json`,
  `mlh-data-bundle.json`), and UI preference keys.
- **Decoupled testing engine**: `DummyBaselineModel` in `harness/models.py`
  (aliases `dummy`, `mock`, `dummy_baseline`, `dummy_classifier`,
  `dummy_regressor`) — instant O(1) fitting, label alignment, and probability
  output without heavy optimization.
- **Mocked process & network probes**: mocked external OS + network calls in
  `test_cloud_jobs_and_env.py` (18.0s → 0.3s); replaced `localhost` with an
  explicit IPv4 binding in `test_llm_api_endpoints.py` (30.5s → 1.0s),
  preventing Windows IPv6 fallback timeouts.

## 3. Approach (decisions + rejected alternatives)

Decouple tests from heavy models + live network via lightweight dummy
estimators and mocked environment probes, achieving sub-second test execution
while preserving the real lifecycle/DAG/gate logic under test. Kept a dual-path
back-compat layer so existing `.ml-harness` workspaces keep working (later fully
removed in the follow-up `mlh-` eradication).

## 4. Evidence

- Full suite **392 tests in 74.13s (reduced from ~140s), 100% pass**.

## 5. Impact & follow-ups

- **Follow-up**: "Complete Eradication of Legacy ml-harness & mlh- Prefixes"
  (2026-10-13) and "Dashboard Metadata, Test Fixtures, and AGENTS.md Quality
  Gate Alignment" (2026-10-14) finished the cleanup (see archived `WORK_LOG.md`).
- **Downstream**: clean `ai-harness` naming boundaries; fast, hermetic unit
  tests.
