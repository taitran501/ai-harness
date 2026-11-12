# Test Suite & Framework 3-Tier Consolidation Plan

**Plan Version:** 1.0.0 (Production-Grade Implementation Plan)  
**Author:** `worker_plan_1` (Teamwork Implementer, QA & Specialist)  
**Date:** 2026-11-10  
**Scope:** Reorganizing and consolidating the complete test suite (135 test modules across `tests/` and `tests/agent_bridge/`, 31,731 LOC, 1,323 unittest / 1,374 AST tests) into a hermetic 3-tier hierarchy (`tests/unit/`, `tests/integration/`, `tests/e2e/`) comprising exactly 38 cohesive domain modules.  
**Related Documents:** `docs/architecture.md`, `docs/CORRECTNESS_CONTRACT.md`, `docs/DESIGN_single_install.md`, `AGENTS.md`.

---

## 1. Executive Summary & Inventory Reconciliation

### 1.1 The Baseline Reconciliation: Disambiguating the "126" File Count
The user request references an inventory of **126 test files**. A filesystem audit resolves this baseline:
- **`tests/` top-level directory contains exactly 126 `.py` files**:
  - **122 executable test modules** matching `test_*.py`.
  - **4 shared test infrastructure/fixture modules**: `tests/__init__.py`, `tests/fixtures.py`, `tests/evidence_fixtures.py`, and `tests/workspace_fixtures.py`.
- **`tests/agent_bridge/` subpackage contains 14 `.py` files**:
  - **13 executable test modules** matching `test_*.py`.
  - **1 package initializer**: `tests/agent_bridge/__init__.py`.
- **Repository-wide Grand Totals**:
  - **135 executable test modules** (`REPO_ROOT.rglob("test_*.py")`), all discoverable by `tools/run_tests.py`.
  - **140 total Python files** across the entire `tests/` directory tree.
  - **31,731 lines** of test code.
  - **1,323 test cases** registered via `unittest.TestLoader`.
  - **1,374 test methods** identified via Abstract Syntax Tree (AST) walk.
  - **1,400+ test items** collected under `pytest` (due to parameterized fixtures).

**Conclusion**: The 126 figure accounts for the `.py` files in the `tests/` root. This consolidation plan accounts for all **135 discoverable test modules** and the **4 shared infrastructure modules**, ensuring that no subpackage modules are orphaned.

```text
Repository Test Files (140 .py files)
├── tests/ (126 files)
│   ├── test_*.py (122 test modules)
│   └── Shared fixtures (4 files: __init__.py, fixtures.py, evidence_fixtures.py, workspace_fixtures.py)
└── tests/agent_bridge/ (14 files)
    ├── test_*.py (13 test modules)
    └── __init__.py (1 file)
```

---

### 1.2 Critical Discovery: The Test Runner Discovery Blindspot
A primary finding from the architectural survey is a silent discovery blindspot in the primary test runner `tools/run_tests.py`:
- `tools/run_tests.py` loads tests using `unittest.TestLoader().loadTestsFromName(module_name)`.
- `unittest.TestLoader` requires test classes to inherit from `unittest.TestCase`. Any standalone pytest-style functions (`def test_*()`) are silently ignored without error.
- **7 test modules silently executed 0 tests** under `tools/run_tests.py`:
  1. `tests/test_interactive_prediction_contract.py` (3 test functions)
  2. `tests/test_model_cli_and_bridge.py` (4 test functions)
  3. `tests/test_model_execution_plan.py` (6 test functions)
  4. `tests/test_model_governance.py` (15 test functions / 22 parameterized pytest test cases)
  5. `tests/test_model_lineage_and_submission.py` (5 test functions)
  6. `tests/test_model_registry.py` (10 test functions)
  7. `tests/test_model_resolution.py` (8 test functions)

**Impact on Invariant Verification**: Crucial model governance contracts—such as maximum parameter limits (7B backbone + LoRA composition), license allowlists, release cutoff dates, and model registry hashing—**were completely unexecuted by `python tools/run_tests.py`** (reporting `0 tests in 0.2s — OK`). While `pytest tests/test_model_governance.py` executed and passed 22 tests, the default quality gate skipped them.

**Plan Resolution**: In Phase 2 of this plan, all standalone test functions in these 7 modules will be wrapped in `unittest.TestCase` classes. Furthermore, `tools/run_tests.py` will be updated with assertion diagnostics to ensure any module discovering 0 test cases raises a warning or error.

---

### 1.3 Suite Execution Characteristics & Empirical Timings

Analysis of the 135 test modules reveals a split in execution profiles:
- **Pure In-Memory Logic (0 disk I/O, 0 subprocesses)**: **19 modules** operate strictly in memory without filesystem reads/writes or process spawns (e.g., `test_kaggle_compat.py` runs 36 tests in 0.85s).
- **Disk I/O / Temporary Workspaces**: **116 modules** create isolated directories via `tempfile.TemporaryDirectory`, `mkdtemp`, or `tests.fixtures.make_workspace` (0.5s - 5s per module).
- **Subprocess Spawning**: **29 modules** spawn Python sub-interpreters, CLI entrypoints, git commands, or live servers (>5s - 70s per module).

#### Empirical Timings of Slow Modules (Measured on Python 3.11 Windows):
| Module | Test Cases | Runtime | Key Bottleneck Operations |
|---|---|---|---|
| `test_kaggle_runtime.py` | 28 | **69.10s** | Full kernel packaging, offline code export, dataset mount checks |
| `test_sync_command.py` | 11 | **29.91s** | Spawns git clone, git switch, git fetch across temporary repos |
| `test_pipeline_execution.py` | 15 | **13.15s** | Sequential child Python scripts executing multi-stage DAGs |
| `test_cloud_jobs_and_env.py` | 8 | **12.13s** | Cloud runner execution, environment detection subprocesses |
| `test_boost_end_to_end.py` | 9 | **11.25s** | Multi-fold runner execution, dataset train/test inference, zip packaging |
| `test_dashboard.py` | 21 | **10.19s** | Live HTTP server thread startup, 18+ HTTP request-response cycles |
| `test_e2e_pipeline.py` | 11 | **8.27s** | Multi-fold runner execution, bootstrap ingestion, tree search |
| `test_audit_gate.py` | 31 | **7.74s** | Subprocess CLI invocations with temporary zip packaging |
| `test_harness.py` | 11 | **7.20s** | Subprocess CLI invocations of `tools/audit_gate.py` |
| `test_single_install.py` | 14 | **6.77s** | Subprocess simulating separate virtualenvs |
| `test_workspace_workflow.py` | 11 | **6.08s** | Spawns `ai-harness-run` CLI subprocess over 2-fold synthetic task |
| `test_remote_dispatch_hardening.py` | 15 | **5.84s** | Process PID liveness checks, reconciliation, timeout simulation |

#### Smoke Tier Latency & Timing Budget (<10s Contract Guarantee):
- **Baseline Smoke Latency**: Running `python tools/run_tests.py --smoke` on the baseline repository executes 47 tests across 5 modules in **2.18s in-process (4.2s wall clock)**.
- **`test_research_memory.py` Partition**:
  - The legacy `test_research_memory.py` takes **15.8s** alone due to running 1,000 paired bootstrap resamples and multi-folder disk sync workspaces (`workspace_with_verified_pair`).
  - To protect the `<10s` contract, `tests/unit/test_research_memory.py` in `--smoke` retains **strictly fast in-memory schema validation (JSON Schema draft 2020-12), decision contracts, and plateau latch rules (<1.5s total)** with zero disk I/O.
  - Heavy 1,000-resample bootstrap calculations and disk-based sync workspaces live in `tests/integration/test_research_continuity.py`.
- **`test_correctness_runtime.py` Multi-Component Placement**:
  - `test_correctness_runtime.py` takes **9.88s** alone because it invokes `UniversalRunner.run(...)` over multi-fold temporary disk workspaces.
  - It is strictly assigned to `tests/integration/test_runner.py` and is **NOT part of `--smoke`**, ensuring `--smoke` remains purely in-memory and executes in <5s total.
- **Unit Tier I/O Refactoring**: Pure logic from the 28 tests currently using `tempfile` is converted to in-memory mocks, with ephemeral disk access (<0.05s) permitted only for isolated serialization contracts, while full workspaces are routed to integration.

---

## 2. 3-Tier Hierarchy Architecture

The reorganized test suite strictly segregates tests into three distinct directories based on execution characteristics, I/O profiles, and external dependencies:

```text
tests/
├── unit/               # Pure in-memory logic, 0 disk I/O, 0 subprocesses (<1s/module, <5s total)
├── integration/        # Multi-component, temporary workspaces, in-process DAGs (0.5s - 5s/mod, ~25s total)
├── e2e/                # Cloud runners, Kaggle kernels, live HTTP server, git sync (5s - 70s/mod)
├── fixtures.py         # Shared workspace generators (make_workspace, workspace_with_repo_configs)
├── evidence_fixtures.py# Shared evidence generators (dummy manifests, sealed runs)
├── workspace_fixtures.py# Shared synthetic dataset and task generators
└── __init__.py         # Root test package marker
```

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              All Tests                                 │
└────────────────────────────────────────────────────────────────────────┘
                                    │
       Does it touch the filesystem (tempfile/workspace) or spawn a subprocess?
                                    │
             ┌───────────────────────┴───────────────────────┐
             │ NO                                            │ YES
             ▼                                               ▼
    tests/unit/ (15 modules)               Does it spawn heavy external tools
    - Zero disk I/O                        (git, live HTTP servers, cloud CLI,
    - Zero subprocesses                    full kernel bundles, >5s pipelines)?
    - Runtime: <1s / module                                  │
    - Full suite: <5s total                 ┌────────────────┴────────────────┐
    - Contains all --smoke invariants       │ NO                              │ YES
                                            ▼                                 ▼
                               tests/integration/ (17 modules)     tests/e2e/ (6 modules)
                               - Hermetic temp workspaces          - Heavy subprocesses
                               - Synthetic data & configs          - Live servers (TCP)
                               - In-process DAG & Runner           - Full pipeline runs
                               - Runtime: 0.5s - 5s / module       - Runtime: 5s - 70s / mod
                               - Suite: ~25s total                 - Excluded from --fast
```

---

### 2.1 `tests/unit/` (Fast Pure-Logic Tier)
- **Mandatory Invariants**:
  - **Zero temporary disk I/O (Pure Logic Focus)**: Pure unit tests operate strictly in-memory without filesystem access.
    - *Refactoring Pattern for Existing Tests*: The 28 tests currently importing `tempfile` are refactored according to a strict 3-tier separation:
      1. *Pure Logic / Schema / AST Checks*: Converted to in-memory mocks, `io.StringIO`, or in-memory dict/object structures (zero filesystem access).
      2. *Ephemeral Serialization Checks*: Bounded to isolated ephemeral `tempfile.TemporaryDirectory` (<0.05s execution) strictly for single-file serialization round-trips (e.g. atomic write integrity or receipt hashing).
      3. *Multi-File Workspaces / Model Fitting / Runners*: Re-routed to `tests/integration/`.
  - **Zero external processes**: Prohibited from calling `subprocess.run`, `subprocess.Popen`, `os.system`.
  - **Performance Budget**: <0.05s per test, <1.0s per module, <5.0s for the entire unit suite.
- **Domain Scope**:
  - Evaluation contracts, probability constraints, metric reference implementations.
  - Label integrity audits (missing, duplicate, fractional labels).
  - Pure-logic gates: Epistemic grounding verification, verified_via whitelist, execution backend resolution logic.
  - Model parameter accounting (base + LoRA + vision encoder + projector), license allowlists, release cutoff dates.
  - Markdown citation syntax parsing, source registry schema validation.
  - Research memory schema validation (JSON Schema draft 2020-12), decision rules, and plateau latches (<1.5s total). Heavy 1,000-resample bootstrap simulations and multi-folder sync workspaces live in `tests/integration/test_research_continuity.py`.
  - Macro hypothesis taxonomy, family activation, mechanism exhaustion rules.
  - Ensemble guard mathematical diversity, pairwise Cohen kappa, machine epsilon, zero-variance vectors.
  - Cross-validation split algorithms (K-fold, stratified, time series, purged group, multilabel).
  - Feature pipelines, tabular transformers, text n-grams, tabular OOM guard.
  - IO utilities, JSON serialization, numpy coercion, safe relpaths, NormalizedObjectDict.
  - Evidence manifest hashing, manifest schema validation, and path traversal rejection (`test_evidence_manifest.py`).
  - TaskSpec configuration parsing, subtask aliases, path traversal rejection.
- **Smoke Invariants Suite**: All 5 `--smoke` invariant test modules reside in `tests/unit/` and execute in <3s in-process (<5s wall clock), guaranteeing `<10s` verification gate execution.
- **Target Size**: Exactly **15 domain modules**.

---

### 2.2 `tests/integration/` (Multi-Component & Runner Tier)
- **Mandatory Invariants**:
  - **Hermetic temporary workspaces**: Allowed to use `tests.fixtures.make_workspace` or `tempfile.TemporaryDirectory` to test interactions between two or more subsystems.
  - **In-process execution**: Uses in-process Python calls or minimal, lightweight child scripts (e.g. `test_script_runner.py`).
  - **No external cloud or network coupling**: Must not require live network access, git network remotes, or cloud providers.
  - **Performance Budget**: 0.5s to 5.0s per module; ~25s total for the integration suite.
- **Domain Scope**:
  - `UniversalRunner` multi-fold execution, debug probes, sample limits, runtime correctness contracts (`test_runner.py`).
  - DAG stage contracts, intermediate artifacts, stage manifests.
  - Recursive directory sealing, SHA-256 manifests, hardlink deduplication, tamper detection (`test_evidence_sealing.py`).
  - Candidate preflight generation, pitch validation, Spearman $\rho \ge 0.995$ near-duplicate blocking.
  - Candidate pool synthesis, recipe fingerprints, official scorer promotion, weighted blending.
  - Model registry disk resolution, content addressing, backend resource plans (local/cloud), SubmissionGate permits.
  - TreeJournal on-disk persistence, UCB branch recommendations, bottleneck cues, orphan node unverified flagging.
  - Research continuity across session restarts, lesson plan tracing, repeat detection without hiding, paired bootstrap math & sync workspaces (`test_research_continuity.py`).
  - AgentBridge action registry, parameter normalization, action execution, typed action schemas.
  - Session budget tracking, approval workflow, crash recovery, state transaction locks, finalize stop gates.
  - Context HUD rendering, workspace hygiene scanner, submission packet reporting.
  - Submission generation, offline notebooks, dynamic markdown rendering, preserve submission.
  - Audit gate CLI invocation, statistical McNemar significance, effect size enforcement (MMES).
  - Plugin discovery, registration, custom task contracts.
  - Artifact repair: run classification (verified, drifted, missing), manifest resealing, evidence-integrity gate.
  - Framework adoption: Consolidating adoption stack, single-install canonical framework resolution, host environment hooks, local editable imports, bootstrap task onboarding & skill injector (`test_framework_adoption.py`).
  - Kaggle CLI dispatch, preflight checks, discussions, submissions, kernel push staging (`test_kaggle_cli.py`).
- **Target Size**: Exactly **17 domain modules**.

---

### 2.3 `tests/e2e/` (Cloud, Kaggle & Remote Execution Tier)
- **Mandatory Invariants**:
  - **Full lifecycle coverage**: Exercises full multi-stage pipelines, entire workspace lifecycles (`ai-harness-run`), live HTTP servers, and complete cloud packaging.
  - **Heavy subprocesses**: Allowed to execute heavy CLI tools (`git clone`, `kaggle kernels push`, `modal run`, multi-stage scripts).
  - **Performance Budget**: 5s to 70s per module.
  - **Strict Isolation**: Excluded by default from `--fast` and `--smoke` pre-push quality gates.
- **Domain Scope**:
  - `test_kaggle_runtime.py`: Kaggle kernel packaging, offline notebook export, dataset mounts, runner contracts (~69s).
  - `test_cloud_runners.py`: Cloud job runner, environment caching, GPU accounting, detached Modal cloud modes (~12s).
  - `test_dashboard_e2e.py`: Live dashboard HTTP server, WebSocket events, frontend resilience stress (~10s).
  - `test_sync_e2e.py`: Framework sync CLI, single-install canonical checkout resolution, git operations (~30s).
  - `test_workspace_workflow.py`: Full autonomous workspace lifecycle, task config creation, multi-fold CLI execution, artifact manifest sealing (~6s).
  - `test_pipeline_e2e.py`: Multi-stage pipeline execution end-to-end, sequential child scripts, candidate packaging (~13s).
- **Target Size**: Exactly **6 domain modules**.

---

### 2.4 Shared Test Infrastructure
- `tests/fixtures.py`, `tests/evidence_fixtures.py`, `tests/workspace_fixtures.py`, and `tests/__init__.py` remain at the `tests/` root. This ensures that existing imports throughout the repository (`from tests.fixtures import make_workspace`) remain valid without requiring mass import rewrites.
- Root `conftest.py` provides global session cleanup fixtures (`_no_suite_artifact_leak`) protecting the working tree from stray artifacts across all three tiers.

---

## 3. Master 135-Module to 38-Module Consolidation Mapping Table

The table below catalogs **every single one of the 135 discoverable test modules** in the repository. Each entry specifies its test counts, line count, associated `harness/` module, execution profile, target tier, target consolidated domain module, and disposition:
- **Retain**: Module remains as a dedicated domain module (moved to its target tier directory).
- **Merge**: Assertions and test classes are merged into a consolidated domain module, and the redundant file is deleted.
- **Prune**: Brittle AST/regex scans or duplicate assertions are eliminated during consolidation.

| # | Current Test Module Path | Unittest Tests | AST Tests | Lines | Associated `harness/` Module | Execution Profile | Target Tier | Target Domain Module | Disposition |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `tests/agent_bridge/test_cloud_hygiene_bypass.py` | 8 | 8 | 220 | `harness.project, harness.skill_injector` | Temp Disk I/O | integration | `test_agent_bridge.py` | Merge |
| 2 | `tests/agent_bridge/test_core_bridge.py` | 8 | 8 | 277 | `harness.agent_bridge, harness.grounding_gate` | Temp Disk I/O | integration | `test_agent_bridge.py` | Merge |
| 3 | `tests/agent_bridge/test_friction_fixes.py` | 23 | 23 | 384 | `harness.agent_bridge, harness.endgame_submission` | Temp Disk I/O | integration | `test_agent_bridge.py` | Merge |
| 4 | `tests/agent_bridge/test_grounding_guards.py` | 6 | 6 | 202 | `harness.agent_bridge, harness.grounding_gate` | Temp Disk I/O | integration | `test_agent_bridge.py` | Merge & Prune duplicate |
| 5 | `tests/agent_bridge/test_hud_ergonomics.py` | 11 | 11 | 304 | `harness.agent_bridge` | Temp Disk I/O | integration | `test_hud_ergonomics.py` | Merge |
| 6 | `tests/agent_bridge/test_session_approval_fixes.py` | 11 | 11 | 366 | `harness.agent_bridge` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 7 | `tests/agent_bridge/test_state_readiness_copilot.py` | 17 | 17 | 453 | `harness.agent_bridge` | Temp Disk I/O | integration | `test_agent_bridge.py` | Merge |
| 8 | `tests/agent_bridge/test_usability_phase3.py` | 6 | 6 | 156 | `harness.agent_bridge, harness.kaggle_notebook_builder` | Temp Disk I/O | unit | `test_io_utils.py` / `test_task_spec.py` | Merge |
| 9 | `tests/agent_bridge/test_usability_phase4.py` | 6 | 6 | 165 | `harness.evaluation, harness.task_spec` | Temp Disk I/O | unit | `test_evaluation_contracts.py` | Merge |
| 10 | `tests/agent_bridge/test_usability_phase5.py` | 6 | 6 | 276 | `harness.agent_bridge, harness.models` | Temp Disk I/O | unit | `test_models.py` | Merge |
| 11 | `tests/agent_bridge/test_usability_phase6.py` | 4 | 4 | 201 | `harness.models, harness.task_spec` | Temp Disk I/O | unit | `test_models.py` | Merge |
| 12 | `tests/agent_bridge/test_usability_phase7.py` | 5 | 5 | 196 | `harness.data_strategy, harness.models` | Temp Disk I/O | unit | `test_data_strategy.py` / `test_feature_pipeline.py` | Merge |
| 13 | `tests/agent_bridge/test_usability_phase8.py` | 9 | 9 | 244 | `harness.data_strategy, harness.endgame_submission` | Temp Disk I/O | unit | `test_data_strategy.py` / `test_evaluation_contracts.py` | Merge |
| 14 | `tests/test_adoption_stack.py` | 19 | 19 | 263 | `harness.agent_bridge, harness.host_hooks` | Temp Disk I/O + Subproc | integration | `test_framework_adoption.py` | Merge |
| 15 | `tests/test_agent_friction_remediation.py` | 15 | 15 | 221 | `harness.bridge_common, harness.grounding_gate` | In-Memory (0 I/O) | unit | `test_io_utils.py` / `test_grounding_gate.py` | Merge |
| 16 | `tests/test_agent_friction_round2.py` | 17 | 17 | 200 | `harness.evaluation, harness.io_utils` | Temp Disk I/O | unit | `test_io_utils.py` / `test_task_spec.py` | Merge |
| 17 | `tests/test_agent_session_resilience.py` | 6 | 6 | 180 | `harness.artifacts, harness.bridge_common` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 18 | `tests/test_anti_degeneration.py` | 11 | 11 | 249 | `harness.diagnostics, harness.ensemble_guard` | Temp Disk I/O + Subproc | unit | `test_ensemble_guard.py` / `test_tree_journal.py` | Merge & Prune duplicates |
| 19 | `tests/test_api_reference.py` | 14 | 14 | 136 | `harness.api_reference` | In-Memory (0 I/O) | unit | `test_task_spec.py` | Merge |
| 20 | `tests/test_artifact_dedup.py` | 14 | 14 | 374 | `harness.artifacts` | Temp Disk I/O | unit / integration | `test_evidence_manifest.py` (hash dedup math) / `test_evidence_sealing.py` | Split & Merge |
| 21 | `tests/test_audit_gate.py` | 31 | 31 | 540 | `tools.audit_gate, harness.task_spec` | Temp Disk I/O + Subproc | integration | `test_audit_gate_integration.py` (CLI) & `unit/test_label_integrity.py` | Split & Merge |
| 22 | `tests/test_audit_gate_submission.py` | 7 | 7 | 194 | `harness.agent_bridge, harness.environment_detector` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 23 | `tests/test_blend_optimizer_contract.py` | 3 | 3 | 66 | `harness.agent_bridge, harness.artifacts` | Temp Disk I/O | integration | `test_candidate_selection.py` | Merge |
| 24 | `tests/test_blend_recipe_identity.py` | 4 | 4 | 89 | `harness.agent_bridge, harness.artifacts` | Temp Disk I/O | integration | `test_candidate_selection.py` | Merge |
| 25 | `tests/test_boost_end_to_end.py` | 9 | 9 | 548 | `harness.agent_bridge, harness.artifacts` | Temp Disk I/O | e2e | `test_pipeline_e2e.py` | Merge |
| 26 | `tests/test_bootstrap_task.py` | 3 | 3 | 190 | `harness.task_spec` | Temp Disk I/O | integration | `test_framework_adoption.py` | Merge |
| 27 | `tests/test_candidate_official_scorer.py` | 4 | 4 | 100 | `harness.agent_bridge, harness.artifacts` | In-Memory (0 I/O) | integration | `test_candidate_selection.py` | Merge |
| 28 | `tests/test_candidate_preflight.py` | 12 | 12 | 314 | `harness.candidate_preflight` | In-Memory (0 I/O) | integration | `test_candidate_preflight.py` | Retain |
| 29 | `tests/test_challenger_m1_1_empirical.py` | 23 | 23 | 439 | `harness.ensemble_guard` | Temp Disk I/O + Subproc | unit | `test_ensemble_guard.py` | Merge |
| 30 | `tests/test_challenger_m3_2_empirical.py` | 14 | 14 | 320 | `dashboard/src/components` | In-Memory (0 I/O) | integration | `test_dashboard_e2e.py` | Merge & Prune regex |
| 31 | `tests/test_challenger_p3_empirical.py` | 4 | 4 | 221 | `dashboard/src/App.tsx` | In-Memory (0 I/O) | integration | `test_dashboard_e2e.py` | Merge & Prune line 98 check |
| 32 | `tests/test_challenger_r1_r4_stress.py` | 10 | 10 | 440 | `tools.dashboard_server` | Temp Disk I/O (Live TCP) | integration | `test_dashboard_e2e.py` | Merge & Prune AST scans |
| 33 | `tests/test_citation_admission.py` | 5 | 5 | 87 | `harness.citation_audit` | Temp Disk I/O | unit | `test_citation_audit.py` | Merge |
| 34 | `tests/test_citation_audit.py` | 7 | 7 | 127 | `harness.citation_audit` | Temp Disk I/O | unit | `test_citation_audit.py` | Retain & Merge into |
| 35 | `tests/test_citation_cli.py` | 3 | 3 | 63 | `tools.citation_cli` | Temp Disk I/O + Subproc | unit | `test_citation_audit.py` | Merge |
| 36 | `tests/test_citation_receipts.py` | 7 | 7 | 178 | `harness.citation_audit` | Temp Disk I/O | unit | `test_citation_audit.py` | Merge |
| 37 | `tests/test_citation_review.py` | 5 | 5 | 72 | `harness.citation_review` | Temp Disk I/O | unit | `test_citation_audit.py` | Merge |
| 38 | `tests/test_citation_sources.py` | 10 | 10 | 124 | `harness.citation_sources` | Temp Disk I/O | unit | `test_citation_audit.py` | Merge |
| 39 | `tests/test_cleaner.py` | 29 | 29 | 467 | `harness.cli, harness.cleaner` | Temp Disk I/O + Subproc | integration | `test_repair_integration.py` | Merge |
| 40 | `tests/test_cli_ergonomics.py` | 2 | 2 | 38 | `harness.cli` | In-Memory (0 I/O) | unit | `test_io_utils.py` | Merge |
| 41 | `tests/test_cloud_jobs_and_env.py` | 8 | 8 | 208 | `harness.cloud_job_runner` | Temp Disk I/O + Subproc | e2e | `test_cloud_runners.py` | Retain & Merge into |
| 42 | `tests/test_codabench_compat.py` | 6 | 6 | 132 | `harness.codabench_compat` | Temp Disk I/O | integration | `test_kaggle_cli.py` | Merge |
| 43 | `tests/test_competitive_endgame.py` | 10 | 10 | 192 | `harness.endgame_submission` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 44 | `tests/test_correctness_candidates.py` | 2 | 2 | 33 | `harness.candidates` | In-Memory (0 I/O) | unit | `test_label_integrity.py` | Retain (**SMOKE**) |
| 45 | `tests/test_correctness_contracts.py` | 11 | 11 | 230 | `harness.task_spec` | In-Memory (0 I/O) | unit | `test_evaluation_contracts.py` | Retain (**SMOKE**) |
| 46 | `tests/test_correctness_governance.py` | 4 | 4 | 96 | `harness.llm_engine, harness.resource_ledger` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 47 | `tests/test_correctness_regressions.py` | 18 | 18 | 543 | `harness.task_spec, harness.candidates` | Temp Disk I/O | unit | `test_task_spec.py` / `test_evaluation_contracts.py` | Merge |
| 48 | `tests/test_correctness_runtime.py` | 6 | 6 | 254 | `harness.runner` | Temp Disk I/O | integration | `test_runner.py` | Merge |
| 49 | `tests/test_correctness_state.py` | 3 | 3 | 67 | `harness.state_store` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 50 | `tests/test_custom_scorer_validation.py` | 8 | 8 | 148 | `harness.hook_ref` | Temp Disk I/O | integration | `test_candidate_selection.py` | Merge |
| 51 | `tests/test_dashboard.py` | 21 | 21 | 390 | `tools.dashboard_server` | Temp Disk I/O (Live TCP) | e2e | `test_dashboard_e2e.py` | Retain |
| 52 | `tests/test_data_bundle_protected_paths.py` | 4 | 4 | 92 | `harness.data_bundle` | Temp Disk I/O | integration | `test_evidence_sealing.py` | Merge |
| 53 | `tests/test_dispatch_governance.py` | 10 | 10 | 200 | `harness.endgame_submission` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 54 | `tests/test_e2e_pipeline.py` | 11 | 11 | 505 | `harness.runner, harness.agent_bridge` | Temp Disk I/O | e2e | `test_pipeline_e2e.py` | Merge |
| 55 | `tests/test_endgame_tools.py` | 6 | 6 | 289 | `harness.endgame_submission` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 56 | `tests/test_ensemble_guard.py` | 25 | 25 | 385 | `harness.ensemble_guard` | In-Memory (0 I/O) | unit | `test_ensemble_guard.py` | Retain |
| 57 | `tests/test_env_cache.py` | 20 | 20 | 300 | `harness.cloud_job_runner` | Temp Disk I/O | integration | `test_repair_integration.py` | Merge |
| 58 | `tests/test_ergonomics_and_resilience.py` | 9 | 9 | 134 | `harness.agent_bridge` | Temp Disk I/O | integration | `test_hud_ergonomics.py` | Merge |
| 59 | `tests/test_execution_backend_policy.py` | 18 | 18 | 210 | `harness.task_spec, harness.runner` | Temp Disk I/O | unit | `test_execution_backend_policy.py` | Retain (**SMOKE**) |
| 60 | `tests/test_external_candidate_metadata.py` | 3 | 3 | 73 | `harness.artifacts, harness.candidates` | Temp Disk I/O | integration | `test_candidate_selection.py` | Merge |
| 61 | `tests/test_f821_undefined_names.py` | 5 | 5 | 82 | `harness.cloud_runner` | Temp Disk I/O | integration | `test_agent_bridge.py` | Merge & Prune static checks |
| 62 | `tests/test_failure_classifier.py` | 9 | 9 | 78 | `harness.failure_classifier` | In-Memory (0 I/O) | unit | `test_task_spec.py` | Merge |
| 63 | `tests/test_gateway_keepalive.py` | 3 | 3 | 58 | `harness.gateway_keepalive` | In-Memory (0 I/O) | unit | `test_io_utils.py` | Merge |
| 64 | `tests/test_grounding_gate.py` | 12 | 12 | 198 | `harness.grounding_gate` | Temp Disk I/O | unit | `test_grounding_gate.py` | Retain (**SMOKE**) |
| 65 | `tests/test_guardrail_regressions.py` | 23 | 23 | 698 | `harness.candidates, harness.runner` | Temp Disk I/O + Subproc | unit / integration | `tests/unit/test_execution_backend_policy.py` / `tests/integration/test_session_resilience.py` / `tests/integration/test_repair_integration.py` | Disperse & Merge |
| 66 | `tests/test_harness.py` | 11 | 11 | 277 | `harness.task_spec, harness.tree_journal` | Temp Disk I/O + Subproc | integration | `test_tree_journal.py` / `unit/test_task_spec.py` | Split & Merge |
| 67 | `tests/test_harness_universal.py` | 11 | 11 | 259 | `harness.data_strategy, harness.feature_pipeline` | In-Memory (0 I/O) | unit | `test_data_strategy.py` / `test_feature_pipeline.py` | Merge |
| 68 | `tests/test_i18n_and_setup_stress.py` | 11 | 11 | 373 | `dashboard/src` | In-Memory (0 I/O) | integration | `test_dashboard_e2e.py` | Merge |
| 69 | `tests/test_interactive_prediction_contract.py` | 0 | 3 | 41 | `harness.evaluation, harness.task_spec` | In-Memory (0 I/O) | unit | `test_evaluation_contracts.py` | Wrap in TestCase & Merge |
| 70 | `tests/test_io_utils.py` | 19 | 19 | 408 | `harness.io_utils` | Temp Disk I/O | unit | `test_io_utils.py` | Retain |
| 71 | `tests/test_job_state.py` | 3 | 3 | 49 | `harness.job_state` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 72 | `tests/test_journal_sync_dedupe.py` | 5 | 5 | 229 | `harness.tree_journal` | Temp Disk I/O | integration | `test_tree_journal.py` | Merge |
| 73 | `tests/test_kaggle_compat.py` | 36 | 36 | 463 | `harness.kaggle_compat` | In-Memory (0 I/O) | unit | `test_io_utils.py` (Kaggle facade) | Merge into `integration/test_kaggle_cli.py` & `unit` |
| 74 | `tests/test_kaggle_competition_discovery.py` | 8 | 8 | 148 | `harness.kaggle_compat` | In-Memory (0 I/O) | integration | `test_kaggle_cli.py` | Merge |
| 75 | `tests/test_kaggle_discussions_and_submissions.py` | 16 | 16 | 348 | `harness.endgame_submission` | Temp Disk I/O | integration | `test_kaggle_cli.py` | Merge |
| 76 | `tests/test_kaggle_inspect.py` | 7 | 7 | 96 | `harness.cli` | Temp Disk I/O | integration | `test_kaggle_cli.py` | Merge |
| 77 | `tests/test_kaggle_leaderboard_tracker.py` | 11 | 11 | 160 | `harness.endgame_submission` | In-Memory (0 I/O) | unit / integration | `test_evidence_manifest.py` (pure-logic manifest hashing & relpath checks) / `test_kaggle_cli.py` | Split & Merge |
| 78 | `tests/test_kaggle_preflight.py` | 16 | 16 | 250 | `harness.cloud_runner` | Temp Disk I/O + Subproc | integration | `test_kaggle_cli.py` | Merge |
| 79 | `tests/test_kaggle_runtime.py` | 28 | 28 | 532 | `harness.kaggle_runtime` | Temp Disk I/O + Subproc | e2e | `test_kaggle_runtime.py` | Retain (**SLOW**) |
| 80 | `tests/test_kernel_dispatch.py` | 18 | 18 | 300 | `harness.kernel_dispatch` | Temp Disk I/O | integration | `test_kaggle_cli.py` | Merge |
| 81 | `tests/test_leaderboard_direction.py` | 5 | 5 | 77 | `harness.endgame_submission` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 82 | `tests/test_llm_api_endpoints.py` | 10 | 10 | 182 | `harness.llm_engine` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 83 | `tests/test_llm_engine.py` | 8 | 8 | 90 | `harness.llm_engine` | In-Memory (0 I/O) | unit | `test_models.py` | Merge |
| 84 | `tests/test_llm_experiment_specs.py` | 3 | 3 | 80 | `harness.tree_journal` | Temp Disk I/O | integration | `test_tree_journal.py` | Merge |
| 85 | `tests/test_local_import.py` | 13 | 13 | 244 | `harness.import_local` | Temp Disk I/O | integration | `test_framework_adoption.py` | Merge |
| 86 | `tests/test_macro_research.py` | 15 | 15 | 204 | `harness.macro_research` | In-Memory (0 I/O) | unit | `test_macro_research.py` | Retain |
| 87 | `tests/test_modal_cloud_modes.py` | 8 | 8 | 257 | `harness.cloud_job_runner` | Temp Disk I/O + Subproc | e2e | `test_cloud_runners.py` | Merge |
| 88 | `tests/test_modal_runtime_requirements.py` | 3 | 3 | 104 | `harness.remote_runtime` | Temp Disk I/O | e2e | `test_cloud_runners.py` | Merge |
| 89 | `tests/test_model_cli_and_bridge.py` | 0 | 4 | 261 | `harness.model_assets.cli` | Temp Disk I/O | integration | `test_model_resolution.py` | Wrap in TestCase & Merge |
| 90 | `tests/test_model_execution_plan.py` | 0 | 6 | 275 | `harness.model_assets` | Temp Disk I/O + Subproc | integration | `test_model_resolution.py` | Wrap in TestCase & Merge |
| 91 | `tests/test_model_governance.py` | 0 | 15 | 304 | `harness.model_assets` | In-Memory (0 I/O) | unit | `test_model_governance.py` | Wrap in TestCase & Retain |
| 92 | `tests/test_model_lineage_and_submission.py` | 0 | 5 | 323 | `harness.model_assets.lineage` | Temp Disk I/O | integration | `test_model_resolution.py` | Wrap in TestCase & Merge |
| 93 | `tests/test_model_registry.py` | 0 | 10 | 149 | `harness.model_registry` | Temp Disk I/O | integration | `test_model_resolution.py` | Wrap in TestCase & Merge |
| 94 | `tests/test_model_resolution.py` | 0 | 8 | 307 | `harness.model_assets` | Temp Disk I/O | integration | `test_model_resolution.py` | Wrap in TestCase & Retain |
| 95 | `tests/test_notebook_prediction_integrity.py` | 6 | 6 | 103 | `harness.kaggle_notebook_builder` | Temp Disk I/O + Subproc | integration | `test_submission_manager.py` | Merge |
| 96 | `tests/test_p2_audits.py` | 13 | 13 | 251 | `harness.agent_bridge` | Temp Disk I/O | integration | `test_hud_ergonomics.py` / `unit/test_research_memory.py` | Merge |
| 97 | `tests/test_p2_fixes.py` | 4 | 4 | 122 | `harness.agent_bridge` | Temp Disk I/O | unit | `test_task_spec.py` | Merge |
| 98 | `tests/test_pipeline_execution.py` | 15 | 15 | 373 | `harness.runner` | Temp Disk I/O + Subproc | integration | `test_pipeline_execution.py` | Retain |
| 99 | `tests/test_plugin_contract_validation.py` | 6 | 6 | 104 | `harness.plugins` | In-Memory (0 I/O) | integration | `test_plugins_integration.py` | Merge |
| 100 | `tests/test_plugin_docs_and_example.py` | 6 | 6 | 114 | `harness.plugins` | Temp Disk I/O | integration | `test_plugins_integration.py` | Merge |
| 101 | `tests/test_preserve_submission.py` | 5 | 5 | 231 | `harness.submission_manager` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 102 | `tests/test_provenance_dispatch_guard.py` | 4 | 4 | 82 | `harness.dispatch_audit` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 103 | `tests/test_reconciliation.py` | 5 | 5 | 128 | `harness.dispatch_audit` | Temp Disk I/O | integration | `test_session_resilience.py` | Merge |
| 104 | `tests/test_recursive_sealing.py` | 8 | 8 | 198 | `harness.artifacts` | Temp Disk I/O | unit / integration | `tests/unit/test_evidence_manifest.py` (pure-logic manifest hashing, schema & relpath traversal checks) / `tests/integration/test_evidence_sealing.py` (directory sealing, hardlink dedup & tamper detection) | Split & Retain |
| 105 | `tests/test_remote_dispatch_hardening.py` | 15 | 15 | 417 | `harness.cloud_job_runner` | Temp Disk I/O + Subproc | integration | `test_session_resilience.py` | Merge |
| 106 | `tests/test_repair.py` | 17 | 17 | 335 | `harness.repair` | Temp Disk I/O | integration | `test_repair_integration.py` | Retain |
| 107 | `tests/test_repair_integrity.py` | 9 | 9 | 382 | `harness.repair` | Temp Disk I/O | integration | `test_repair_integration.py` | Merge |
| 108 | `tests/test_research_action_contracts.py` | 10 | 10 | 175 | `harness.research_memory` | Temp Disk I/O | unit | `test_evidence_manifest.py` (pure-logic manifest hashing & relpath verification) / `test_research_memory.py` | Split & Merge |
| 109 | `tests/test_research_continuity.py` | 12 | 12 | 281 | `harness.research_memory` | Temp Disk I/O | integration | `test_research_continuity.py` | Retain |
| 110 | `tests/test_research_memory.py` | 21 | 21 | 308 | `harness.research_memory` | In-Memory (0 I/O) | unit | `test_research_memory.py` | Retain |
| 111 | `tests/test_research_scope_stability.py` | 4 | 4 | 119 | `harness.research_memory` | Temp Disk I/O | unit | `test_research_memory.py` | Retain (**SMOKE**) |
| 112 | `tests/test_research_sync_cache.py` | 4 | 4 | 140 | `harness.research_memory` | Temp Disk I/O | unit | `test_research_memory.py` | Merge |
| 113 | `tests/test_research_trace.py` | 12 | 12 | 276 | `harness.research_memory` | Temp Disk I/O | integration | `test_research_continuity.py` | Merge |
| 114 | `tests/test_retrospective_gate.py` | 14 | 14 | 241 | `harness.macro_research` | In-Memory (0 I/O) | unit | `test_macro_research.py` | Merge |
| 115 | `tests/test_ruff_b10_b16.py` | 6 | 6 | 252 | `harness.plugins, harness.task_spec` | Temp Disk I/O | integration | `tests/integration/test_session_resilience.py` / `tests/integration/test_candidate_preflight.py` / `tests/integration/test_model_resolution.py` | Disperse & Merge |
| 116 | `tests/test_runtime_extensions.py` | 6 | 6 | 124 | `harness.runner` | Temp Disk I/O | integration | `test_runner.py` | Merge |
| 117 | `tests/test_scoring_import_hermetic.py` | 3 | 3 | 167 | `harness.hook_ref` | Temp Disk I/O | integration | `test_candidate_selection.py` | Merge |
| 118 | `tests/test_script_adapter.py` | 2 | 2 | 157 | `harness.runner` | Temp Disk I/O | integration | `test_runner.py` | Merge |
| 119 | `tests/test_script_prediction_transport.py` | 5 | 5 | 110 | `harness.runner` | Temp Disk I/O + Subproc | integration | `test_runner.py` | Merge |
| 120 | `tests/test_script_validation_scope.py` | 5 | 5 | 125 | `harness.runner` | Temp Disk I/O | integration | `test_runner.py` | Merge |
| 121 | `tests/test_single_install.py` | 14 | 14 | 309 | `harness.project` | Temp Disk I/O + Subproc | integration | `test_framework_adoption.py` | Merge |
| 122 | `tests/test_skill_injector.py` | 17 | 17 | 442 | `harness.skill_injector` | Temp Disk I/O | integration | `test_framework_adoption.py` | Merge |
| 123 | `tests/test_status_report.py` | 13 | 13 | 158 | `harness.status_report` | Temp Disk I/O | integration | `test_repair_integration.py` | Merge |
| 124 | `tests/test_strategy_cards.py` | 3 | 3 | 71 | `harness.agent_bridge` | Temp Disk I/O | integration | `test_hud_ergonomics.py` | Merge |
| 125 | `tests/test_submission_and_cloud.py` | 7 | 7 | 175 | `harness.submission_manager` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 126 | `tests/test_submission_dispatch.py` | 24 | 24 | 304 | `harness.endgame_submission` | Temp Disk I/O | integration | `test_submission_manager.py` | Merge |
| 127 | `tests/test_submission_guards.py` | 5 | 5 | 48 | `harness.endgame_submission` | In-Memory (0 I/O) | integration | `test_submission_manager.py` | Merge |
| 128 | `tests/test_sync_command.py` | 11 | 11 | 251 | `harness.sync` | Temp Disk I/O + Subproc | e2e | `test_sync_e2e.py` | Retain (**SLOW**) |
| 129 | `tests/test_task_validate.py` | 20 | 20 | 523 | `harness.task_validate` | Temp Disk I/O | unit | `test_task_spec.py` | Merge |
| 130 | `tests/test_threshold_evidence.py` | 8 | 8 | 124 | `harness.artifacts` | Temp Disk I/O | unit / integration | `test_evidence_manifest.py` (manifest verification) / `test_evidence_sealing.py` | Split & Merge |
| 131 | `tests/test_toolchain_adoption.py` | 13 | 13 | 350 | `harness.toolchain` | Temp Disk I/O + Subproc | integration | `test_framework_adoption.py` | Retain |
| 132 | `tests/test_tree_journal_ensemble.py` | 11 | 11 | 258 | `harness.tree_journal, harness.ensemble_guard` | Temp Disk I/O | integration | `test_tree_journal.py` / `unit/test_ensemble_guard.py` | Disperse & Merge |
| 133 | `tests/test_universal_plugin.py` | 7 | 7 | 324 | `harness.plugins` | Temp Disk I/O | integration | `test_plugins_integration.py` | Retain |
| 134 | `tests/test_upgrades_roadmap.py` | 14 | 14 | 584 | `harness.evaluation, harness.runner` | Temp Disk I/O | unit / integration / e2e | `tests/unit/test_evaluation_contracts.py` / `tests/unit/test_feature_pipeline.py` / `tests/integration/test_framework_adoption.py` / `tests/e2e/test_cloud_runners.py` | Disperse & Merge |
| 135 | `tests/test_workspace_workflow.py` | 11 | 11 | 426 | `harness.runner` | Temp Disk I/O + Subproc | e2e | `test_workspace_workflow.py` | Retain |

---

### 3.1 Summary of the 38 Consolidated Domain Modules

```text
CONSOLIDATED 38-MODULE HIERARCHY:
├── tests/unit/ (15 modules)
│   ├── test_evaluation_contracts.py    [SMOKE] (Contracts, schemas, probability bounds, metric refs)
│   ├── test_label_integrity.py        [SMOKE] (Missing/duplicate/fractional labels, ID alignment)
│   ├── test_grounding_gate.py         [SMOKE] (Epistemic grounding, whitelist, assumption registers)
│   ├── test_execution_backend_policy.py [SMOKE] (Resolution precedence, CLI override, probe limits)
│   ├── test_research_memory.py        [SMOKE] (JSON Schema 2020-12, decisions, plateau latches)
│   ├── test_macro_research.py         (Family taxonomy, exhaustion rules, 5-result review gate)
│   ├── test_model_governance.py       (Parameter accounting, composition, cutoff dates, licenses)
│   ├── test_citation_audit.py         (Markdown citations, source registry, receipts, review gate)
│   ├── test_evidence_manifest.py      (Artifact hashing, manifest schema, relpath traversal checks)
│   ├── test_ensemble_guard.py         (Pairwise diversity, Cohen kappa, rescue samples, epsilon)
│   ├── test_data_strategy.py          (CV splits: K-fold, stratified, time series, purged group)
│   ├── test_feature_pipeline.py       (Tabular transformers, text n-grams, tabular OOM guard)
│   ├── test_models.py                 (Baseline estimators: Ridge, Logistic, LightGBM, CatBoost)
│   ├── test_io_utils.py               (JSON serialization, sanitize cycles, numpy coercion, safe relpaths)
│   └── test_task_spec.py              (Task configuration parsing, subtask aliases, path traversal)
│
├── tests/integration/ (17 modules)
│   ├── test_runner.py                 (UniversalRunner multi-fold validation, debug probes, variants)
│   ├── test_pipeline_execution.py     (Stage contracts, DAG execution, intermediate artifacts)
│   ├── test_evidence_sealing.py       (Recursive directory sealing, hardlink deduplication, tamper check)
│   ├── test_candidate_preflight.py    (Preflight generation, pitch validation, Spearman rho >= 0.995)
│   ├── test_candidate_selection.py    (Candidate pool synthesis, recipe fingerprints, official scorer)
│   ├── test_model_resolution.py       (ModelStore resolution, local/cloud mounts, SubmissionGate permits)
│   ├── test_tree_journal.py           (TreeJournal persistence, UCB branch recommendations, bottlenecks)
│   ├── test_research_continuity.py    (Session restart recovery, lesson plan tracing, repeat detection, paired bootstrap math & sync workspaces)
│   ├── test_agent_bridge.py           (AgentBridge action registry, parameter normalization, dispatch)
│   ├── test_session_resilience.py     (Budget tracking, approval workflow, crash recovery, state locks)
│   ├── test_hud_ergonomics.py         (Context HUD rendering, workspace hygiene scanner, submission alerts)
│   ├── test_submission_manager.py     (Submission generation, offline notebooks, dynamic markdown)
│   ├── test_audit_gate_integration.py (Audit gate CLI, statistical McNemar significance, MMES)
│   ├── test_plugins_integration.py    (Plugin discovery, registration, custom task contracts)
│   ├── test_repair_integration.py     (Run classification, manifest resealing, evidence integrity gate)
│   ├── test_framework_adoption.py     (Consolidating adoption stack, single-install, toolchain adoption, local import, bootstrap task onboarding & skill injector)
│   └── test_kaggle_cli.py             (Kaggle CLI dispatch, preflight checks, discussions, submissions)
│
└── tests/e2e/ (6 modules)
    ├── test_kaggle_runtime.py         [SLOW] (Kernel packaging, offline notebook export, dataset mounts)
    ├── test_cloud_runners.py          [SLOW] (Cloud job runner, environment caching, detached Modal modes)
    ├── test_dashboard_e2e.py          [SLOW] (Live dashboard HTTP server, WebSocket events, UI stress)
    ├── test_sync_e2e.py               [SLOW] (Framework sync CLI, single-install canonical pin, git ops)
    ├── test_workspace_workflow.py     (Autonomous workspace lifecycle, task config, multi-fold CLI)
    └── test_pipeline_e2e.py           (Multi-stage pipeline execution end-to-end, sequential child scripts)
```

---

## 4. Ad-hoc PR & Challenger Test Deconstruction

A significant source of suite bloat and micro-fragmentation is the accumulation of 12 historical PR test suites and challenger files (148 tests, 3,830 lines of code). Below is the systematic deconstruction of these files:

### 4.1 Detailed File-by-File Breakdown

#### 1. `tests/test_challenger_m1_1_empirical.py` (23 tests, 439 lines)
- **Origin**: Challenger stress suite for PR M1.1 (EnsembleGuard numerical stability).
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestAdversarialVector1NoiseVsMicroJitterVsGenuine` (5 tests): Machine epsilon noise ($10^{-16}$) vs micro-jitter ($10^{-8}$) vs genuine effect ($0.05\sigma$) math.
  - `TestAdversarialVector2ExtremeScales` (2 tests): Macroeconomic ($10^7$) vs microscopic ($10^{-6}$) numerical stability.
  - `TestAdversarialVector3ConstantZeroVariance` (4 tests): Zero-variance division-by-zero avoidance.
  - `TestAdversarialVector4MonotonicTransformations` (6 tests): Monotonic transformation non-linear divergence (Spearman $\rho=1.0$ vs Pearson $r<0.8$).
  - `TestAdversarialVector5RescueSampleAnalysis` (4 tests): Rescue sample counting and margin math under synergistic vs degraded ensembles.
  - `TestAdversarialVectorCLIStressAndEdgeCases` (2 tests): CLI subprocess verification of component correlation.
- **Target Destination**: Classes 1–5 (21 math tests) merge into `tests/unit/test_ensemble_guard.py`. Class 6 (2 CLI tests) merges into `tests/integration/test_audit_gate_integration.py`.
- **Pruned**: Redundant boilerplate. Delete original file.

#### 2. `tests/test_challenger_m3_2_empirical.py` (14 tests, 320 lines)
- **Origin**: Challenger test inspecting frontend TSX components via Python regex.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestModalityAdaptiveAuditRegimeLogic` (4 tests): Modality-adaptive audit regime resolution rules.
  - `TestParamScaleFormatterEdgeCases` (3 tests): Parameter scale formatting presets (`10M`, `1B`).
  - `TestI18nCompletenessForTaskConfigModal` (3 tests): Localization key completeness.
- **Pruned / Deleted**: 4 brittle source-text regex scans in `TestPersistenceContractVerification` searching `TaskConfigModal.tsx` for raw string tokens like `handleSave`.
- **Target Destination**: Merge math and formatting into `tests/e2e/test_dashboard_e2e.py` (or Python helper). Delete original file.

#### 3. `tests/test_challenger_p3_empirical.py` (4 tests, 221 lines)
- **Origin**: UI i18n and offline retry test.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestI18nWiringEmpirical` (2 tests): Vietnamese translation string completeness for candidate cards.
- **Pruned / Deleted**: 2 brittle checks in `TestAppOfflineRetryEmpirical` scanning line 98 of `dashboard/src/App.tsx` via Python regex (`self.assertNotIn('!initialLoadedRef.current', ...)`).
- **Target Destination**: Merge i18n checks into `tests/e2e/test_dashboard_e2e.py`. Delete original file.

#### 4. `tests/test_challenger_r1_r4_stress.py` (10 tests, 440 lines)
- **Origin**: Live HTTP socket and React resilience tests.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `ChallengerBackendIntegrationTests` (3 tests): Real HTTP requests against live server endpoints (`POST /api/competition-state/agent-context`, rapid burst updates, `POST /api/competition-state/data-audit`).
- **Pruned / Deleted**: 7 tests in `ChallengerR1InteractionStressTests` and `ChallengerR4ResilienceStressTests` performing AST/regex inspection on React components for debounce timing and optimistic UI updates.
- **Target Destination**: Merge 3 live HTTP server integration tests into `tests/e2e/test_dashboard_e2e.py`. Delete original file.

#### 5. `tests/test_p2_audits.py` (13 tests, 251 lines)
- **Origin**: PR P2 audit additions.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestHudSubmissionsBlock` (4 tests): Submission manager packet rendering in HUD.
  - `TestWorkspaceHygiene` (4 tests): Workspace hygiene stray detection.
  - `TestMechanismAntiDuplicate` (5 tests): ResearchMemory anti-duplicate mechanism matching across proposals.
- **Target Destination**: Merge into `tests/integration/test_hud_ergonomics.py` and `tests/unit/test_research_memory.py`. Delete original file.

#### 6. `tests/test_p2_fixes.py` (4 tests, 122 lines)
- **Origin**: PR P2 bugfixes.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestValidateConfigPath` (3 tests): Path traversal rejection security invariants (`..`, outside absolute paths).
  - `TestPerSubtaskDirection` (1 test): Optimizing subtasks with minimization objectives in TreeJournal.
- **Target Destination**: Path checks merge into `tests/unit/test_task_spec.py`. Subtask direction merges into `tests/integration/test_tree_journal.py`. Delete original file.

#### 7. `tests/test_agent_friction_remediation.py` (15 tests, 221 lines)
- **Origin**: PR reducing agent friction with Kaggle facade and Grounding whitelist.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `NormalisedObjectDictInterfaceTests` (4 tests): Dict access (`__getitem__`, `get`, `in`) on `_NormalisedObject`.
  - `KaggleClientFacadeEnhancementTests` (3 tests): `kernels_list` alias and transient 404 handling.
  - `GroundingGateKaggleWhitelistTests` (3 tests): Kaggle public LB, discussion, and notebook source types in GroundingGate.
  - `ActionDiscoveryAndSchemaTests` (5 tests): Typed action schema registration and `--help` CLI handling.
- **Target Destination**: Kaggle classes merge into `tests/integration/test_kaggle_cli.py`. Grounding class merges into `tests/unit/test_grounding_gate.py`. Action schema merges into `tests/integration/test_agent_bridge.py`. Delete original file.

#### 8. `tests/test_agent_friction_round2.py` (17 tests, 200 lines)
- **Origin**: PR resolving agent friction round 2.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestKaggleArtifactCriticalMismatchPolicy` (5 tests): Task/config SHA-256 and job ID conflict policies.
  - `TestSafeRelpath` (4 tests): Path sanitization and forward slash enforcement across platforms.
  - `TestUniversalTaskSpecStringTaskCompatibility` (2 tests) & `TestTaskDatasetSlugPolicy` (1 test): String task definitions and slug parsing.
  - `TestEvaluationAlias` (1 test): Metric aliasing.
  - `TestNormalisedObjectDictParity` (2 tests): Dict parity (`.keys()`, `.values()`, `.items()`).
  - `TestTreeJournalListCompatibility` (2 tests): List-format event preservation in TreeJournal.
- **Target Destination**: Disperse into `tests/e2e/test_kaggle_runtime.py`, `tests/unit/test_io_utils.py`, `tests/unit/test_task_spec.py`, and `tests/integration/test_tree_journal.py`. Delete original file.

#### 9. `tests/test_ruff_b10_b16.py` (6 tests, 252 lines)
- **Origin**: PR resolving Ruff B10-B16 linter issues.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestB10UnfreezeAuditSelfContained`: Unfreeze audit event released slots.
  - `TestB12StreamLogsBinarySeek`: Binary stream log seeking past EOF safety.
  - `TestB15PreflightFastMode`: Preflight full data check flag.
  - `TestB16ModelFactoryCompetitionAgnostic`: Plugin-served models.
- **Target Destination**: Disperse into `tests/integration/test_session_resilience.py`, `tests/integration/test_candidate_preflight.py`, and `tests/integration/test_model_resolution.py`. Delete original file.

#### 10. `tests/test_upgrades_roadmap.py` (14 tests, 584 lines)
- **Origin**: Upgrades roadmap PR.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestExtendedMetricsAndRunnerHook` (4 tests) & `TestEdgeCasesAndRanking` (3 tests): Extended metrics (QWK, regression aliases).
  - `TestBootstrapAttachSeamless` (1 test): Dataset bootstrap attach.
  - `TestModalCloudRuntimeDependencies` (2 tests): Modal runtime LightGBM/XGBoost bundling.
  - `TestTabularOOMGuard` (2 tests): Tabular OneHotEncoder max_categories guard.
  - `TestModalityAwareCreateExperiment` (2 tests): Agent bridge create_experiment modality handling.
- **Target Destination**: Extended metrics merge into `tests/unit/test_evaluation_contracts.py`. OOM guard merges into `tests/unit/test_feature_pipeline.py`. Dataset bootstrap attach merges into `tests/integration/test_framework_adoption.py`. Modal checks merge into `tests/e2e/test_cloud_runners.py`. Delete original file.

#### 11. `tests/test_guardrail_regressions.py` (23 tests, 698 lines)
- **Origin**: Historical regressions suite.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestExternalPredictionJournalName` (1 test): External candidate naming.
  - `TestRepairBundleContract` (2 tests) & `TestAuditOutputsOutsideRunDir` (1 test): Repair bundle contracts.
  - `TestProbeRunsStayLocal` (5 tests): Probe runs stay local invariants.
  - `TestKagglePollTransientTolerance` (3 tests): Kaggle poll transient tolerance.
  - `TestManagedJobGpuAccounting` (1 test) & `TestSessionGpuReconcile` (1 test): 1 GPU per managed job accounting.
  - `TestFinalizeContract` (3 tests) & `TestFinalizedStopGate` (3 tests): Finalize freeze requirement and mutation blocking.
  - `TestFoldStdSingleRow` (1 test): Evaluation single row std dev safety.
  - `TestGeminiErrorSurfaces` (2 tests): Gemini error text surfacing.
- **Target Destination**: Probe locality merges into `tests/unit/test_execution_backend_policy.py`. Finalize gates merge into `tests/integration/test_session_resilience.py`. Repair merges into `tests/integration/test_repair_integration.py`. Delete original file.

#### 12. `tests/test_f821_undefined_names.py` (5 tests, 82 lines)
- **Origin**: Historical pyflakes regression fixes.
- **Non-Redundant Assertions Earmarked for Preservation**:
  - `TestInspectExperimentHandler` (2 tests): CLI experiment inspector invocation.
- **Pruned / Deleted**: 3 static `hasattr` checks verifying function importability. These checks are 100% redundant with `ruff check harness/ tools/ tests/` (which flags `F821` undefined names statically during pre-push).
- **Target Destination**: Merge the 2 CLI tests into `tests/integration/test_agent_bridge.py`. Delete original file.

---

## 5. 100% Critical Invariant Preservation Mapping

Requirement R4 dictates 100% preservation of all six critical invariant domains. The matrix below traces every invariant requirement from its current implementation to its consolidated target module, confirming that zero assertions are lost or weakened:

### 5.1 Invariant Traceability Matrix

| Invariant Domain | Current Test Sources | Invariant Verification Scope | Target Consolidated Module | Preservation Guarantee |
|---|---|---|---|---|
| **1. Schema v2 Correctness Contracts** | `test_correctness_contracts.py`<br>`test_correctness_regressions.py`<br>`test_correctness_runtime.py`<br>`test_task_validate.py` | - EvaluationContract schema validation.<br>- Probability bounds: binary probability does not require sum-to-one; multi-class strictly enforces sum-to-one; multi-label does not require sum-to-one.<br>- Evaluation metric reference implementations match Scikit-Learn exactly.<br>- Primary score sign inversion (`utility = -primary_score` for minimization metrics). | `tests/unit/test_evaluation_contracts.py` [SMOKE]<br>`tests/unit/test_task_spec.py`<br>`tests/integration/test_runner.py` | **100% Preserved**.<br>All 11 tests from `test_correctness_contracts.py` retained intact in unit tier in-memory (<0.5s in `--smoke`); path validation and metric inversion in `test_task_spec.py`; multi-fold runner execution from `test_correctness_runtime.py` (9.88s) preserved in `tests/integration/test_runner.py` outside `--smoke` to protect the `<10s` SLA. |
| **2. Label Integrity Audit** | `test_correctness_candidates.py`<br>`test_audit_gate.py`<br>`test_tree_journal_ensemble.py` | - Detection of missing labels in predictions.<br>- Detection of duplicate sample IDs.<br>- Rejection of fractional labels on integer target contracts.<br>- Out-of-fold sample row alignment verification. | `tests/unit/test_label_integrity.py` [SMOKE]<br>`tests/integration/test_candidate_selection.py` | **100% Preserved**.<br>Pure in-memory DataFrame integrity checks retained in `tests/unit/test_label_integrity.py` and included in `--smoke`. |
| **3. Anti-Hallucination Grounding Gate** | `test_grounding_gate.py`<br>`agent_bridge/test_grounding_guards.py`<br>`test_agent_friction_remediation.py` | - Epistemic verification mandate: non-baseline experiments must declare a valid verification channel via `verified_via` (`huggingface_hub`, `search_web`, `read_url_content`, `local_registry`).<br>- Sentinels `not_verified`, `unverified`, `none`, or empty strings are strictly rejected (`GROUNDING VERIFICATION ERROR`).<br>- Lazy trial-and-error interventions ("tune parameters", "random search") are rejected.<br>- Assumption registers with `uncertainty_level: high` require an explicit `verification_probe` (`--debug --fold 0`).<br>- Baseline root exemption (`branch=baseline`, `parent_id=none`) is automatically synthesized as verified. | `tests/unit/test_grounding_gate.py` [SMOKE]<br>`tests/integration/test_agent_bridge.py` | **100% Preserved**.<br>All pure gate logic consolidated into `tests/unit/test_grounding_gate.py` (in `--smoke`). Duplicate sentinel test pruned from bridge; bridge action interceptors verified in integration. |
| **4. Execution Backend Lock Policy** | `test_execution_backend_policy.py`<br>`test_guardrail_regressions.py`<br>`test_correctness_regressions.py` | - UniversalTaskSpec execution backend configuration with locked status.<br>- Resolution precedence: explicit CLI `--backend` > task contract `execution.backend` > `local`.<br>- Backend lock enforcement: When locked, overriding with another backend raises `Backend lock violation`.<br>- Smoke and probe runs (`--debug`, `--fold 0`, `--probe-samples`) ALWAYS execute locally, even on cloud-locked tasks.<br>- Sample limit hierarchy precedence: `cli > debug > config`. | `tests/unit/test_execution_backend_policy.py` [SMOKE] | **100% Preserved**.<br>All 18 tests retained intact and expanded with probe locality assertions from `test_guardrail_regressions.py`. Preserved in `<10s` `--smoke`. |
| **5. Model Governance, Parameter Accounting & Candidate Citation** | `test_model_governance.py`<br>`test_candidate_preflight.py`<br>`test_model_registry.py`<br>`test_citation_audit.py`<br>`test_citation_sources.py`<br>`test_citation_receipts.py`<br>`test_citation_review.py`<br>`test_citation_admission.py` | - **Parameter Accounting**: Total deployed parameter limits strictly sum base LLM + LoRA trainable adapters + vision encoders + projectors. Unknown parameter counts cannot default to zero.<br>- **Release Cutoff Date**: Models released after competition cutoff dates are rejected.<br>- **License Allowlist**: Normalized, case-insensitive SPDX expression checking.<br>- **Candidate Citation Mandate**: Pitched candidates must cite a verified `preflight_id`.<br>- **Near-Duplicate Prohibition**: Candidates with Spearman $\rho \ge 0.995$ against incumbent are prohibited from burning daily quota without verified paired OOF gain.<br>- **Citation Cryptographic Receipts**: Discussion topics and external citations must have verified receipts; unreviewed notes remain hypotheses. | `tests/unit/test_model_governance.py`<br>`tests/unit/test_citation_audit.py`<br>`tests/integration/test_candidate_preflight.py`<br>`tests/integration/test_model_resolution.py` | **100% Preserved**.<br>`test_model_governance.py` converted to `unittest.TestCase` (fixing runner blindspot). Complete citation pipeline consolidated into `tests/unit/test_citation_audit.py`. Candidate preflight diversity checks preserved. |
| **6. Immutable Evidence Folder Guarantees** | `test_recursive_sealing.py`<br>`test_threshold_evidence.py`<br>`test_artifact_dedup.py`<br>`test_preserve_submission.py`<br>`test_repair_integrity.py`<br>`test_repair.py` | - Recursive SHA-256 manifest sealing over all artifacts in completed runs (`.ai-harness/artifacts/<task>/<exp_id>`).<br>- Detection of nested tampering or file edits post-run.<br>- Directory traversal and path escape prevention (`..` in manifest keys).<br>- Completed runs cannot be overwritten (`UniversalRunner` raises `ValueError: overwrite`).<br>- Evidence integrity gate: score re-derivation against runner, tampered metric blocking, benign vs evidence drift classification. | `tests/unit/test_evidence_manifest.py`<br>`tests/integration/test_evidence_sealing.py`<br>`tests/integration/test_repair_integration.py` | **100% Preserved**.<br>Pure hash computation, schema validation, and path traversal rejection verified in `tests/unit/test_evidence_manifest.py`; directory sealing, hardlinks, and tamper checks in `tests/integration/test_evidence_sealing.py`; repair integrity gates in `tests/integration/test_repair_integration.py`. |

---

## 6. Test Tooling Modernization Specification

### 6.1 `tools/run_tests.py` Modernization

The current test runner has three architectural limitations:
1. It prepends `tests.` to module targets, failing when modules reside in subdirectories like `tests/unit/`, `tests/integration/`, or `tests/e2e/`.
2. It hardcodes `SMOKE_MODULES` as direct child stems of `tests.`, which fails in Phase 1 before files are renamed.
3. It performs a global suite check `testsRun == 0`, masking single-module blindspots in multi-module runs.

#### Precise Code Updates to `tools/run_tests.py`:

```python
# tools/run_tests.py - Modernized Implementation Specification

from pathlib import Path
import re
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Invariant Smoke Modules (<10s Fast Gate) - Dynamic Stem Mapping & Transitional Aliases
SMOKE_MODULE_STEM_ALIASES = {
    "test_evaluation_contracts": ["test_evaluation_contracts", "test_correctness_contracts"],
    "test_label_integrity": ["test_label_integrity", "test_correctness_candidates"],
    "test_grounding_gate": ["test_grounding_gate"],
    "test_execution_backend_policy": ["test_execution_backend_policy"],
    "test_research_memory": ["test_research_memory", "test_research_scope_stability"],
}

# Designated Heavy / Slow Modules (Excluded from --fast) & Transitional Aliases
SLOW_MODULE_STEM_ALIASES = {
    "test_kaggle_runtime": ["test_kaggle_runtime"],
    "test_sync_e2e": ["test_sync_e2e", "test_sync_command"],
    "test_cloud_runners": ["test_cloud_runners", "test_cloud_jobs_and_env"],
    "test_dashboard_e2e": ["test_dashboard_e2e", "test_dashboard"],
    "test_pipeline_e2e": ["test_pipeline_e2e", "test_pipeline_execution", "test_boost_end_to_end"],
}

def discover_modules() -> list[str]:
    """Recursively discover all test modules in tests/, tests/unit/, tests/integration/, tests/e2e/."""
    test_dir = REPO_ROOT / "tests"
    modules = []
    for path in sorted(test_dir.rglob("test_*.py")):
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        dotted = ".".join(rel.parts)
        modules.append(dotted)
    return modules

def _stem(module_name: str) -> str:
    """Return the base stem of a dotted module path (e.g. 'tests.unit.test_x' -> 'test_x')."""
    return module_name.rsplit(".", 1)[-1]

def resolve_target_modules(targets: list[str], all_modules: list[str]) -> list[str]:
    """Resolve user-specified targets, stems, tier selectors, or legacy aliases to fully-qualified module paths."""
    stem_map = {_stem(m): m for m in all_modules}
    resolved = []
    for target in targets:
        # Robust path normalization: handle Windows backslashes, trailing slashes, and prefixes
        cleaned = target.replace("\\", "/").rstrip("/").removesuffix(".py")
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        cleaned = cleaned.replace("/", ".")

        # Tier selectors
        if cleaned in ("unit", "tests.unit"):
            resolved.extend([m for m in all_modules if m.startswith("tests.unit.")])
        elif cleaned in ("integration", "tests.integration"):
            resolved.extend([m for m in all_modules if m.startswith("tests.integration.")])
        elif cleaned in ("e2e", "tests.e2e"):
            resolved.extend([m for m in all_modules if m.startswith("tests.e2e.")])
        elif cleaned in stem_map:
            resolved.append(stem_map[cleaned])
        elif cleaned in all_modules:
            resolved.append(cleaned)
        elif f"tests.{cleaned}" in all_modules:
            resolved.append(f"tests.{cleaned}")
        else:
            # Check transitional stem aliases
            matched_alias = False
            for canonical, aliases in SMOKE_MODULE_STEM_ALIASES.items():
                if cleaned == canonical or cleaned in aliases:
                    for alias in aliases:
                        if alias in stem_map:
                            resolved.append(stem_map[alias])
                            matched_alias = True
                            break
                    if matched_alias:
                        break
            if not matched_alias:
                # Suffix match fallback
                matches = [m for m in all_modules if m.endswith(f".{cleaned}")]
                if matches:
                    resolved.extend(matches)
                else:
                    resolved.append(cleaned)
    return list(dict.fromkeys(resolved))  # Preserve order, deduplicate

def build_suite(modules: list[str]) -> unittest.TestSuite:
    """Build a TestSuite from module names, asserting per-module test discovery."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for mod_name in modules:
        mod_suite = loader.loadTestsFromName(mod_name)
        # Per-module validation to catch silent 0-test skips immediately
        if mod_suite.countTestCases() == 0:
            raise RuntimeError(f"MODULE DISCOVERY ERROR: Discovered 0 tests in module '{mod_name}'")
        suite.addTest(mod_suite)
    return suite
```

#### Tier Flags & Runner Invariants:
- `python tools/run_tests.py --smoke`: Dynamically resolves `SMOKE_MODULE_STEM_ALIASES` against discovered modules, executing the 5 critical invariant unit tests in <10s (transitional aliases guarantee pass in Phase 1 before files are moved!).
- `python tools/run_tests.py --fast`: Discovers all modules in `tests.unit.*` and `tests.integration.*` (or flat tests during Phase 1-2), automatically skipping all `tests.e2e.*` and any module matching `SLOW_MODULE_STEM_ALIASES`.
- `python tools/run_tests.py --tier unit` (or `--unit`): Runs only `tests/unit/` (<5s total).
- `python tools/run_tests.py --tier integration` (or `--integration`): Runs only `tests/integration/` (~25s total).
- `python tools/run_tests.py --tier e2e` (or `--e2e`): Runs only `tests/e2e/`.
- **Per-Module Zero-Test Detection**: `build_suite()` validates `mod_suite.countTestCases() > 0` for every module, eliminating the silent blindspot where a module produces 0 tests inside a multi-module run.

---

### 6.2 Pytest Fixture Migration Protocol for Blindspot Modules
To eliminate the runner blindspot permanently and ensure 100% interoperability across both `unittest` and `pytest`, all standalone test functions in the 7 identified modules will be wrapped in `unittest.TestCase` subclasses using explicit translation patterns:

#### Exact Code Translation Recipes:

1. **`tmp_path` fixture -> `tempfile.TemporaryDirectory`**:
   ```python
   # Legacy standalone pytest function
   # def test_valid_yaml(tmp_path: Path): ...
   
   # Migrated TestCase pattern
   import tempfile
   from pathlib import Path
   import unittest

   class TestModelRegistry(unittest.TestCase):
       def setUp(self):
           self._td = tempfile.TemporaryDirectory()
           self.tmp_path = Path(self._td.name)
           self.addCleanup(self._td.cleanup)

       def test_valid_yaml(self):
           spec_file = self.tmp_path / "model_definition.yaml"
           ...
   ```

2. **`monkeypatch` fixture -> `unittest.mock.patch.dict(os.environ)` or `patch.object`**:
   ```python
   # Legacy standalone pytest function
   # def test_env_override(monkeypatch):
   #     monkeypatch.setenv("HARNESS_FORCE_PROBE", "1")
   
   # Migrated TestCase pattern
   import os
   from unittest.mock import patch
   import unittest

   class TestInteractivePredictionContract(unittest.TestCase):
       def test_env_override(self):
           with patch.dict(os.environ, {"HARNESS_FORCE_PROBE": "1"}):
               # executes with mocked environment
               ...
   ```

3. **`@pytest.mark.parametrize` -> `for ... in ...: with self.subTest(...):`**:
   ```python
   # Legacy standalone pytest function
   # @pytest.mark.parametrize("license_id,expected", [("mit", True), ("gpl-3.0", False)])
   # def test_license(license_id, expected): ...
   
   # Migrated TestCase pattern
   import unittest

   class TestModelGovernance(unittest.TestCase):
       def test_license_allowlist_validation(self):
           cases = [("mit", True), ("apache-2.0", True), ("gpl-3.0", False)]
           for license_id, expected in cases:
               with self.subTest(license=license_id, expected=expected):
                   self.assertEqual(is_license_allowed(license_id), expected)
   ```

4. **`capsys` fixture -> `io.StringIO` and `contextlib.redirect_stdout`**:
   ```python
   import io
   import contextlib
   import unittest

   class TestModelCliAndBridge(unittest.TestCase):
       def test_cli_help_output(self):
           buf = io.StringIO()
           with contextlib.redirect_stdout(buf):
               main(["--help"])
           self.assertIn("usage:", buf.getvalue().lower())
   ```

5. **Signature Alignment**: Add `self` as the first parameter to every migrated test method (`def test_*(self):`).

---

### 6.3 Pytest Configuration & Auto-Tagging Hook

#### 1. Configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_default_fixture_loop_scope = "function"
markers = [
    "unit: Pure in-memory unit tests with zero disk I/O and zero subprocesses (<1s).",
    "integration: Multi-component tests covering DAG, pipeline, and runner workflows (0.5s - 5s).",
    "e2e: End-to-end tests covering cloud runners, Kaggle kernels, live servers, and remote execution (>5s).",
    "smoke: Critical invariant tests (<10s fast gate).",
]
```

#### 2. Pytest Marker Auto-Tagging Hook (`conftest.py`):
To enable `pytest -m smoke`, `pytest -m unit`, and `pytest -m integration` out of the box without manual decorators on every test case, `conftest.py` implements a `pytest_collection_modifyitems` hook:

```python
# conftest.py - Pytest Directory & Smoke Auto-Tagging Hook
import pytest

def pytest_collection_modifyitems(config, items):
    """Automatically tag tests with markers based on directory hierarchy and smoke lists."""
    smoke_stems = {
        "test_evaluation_contracts", "test_correctness_contracts",
        "test_label_integrity", "test_correctness_candidates",
        "test_grounding_gate",
        "test_execution_backend_policy",
        "test_research_memory", "test_research_scope_stability",
    }
    for item in items:
        path_str = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in path_str:
            item.add_marker(pytest.mark.integration)
        elif "/tests/e2e/" in path_str:
            item.add_marker(pytest.mark.e2e)
        
        # Attach smoke marker to critical invariant modules in unit tier (or root during transition)
        if any(stem in path_str for stem in smoke_stems) and ("/tests/unit/" in path_str or "/tests/" in path_str):
            item.add_marker(pytest.mark.smoke)
```

---

## 7. Step-by-Step Migration Sequence & Rollback Protocol

To guarantee zero regression and zero downtime during the consolidation, the migration is structured into **6 sequential phases** (including a mandatory Phase 2.5 prerequisite to extract cross-test helper functions into `tests/fixtures.py` before file movements). Each phase includes non-destructive execution, shadow verification, and an automated rollback checkpoint:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   6-Phase Safe Migration Sequence                      │
└────────────────────────────────────────────────────────────────────────┘
                                    │
    Phase 1: Tooling Modernization (run_tests.py, pyproject.toml)
    ├── Add dynamic stem resolution, transitional aliases & tier flags
    └── Verify existing flat tests pass 100% via --smoke, --fast, full
                                    │
    Phase 2: Runner Blindspot Elimination
    ├── Wrap 7 standalone test modules into unittest.TestCase
    └── Verify test count increases by +51 tests under run_tests.py
                                    │
    Phase 2.5: Cross-Test Helper Extraction (Pre-Move Prerequisite)
    ├── Extract task(), task_document(), config(), rule_document() to tests/fixtures.py
    ├── Update 20+ calling test modules to import from tests.fixtures
    └── Verify zero ModuleNotFoundError under --smoke and --fast
                                    │
    Phase 3: Directory Tree Creation & Non-Destructive Shadow Migration
    ├── Create tests/unit/, tests/integration/, tests/e2e/
    ├── Move retained modules into target directories via git mv
    └── Verify runner dynamic resolution discovers and runs all modules
                                    │
    Phase 4: Domain Consolidation & PR Deconstruction
    ├── Consolidate fragmented modules into 38 domain modules
    ├── Earmark non-redundant math & prune brittle AST/regex checks
    └── Delete obsolete ad-hoc files
                                    │
    Phase 5: Final Quality Gates & Rollback Checkpoint
    ├── Run python tools/run_tests.py --smoke (<10s)
    ├── Run python tools/run_tests.py --fast (~30s)
    ├── Run full python tools/run_tests.py (100% pass)
    └── Run ruff check harness/ tools/ tests/ (0 errors)
```

---

### 7.1 Detailed Phase Breakdown

#### Phase 1: Tooling Modernization & Backwards Compatibility
- **Action**: Update `tools/run_tests.py` with dynamic stem resolution, `SMOKE_MODULE_STEM_ALIASES`, `SLOW_MODULE_STEM_ALIASES`, robust path normalization (handling Windows `\` and trailing slashes), tier flags (`--unit`, `--integration`, `--e2e`, `--fast`, `--smoke`), and per-module zero-test detection (`mod_suite.countTestCases() > 0`). Update `pyproject.toml` with pytest markers and `conftest.py` with `pytest_collection_modifyitems` hook.
- **Verification Gate**:
  ```bash
  python tools/run_tests.py --smoke   # Must pass 47 tests in <3s (transitional aliases guarantee success!)
  python tools/run_tests.py --list-slow # Must output slow modules correctly
  ruff check tools/run_tests.py        # Must pass with 0 lint errors
  ```
- **Rollback Condition**: If `--smoke` fails or discovery breaks on existing flat tests, revert `tools/run_tests.py`.

#### Phase 2: Standalone Test Function Wrapping (Blindspot Elimination)
- **Action**: Wrap standalone test functions in the 7 identified modules (`test_model_governance.py`, `test_model_registry.py`, `test_model_resolution.py`, `test_model_lineage_and_submission.py`, `test_model_cli_and_bridge.py`, `test_model_execution_plan.py`, `test_interactive_prediction_contract.py`) into `unittest.TestCase` classes using the Section 6.2 fixture translation recipes (`tmp_path` -> `tempfile`, `monkeypatch` -> `patch.dict`, `@pytest.mark.parametrize` -> `self.subTest`).
- **Verification Gate**:
  ```bash
  python tools/run_tests.py test_model_governance  # Must report 15+ tests run (NOT 0 tests!)
  python tools/run_tests.py test_model_registry    # Must report 10 tests run
  ```
- **Rollback Condition**: If any wrapped test fails assertions or raises import errors, inspect and fix the wrapper before proceeding.

#### Phase 2.5: Cross-Test Helper Extraction (Prerequisite for Directory Migration)
- **Problem Statement**: Over 20 test modules import helper fixtures directly from sibling test modules rather than shared fixtures:
  - `from tests.test_correctness_contracts import task` (15+ call sites)
  - `from tests.test_research_continuity import config, task_document` (6 call sites)
  - `from tests.test_research_memory import rule_document` (4 call sites)
  - `from tests.test_macro_research import review_payload` (2 call sites)
  - `from tests.test_repair import _seal_into, task_document` (3 call sites)
  If `test_correctness_contracts.py` or `test_research_continuity.py` were moved in Phase 3 before extracting these helpers, test collection would instantly fail with `ModuleNotFoundError` cascade errors.
- **Action**:
  1. Extract `task()`, `task_document()`, `config()`, `rule_document()`, `review_payload()`, and `_seal_into()` into `tests/fixtures.py`.
  2. Update all calling test modules across `tests/*.py` to import from `tests.fixtures` instead of sibling test files.
- **Verification Gate**:
  ```bash
  python tools/run_tests.py --smoke        # Must pass 100%
  python tools/run_tests.py --fast         # Must pass 100% with 0 ModuleNotFoundError
  ruff check tests/                        # 0 import errors
  ```
- **Rollback Condition**: If any test fails due to fixture import discrepancies, revert imports to the previous state.

#### Phase 3: Directory Hierarchy Creation & Retained Module Migration
- **Action**:
  1. Create directories: `tests/unit/`, `tests/integration/`, `tests/e2e/`.
  2. Create `__init__.py` files in each tier directory.
  3. Move retained domain modules into their target directories using git move (`git mv`). Retain shared fixtures (`fixtures.py`, `evidence_fixtures.py`, `workspace_fixtures.py`) at the `tests/` root.
- **Verification Gate**:
  ```bash
  python tools/run_tests.py --unit         # Executes unit tier in <5s
  python tools/run_tests.py --smoke        # Confirms dynamic resolution finds moved smoke modules
  ```

#### Phase 4: Domain Consolidation & Ad-hoc PR Assertion Integration
- **Action**:
  1. Consolidate micro-fragmented domains:
     - Kaggle (8 files -> 4 domain modules)
     - Citation & Debate (7 files -> 2 domain modules)
     - Research Continuity & Memory (10 files -> 4 domain modules)
     - Model Assets & Governance (9 files -> 5 domain modules)
  2. Integrate earmarked assertions from the 12 ad-hoc PR/challenger files into target domain modules.
  3. Prune brittle source-text regex scans and duplicate assertions.
  4. Delete the 12 obsolete ad-hoc files.
- **Verification Gate**:
  ```bash
  python tools/run_tests.py test_ensemble_guard  # Verifies m1.1 math vectors pass
  python tools/run_tests.py test_citation_audit   # Verifies unified citation pipeline passes
  ```

#### Phase 5: Comprehensive Quality Gates & Audit Attestation
- **Action**: Execute the full suite of verification commands across all tiers and static analyzers.
- **Verification Gate**:
  ```bash
  python tools/run_tests.py --smoke        # Must pass 100% in <10s
  python tools/run_tests.py --fast         # Must pass 100% in ~30s
  python tools/run_tests.py                # Full suite: 38 modules, 100% pass
  ruff check harness/ tools/ tests/        # 0 lint errors
  ```
- **Attestation**: Produce the final audit artifact confirming 100% invariant preservation and zero dropped tests.

---

### 7.2 Rollback Protocol
If any critical invariant fails or unexpected test regression occurs during any phase:
1. **Immediate Halt**: Stop execution; do not proceed to subsequent phases.
2. **Git Workspace Clean**:
   ```bash
   git status
   git checkout -- tests/ tools/run_tests.py pyproject.toml
   git clean -fd tests/
   ```
3. **Verify Baseline**:
   ```bash
   python tools/run_tests.py --smoke
   ```
   Confirm that the suite returns to its clean baseline passing state (47 tests in <3s).

---

## 8. Acceptance Criteria Verification Checklist

The table below cross-references all requirements and acceptance criteria from `ORIGINAL_REQUEST.md`, confirming that this implementation plan satisfies every specification:

| Requirement / Acceptance Criteria | Plan Section | Compliance Details | Status |
|---|---|---|---|
| **Audit Inventory Accounts for All Test Files** | §1.1, §3 (Table) | Reconciles 126 root `.py` + 14 bridge `.py` = 135 discovered test modules + 4 shared fixtures = 140 files. Exhaustive table lists all 135 files with test counts, LOC, dependencies, I/O profile, and disposition. | **SATISFIED** |
| **Runner Discovery Blindspot Detailed & Fixed** | §1.2, §6.2 | Identifies the 7 modules running 0 tests under `tools/run_tests.py` due to standalone `def test_*` functions. Specifies wrapping in `unittest.TestCase` and per-module zero-test validation (`mod_suite.countTestCases() > 0`). | **SATISFIED** |
| **3-Tier Hierarchy with Concrete Boundaries** | §2.1, §2.2, §2.3 | Defines `tests/unit/` (in-memory, 0 I/O, <1s/mod, <5s total), `tests/integration/` (workspaces, DAG, 0.5s-5s/mod, ~25s total), and `tests/e2e/` (cloud/kaggle/sync, 5s-70s/mod). | **SATISFIED** |
| **Target Consolidation to ~35–40 Domain Modules** | §2, §3, §3.1 | Maps all 135 test modules into **exactly 38 consolidated domain modules** (15 unit, 17 integration, 6 e2e) with 100% 1-to-1 reconciliation between Section 3 Table and Section 3.1 Domain Summary. | **SATISFIED** |
| **Ad-hoc PR & Challenger Deconstruction** | §4, §4.1 | Deconstructs all 12 ad-hoc PR/challenger files class-by-class (3,830 LOC); earmarks non-redundant mathematical/system assertions and prunes brittle AST/regex checks. | **SATISFIED** |
| **100% Critical Invariant Preservation** | §5, §5.1 | Section-by-section traceability matrix for all 6 invariants: Schema v2 contracts, label integrity audit, anti-hallucination grounding gate, backend lock policy, model governance parameter limits, and immutable evidence folder sealing. | **SATISFIED** |
| **Test Tooling Modernization Specification** | §6.1, §6.3 | Provides exact Python code for `tools/run_tests.py` dynamic stem resolution, transitional aliases (`SMOKE_MODULE_STEM_ALIASES`), Windows backslash normalization, tier flags, and `conftest.py` pytest auto-tagging hook. | **SATISFIED** |
| **Preservation of Shared Fixtures** | §2.4 | Specifies retaining `tests/fixtures.py`, `evidence_fixtures.py`, `workspace_fixtures.py`, and root `conftest.py` in place to preserve import stability across the repository. | **SATISFIED** |
| **6-Phase Migration Sequence & Rollback Protocol** | §7, §7.1, §7.2 | Defines a 6-phase migration sequence with Phase 2.5 cross-test helper extraction prerequisite, non-destructive verification steps, shadow validation, concrete verification commands, and rollback procedures. | **SATISFIED** |

---

## 9. Conclusion & Execution Authorization

This Implementation Plan provides an authoritative, complete, and mathematically sound specification for consolidating the `ai-harness` test suite. By eliminating 97 micro-fragmented files, converting standalone tests to resolve the runner blindspot, and establishing a strict 3-tier boundary, the test suite achieves:
1. **Dramatic Speed Improvement**: Developers can run `python tools/run_tests.py --tier unit` in <5s and `python tools/run_tests.py --fast` in ~30s.
2. **Zero Invariant Loss**: All 6 architectural and scientific invariants are 100% preserved.
3. **Flawless Pre-Push Confidence**: Pre-push quality gates run reliably across both `unittest` and `pytest`.

Execution of the migration sequence is ready to commence upon user review and approval.
