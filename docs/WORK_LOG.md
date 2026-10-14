# Repository work log

This journal tracks development of AI-Harness. Read the latest entry after a
restart and append an entry after a meaningful implementation/validation step
or before handing off. Earlier entries are historical evidence, not a claim
about the latest code. Task experiments have their own generated
`RESEARCH_PROGRESS.md` in the resolved artifact root.

Each entry records: request/observation, completed work, evidence, lesson,
remaining uncertainty, and next action. Record failures and abandoned directions
when they affect the next decision. Keep notes concise; do not copy transcripts,
raw datasets, or internal reasoning. Do not invent results or reconstruct lost
history from guesses.

## 2026-10-10 — Durable insight traces

- Observation: flat research notes did not preserve an insight's identity,
  planned test, experiment links, lessons, and derived directions.
- Completed: insight IDs and parent links; bounded probe plans; linked decisions
  with lessons and revisit conditions; paginated trace; pending-result triage;
  checkpoint insight IDs; generated workspace guidance and phase revision updates.
- Corrected: repeated-config metadata could erase insight links; reassigned unrun
  configs could attribute a new result to an older insight; smoke failures lacked
  experiment IDs in the research ledger.
- Evidence: 387 local tests passed in 140.262 seconds on a disposable checkout.
  Template single-fold smoke and a synthetic 12-row/two-fold NLP trace passed;
  restart recovered the linked result, lesson, derived insight and checkpoint.
  Hash checks preserved 43 configs/artifact/document files in the real workspace.
- Lesson: recoverable workflow records are necessary, but do not establish better
  agent reasoning or competition performance.
- Remaining: actual-agent continuity quality has not been measured.
- Next action: add a human-readable progress journal, then evaluate continuity
  with a fixed task, bounded budget, and an intentional context interruption.

## 2026-10-10 — Human-readable ongoing tracking

- Request: make completed work and next steps visible in Markdown while the agent works.
- Completed: generated task-level `RESEARCH_PROGRESS.md` with resume instructions,
  insight statuses and chronological notes/results. Research events and context
  reads refresh it; the HUD exposes its path. This development log and `AGENTS.md`
  now specify when assistants should record repository work.
- Boundary: unrecorded insights and ordinary code edits are not inferred
  automatically. Agents must summarize meaningful work through the research
  actions or this development journal. Tracking does not authorize continuous compute.
- Validation: 24 focused tests passed; all 390 tests passed in 151.749 seconds
  on an isolated checkout. Template smoke and synthetic two-fold/restart integration
  passed. Markdown refresh, evidence invalidation, concurrent notes, preservation
  of user-owned files, and recovery after report-write failure were exercised.
  Hash checks confirmed all 43 protected real-workspace files were unchanged.
- Lesson: event-driven Markdown makes recorded work visible, but recording discipline
  is still necessary; a generated journal cannot recover an insight never recorded.
- Next action: design the actual-agent continuity evaluation with a fixed task,
  interrupted context, repetition counts, interpretation completeness and a budget.

## 2026-10-13 — Canonical AI-Harness Rebranding & Decoupled Fast Testing Architecture

- Request: audit repository for legacy `ml-harness` bias, rebrand to `ai-harness`, resolve unit tests running heavy local models and stalling test cycles, and commit changes cleanly to remote repo.
- Completed:
  - **Comprehensive Rebranding**: Standardized canonical name to `ai-harness` across runtime (`harness/project.py`, `harness/runner.py`, `harness/skill_injector.py`, `harness/phase_workflow.py`, `harness/cli.py`, `harness/plugins.py`, `harness/environment_detector.py`, `harness/data_strategy.py`, `harness/llm_engine.py`, `harness/templates/`), developer tools (`tools/bootstrap_task.py`, `tools/dashboard_server.py`, `tools/agent_bridge_cli.py`, `tools/agent_runtime_cli.py`), Web dashboard (`dashboard/src/App.tsx`, `dashboard/src/constants/i18n.ts`, modals), skills (`.agents/skills/ai-harness/SKILL.md`), and documentation (`docs/README.md`, `docs/agent_workflow.md`, `docs/CORRECTNESS_CONTRACT.md`, `docs/plugin_architecture.md`, prompts).
  - **Dual-path Backward Compatibility**: Preserved automatic fallback resolution for legacy `.ml-harness` workspaces, manifests (`ml-harness-runtime.json`, `mlh-data-bundle.json`), and UI preference storage keys.
  - **Decoupled Testing Engine**: Implemented `DummyBaselineModel` in `harness/models.py` (registered under aliases `dummy`, `mock`, `dummy_baseline`, `dummy_classifier`, `dummy_regressor`) providing instant `O(1)` fitting, label alignment, and probability matrix output without heavy mathematical optimization.
  - **Mocked Process & Network Probes**: Mocked external OS and network calls in `test_cloud_jobs_and_env.py` (dropped from 18.0s to 0.3s) and replaced `localhost` with explicit `127.0.0.1` IPv4 binding in `test_llm_api_endpoints.py` (dropped from 30.5s to 1.0s), preventing Windows IPv6 fallback timeouts.
- Evidence: Full test suite (`python -m unittest discover -s tests -p "test_*.py"`) completed 392 tests in 74.13s (reduced from ~140s) with 100% pass rate.
- Lesson: Harness unit tests validating lifecycle DAG transitions, anti-leakage gates, and orchestration contracts should never depend on heavy numerical convergence or live network sockets; decoupling via lightweight dummy estimators and mocked environment probes achieves sub-second test execution.
- Next action: Partition changes into logical commits and push to remote `origin/dev`.

## 2026-10-13 — Complete Eradication of Legacy ml-harness & mlh- Prefixes

- Request: check if any traces of `ml-harness` remain in the repository and purge them completely.
- Completed:
  - **Script & CLI Removal**: Removed legacy `ml-harness*` script aliases from `pyproject.toml`.
  - **Storage & Runtime Purity**: Removed `ml_harness_ui_lang` key in `i18n.ts`, removed fallback `LEGACY_DATA_MANIFEST_NAME = "mlh-data-bundle.json"`, and eliminated `LEGACY_MANIFEST_RELATIVE_PATH` and `.ml-harness` checks in `harness/project.py`.
  - **Cloud & Resource Prefix Modernization**: Updated dataset, volume, code slug, and tempdir prefixes across `harness/cloud_policy.py`, `harness/cloud_runner.py`, `harness/templates/kaggle_template.py`, and `harness/kaggle_runtime.py` from `mlh-` to `aih-`.
  - **Web Assets & Test Assertions**: Updated pre-built web asset bundle and HTML title to `AI-Harness`, cleaned `.gitignore`, and updated all corresponding test assertions and fixtures in `test_workspace_workflow.py`, `test_kaggle_runtime.py`, `test_research_trace.py`, `test_bootstrap_task.py`, and `test_universal_plugin.py`.
- Evidence: `git grep -inE "ml[-_]harness"` yields 0 matches in codebase files; full test suite passed 392/392 tests in 79.0s.
- Lesson: Full deprecation of legacy prefixes ensures clean naming boundaries without ambiguous dual paths across cloud policies and local file caches.
- Next action: Commit and push changes to remote repository.
## 2026-10-14 — Dashboard Metadata, Test Fixtures, and AGENTS.md Quality Gate Alignment

- Request: audit repository for any lingering legacy metadata or clean-up targets.
- Completed:
  - **Dashboard Source Sync**: Updated `dashboard/package.json` name from `mldl-experiment-dashboard` to `ai-harness-dashboard`, and updated `dashboard/index.html` title to `AI-Harness & Priming Dashboard`.
  - **Skill Fixture Modernization**: Modernized mock skill directory path in `tests/test_bootstrap_task.py` from `.agents/skills/ml-experiment-harness` to canonical `.agents/skills/ai-harness`.
  - **Documentation & Quality Gates**: Updated `AGENTS.md` test count baseline from 330+ to 390+ tests, and clarified invocation context for single-fold smoke probe validation.
- Evidence: Full test suite (`python -m unittest discover -s tests -p "test_*.py"`) executes 392/392 tests with 100% pass rate in ~79s.
- Lesson: Keeping development documentation, package descriptors, and test fixtures synchronized prevents subtle drift between the core harness and auxiliary web/skill scaffolding.
## 2026-10-14 — Fix Kaggle Windows MAX_PATH Overflow and Valid License Identification

- Request: investigate and fix two root causes under `kaggle_bundle_mode: dataset` on Windows: `[Errno 2] No such file or directory` due to path length exceeding Windows `MAX_PATH` (260 characters) during Kaggle CLI token flattening, and `Dataset creation error: Please select a valid license` due to unrecognized `copyright-authors` license name in `dataset-metadata.json`.
- Completed:
  - **Windows MAX_PATH Protection**: Shortened code staging directory structure from `stage / "code-dataset" / bundle_sha256` (77 chars) to compact isolated `stage / "code" / bundle_sha256[:12]` (17 chars). Shortened bundle filename from `aih-code-{64-char-hash}.zip` (77 chars) to `aih-code-{16-char-hash}.zip` (29 chars). Shortened staging prefix in `cloud_runner.py` to `aih-kaggle-{job_id[:8]}-`. This reduces the flattened upload token file path from ~285+ characters to ~145 characters, safely below the 260-character limit.
  - **Valid Kaggle License Specification**: Changed hardcoded `"licenses": [{"name": "copyright-authors"}]` in `harness/templates/kaggle_template.py` to Kaggle's recognized custom license identifier `"licenses": [{"name": "other"}]` while preserving the full retained rights in the description. Also standardized code dataset title to `AI Harness Code {bundle_sha256[:32]}` and updated `docs/kaggle_execution.md`.
  - **Verification**: Updated `tests/test_kaggle_runtime.py` with explicit assertions verifying `"other"` license metadata, "AI Harness Code" title, and relative archive path length < 50 characters.
- Evidence: 392/392 tests passed (100%) across the full test suite.
- Lesson: Windows file paths when manipulated by third-party CLI tools (like Kaggle's flattened upload token files) require proactive defensive length budgeting; custom licenses for Kaggle datasets must conform to Kaggle's recognized license taxonomy (`other`).
- Next action: Commit and push changes to remote `origin/dev`.
