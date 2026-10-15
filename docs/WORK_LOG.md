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

## 2026-10-15 — Agent Usability, Experiment ID Resilience, OOF Serialization & Kaggle Exporter

- Request: analyze multi-agent friction across ~35 historical transcripts and active repositories (`aimo`, `airline_prediction`, `zindi`, `rsna-knee`), identify pain points, and enhance framework usability and resilience for AI coding assistants.
- Completed:
  - **CLI Shell Resilience**: Supported flexible JSON/YAML/single-quote parsing in `tools/agent_bridge_cli.py` to prevent PowerShell `ScriptBlock` syntax errors; promoted flat dynamic arguments (`--branch feat --descriptor v1`).
  - **Experiment ID Allocation & Collision Prevention**: Updated `AgentBridge.create_experiment` and added `AgentBridge.get_next_experiment_num` and `TypedActionName.GET_NEXT_EXPERIMENT_ID` action. Scans both `configs/` and `.ai-harness/artifacts/<task_id>/` (and legacy `.ml-harness`), guaranteeing auto-increment IDs (`exp002`, `exp003`) and guarding against accidental overwrites.
  - **OOF & Test Prediction Serialization**: Decomposed multiclass/binary probability predictions into native float columns (`proba_<target>_<class>` and `proba_<target>_1`) alongside legacy JSON strings; exported optional `oof_predictions.parquet` for instant vectorized downstream blending/stacking.
  - **Kaggle Private Dataset Resilience**: Defaulted `data_config.license` to `"CC0-1.0"` with notice log instead of failing with hard `RuntimeError`.
  - **Standalone Kaggle Notebook Exporter**: Implemented `harness/kaggle_notebook_builder.py` and CLI subcommand `ai-harness export-kaggle-notebook` to automatically bundle offline wheels installer, model weights loading, and submission validation into a standalone `.ipynb`.
  - **Compatibility Aliases & Documentation**: Added `ml-harness` and `ml-harness-agent` entrypoint aliases in `pyproject.toml`; updated `AGENTS.md` with PowerShell-safe CLI usage guidelines.
- Next action: Open PR into `main` via `gh pr create`.

## 2026-10-15 — Agent Usability Audit: Label Normalization, Import Aliases & Kaggle Remote Robustness

- Request: Deep audit of historical transcripts across ~35 conversation sessions in `~/.gemini/antigravity/brain` using `ai-harness` (`aimo`, `airline_prediction`, `zindi`, `rsna-knee`). Identify remaining friction points, errors, and workarounds; implement framework improvements and verify with pre-push quality gates.
- Completed:
  - **Label & Class Normalization in Evaluation**: Resolved `ValueError: Ground truth outside declared classes` and `ValueError: Predictions outside target domain` in `harness/evaluation.py` by implementing `_normalize_class_value` and `_normalize_class_array`. Integer-valued floats (e.g. `0.0`, `1.0` in pandas) now cleanly match declared integer classes (`[0, 1]`) and string classes (`['0', '1']`). Added actionable diagnostic guidance when models output continuous probabilities into discrete prediction fields.
  - **Transparent Package & Module Aliases**: Created `ai_harness` and `ml_harness` alias packages with `ModuleSpec` loader forwarding submodules (`ai_harness.task_spec`, `ml_harness.runner`) directly to `harness` module instances, eliminating `ModuleNotFoundError: No module named 'ai_harness'` and `'ml_harness'`. Created `harness/agent.py` forwarding to `harness.agent_bridge`. Added `Runner = UniversalRunner` and `build_parser()` in `harness/runner.py`. Exported `SubmissionValidator` and `kaggle_manager` in `harness/__init__.py`.
  - **Kaggle Remote Runner Error Logs**: Added `KaggleKernelManager.get_kernel_error_log` which downloads and extracts remote kernel stderr tracebacks upon failure, eliminating blind `KernelWorkerStatus.ERROR` exceptions in `harness/cloud_runner.py`.
  - **Kaggle API Transient Error Tolerance**: Upgraded `wait_for_dataset` to tolerate transient HTTP 403 Forbidden / 404 Not Found propagation glitches for up to 5 consecutive attempts.
  - **Identity Mismatch Tolerance & CLI Pull**: Supported non-fatal `core_version` mismatch warnings and `--force-import` in `pull_kernel_output` and runner; added `ai-harness cloud pull --exp-id ... [--kernel-slug ...] [--force-import]` CLI subcommand.
- Evidence: 425/425 unit tests passed in 107.206s (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 6 dedicated test cases in `tests/test_agent_usability_audit.py`.
- Lesson: Submodule import aliases in Python require dedicated `importlib.machinery.ModuleSpec` loaders to guarantee module identity (`is`); remote runner exceptions must proactively download and parse remote stream logs so that AI agents receive actionable tracebacks rather than opaque cloud worker status strings.
- Next action: Commit and push changes to remote branch `feat/agent-usability-p3` and update PR #29.

## 2026-10-15 — Agent Usability Phase 5: CatBoost Built-in, Feature Diagnostics, Project Defaults & Split Robustness

- Request: Continue deep transcript audit across `~/.gemini/antigravity/brain`, identify remaining failure modes, error messages, and missing capabilities for AI coding assistants working with `ai-harness`, implement features/fixes, test thoroughly, and merge into `main`.
- Completed:
  - **Built-in CatBoost Model Wrapper**: Added `CatBoostModel` wrapper to `harness/models.py` (registered under `catboost`, `cb`, `catboost_classifier`, `catboost_regressor`, `cb_classifier`, `cb_regressor`), with automated classification, regression, probability output, parameter calculation, and default `allow_writing_files: False` (preventing transient `catboost_info/` directories from polluting project workspaces).
  - **Feature Validation Diagnostics**: Replaced generic opaque `ValueError("Input features must exist and exclude targets, IDs, groups and split metadata")` in `harness/models.py` with `validate_model_features()` providing explicit diagnostics for empty feature sets, specific missing columns (with a sample of available DataFrame columns), and forbidden metadata overlaps.
  - **HarnessProject Default Auto-Discovery**: Added dataclass field defaults and `__post_init__` to `HarnessProject` in `harness/project.py`. Direct instantiation via `HarnessProject(".")` or `HarnessProject()` now auto-populates `manifest_path`, `task_config_path`, `plugin_id`, and `artifacts_dir`, eliminating `TypeError: missing 5 required positional arguments`.
  - **Split File Column Aliases & ID Coverage Diagnostics**: In `harness/runner.py`, enhanced split file loading to detect fold column aliases (`fold`, `kfold`, `split`, `fold_id`, `cv_fold`, `Fold`, `Split`), perform case-insensitive ID column matching, and provide clear diagnostics displaying the uncovered ID count and sample uncovered IDs if IDs don't match. Extracted `UniversalRunner._load_training_data()` and honored `paths.train_path` override in experiment configs.
  - **Task Specification Flexibility**: In `harness/task_spec.py`, added support for dictionary-formatted `inputs` in `UniversalTaskSpec.from_dict` alongside list-of-dicts; supported `id_column` and `split` path declarations under top-level task metadata.
  - **Agent Bridge Usability & CLI Serialization**: Added `AgentBridge.act` method alias matching `execute_action`; added `harness.agent_bridge.main` CLI forwarder for `python -m harness.agent_bridge`; implemented `_json_serial` fallback serializer in `tools/agent_bridge_cli.py` handling `Path`, `datetime`, numpy scalars/ndarrays, and custom objects to eliminate `TypeError: Object of type ... is not JSON serializable`. Added `catboost_info/` to `.gitignore`.
- Evidence: 431/431 unit tests passed in 111.122s across the full repository test suite (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 6 dedicated test cases in `tests/test_agent_usability_p5.py`.
- Lesson: Providing friendly defaults on core orchestration dataclasses and forgiving alias resolution on data splits and inputs dramatically reduces syntax friction for AI agents without compromising contract rigor.
- Next action: Create feature branch `feat/catboost-and-diagnostics`, push, open PR, and squash-merge into `main`.

## 2026-10-15 — Agent Usability Phase 6: Built-in RandomForest, Extended Linear Aliases, TreeJournal Load Flexibility & Cross-Environment Dataset Fallback

- Request: Continuous audit across 292 transcript files in `~/.gemini/antigravity/brain`, identify remaining friction points, implement features/fixes, test thoroughly, and merge into `main`.
- Completed:
  - **Built-in RandomForestModel Wrapper**: Implemented `RandomForestModel` in `harness/models.py` (registered under `random_forest`, `randomforest`, `rf`, `rf_classifier`, `rf_regressor`) providing out-of-the-box multi-threaded (`n_jobs=-1`) Random Forest classification, regression, probability outputs (`proba_<target>`), and parameter counting on tabular and vectorized text features.
  - **Extended Baseline Model Aliases & Lasso Regression**: Expanded `GenericBaselineModel.NAMES` to recognize `logreg`, `logistic`, `linear`, `linear_regression`, `linear_model`, `lasso`, and `lasso_regression`. Added native `Lasso` estimator support for regression with proper parameter filtering.
  - **TreeJournal Load Flexibility**: Updated `TreeJournal.load` in `harness/tree_journal.py` to seamlessly handle both instance invocations (`journal.load()`) and direct class calls (`TreeJournal.load(path_str)`), eliminating `AttributeError: 'str' object has no attribute 'storage_path'`. Added `TreeJournal.load_from_file(path)` classmethod.
  - **Cross-Environment / Portable Dataset Path Fallback**: In `UniversalTaskSpec.load_dataset_split` (`harness/task_spec.py`), added automatic basename fallback resolution in `root / p.name` and `root / "dataset" / p.name` when an absolute path from a foreign container or host (such as `/kaggle/working/dataset/train.csv` or a foreign Windows/Linux path) does not exist on the current machine.
- Evidence: 435/435 unit tests passed in 92.509s across the full repository test suite (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 4 dedicated test cases in `tests/test_agent_usability_p6.py`.
- Lesson: Providing flexible invocation signatures on persistence classes and portable fallback resolution on file paths ensures experiments transfer seamlessly across cloud and local runner boundaries.
- Next action: Create feature branch `feat/rf-and-portability`, push, open PR, and squash-merge into `main`.

## 2026-10-15 — Agent Usability Phase 7: Non-NLP Modalities (Tabular Robustness, Time-Series CV, Media Path Resolution)

- Request: Expand support for non-NLP problem modalities (Tabular, Time-Series / Sequential, Computer Vision, Audio, Multimodal) and address real-world failure patterns encountered by autonomous coding agents.
- Completed:
  - **RobustTabularTransformer**: Implemented `RobustTabularTransformer` in `harness/models.py` (and exported in `harness/__init__.py`), which median-imputes missing numeric features and ordinally encodes categorical strings with missing/unseen values safely mapped to `-1.0`. Defaulted `RandomForestModel` to `RobustTabularTransformer`, permanently fixing `ValueError: could not convert string to float` when fitting tabular datasets with mixed string and numeric columns.
  - **Native TimeSeriesSplit Cross-Validation**: Added `TIME_SERIES = "time_series"` and `TEMPORAL = "temporal"` to `SplitMode` in `harness/data_strategy.py`, along with parsing for aliases (`time_series`, `time_series_split`, `timeseries`, `temporal`). Added `strategy: "time_series"` branch in `UniversalRunner._resolve_validation_splits` (`harness/runner.py`), enabling leakage-free temporal forward validation splits for financial, forecasting, and sequential tabular problems.
  - **Runner Model Family Auto-Detection**: Added `random_forest` / `rf` detection to `UniversalRunner` so that `metrics.json` records `"model_family": "random_forest"`.
  - **Media & File Path Resolution**: Added `InputVariantSpec.resolve_media_paths(df, project_root, column, extensions)` in `harness/task_spec.py` with `media_dir` configuration support. Proactively resolves relative filenames in image/audio CSVs across `project_root`, `dataset/`, `images/`, `audio/`, and `/kaggle/input/` subdirectories with extension completion (`.jpg`, `.png`, etc.), ensuring CV and Audio pipelines resolve files seamlessly in local, Docker, and Kaggle environments.
- Evidence: Full test suite passes 100% (440/440 tests) in 106.103s (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 5 dedicated test cases in `tests/test_agent_usability_p7.py`.
- Lesson: For non-GBDT tabular estimators, tree-safe numeric encoding with `-1.0` unseen-category sentinel values prevents catastrophic string conversion failures while preserving tree partition boundaries; time-series cross-validation requires expanding-window splits where validation samples strictly succeed training partitions.
- Next action: Commit, push to `feat/non-nlp-modalities`, create PR #33, and squash-merge into `main`.

## 2026-10-15 — Agent Usability Phase 8: Universal Modality Expansion (Forecasting, Financial Splits, CV, Audio & Multilabel)

- Request: Expand framework across all competition paradigms (Financial / Tabular Sequential, Computer Vision, Audio / Speech, NLP Generation & Seq2Seq, Multilabel Classification) so agents never encounter missing metrics, improper splits, or validation barriers.
- Completed:
  - **Comprehensive Metric Expansion**:
    - **Time-Series / Financial**: Implemented `compute_rmsle`, `compute_smape`, `compute_wape`, `compute_pinball_loss`, `compute_pearson`, and `compute_mrr` with aliases (`rmsle`, `smape`, `wape`, `pinball`, `quantile_loss`, `medae`, `pearson`, `mrr`).
    - **Computer Vision**: Implemented `compute_dice_score`, `compute_iou_score`, `mask_to_rle`, `rle_to_mask` (with integer-float parsing), `compute_box_iou`, and `compute_map_50` with aliases (`dice`, `iou`, `bbox_iou`, `map_50`, `map50`).
    - **Speech & Audio**: Implemented `compute_wer` and `compute_cer` with aliases (`wer`, `cer`).
    - **NLP Generation & Seq2Seq**: Implemented `compute_bleu` (with Laplace smoothing for short sentences), `compute_exact_match`, `compute_token_f1`, and `compute_token_accuracy` with aliases (`bleu`, `exact_match`, `token_f1`, `token_accuracy`).
    - **Classification & Probabilities**: Added `balanced_accuracy`, `weighted_f1`, `mcc` (Matthews correlation), `pr_auc`, and `brier_score`.
    - **Direction Inference**: Implemented `default_metric_direction(metric)` automatically inferring minimize vs. maximize for all new metrics.
  - **Financial & Sequential Data Strategy**:
    - Implemented `PurgedGroupTimeSeriesSplit` (and `SplitMode.PURGED_GROUP_TIME_SERIES`) with configurable embargo buffer (`embargo_pct`), preventing lookahead and auto-correlation leakage across sequential trading sessions / dates.
  - **Multilabel Stratified Split**:
    - Implemented `MultilabelStratifiedKFold` (and `SplitMode.MULTILABEL_STRATIFIED_KFOLD`) providing greedy iterative multi-label stratification without external dependencies.
  - **Task Specification & Auto-Normalization**:
    - Defined `TASK_TYPE_NORMALIZATIONS` mapping aliases (`forecasting`, `ts_forecast`, `segmentation`, `detection`, `ner`, `asr`, `generation`, `seq2seq`, `retrieval`, `ranking`) to canonical task types.
    - Updated `SubtaskSpec.from_dict` to auto-normalize task types and infer default metric direction.
  - **Submission Validator Flexibility**:
    - Updated `SubmissionPreflightValidator` to accept non-numeric predictions (such as RLE masks, ASR text transcripts, and string class names) when `probability_mode=False`.
  - **Runner Split Integration**:
    - Added runner branches for `purged_group_time_series` / `financial_time_series` and `multilabel_stratified` / `multilabel_stratified_kfold` in `UniversalRunner._resolve_validation_splits`.
- Evidence: Full test suite passes 100% (449/449 tests) in 94.790s (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 9 dedicated unit tests in `tests/test_agent_usability_p8.py`.
- Lesson: Providing built-in domain metrics, anti-leakage purged financial splits, and string-tolerant submission validation enables autonomous agents to immediately tackle competitive tasks across CV, audio, NLP generation, and quantitative finance without writing fragile custom scoring loops.
- Next action: Stage, commit, push to `feat/universal-modalities-expansion`, create PR #34, and squash-merge into `main`.



