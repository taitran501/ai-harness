# Run Evidence and Correctness Contract (Schema v2)

This document specifies the invariants, evidence contracts, and verification gates enforced by the AI-Harness runtime. All execution backends, agents, and CLI tools must comply with this contract.

---

## 1. Execution Engine and Estimators

### Canonical Runner
- The canonical execution entry point for local and remote runs is `harness.runner` (`ai-harness-run`).
- The legacy script `train.py` delegates directly to `harness.runner`.

### Supported Built-in Estimators
- Built-in estimators include: `baseline`, `logistic_regression`, `ridge`, `tfidf_logistic_regression`, and `tfidf_ridge`.
- The `baseline` estimator resolves dynamically based on target task type (e.g., majority class for classification, mean value for regression).
- Any other model architecture requires a registered task plugin or an explicit model factory (`model.factory`).
- Invalid configuration parameters or unsupported generic metrics fail immediately instead of silently falling back to a default estimator or scorer.

---

## 2. Predictions and Scoring Invariants

### Prediction Batch Contract (`PredictionBatch`)
- Every prediction batch carries instance IDs, predicted labels, class probabilities, and declared class ordering.
- Probability-based metrics (such as `roc_auc` or `logloss`) strictly reject label-only inputs.
- Supported generic evaluation metrics: `accuracy`, `macro_f1`, `micro_f1`, `roc_auc`, `logloss`, `rmse`, `mae`, and `r2`.
- Custom plugin scorers retain their official aggregate score.
- Multi-task weighted score aggregation requires matching subtask metrics and optimization directions.
- Every declared target column must be completely evaluated across all instances.

### Primary Score vs. Utility
- `primary_score` records the exact mathematical value of the evaluated metric.
- `utility` normalizes the score such that higher is always better (negated for minimization metrics like RMSE or logloss).
- Model ranking, tree exploration, UCB recommendations, and candidate promotion strictly evaluate `utility`.
- Per-fold scores serve as diagnostic telemetry only. The authoritative primary score is computed globally on complete Out-Of-Fold (OOF) predictions.
- Tasks with multiple input variants require an explicit `data.selection_variant`.
- The task random seed is authoritative; if omitted, the runtime defaults to 42.
- Declared split and development dataset paths never silently fall back if missing.

---

## 3. Immutable Run Evidence and Manifest v2

### Directory Exclusivity
- Full validation runs claim `.ai-harness/artifacts/<task-id>/<exp_id>` exclusively.
- Re-running an experiment under an existing full-run ID is prohibited; new runs require a unique sequential experiment ID.
- The `--force` flag does not overwrite completed run evidence.
- Debug probes, single-fold checks, and sample probes write exclusively to `probes/<exp_id>/<run_id>` and never enter the official solution tree or candidate pool.

### Manifest v2 Specification
Every full run generates a cryptographic manifest recording:
- **Run Identity**: Experiment ID, run kind (`full` vs. `probe`), and terminal status.
- **Task Identity**: Task name, platform, and scorer specification.
- **Input Hashes**: SHA-256 hashes of the resolved configuration, training dataset, and validation split file.
- **Estimator Details**: Actual resolved estimator hyperparameters and feature pipeline settings.
- **Artifact Hashes**: SHA-256 hashes of generated artifacts (`metrics.json`, `oof_predictions.csv`, evaluation truth, and resolved config).

Consumers (such as the candidate pool and dashboard) reject artifacts with missing or mismatched manifest hashes. Historical artifacts lacking Schema v2 manifests remain visible as `LEGACY_UNVERIFIED` and cannot establish a baseline anchor or be promoted to candidate status.

---

## 4. Candidate Management and Ensembling

### Candidate Promotion
- Promoting an experiment to the candidate pool requires its exact `exp_id`.
- Candidate IDs are unique and retain source experiment IDs and manifest hashes.
- Ensemble source models must share identical task, scorer, dataset, and split identities, and must cover identical instance ID sets (1-to-1 joins).

### Blending and Voting Rules
- **Classification Voting**: Uses the declared task class order for deterministic tie-breaking.
- **Probability Blending**: Requires identical probability class ordering across all constituent models.
- **Regression Blending**: Operates directly on continuous out-of-fold predictions.
- All positive-weight targets must satisfy configured diversity thresholds (`harness/ensemble_guard.py`).
- An ensemble candidate must strictly outperform its best constituent under the declared validation protocol before publication.

---

## 5. Submission Packaging and Audit Gates

### Submission Contract
- Packaging a competition submission requires an active task plugin that implements both `submission_builder` and `submission_validator`.
- Submissions require a hashed `test_predictions.csv` explicitly tagged as `test` in the full-run manifest.
- **Out-of-fold validation predictions are never accepted as test submission predictions.**
- Generic declarative tasks lacking official submission hooks fail closed when packaging is attempted.

### Standalone Pre-Submission Audit (`ai-harness-audit`)
- Supports an explicit `--id-column` flag.
- Compares prediction columns against test ground specifications.
- Strictly rejects null values, duplicate row IDs, mismatched row counts, and invalid domain values (e.g., probabilities outside $[0, 1]$).

---

## 6. Governance, Concurrency, and Resource Budgets

### Approval Governance
- Dashboard endpoints for candidate promotion, candidate freezing, emergency unfreezing, cloud dispatch, and metric scoring delegate directly to the Agent Bridge.
- Explicit user approval (`--user-approved`) is mandatory for high-stakes actions: `freeze_candidate`, `prepare_submission`, and `emergency_unfreeze`.
- Candidate freezing validates candidate eligibility, unique instance keys, and allocated slot limits. Unfreezing requires an explicit audit reason.

### Resource Reservations and Budgets
- Compute resource reservations are recorded as durable JSON transactions, separated for training, probes, and candidate operations.
- Terminal resource accounting is idempotent.
- Failed runs count consumed resources; unverified GPU allocations remain null.
- A local wall-time budget terminates the running process tree if exceeded.
- Remote hard runtime budgets and hard GPU budgets are rejected for Kaggle and Modal because these backends do not provide verified immediate termination guarantees.
- Halting a local monitoring process never certifies remote job cancellation.

### State Concurrency and File Locks
- State files use reentrant thread locks combined with OS-level file locks (`msvcrt` on Windows, `fcntl` on Unix/Linux).
- State updates perform fresh disk reads and atomic file writes.
- File locks protect state updates exclusively and are released immediately after writing.
- Non-conflicting snapshot updates and append-only event logs merge automatically; conflicting scalar updates fail explicitly.
- Task configuration caching hashes file content.
- Network requests always verify TLS certificates; `HARNESS_CA_BUNDLE` allows specifying a custom trusted CA certificate file.

---

## 7. Verification Scope and Boundaries

- **NLP / Text Tasks**: Strictly verified and benchmarked across all runtime pipelines.
- **Non-NLP Modalities**: Tabular, Vision, Audio, and Multimodal support is experimental work-in-progress on the roadmap.
- **Feature Extraction**: Text metadata and TF-IDF feature pipelines are supported in `UniversalFeaturePipeline`. Probability calibration and decision threshold tuning are utility functions requiring explicit integration, not built-in runner models.
- **Test Integrity**: External provider calls in regression tests are strictly mocked. Live cloud verification runs require separately budgeted executions.

---

## 8. Research Memory Epistemic Contract

- Agent prose, including insight narratives, checkpoints, and declared decisions, is never machine-verified evidence.
- Internal metrics and model capabilities require references to validated immutable artifacts; research decisions require machine evaluation receipts.
- External source references establish provenance only. Claims from papers, discussions, or public leaderboards remain external and cannot establish local execution or local model performance.
- Current research state in the Context HUD and `RESEARCH_PROGRESS.md` is rendered from live validators. Ungrounded notes and legacy assertions remain in the append-only ledger but are quarantined from default resume context and active macro-family state.

## 9. Candidate Provenance and Dispatch Script Guard

- Full runs record `provenance_version: 1` and a provenance classification. Imported prediction values are always `external_prediction`, even when their score and file integrity are verified; they are not submission eligible and cannot be candidate constituents.
- Candidate recipes bind source IDs, source manifest hashes, and OOF/test prediction hashes through `recipe_binding_sha256`. Candidate construction and blend optimization revalidate provenance before reading predictions.
- `verified` describes artifact integrity and scoring. `submission_eligible` is a separate derived decision; `eligible` is only its compatibility mirror. Test predictions and a supported promotion decision are required before dispatch.
- `ai-harness audit-scripts` scans root Python files and `scratch/` for raw Kaggle submission and kernel push calls. Unacknowledged blocking findings lock submission dispatch with `DISPATCH_BACKDOOR_DETECTED`.
- User approval and diversity preflight receipts do not substitute for a supported decision evaluation. Submission payloads must resolve to a validated run or candidate manifest.

## 10. Citation Evidence and Admission Contract

- `ai-harness audit-claims` reports scan coverage, evidence integrity, claim support, and admission separately. A clean scan, keyword overlap, or an exact excerpt alone does not establish that a claim is supported.
- Strict audits require complete selected-document coverage, explicit claim-to-evidence bindings, source and locator validation, and a current review verdict. Empty or failed scans cannot produce a successful verification receipt.
- Discussion snapshots use explicit versioned schema adapters. Source identity, the snapshot hash, segment locator, excerpt offsets, and excerpt hash are bound into the audit/review records. Authors remain metadata and cannot count as message text.
- Legacy `source_refs` remain provenance declarations. They are unverified until the source is registered and the specific claim is reviewed. A grounded hypothesis may be eligible for a bounded probe; an external reported result does not establish local performance.
- Demonstrated local gain still requires the harness's machine-evaluated decision-rule receipt over comparable local evidence. Citation receipts do not replace promotion, diversity, backend-lock, stagnation, approval, or quota gates.
- Citation dependencies are revalidated at research, direct-run, candidate-pitch, submission-permit, and remote-dispatch boundaries. Subjects without external citation dependencies receive `not_applicable`.
- Source snapshots and audit receipts are stored outside immutable experiment directories. Receipts detect stale or edited evidence, but do not cryptographically authenticate an actor who can rewrite the complete workspace.
