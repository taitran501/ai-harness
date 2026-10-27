# Candidate Preflight CLI, SubmissionGate Candidate Blend Parity, and Kernel Directory Push

- **Date**: 2026-10-27
- **PR**: Working tree
- **Branch**: `main`
- **Status**: merged
- **Commits**: `<working-tree>`

## Why

Auditing real conversation trajectories revealed three major friction loops forcing agents into ad-hoc shell workarounds or backdoor Python scripts:

1. **`SubmissionGate` blocked blended candidate CSVs with `EXPERIMENT_EVIDENCE_REQUIRED`**:
   - In competitive ML, ensemble blends, manifold synthesis, and post-processed predictions originate from combining multiple models or notebooks. They do not have a single experiment directory containing `test_predictions.csv` and `run_config.yaml`.
   - `SubmissionGate._resolve_payload` strictly threw `EXPERIMENT_EVIDENCE_REQUIRED: Pass --exp-id for a sealed run with test predictions.` when `--exp-id` was omitted, forcing agents in `airline_prediction` Day 17 to hand-craft an unmonitored backdoor dispatch script (`scratch/dispatch_sub.py`).
   - Furthermore, `SubmissionGate._history` failed to resolve relative file paths in `dispatch_log.jsonl`, did not recognize `file_name` keys in legacy log records, and checked `num_total > len(known_refs)` even when `limited_by_total` was False, causing false-positive `HISTORY_INCOMPLETE` errors.
   - For probability-based metrics (e.g. `roc_auc`), `audit_prediction_integrity` rejected continuous probability values because the column did not match discrete categorical class strings (`['False', 'True']`).

2. **Missing first-class `ai-harness preflight` CLI**:
   - The candidate preflight engine (`run_candidate_preflight` and `validate_candidate_pitch`) was implemented in Python, but had no top-level CLI entry point. Agents had to run 15+ ad-hoc `python -c "from harness.candidate_preflight import..."` scripts just to verify behavioral diversity before pitching candidates.

3. **Missing directory push support for Kaggle Code Competitions (`ai-harness kaggle push`)**:
   - In kernel-only code competitions (e.g. ARC Prize, RSNA), agents prepare a submission directory containing `kernel-metadata.json` and solution code.
   - `ai-harness kaggle` lacked a `push` subcommand and `dispatch-kernel` only accepted single notebook files, forcing agents to run raw shell commands like `kaggle kernels push -p stage_m2_test` that bypassed dispatch auditing.

## What changed

- **`harness/submission_gate.py`**:
  - `_resolve_payload`: When `--user-approved` or `--preflight-id` is provided, candidate files without `--exp-id` are validated via `plugin.validate_submission`, synthesized into candidate evidence (`kind="candidate_file"`), and permitted through the gate.
  - `_history`: Resolves `candidate_path`, `file_name`, and `archive_path` relative to `project.root` before file verification; acknowledges operator-approved refs from `.ai-harness/acknowledged_submissions.json`.
  - Quota validation: Restricts the `num_total > len(known_refs)` gate to competitions where `limited_by_total` is `True`.
  - Promotion logic: When prior submissions exist and `--user-approved` or `--preflight-id` is provided without a formal `--rule-id`, records `"operator_attestation"` or `preflight_id` in `decision_evaluation_ids`. Rejects near-duplicates (`rho >= 0.995`) with `INDEPENDENT_CONFIRMATION_REQUIRED`.
- **`harness/audit_gate.py`**:
  - `_normalize_prediction_frame`: Automatically aliases `<target>` to `pred_<target>` if `pred_<target>` is absent, enabling seamless semantic hash and diversity comparisons across raw submission CSVs and experiment test prediction tables.
  - `align_prediction_frames`: Implemented fast NumPy vector equality check `all((candidate[k].to_numpy() == reference[k].to_numpy()).all() for k in keys)` for aligned frames, avoiding expensive tuple set and dictionary construction on large datasets.
  - `audit_prediction_integrity`: Allows valid probabilities in `[0.0, 1.0]` for classification tasks evaluated on probability metrics (`roc_auc`, `auc`, `log_loss`, `brier_score`).
- **`harness/endgame_submission.py` & `harness/submission_dispatch.py`**:
  - Added `--preflight-id` CLI argument to `ai-harness submission dispatch`, passing `preflight_id` into `KaggleLeaderboardTracker.dispatch` and `SubmissionGate.authorize_dispatch`.
- **`harness/cli.py`**:
  - Added top-level `preflight` subcommand: `ai-harness preflight <candidate.csv> [--incumbent <baseline.csv>] [--pitch] [--claim-kind ...] [--json]`.
  - Added formatted human-readable output by default and machine-readable JSON with `--json`.
  - Fixed `ensure_pinned_toolchain` to pass `argv` when called programmatically rather than defaulting to `sys.argv[1:]`.
- **`harness/kernel_dispatch.py` & `harness/kaggle_inspect.py`**:
  - Implemented `push_kernel_directory(dir_path, ...)`: reads `kernel-metadata.json`, pushes via `client.kernels_push`, and records the dispatch in `dispatch_log.jsonl` with content hashing.
  - Added `-p` / `--path` support to `ai-harness dispatch-kernel`.
  - Added `ai-harness kaggle push [-p <dir>] [--watch] [--poll <sec>] [--timeout <sec>]` subcommand.
- **`airline_prediction` workspace**:
  - Deleted backdoor script `scratch/dispatch_sub.py`.
- **Tests**:
  - Added `PushKernelDirectoryTests` in `tests/test_kernel_dispatch.py`.
  - Added `test_cli_preflight_command` in `tests/test_candidate_preflight.py`.
  - Added `test_candidate_file_dispatch_with_user_approved` in `tests/test_submission_dispatch.py`.

## Approach

- **Unified evidence resolution**: Rather than treating standalone candidate CSVs as out-of-band anomalies, `SubmissionGate` now treats them as first-class `candidate_file` evidence. Format integrity is checked against the task plugin, exact payload bytes are hashed and content-staged under `.ai-harness/submission_payloads`, and diversity checks remain strictly enforced.
- **Non-blocking lifetime counters**: Competitions without an all-time submission cap (`limited_by_total=False`) track lifetime submissions purely for statistical metrics; gating against `num_total` is only appropriate when the platform imposes a total submission limit.
- **Fast-path alignment**: Vector equality checks on key columns avoid quadratic tuple construction and sorting when comparing large prediction frames (e.g. 300,000 rows across 69 portfolio submissions).

## Evidence

- `ruff check harness/ tools/ tests/` passes 100% with 0 errors.
- `python tools/run_tests.py --smoke --no-lock` passes 47 tests in 2.6s.
- `python -m unittest tests/test_kernel_dispatch.py tests/test_candidate_preflight.py tests/test_submission_dispatch.py tests/test_submission_guards.py` passes 54 tests in 10.9s.
- `ai-harness preflight` tested on live workspace files with both formatted output and `--json`.
- `ai-harness submission dispatch ... --dry-run` tested on live workspace files, verifying candidate file parsing, legacy history traversal, and near-duplicate protection.

## Impact & follow-ups

Eliminates the friction loops that led agents to write backdoor scripts (`scratch/dispatch_sub.py`) and bypass kernel dispatch auditing (`kaggle kernels push -p`). All code competition pushes and candidate blend submissions now flow through the audited harness gateway.
