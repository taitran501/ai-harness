# Architectural hardening and boundary consolidation

- **Date**: 2026-11-14
- **PR**: #137 (host review pending)
- **Branch**: `fix/harness-invariants-and-audit-hardening`
- **Status**: open, uncommitted working tree
- **Commits**: None; no commit or push was performed.

## Why

Framework lookup, state loading, prediction transport, validation scope and artifact
publication had different acceptance rules at adjacent boundaries. An invalid
explicit path could fall back; a transaction could reset corrupt state; test
predictions could lose identity; selected folds could define their own full-run
coverage; provider import could partially populate or remove a destination. The
HUD could hide blocking policy behind an unbounded diagnostic dump.

## What changed

1. Framework resolution is shared and fail closed for explicit CLI/environment
   settings. Sync also checks package and Git/worktree markers. Pin-disable flags
   no longer conceal bad configuration.
2. Authoritative JSON reads distinguish absence from unreadable, malformed,
   incorrectly encoded, invalid-schema and unsupported-version state. Transactions
   validate before yielding and before persistence; session snapshots, competition
   state and research append paths cannot reset corruption. Legacy session IDs and
   supported scope-history forms remain explicit migrations.
3. Prediction payload checks work without labels, retain explicit IDs, align
   persisted frames only after exact identity checks, validate supplied probability
   domains and class order, and protect candidate sources and test partitions.
   Bootstrap declares classification classes from training data even when the task
   type was supplied explicitly. Nonfinite estimator metadata uses explicit tagged
   JSON values without relaxing numeric score/prediction checks. Remote dispatch
   ignores task-config environment overrides owned by other workspaces. External predictions retain an inspectable
   external run kind and cannot supply verified full-run evidence.
4. Required folds and IDs are resolved separately from execution selection and
   cached across adapters. Runs seal partitions plus observed scope. Partial runs
   are probes; full admission and remote/comparison reservations require complete
   scope. Legacy runs without sealed scope fail admission without rewriting them.
5. Local completion, Modal attached returns/recovery and Kaggle imports use one
   publisher. It validates full file inventory, safe paths, fresh hashes, metrics,
   predictions and scope; copies privately; revalidates; and uses an atomic
   no-replace rename under an external lock. Exact manifest/file bytes are
   idempotent; conflicting or empty destinations remain untouched. Force import
   cannot override validation. Local execution reservations prevent rerunning an
   existing experiment; post-publication indexing failure retains completed data.
6. Default Markdown HUD rendering reserves sections in objective/blocker/incumbent/
   result/action order and caps output at 4,000 characters. Huge diagnostics are
   bounded with omission markers; JSON and inspection actions retain detail.
   Fatal authoritative-state errors still propagate.

## Approach

Implemented the five blueprint phases in dependency order. Shared readers,
validators and publication operations replace boundary-specific permissive
behavior. Existing production admission interfaces remain fail closed. Synthetic
fixtures explicitly seal scope and utility instead of weakening validation to
accept incomplete legacy fixtures. Storage publication creates independent final
copies; byte equality remains guaranteed, while hardlink identity to mutable
canonical/staging sources is intentionally no longer guaranteed.

## Evidence

All provider transports are mocked. Real Windows filesystem publication,
existing-directory rejection and multiprocess identical/conflicting publication
were exercised. No live cloud training/submission or production quality is claimed.

| Command / gate | Result | Elapsed |
| --- | --- | --- |
| `ruff check harness/ tools/ tests/` | Clean | <1 s |
| `python tools/run_tests.py --smoke` | 79 tests, no failures/skips | 7.4 s |
| `python tools/run_tests.py --fast` | 1,084 tests, no failures, 2 existing skips | 214.4 s |
| Affected slow modules: cloud runners, Kaggle runtime, sync, tree journal | 86 tests, no failures/skips | 95.6 s |
| Final focused regressions: citation, pipeline E2E, publication/provider and audit | 81 tests, no failures, 1 existing skip | 30.3 s |
| `python tools/run_tests.py` | 1,258 tests, no failures, 2 existing skips | 342.7 s |
| Final publication/audit gate, including custom-plugin recovery | 24 tests, no failures/skips | 10.4 s |

The smoke tier grew from 49 tests / 3.3 seconds to 79 tests / 7.4 seconds, after
measuring the new representatives. Multiprocess publication and provider transport
integration remain outside smoke. The final isolated custom-plugin resolver branch was verified in the 24-test
publication/audit gate after the broad suites. Verification logs remain locally under ignored
`scratch/`; the final verification summary is `scratch/boundary-verification.json`.

Linux runtime verification was unavailable: `wsl.exe --list --quiet` failed with
`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`. Linux uses
`renameat2(RENAME_NOREPLACE)` and fails explicitly if unavailable. There is no
verified Linux execution result from this checkout; the portable filesystem tests
remain a Linux acceptance gate for the host/CI.

## Tests added

33 new test methods across eight modules:

| Module | Methods | Contracts exercised |
| --- | ---: | --- |
| `tests/unit/test_framework_resolution.py` | 2 | Invalid explicit sources, pin-disable behavior, relative paths and worktree parity |
| `tests/unit/test_state_integrity.py` | 8 | Strict JSON/schema/encoding reads, transactions, stale snapshots, legacy IDs, named sessions and research corruption |
| `tests/integration/test_prediction_boundaries.py` | 5 | Exact in-memory IDs, persisted shuffle, ID type preservation, test probability domains and shared wrong candidate test IDs |
| `tests/integration/test_validation_scope.py` | 6 | Required vs selected folds, task-declared subsets, duplicates/overrides, forged full tags and post-publication indexing failure |
| `tests/unit/test_artifact_publication.py` | 7 | Exact retries, conflicts/empty destinations, fresh hashes, unsafe inventories, copy failures, rename races and reparse-point rejection before descent |
| `tests/integration/test_artifact_publication.py` | 1 | Real multiprocess identical and conflicting publishers |
| `tests/integration/test_provider_publication.py` | 2 | Shared Kaggle/Modal publication, custom-plugin scorer identity, exact retries and conflict preservation |
| `tests/unit/test_context_rendering_budget.py` | 2 | Megabyte diagnostics, thousands of records, Unicode, deterministic rendering, policy priority and exact budgets |

Existing runner, prediction/scoring, provider, evidence, repair, research,
adoption and synthetic-fixture tests were updated to assert the new contracts.

## Impact & follow-ups

Invalid explicit paths, corrupt state, missing/misaligned IDs, undeclared/reversed
class order, non-normalized single-label probabilities, unverified validation
scope, unsafe payloads and conflicting imports now fail explicitly. Correct the
source/configuration or choose a new experiment identity; do not reset state or
re-seal historical artifacts to bypass the failure. Unsupported atomic-rename
platforms/filesystems fail explicitly. Archived `docs/WORK_LOG.md` is unchanged.

No branches were switched or created. All changes remain in this repository for
host review, commit, push and the PR #137 update. Linux filesystem execution is
still required before claiming cross-platform runtime verification.

## Changed-file inventory

58 changed or new delivery files. Verification logs and generated caches are ignored.

### Runtime and adapters (25)

- [`harness/agent_bridge.py`](../../harness/agent_bridge.py)
- [`harness/artifacts.py`](../../harness/artifacts.py)
- [`harness/bridge_actions.py`](../../harness/bridge_actions.py)
- [`harness/bridge_common.py`](../../harness/bridge_common.py)
- [`harness/candidates.py`](../../harness/candidates.py)
- [`harness/cloud_job_runner.py`](../../harness/cloud_job_runner.py)
- [`harness/cloud_runner.py`](../../harness/cloud_runner.py)
- [`harness/competition_state.py`](../../harness/competition_state.py)
- [`harness/decision_rules.py`](../../harness/decision_rules.py)
- [`harness/evaluation.py`](../../harness/evaluation.py)
- [`harness/oof_revalidation.py`](../../harness/oof_revalidation.py)
- [`harness/provenance.py`](../../harness/provenance.py)
- [`harness/research_memory.py`](../../harness/research_memory.py)
- [`harness/run_data.py`](../../harness/run_data.py)
- [`harness/runner.py`](../../harness/runner.py)
- [`harness/script_predictions.py`](../../harness/script_predictions.py)
- [`harness/state_store.py`](../../harness/state_store.py)
- [`harness/submission_manager.py`](../../harness/submission_manager.py)
- [`harness/sync.py`](../../harness/sync.py)
- [`harness/task_spec.py`](../../harness/task_spec.py)
- [`harness/templates/kaggle_template.py`](../../harness/templates/kaggle_template.py)
- [`harness/templates/modal_template.py`](../../harness/templates/modal_template.py)
- [`harness/threshold_data.py`](../../harness/threshold_data.py)
- [`harness/toolchain.py`](../../harness/toolchain.py)
- [`harness/validation_scope.py`](../../harness/validation_scope.py)

### Tools (2)

- [`tools/bootstrap_task.py`](../../tools/bootstrap_task.py)
- [`tools/run_tests.py`](../../tools/run_tests.py)

### Tests and synthetic evidence fixtures (23)

- [`tests/e2e/test_cloud_runners.py`](../../tests/e2e/test_cloud_runners.py)
- [`tests/e2e/test_pipeline_e2e.py`](../../tests/e2e/test_pipeline_e2e.py)
- [`tests/e2e/test_sync_e2e.py`](../../tests/e2e/test_sync_e2e.py)
- [`tests/evidence_fixtures.py`](../../tests/evidence_fixtures.py)
- [`tests/fixtures.py`](../../tests/fixtures.py)
- [`tests/integration/test_artifact_publication.py`](../../tests/integration/test_artifact_publication.py)
- [`tests/integration/test_audit_hardening.py`](../../tests/integration/test_audit_hardening.py)
- [`tests/integration/test_candidate_selection.py`](../../tests/integration/test_candidate_selection.py)
- [`tests/integration/test_evidence_sealing.py`](../../tests/integration/test_evidence_sealing.py)
- [`tests/integration/test_framework_adoption.py`](../../tests/integration/test_framework_adoption.py)
- [`tests/integration/test_prediction_boundaries.py`](../../tests/integration/test_prediction_boundaries.py)
- [`tests/integration/test_provider_publication.py`](../../tests/integration/test_provider_publication.py)
- [`tests/integration/test_repair_integration.py`](../../tests/integration/test_repair_integration.py)
- [`tests/integration/test_research_continuity.py`](../../tests/integration/test_research_continuity.py)
- [`tests/integration/test_runner.py`](../../tests/integration/test_runner.py)
- [`tests/integration/test_submission_manager.py`](../../tests/integration/test_submission_manager.py)
- [`tests/integration/test_validation_scope.py`](../../tests/integration/test_validation_scope.py)
- [`tests/unit/test_artifact_publication.py`](../../tests/unit/test_artifact_publication.py)
- [`tests/unit/test_context_rendering_budget.py`](../../tests/unit/test_context_rendering_budget.py)
- [`tests/unit/test_evaluation_contracts.py`](../../tests/unit/test_evaluation_contracts.py)
- [`tests/unit/test_evidence_manifest.py`](../../tests/unit/test_evidence_manifest.py)
- [`tests/unit/test_framework_resolution.py`](../../tests/unit/test_framework_resolution.py)
- [`tests/unit/test_state_integrity.py`](../../tests/unit/test_state_integrity.py)

### Documentation (8)

- [`docs/CORRECTNESS_CONTRACT.md`](../../docs/CORRECTNESS_CONTRACT.md)
- [`docs/DESIGN_single_install.md`](../../docs/DESIGN_single_install.md)
- [`docs/TASK_SPEC_TEMPLATE.md`](../../docs/TASK_SPEC_TEMPLATE.md)
- [`docs/agent_workflow.md`](../../docs/agent_workflow.md)
- [`docs/changes/README.md`](../../docs/changes/README.md)
- [`docs/changes/boundary-consolidation.md`](../../docs/changes/boundary-consolidation.md)
- [`docs/kaggle_execution.md`](../../docs/kaggle_execution.md)
- [`docs/plugin_architecture.md`](../../docs/plugin_architecture.md)

