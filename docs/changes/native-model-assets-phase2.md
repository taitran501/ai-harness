# Native Model Execution Plan and Pre-Run Admission (Phase 2)

- **Date**: 2026-11-03
- **PR**: Working tree
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: `— (not committed)`

## Why

Phase 1 added typed model definitions and policy evaluation, but execution paths
did not resolve the same inherited config before local and cloud dispatch.
Parameter limits were checked after fitting and counted trainable weights, while
the Kaggle dry-run could upload its dataset before returning a dispatch plan.
The native `dispatch-kernel` entry point also had no task-scoped model admission
check.

## What changed

- Added `harness/model_assets/resolution.py` with a shared effective-config
  resolver, immutable `ModelPlan`, portable `model_lock.json` data, policy
  receipts, and `ModelAdmissionError`.
- Integrated pre-run model admission in `runner.main()`, `UniversalRunner`,
  cloud preparation, and the native kernel dispatch routes.
- Persisted `model_lock.json` and `model_policy_receipt.json` in local run
  artifacts. Required governance rejects missing, invalid, unknown, or
  over-limit model declarations before plugin/model setup or training.
- Passed explicit base-config paths through Modal and Kaggle staging so remote
  execution inherits the same base config as local execution.
- Moved Kaggle dry-run return ahead of dataset upload and kernel/code dispatch.
- Split validator APIs into trainable-budget and logical-total counts. The
  legacy `max_params_limit` remains a trainable budget; registry policy counts
  require a logical-total declaration and fail closed when that fact is absent.
- Added execution-plan, pre-run rejection, direct kernel admission, validator,
  and cloud dry-run regression tests.

## Approach

The resolver performs one recursive base/experiment merge for every backend.
Selected `model_assets.roots` resolve through the task's bounded catalog or a
content-addressed definition, and the resulting lock binds definition digests
to the task policy digest. Local and cloud boundaries evaluate the same plan
and only required governance blocks execution; legacy runs still receive an
explicit unverified lock and receipt. Competition restrictions without an
explicit governance mode produce a migration blocker instead of being ignored.

Admission occurs before plugin loading, model construction, provider uploads,
or kernel pushes. Cloud dry-run still builds the local package plan, but returns
before any dataset or code upload and before kernel dispatch.

## Evidence

Pending final full-suite and Ruff verification.

## Impact & follow-ups

Declared definitions and policy are now bound to each run, but this phase does
not attest that unrestricted scripts or plugins loaded only the declared
weights, nor does it resolve or hash model resource bytes at runtime. Runtime
attestation and backend resource verification remain later milestones.
