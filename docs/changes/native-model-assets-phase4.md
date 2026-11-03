# Native Model Candidate Lineage and Dispatch Binding (Phase 4)

- **Date**: 2026-11-03
- **PR**: Working tree
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: `— (not committed)`

## Why

Run-level model admission did not carry a complete model closure into an
ensemble candidate or bind that closure to preflight and dispatch evidence.
Two individually compliant models could therefore exceed an aggregate
competition cap after blending, and a preflight receipt described predictions
without identifying the exact model subject and policy that produced them.

## What changed

- Added `harness/model_assets/lineage.py` with a content-addressed
  `CandidateSubject`, source-lock closure resolution, and current-policy
  evaluation for individual and aggregate deployed models.
- Candidate construction now records source-manifest and model-lock bindings,
  evaluates the full constituent closure, and rejects over-cap or stale-policy
  compositions. Candidate validation recomputes those bindings.
- Candidate preflight now writes schema v3 receipts with subject, aggregate
  model-lock, and model-policy digests plus compliance status. Readers continue
  to validate v1 and v2 receipts.
- SubmissionGate recomputes model compliance from sealed run evidence, checks
  cited v3 receipt bindings before authorization, and records the same model
  identity in the durable single-use dispatch intent. Required governance
  rejects legacy receipts without the new bindings.
- Added `--source-exp-id` for preflight inputs stored outside their source run,
  and regression tests for aggregate caps, v3 bindings, legacy reads, and
  permit binding.

## Approach

Candidate composition preserves one inference member per source run instead
of deduplicating equal model definitions: blending two runs deploys two model
instances. Locks and policy receipts are checked against the source artifact
manifest, definitions are parsed through the typed registry model, and current
competition rules are evaluated again over the complete candidate closure.
Candidate subjects bind task, recipe, source-manifest identities, model locks,
policy identity, lineage, and normalized prediction semantics. Raw prediction
and outbound payload hashes remain separate.

Schema v3 is additive. Existing v1/v2 preflight documents remain readable;
required model governance accepts them only when no new binding is required,
and governed dispatch requires a matching v3 subject, lock, policy, and
eligible compliance status.

## Evidence

Pending full pytest and Ruff verification.

## Impact & follow-ups

Legacy or external prediction files can still receive a v3 preflight receipt,
but remain explicitly unverified until tied to a source run. Callers can pass
`--source-exp-id` to bind an exported submission to validated source evidence.
Runtime attestation of undeclared model loads remains outside this phase.
