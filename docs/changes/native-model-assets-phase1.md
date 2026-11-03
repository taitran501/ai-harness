# Native Model Assets and Governance Phase 1

- **Date**: 2026-11-03
- **PR**: Working tree
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: `— (not committed)`

## Why

AI-Harness needs a bounded, native representation of model components and a
pure policy evaluator that counts deployed logical parameters rather than only
trainable adapter parameters. The legacy built-in model-name registry is a
separate fast import path and must keep its API and cache behavior.

## What changed

- Added typed model definitions, canonical definition digests, structured
  policy assessments, and competition rule normalization in
  `harness/model_assets/`.
- Added model and composition checks for pinned references, pretrained/API
  restrictions, full parameter totals, release cutoffs, and license allowlists.
- Added a YAML/JSON definition store with a 16 KiB file cap, strict field
  validation, alias/duplicate-key rejection, digest verification, and atomic
  content-addressed registration.
- Added non-mutating competition and model-governance accessors to
  `UniversalTaskSpec` and documented the separation from the lightweight
  built-in name registry.
- Preserved legacy scorer identity when additive governance accessors are
  added to `task_spec.py`.
- Added focused governance and registry test suites.

## Approach

Definitions normalize to a fixed schema before hashing or storage. Unknown
fields fail closed so bulky or unsupported metadata cannot enter the model
object catalog. Policy evaluation is a pure function over typed rules and
model facts; an unknown parameter count blocks a decision whenever an
applicable cap exists. Composition checks sum the deployed component closure.
The task-spec identity helper strips only the two additive policy properties
from the source string used by the legacy scorer digest, preserving the exact
pre-change identity for scoring code.

## Evidence

Pending final focused tests, full pytest suite, smoke suite, and ruff checks.

## Impact & follow-ups

Phase 1 provides definitions, storage, and pure policy evaluation. Execution
planning, runtime attestation, candidate binding, and backend enforcement
remain later milestones.
