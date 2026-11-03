# Native Model Resource Resolution and Backend Preservation (Phase 3)

- **Date**: 2026-11-03
- **PR**: Working tree
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: `— (not committed)`

## Why

Phase 2 binds a run to a governed model definition and policy decision, but it
does not identify the bytes that a backend should mount. Cloud jobs could
therefore start with a model lock while silently relying on a mutable cache or
an undeclared download. Remote artifact filters also needed to retain the
model evidence used to explain each run.

## What changed

- Added `BackendResourcePlan` and `ModelResolver` for Local, Kaggle Dataset, and
  Modal Volume paths, along with manifest-based file size and SHA-256 checks.
- Added deterministic local cache and pinned Hugging Face snapshot lookup. The
  resolver does not fetch assets; offline plans fail when mounts, manifests, or
  expected weight files are missing or changed.
- Passed backend resource plans through cloud preparation. Kaggle metadata
  unions model Dataset sources with task/code Dataset sources, and its runtime
  verifies the mounted files before starting the harness runner.
- Added a dedicated Modal model Volume mount at `/vol/models` for declared
  resources, with startup verification and an exclusion for workspace model
  cache bytes from the container code image.
- Kept sealed model lock, policy receipt, loaded model, output, and lineage
  records in Kaggle exports and Modal artifact bundles.
- Added focused Local, Kaggle, Modal, offline failure, serialization, and export
  tests.

## Approach

The portable `ModelPlan` remains separate from backend paths. The new resource
plan translates each locked root into a backend mount and carries the selected
source identity plus manifest file records. Verification hashes each expected
file directly instead of using the general artifact hash cache. Planning and
verification perform no downloads. Kaggle model sources enter
`dataset_sources`, because those are Kaggle Datasets that must appear under
`/kaggle/input`; the Kaggle `model_sources` metadata field remains empty.
Modal mounts a separately named model Volume only when resources are declared.
Both remote runners verify mounted bytes before invoking model code.

## Evidence

Pending the requested full pytest and Ruff runs.

## Impact & follow-ups

Model artifact manifests must provide relative paths, byte counts when known,
and SHA-256 digests. Kaggle locations must name a Dataset, and Modal locations
must point into a provisioned model Volume. This phase verifies existing
resources but does not upload model bytes or add a model cache-fetch command.
