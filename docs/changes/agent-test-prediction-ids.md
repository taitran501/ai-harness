# Align custom test predictions by sample identity

- **Date**: 2026-10-22
- **PR**: [#120](https://github.com/taitran501/ai-harness/pull/120)
- **Branch**: `fix/agent-test-prediction-ids`
- **Status**: open

## Why

A model could correctly export `(id, prediction)` rows in batch/sort order, but script and pipeline adapters discarded those IDs and attached values to test-dataset row positions. Corrupted predictions still received valid hashes and test-evidence tags.

## What changed

Both adapters align test output to the declared dataset by exact one-to-one IDs before extracting labels, probabilities and class metadata. Missing, extra, duplicate or null IDs fail closed.

## Approach

Centralize identity alignment in the prediction-transport helper. Preserve every output value with its sample ID; never guess positional correspondence.

## Evidence

Subprocess tests reverse test row order in both execution modes and check final IDs, labels, probabilities and sealed evidence. Negative tests cover malformed ID sets. Full-suite and Ruff results appear in the PR.

## Impact & follow-ups

Custom scripts must provide test IDs, including when their row count matches the dataset. Completed artifacts are untouched.
