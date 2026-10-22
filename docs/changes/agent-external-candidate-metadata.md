# Normalize imported prediction artifacts for candidate selection

- **Date**: 2026-10-22
- **PR**: [#122](https://github.com/taitran501/ai-harness/pull/122)
- **Branch**: `fix/agent-external-candidate-metadata`
- **Status**: merged
- **Merge commit**: `27cd1d87c843c2ce7629cf381ec407301e9e01b3`

## Why

`register_external_prediction` accepted minimal CSV/NumPy predictions and produced verified evidence, but promotion failed because candidate readers required `input_variant`. Raw target aliases and probability class ordering also needed canonical transport.

## What changed

Imported managed OOF always uses `input_variant: external`, canonical `pred_*` target columns and a fold column. Existing exporter fold assignments are preserved; absent fold metadata is explicitly unknown (-1). Shared prediction parsing retains declared probability class order for scoring.

## Approach

Normalize only the newly created managed artifact, leaving the original external file unchanged. Do not forge CV fold assignments or mutate completed runs.

## Evidence

Bridge integration tests register and promote shuffled minimal CSVs, NumPy exports, and classification probabilities with reversed class order. Tests check scorer equality, sealed evidence, preserved source hashes and fold metadata. Full-suite and Ruff results appear in the PR.

## Impact & follow-ups

Import success now produces a candidate-readable artifact even when an external exporter knows nothing about harness-internal metadata. External provenance remains explicitly labeled.

## Pre-merge review

Reviewed registration -> immutable run validation -> AgentBridge promotion for shuffled CSV target aliases, NumPy exports and reversed probability class order. Managed metadata is normalized without rewriting the source export or completed evidence. No blocking findings.

A fresh integration run passed: 1,053 tests, 2 skipped, no failures/errors. Ruff and diff whitespace checks passed.
