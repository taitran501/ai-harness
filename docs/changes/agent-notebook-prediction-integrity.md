# Stop Kaggle notebooks from fabricating or misaligning submissions

- **Date**: 2026-10-23
- **PR**: [#125](https://github.com/taitran501/ai-harness/pull/125)
- **Branch**: `fix/agent-notebook-prediction-integrity`
- **Status**: open

## Why

An agent following `export-kaggle-notebook` could execute the exported notebook without any prediction artifact and receive a SUCCESS message plus a submission filled with 0.5. When a real artifact existed, values were copied by position under test IDs, silently changing their meaning if rows were shuffled. Missing target columns also caused internal metadata to be copied into the submission.

## What changed

Generated notebooks require a real prediction CSV and exact, unique, non-null test IDs, align predictions by ID, select only the requested target or its `pred_` alias, and reject missing/ambiguous artifacts and nonfinite numeric output before writing. Explicit test-path and wheel-pattern arguments are honored. Failed offline dependency installation propagates instead of printing success.

## Approach

Keep this exporter a standalone assembler for precomputed scalar predictions. State that model-specific checkpoint inference must be implemented separately; never synthesize replacement predictions or IDs. Prefer attached `test_predictions.csv` over packaged submissions, allow local artifact verification, and reject ambiguous attached inputs instead of choosing an arbitrary file.

## Evidence

Before the fix, executing generated cells with no artifact produced `{'id': [2, 1], 'target': [0.5, 0.5]}` and printed SUCCESS. New tests execute the generated notebook code, covering shuffled predictions, explicit dataset selection, malformed/absent IDs, missing targets, nonfinite values, ambiguous artifacts and failed wheel installation. Full suite: 1,062 tests, 2 skipped, no failures/errors; Ruff and diff whitespace checks pass.

## Impact & follow-ups

Missing inference now stops at an actionable error. Competitions requiring fresh hidden-test inference still require the agent's model-specific inference implementation; this exporter does not claim to supply one. Existing completed evidence is unchanged.
