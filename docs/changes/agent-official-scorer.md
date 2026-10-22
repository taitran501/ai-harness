# Preserve the official scorer throughout candidate selection

- **Date**: 2026-10-22
- **PR**: [#118](https://github.com/taitran501/ai-harness/pull/118)
- **Branch**: `fix/agent-official-scorer`
- **Status**: open

## Why

An agent using `paths.metrics_evaluator` could complete full validation and lock its baseline, then fail promotion because the candidate builder invoked the generic scorer. Blend optimization had the same divergence. A valid custom competition metric must stay authoritative after training.

## What changed

Runner, repair revalidation, candidate construction and blend optimization share `score_predictions`. Candidate sources must agree on custom-scorer identity as well as task, data and split identities. Optimizer sources are validated before scoring.

## Approach

Reuse the existing evidence-revalidation scoring path rather than adding a second hook dispatcher. Keep generic scorer identities unchanged and retain the runner's preflight skip and first-failure warning behavior. Source configs remain immutable.

## Evidence

Regression tests exercise real runner output -> Bridge promotion, custom-metric optimization, and rejection of individually valid sources with different scorer implementations. `python tools/run_tests.py`: 1039 tests, 2 skipped, no failures/errors. Ruff and diff whitespace checks pass.

## Impact & follow-ups

Custom-hook runs can enter candidate selection without replacing their metric with a generic proxy. External import metadata and script prediction transport are addressed separately.
