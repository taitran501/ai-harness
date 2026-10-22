# Derive threshold recommendations from the declared binary evidence

- **Date**: 2026-10-23
- **PR**: [#126](https://github.com/taitran501/ai-harness/pull/126)
- **Branch**: `fix/agent-threshold-evidence`
- **Status**: open

## Why

An agent could run `optimize_threshold` on a perfect binary classifier and receive accuracy 0.5 when the model stored probabilities in reversed class order. A different raw target-column name made the same action fail to find probabilities. The handler also combined all OOF input variants, silently median-binarized regression targets, and labeled an F1 search as any unsupported requested metric.

## What changed

Threshold input loading validates completed source evidence, reads only its selected variant, aligns OOF IDs with truth, resolves prediction columns by subtask ID, and selects the last declared class's probability by label. String labels work without median conversion. Observations identify the target, positive class, selected variant and sample count. Full-precision thresholds are returned.

The search rejects unsupported metrics, non-binary labels and invalid probabilities, and always includes the 0.5 baseline even for a bounded grid. Agent guidance explains the supported scope and how to apply recommendations to new predictions.

## Approach

Move evidence extraction into a small read-only helper and reuse the existing ID/class normalization contracts. Preserve immutable artifacts and scorer identity sources. This is an auxiliary binary F1/accuracy/kappa recommendation, not a replacement official scorer or automatic mutation of an existing run.

## Evidence

Before the fix, real imported evidence returned accuracy 0.5 for a perfect reversed-order classifier; aliasing the truth column failed; requesting roc_auc returned an F1 value labeled roc_auc. Regression tests now cover both class orders, target aliases, string labels, selected versus decoy variants, source hash drift, absent probabilities, unsupported metrics, invalid probabilities and bounded grids. Full suite: 1,070 tests, 2 skipped, no failures/errors; Ruff and diff whitespace checks pass.

## Impact & follow-ups

The action requires one binary target with two declared classes and probability/class-order metadata. Regression, multiclass and multi-target threshold policies remain agent-owned custom work. Completed evidence is never rewritten.
