# Reject incompatible probability blends before optimization

- **Date**: 2026-10-23
- **PR**: [#124](https://github.com/taitran501/ai-harness/pull/124)
- **Branch**: `fix/agent-blend-optimizer-contract`
- **Status**: open

## Why

An agent could register two valid classifiers whose probability columns used different class orders. `optimize_blend` averaged their columns positionally and returned SUCCESS with a wrong score, while candidate construction rejected the same sources. A complementary synthetic pair should average to accuracy 1.0 after class alignment, but the optimizer reported 0.5 without explaining the incompatible ordering.

## What changed

The optimizer enforces the candidate builder's matching class-order requirement before scoring a probability blend. It also rejects repeated component IDs and unknown blend modes instead of reporting an optimization that candidate construction cannot reproduce.

## Approach

Preserve the existing explicit source contract rather than silently reordering stored evidence. Agents receive an actionable class-order error and can regenerate compatible predictions. Completed sources and scorer identity code are unchanged.

## Evidence

The new regression failed on the previous implementation: the Bridge returned SUCCESS for incompatible orders. Tests now cover rejection through the real external-import -> AgentBridge optimizer path, a compatible pair whose reported optimum and constructed candidate both score 1.0, and invalid component/mode requests. Full suite: 1,056 tests, 2 skipped, no failures/errors; Ruff and diff whitespace checks pass.

## Impact & follow-ups

Optimization no longer reports a misleading score for incompatible probability columns. No migration or evidence rewrite is required.
