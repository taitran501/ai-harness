# Preserve probability class order in custom execution

- **Date**: 2026-10-22
- **PR**: [#119](https://github.com/taitran501/ai-harness/pull/119)
- **Branch**: `fix/agent-probability-class-order`
- **Status**: merged
- **Merge commit**: `d79d6c1abc4651f6fe3d1a23bfb80acb50b550ca`

## Why

A script could export valid OOF probabilities with an explicit class order, yet ROC-AUC/logloss failed because the adapter dropped `classes_*` before evaluation. Test inference also silently substituted the task's class order for the model's order.

## What changed

Script and pipeline OOF records preserve `classes_*`. Shared test-prediction extraction carries a consistent declared class order into final inference and rejects conflicting orders within a file.

## Approach

Keep model ordering intact through transport; metric evaluation already validates and reorders probability matrices. Do not infer an ordering from task labels when the model explicitly supplies it.

## Evidence

Real subprocess regressions cover script/pipeline ROC-AUC and logloss, reversed model class order, test label decoding and sealed test evidence. A negative case rejects per-row class-order changes. Full-suite and Ruff results appear in the PR.

## Impact & follow-ups

No change to generic scorer identities or completed evidence. Test-ID alignment is a separate follow-up.

## Pre-merge review

Traced real subprocess CSV output through OOF scoring, test averaging, label decoding and evidence validation in both execution modes. Class metadata is carried alongside probability columns and inconsistent test class order is rejected. No blocking findings.

A fresh integration run passed: 1,053 tests, 2 skipped, no failures/errors. Ruff and diff whitespace checks passed.
