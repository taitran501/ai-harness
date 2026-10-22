# Preserve probability class order in custom execution

- **Date**: 2026-10-22
- **PR**: pending creation
- **Branch**: `fix/agent-probability-class-order`
- **Status**: open

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
