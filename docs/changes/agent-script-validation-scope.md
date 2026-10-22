# Make custom execution honor the resolved validation scope

- **Date**: 2026-10-22
- **PR**: pending creation
- **Branch**: `fix/agent-script-validation-scope`
- **Status**: open

## Why

Single-fold smoke exported every row as fold zero when folds were generated, or let a child silently execute all CV folds when folds already existed. Holdout/time-series scripts were incorrectly required to predict the entire training dataset rather than the validation subset.

## What changed

A shared partition export supplies exact selected train/validation IDs, probe sample bounds and run kind to both adapters. Children read `load_validation_manifest`; the parent evaluates precisely those validation IDs and rejects extra or missing output. The full compatibility fold map is retained without collapsing unselected rows into fold zero. Generated skills, API reference, workflow docs and the training scaffold describe the contract.

## Approach

Keep task-owned modeling and loading in the child while providing an explicit execution scope. Seal the partition manifest and hash actual bounded splits. Do not claim a successful smoke when a legacy script ignores scope and returns full-CV output.

## Evidence

Real subprocess matrix covers both adapters with full CV, generated/preset single-fold smoke, bounded probes, ratio holdout, separate dev holdout and temporal CV. Extra OOF IDs fail; probes never enter the full-run tree. Existing adapter/pipeline tests and Ruff pass; full-suite results appear in the PR.

## Impact & follow-ups

Existing full-CV scripts retain splits.tsv. Probe/holdout scripts must consume the explicit partition contract. The harness checks output scope; task code remains responsible for actually fitting only on the supplied training partition.
