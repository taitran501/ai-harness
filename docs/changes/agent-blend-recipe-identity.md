# Preserve source-to-weight mapping in candidate recipe identity

- **Date**: 2026-10-22
- **PR**: pending creation
- **Branch**: `fix/agent-blend-recipe-identity`
- **Status**: open

## Why

Sorting source IDs and weights independently made A=90%, B=10% collide with A=10%, B=90%. The Bridge rejected the second scientifically distinct recipe as a duplicate.

## What changed

Fingerprints sort source/normalized-weight pairs together. Source reordering, proportional scaling and default equal weights canonicalize consistently. Invalid weights are rejected. Duplicate detection recomputes existing recipe identities from stored details rather than trusting the old lossy fingerprint.

## Approach

Use exact decimal fractions for normalized ratios to avoid introducing rounding collisions. Migration is read-only: completed candidate manifests and their hashes are not rewritten. Existing explicit revisit behavior is retained.

## Evidence

Unit cases cover swapped weights, reordered sources, scaled/default weights and invalid values. A real Bridge test builds both distinct blends, simulates an old stored fingerprint, then verifies that the equivalent reordered recipe is blocked while the original candidate remains unchanged. Full-suite and Ruff results appear in the PR.

## Impact & follow-ups

Agents can explore genuinely different blend weights without supplying a misleading revisit reason. Existing duplicate protection remains effective across the migration.
