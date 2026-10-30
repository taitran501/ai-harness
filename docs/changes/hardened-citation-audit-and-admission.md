# Hardened citation audit, evidence registry, and admission protocol

- **Date**: 2026-10-31
- **PR**: Working tree
- **Branch**: `main`
- **Status**: open
- **Commits**: —

## Why

The initial citation scanner could find discussion files and compare nearby
keywords, but it could not prove that a particular source segment supported a
particular claim. Empty scans and scan failures also needed explicit status and
exit semantics. The implementation now separates coverage, evidence integrity,
claim support, empirical status, and policy admission. It supersedes the initial
scanner's heuristic verification wording while retaining that scan as advisory
lint.

## What changed

- Added versioned source records, explicit schema adapters, content-addressed
  discussion snapshots, stable segment locators, explicit claim/evidence
  bindings, independent verifier execution records, and hash-bound audit
  receipts under `.ai-harness/`.
- Replaced broad topic-file traversal with task-aware Markdown discovery and
  selected-document scanning. `audit-claims` now has strict mode, claims files,
  repeatable file selection, JSON output, and distinct input/admission exit codes.
- Revalidated citation dependencies at research recording, grounding, direct
  run, candidate pitch, submission authorization, and remote dispatch boundaries.
  Remote capsules carry portable validated evidence and review material.
- Added v2 candidate-preflight and submission-gate schemas while retaining v1
  schemas for compatibility. Added separate grounded-hypothesis,
  external-reported-result, and demonstrated-local-gain outcomes to debate skill
  generation.
- Added adversarial regressions for locator/excerpt mismatch, question versus
  affirmative answer, negation, invented entities, ambiguous line association,
  same-line claim independence, duplicate order, BOM/decoding failures, receipt
  tampering, CLI exits, and admission behavior.
- Updated the research trace, continuity, and correctness contracts. Historical
  ledger entries and completed experiment evidence remain untouched.

## Approach

Mechanical checks establish source identity and excerpt integrity only. A claim
becomes supported only through a separate structured verifier verdict bound to
the current claim/evidence hashes and execution provenance. Heuristic token
overlap remains a warning, never a support verdict. Hypotheses may proceed to a
bounded probe; external reported results remain external; demonstrated local
gain continues to depend on the existing machine-evaluated decision rule.

Source discovery is explicit and adapter-based. The audit service reads only the
selected documents and referenced snapshots, caches each distinct source within
an invocation, and stores receipts outside immutable experiment directories.
The verifier provenance records a separate execution but does not claim
cryptographic actor authentication against a complete workspace rewrite.

## Evidence

- Targeted citation, admission, research-memory, grounding, candidate-preflight,
  submission-dispatch, skill, provenance, correctness-regression, and remote
  hardening suites: **131 tests passed, 1 skipped** across the final targeted
  runs. The mocked cloud admission case asserts provider preflight is never
  called when a discussion receipt is missing.
- `python tools/run_tests.py --smoke --no-lock`: **47 tests passed**.
- `ruff check harness/ tools/ tests/`: passed.
- `python -m compileall -q harness tests`: passed before the final test/documentation
  edits.
- CLI subprocess regressions verify JSON/text results and the 0/1/2 exit-code
  contract. Live cloud dispatch and unmocked verifier-service calls were not run.

## Impact & follow-ups

Strict independent review requires a configured verifier adapter and fails
explicitly when none is available. Historical citations remain unverified until
registered and reviewed; no receipts were manufactured for prior runs. Receipt
hashes detect stale or altered evidence but cannot authenticate a workspace owner
who rewrites all local state.
