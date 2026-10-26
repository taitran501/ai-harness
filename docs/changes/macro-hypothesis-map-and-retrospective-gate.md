# Macro hypothesis map and periodic retrospective gate

- **Date**: 2026-10-26
- **PR**: Working tree
- **Branch**: `main` (uncommitted implementation)
- **Status**: local implementation
- **Commits**: None

## Why

Long campaigns retain local insight history but can repeatedly adjust nearby
parameters without noticing untouched research families. The existing action-count
checkpoint and stagnation latch do not provide a broad coverage review. The design
in `scratch/codex_metacognition_discussion.md` calls for evidence-derived coverage
and a retrospective at the next costly boundary.

## What changed

- Added versioned macro map and review schemas, nine canonical families, and
  family/source metadata on insight registration. Legacy insights remain visible
  under Unclassified.
- Added a derived map with mixed evidence counts, comparison-group IDs, live
  machine-evaluated support, bounded exhaustion and deterministic evidence digests.
- Added typed `get_macro_hypothesis_map`, `check_retrospective_status` and
  `record_macro_review` actions through the existing flat/JSON CLI. Reviews validate
  task, digest, run count and mechanism ID links before appending a milestone.
- Added five-new-verified-run cadence and unreviewed stagnation-event triggers,
  enforced at Bridge, local runner, cloud manager, direct remote preparation, and
  submission authorization boundaries. Registered stagnation recovery probes retain
  their rule/generation/candidate-budget exception; smoke/probe execution remains
  available.
- Elevated coverage and due status in the HUD, session-start CLI response and
  research context. Review records appear in generated progress Markdown. Updated
  AGENTS.md, the skill generator and research continuity/trace documentation.

## Approach

The ledger and sealed evidence remain authoritative. No second editable map is
stored. An active direction takes display precedence over supported mechanisms;
counts preserve both. Support requires an intact machine receipt and unchanged,
currently verified full-run evidence. Exhaustion requires terminal machine evidence,
a bounded plan with stop criteria, and revisit conditions for every direction.

A review stores covered run IDs and stagnation-event identities inside its ledger
event. ID set differences prevent lost old artifacts from reducing the number of
new results. The review's digest binds the current family projection, ledger events
and run evidence. Invalid reviews append nothing. Reviews do not clear the separate
stagnation latch. Remote workers use the existing bound execution capsule; their
origin workspace performs the retrospective check before dispatch.

## Evidence

- Requested focused suite: 77 tests passed across `test_macro_research`,
  `test_retrospective_gate`, `test_audit_gate`, and `test_submission_dispatch`.
- Smoke tier: 47 tests passed.
- `ruff check harness/ tools/ tests/`: clean.
- Broader regression checks are recorded below when complete.

New tests use shared disposable workspace fixtures and genuinely sealed synthetic
evidence. They cover mixed states, missing stopping/revisit criteria, receipt and
artifact tampering, stale/invalid review claims, restart persistence, excluded
probes/failures, stagnation generations, bounded recovery and direct costly paths.
CLI flat flags and JSON review payloads are exercised through the real Bridge.

## Impact & follow-ups

Existing campaigns with five verified results immediately require a retrospective
before further ordinary full runs or submissions. Completed evidence is unchanged.
An untouched lead's priority and orthogonality rationale are scientist judgments;
the runtime validates references and state, not the quality of scientific prose.
Legacy text decisions cannot establish machine-supported or exhausted coverage.
