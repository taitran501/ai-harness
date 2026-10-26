# Epistemic memory, workspace confinement, and reconciliation hardening

- **Date**: 2026-10-26
- **PR**: — (not opened)
- **Branch**: `main` (uncommitted working tree)
- **Status**: uncommitted
- **Commits**: —

## Why

Checkpoint narratives could appear alongside verified experiment results without a clear epistemic boundary. Provider query failures could be reported as `CLEAN`, and candidate preflight results had no typed citation check to bind a pitch to the current prediction file or require paired OOF support for near duplicates.

## What changed

- Extracted `render_markdown_progress`; it reports live-validated runs and machine-evaluated decisions separately from explicitly labeled checkpoint notes. Checkpoint rows are marked `note [unverified]`.
- Marked checkpoint and open-question lines as `[UNVERIFIED NOTE]` in the Context HUD and added cached provider status, reason, count when known, and last-checked time.
- Changed unavailable provider queries to `UNKNOWN / PROVIDER_UNAVAILABLE` with a null provider count. `CLEAN` now requires a successful query with no unexplained submissions.
- Stored preflight receipts under `.ai-harness/artifacts/preflight/` and added receipt loading and candidate-pitch validation. The typed action checks receipt identity and candidate hash; near duplicates require indexed, machine-supported positive paired OOF evidence tied to the candidate and incumbent.
- Added the candidate citation mandate and workspace anti-dredging instructions to `AGENTS.md` and generated skills.
- Added focused reconciliation, memory rendering, candidate validation, HUD, and generated-skill tests.

## Approach

Verified progress facts are passed to a pure renderer from live run validation and machine evaluation receipts. Checkpoint prose stays available for continuity but is always called out as an unverified hypothesis. Provider status distinguishes an empty successful response from an unavailable query. Candidate citations are resolved only from validated IDs in the active workspace, and the current candidate file must match the stored SHA-256. Paired OOF support is accepted only from an indexed, intact decision-evaluation receipt with a supported outcome, positive gain, passing uncertainty check, and matching full-run candidate/control IDs on the OOF partition.

Workspace confinement text is an instruction boundary, not an implementation of OS-level access control. IDE and operating-system tool permissions must enforce the filesystem boundary independently.

## Evidence

- `ruff check harness/ tools/ tests/` — passed.
- `python tools/run_tests.py --smoke` — 47 tests passed across 5 modules.
- Focused affected modules — 49 tests passed.
- `python tools/run_tests.py --fast` — 1,130 tests passed, 1 skipped.

## Impact & follow-ups

Free-form conversation can still bypass a typed action unless the host applies response middleware; generated instructions require the citation and the harness validates pitches submitted through its typed action. The airline artifact-memory quarantine described in the research plan was not performed: its target is outside this repository's writable workspace, and this change did not read or modify those files.
