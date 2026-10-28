# Research memory epistemic quarantine

- **Date**: 2026-10-28
- **PR**: Working tree
- **Branch**: current checkout
- **Status**: open
- **Commits**: —

## Why

Legacy research prose can state unsupported architectures, local execution, or scores. Previously, generated resume context repeated those statements and macro maps treated every insight as active, allowing unsupported claims to look like current research state.

## What changed

- Added `ResearchMemory.audit_research_memory()` and the read-only `ai-harness audit-memory` report, classifying entries against live validated runs and machine evaluation receipts without deleting ledger entries.
- Quarantined unsupported insight titles and checkpoint conclusions from the default progress resume; added an epistemic warning, kept registered probe plans visible, and added the audit command hint.
- Updated Context HUD rendering to suppress ungrounded checkpoints and distinguish machine-evaluated decisions from unverified interpretations.
- Macro families become active only through validated run evidence or preregistered probe plans.
- Clarified the agent and correctness contracts and added focused regression coverage.

## Approach

Keep `research_memory.json` append-only and generate a separate evidence-backed projection. External references are classified as external provenance and do not count as local execution. The default resume is conservative; detailed historical notes remain available through audit/search and trace workflows.

## Evidence

`ruff check harness/` passed. `python tools/run_tests.py --smoke --no-lock` passed (47 tests). Focused research-memory, macro-research, and HUD suite passed (30 tests).

## Impact & follow-ups

Legacy entries remain unchanged and can be inspected using `ai-harness audit-memory --project-root <workspace>`. Pattern-based identification of old execution claims is deliberately conservative and should be reviewed as a classification, not a substitute for artifact validation.
