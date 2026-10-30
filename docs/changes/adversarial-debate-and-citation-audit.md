# Adversarial dual-agent debate skill and automated claim citation audit

- **Date**: 2026-10-31
- **PR**: Working tree
- **Branch**: `main`
- **Status**: open
- **Commits**: `<pending>`

## Why

Single-agent workflows suffer from self-reinforcing confirmation bias: when reviewing discussions and benchmark outcomes, a lone agent often attributes noisy leaderboard movements to speculative mechanisms, compounds conjectures in subsequent steps, and even fabricates citations (e.g. citing topic numbers for claims not contained in them). In chat `bcb814a9-52cc-4bb9-98ac-9282e8a3bce4`, an agent generated a 200KB+ compendium attributing `shiiin9 D-prime` scheduling to `#750885` (which was actually Daniel Franzen's concurrency negative result) and elevated an unverified forum question in `#738766` into a definitive mathematical scoring law.

To prevent this epistemic failure mode, research memory and hypothesis formulation require both an adversarial dual-agent cross-examination protocol and an automated machine-enforced citation audit gate.

## What changed

- **`harness/audit_gate.py`**:
  - Added `extract_topic_text(raw_json)`: Recursively extracts textual content across titles, bodies, and comment threads from Kaggle discussion JSON files.
  - Added `scan_markdown_citations(project_root, file_path=None)`: Parses markdown documents for `#<topic_id>` citations, verifies that the corresponding discussion JSON files exist on disk (`MISSING_TOPIC_FILE`), and inspects surrounding claim terms against actual topic content to flag fabricated or ungrounded claims (`UNGROUNDED_TOPIC_CITATION`).
- **`harness/cli.py`**:
  - Added `audit-claims` CLI subcommand (`ai-harness audit-claims [--file <path>] [--json]`) to audit citations in markdown documents against disk files.
- **`harness/skill_injector.py`**:
  - Added `generate_debate_skill_content()` generating `/ai-harness-debate` (Adversarial Epistemic Debate & Cross-Examination Protocol).
  - Updated `generate_modular_skills()` to include `ai-harness-debate`.
  - Updated master skill (`/ai-harness`) and explore skill (`/ai-harness-explore`) with references to `/ai-harness-debate`.
- **`tests/test_provenance_dispatch_guard.py`**:
  - Added unit test `test_scan_markdown_citations_finds_missing_and_ungrounded_topics` verifying detection of missing discussion files and ungrounded topic citations.
- **`tests/test_skill_injector.py`**:
  - Updated modular skill generation, global installation (6 skills total), workspace installation, and CLI tests to cover `ai-harness-debate`.

## Approach

The system combines human-facing agent instructions with automated machine verification:
1. **The Protocol (`/ai-harness-debate`)**: Structures the research workflow into two opposing personas:
   - *Agent 1 (Proposer)*: Must format claims as formal Candidate Insight Cards with verbatim text excerpts and acknowledged confounders.
   - *Agent 2 (Auditor)*: Must physically verify the referenced file on disk, challenge statistical noise against benchmark variance, reject bundled multi-variable changes, and formulate counter-hypotheses.
2. **Automated Enforcement (`ai-harness audit-claims`)**: Scans markdown documents (compendiums, strategy notes, `RESEARCH_PROGRESS.md`) to verify that cited topics exist on disk and that technical keywords in claim lines actually appear in the referenced files.

## Evidence

- `tests/test_provenance_dispatch_guard.py`: 4/4 passed in 0.1s.
- `tests/test_skill_injector.py`: 12/12 passed in 10.1s.
- Smoke suite (`python tools/run_tests.py --smoke --no-lock`): 47/47 passed in 4.2s.
- Ruff clean (`ruff check harness/ tools/ tests/`): 0 errors.
- Real-world validation: `ai-harness audit-claims` run against `arc3_community_insights_compendium.md` from chat `bcb814a9` accurately flagged ungrounded citations (including `#750885` and `#738766`).

## Impact & follow-ups

All attached and newly created workspaces receive the `/ai-harness-debate` slash command upon running `ai-harness sync` or `ai-harness attach`. Global Antigravity skills now include `/ai-harness-debate`.
