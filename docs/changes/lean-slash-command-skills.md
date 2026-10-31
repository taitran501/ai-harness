# Lean slash command skills (`/harness`, `/debate`) and submission doc consolidation

- **Date**: 2026-10-31
- **PR**: Working tree
- **Branch**: `main`
- **Status**: open
- **Commits**: —

## Why

The previous slash-command configuration was bloated into 6 separate micro-skills with repetitive prefixes (`ai-harness`, `ai-experiment-harness`, `ai-harness-resume`, `ai-harness-explore`, `ai-harness-submit`, `ai-harness-debate`). Furthermore, `submit` is a competition-internal phase rather than a general-purpose Gemini assistant skill. Exposing `/submit` globally in Gemini is out-of-context for non-competition tasks.

## What changed

- **Consolidated into 2 purpose-driven slash skills**:
  1. `/harness`: Autonomous ML experiment runner (data inspection, baselines, single-fold smoke tests, cross-validation, progress resumption via Context HUD, and submission governance/preflight gate).
  2. `/debate`: Adversarial cross-examination between agents and claim citation audit against raw disk topic files to prevent hallucinations.
- **Removed `/submit` from slash skills**: Folded submission governance, zero CV gain mandate, candidate preflight ($\rho < 0.995$), and human confirmation rules directly into `harness` (under Section 6).
- **Human-friendly descriptions**: Replaced verbose jargon with clear, 1-line descriptions explaining exactly when the operator should trigger the skill.
- **Global & workspace purge**: Updated `install_global_skills` and `install_workspace_skills` to automatically purge all legacy bloated directories (`ai-experiment-harness`, `ai-harness-*`, `ml-harness*`, `submit`), keeping `~/.gemini/config/skills` clean and aligned with other Gemini skills (`brainstorm`, `codex-cli`, `deep-research`, etc.).
- **Workspace compatibility**: Retained `ai-harness` twin in `.agents/skills/` so legacy workspace hooks and scripts continue functioning without disruption.

## Evidence

- `python -m unittest tests/test_skill_injector.py`: 12 tests passed.
- `python tools/run_tests.py --smoke --no-lock`: 47 tests passed.
- `python tools/run_tests.py --fast --no-lock`: 1,189 tests passed (100% OK).
- `ruff check harness/ tools/ tests/`: passed with 0 errors.
- Verified global skills directory (`C:\Users\trant\.gemini\config\skills`) now contains only `harness` and `debate` alongside standard Gemini skills.
