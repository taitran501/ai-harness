# Lean slash command skills (`/harness`, `/debate`, `/submit`)

- **Date**: 2026-10-31
- **PR**: Working tree
- **Branch**: `main`
- **Status**: open
- **Commits**: —

## Why

The previous slash-command configuration was bloated into 6 separate micro-skills with repetitive prefixes (`ai-harness`, `ai-experiment-harness`, `ai-harness-resume`, `ai-harness-explore`, `ai-harness-submit`, `ai-harness-debate`). Their descriptions were dense with academic jargon, making it confusing for a human operator looking at the Antigravity `/` command menu to know which skill to apply.

## What changed

- **Consolidated into 3 clean, intuitive slash skills**:
  1. `/harness`: Autonomous ML experiment runner (data inspection, baselines, single-fold smoke tests, cross-validation, and progress resumption via Context HUD).
  2. `/debate`: Adversarial cross-examination between agents and claim citation audit against raw disk topic files to prevent hallucinations.
  3. `/submit`: Prediction diversity checks, out-of-fold gain auditing, and safe competition submission generation with operator approval.
- **Human-friendly descriptions**: Replaced verbose jargon with clear, 1-line descriptions explaining exactly when the operator should trigger the skill.
- **Global & workspace purge**: Updated `install_global_skills` and `install_workspace_skills` to automatically purge all legacy bloated directories (`ai-experiment-harness`, `ai-harness-*`, `ml-harness*`), keeping `~/.gemini/config/skills` clean and aligned with other Gemini skills (`brainstorm`, `codex-cli`, `deep-research`, etc.).
- **Workspace compatibility**: Retained `ai-harness` twin in `.agents/skills/` so legacy workspace hooks and scripts continue functioning without disruption.

## Evidence

- `python -m unittest tests/test_skill_injector.py`: 12 tests passed.
- `python tools/run_tests.py --smoke --no-lock`: 47 tests passed.
- `python tools/run_tests.py --fast --no-lock`: 1,189 tests passed (100% OK).
- `ruff check harness/ tools/ tests/`: passed with 0 errors.
- Verified global skills directory (`C:\Users\trant\.gemini\config\skills`) now contains only `harness`, `debate`, `submit` alongside other standard skills.
