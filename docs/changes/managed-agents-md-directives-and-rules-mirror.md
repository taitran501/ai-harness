# Strengthen Managed AGENTS.md Directives and Mirror Rules for Coding Agents

- **Date**: 2026-11-02
- **PR**: [#136](https://github.com/taitran501/ai-harness/pull/136)
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: `<sha>`

## Why

When AI coding assistants (Antigravity, Cursor, Claude Code, Codex) enter an attached workspace, the only context file automatically injected into their system context on every turn is `AGENTS.md` (and `.agents/rules/*.md` in Antigravity).
Previously, `render_agents_md_body()` in `harness/project.py` only outputted a brief 16-line procedural notice. Consequently, agents frequently ignored the harness lifecycle:
1. They defaulted to a conversational persona, outputting generic machine learning essays and engaging in open-ended brainstorming without physical harness evidence.
2. They wrote loose Python scripts (`train.py`, `infer.py`, `predict.sh`) running detached in the terminal outside harness tracking and cross-validation audit gates.
3. They generated ad-hoc unaligned submission files in the workspace root without checking schema contracts, running candidate preflight, or registering submissions.
4. They finished turns without recording research continuity (`record_research_checkpoint`, `record_research_insight`), losing context across session boundaries.
5. In monorepo workspaces containing multiple competition sub-projects (e.g. `zalo_ai_challenge` containing `datamate/` and `lifemate/`), agents mixed root and sub-project contexts.

## What changed

- **`harness/project.py`**:
  - Expanded `render_agents_md_body()` into a comprehensive, non-negotiable **System Execution Directive** covering:
    1. **No Ungrounded Brainstorming / No Ad-Hoc Scripts**: Strictly prohibiting loose terminal scripts and conversational speculation in a vacuum; requiring declarative configs in `configs/exp<NNN>_<descriptor>.yaml` or `pipeline`/`script_adapter` integration.
    2. **Mandatory First Action**: Requiring `ai-harness-agent --project-root . context` (or `/harness`) as the mandatory first action before taking any steps or proposing hypotheses.
    3. **Experiment Lifecycle**: Fast single-fold smoke probe (`--debug --fold 0`) before full multi-fold compute.
    4. **Standardized Submissions & Governance**: Requiring candidate preflight (Spearman rho < 0.995 vs incumbent) and submission registration (`register_submission`) to ensure test predictions and recipes are recorded.
    5. **Mandatory Research Journaling**: Requiring `record_research_insight`, `record_research_decision`, and `record_research_checkpoint` after every iteration.
    6. **Multi-Task / Monorepo Workspaces**: Mandating `--project-root <task_dir>` when operating in repositories with multiple sub-tasks.
    7. **Framework Upgrades**: Reaffirming `ai-harness sync`.
  - Updated `update_agents_md(root)` to also mirror these directives into `.agents/rules/harness.md`, ensuring native discovery by Antigravity rule engines.
- **`tests/agent_bridge/test_cloud_hygiene_bypass.py`**:
  - Added assertions verifying that `AGENTS.md` and `.agents/rules/harness.md` contain the non-negotiable directives on attach.

## Evidence

- Unit tests: 8/8 tests pass in `tests.agent_bridge.test_cloud_hygiene_bypass`.
- Smoke tier: 47/47 tests pass in 2.9s (`python tools/run_tests.py --smoke`).
- Static lint: `ruff check harness/ tools/ tests/` (all checks pass).
- Live workspace verification: Successfully refreshed `AGENTS.md` and created `.agents/rules/harness.md` in `zalo_ai_challenge`, `zalo_ai_challenge/datamate`, and `zalo_ai_challenge/lifemate`.

## Impact & follow-ups

None. Any workspace running `ai-harness attach` or `ai-harness sync` automatically receives the refreshed directives.
