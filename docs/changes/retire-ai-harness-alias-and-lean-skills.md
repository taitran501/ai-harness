# Retire /ai-harness Alias, Align Skill Producers, and Safe Alias Migration

- **Date**: 2026-11-02
- **PR**: [#136](https://github.com/taitran501/ai-harness/pull/136)
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: `<sha>`

## Why

AI-Harness was still spawning a duplicate `/ai-harness` skill in `.agents/skills/ai-harness` and autocomplete surfaces, even though `/harness` was established as the primary canonical slash command.
An architectural audit using Codex (`gpt-6.1-sol` with high reasoning effort) identified multiple contributing factors:
1. `harness/skill_injector.py` was explicitly writing `twin_dest` to `.agents/skills/ai-harness/SKILL.md` in `install_workspace_skills()`.
2. `tools/bootstrap_task.py` explicitly targeted `root / ".agents" / "skills" / "ai-harness" / "SKILL.md"` when bootstrapping new competition workspaces.
3. `harness/project.py` AGENTS.md template directed agents to read `.agents/skills/ai-harness/SKILL.md`.
4. `harness/host_hooks.py` referenced `.agents/skills/ai-harness/SKILL.md` in blocked-command stderr diagnostics.
5. Re-attaching an already-attached workspace (`ALREADY_ATTACHED`) returned before refreshing workspace skills, leaving obsolete alias directories behind.
6. Documentation files (`README.md`, `docs/prompts/onboarding.md`, `docs/plugin_architecture.md`) still instructed operators and agents to use `.agents/skills/ai-harness/`.

## What changed

- **`harness/skill_injector.py`**:
  - Removed `twin_dest` creation for `ai-harness` in `install_workspace_skills()`.
  - Implemented `_retire_generated_alias(skills_dir)`: safely detects previously generated `ai-harness/SKILL.md` matching known compatibility entry points, ensures it is not a symlink, unlinks the file, and removes the directory only if empty (preserving adjacent user-created files like `notes.txt`).
  - Added alias retirement invocation to both `install_global_skills()` and `install_workspace_skills()`.
  - Disallowed generating or injecting into `ai-harness` (`ValueError("/ai-harness is retired; use /harness")`).
  - Simplified `/harness` description frontmatter to a punchy, human-friendly single sentence.
- **`harness/project.py`**:
  - Pointed `AGENTS.md` managed template to `.agents/skills/harness/SKILL.md`.
  - Changed default attachment `skill_path` to `root / ".agents" / "skills" / "harness" / "SKILL.md"`.
  - In `same_attachment` (`ALREADY_ATTACHED`), invoke `install_workspace_skills()` to refresh missing canonical skills and retire legacy aliases automatically.
- **`harness/host_hooks.py`**:
  - Updated hook message to cite `.agents/skills/harness/SKILL.md`.
- **`tools/bootstrap_task.py`**:
  - Updated task bootstrap engine to inject and report `.agents/skills/harness/SKILL.md`.
- **Documentation**:
  - Updated `README.md`, `docs/prompts/onboarding.md`, and `docs/plugin_architecture.md` to reference `.agents/skills/harness/SKILL.md`.
- **Tests**:
  - Updated `tests/test_bootstrap_task.py`, `tests/test_upgrades_roadmap.py`, and `tests/agent_bridge/test_cloud_hygiene_bypass.py` to assert canonical `harness/SKILL.md` and verify `ai-harness` is absent.
  - Added comprehensive tests in `tests/test_skill_injector.py`:
    - `test_owned_alias_is_retired_in_both_scopes`
    - `test_alias_retirement_preserves_custom_files`
    - `test_retired_alias_cannot_be_generated_explicitly`
    - `test_reattach_refreshes_missing_skill_and_retires_alias`

## Approach

Following Codex `gpt-6.1-sol` recommendations:
- Standardized on a single, clean slash command: `/harness`.
- Rather than blindly running `shutil.rmtree()` on `ai-harness` (which could delete user custom files or follow symlinks), `_retire_generated_alias()` verifies file ownership, protects against symlinks, safely unlinks only recognized generated files, and removes empty directories.
- Ensured both generators (`skill_injector`, `bootstrap_task`) and consumers (`project.py`, `host_hooks.py`, tests, documentation) are fully aligned.

## Evidence

- Audited with OpenAI Codex CLI (`gpt-6.1-sol` with `model_reasoning_effort=high`, report saved to `scratch/codex_skills_audit_report.md`).
- Targeted tests: 42 passed in 11.5s (`python -m unittest tests.test_skill_injector tests.test_bootstrap_task tests.test_upgrades_roadmap tests.agent_bridge.test_cloud_hygiene_bypass`).
- Smoke tier gate: 47 passed in 2.9s (`python tools/run_tests.py --smoke`).
- Fast tier gate: 1210 tests passed across 122 modules (`python tools/run_tests.py --fast`).
- Static lint: `ruff check harness/ tools/ tests/` (all checks passed).
- Live workspace verification: `.agents/skills/ai-harness` successfully retired; `.agents/skills/harness` is present and lean.

## Impact & follow-ups

None. Legacy workspaces automatically have `/ai-harness` retired on next `sync` or `attach`.
