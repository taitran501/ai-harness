# Modular slash-command skills and global synchronization

- **Date**: 2026-10-28
- **PR**: Working tree
- **Branch**: `main`
- **Status**: open
- **Commits**: `<pending>`

## Why

Assistants (e.g. Antigravity) and users navigate the ML harness across different conversation sessions and workspaces. Previously, harness skills were injected only as a monolithic workspace skill (`.agents/skills/ai-harness/SKILL.md`) or were unavailable as granular slash commands across new chats. When starting a new session or resuming an experiment, agents often hallucinated workflow actions or lacked immediate step-by-step guidance on anti-hallucination triage, discussion mining, and submission governance unless the user manually explained them.

Furthermore, Antigravity discovers slash commands via two primary locations: global skills (`~/.gemini/config/skills/<name>/SKILL.md`) and workspace skills (`<workspace>/.agents/skills/<name>/SKILL.md`). Without global skill installation and modularization, typing `/` in chat did not present specialized harness commands like `/ai-harness-resume`, `/ai-harness-explore`, or `/ai-harness-submit`.

## What changed

- **`harness/skill_injector.py`**:
  - Added modular skill generators:
    - `generate_modular_skills`: builds a dictionary mapping skill names to their markdown content and target directory structure.
    - `generate_resume_skill_content`: generates `/ai-harness-resume` (session restart triage, Context HUD inspection, `RESEARCH_PROGRESS.md` "Resume here" check, quarantine of ungrounded notes, baseline anchor & stagnation latch verification).
    - `generate_explore_skill_content`: generates `/ai-harness-explore` (discussion mining, hypothesis generation, bounded `DecisionRuleV1` probe planning, single-fold smoke test validation).
    - `generate_submit_skill_content`: generates `/ai-harness-submit` (local CV improvement mandate, prediction provenance check, candidate preflight diversity test, raw script audit, exact-payload operator approval).
  - Added `get_global_skills_dir()`: locates `GEMINI_SKILLS_DIR`, `ANTIGRAVITY_SKILLS_DIR`, or `~/.gemini/config/skills`.
  - Added `install_global_skills()`: atomically writes modular skills (`ai-harness`, `ai-experiment-harness`, `ai-harness-resume`, `ai-harness-explore`, `ai-harness-submit`) to the global skills directory.
  - Added `install_workspace_skills()`: atomically writes modular skills to `<project_root>/.agents/skills/` and cleans up legacy `ml-harness` folders.
  - Refactored `inject_skill_file()`: now coordinates with `install_workspace_skills()` while keeping backward compatibility for custom output paths.
  - Updated `generate_skill_content()`: added clear tip banners directing assistants to the modular slash skills.
- **`harness/cli.py`**:
  - Added `install-skills` subcommand supporting `--global`, `--project-root`, `--all`, and `--json`.
  - Added CLI handler `_run_install_skills`.
- **`harness/sync.py`**:
  - Embedded automatic global skill installation into `ai-harness sync`.
- **`harness/project.py`**:
  - Attached automatic global skill installation into `ai-harness attach`.
- **`tests/test_skill_injector.py`**:
  - Added comprehensive test coverage:
    - `test_generate_modular_skills`: validates YAML frontmatter and key directives across all 4 skills.
    - `test_install_global_skills`: validates file layout and writing in custom global targets.
    - `test_install_workspace_skills`: validates workspace installation and legacy cleanup.
    - `test_cli_install_skills`: validates CLI invocations and JSON payload output.

## Approach

Antigravity's slash command engine discovers custom skills from both `~/.gemini/config/skills` (user global) and `.agents/skills` (workspace local). By modularizing harness guidance into targeted sub-skills (`ai-harness-resume`, `ai-harness-explore`, `ai-harness-submit`, alongside the overarching `ai-harness`), agents and users can invoke specific phases with immediate contextual constraints directly via `/`.

The synchronization mechanism is wired into both the explicit CLI (`ai-harness install-skills`) and automated project lifecycle events (`ai-harness sync`, `ai-harness attach`), ensuring that global and local skills stay in lockstep with the framework without requiring manual copying or maintenance.

## Evidence

- Unit tests in `tests/test_skill_injector.py` pass 100%.
- Verified global skills written to `C:\Users\trant\.gemini\config\skills`:
  - `ai-harness/SKILL.md`
  - `ai-experiment-harness/SKILL.md`
  - `ai-harness-resume/SKILL.md`
  - `ai-harness-explore/SKILL.md`
  - `ai-harness-submit/SKILL.md`
- Smoke test tier (`python tools/run_tests.py --smoke --no-lock`): 47/47 passed.
- Ruff clean (`ruff check harness/ tools/ tests/`): 0 errors.

## Impact & follow-ups

None. Existing workspaces seamlessly acquire modular skills upon next `ai-harness sync` or `ai-harness attach`.
