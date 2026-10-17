# Skill Source Fix: Add Pipeline Section to Generated SKILL.md Source

- **Date**: 2026-10-17
- **PR**: [#83](https://github.com/taitran501/ai-harness/pull/83)
- **Branch**: `— (squash commit 0127a30; branch name not recorded)`
- **Status**: merged
- **Commits**: `0127a30`

## Why

PR #82 added a pipeline section to `.agents/skills/ai-harness/SKILL.md` — but that file is **gitignored and generated** (per-workspace output of `harness/skill_injector.py`). Editing it directly meant the change lived only in one workspace and was lost on the next `attach`/`sync` (which regenerates the skill from source). The durable source of truth is `skill_injector.py`; the generated file is derived output. This change moves the pipeline section into the source so every regenerated SKILL.md carries it.

## What changed

- **`harness/skill_injector.py`**: the generated SKILL.md template now includes the pipeline section (valid `approach × mode` combo table, template + scaffold references, audit-gate guarantee) — previously it existed only in the hand-edited, gitignored output.

## Approach

Reinforces the rule: **never edit `.agents/skills/ai-harness/SKILL.md` directly; always update `harness/skill_injector.py` (the source).** The pipeline content was ported verbatim from the hand-edited output into the generator, so the file is now reproducible from source. This is the same class of bug the `SKILL.md` persistence convention exists to prevent (editing derived output instead of its generator).

## Evidence

- Generated SKILL.md now contains the pipeline section (verified via `generate_skill_content`).
- `tests/test_skill_injector.py` passes.

## Impact & follow-ups

Pipeline discoverability in the generated skill is now durable across `attach`/`sync` and across workspaces. Reinforces the "edit the source, not the generated output" convention (now also stated in `AGENTS.md`).
