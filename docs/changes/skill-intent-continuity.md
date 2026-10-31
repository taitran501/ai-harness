# Skill intent and research continuity

- **Date**: 2026-10-31
- **PR**: —
- **Branch**: `feat/research-workflow-skills`
- **Status**: Working tree
- **Commits**: —

## Why

In competition conversations, a user may invoke a skill or use a short continuation
to request a reminder, research, a plan, or the next authorized action. Agents have
treated review as a terminal recommendation and repeated earlier agent narratives
despite later user corrections. The generated skill also displayed an undeclared
validation scheme when the task declared one in `task.validation`.

## What changed

- The generated `/harness` skill interprets short requests in the current objective
  and respects their requested scope while continuing authorized work.
- The generated `/debate` skill ties its review to the active decision and next
  appropriate action, and distinguishes observations and evaluation evidence from
  causal interpretations.
- A task-aware `/research` skill connects investigation, previous experiments,
  evidence and the next decision without mandating a fixed research method.
- Validation information comes from the task contract, with legacy field fallback.
- Skill injector tests reflect the current workflow and cover validation field
  precedence and backward compatibility.
- Pre-push validation exposed seven failures also reproduced on the parent commit:
  update legacy alias/HUD assertions to test the canonical skill and current policy,
  keep the research frontier compact, and reject a nested Kaggle mount owned by a
  different account when an explicit owner was requested. Flat mounts still work;
  a nested directory with no matching owner cannot be safely identified by name
  alone and now fails closed.

## Approach

Keep the existing execution/approval gates and connect `/research` to `/harness`
and `/debate` without introducing a competition-specific recipe. A research-only
request remains research-only, while a request to continue authorized work should not end at a
passive list of options. Actual behavioral compliance requires evaluation with a
real agent; generated instructions alone cannot guarantee it.

## Evidence

`python -m unittest tests.test_skill_injector -q`: 13 tests passed.
`python tools/run_tests.py --smoke`: 47 tests passed. Ruff passed for the changed
Python files, `git diff --check` passed, and all three regenerated airline skills
passed `skill-creator`'s `quick_validate.py`. Real Antigravity conversation
behavior has not been replayed.

The seven initial full-suite failures were reproduced independently on parent
`078cc48`; all seven targeted regressions pass after the fixes above. A subsequent
full run exposed a conflicting ownerless-nested-mount expectation, which was
adjusted to require a verifiable owner instead of guessing from a directory name.

## Impact & follow-ups

Existing workspaces need the regenerated skills; a local patch based on an
unmerged framework branch can be overwritten by a later framework sync. Durable
user-intent state and independent agent behavior evaluation remain follow-ups.
