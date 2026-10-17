# Agent-bypass audit: why agents skip the harness (and now can't silently)

- **Date**: 2026-10-17
- **PR**: [#60](https://github.com/taitran501/ai-harness/pull/60)
- **Status**: merged (squash)
- **Area**: status / project / skill_injector / hygiene

## 1. Why (motivation + thought)

Audit adoption: what happens when an agent is lazy or simply has no intention
of using the harness? Findings (all empirically verified in `rsna-knee`): the
agent had bypassed the harness 100% for cloud runs — 22/22 `kaggle_sub_*` dirs
were manual kernel pushes (no `artifact_manifest.json`), zero harness
experiments/sessions/warnings, dated recent (not legacy). Three root causes:

1. A real bug — `status` reported kaggle/modal as *unavailable* even with an
   installed + authenticated CLI (looked for `available`/`detected` keys while
   `EnvironmentDetector` reports `installed`/`authenticated`/
   `has_credentials`). An agent reading the status concluded dispatch was
   impossible and pushed manually.
2. Structural — no workspace had a universal agent entry point;
   `.agents/skills/` only load for harnesses that support them and when the
   task matches.
3. A maintenance trap — two SKILL.md generators (`attach` inline block vs
   `skill_injector`), so `attach` wrote v1 and `sync` silently overwrote with
   v2; neither contained the other's content.

Dispatch itself was fully functional — only the *reporting* was wrong, which is
what steered the agent to the manual path.

## 2. What (concrete changes)

- **`harness/status_report.py`**: per-provider availability mapping (modal:
  installed+has_credentials; kaggle: installed+authenticated; docker:
  installed+running; gpu: detected; llm: total_configured).
- **`harness/project.py`**: `attach` now generates a managed `AGENTS.md` block
  at the workspace root (universal entry point for any coding agent) — points
  to the skill SOP, the context command, the experiment lifecycle, and
  non-negotiable no-bypass rules (no manual kernel pushes, report dispatch
  failures, immutable artifacts). User-owned AGENTS.md without the marker is
  never touched; managed blocks refresh in place.
- **`harness/project.py` + `harness/skill_injector.py`**: consolidated skill
  generation — `attach_project` now calls `inject_skill_file` (single source of
  truth) with `HARNESS_PROJECT_ROOT` pinning; the lean inline block's unique
  content was ported into the skill_injector SOP.
- **`harness/hygiene.py`**: run-output dirs lacking the harness
  `artifact_manifest.json` aggregate into a `bypassed_run_output` finding;
  sealed dirs keep the existing `stale_run_output` advisory.

## 3. Approach (decisions + rejected alternatives)

Three cheap signals close the blind spot: report your own capabilities
truthfully (a wrong status is an adoption bug, not a cosmetic one), give every
agent a universal entry point (managed AGENTS.md), and make the bypass
detectable (the artifact manifest is a free, reliable seal). Consolidated to ONE
skill generator (single source of truth) rather than patching the two divergent
ones separately.

## 4. Evidence

- New `tests/test_agent_bypass_audit.py` (8 tests: availability mapping with a
  mocked detector; bypass vs sealed hygiene aggregation; AGENTS.md
  create/user-owned/refresh; attached-skill == sync-skill content).
- Fixes A (status) and B (AGENTS.md) bite-checked by reverting. Full suite 697
  OK; ruff gate clean. rsna-knee HUD now shows: 35 bypassed, 3 stale, 1 stray
  file.

## 5. Impact & follow-ups

- **Downstream**: the adoption failure was invisible because the framework only
  measures what flows through it — bypassed runs leave no trace *by design*.
  This makes the bypass measurable and the entry point universal.
- Feeds the adoption stack (`adoption-stack-t3-t5.md`) as the
  detection + truthful-status layer.
