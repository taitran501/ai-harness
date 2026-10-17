# Adoption stack T3–T5 (external-repo research-backed enforcement)

- **Date**: 2026-10-17
- **PR**: [#61](https://github.com/taitran501/ai-harness/pull/61) (T3–T5; T0–T2 in [#60](https://github.com/taitran501/ai-harness/pull/60))
- **Status**: merged (squash)
- **Area**: agent-bridge / host hooks / status

## 1. Why (motivation + thought)

Question: what do external projects do when agents bypass/skip prescribed
tooling? Web-verified, the field converges on a 3-layer guardrail model:
(1) instruction files (AGENTS.md) — advisory only, adherence drops past
~200 lines; (2) per-host tool hooks (Claude Code PreToolUse, Cursor
beforeShellExecution) — exit 2 blocks with the reason fed back, "hooks run
outside the model so prompt injection can't talk its way past them";
(3) repo layer (pre-commit/CI/branch protection) — binds every change but is
evadable (documented case of Claude Code using `--no-verify` across consecutive
commits). Closest ML analog: agentic-experiments (aexp) — *deterministic
hooks*. Counter-lesson: don't wrap APIs 1:1. Empirical local fact that shaped
the plan: 4/5 workspaces are NOT git repos — layer 3 is structurally absent, so
the host-agnostic layer the harness controls is its own CLI/HUD surface.

## 2. What (concrete changes)

- **T3 session-start adoption gate** (`agent_bridge`): the Context HUD (which
  the SOP mandates reading at session start) renders a `⛔ Adoption protocol`
  block when bypass findings exist or 0 harness experiments — aexp's
  session-start refusal implemented in the layer every host must read.
- **T4 per-host guardrail hooks** (`harness/host_hooks.py`, new): `attach`
  generates `.claude/settings.json` (PreToolUse/Bash) + `.cursor/hooks.json`
  (beforeShellExecution) + the hook script at
  `.ai-harness/hooks/block_manual_kaggle.py`. Blocks `kaggle kernels push` /
  `competitions submit` with exit 2 + a reason pointing at the harness path.
  Invariants: fail-open on any parse error (a guardrail that breaks the host is
  worse than none), merge-safe (user hooks preserved, only the
  `managed_by: ai-harness` entry replaced), human override
  `AI_HARNESS_ALLOW_MANUAL=1`.
- **T5 adoption health in status** (`status_report`): `adoption` section —
  experiments / bypassed run-outputs / sessions + verdict
  (healthy | bypass_detected | no_activity).

## 3. Approach (decisions + rejected alternatives)

Three layers, each targeting a different evasion path. T3 lives in the HUD
(every host reads it per the SOP). T4 is the hard block (per-host hooks,
exit 2) — the layer prompt-injection can't talk its way past. T5 makes the
state machine-readable. Rejected a 1:1 MCP wrapper (the counter-lesson) and a
pre-commit-only approach (evadable, and 4/5 workspaces aren't git repos).

## 4. Evidence

- New `tests/test_adoption_stack.py` (19 tests: hook block/allow/fail-open/
  override via real subprocess; config create/merge/idempotence/unreadable-skip;
  adoption verdicts; HUD gate present/wording).
- E2E in rsna-knee: hook blocks manual push rc=2, benign rc=0, override rc=0;
  HUD gate + status adoption line verified. Full suite 716 OK; ruff gate clean.

## 5. Impact & follow-ups

- **Follow-up**: T6 — an `ai-harness` MCP server exposing ~6–8 domain tools
  (context, create_experiment, run_smoke_test, run_full_validation,
  record_research_*) — the Toolshed adoption pattern. Big build; T3–T5 gives
  most of the adherence value without it.
- **Downstream**: the generated SKILL.md documents the manual-run hook block
  (exit 2), the report-don't-fall-back rule, and `AI_HARNESS_ALLOW_MANUAL=1`
  as the human operator's privilege.
