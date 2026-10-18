# M4 refactoring (PR-E): AgentBridge action-handler extraction

- **Date**: 2026-10-18
- **PR**: — (branch `refactor/agent-bridge-handler-extraction`)
- **Branch**: `refactor/agent-bridge-handler-extraction`
- **Status**: open
- **Commits**: —

## Why

External review finding **M4**: `harness/agent_bridge.py` was the largest
module in the framework (4098 lines, `AgentBridge` god class with 65
methods). Slice E of the M4 refactor splits it into three files with a
single-responsibility layout, proving the behavior-preserving extraction
pattern on the highest-value module after slice D (dashboard, PR #98).

## What changed

**New: `harness/bridge_common.py`** (215 lines) — the bridge's typed
vocabulary, moved **verbatim** from `harness/agent_bridge.py` L43–L234:
- `ApprovalTier`, `TypedActionName` (enums)
- `MAX_SESSION_ACTIONS`, `MAX_SESSION_OBSERVATIONS` (session caps)
- `DEFAULT_ACTION_APPROVAL_TIERS`
- dataclasses: `SessionBudget`, `ComputeUsage`, `TypedActionCall`,
  `ActionObservation`, `BridgeSession`
- `_TYPED_ACTION_NAMES` guard set (used by the class-level proxy metaclass)

Single source of truth for the typed action vocabulary; pure stdlib
imports, no `harness` imports (no cycle risk).

**New: `harness/bridge_actions.py`** (1893 lines) — `ActionHandlerMixin`
holding the 30 typed `_handle_*` methods, moved **verbatim** (1862-line
contiguous block, byte-identical, verified by diff). Handlers lazy-import
heavier harness modules inside their bodies, so the module needs only the
shared types from `bridge_common` plus `GroundingGate` at module scope.

**`harness/agent_bridge.py`** (4098 → 2052 lines) — now the session
lifecycle, context packets, guardrails, `execute_action` dispatch, and
module functions (`_truncate_text`, `_research_continuity_hud_lines`,
`adoption_gate_lines`, `_AgentBridgeMeta`). `AgentBridge` mixes in
`ActionHandlerMixin`; the file re-exports every moved name
(`TypedActionName`, `ActionObservation`, `ApprovalTier`,
`SessionBudget`, `BridgeSession`, `MAX_SESSION_*`,
`DEFAULT_ACTION_APPROVAL_TIERS`, `_TYPED_ACTION_NAMES`), so all 25+
external import sites are unaffected.

## Approach

- **Pure mechanical move** — no logic line changed; both moved blocks are
  byte-identical to the original (verified against `git show HEAD`).
- **Mixin, not reimplementation** — `class AgentBridge(ActionHandlerMixin,
  metaclass=_AgentBridgeMeta)`; method resolution order and the
  class-level proxy metaclass behavior are unchanged.
- **Dependency audit before moving**: the 30 handlers reference only
  `ActionObservation`/`TypedActionName` from the module scope (now imported
  from `bridge_common`), stdlib/third-party names, and lazy in-body imports.
  No handler calls a module-level helper that stayed behind, so no circular
  import is possible; `bridge_common` is dependency-free.
- **Facade compatibility**: no monkeypatch/patch target touches the moved
  names (`grep` verified: only `harness.agent_bridge.SubmissionManager` is
  patched, and that import stayed).

## Evidence

- Byte-equivalence: types block (192 lines) and handler block (1862 lines)
  are byte-identical to `harness/agent_bridge.py` at `HEAD`.
- Agent-bridge test files (`test_agent_bridge*`, `test_agent_ergonomics`,
  `test_agent_friction_fixes`, `test_agent_usability_p3/p5`,
  `test_boost_end_to_end`): 75 passed.
- Full suite: `python tools/run_tests.py` → **908 tests OK**.
- `ruff check harness/` clean.
- Smoke import: facade names resolve from both `agent_bridge` and
  `bridge_common` (same objects); `AgentBridge.act is AgentBridge.execute_action`;
  metaclass proxy (`AgentBridge.review_hypothesis`) still callable.

## Impact & follow-ups

- No behavior change; module layout only. New files are inside the
  `harness` package; `harness.agent_bridge` remains the canonical entry.
- PR-F: extract `UniversalRunner` pipeline/data clusters into mixin modules.
