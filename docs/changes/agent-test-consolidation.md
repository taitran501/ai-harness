# Test consolidation: agent-bridge tests into a package, PR-number files renamed

- **Date**: 2026-10-18
- **PR**: [#102](https://github.com/taitran501/ai-harness/pull/102) (open)
- **Branch**: `refactor/agent-test-consolidation`
- **Status**: open
- **Commits**: — (recorded at merge)

## Why

The 2026-10-18 tech-debt audit found the agent-domain tests were spread
across 13 top-level files named by *when* they were written (phase
numbers, PR numbers, "p0_p1 fixes") rather than *what* they cover:
`test_agent_usability_p3/p5/p6/p7/p8`, `test_agent_bridge_p0_p1_fixes`,
`test_pr_a_correctness`, `test_pr_b_guardrails`,
`test_harness_p1_p2`, `test_harness_p3`, `test_pr_c_state_agent_friction`.
Six months later nobody can tell from the filename what regresses where,
and the phase/PR numbers carry no meaning outside their original
sessions. 115 agent tests in 13 flat files also made the tests/
directory noisy (87 files).

## What changed

**New package `tests/agent_bridge/`** (13 files moved, feature-based
names):

| Before | After |
|---|---|
| `test_agent_bridge.py` | `tests/agent_bridge/test_core_bridge.py` |
| `test_agent_bridge_anti_bypass.py` | `tests/agent_bridge/test_grounding_guards.py` |
| `test_agent_bridge_p0_p1_fixes.py` | `tests/agent_bridge/test_session_approval_fixes.py` |
| `test_agent_bypass_audit.py` | `tests/agent_bridge/test_cloud_hygiene_bypass.py` |
| `test_agent_ergonomics.py` | `tests/agent_bridge/test_hud_ergonomics.py` |
| `test_agent_friction_fixes.py` | `tests/agent_bridge/test_friction_fixes.py` |
| `test_agent_usability_audit.py` | `tests/agent_bridge/test_usability_phase4.py` |
| `test_agent_usability_p3.py` | `tests/agent_bridge/test_usability_phase3.py` |
| `test_agent_usability_p5.py` | `tests/agent_bridge/test_usability_phase5.py` |
| `test_agent_usability_p6.py` | `tests/agent_bridge/test_usability_phase6.py` |
| `test_agent_usability_p7.py` | `tests/agent_bridge/test_usability_phase7.py` |
| `test_agent_usability_p8.py` | `tests/agent_bridge/test_usability_phase8.py` |
| `test_pr_c_state_agent_friction.py` | `tests/agent_bridge/test_state_readiness_copilot.py` |

Phase numbers were preserved in the usability names (p3–p8 →
phase3–phase8) because they refer to real product phases documented in
each module's docstring — kept for traceability, demoted to a
clearly-labeled suffix.

**Renamed in place** (4 files, PR-number names → feature names):

| Before | After |
|---|---|
| `test_pr_a_correctness.py` | `tests/test_correctness_regressions.py` |
| `test_pr_b_guardrails.py` | `tests/test_guardrail_regressions.py` |
| `test_harness_p1_p2.py` | `tests/test_tree_journal_ensemble.py` |
| `test_harness_p3.py` | `tests/test_audit_gate_submission.py` |

**One mechanical fix, forced by the move**: three moved files
(`test_core_bridge`, `test_grounding_guards`, `test_friction_fixes`)
declared their own `SOURCE_ROOT = Path(__file__).resolve().parent.parent`
to reach the repo root (they copy the real `configs/` dir into temp
workspaces). One level deeper in the package that path resolved to
`tests/`. They now import the existing shared
`tests.fixtures.SOURCE_ROOT` constant instead — single source of truth,
depth-independent. No other test logic changed.

## Approach

- **Pure renames first, regrouping second** — the alternative (splitting
  115 tests across 13 files into new feature-grouped files) requires
  rewriting test-class boundaries and risks subtle cross-test
  dependencies; the filename problem is solved by the move + rename
  alone, with content byte-identical (verified via `git diff -M100%`:
  14 of 17 moves at 100% similarity, 3 with the single import-line fix
  above).
- **Why a package and not just renames in place**: 13 files is already
  15% of the tests/ directory; the agent domain is the one with the
  highest churn (usability phases keep landing). A subpackage keeps
  future additions contained and matches how the code under test is
  organized (`harness/agent_bridge.py` + `harness/bridge_*.py`).
- **Depends on PR #101** (recursive dotted-name discovery in
  `tools/run_tests.py`) — on main the moved modules would silently stop
  being collected. Verified merged before this branch was cut.
- No cross-file test imports were affected: all absolute
  (`from tests.fixtures import ...`, `from tests.test_correctness_contracts
  import task`); no relative imports exist in the suite.

## Evidence

- `git diff -M100% --cached --stat`: 17 renames, 0 insertions / 0
  deletions for the pure renames; the 3 fixed files show exactly the
  import-line change.
- Full suite: `python tools/run_tests.py` → 908 tests OK (200.0 s),
  87 modules discovered (13 under `tests.agent_bridge.*`).
- `python -m pytest tests/ -q --co` → 925 collected, identical to the
  pre-move count measured on main (the 925-vs-908 pytest/unittest gap
  is pre-existing: pytest also collects non-TestCase `test_*` items).
- `python tools/run_tests.py tests/agent_bridge/test_usability_phase5.py`
  and `tests.agent_bridge.test_core_bridge` both resolve and pass.
- `--fast` / `--smoke` tiers unaffected (no moved module is a tier
  member).
- `ruff check tests/ tools/` clean (I001 import order respected).

## Impact & follow-ups

- Discoverability: agent regressions now live under one package with
  feature names; new agent tests go in `tests/agent_bridge/`.
- Historical docs (WORK_LOG, older change docs) reference the old file
  names — intentional, not updated.
- Follow-up (not this PR): deeper feature-regrouping of the usability
  phase files if their content itself is reorganized; god-function
  decomposition (C) is independent.
