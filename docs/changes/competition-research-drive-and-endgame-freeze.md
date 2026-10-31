# Competition Research Drive and Endgame Freeze Policy

- **Date**: 2026-10-31
- **PR**: — (not opened)
- **Branch**: `main`
- **Status**: Working tree
- **Commits**: —

## Why

Agents could acknowledge requests for new directions, then return to nearby blend or
rank-nudge variants. The active-competition stagnation latch also blocked ordinary
full validation after three failed comparisons, while the state-machine HUD
recommended freezing a final portfolio as soon as candidates existed. In addition,
the continuation rule lived deep in a broad skill and could be missed at session
entry or after a new result. Those gaps could turn a plateau or a good interim public
score into a premature end to research.

## What changed

- Added `docs/competition_research_protocol.md` with a durable competition objective,
  novelty, quota, evidence-separation, stagnation, and endgame policy.
- Updated `AGENTS.md`, research-continuity docs, the phase-7 prompt, the workflow
  guide, and generated/checked-in skills to carry the latest user objective and
  corrections into the next action.
- Changed stagnation handling for competition tasks: a full run after a pause must
  link the exact insight, probe plan, and registered decision rule; verified support
  clears the promotion pause. Generic non-competition tasks retain the bounded
  recovery gate.
- Changed the competition-state HUD to recommend continued research while
  candidates remain provisional, instead of prompting an early freeze.
- Changed the Context HUD to distinguish competition research from the generic
  recovery pause.
- Promoted the competition loop to the top of the generated Antigravity `/harness`
  skill, with explicit triggers for session start/resume, result review, and
  candidate decisions. The active competition directive is also repeated in the
  Context HUD, `start-session` response, and run/promotion/submission/freeze
  observations so an agent sees it again at the point of decision.
- Reduced the generated `/ai-harness` compatibility alias to a short router so it
  no longer duplicates the full `/harness` instruction set. Corrected the discussion
  sort and candidate-preflight command examples against the current CLI, and aligned
  submission wording with the registered promotion gate while retaining public-LB
  results as external feedback.

## Approach

The fix separates three different decisions: whether to keep researching, whether to
promote/submit a candidate, and when to lock the final portfolio. Stagnation still
blocks promotion and submission of the stalled line. Active competitions can keep
running full validation when each candidate is linked to a preregistered insight,
probe and machine-evaluable rule. A machine-supported result can reopen promotion.
UCB remains a recommendation; it no longer outranks the user's latest correction or
an evidence-backed untouched mechanism.

The final-selection phase is now presented as an endgame action at official close.
The freeze action itself remains approval-gated and callable by an explicitly
approved operator; the agent instruction and automatic HUD recommendation defer it
until the endgame.

## Evidence

- Reviewed the source diff and generated skill output. Python syntax compilation,
  `ruff check harness/ tools/ tests/`, and `git diff --check` passed; the test suite
  was not run.

## Impact & follow-ups

Existing competition workspaces receive the updated dynamic skill when they sync or
reattach the framework. The current workspace and global Antigravity `/harness` skill
are regenerated from the canonical source. Candidate dispatch remains blocked by the
normal evidence, preflight and approval gates during stagnation. The runtime does not
yet enforce the official deadline inside `freeze_candidate`; the endgame-only
restriction is carried by repository, skill, phase-prompt, and repeated HUD/action
instructions while the explicit operator approval gate remains in force.
