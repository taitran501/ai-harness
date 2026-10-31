# Competition Research Drive & Evidence Discipline

This protocol applies whenever the attached task has an active external competition
or leaderboard. The agent owns scientific direction; AI-Harness preserves evidence,
blocks unsafe dispatch, and keeps competition state consistent across sessions.

Antigravity receives this protocol through the generated `/harness` skill and a
repeated `competition_research_directive` in the Context HUD, session-start response,
and candidate action observations. Use those reminders at session entry, after each
completed result, and before promotion, submission, or freeze decisions.

## Keep the competition active

Research continues through the official competition submission cutoff/endgame. A high
public score, a plateau, a completed session/day, exhausted daily quota, a `best`
label, or one supported candidate is not a completion signal. If the user withdraws
earlier, record a user-directed withdrawal or pause; do not label it a competition
freeze or finalization.

`freeze_candidate` creates a hard portfolio lock that can stop experiments and
submissions. It is an endgame operation, not a way to protect a good score. Keep
candidates provisional while the competition is open. Do not recommend or invoke a
freeze until the final submission cutoff/endgame and the user asks for final
selection. Finalize only after checking the official deadline, current submission
status, and the relevant evaluation results.

## Resume from the real objective and live state

At each new session or day:

1. Read `ai-harness-agent context`, the verified resume section in
   `RESEARCH_PROGRESS.md`, and the latest research checkpoint.
2. Carry forward the user's latest target and corrections. If the user says to stop
   mixing and test another mechanism, make that constraint visible in the next
   action and checkpoint; do not merely agree in prose.
3. Refresh the official competition metric, direction, deadline, rules, current
   leaderboard, latest submission statuses, and remaining quota. Use the live
   harness commands and official platform source; treat cached ranks and old chat
   summaries as stale until checked.
4. Review verified full runs, registered decisions, the macro hypothesis map,
   diagnostics, prior submission recipes, and relevant recent discussions before
   proposing another candidate.

Preserve the user's evolving goal. A past target such as Top 100 does not silently
replace a later target such as Top 10. If the current rank target is clear from the
active checkpoint and latest user correction, act on it without asking the user to
repeat it.

## Prefer real mechanism changes

Before an experiment or submission, compare it against the full verified research
history and recent candidates. State:

- **Mechanism family** and specific source: diagnostic slice, EDA result, or a
  directly inspected discussion with a stable reference.
- **What changes** and how the mechanism differs from the closest previous attempt.
- **Control**, official metric/direction, expected observation, alternative
  explanation, decision rule, compute budget, and stop condition.
- **How the result will change the next decision**, including what will happen if
  the expected observation does not occur.

Record the insight and plan before implementation or execution. Link the insight and
probe plan to the experiment config. Use semantic mechanism comparison in addition
to exact config fingerprints: renaming an existing blend, nudging its coefficient,
or changing a model seed does not make it a new direction. A parameter sweep is
valid when it tests a predeclared sensitivity question with a fixed control and
bounded rule; describe it as a sweep, not a breakthrough. Do not claim causality,
SOTA, or a breakthrough until the registered evidence supports that statement.

When the user explicitly asks for a new direction, prioritize the highest-value
untested mechanism or family from evidence and discussion mining. UCB and incumbent
score are inputs to that choice, not vetoes against a justified pivot. Avoid a
passive list: select and execute a bounded next probe when the existing authorization
allows it.

## Use submission quota as an experiment budget

Quota is a ceiling, not a requirement to submit every slot. Available daily slots are
also room to learn from meaningfully distinct candidates; do not preserve a safe
incumbent just because it already scores well. Allocate slots to candidates whose
hypotheses were registered before the result is known and which pass the task's
promotion and dispatch gates. Do not spend several slots exploring tiny coefficient
changes around the same prediction vectors unless that is the bounded, preregistered
question.

Maintain separate evidence columns:

1. **Local validation**: verified full-run metric, paired comparison, uncertainty,
   and split/scorer identity.
2. **External evaluation**: public score/rank, status, submission time, and current
   leaderboard frontier.
3. **Private evaluation**: only when actually reported by the platform.
4. **Discussion evidence and hypotheses**: sourced claims, not local results.

A public leaderboard change is useful external evidence, but it does not prove which
mechanism caused it or guarantee private-board performance. Repeatedly selecting
variants on public scores can overfit that board. State uncertainty plainly; never
invent certainty to justify preserving or discarding a candidate. Follow the
existing task-specific validation, candidate preflight, exact-payload approval, and
dispatch rules.

## Stagnation triggers reassessment, not a research stop

After three comparable non-improvements, stop promoting or submitting the stalled
line and inspect its diagnostics, research family, prior lessons, and relevant
discussions. The stagnation latch pauses promotion/submission for that line. In an
active competition task, it does **not** block full validation of a distinct,
preregistered mechanism. Generic non-competition tasks retain their configured
bounded recovery gate. Candidate dispatch remains paused until a verified,
machine-evaluated supported result clears the competition latch.

If `RETROSPECTIVE_REQUIRED` appears, complete the required macro review, then resume
the competition loop in the same work session. A retrospective is a decision
checkpoint, not a declaration that research is exhausted. A new session, remaining
quota, public rank, or narrative does not clear a runtime latch.

## Report without hype

For each result, label it as one of:

- a verified local run and its machine-evaluated decision;
- an external leaderboard observation;
- a sourced external claim;
- an untested hypothesis.

Do not call a candidate “best” without naming the metric, evidence scope and
comparison. Do not call a task frozen, finished, exhausted or final while the
competition is open. After every user correction, state the next concrete action
that implements it and preserve that action in the checkpoint.
