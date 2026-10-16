# Research continuity

The external assistant owns scientific reasoning. The harness retains evidence,
decisions, and restart instructions so that a fresh session can continue the work.
For insight IDs, bounded probe plans, linked experiment history, lessons and resume
triage, see [the research trace protocol](research_trace.md).
`AgentBridge.start_session()` only starts an active session. It does not invent an
experiment, synthesize a portfolio, or propose freezing it.

## Durable state and evidence

`research_memory.json` lives in the resolved task artifact root, alongside the
journal, outside immutable completed experiment directories. Updates use the same
cross-process lock and atomic JSON transactions as other harness state. Records are
scoped by task and scorer identity; changing either preserves the old ledger but
does not present its decisions as current-task evidence.

The ledger records proposals, declared scientific decisions, execution observations,
and checkpoints. Completed full runs are discovered from all artifact directories
and revalidated against manifest hashes whenever queried. Probes, failed runs, and
legacy artifacts cannot become verified full evidence. A prior decision remains an
interpretation; its evidence links are checked again when displaying context.

The compact HUD shows a bounded recent summary and the latest checkpoint. Search
reads the complete ledger and full-run history, including experiments older than the
journal's display window. Restart recovery therefore does not depend on one session
or on an agent remembering to read the old Markdown graveyard.

## Iteration protocol

Use `ai-harness-agent --project-root .`:

1. Read `context` after restart/compaction and before planning an iteration.
2. Call `act --action search_research_memory` with `query` describing the mechanism,
   error slice, model, or previous decision.
3. Call `act --action review_hypothesis` with an inline `config` object or a
   workspace-relative `config_path`. Read exact matches and related approaches.
4. Record `record_research_insight` and save its ID. Use `plan_research_probe` to
   state the control, one intervention, alternative explanation, expected result,
   budget, and decision/stop rules. Link `research.insight_id` in the config (or
   pass `insight_id` to the bridge action). Implement through
   `create_experiment` or `write_experiment_config`, then run smoke/full validation.
5. Call `record_research_decision` with `outcome` (`supported`, `rejected`,
   `inconclusive`, `deferred`), `conclusion`, optional verified experiment IDs in
   `evidence`, and `next_action`. Linked decisions also require `insight_id` and
   `lesson`; rejected/deferred directions require a `revisit_when` condition.
6. Call `record_research_checkpoint` with `summary`, `next_action`, and
   `open_questions` and active `insight_ids`. Use `trace_research_insight` to recover
   the thread; link a derived insight via `parent_insight_id`. Save enough context
   to resume without reconstructing the chat.

Example action payloads for `--params`:

On PowerShell, save these objects to UTF-8 JSON files and use `--param-file` to
avoid shell quoting ambiguity; see [runtime extensions](runtime_extensions.md).

```json
{"query":"class weighting minority recall", "limit":8}
```

```json
{"config_path":"configs/exp012_loss_weighted.yaml"}
```

```json
{"outcome":"inconclusive", "conclusion":"The full score stayed below the incumbent; minority recall improved but variance needs investigation.", "evidence":["exp012_loss_weighted"], "next_action":"Inspect per-fold error slices before changing the loss again."}
```

```json
{"summary":"Weighted loss improved minority recall without a full-score gain. The previous augmentation route is deferred pending a noise audit.", "next_action":"Compare error slices for the incumbent and weighted model, then propose one discriminating probe.", "open_questions":["Are the remaining errors label noise or missing context?"]}
```

No linked full run means the decision is a declared interpretation without verified
measurement. Do not call a crashing implementation a rejected mechanism.

## Repetition and related approaches

The fingerprint compares the resolved declared execution configuration, including
the task/scorer scope. Experiment name, parent, narrative, grounding prose, reporting
location, and resolved runtime provenance do not create a new intervention.
Creation, config writes, and full validation through Agent Bridge reject the same
declared configuration under another experiment ID unless an explicit
`revisit_reason` explains why it is intentional. The reason is retained in the config
when created/written. Editing a proposal and executing its own unrun config remains
possible; completed evidence is still immutable.

This fingerprint does not establish identical data contents or identical custom
implementation. Replication, changed data, and code revisions are legitimate reasons
to revisit. Lexical narrative overlap and identical declared model settings produce
related-approach warnings, not a permanent ban. The assistant must explain the
mechanistic difference; the harness cannot prove semantic equivalence or generate
a better hypothesis on the assistant's behalf.

## Stagnation and stopping

Progress is measured against the comparable incumbent, not just the initial
baseline. Full runs are compared within task/scorer/custom-scorer/data/split identity.
The context counts consecutive non-improvements across sessions; equal scores are
not progress even when the meaningful-improvement threshold is zero. Probes and
execution failures do not contribute to this scientific plateau count.

After three comparable non-improvements, the default advisory is
`REASSESS_RECOMMENDED`. Inspect verified diagnostic slices, search previous lessons,
enumerate alternative mechanisms, and choose a cheap probe that distinguishes them.
Record its control, observable result, promotion criterion, and budget/stop rule.
The advisory neither freezes the task nor demonstrates that the agent has reached
a capability ceiling. Explicit resource budgets, user cancellation, freeze approval,
and configured hard stopping rules retain their existing authority.

The bridge HUD, action observations, dashboard-generated prompts, phase prompts,
repository instructions, and newly attached workspace skills expose this protocol.
Agents bypassing these interfaces can still ignore it; direct runner invocations do
not enforce the novelty gate. Completed direct runs are discoverable as verified
history when the bridge is next read. Existing user prompt packs are preserved.
