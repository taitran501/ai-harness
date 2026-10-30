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

### Macro coverage and periodic retrospectives

The HUD and `get_macro_hypothesis_map` project nine canonical families from the
ledger: `data_and_splits`, `representation`, `architecture`, `target`,
`optimization`, `ensemble`, `post_processing`, `domain_signals`, and
`unclassified`. Record `family_id` and optional `source_refs` on each new insight;
unknown or missing families stay unclassified. Counts retain comparison groups;
the map never compares scores across split/data/scorer identities.

An open direction takes precedence in the family display (`active`), even alongside
supported mechanisms. Support requires a machine-evaluated decision whose receipt
and full-run dependencies still verify. Exhaustion applies only to registered
directions within their bounded probe budgets and stopping criteria, with terminal
evidence and documented revisit conditions. Unstructured rejection alone cannot
exhaust a family. An untried family has no registered insights or verified runs.

Every five newly verified full-run results since the last review, or a stagnation
pause not covered by that review, `RETROSPECTIVE_REQUIRED` blocks the next ordinary
full run and competition submission. Smoke/probe runs, failures, EDA and Bridge
action counts do not advance this cadence. Run-ID milestones persist across
sessions and prevent lost/tampered old evidence from discounting new results.
Inspection, notes and smoke probes remain available. The exception is a registered
stagnation recovery probe using its registered rule, pause generation and bounded
candidate budget; a freeform `recovery_probe` flag grants no exemption. Recording
a review does not clear the separate stagnation pause.

Retrieve a current `map_digest`, then submit a review through
`ai-harness-agent act --action record_macro_review --param-file retrospective.json`.
The CLI also accepts flat flags (nested values as JSON) and `--params`. The Bridge
provides session ID, task ID, review ID and current run count; explicit task/count
must match. Supported/rejected claims must cite real linked insight, decision and
candidate-run IDs. Empty evidence lists are allowed when there are no mechanisms
to cite. The untouched lead must identify an untried family while such gaps remain;
when all families have been explored, describe an untouched direction within a
family instead. Its priority and rationale are the scientist's judgment.

Example review body (replace the digest and evidence IDs with current values):

```json
{
  "map_digest": "<from get_macro_hypothesis_map>",
  "changes_since_last_review": "Representation probes are complete; target formulation remains untouched.",
  "supported_mechanisms": [],
  "rejected_mechanisms": [],
  "highest_priority_untouched": {
    "family_id": "target",
    "proposed_direction": "Test a target transformation against the fixed control",
    "rationale": "Residual slices suggest scale-dependent errors",
    "source_ref": "diagnostics:residual-slices"
  },
  "next_probe_rationale": "Changes the learning target rather than another nearby model parameter.",
  "next_action": "Register one bounded smoke probe before full validation.",
  "budget_limit": 1,
  "revisit_conditions": "Reassess after the registered stop criterion is reached."
}
```

Each mechanism claim has `family_id`, `insight_id`, `decision_id`, `run_id`, and
`mechanism_summary`. Reviews are appended to `research_memory.json` with covered
run IDs and stagnation events as the retrospective milestone, and included in
`RESEARCH_PROGRESS.md`. Stale map digests or invented support claims do not advance
the milestone. Coverage gaps are derived; high-priority leads require an explicit
sourced review, never keyword inference.

### Per-iteration actions

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

## Citation continuity and evidence status

External citation state lives under `.ai-harness/evidence/` and
`.ai-harness/citation_audits/`, outside completed run directories. Audit discovery
uses root Markdown, immediate `.ai-harness/*.md`, explicitly configured files,
and the active task's resolved `ResearchMemory.progress_path`; it does not choose
a progress file by traversing other task artifacts. A missing or ambiguous active
task remains an incomplete scope.

Historical `source_refs` and topic mentions remain provenance declarations until
their exact source version and claim binding are registered and reviewed. Never
promote a legacy citation to verified because a topic file happens to exist or a
keyword appears in it. Revalidate the audit/review receipt when an insight,
candidate, submission, or remote execution depends on it. A grounded hypothesis
can proceed to a bounded probe, while external reported results and demonstrated
local gains retain separate status. Receipt hashes detect edits and stale inputs;
they do not authenticate a verifier against full-workspace rewrite.
