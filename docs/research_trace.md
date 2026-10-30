# Research trace: from insight to the next experiment

The ledger already retains proposals, execution observations, scientific decisions
and checkpoints. Flat notes are not enough to reconstruct which insight motivated
a probe, what the probe was meant to distinguish, or which lesson led to a new
direction. Research traces add explicit links to the existing ledger, outside
immutable experiment folders. They store concise observations and decisions, not
chat transcripts or private internal reasoning.

## Patterns inspected in other repositories

Sources inspected on 2026-10-10. This is a comparison of workflow mechanisms, not
a claim of superior competition scores or agent reasoning quality.

| Repository | Mechanism in the inspected code | What AI-Harness adopts |
| --- | --- | --- |
| [AIDE journal](https://github.com/WecoAI/aideml/blob/main/aide/journal.py) and [agent](https://github.com/WecoAI/aideml/blob/main/aide/agent.py) | Nodes link parents/children and retain plans, execution errors, analysis and metrics; improvement prompts receive a journal summary. | Explicit insight lineage, with experiments and interpretations retrievable by ID. Keep execution failure distinct from scientific rejection. |
| [AI Scientist experiments](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_experiments.py) and [ideas](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/generate_ideas.py) | Bounded experiment runs feed results back into planning and ask for per-run notes; ideation uses prior ideas and reflection. | Record what a result taught us and why a derived direction follows; preserve the link from the prior lesson. |
| [autoresearch program](https://github.com/karpathy/autoresearch/blob/master/program.md) | A compact result ledger records commit, metric, resource use, and keep/discard/crash; experiments use a fixed training budget. | Declare a bounded probe and explicit decision/stop rules before a linked intervention. Actual resource enforcement remains in the harness session ledger. |

AI-Harness retains its task/scorer identity, verified sealed evidence and external
assistant ownership of scientific decisions. It does not adopt an unbounded run
loop, automatic model selection, or destructive rollback of completed evidence.

## One insight, one durable identity

1. Search `search_research_memory` and review the hypothesis before proposing a
   new direction. Related mechanisms are advisory; a renamed insight does not
   make the same execution configuration novel.
2. Call `record_research_insight` with `title`, `observation`, `mechanism`, and
   `next_action`, plus a canonical `family_id` and optional provenance `source_refs`.
   Legacy or unknown families remain `unclassified`. An observation is a declared interpretation unless backed by
   optional verified full-run IDs in `evidence`. Save the returned `id`.
3. Call `plan_research_probe` with that `insight_id` and all seven plan fields:
   `control`, one `intervention`, `expected_observation`, `alternative_explanation`,
   `decision_rule`, `budget`, and `stop_rule`. The plan must explain how the
   intervention distinguishes explanations, rather than just "try a new model".
4. Link an experiment with `research.insight_id` in YAML, or pass `insight_id`
   to `create_experiment` / `write_experiment_config`. Bridge creation, writes,
   and full validation require a recorded plan for linked configurations. A
   proposal retains the plan ID used when it was registered. Scientific metadata
   is excluded from the execution fingerprint, so links cannot bypass novelty.
5. Run smoke/probe validation before authorized full compute. Execution failures
   and smoke results remain engineering observations. They do not verify a full
   score or reject the scientific mechanism.
6. Interpret the result with `record_research_decision`, including `insight_id`,
   `outcome`, `conclusion`, verified run IDs in `evidence`, `lesson`, and
   `next_action`. Rejected/deferred directions also require `revisit_when`: the
   changed condition or new evidence that would justify revisiting the direction.
7. If the lesson motivates another mechanism, record a new insight with
   `parent_insight_id`. Save a checkpoint with active `insight_ids` and unresolved
   questions so the next session recovers both the task and the research threads.

These fields record a reviewable scientific contract. Textual probe budgets do
not reserve compute or enforce termination: existing session/resource budgets,
approval rules and user cancellation continue to govern execution.

## Example: recall improved but the full metric did not

This is an illustrative NLP example, not a measured repository result. The
observation is that minority recall is poor; the proposed mechanism is loss
imbalance. A weighting probe should distinguish this from annotation noise.
If full evidence shows a recall gain with no reliable full-metric improvement,
the lesson is narrower than "weighting never works": it did not resolve the
current bottleneck under this configuration. The next direction may be a label
audit, linked to this lesson rather than a new weighting idea with a new name.

Save the following objects as UTF-8 JSON and use `--param-file` on PowerShell.
Replace `INSIGHT_ID` with the ID returned by the first action.

```json
{
  "title": "Minority recall bottleneck",
  "observation": "Verified incumbent diagnostics show low minority recall.",
  "mechanism": "Underweighted minority examples may explain the recall gap.",
  "next_action": "Compare weighting against the fixed incumbent on unchanged folds.",
  "evidence": ["exp001_baseline"]
}
```

```powershell
ai-harness-agent --project-root . act --action record_research_insight --param-file insight.json
```

```json
{
  "insight_id": "INSIGHT_ID",
  "control": "exp001_baseline, unchanged data, folds, features and seed",
  "intervention": "Change only class weighting",
  "expected_observation": "Minority recall rises consistently across folds",
  "alternative_explanation": "Annotation noise rather than loss imbalance",
  "decision_rule": "Require a full-metric gain and stable minority improvement before promotion",
  "budget": "One fold smoke, then one full validation within the authorized session budget",
  "stop_rule": "Stop this probe on execution error or exhausted budget; diagnose before retrying"
}
```

```powershell
ai-harness-agent --project-root . act --action plan_research_probe --param-file probe.json
ai-harness-agent --project-root . act --action trace_research_insight --param-file trace.json
```

`trace.json` is `{"insight_id":"INSIGHT_ID","offset":0,"limit":20}`.
The trace returns the original observation, plans, linked proposals/execution
observations, verified results, decisions, derived insights and current state.
Follow `next_offset` for older/later pages; pages are chronological and capped at
30 events. The latest results/children are bounded separately, with total counts.

## Resume and rebrainstorm

### Human-readable progress file

The resolved task artifact root contains `RESEARCH_PROGRESS.md`, alongside
`research_memory.json`. Every recorded insight, plan, proposal, execution,
decision or checkpoint refreshes this generated Markdown journal. Calling
`context` refreshes it too, so completed direct-run results appear even when
they did not pass through the Bridge. This is event-driven tracking, not a
background agent or an endless execution loop.

The file has a resume section (latest checkpoint and pending results), a table
of all insight statuses, and a chronological journal containing observations,
probe contracts, work/results, lessons and next actions. The HUD returns its
absolute `progress_path`. Task/scorer scope is explicit; switching scope changes
the current Markdown view while retaining earlier records in the JSON ledger.

Record a note after a meaningful finding or result, before pivoting, and before
stopping/compaction. A useful note states what happened, evidence, what it taught
us and the next bounded action. Record rejected approaches as well as successes.
Agent work that does not call these recording actions is not automatically
summarized: filesystem edits and unrecorded insights are outside this journal.

JSON is authoritative; edit notes through the Bridge rather than this generated
file. A pre-existing unmarked Markdown file is preserved, and report write failures
surface as `progress_warning` without duplicating or losing committed JSON notes.
The full journal can grow; read the resume section and use scoped search/trace
instead of putting the entire file into agent context.

For development of this repository, document framework-repo changes in
[docs/changes/](changes/README.md) — one file per change, from
`docs/changes/_TEMPLATE.md`, indexed in `docs/changes/README.md` — with what
changed, why, the approach and the evidence. `docs/WORK_LOG.md` is a
read-only archive; do not append to it.

`context` adds `research_frontier`, `open_insight_count`, `unreviewed_results`,
and `unreviewed_result_count`. Each frontier item carries the observation,
mechanism, the assistant's concrete `next_action`, and a `required_step` indicating
the missing workflow work. The bounded HUD prioritizes:

- `needs_evidence_review`: a decision's evidence no longer passes verification.
- `needs_interpretation`: full results exist without an insight-linked decision.
- `needs_diagnosis`: execution or probe output needs interpretation before another
  scientific claim; an implementation crash is not a rejected mechanism.
- `needs_probe_plan`: an insight needs a discriminating, bounded plan.
- `ready_for_probe`: a plan is recorded; implement its single intervention.

An inconclusive decision keeps the thread open with its declared next action.
Supported, rejected and deferred threads remain searchable and traceable. Their
status is the assistant's recorded interpretation, not automatic candidate
promotion or proof of a causal mechanism. Evidence links are revalidated live.
Decisions across iterations retain all acknowledged run IDs, so a later decision
does not accidentally make previously interpreted results appear new again.

Read pending results before proposing another neighboring approach. When the
verified incumbent plateaus, inspect diagnostic slices, retrieve old lessons and
use a new bounded probe to distinguish explanations. Trace the parent lesson and
state what changed. The harness does not invent the next hypothesis, conclude a
capability ceiling, or freeze a task because an insight failed.

## Compatibility and verification boundary

The existing append-only ledger is extended without rewriting old records.
Legacy unlinked configs and decisions continue to work. Direct full runs are
discoverable as pending results; a run that records `research.insight_id` is also
linked to its insight. Bridge smoke/failure observations link by experiment ID.
Direct runner invocations still bypass Bridge plan/novelty gates. Existing user
prompt packs and workspace skills are preserved; newly attached skills expose
the trace actions, and rendered phase/dashboard prompts consume the same HUD.
Research notes now participate in phase revision tracking so a recorded lesson
can make an iteration phase revisitable without waiting for more expensive runs.

Tests exercise restart recovery, parent/child lineage, plan gates, repeated-config
checks, live evidence invalidation, pagination, old direct results, bridge smoke
failure linkage, generated workspace skills and phase revisions. These are
workflow/correctness checks; improved reasoning or competition scores require
separate bounded evaluations with an actual agent and task.

## External citations and claim review

`source_refs` identify where an idea came from; by themselves they do not verify
the source or the claim. Register external discussion payloads with an explicit
schema adapter so the registry can preserve the raw snapshot hash, normalized
segments, author metadata, stable comment IDs (or snapshot-bound locators), and
retrieval completeness. Registration is explicit and does not search artifact
trees for topic files.

For a claim that must pass a strict gate, provide a versioned claim record with
the exact claim text and one or more evidence references. Each reference pins a
source version, locator, excerpt, and excerpt offsets. A separate verifier
execution records the current claim/evidence hashes and a structured verdict;
persona switching does not count as independent review. Missing execution
provenance remains `not_reviewed`.

Keep these outcomes distinct:

- **Grounded hypothesis**: the source supports the motivation, and a bounded
  probe can test local applicability.
- **External reported result**: the source supports what its author reported;
  it does not establish a gain on this task.
- **Demonstrated local gain**: the existing decision-rule evaluator supports the
  result on comparable, sealed local evidence.

The citation audit reports `complete`, `partial`, or `nothing_scanned` coverage;
`verified`, `unverified`, or `invalid` evidence integrity; and
`supported`, `contradicted`, `insufficient`, or `not_reviewed` claim support.
Keyword matching is advisory lint only. Correct a citation by creating a new
reviewed version while preserving the prior verdict and receipt.
