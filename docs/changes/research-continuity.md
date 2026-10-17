# Research continuity: durable insight traces + human-readable tracking

- **Date**: 2026-10-10
- **PR**: research-memory series
- **Status**: merged
- **Area**: research memory / progress tracking

## 1. Why (motivation + thought)

Flat research notes did not preserve an insight's identity, planned test,
experiment links, lessons, and derived directions. Goal: make completed work and
next steps visible in Markdown while the agent works, and make research records
recoverable across restarts/compaction.

## 2. What (concrete changes)

- **Durable insight traces**: insight IDs + parent links; bounded probe plans;
  linked decisions with lessons and revisit conditions; paginated trace;
  pending-result triage; checkpoint insight IDs. Corrected: repeated-config
  metadata could erase insight links; reassigned unrun configs could attribute a
  new result to an older insight; smoke failures lacked experiment IDs in the
  research ledger.
- **Human-readable ongoing tracking**: generated task-level
  `RESEARCH_PROGRESS.md` with resume instructions, insight statuses, and
  chronological notes/results. Research events and context reads refresh it; the
  HUD exposes its path. The development log + `AGENTS.md` specify when to record
  repo work.
- Boundary: unrecorded insights and ordinary code edits are NOT inferred
  automatically — agents must summarize meaningful work through the research
  actions. Tracking does not authorize continuous compute.

## 3. Approach (decisions + rejected alternatives)

Event-driven Markdown + a stable, identity-carrying insight trace. Recoverable
records are necessary but do not by themselves establish better reasoning or
competition performance — so the follow-up measured continuity on a fixed task
with a bounded budget + an intentional context interruption.

## 4. Evidence

- 387 local tests (insight traces) + 390 tests (tracking) pass; template
  single-fold smoke + a synthetic 12-row/two-fold NLP trace; restart recovered
  the linked result, lesson, derived insight and checkpoint. Hash checks
  preserved 43 protected real-workspace files.

## 5. Impact & follow-ups

- **Follow-up**: the actual-agent continuity evaluation (fixed task,
  interrupted context, repetition counts, interpretation completeness, budget).
- **Downstream**: `docs/research_continuity.md` + `docs/research_trace.md`; the
  insight / probe / record / checkpoint actions in the SOP (the generated
  SKILL.md research section).
