# Repository work log

This journal tracks development of AI-Harness. Read the latest entry after a
restart and append an entry after a meaningful implementation/validation step
or before handing off. Earlier entries are historical evidence, not a claim
about the latest code. Task experiments have their own generated
`RESEARCH_PROGRESS.md` in the resolved artifact root.

Each entry records: request/observation, completed work, evidence, lesson,
remaining uncertainty, and next action. Record failures and abandoned directions
when they affect the next decision. Keep notes concise; do not copy transcripts,
raw datasets, or internal reasoning. Do not invent results or reconstruct lost
history from guesses.

## 2026-10-10 — Durable insight traces

- Observation: flat research notes did not preserve an insight's identity,
  planned test, experiment links, lessons, and derived directions.
- Completed: insight IDs and parent links; bounded probe plans; linked decisions
  with lessons and revisit conditions; paginated trace; pending-result triage;
  checkpoint insight IDs; generated workspace guidance and phase revision updates.
- Corrected: repeated-config metadata could erase insight links; reassigned unrun
  configs could attribute a new result to an older insight; smoke failures lacked
  experiment IDs in the research ledger.
- Evidence: 387 local tests passed in 140.262 seconds on a disposable checkout.
  Template single-fold smoke and a synthetic 12-row/two-fold NLP trace passed;
  restart recovered the linked result, lesson, derived insight and checkpoint.
  Hash checks preserved 43 configs/artifact/document files in the real workspace.
- Lesson: recoverable workflow records are necessary, but do not establish better
  agent reasoning or competition performance.
- Remaining: actual-agent continuity quality has not been measured.
- Next action: add a human-readable progress journal, then evaluate continuity
  with a fixed task, bounded budget, and an intentional context interruption.

## 2026-10-10 — Human-readable ongoing tracking

- Request: make completed work and next steps visible in Markdown while the agent works.
- Completed: generated task-level `RESEARCH_PROGRESS.md` with resume instructions,
  insight statuses and chronological notes/results. Research events and context
  reads refresh it; the HUD exposes its path. This development log and `AGENTS.md`
  now specify when assistants should record repository work.
- Boundary: unrecorded insights and ordinary code edits are not inferred
  automatically. Agents must summarize meaningful work through the research
  actions or this development journal. Tracking does not authorize continuous compute.
- Validation: 24 focused tests passed; all 390 tests passed in 151.749 seconds
  on an isolated checkout. Template smoke and synthetic two-fold/restart integration
  passed. Markdown refresh, evidence invalidation, concurrent notes, preservation
  of user-owned files, and recovery after report-write failure were exercised.
  Hash checks confirmed all 43 protected real-workspace files were unchanged.
- Lesson: event-driven Markdown makes recorded work visible, but recording discipline
  is still necessary; a generated journal cannot recover an insight never recorded.
- Next action: design the actual-agent continuity evaluation with a fixed task,
  interrupted context, repetition counts, interpretation completeness and a budget.
