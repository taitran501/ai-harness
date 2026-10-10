# AGENTS.md — Canonical Assistant Context & Navigation Map

Canonical context file for AI coding assistants (Antigravity, OpenAI Codex, Claude Code, Cursor).

> **Read this file like a table of contents.** Detail lives in `docs/` and task skills. This file stays lean (~100 lines) per OpenAI's [harness-engineering](https://openai.com/index/harness-engineering/) practice to maximize task context economy.

> [!IMPORTANT]
> **Implementation and Verification Scope**
> Competition-level verification currently covers **NLP / Text** tasks (sequence classification, transformer fine-tuning, prompt evaluation).
> **Tabular has implemented and locally tested paths** for categorical handling, scikit-learn pipelines, LightGBM, and XGBoost; its performance remains unverified in competitive conditions. **Vision, Audio, Multimodal** remain experimental WIP / Roadmap.

---

## 1. Documentation & Architecture Map

- **[docs/architecture.md](docs/architecture.md)** — 3-layer architecture, decoupling, and empirical evolution roadmap.
- **[docs/agent_workflow.md](docs/agent_workflow.md)** — Full 8-phase workflow, detailed CLI guides, and agent conventions.
- **[docs/research_continuity.md](docs/research_continuity.md)** — Durable research memory, novelty review, restart checkpoints, and stagnation reassessment.
- **[docs/research_trace.md](docs/research_trace.md)** — Insight IDs, bounded probe plans, per-insight lessons, lineage, and resume triage.
- **[docs/WORK_LOG.md](docs/WORK_LOG.md)** — Repository development journal: completed changes, validation, and next steps. Update after meaningful implementation steps and before handing off; do not invent results.
- **[docs/CORRECTNESS_CONTRACT.md](docs/CORRECTNESS_CONTRACT.md)** — Schema v2 rules, immutable runs, and verification gates.
- **[docs/TASK_SPEC_TEMPLATE.md](docs/TASK_SPEC_TEMPLATE.md)** — Declarative `configs/task.yaml` specification schema.
- **[docs/plugin_architecture.md](docs/plugin_architecture.md)** — Pluggable task adapters, registration, and discovery.
- **[docs/kaggle_execution.md](docs/kaggle_execution.md)** — Kaggle CLI setup, data mounts, dependency policy, polling, and verified artifact import.
- **[docs/prompts/README.md](docs/prompts/README.md)** — Prompt library across onboarding, iteration, and ensembling phases.

---

## 2. System Layers & Working Principles

```text
Assistant / Engineer (Layer 1)  ──>  Agent Bridge (Layer 2)  ──>  Harness Runtime (Layer 3)
(Scientific hypotheses, review)     (Context HUD, session gates)    (Runner, DAG, audit gates)
```

1. **Non-Prescriptive Autonomy**: In the AIDE/Agentic ML paradigm, the assistant is the scientist. The harness is an execution and audit layer—it **never hardcodes feature preprocessing** (e.g. forced scalers) or constrains model architectures.
2. **Context Economy**: Never print or dump raw datasets into prompt memory. Use `ai-harness-agent context` for compressed HUD state (~1k tokens).
3. **Atomic Interventions**: Modify one hypothesis variable at a time so performance deltas are cleanly attributable.
4. **Fast Probes Before Full Runs**: Always verify with a single-fold smoke test (`--debug --fold 0`) before committing full multi-fold compute.
5. **Immutable Evidence**: Full runs under `.ai-harness/artifacts/<task-id>/<exp_id>` are strictly immutable. Never mutate completed evidence folders.
6. **Research Continuity**: After restart/compaction, read `context`, the latest checkpoint, and pending results in `research_frontier`. Before brainstorming/implementation, use `search_research_memory` and `review_hypothesis`. Record an insight with `record_research_insight`, plan its bounded discriminating probe with `plan_research_probe`, and link `research.insight_id` in the experiment config. Use `trace_research_insight` to recover its history. Record a decision with lesson/next action and checkpoint after each iteration; link derived insights to their parent. An intentional repeated configuration requires `revisit_reason`; related approaches need a mechanism comparison. Plateau calls for diagnostics and reassessment, never automatic freeze.
7. **Written Progress**: Record research notes after meaningful observations/results and before stopping. The harness generates task-level `RESEARCH_PROGRESS.md`; its path is in the context HUD. Read its resume section after restart, then retrieve only the relevant insight history. For repository code changes, append a concise entry to `docs/WORK_LOG.md` with work, evidence, lesson, and next action.

---

## 3. Essential CLI Commands

```bash
# Attach task & view compressed context HUD
ai-harness attach --project-root . --task-config configs/task.yaml
ai-harness-agent --project-root . context

# Start session & run experiments
ai-harness-agent --project-root . start-session --goal "Improve metric" --max-experiments 3
ai-harness-agent --project-root . act --action create_experiment --params '{"branch": "feat", "descriptor": "v1"}'

# Execute smoke test & full cross-validation
ai-harness-run --project-root . --config configs/<exp_config>.yaml --debug --fold 0
ai-harness-run --project-root . --config configs/<exp_config>.yaml

# Web dashboard
ai-harness-dashboard --project-root .
```

---

## 4. Pre-Push Quality Gates

Run before pushing changes to this repository:
```bash
python -m unittest discover -s tests -p "test_*.py"   # Full test suite (330+ tests must pass 100%)
ai-harness-run --config configs/exp000_template.yaml --debug --fold 0  # Smoke probe validation
```
