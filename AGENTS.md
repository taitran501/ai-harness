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
- **[docs/agent_workflow.md](docs/agent_workflow.md)** — Full 8-phase workflow, detailed CLI guides, agent conventions, and multi-stage pipeline execution (`execution.mode: pipeline`, for train/blend/stack with per-stage audit trail).
- **[docs/research_continuity.md](docs/research_continuity.md)** — Durable research memory, novelty review, restart checkpoints, and stagnation reassessment.
- **[docs/research_trace.md](docs/research_trace.md)** — Insight IDs, bounded probe plans, per-insight lessons, lineage, and resume triage.
- **[docs/changes/](docs/changes/README.md)** — Per-change documentation (one file per change/PR: why/what/approach/evidence/impact). The **active** change log — write a new file here for every framework-repo change, before committing.
- **[docs/WORK_LOG.md](docs/WORK_LOG.md)** — Historical development journal (**ARCHIVED** 2026-10-17, read-only). New changes go to `docs/changes/`; do not append here.
- **[docs/CORRECTNESS_CONTRACT.md](docs/CORRECTNESS_CONTRACT.md)** — Schema v2 rules, immutable runs, and verification gates.
- **[docs/TASK_SPEC_TEMPLATE.md](docs/TASK_SPEC_TEMPLATE.md)** — Declarative `configs/task.yaml` specification schema.
- **[docs/plugin_architecture.md](docs/plugin_architecture.md)** — Pluggable task adapters, registration, and discovery.
- **[docs/kaggle_execution.md](docs/kaggle_execution.md)** — Kaggle CLI setup, data mounts, dependency policy, polling, and verified artifact import.
- **[docs/DESIGN_single_install.md](docs/DESIGN_single_install.md)** — Single-install architecture: one canonical framework checkout shared by all workspaces, `HARNESS_FRAMEWORK_DIR` resolution, and how sync/status behave under it.
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
7. **Written Progress**: Record research notes after meaningful observations/results and before stopping. The harness generates task-level `RESEARCH_PROGRESS.md`; its path is in the context HUD. Read its resume section after restart, then retrieve only the relevant insight history. For framework-repo code changes, write a dedicated change doc at `docs/changes/<YYYY-MM-DD>-<slug>.md` (copy `docs/changes/_TEMPLATE.md`) covering why / what / approach / evidence / impact, and add a row to the `docs/changes/README.md` index — do this **before** committing the PR. `docs/WORK_LOG.md` is now a read-only archive (no new entries).
8. **Execution Backend Policy**: A task contract may declare `execution: {backend: kaggle, locked: true}`. `run_full_validation` resolves the backend as explicit `--backend` param > `execution.backend` > `local`; when locked, other backends are rejected (`Backend lock violation`). Smoke tests/probes always run locally. When updating a competition workspace after a framework release, use `ai-harness sync` instead of hand-written multi-step sync prompts.
9. **Single-Install Framework**: All workspaces share ONE canonical framework checkout located by `HARNESS_FRAMEWORK_DIR` (falling back to a legacy per-workspace `<workspace>/ai-harness` clone only until migrated). `import harness` therefore always resolves to the same code, eliminating cross-workspace drift and the re-exec tax. Under single-install, `ai-harness sync` pulls the shared checkout but does **not** retarget the global editable install per workspace; `ai-harness status` reports a `framework` section (`single_install` flag + stale-install advisory). Remote dispatch bundles `harness/` from the installed (canonical) source, so cloud runs are unaffected.

---

## 3. Essential CLI Commands

```bash
# Attach task & view compressed context HUD
ai-harness attach --project-root . --task-config configs/task.yaml
ai-harness-agent --project-root . context

# Update the shared framework (canonical single-install, or legacy workspace clone)
ai-harness sync --project-root .
# In single-install (HARNESS_FRAMEWORK_DIR set) this pulls the canonical
# checkout and re-attaches; it does NOT retarget the global editable install.

# Start session & run experiments (flat CLI flags avoid PowerShell JSON escaping errors)
ai-harness-agent --project-root . start-session --goal "Improve metric" --max-experiments 3
ai-harness-agent --project-root . act --action create_experiment --branch feat --descriptor v1
ai-harness-agent --project-root . act --action get_next_experiment_id

# Execute smoke test & full cross-validation
ai-harness-run --project-root . --config configs/<exp_config>.yaml --debug --fold 0
ai-harness-run --project-root . --config configs/<exp_config>.yaml

# Export Kaggle Code Competition offline submission notebook
ai-harness export-kaggle-notebook --exp-dir .ai-harness/artifacts/<task>/<exp_id> --output submission.ipynb

# Web dashboard
ai-harness-dashboard --project-root .
```

---

## 4. Pre-Push Quality Gates

Run before pushing changes to this repository:
```bash
python tools/run_tests.py                # Full suite (all modules must pass 100%)
python tools/run_tests.py --fast         # Fast tier during iteration (skips 4 slow pipeline modules)
python tools/run_tests.py --smoke        # <10s critical-invariants gate for rapid iteration
ruff check harness/ tools/ tests/ --select F,E7 # Static gate: undefined names (F821) & pyflakes (zero-noise incl. tests/)
```

The suite is tiered: `--fast` runs everything except the four slow
full-pipeline modules (`test_kaggle_runtime`, `test_dashboard`,
`test_audit_gate`, `test_harness`); `--smoke` runs only the framework's
critical invariants (correctness contract, label-integrity audit,
anti-hallucination grounding gate, research-memory scope stability,
backend-lock policy) in ~2 s for instant feedback while iterating.
Always run the **full** suite before pushing; use `--fast` while
iterating; `--smoke` for a fast core sanity check. Shared workspace
fixtures live in `tests/fixtures.py` — reuse
`make_workspace()` / `workspace_with_repo_configs()` / `copy_repo_configs()`
instead of copy-pasting workspace setup into new test files.

In a competition workspace with attached data:
```bash
ai-harness-run --config configs/<exp_config>.yaml --debug --fold 0  # Smoke probe validation
```
