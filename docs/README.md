# AI-Harness Documentation

AI-Harness is an experimentation and validation harness for competitive machine learning and data science benchmarks. It separates the creative reasoning of AI coding assistants (or human engineers) from the execution, cross-validation, and audit pipeline.

This directory contains technical documentation, system specifications, and prompt contracts for engineers and AI agents working with the framework.

---

## 1. System Overview and Core Idea

In autonomous machine learning systems, two failure modes frequently occur:
1. **Rigid AutoML**: Traditional AutoML tools hardcode feature engineering, constrain model search to fixed templates, and take away the engineer's control.
2. **Unconstrained Code Generation**: Unconstrained AI coding assistants generate ad-hoc training scripts, easily leak validation labels into training sets, evaluate models inconsistently across folds, and fill their context windows with multi-megabyte log files.

AI-Harness resolves this tension through a decoupled **three-layer architecture**:

```text
Layer 1: Reasoning and Planning (AI Assistant or Human Engineer)
   - Formulates atomic hypotheses.
   - Diagnoses failure slices from structured diagnostics.
   - Designs feature pipelines, model architectures, and loss formulations.
   |
   v
Layer 2: Agent Bridge (CLI & HTTP Bridge)
   - Provides compressed context Heads-Up Displays (HUD ~1k tokens).
   - Manages experiment budgets and action permissions.
   - Recommends next-step exploration branches via Upper Confidence Bound (UCB).
   |
   v
Layer 3: Execution Runtime and Solution Tree
   - Enforces strict cross-validation and anti-leakage grouping invariants.
   - Tracks experiments in an immutable Solution Tree (DAG).
   - Enforces deterministic audit gates before submissions.
```

The assistant acts as the creative scientist, while the harness serves as the rigorous laboratory and auditor.

---

## 2. Core Features

- **Decoupled Architecture**: The core runtime is task-agnostic. Competition-specific datasets, custom metrics, and submission packaging are isolated in declarative configurations or standalone task plugins.
- **Solution Tree DAG**: Experiments are tracked as a directed acyclic graph rather than a flat run list. Parent-child relationships, hyperparameter changes, and score deltas are recorded deterministically.
- **Upper Confidence Bound (UCB) Exploration**: An exploration-exploitation heuristic guides the assistant to balance refining strong models against exploring untested feature or backbone families.
- **Fast Probe Verification**: Fast single-fold debug probes (`--debug --fold 0`) verify pipeline syntax, tensor shapes, and loss convergence before committing full multi-fold compute.
- **Strict Anti-Leakage Contracts**: Group-aware validation splitting prevents entity overlap across folds. Target metrics are computed strictly on out-of-fold predictions.
- **Immutable Artifacts and Audit Gates**: Completed validation runs receive cryptographic manifest hashes. Pre-submission audit gates verify prediction completeness, column ordering, ID validity, and domain ranges.
- **Web Dashboard**: An integrated React application provides real-time tree visualization, leaderboard ranking, phase progression, and diagnostic views.

---

## 3. Mental Model: Engine vs. Competition Workspace

It is essential to distinguish between the two operating environments:

1. **The AI-Harness Package (Engine)**:
   - This repository contains the reusable runtime, CLI tools, web dashboard, and plugin interfaces.
   - Installed globally or inside a Python environment via `pip install -e .` or `uv tool install`.

2. **The Competition Workspace**:
   - A separate project directory dedicated to a specific competition or dataset.
   - Contains competition data (`dataset/`), task configuration (`configs/task.yaml`), validation splits (`splits/`), and generated runs (`.ai-harness/artifacts/`).
   - Attached to the installed harness via:
     ```bash
     ai-harness attach --project-root . --task-config configs/task.yaml
     ```

---

## 4. The 8-Phase Competition Journey

Competitions progress through eight sequential, evidence-gated lifecycle phases:

| Phase | Identifier | Core Objective | Primary Tooling |
|---|---|---|---|
| **Phase 1** | `setup` | Task configuration, data discovery, and environment verification | `ai-harness attach`, `ai-harness-bootstrap` |
| **Phase 2** | `understand` | Exploratory data analysis, label distributions, anti-leakage audit | `ai-harness eda run`, `docs/TASK.md` |
| **Phase 3** | `baseline` | Minimal working model, smoke probe, and locked baseline anchor | `ai-harness-run --debug --fold 0`, `establish_baseline_anchor` |
| **Phase 4** | `experiment` | Iterative AIDE UCB exploration loop with single-variable interventions | `create_experiment`, `run_full_validation` |
| **Phase 5** | `select` | Identifying top models, error diversity, and candidate pool management | `promote_candidate`, `inspect_diagnostics` |
| **Phase 6** | `public_lb` | Tracking external evaluation scores (e.g., Kaggle public leaderboard) | `record_external_score` |
| **Phase 7** | `freeze` | Model blending, ensembling, and locking final candidate slots | `build_ensemble`, `freeze_candidate` |
| **Phase 8** | `finalization` | Deterministic submission verification, packaging, and sign-off | `prepare_submission`, `ai-harness-audit` |

---

## 5. Documentation Map

Use the following reading guide based on your current task:

| Document | Primary Audience | When to Read |
|---|---|---|
| **[changes/](changes/README.md)** | Engineers & Agents | **Active change log** — one file per framework-repo change/PR (why/what/approach/evidence/impact). Write a new file here for every change, before committing. |
| **[architecture.md](architecture.md)** | Engineers & Agents | Understanding the three-layer system design, Solution Tree DAG, and execution backends. |
| **[agent_workflow.md](agent_workflow.md)** | AI Coding Assistants | Practical guide to interacting with the harness via CLI, reading context HUDs, and managing sessions. |
| **[CORRECTNESS_CONTRACT.md](CORRECTNESS_CONTRACT.md)** | Engineers & Agents | Critical specification on run immutability, manifest hashing, scoring rules, and submission validation. |
| **[TASK_SPEC_TEMPLATE.md](TASK_SPEC_TEMPLATE.md)** | Engineers & Agents | Schema reference for writing `configs/task.yaml` and human context files (`docs/TASK.md`). |
| **[plugin_architecture.md](plugin_architecture.md)** | Engineers | Implementing custom task plugins (`TaskPlugin`), entry points, and workspace prompt packs. |
| **[knowledge_base.md](knowledge_base.md)** | Engineers & Agents | Modeling recommendations, backbone references, and loss formulation guidelines. |
| **[prompts/README.md](prompts/README.md)** | AI Coding Assistants | Central index of system prompt contracts for onboarding, iteration, ensembling, and auditing. |

---

## 6. Verification Scope Notice

AI-Harness is currently **strictly verified and benchmarked on Natural Language Processing (NLP / Text)** tasks (such as sequence classification, transformer fine-tuning, cross-lingual transfer, and prompt evaluation).

Support for other modalities (**Tabular, Computer Vision, Audio, Multimodal**) represents experimental work-in-progress on the development roadmap and is not yet benchmarked under competitive conditions. The core runtime remains flexible, allowing agents to author custom model factories and pipelines as needed.
