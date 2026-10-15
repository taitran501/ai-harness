# ai-harness

Modular experimentation harness and dashboard for competitive machine learning and data science projects (Kaggle, Codabench, DrivenData, and academic benchmarks).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dashboard](https://img.shields.io/badge/Dashboard-React%20%2B%20Vite-purple.svg)](dashboard/)

## Overview

ai-harness provides experiment tracking, validation orchestration, and agent integration for competitive data science workflows. It connects local runs, cloud GPU execution (Modal, Kaggle), and AI coding assistants (Antigravity, Codex, Claude Code) through a structured solution tree and verification gates.

The core runtime is task-agnostic. Task-specific data loading, model architectures, and competition metrics are loaded via standalone task adapters.

> **Scope Notice**: Competition-level validation currently covers **NLP / Text** tasks. **Tabular already has implemented and locally tested execution paths**: native categorical handling, scikit-learn pipelines, LightGBM, and XGBoost, including regression/classification integration tests. Tabular competition-level generalization is still unverified; Vision, Audio, and Multimodal remain experimental roadmap work. Local engineering tests and competition results are separate evidence.

### Agent research continuity

The harness preserves proposals, decisions, execution observations, and restart
checkpoints across sessions. Read `ai-harness-agent context`, search older work with
`search_research_memory`, and use `review_hypothesis` before implementation. Exact
declared configuration repeats require a `revisit_reason`; related approaches get
an advisory comparison. Plateau triggers reassessment against the comparable
incumbent and verified diagnostics, without automatically freezing the task.
See [the research continuity protocol](docs/research_continuity.md) and
[runtime extensions](docs/runtime_extensions.md).

On Windows PowerShell, put action parameters in a UTF-8 JSON file to avoid shell
quoting ambiguity:

```powershell
ai-harness-agent --project-root . act --action record_research_checkpoint --param-file checkpoint.json
ai-harness-run --project-root . --config configs/exp012_loss_weighted.yaml --debug --fold 0 --unbuffered
```

Fold progress is always flushed to stdout and `run.log`. `--unbuffered` additionally
flushes redirected Python stdout/stderr; managed Python subprocesses use `-u`.

## Quick start

### CLI Installation

Install globally using `uv` (recommended):

```bash
uv tool install git+https://github.com/taitran501/ai-harness.git
```

Or install in your active Python environment:

```bash
pip install git+https://github.com/taitran501/ai-harness.git
```

For local development:

```bash
git clone https://github.com/taitran501/ai-harness.git
cd ai-harness
pip install -e .
```

### Attach to a workspace

Attach the harness to your competition repository:

```bash
ai-harness attach --project-root . --task-config configs/task.yaml
```

This writes `.ai-harness/project.yaml` and configures the agent skill under `.agents/skills/ai-harness/`.

### AI agent integration

Query task state and retrieve context for your coding agent:

```bash
ai-harness-agent context
```

Start an agent session with an experiment budget:

```bash
ai-harness-agent start-session --goal "Improve validation score" --max-experiments 3
```

Launch the local web dashboard:

```bash
ai-harness-dashboard
```

### Diagnose before you dispatch (recommended first step for agents)

Run one command to get install + workspace + cloud health as JSON:

```bash
ai-harness status              # JSON (default)
ai-harness status --markdown   # human-readable summary
ai-harness status --no-cloud   # skip cloud checks (offline-friendly)
```

If a Kaggle kernel fails, inspect its state **without re-dispatching** (no new push, no quota burned):

```bash
ai-harness kaggle status <owner/kernel-slug>   # current state
ai-harness kaggle log    <owner/kernel-slug>   # failure traceback/stderr
ai-harness kaggle output <owner/kernel-slug>   # list output files
```

Preview a cloud dispatch without pushing anything:

```bash
ai-harness-run --config configs/exp001.yaml --backend kaggle --dry-run
```

## Features

| Component | Description |
|---|---|
| **NLP Transformer Core** | Pretrained encoder fine-tuning (DeBERTa, RoBERTa, ModernBERT), LoRA/QLoRA via Unsloth, and LLM prompting |
| **Validation Protocols** | Stratified, grouped, and time-series cross-validation with debug probe runs (`--debug --fold 0`) |
| **Solution Tree (AIDE)** | Lineage DAG tracking with Upper Confidence Bound (UCB) next-step exploration recommendations |
| **SWE-bench Style Audit** | Deterministic pre-submission audit gate preventing metric leakage and corrupted artifacts |
| **Execution Backends** | Run locally, or dispatch jobs to Modal.com and Kaggle kernels with checksum verification |
| **Interactive Dashboard** | React + Vite UI for tracking iterations, visualizing tree lineages, and inspecting metrics |

## Repository structure

```text
ai-harness/
├── harness/           # Core runtime, runner, and plugin API
├── tools/             # Agent bridge, dashboard, and CLI entry points
├── dashboard/         # React dashboard source
├── configs/           # Reference task and experiment configurations
├── task_plugins/      # Task-specific competition adapters
└── .agents/skills/    # Pre-built agent skills for Antigravity, Claude Code, and Codex
```

## License

This project is licensed under the [MIT License](LICENSE).
