# ai-harness

Modular experimentation harness and dashboard for competitive machine learning and data science projects (Kaggle, Codabench, DrivenData, and academic benchmarks).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dashboard](https://img.shields.io/badge/Dashboard-React%20%2B%20Vite-purple.svg)](dashboard/)

## Overview

ai-harness provides experiment tracking, validation orchestration, and agent integration for competitive data science workflows. It connects local runs, cloud GPU execution (Modal, Kaggle), and AI coding assistants (Antigravity, Codex, Claude Code) through a structured solution tree and verification gates.

The core runtime is task-agnostic. Task-specific data loading, model architectures, and competition metrics are loaded via standalone task adapters.

> **Scope Notice**: ai-harness is validated and benchmarked for **Natural Language Processing (NLP / Text)** tasks (e.g., text sequence classification, transformer fine-tuning, prompt evaluation). Other modalities (Tabular, Vision, Audio, Multimodal) are experimental roadmap items and are not yet validated in competition settings.

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
