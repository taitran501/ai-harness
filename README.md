# SemEval-2027: StereoQueerEval

Multilingual benchmark for evaluating stereotypes and queer representations across online discourse in English, Italian, and Dutch.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Harness: ml--harness](https://img.shields.io/badge/Execution%20Engine-ml--harness-blueviolet.svg)](ml-harness/)
[![Validation: 5--Fold CV](https://img.shields.io/badge/Validation-5--Fold%20CV-orange.svg)](splits/)

---

## 1. Overview

This repository contains the end-to-end competition pipeline for **SemEval-2027: StereoQueerEval**. The benchmark evaluates computational models on their ability to detect subtle stereotyping, bias, and sentiment towards LGBTQ+ communities in social comments across three languages:
- **English (`EN`)**
- **Italian (`IT`)**
- **Dutch (`NL`)**

The project separates competition data and modeling logic from the execution engine:
- **Competition Data & Models**: Reside at the root level (`dataset/`, `splits/`, `src/`, `configs/`).
- **Pluggable Execution Harness**: Resides in [`ml-harness/`](ml-harness/), a modular and task-agnostic experimentation framework providing automated cross-validation, AIDE solution tree search, SWE-agent audit gates, and an interactive dashboard.

---

## 2. Directory Layout

```text
semeval_2027/
├── dataset/                     # Tracked competition datasets
│   ├── StereoQueerEval_EN_training.tsv
│   ├── StereoQueerEval_IT_training.tsv
│   ├── StereoQueerEval_NL_training.tsv
│   └── StereoQueerEval_training.zip
├── splits/                      # Leak-free cross-validation assignments
│   ├── fold_assignments.tsv     # 5-fold video-grouped split mapping
│   └── split_manifest.json      # Distribution and stratification hashes
├── configs/                     # Competition task and experiment configs
│   ├── task.yaml                # StereoQueerEval metric & subtask schema
│   ├── base.yaml                # Base training configuration
│   ├── exp000_template.yaml     # Epistemic grounding experiment template
│   └── exp001_tfidf_baseline.yaml # Verified TF-IDF baseline
├── src/                         # Competition-specific modeling & feature logic
│   ├── data.py                  # Dataset loading and multilingual preprocessing
│   ├── features.py              # Context concatenation and n-gram extraction
│   ├── models.py                # Classifier definitions (Classical & Transformers)
│   └── utils.py                 # Metric computation and artifact serialization
├── artifacts/                   # Experiment artifacts, OOF predictions & tree logs
│   ├── journal.json             # AIDE Solution Tree DAG
│   └── exp001_baseline_tfidf/   # Baseline model artifacts
├── tools/                       # CLI forwarders to ml-harness
│   ├── agent_bridge_cli.py      # Antigravity/Codex bidirectional agent bridge
│   ├── journal_tool.py          # Solution Tree and UCB branch recommender
│   ├── audit_gate.py            # SWE-Agent deterministic submission gate
│   ├── dashboard_server.py      # Interactive React + REST server
│   └── submission_tool.py       # Submission packaging utility
├── ml-harness/                  # Standalone execution harness engine
│   ├── harness/                 # Core cross-validation, runner, and guards
│   ├── dashboard/               # React + Vite frontend
│   ├── tools/                   # Harness CLI utilities
│   ├── tests/                   # 145 unit and integration tests
│   ├── docs/                    # Architecture and knowledge base
│   └── README.md                # Dedicated ml-harness documentation
├── train.py                     # Competition runner entrypoint
├── EXPERIMENTS.md               # Leaderboard, iterations, and idea graveyard
└── AGENTS.md                    # Workflow conventions for AI assistants
```

---

## 3. Quick Start

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r ml-harness/requirements.txt
```

### Running Experiments

```bash
# Fast smoke test (single fold debug probe)
python train.py --config configs/exp001_tfidf_baseline.yaml --debug --fold 0

# Full 5-fold cross-validation
python train.py --config configs/exp001_tfidf_baseline.yaml

# Re-run or overwrite completed experiment
python train.py --config configs/exp001_tfidf_baseline.yaml --force
```

### Inspecting Solution Tree & UCB Recommendations

```bash
# View all experiments and performance in the solution DAG
python tools/journal_tool.py --summary

# Query the next recommended atomic intervention (UCB-balanced)
python tools/journal_tool.py --recommend
```

### Starting the Web Dashboard

```bash
# Launch local web dashboard (serves on http://localhost:8000)
python tools/dashboard_server.py --port 8000
```

---

## 4. AI Assistant Workflow (Agent Bridge)

The project includes an **Agent Bridge** (`tools/agent_bridge_cli.py`) that allows AI coding assistants (such as Antigravity, OpenAI Codex, or Claude Code) to interact with the competition environment under strict guardrails:

```bash
# 1. Get compressed context packet (current stage, best CV, UCB recommendations)
python tools/agent_bridge_cli.py context

# 2. Start a work session with an experiment budget
python tools/agent_bridge_cli.py start-session --goal "Improve English Macro-F1" --max-experiments 3

# 3. Create a verified experiment configuration
python tools/agent_bridge_cli.py act --action create_experiment --params '{"branch": "feat", "descriptor": "title_context"}'

# 4. Run a fast smoke test
python tools/agent_bridge_cli.py act --action run_smoke_test --params '{"config_path": "configs/exp002_feat_title_context.yaml"}'

# 5. Run full validation
python tools/agent_bridge_cli.py act --action run_full_validation --params '{"config_path": "configs/exp002_feat_title_context.yaml", "backend": "local"}'
```

See [`AGENTS.md`](AGENTS.md) for full instructions and available typed actions.

---

## 5. Pluggable Harness Subfolder (`ml-harness/`)

The experimentation harness inside [`ml-harness/`](ml-harness/) is completely independent of the StereoQueerEval dataset and schema. It can be copied into any other competition or benchmark workspace with zero modifications.

For detailed documentation on the standalone harness, its REST API, and cloud execution backends, refer to [`ml-harness/README.md`](ml-harness/README.md).
