# ML-Harness

A modular experimentation harness and dashboard for competitive machine learning and data science projects (Kaggle, Codabench, DrivenData, and academic benchmarks).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dashboard](https://img.shields.io/badge/Dashboard-React%20%2B%20Vite-purple.svg)](dashboard/)
[![Tests](https://img.shields.io/badge/Tests-145%20passing-brightgreen.svg)](tests/)

---

## Overview

**ML-Harness** is a Python framework designed to streamline running, tracking, and evaluating machine learning experiments. It works for standard data science workflows whether you run models locally, dispatch heavy training jobs to cloud GPUs (Modal or Kaggle), or collaborate with AI coding assistants (such as Antigravity or Codex).

The harness is **task-agnostic**. The core runtime (cross-validation, experiment lineage tracking, ensembling checks, submission verification, and web UI) is decoupled from any specific competition. You can adapt it to a new tabular, text, vision, or multi-task problem by editing a single configuration file (`configs/task.yaml`).

---

## Features

### 1. Broad Modeling Support
- **Classical ML & Gradient Boosting**: Native support for `scikit-learn`, `LightGBM`, `XGBoost`, and `CatBoost` with tabular preprocessing pipelines.
- **Deep Learning Transformers**: Fine-tuning pretrained text encoders (`DeBERTa`, `RoBERTa`, `ModernBERT`, etc.) via Hugging Face with customizable heads, learning rate schedules, and mixed precision.
- **PEFT / QLoRA**: Script generation and execution for 4-bit quantized LoRA fine-tuning using `Unsloth` and Hugging Face PEFT for consumer-grade GPU memory footprints.
- **LLM Prompting & Reasoning**: Support for zero-shot, few-shot, and Chain-of-Thought (CoT) prompting with structured JSON output parsing across OpenAI, Gemini, and DeepSeek backends.

### 2. Flexible Validation Protocols
- **Holdout Validation**: Fast validation against a designated evaluation dataset (`dev_split_name`).
- **Cross-Validation**: $k$-fold cross-validation with anti-leakage grouping (`GroupKFold`, `StratifiedKFold`, `TimeSeriesSplit`).
- **Debug Probes**: Low-cost, sampled-row smoke tests (`--debug --fold 0`) to verify syntax, pipeline wiring, and convergence before launching long runs.

### 3. Experiment Tracking & Solution Tree
- **Lineage DAG**: Experiments are tracked in `artifacts/journal.json` with parent-child relationships, hypotheses, and evaluation metrics.
- **Upper Confidence Bound (UCB) Search**: Recommends next steps by balancing exploitation of the current best model with exploration of under-tested model families.
- **Human-Readable Logs**: Summaries and notes are maintained in `EXPERIMENTS.md` (tracking best models and an idea graveyard of approaches that did not pan out).

### 4. Verification & Guardrails
- **Epistemic Grounding**: Optional metadata verification ensuring configurations specify their rationale, reference sources, and assumptions before running.
- **Ensemble Diversity Guard**: Rejects pseudo-ensembles by enforcing prediction divergence thresholds (Cohen's Kappa $\kappa \le 0.85$ and minimum disagreement $\ge 5.0\%$) between models before blending.
- **Pre-Submission Audit**: Validates submission files against baselines (checking for duplicate predictions, minimum effect size $\ge 2.0\%$, and format errors) before packaging `submission.zip` or `submission.csv`.

### 5. Multi-Backend Execution
- **Local**: Run directly on your development machine (`python train.py`).
- **Modal.com**: Serverless GPU execution with persistent volume caching and GPU selection (A10G, A100, H100, T4, L4).
- **Kaggle API / CLI**: Automated script packaging, kernel push, remote execution monitoring, and artifact download.

### 6. Interactive Web Dashboard
A local web UI built with React, Vite, and Tailwind CSS provides:
- **Solution Tree View**: Interactive graph of all experiment iterations and their performance deltas.
- **Leaderboard**: Compare validation scores, parameters, and training logs across models.
- **Execution Runner**: Configure and trigger runs directly from the browser.
- **LLM Studio**: Test prompts against sample data, search Hugging Face Hub models, and generate Unsloth training scripts.
- **Task Configurator**: Switch between task presets (e.g., Kaggle house prices regression, binary classification, Codabench multi-task NLP) or edit task definitions interactively.
- **Submissions Hub**: Track generated submission files and audit receipts.

---

## Directory Structure

```text
ml-harness/
├── configs/                    # Declarative configuration files
│   ├── base.yaml               # Default runtime and path settings
│   ├── task.yaml               # Active competition/benchmark definition
│   ├── tasks/                  # Task presets (Kaggle tabular, NLP, etc.)
│   ├── exp000_template.yaml    # Starter experiment template
│   └── exp001_tfidf_baseline.yaml # Baseline configuration
├── dashboard/                  # React + Vite frontend source code
├── docs/                       # Reference documentation
│   ├── AIDE_SWE_ARCHITECTURE.md# Background on MCTS tree search and SWE-agent patterns
│   └── KNOWLEDGE_BASE.md       # Modeling reference guide (NLP, Tabular, Loss functions)
├── harness/                    # Core Python library
│   ├── runner.py               # Universal experiment execution engine
│   ├── task_spec.py            # Task definition parsing and metric evaluation
│   ├── agent_bridge.py         # Session management and assistant bridge
│   ├── tree_journal.py         # Solution tree and UCB recommendation
│   ├── diagnostics.py          # Error slice analysis and confusion matrices
│   ├── ensemble_guard.py       # Candidate diversity validation
│   ├── grounding_gate.py       # Configuration and assumption checks
│   ├── submission_manager.py   # Submission registration and tracking
│   └── cloud_runner.py         # Modal and Kaggle execution dispatch
├── submissions/                # Packaged submissions and audit records
├── tests/                      # Unit and integration test suite
├── tools/                      # CLI utilities and dashboard server
│   ├── dashboard_server.py     # Local dashboard HTTP API server
│   ├── agent_bridge_cli.py     # CLI for assistant session management
│   ├── journal_tool.py         # CLI for inspecting solution tree & UCB suggestions
│   ├── audit_gate.py           # Pre-submission verification tool
│   ├── submission_tool.py      # Submission packaging CLI
│   └── web/                    # Built static dashboard assets served by Python
├── train.py                    # Main CLI entrypoint for running experiments
├── AGENTS.md                   # Assistant operating guidelines and command reference
├── EXPERIMENTS.md              # Experiment log and idea graveyard
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/taitran501/ml-harness.git
cd ml-harness

# Install Python dependencies
pip install -r requirements.txt

# (Optional) If modifying the React dashboard frontend:
cd dashboard && pnpm install && pnpm build && cd ..
```

### 2. Dataset Acquisition

If starting fresh without a dataset, download your competition data into a `dataset/` directory:

- **Kaggle CLI**:
  ```bash
  kaggle competitions download -c <competition-name> -p dataset/
  tar -xf dataset/<competition-name>.zip -C dataset/
  ```
- **Direct Download / URL**:
  ```bash
  curl -L "<download-url>" -o dataset/data.zip
  tar -xf dataset/data.zip -C dataset/
  ```
- **Hugging Face Datasets**:
  ```bash
  python -c "from datasets import load_dataset; ds = load_dataset('<dataset_name>'); ds['train'].to_pandas().to_csv('dataset/train.csv', index=False)"
  ```

---

### 3. Choose Your Workflow: Agent vs. Manual

#### Option A: Agent-Driven (Zero Configuration)
If using an AI coding assistant (Antigravity IDE, Claude Code, OpenAI Codex):
1. Open the repository in your editor.
2. Prompt the assistant:
   > *"Initialize the harness for my dataset in `dataset/train.csv`, configure `configs/task.yaml`, inject the skill, and run a smoke test."*
3. The assistant auto-detects columns, generates the configuration, updates `.agents/skills/`, and verifies convergence on a debug probe.

#### Option B: Manual Setup (3 Commands)
```bash
# 1. Select a task preset matching your problem
cp configs/tasks/kaggle_tabular_regression.yaml configs/task.yaml
# (Or kaggle_binary_classification.yaml, stereoqueereval.yaml, etc.)

# 2. Inject competition skill into .agents/skills/
python tools/inject_skill.py

# 3. Run a fast smoke test (single fold debug probe)
python train.py --config configs/exp001_tfidf_baseline.yaml --debug --fold 0
```

---

### 4. Task Configuration Reference (`configs/task.yaml`)

A minimal `configs/task.yaml` looks like this:

```yaml
task:
  name: "MyCompetition"
  platform: "kaggle"
  task_type: "classification"
  primary_metric: "macro_f1"
  direction: "maximize"

data:
  train_path: "dataset/train.csv"
  id_column: "id"
  target_column: "target"

validation:
  strategy: "kfold"
  n_folds: 5
  seed: 42
  probe_samples: 50
```

### 3. Launch the Web Dashboard

```bash
python tools/dashboard_server.py --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser. You can inspect your solution tree, edit task parameters, generate new configs, or test prompts in the LLM Studio.

### 4. Run an Experiment

Use `train.py` to run models. Use `--debug --fold 0` for a fast smoke test:

```bash
# Fast smoke test on fold 0 with sample probe
python train.py --config configs/exp001_tfidf_baseline.yaml --debug --fold 0 --force

# Full validation run across all folds
python train.py --config configs/exp001_tfidf_baseline.yaml --force
```

### 5. Inspect the Solution Tree

```bash
# View current experiment tree summary
python tools/journal_tool.py --summary

# Ask for the next recommended branch using UCB
python tools/journal_tool.py --recommend
```

### 6. Verify and Package a Submission

Before submitting to a competition platform, check your predictions with `audit_gate.py`:

```bash
# Compare candidate predictions against your baseline
python tools/audit_gate.py \
  --candidate artifacts/exp002_my_model/oof_predictions.csv \
  --baseline artifacts/exp001_tfidf_baseline/oof_predictions.csv \
  --output-zip submissions/sub002.zip
```

If the checks pass (non-duplicate predictions, formatting verified), the script packages your submission and saves an `AUDIT_RECEIPT.json`.

### 7. Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Working with AI Assistants

If you use an AI coding assistant (such as Antigravity, OpenAI Codex, or Claude Code), the assistant can interact with ML-Harness using the CLI tools:

- **Context Overview**: `python tools/agent_bridge_cli.py context` returns a structured summary of the current task, top models, recommended branches, and recent bottlenecks.
- **Session Tracking**: `python tools/agent_bridge_cli.py start-session --goal "Improve F1 score" --max-experiments 3` tracks progress and resource usage.
- **Typed Actions**: The assistant can invoke actions like `create_experiment`, `run_smoke_test`, `run_full_validation`, or `promote_candidate` through `agent_bridge_cli.py act`.

Detailed guidelines and action contracts are documented in [AGENTS.md](AGENTS.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
