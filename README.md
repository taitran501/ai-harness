# ML-Harness: Universal Agentic ML/DL Experimentation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dashboard: React+Vite](https://img.shields.io/badge/Dashboard-React%20%2B%20Vite%20%2B%20Tailwind-purple.svg)](dashboard/)
[![Audit Gate: SWE--Agent](https://img.shields.io/badge/Audit%20Gate-SWE--Agent%20Verified-emerald.svg)](tools/audit_gate.py)
[![Cloud: Modal + Kaggle](https://img.shields.io/badge/Cloud-Modal%20%7C%20Kaggle%20CLI-orange.svg)](harness/cloud_runner.py)
[![Models: ML + DL + LLMs](https://img.shields.io/badge/Modeling-Classical%20%7C%20Transformers%20%7C%20QLoRA%20%7C%20CoT-red.svg)](harness/runner.py)

**ML-Harness** is an autonomous, pluggable experimentation framework and execution harness designed for competitive data science (Kaggle, Codabench, DrivenData, SemEval) and machine learning research. It bridges AI coding agents (**Antigravity**, **OpenAI Codex**, **SWE-agent**, **AIDE**) and human researchers with deterministic guardrails that eliminate heuristic degeneration, confabulation, data leakage, and premature convergence.

> **Operating Boundary**:  
> **“Antigravity thinks. AgentBridge coordinates. Harness executes and verifies.”**  
> Core harness primitives (OOF evaluation, solution tree DAGs, deterministic audit gates, candidate pools) are 100% decoupled from competition domains. Any competition can be attached simply by providing a declarative `configs/task.yaml`.

---

## Key Capabilities

```text
Antigravity IDE (Layer 1)
   │  Scientific Reasoning / Hypothesis / Observe -> Decide -> Act
   ▼
AgentBridge (Layer 2)
   │  Context Packet / Budget Guards / Typed Actions / Human Approval
   ▼
Universal Execution Harness (Layer 3)
   ├─ Universal Runner (ML / DL / Unsloth QLoRA / Reasoning LLMs)
   ├─ Dynamic TaskSpec (Holdout Dev / GroupKFold / Probe Validation)
   ├─ MCTS Solution Tree & UCB Lineage (AIDE-style DAG)
   ├─ Deterministic Guardrails (GroundingGate, EnsembleGuard, AuditGate)
   └─ Execution Backends (Local / Modal Serverless GPU / Kaggle API)
```

### 1. Full Modeling Spectrum Supported
- **Classical ML & Gradient Boosting**: Native support for `scikit-learn`, `LightGBM`, `XGBoost`, and `CatBoost` with automated hyperparameter pipelines.
- **Deep Learning Transformers**: Pretrained encoder fine-tuning (`DeBERTa-v3`, `XLM-RoBERTa`, `ModernBERT`) with custom heads, cosine schedules, and mixed precision.
- **Fast QLoRA (Local / Open-Weights)**: Integrated 4-bit NormalFloat (NF4) parameter-efficient fine-tuning via `Unsloth` and Hugging Face PEFT with Triton kernels (<9GB VRAM footprint).
- **Reasoning LLMs & Prompting**: Zero-shot and few-shot Chain-of-Thought (CoT) prompting with structured JSON classification schema parsing (Gemini, OpenAI, DeepSeek-R1).

### 2. Flexible Validation Protocols
- **Holdout Dev Split**: Fast evaluation against a single dedicated validation file (`dev_split_name`).
- **Grouped / Stratified Cross-Validation**: $k$-fold cross-validation with anti-leakage grouping (`GroupKFold`, `StratifiedKFold`, `TimeSeriesSplit`).
- **Low-Cost Sample Probes**: Instant single-fold or sampled-row verification probes to prove gradient convergence and schema validity before spending GPU compute.

### 3. MCTS Solution Tree & UCB Exploration (AIDE-Style)
- Formulates experiment lineage as a directed acyclic graph (DAG) in `artifacts/journal.json`.
- Implements **Upper Confidence Bound (UCB)** exploration:
  $$\text{Score}_{\text{UCB}}(N) = \mu(N) + c \cdot \sqrt{\frac{\ln(\text{Total Visits})}{\text{Visits}(N) + 1}}$$
- Balances exploiting the current top model vs. exploring under-sampled model families, preventing agents from myopically giving up on promising branches.

### 4. Deterministic SWE-Agent Guardrails
- **GroundingGate**: Requires literature/empirical citations and explicit assumption registers (Edwards & Schuster, 2026) before any configuration is persisted.
- **EnsembleGuard**: Rejects pseudo-ensembles. Enforces **Cohen's Kappa ($\kappa \le 0.85$)** and **Disagreement Rate ($\ge 5.0\%$)** to ensure candidate diversity.
- **AuditGate**: Mandates a **Minimum Meaningful Effect Size (MMES $\ge 2.0\%$)**, SHA-256 duplicate detection, and McNemar significance testing before signing cryptographic `AUDIT_RECEIPT.json` packages.

### 5. Multi-Cloud Execution Hub (Modal & Kaggle)
- **Modal.com**: Serverless execution with persistent volume caching (`modal volume`) and GPU selection (A10G, A100, H100, T4, L4).
- **Kaggle CLI**: Automated kernel generation (`run_kaggle.py` + `kernel-metadata.json`), dataset push, execution dispatch, and artifact synchronization.
- **Local Pre-Flight Checks**: Static syntactic and schema verification via `CloudPreflightInspector` prior to remote dispatch.

### 6. Interactive Minimalist WebApp & Dashboard
- Modern responsive dashboard built with **React, Vite, Tailwind CSS, and Vis-Network**.
- **Competition Journey Workspace**: Visual tracking across all 8 phases of competition lifecycle.
- **Execution Runner**: One-click configuration generator, command launcher, and console log viewer.
- **LLM Studio**: Interactive model playground, prompt engineering with variable interpolation, Hugging Face Hub live search, and 1-click Unsloth training script generation.
- **Solution Tree Canvas**: Interactive DAG visualization with metric deltas, model family tags, and node inspectors.
- **Dynamic Task Configurator**: Built-in presets for Kaggle tabular regression, binary classification, Codabench multi-task NLP, and custom benchmarks.

---

## Directory Layout

```text
ml-harness/
├── .agents/                    # Universal AI agent skill definitions & prompt contracts
├── configs/                    # Declarative task & experiment specifications
│   ├── base.yaml               # Universal runtime defaults
│   ├── task.yaml               # Active competition declarative contract
│   ├── tasks/                  # Predefined competition presets (Kaggle, Codabench, Vision)
│   ├── exp000_template.yaml    # Universal experiment template with grounding registers
│   └── exp001_tfidf_baseline.yaml # Reference baseline configuration
├── dashboard/                  # React + TypeScript + Vite + Tailwind CSS web dashboard
├── docs/                       # Architectural documentation & empirical knowledge base
│   ├── AIDE_SWE_ARCHITECTURE.md# Research paper analysis (AIDE MCTS & SWE-agent ACI)
│   └── KNOWLEDGE_BASE.md       # Empirical catalog of backbones, loss functions & prior benchmarks
├── harness/                    # Core experimentation platform engines
│   ├── runner.py               # Universal ML/DL/LLM execution runner
│   ├── task_spec.py            # Universal declarative task contract & dynamic evaluator
│   ├── agent_bridge.py         # 3-Layer Agent Bridge, stateful sessions & typed actions
│   ├── competition_state.py    # 8-Phase Journey, Candidate Pool & Freeze governance
│   ├── tree_journal.py         # MCTS Solution Tree & UCB exploration algorithm
│   ├── diagnostics.py          # Universal failure diagnostics & slice analysis
│   ├── ensemble_guard.py       # Candidate diversity & pseudo-ensemble prevention
│   ├── grounding_gate.py       # Epistemic grounding & assumption register validation
│   └── cloud_runner.py         # Cloud execution orchestrator (Modal & Kaggle)
├── submissions/                # Official submissions registry, metadata & audit receipts
├── tests/                      # Comprehensive unit and integration test suite (145 tests)
├── tools/                      # CLI utilities & high-performance dashboard server
│   ├── agent_bridge_cli.py     # Typed action execution & session telemetry CLI
│   ├── audit_gate.py           # Pre-submission statistical verification & SHA-256 audit gate
│   ├── dashboard_server.py     # REST API server & web application host
│   ├── journal_tool.py         # Solution Tree operations & UCB recommendation CLI
│   ├── submission_tool.py      # Registration tool for verified competition submissions
│   └── web/                    # Production-built web dashboard assets
├── train.py                    # Universal CLI entrypoint delegating to harness.runner
├── AGENTS.md                   # Operating contract & scientific guidelines for AI agents
├── EXPERIMENTS.md              # Tier 1 Global Memory: Hall of Fame & Idea Graveyard
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/taitran501/ml-harness.git
cd ml-harness

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install dashboard frontend dependencies if customizing React components
cd dashboard && pnpm install && cd ..
```

### 2. Configure Your Competition Task

Point `configs/task.yaml` to your competition dataset, or select one of the built-in presets:

```bash
# Example: Use a built-in Kaggle tabular regression preset
cp configs/tasks/kaggle_tabular_regression.yaml configs/task.yaml

# Or inject dynamically via CLI
python tools/inject_skill.py --preset kaggle_binary_classification
```

### 3. Launch the WebApp Dashboard

```bash
python tools/dashboard_server.py --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** to access the Solution Tree, inspect error slices, launch runs via the Execution Runner, or explore the LLM Studio.

### 4. Run Baseline & Verification Probes

```bash
# Fast smoke test on a single fold or debug slice (Fold 0)
python train.py --config configs/exp001_tfidf_baseline.yaml --debug --fold 0 --force

# Full validation execution
python train.py --config configs/exp001_tfidf_baseline.yaml --force
```

### 5. Solution Tree Lineage & UCB Recommendations

```bash
# View Solution Tree summary and top models
python tools/journal_tool.py --summary

# Query optimal branch recommendation via UCB
python tools/journal_tool.py --recommend
```

### 6. Audit & Package a Submission

```bash
# Verify candidate predictions against parent baseline (enforces MMES >= 2.0%)
python tools/audit_gate.py \
  --candidate artifacts/exp002_feat_pipeline/oof_predictions.csv \
  --baseline artifacts/exp001_tfidf_baseline/oof_predictions.csv \
  --output-zip submissions/sub002.zip

# Register verified submission in the local registry
python tools/submission_tool.py register \
  --exp-id exp002_feat_pipeline \
  --candidate artifacts/exp002_feat_pipeline/oof_predictions.csv
```

### 7. Run Full Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## AI Agent Collaboration & Priming

To prime an autonomous coding agent (Antigravity inside IDE, Codex CLI, Claude Code, AIDE), copy the formatted prompt from the **Priming** tab on the web dashboard or fetch it programmatically:

```bash
# Get priming prompt for Antigravity IDE
curl http://localhost:8080/api/priming_prompt?ide=antigravity&intent=iterate

# Get priming prompt for OpenAI Codex CLI
curl http://localhost:8080/api/priming_prompt?ide=codex&intent=attach
```

Supported priming templates:
- **`antigravity`**: Multi-agent collaborative reasoning with mandatory Epistemic Grounding and Assumption Register.
- **`codex`**: Autonomous terminal coding agents adhering strictly to `AGENTS.md` and typed action contracts.
- **`aide`**: MCTS/UCB tree exploration and atomic hypothesis formulation.
- **`minimal`**: Compact prompt for rapid debugging and single-variable probing.

---

## License

MIT License. Designed for robust, reproducible, and verifiable competitive data science and machine learning research.
