# ML-Harness: Autonomous Agentic ML/DL Experimentation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dashboard: React+Vite](https://img.shields.io/badge/Dashboard-React%20%2B%20Vite%20%2B%20Tailwind-purple.svg)](dashboard/)
[![Audit Gate: SWE--Agent](https://img.shields.io/badge/Audit%20Gate-SWE--Agent%20Verified-emerald.svg)](harness/audit_gate.py)
[![Cloud: Modal + Kaggle](https://img.shields.io/badge/Cloud-Modal%20%7C%20Kaggle%20CLI-orange.svg)](harness/cloud_runner.py)

**ML-Harness** is a rigorous, task-agnostic experimentation framework and execution harness designed for autonomous coding agents (Antigravity, OpenAI Codex, SWE-agent, AIDE) and human ML researchers. It prevents heuristic degeneration, confabulation, and premature convergence across complex machine learning and competitive data science competitions.

---

## Key Architectural Innovations

### 1. MCTS Solution Tree & UCB Lineage Tracking
- Formulates iterative ML/DL development as a Monte Carlo Tree Search (MCTS) DAG.
- Implements **Upper Confidence Bound (UCB)** branch recommendation to balance exploiting the current top model vs. exploring under-sampled model families.
- Automatically tracks parent-child lineage, atomic hypotheses, and post-mortems in `artifacts/journal.json`.

### 2. SWE-Agent Deterministic & Statistical Audit Gate
- Enforces a **Minimum Meaningful Effect Size (MMES $\ge 2.0\%$)** on validation samples to prevent submitting random seed noise or trivial variations.
- **Anti-Pseudo-Ensemble Verification**: Requires constituent models to exhibit **Cohen's Kappa ($\kappa \le 0.85$)** and a **Disagreement Rate ($\ge 5.0\%$)** to ensure genuine diversity before ensembling.
- Generates cryptographically verifiable `AUDIT_RECEIPT.json` receipts required for all competition submissions.

### 3. Epistemic Grounding & Assumption Register
- Inspired by Edwards & Schuster (2026) (*Ask or Assume?*), models must declare empirical grounding sources (literature citations, verified local registries) and explicit assumption registers before execution.
- High-uncertainty configurations trigger automated single-fold verification probes before committing multi-fold or remote GPU resources.

### 4. Cloud Execution Hub (Modal.com & Kaggle CLI)
- **Dataset Management**:
  - Modal: Persistent volume creation (`modal volume create/put`) with automatic code mounts.
  - Kaggle: Automated dataset staging (`kaggle datasets init/create/version`).
- **One-Click Runner Staging**: Generates containerized execution scripts (`tmp/modal/app_<id>.py` or `tmp/kaggle/<id>/run_kaggle.py` + `kernel-metadata.json`) with GPU hardware selection (A10G, A100, H100, L4, T4 x 2, P100).
- **Local Pre-Flight Inspection**: Syntactic and schema validation via `CloudPreflightInspector` prior to remote dispatch.
- **1-Click Tree Sync**: Fetches remote execution artifacts into local `artifacts/<exp_id>/` and updates the DAG in real time.

### 5. Interactive Minimalist Dashboard
- Single-page application built with **React, Vite, Tailwind CSS, and Vis-Network**.
- **Solution Tree Canvas**: Pan-and-zoom DAG visualization of all experimental nodes with delta metrics and branch tags.
- **Dynamic Leaderboard**: Full historical ranking with approach filters (ML, DL, Ensemble).
- **Submissions Hub**: Official leaderboard vs. local CV divergence tracking ($\Delta \text{LB} - \text{CV} < -0.05$ alerts).
- **Agent Priming Center**: One-click generation of prompt injection payloads tailored for OpenAI Codex CLI, Antigravity, AIDE, and Minimal Probes.
- **Expandable Task Configuration**: Configurable scoring functions (Macro-F1, ROC-AUC, RMSE, etc.), optimization directions, baseline floors, target goals, and composite weights.

---

## Directory Structure

```
ml-harness/
├── .agents/                    # Autonomous agent skill definitions & prompt contracts
├── configs/                    # Declarative experiment & task specifications
│   ├── task.yaml               # Universal task definition & multi-metric score configuration
│   ├── exp000_template.yaml    # Universal experiment template with grounding registers
│   └── exp001_tfidf_baseline.yaml # Reference baseline configuration
├── dashboard/                  # React + TypeScript + Vite + Tailwind CSS dashboard
├── docs/                       # Grounding knowledge base & architectural documentation
│   └── KNOWLEDGE_BASE.md       # Empirical literature citations & verified architectures
├── harness/                    # Core experimentation platform engines
│   ├── tree_journal.py         # MCTS Solution Tree & UCB exploration algorithm
│   ├── audit_gate.py           # Statistical audit gate & submission verification
│   ├── grounding_gate.py       # Epistemic grounding & assumption register validation
│   ├── cloud_runner.py         # Cloud execution orchestrator (Modal & Kaggle)
│   ├── cloud_inspector.py      # Local pre-flight syntax & dependency inspector
│   └── anti_degeneration.py    # Architectural diversity & pseudo-ensemble prevention
├── submissions/                # Submission packages, audit receipts, and submission tracker
├── tests/                      # Full unit test suite (100% passing)
├── tools/                      # CLI utilities & high-performance dashboard server
│   ├── dashboard_server.py     # REST API server & web application host
│   ├── journal_tool.py         # CLI for tree operations & UCB recommendation
│   ├── submission_tool.py      # CLI for registering verified competition submissions
│   └── web/                    # Production-built web dashboard assets
├── train.py                    # Unified experiment runner with grounding & audit integration
├── AGENTS.md                   # Operating contract & scientific guidelines for AI agents
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

# (Optional) Install dashboard frontend dependencies if modifying React components
cd dashboard && pnpm install && cd ..
```

### 2. Launch the Interactive Dashboard

```bash
python tools/dashboard_server.py --port 8080
```
Open [http://localhost:8080](http://localhost:8080) to inspect the Solution Tree, review the Leaderboard, configure Cloud Runners, or copy priming prompts for autonomous agents.

### 3. Run Experiments

```bash
# Smoke test on a single fold (Fold 0) - use --force if re-running existing baseline
python train.py --config configs/exp001_tfidf_baseline.yaml --debug --fold 0 --force

# Run full cross-validation
python train.py --config configs/exp001_tfidf_baseline.yaml --force
```

### 4. Run Tree Operations & UCB Recommendation

```bash
# View Solution Tree summary
python tools/journal_tool.py --summary

# Query optimal branch recommendation via UCB
python tools/journal_tool.py --recommend
```

### 5. Audit & Register a Submission

```bash
# Verify initial baseline predictions (--allow-initial for root baseline)
python tools/audit_gate.py --candidate artifacts/exp001_baseline_tfidf/oof_predictions.csv --allow-initial --output-zip submissions/sub001.zip

# Verify candidate predictions against parent baseline (enforces min prediction delta >= 2.0%)
python tools/audit_gate.py --candidate artifacts/exp002_feat_video_title/oof_predictions.csv --baseline artifacts/exp001_baseline_tfidf/oof_predictions.csv --output-zip submissions/sub002.zip

# Register verified submission
python tools/submission_tool.py register --exp-id exp001_baseline_tfidf --candidate artifacts/exp001_baseline_tfidf/oof_predictions.csv --allow-initial
```

### 6. Cloud Execution Dispatch

```bash
# Run local pre-flight checks
python harness/cloud_inspector.py --config configs/exp001_tfidf_baseline.yaml

# Dispatch to Modal or Kaggle
python harness/cloud_runner.py --backend modal --config configs/exp001_tfidf_baseline.yaml
python harness/cloud_runner.py --backend kaggle --config configs/exp001_tfidf_baseline.yaml
```

### 7. Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Agent Integration

To prime an autonomous coding agent (Antigravity, Codex CLI, Claude Code, AIDE), copy the formatted system directive from the **Agent Prompts** tab on the dashboard or query the API directly:

```bash
curl http://localhost:8080/api/priming_prompt?template=codex
```

Supported priming templates:
- **`codex`**: OpenAI Codex CLI & autonomous coding agents with AGENTS.md conventions.
- **`antigravity`**: Multi-agent collaborative research with mandatory 4-step Acknowledgment Gate.
- **`aide`**: MCTS/UCB tree exploration and atomic hypothesis formulation.
- **`minimal`**: Compact prompt for rapid debugging and single-variable probing.

### Dynamic Competition Skill Injection

The AI Agent skill (`.agents/skills/ml-experiment-harness/SKILL.md`) dynamically adapts to any competition or benchmark (Kaggle, Codabench, DrivenData):

```bash
# Inject skill directly from active task config
python tools/inject_skill.py --config configs/task.yaml

# Or inject from built-in competition presets
python tools/inject_skill.py --preset kaggle_tabular_regression
python tools/inject_skill.py --preset kaggle_binary_classification
python tools/inject_skill.py --preset vision_multiclass

# Or sync via journal CLI
python tools/journal_tool.py --inject-skill
```

When updating task settings or presets in the Web Dashboard, the skill is automatically re-generated and injected in real time.

---

## License

MIT License. Designed for robust, reproducible, and verifiable machine learning research.
