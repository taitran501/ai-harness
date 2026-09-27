# Architecture & System Design

This document details the architectural design of **ML-Harness**, drawing on principles from tree-search exploration (AIDE) and deterministic execution verification (SWE-agent).

---

## 1. System Layers

The harness organizes competitive machine learning workflows into three decoupled layers:

```text
Layer 1: Reasoning & Planning (Assistant / Engineer)
   │ Hypotheses, failure slice diagnostics, literature retrieval
   ▼
Layer 2: Agent Bridge (tools/agent_bridge_cli.py)
   │ Context compression, session budgets, typed action governance
   ▼
Layer 3: Execution Runtime (harness/)
   │ Validation schemes, Solution Tree DAG, guardrails & audit gates
```

---

## 2. Core Components

### 2.1 Solution Tree DAG (`harness/tree_journal.py`)
Rather than maintaining a flat, unstructured list of experiments, ML-Harness organizes iterations as a Directed Acyclic Graph (DAG):
- Each experiment node records its parent, hypothesis, hyperparameters, primary validation score, and execution artifacts.
- **Compressed State**: To prevent context window bloat for AI assistants, the journal generates concise summaries (~1,000 tokens) capturing only the top models, recent deltas, and bottleneck diagnostics.

### 2.2 UCB Search & Exploration (`tools/journal_tool.py`)
To prevent prematurely abandoning promising directions or getting stuck in local optima, the harness uses an Upper Confidence Bound (UCB) heuristic:

$$\text{Score}_{\text{UCB}}(N) = \text{Metric}(N) + c \cdot \sqrt{\frac{\ln(\text{Total Runs})}{\text{Visits}(N) + 1}}$$

This balances **exploitation** (iterating on the current highest-performing model family) with **exploration** (testing under-explored feature pipelines or model backends).

### 2.3 Universal Execution Engine (`harness/runner.py`)
The execution engine is task-agnostic and supports multiple modeling paradigms:
- **Classical ML**: Scikit-learn pipelines, LightGBM, XGBoost, CatBoost.
- **Deep Transformers**: Pretrained Hugging Face encoders with customizable task heads.
- **PEFT / QLoRA**: 4-bit quantized LoRA fine-tuning for consumer GPUs.
- **LLM Prompting & Reasoning**: Multi-step prompting with structured output validation.

### 2.4 Guardrails & Verification Gates
To maintain experimental rigor:
- **Epistemic Grounding Gate (`harness/grounding_gate.py`)**: Checks experiment configurations for explicit hypotheses, literature references, and assumptions before running.
- **Ensemble Diversity Guard (`harness/ensemble_guard.py`)**: Rejects pseudo-ensembles by enforcing prediction divergence thresholds ($\kappa \le 0.85$, disagreement $\ge 5\%$).
- **Deterministic Audit Gate (`tools/audit_gate.py`)**: Validates submission files against baselines before packaging.

---

## 3. Execution Backends

ML-Harness supports three execution backends:
1. **Local**: Direct execution on the developer machine via `python train.py`.
2. **Modal.com**: Serverless GPU dispatch with persistent volume caching.
3. **Kaggle**: Automated notebook packaging, push, and artifact synchronization.
