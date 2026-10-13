# Architecture and System Design

This document details the architectural design of **AI-Harness**, which decouples creative machine learning exploration from execution and deterministic validation.

---

## 1. System Layers

AI-Harness organizes machine learning workflows into three decoupled layers:

```text
Layer 1: Reasoning and Planning (Assistant / Engineer)
   |  Formulates hypotheses, inspects failure slices, designs pipelines.
   v
Layer 2: Agent Bridge (tools/agent_bridge_cli.py & harness/agent_bridge.py)
   |  Provides context compression, session budgets, and typed action governance.
   v
Layer 3: Execution Runtime (harness/)
      Executes cross-validation, maintains the Solution Tree DAG, and enforces audit gates.
```

### Layer 1: Reasoning and Planning
The engineer or AI coding assistant (Antigravity, Codex, Claude Code) functions as the research scientist. Layer 1 analyzes problem specifications, reviews diagnostic reports from past runs, retrieves relevant domain techniques, and formulates specific atomic hypotheses.

### Layer 2: Agent Bridge
The bridge layer mediates between the reasoning assistant and the execution engine:
- **Context Compression**: Distills multi-megabyte run histories and logs into a compact Heads-Up Display (~1,000 tokens) to preserve the assistant's context window.
- **Session and Budget Management**: Tracks experiment quotas and execution runtime limits to prevent compute runaway.
- **Typed Action Governance**: Translates high-level agent intentions (`create_experiment`, `run_smoke_test`, `promote_candidate`) into validated system calls.

### Layer 3: Execution Runtime
The execution engine runs pipelines against declared datasets and validation schemes:
- **Solution Tree DAG**: Records parent-child lineages and metrics in an immutable graph structure.
- **Validation Engine**: Executes multi-fold cross-validation with strict anti-leakage grouping invariants.
- **Audit and Guardrails**: Verifies prediction integrity, checks manifest hashes, and blocks invalid submissions.

---

## 2. Core Components

### 2.1 Solution Tree DAG (`harness/tree_journal.py`)
Rather than maintaining an unstructured flat list of runs, AI-Harness organizes iterations as a Directed Acyclic Graph (DAG):
- **Lineage Tracking**: Each node records its parent experiment ID, hypothesis statement, hyperparameters, cross-validation metrics, and output artifact paths.
- **State Compression**: Summaries highlight the best model, recent score deltas, and current error bottlenecks without dumping raw prediction files.

### 2.2 Upper Confidence Bound (UCB) Search (`tools/journal_tool.py`)
To prevent premature convergence on a single model family or early abandonment of promising directions, the harness applies an Upper Confidence Bound heuristic:

$$\text{Score}_{\text{UCB}}(N) = \text{Metric}(N) + c \cdot \sqrt{\frac{\ln(\text{Total Runs})}{\text{Visits}(N) + 1}}$$

Where:
- $\text{Metric}(N)$ represents the normalized utility of node $N$.
- $\text{Total Runs}$ is the cumulative experiment count.
- $\text{Visits}(N)$ is the number of times branch $N$ has been iterated upon.
- $c$ is an exploration parameter balancing exploitation of top models with exploration of under-tested intervention branches (`feat`, `model`, `loss`, `post`).

### 2.3 Universal Execution Engine (`harness/runner.py`)
The execution engine supports diverse modeling paradigms:
- **Deep Transformers (Verified Core - NLP)**: Pretrained Hugging Face encoders (such as DeBERTa, RoBERTa, ModernBERT) with customizable task heads.
- **PEFT / QLoRA (Verified Core - NLP)**: Parameter-efficient fine-tuning and quantized training for consumer GPUs.
- **LLM Prompting (Verified Core - NLP)**: Structured prompt pipelines and few-shot reasoning workflows.
- **Classical Gradient Boosting (Tabular; locally tested, competition verification pending)**: Scikit-learn pipelines, LightGBM, XGBoost, and optional CatBoost.

**The Non-Prescriptive Principle**: In this framework, the assistant or engineer designs the model factory (`model.factory`), feature representations, and hyperparameter grids. The harness runtime acts strictly as an execution runner and verification auditor—it never imposes rigid preprocessing or forces specific scalers.

### 2.4 Guardrails and Verification Gates
- **Grounding Gate (`harness/grounding_gate.py`)**: Validates that experiment configurations declare explicit hypotheses, rationale, and assumption registers prior to execution.
- **Ensemble Diversity Guard (`harness/ensemble_guard.py`)**: Checks prediction correlation thresholds between ensemble constituents to ensure meaningful diversity.
- **Submission Gate (`tools/audit_gate.py`)**: Verifies prediction format, row count, null absence, domain validity, and column ordering before allowing submission packaging.

---

## 3. Execution Backends

AI-Harness supports three execution targets:
1. **Local**: Direct execution on the local host machine via `ai-harness-run --project-root .`.
2. **Modal.com**: Serverless GPU dispatch with persistent volume caching for heavy transformer training.
3. **Kaggle**: Automated kernel generation, push, and remote artifact synchronization.

---

## 4. Verification Scope and Development Roadmap

### Verified Domain Scope: Natural Language Processing (NLP / Text)
The harness runtime, baseline suites, and default diagnostics are benchmarked and verified for NLP/Text tasks:
- Sequence classification (single-label and multi-label).
- Cross-lingual transfer (XLM-RoBERTa, mDeBERTa).
- Modern transformer backends (ModernBERT, DeBERTa-v3).
- LLM prompting and QLoRA fine-tuning.

### Experimental Modalities (Roadmap WIP)
Support for other modalities is currently in progress:
- **Tabular Modeling**: Implemented categorical handling and locally tested LightGBM/XGBoost classification/regression paths; optional CatBoost support. Competitive performance across tasks remains unverified.
- **Computer Vision**: Backbone integration via `timm` (ConvNeXt, Swin, ViT) and spatial augmentations.
- **Audio Processing**: Acoustic features and speech models (Whisper, Wav2Vec2, AST).
- **Multimodal Fusion**: Cross-attention architectures combining text, tabular, and visual features.
