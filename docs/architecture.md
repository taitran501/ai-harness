# Architecture & System Design

This document details the architectural design of **AI-Harness**, drawing on principles from tree-search exploration (AIDE) and deterministic execution verification (SWE-agent).

> [!IMPORTANT]
> **Verification Scope: Strictly NLP / Text Only**
> AI-Harness is currently **strictly verified and working well ONLY on Natural Language Processing (NLP / Text)** benchmarks (such as sequence classification, transformer fine-tuning, and prompt evaluation).
> Support for all other modalities (**Tabular, Computer Vision, Audio, Multimodal**) is **experimental work-in-progress (WIP / Roadmap)**, has NOT been verified in competitive conditions, and is NOT considered ready or working well in practice. Never claim or assume that non-NLP modalities are production-ready or battle-tested.

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
Rather than maintaining a flat, unstructured list of experiments, AI-Harness organizes iterations as a Directed Acyclic Graph (DAG):
- Each experiment node records its parent, hypothesis, hyperparameters, primary validation score, and execution artifacts.
- **Compressed State**: To prevent context window bloat for AI assistants, the journal generates concise summaries (~1,000 tokens) capturing only the top models, recent deltas, and bottleneck diagnostics.

### 2.2 UCB Search & Exploration (`tools/journal_tool.py`)
To prevent prematurely abandoning promising directions or getting stuck in local optima, the harness uses an Upper Confidence Bound (UCB) heuristic:

$$\text{Score}_{\text{UCB}}(N) = \text{Metric}(N) + c \cdot \sqrt{\frac{\ln(\text{Total Runs})}{\text{Visits}(N) + 1}}$$

This balances **exploitation** (iterating on the current highest-performing model family) with **exploration** (testing under-explored feature pipelines or model backends).

### 2.3 Universal Execution Engine (`harness/runner.py`)
The execution engine is task-agnostic and supports multiple modeling paradigms:
- **Deep Transformers (Verified Core - NLP)**: Pretrained Hugging Face encoders with customizable task heads.
- **PEFT / QLoRA (Verified Core - NLP)**: 4-bit quantized LoRA fine-tuning for consumer GPUs.
- **LLM Prompting & Reasoning (Verified Core - NLP)**: Multi-step prompting with structured output validation.
- **Classical ML (Experimental / WIP — Tabular)**: Scikit-learn pipelines, LightGBM, XGBoost, CatBoost (experimental scaffolding; unverified in competitive practice).

**Non-Prescriptive Principle**: In the AIDE / Agentic ML paradigm, the agent (Antigravity / Codex) is the creative scientist responsible for designing and modifying `model.factory`, feature pipelines, and hyperparameters. The harness runtime acts purely as an execution runtime and audit gate—it never imposes rigid preprocessing rules (such as forcing scalers or restrictive categorical limits) or dictates model choices.

### 2.4 Guardrails & Verification Gates
To maintain experimental rigor:
- **Epistemic Grounding Gate (`harness/grounding_gate.py`)**: Checks experiment configurations for explicit hypotheses, literature references, and assumptions before running.
- **Ensemble Diversity Guard (`harness/ensemble_guard.py`)**: Applies configured diversity thresholds when the task declares them.
- **Submission Gate**: Uses the selected task plugin's submission builder and validator. The generic audit applies only to declarative tasks.

---

## 3. Execution Backends

AI-Harness supports three execution backends:
1. **Local**: Direct execution on the developer machine via `ai-harness-run --project-root .`.
2. **Modal.com**: Serverless GPU dispatch with persistent volume caching.
3. **Kaggle**: Automated notebook packaging, push, and artifact synchronization.

---

## 4. Empirical Scope & Evolution Roadmap

### 4.1 Verified Domain Scope: Strictly NLP / Text Only
> [!IMPORTANT]
> **Production & Competition Readiness**: The harness runtime, baseline suites, and diagnostics are currently **strictly validated and working well ONLY on NLP / Text** benchmarks:
> - **NLP / Text Modeling (Verified Core)**: Sequence classification, cross-lingual transfer (XLM-R, DeBERTa, ModernBERT), LLM prompting, and QLoRA fine-tuning (empirically validated on NLP classification benchmarks).
> - **Other Modalities are Experimental WIP**: Tabular, Computer Vision, Audio, and Multimodal are **NOT** yet verified and **NOT** working well in practice; they are experimental works-in-progress (WIP) on the development roadmap, not ready for production or competition use.

### 4.2 Transparency on Historical LLM / NLP Bias
Early scaffolding within the project carried an implicit bias toward textual problem structures (e.g. TF-IDF feature extraction, transformer encoders, and language model inference). We explicitly recognize this bias:
- Built-in heuristics, defaults, and TreeJournal clustering now avoid pigeonholing models into narrow textual families.
- The execution layer treats all model definitions, pipelines, and inputs as flexible, pluggable, and agent-directed.

### 4.3 Experimental & Roadmap Modalities (WIP — Unverified)
To expand beyond the currently verified NLP scope, active and planned development targets:
1. **Tabular Modeling**: Classical gradient boosting (LightGBM, XGBoost, CatBoost) with native categorical handling and cross-validation schemes (experimental scaffolding; currently unverified in real competition conditions).
2. **Computer Vision**: Backbone architectures via `timm` (ConvNeXt, Swin, ViT), visual augmentation pipelines, and image classification/segmentation metrics.
3. **Audio Processing**: Acoustic feature extraction, spectrogram transforms, and models such as Whisper, Wav2Vec2, and Audio Spectrogram Transformers (AST).
4. **Multimodal Fusion**: Cross-attention and early/late fusion architectures combining tabular, textual, and visual modalities (CLIP, SigLIP, Qwen2-VL).
