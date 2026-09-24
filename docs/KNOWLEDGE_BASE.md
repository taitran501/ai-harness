# KNOWLEDGE_BASE.md — Domain Knowledge & Model Registry

This document serves as the **authoritative in-tree knowledge registry** for SemEval-2027 Task 8 (StereoQueerEval).
AI Agents must consult this document or conduct fresh external retrieval (via `search_web` / Hugging Face Hub)
before proposing any model, loss function, or feature intervention.

---

## 1. Verified Model Backbones (Multilingual: EN, IT, NL)

The dataset contains comments in **English (EN)**, **Italian (IT)**, and **Dutch (NL)**, with Persian (FA) as zero-shot.
Agents MUST NOT propose English-only backbones (e.g. `bert-base-uncased`, `roberta-base`) for multilingual evaluation.

| Backbone Model ID | Parameters | Multilingual Coverage | Architecture Highlights | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **`microsoft/mdeberta-v3-base`** | ~86M backbone (+160M embedding) | 100+ languages (incl. EN, IT, NL) | Disentangled Attention, Replaced Token Detection (RTD). | **Recommended baseline** for hate speech & stereotype classification. Strong semantic representations. |
| **`FacebookAI/xlm-roberta-base`** | ~125M backbone (+150M embedding) | 100 languages (incl. EN, IT, NL) | Robust multilingual masked LM, byte-level BPE. | Very stable on noisy social media comments; resistant to punctuation variations. |
| **`FacebookAI/xlm-roberta-large`** | ~355M backbone | 100 languages | Scaled XLM-R capacity. | Offload to Cloud GPU (`harness/cloud_runner.py`). High capacity for rare identity distinctions. |
| **`answerdotai/ModernBERT-base`** | ~149M total | Multilingual transfer capable (RoPE, 8192 context) | Modernized encoder: FlashAttention-2, rotary embeddings, unpadding. | Fast inference, long YouTube comments + descriptions. |
| **`Alibaba-NLP/bge-m3`** | ~567M total | High-coverage multilingual embedding | Multi-Functionality, Multi-Granularity, Multi-Linguality. | Pre-extracting dense semantic embeddings for lightweight classifiers. |

---

## 2. Hard Imbalance & Subtask C Identity Strategies

### The Bottleneck
Subtask C evaluates 9 LGBTQIA+ identity labels (`l`, `g`, `b`, `t`, `q`, `i`, `a`, `lgbtqia+`, `nb`) under the **`oracle_hate`** condition (`hate_speech != "no"`).
- Labels `g` (gay), `l` (lesbian), and `t` (trans) have substantial support.
- Labels `a` (asexual), `i` (intersex), and `nb` (non-binary) have **very low support (<15 samples)** across the entire training set.
- Standard Binary Cross Entropy (BCE) collapses on the extreme tail, yielding **0.00 Macro-F1** on rare classes.

### Scientifically Grounded Loss Interventions
1. **Focal Loss ($\gamma \in [1.5, 2.5]$, $\alpha \in [0.25, 0.75]$)**:
   - Downweights easy negative samples ($p_t \to 1$), forcing gradients to focus on ambiguous and rare positive samples:
     $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
2. **Class-Balanced Loss (Cui et al., CVPR 2019)**:
   - Weights loss by the *effective number of samples* $E_n = \frac{1 - \beta^n}{1 - \beta}$ where $\beta \in [0.99, 0.9999]$.
   - Prevents dominating gradients from majority identities.
3. **Asymmetric Loss (ASL) for Multi-Label (Ridnik et al., ICCV 2021)**:
   - Sets asymmetric focusing parameters $\gamma_+$ and $\gamma_-$, with hard thresholding $\max(p - m, 0)$ for negative samples to eliminate noisy false negatives.
4. **Post-Hoc Threshold Optimization**:
   - Rather than fixing a uniform threshold $0.5$ for all identities, optimize per-class decision thresholds on validation OOF probabilities to maximize per-class F1.

---

## 3. Multi-Task Learning (MTL) & Hierarchical Routing

### The Inter-Task Relationship
- **Subtask A** (Stereotype: `no`/`yes`) and **Subtask B** (Hate Speech: `no`/`yes_explicit`/`yes_implicit`) exhibit high correlation: hateful comments frequently incorporate stereotypes.
- **Subtask C** (Target Type & Identities) is strictly conditioned on `hate_speech != "no"`.

### Recommended Architectures
```
                   [Multilingual Backbone (mDeBERTa / XLM-R)]
                                      │
                                      ▼
                           [Shared Semantic Pooling]
                           ├── Head A: Stereotype (Binary Logits)
                           ├── Head B: Hate Speech (3-Class Logits)
                           └── Head C: (Hierarchical Masked Gating)
                                ├── Target Type Head (Individual vs Group)
                                └── Identity Head (9 Sigmoids)
                                (Loss masked out if true hate_speech == "no")
```

---

## 4. Input Text Representations & Context Noise

| Representation | Format | Empirical Findings & Known Graveyard Status |
| :--- | :--- | :--- |
| **`comment`** | YouTube comment text only | High precision. Captures local hate/stereotype cues directly. |
| **`title_comment`** | `[TITLE] {yt_title} [SEP] {comment}` | **Recommended balance**. Provides video context without overwhelming comment signal. |
| **`full_context`** | `[TITLE] {yt_title} [DESC] {yt_desc} [COMMENT] {comment}` | **Idea Graveyard Warning**: YouTube descriptions are full of links, promotional boilerplate, and copyright notices. They introduce severe noise for Subtask A & B. Do NOT use raw descriptions without semantic filtering. |

---

## 5. Epistemic Protocol: Mandatory Retrieval Before Implementation

Before creating a new experiment:
1. Identify the specific bottleneck in `artifacts/<parent_id>/DIAGNOSTICS.md`.
2. Check if the proposed solution is already documented in this `KNOWLEDGE_BASE.md` or refuted in the `Idea Graveyard` (`EXPERIMENTS.md`).
3. If proposing a novel library, loss, or model outside this document:
   - Call `search_web` or inspect Hugging Face / ArXiv.
   - Record the evidence in `grounding:` and `assumption_register:` in the config YAML.
