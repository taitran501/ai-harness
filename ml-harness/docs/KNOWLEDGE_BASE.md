# Universal ML Competition Knowledge Base & Grounding Catalog

This document serves as the **authoritative in-tree empirical knowledge registry** for ML-Harness.
AI Agents and human researchers must consult this document or conduct fresh external retrieval (via `search_web` / Hugging Face Hub / Semantic Scholar) before proposing any model backbone, loss function, or feature intervention.

---

## 1. Natural Language Processing & Modern Transformers

For text classification, sequence labeling, and multi-task textual reasoning:

| Backbone Model ID | Parameters | Modality / Coverage | Key Architecture Highlights | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **`microsoft/deberta-v3-base`** | ~86M (+160M emb) | English | Disentangled Attention, Replaced Token Detection (RTD) | SOTA encoder for English classification, reasoning, and NLI. |
| **`microsoft/mdeberta-v3-base`** | ~86M (+160M emb) | 100+ languages | Disentangled attention with multilingual vocab | Recommended baseline for cross-lingual transfer & multilingual classification. |
| **`FacebookAI/xlm-roberta-base`** | ~125M (+150M emb) | 100 languages | Robust multilingual masked LM, byte-level BPE | Extremely stable on noisy social media text, slang, and multilingual subsets. |
| **`answerdotai/ModernBERT-base`** | ~149M total | English / Transfer | FlashAttention-2, RoPE rotary embeddings, 8192 context | Ultra-fast inference, long documents, and rich contextual fusion. |
| **`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** | ~7B (4-bit NF4) | Multilingual | Native 4-bit QLoRA, Triton kernels, <9GB VRAM footprint | Generative classification, complex world knowledge, implicit nuance. |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`** | ~14B (4-bit AWQ) | Multilingual | Distilled multi-step reasoning traces | Zero-shot & few-shot Chain-of-Thought reasoning for subtle coded subtext. |

---

## 2. Tabular & Structured Learning

For structured tabular benchmarks (Kaggle, DrivenData):

| Framework / Model | Engine Highlights | Optimal Hyperparameter Priors | Best Suited For |
| :--- | :--- | :--- | :--- |
| **`LightGBM`** | Histogram-based gradient boosting, leaf-wise tree growth | `learning_rate: 0.03-0.08`, `num_leaves: 31-127`, `subsample: 0.8`, `colsample_bytree: 0.8` | High-speed baseline, large datasets, dense continuous features. |
| **`XGBoost`** | Depth-wise tree growth, exact greedy & histogram methods | `max_depth: 4-8`, `eta: 0.05`, `subsample: 0.85`, `colsample_bytree: 0.8` | Competitive tabular accuracy, robust L1/L2 leaf regularization. |
| **`CatBoost`** | Symmetric (oblivious) trees, ordered target encoding | `depth: 6-8`, `l2_leaf_reg: 3-10`, `iterations: 1000-2500` | Heavy categorical variables, default parameter resilience. |
| **`TabNet`** | Sequential attentive mechanism for sparse feature selection | `n_d: 8-64`, `n_a: 8-64`, `gamma: 1.3-2.0` | Interpretable deep tabular representations and multi-modal feature fusion. |

---

## 3. Severe Class Imbalance & Skew Mitigation

When minority classes or rare target labels have extreme imbalance:

### 1. Focal Loss (Lin et al., ICCV 2017)
Downweights easy negative samples ($p_t \to 1$), dynamically focusing gradients on ambiguous and rare positive samples:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- Typical hyperparameter priors: $\gamma \in [1.5, 2.5]$, $\alpha \in [0.25, 0.75]$.

### 2. Class-Balanced Loss via Effective Number of Samples (Cui et al., CVPR 2019)
Weights loss inversely by the *effective volume of data* $E_n = \frac{1 - \beta^n}{1 - \beta}$, avoiding gradient dominance by majority categories:
$$\beta \in [0.99, 0.9999]$$

### 3. Asymmetric Loss for Multi-Label (ASL, Ridnik et al., ICCV 2021)
Applies asymmetric focusing parameters $\gamma_+$ and $\gamma_-$ with hard probability margin thresholding $\max(p - m, 0)$ to completely discard trivial negative background noise.

### 4. Out-of-Fold Threshold Calibration (Nelder-Mead Search)
Rather than fixing an arbitrary decision threshold ($0.5$), optimize class-specific decision thresholds on out-of-fold validation probabilities to maximize primary evaluation metrics directly (e.g. Macro-F1, Balanced Accuracy) without model retraining.

---

## 4. Multi-Task Learning, Routing & Ensembling

### 1. Multi-Task Heads
When predicting multiple related labels (e.g., primary sentiment + fine-grained aspect, or parent topic + sub-intent):
- Share low-level contextual representations across early encoder layers.
- Implement task-specific classification heads with dynamically weighted loss:
  $$\mathcal{L}_{\text{total}} = \sum_{k=1}^K w_k \cdot \mathcal{L}_k$$

### 2. Specialist Router Architecture
- Route broad instances to fast, comment-focused or tabular models.
- Route hard, ambiguous, or conditional instances to heavy deep reasoning / LLM models.

### 3. Ensemble Diversity Requirements (EnsembleGuard)
To prevent wasteful pseudo-ensembles and over-fitting:
- **Cohen's Kappa ($\kappa \le 0.85$)**: Candidates must exhibit genuine diversity in prediction distributions.
- **Disagreement Rate ($\ge 5.0\%$)**: Candidate pairs must disagree on at least 5% of validation samples.
- **Minimum Meaningful Effect Size (MMES $\ge 2.0\%$)**: Candidate predictions must demonstrably differ from baseline before submission sign-off.
