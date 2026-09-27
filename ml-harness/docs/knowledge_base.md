# Modeling Reference Guide

This document lists recommended backbones, modeling frameworks, and loss functions for competitive machine learning benchmarks.

---

## 1. Natural Language Processing

| Model Backbone | Parameters | Modality / Coverage | Key Highlights | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **`microsoft/deberta-v3-base`** | ~86M (+160M emb) | English | Disentangled Attention, RTD training | Strong default baseline for English classification. |
| **`microsoft/mdeberta-v3-base`** | ~86M (+160M emb) | 100+ languages | Multilingual vocabulary | Multilingual classification and cross-lingual transfer. |
| **`FacebookAI/xlm-roberta-base`** | ~125M (+150M emb) | 100 languages | Robust masked language model | Social media text, noisy text, informal slang. |
| **`answerdotai/ModernBERT-base`** | ~149M total | English / Transfer | FlashAttention-2, 8192 context window | Fast inference and longer document classification. |
| **`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** | ~7B (4-bit NF4) | Multilingual | Native 4-bit QLoRA, low VRAM footprint | Generative classification, complex world knowledge. |

---

## 2. Tabular Modeling

| Model / Library | Engine Type | Recommended Hyperparameter Priors | Best Suited For |
| :--- | :--- | :--- | :--- |
| **`LightGBM`** | Histogram gradient boosting, leaf-wise | `learning_rate: 0.03-0.08`, `num_leaves: 31-127`, `subsample: 0.8` | Fast baseline, large tabular datasets. |
| **`XGBoost`** | Depth-wise gradient boosting | `max_depth: 4-8`, `eta: 0.05`, `subsample: 0.85` | Competitive accuracy, L1/L2 regularization. |
| **`CatBoost`** | Symmetric (oblivious) decision trees | `depth: 6-8`, `l2_leaf_reg: 3-10`, `iterations: 1000-2500` | Heavy categorical features, robust defaults. |

---

## 3. Loss Functions & Class Imbalance

1. **Focal Loss (Lin et al., ICCV 2017)**:
   Downweights easy negative examples to focus training on ambiguous boundary cases:
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Recommended priors: $\gamma \in [1.5, 2.5]$, $\alpha \in [0.25, 0.75]$.

2. **Class-Balanced Loss (Cui et al., CVPR 2019)**:
   Weights loss inversely by effective sample volume $E_n = \frac{1 - \beta^n}{1 - \beta}$ with $\beta \in [0.99, 0.9999]$.

3. **Asymmetric Loss (ASL, Ridnik et al., ICCV 2021)**:
   Decouples positive and negative focusing parameters for extreme multi-label sparsity.

4. **Post-Hoc Decision Threshold Tuning**:
   Tune class decision thresholds on out-of-fold validation probabilities (e.g. via Nelder-Mead optimization) to directly maximize Macro-F1 without model retraining.
