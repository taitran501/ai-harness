# Modeling Reference Guide

This document lists recommended backbones, modeling frameworks, and loss functions for competitive machine learning benchmarks.

> [!IMPORTANT]
> **Current Verification Scope: NLP / Text ONLY**
> ML-Harness defaults and empirical registries are currently **strictly verified and working well ONLY on Natural Language Processing (NLP / Text)** benchmarks (e.g., `stereoqueereval`, sequence classification, transformer fine-tuning, and prompt evaluation).
>
> All other modalities (**Tabular, Computer Vision, Audio, Multimodal**) are **experimental works-in-progress (WIP / Roadmap)**, have **NOT** been verified or tested in competitive conditions, and are **NOT** considered ready or working well in practice.
>
> In accordance with the AIDE / Agentic ML philosophy, the coding agent (Antigravity/Codex) is autonomous: **Section 1 (NLP) contains verified reference baselines, whereas all subsequent sections are experimental/roadmap references, NOT hardcoded constraints or certified baselines**. The agent is free to define, modify, or author novel model architectures (`model.factory`), feature representations, and loss formulations.

---

## 1. Natural Language Processing (Verified Core)

| Model Backbone | Parameters | Modality / Coverage | Key Highlights | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **`microsoft/deberta-v3-base`** | ~86M (+160M emb) | English | Disentangled Attention, RTD training | Strong default baseline for English classification. |
| **`microsoft/mdeberta-v3-base`** | ~86M (+160M emb) | 100+ languages | Multilingual vocabulary | Multilingual classification and cross-lingual transfer. |
| **`FacebookAI/xlm-roberta-base`** | ~125M (+150M emb) | 100 languages | Robust masked language model | Social media text, noisy text, informal slang. |
| **`answerdotai/ModernBERT-base`** | ~149M total | English / Transfer | FlashAttention-2, 8192 context window | Fast inference and longer document classification. |
| **`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** | ~7B (4-bit NF4) | Multilingual | Native 4-bit QLoRA, low VRAM footprint | Generative classification, complex world knowledge. |

---

## 2. Tabular Modeling (Experimental / WIP — Unverified in Practice)

> [!WARNING]
> Tabular modeling support is currently experimental work-in-progress (WIP). It has **NOT** been verified or benchmarked in real competitive conditions, and is **NOT** currently considered working well. Use with caution.

| Model / Library | Engine Type | Recommended Hyperparameter Priors | Best Suited For |
| :--- | :--- | :--- | :--- |
| **`LightGBM`** | Histogram gradient boosting, leaf-wise | `learning_rate: 0.03-0.08`, `num_leaves: 31-127`, `subsample: 0.8` | Fast baseline, large tabular datasets. Uses native categorical splitting. |
| **`XGBoost`** | Depth-wise gradient boosting | `max_depth: 4-8`, `eta: 0.05`, `subsample: 0.85` | Competitive accuracy, L1/L2 regularization, `enable_categorical=True`. |
| **`CatBoost`** | Symmetric (oblivious) decision trees | `depth: 6-8`, `l2_leaf_reg: 3-10`, `iterations: 1000-2500` | Heavy categorical features, robust defaults. |

---

## 3. Computer Vision (Roadmap / WIP — Unverified)

> [!NOTE]
> Computer Vision backbones are planned roadmap items (WIP) and are not yet verified in practice.

| Model Backbone | Parameters | Framework / Library | Key Highlights | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **`convnext_small.fb_in22k_ft_in1k`** | ~50M | `timm` / PyTorch | Modern pure-convolutional inductive bias | Image classification, spatial feature extraction. |
| **`swin_base_patch4_window7_224`** | ~88M | `timm` / Hugging Face | Shifted-window hierarchical ViT | Dense prediction, multi-scale image representations. |
| **`vit_base_patch16_224`** | ~86M | `timm` / Hugging Face | Standard global attention Vision Transformer | High-resolution visual understanding with pretraining. |
| **`efficientnet_b3.ra2_in1k`** | ~12M | `timm` / PyTorch | Compound scaling, fast inference | Low-compute / edge vision tasks. |

---

## 4. Audio Processing (Roadmap / WIP — Unverified)

> [!NOTE]
> Audio processing models are planned roadmap items (WIP) and are not yet verified in practice.

| Model Backbone | Parameters | Framework / Library | Key Highlights | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **`openai/whisper-small`** | ~244M | Hugging Face Transformers | Encoder-decoder speech recognition & representations | Multilingual audio transcription and acoustic classification. |
| **`facebook/wav2vec2-base`** | ~95M | Hugging Face Transformers | Self-supervised raw audio waveform modeling | Phonetic classification, emotion detection, voice recognition. |
| **`MIT/ast-finetuned-audioset-10-10-0.4593`** | ~86M | Hugging Face Transformers | Audio Spectrogram Transformer (AST) | General environmental sound and event detection. |

---

## 5. Multimodal Fusion & Vision-Language (Roadmap / WIP — Unverified)

> [!NOTE]
> Multimodal fusion architectures are planned roadmap items (WIP) and are not yet verified in practice.

| Model Backbone | Parameters | Modality / Coverage | Key Highlights | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **`openai/clip-vit-base-patch32`** | ~151M | Image + Text | Contrastive joint embedding space | Zero-shot image classification, text-image retrieval. |
| **`google/siglip-base-patch16-224`** | ~204M | Image + Text | Sigmoid loss for language-image pretraining | Robust multi-label multimodal alignment and zero-shot probing. |
| **`Qwen/Qwen2-VL-7B-Instruct`** | ~7B | Vision-Language | High-resolution spatial reasoning & OCR | Visual question answering, diagram/document reasoning. |

---

## 6. Loss Functions & Class Imbalance

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
