# StereoQueerEval Experiment Leaderboard

This document serves as the central tracking registry for all cross-validation experiments.
All metrics represent the mean over 5 video-grouped folds (`splits/fold_assignments.tsv`).
Subtask C metrics are computed conditionally on gold hateful comments (`oracle_hate`).

---

## 1. Current Best Models (Hall of Fame)

Use these models as the parent baseline for new iterations:

| Subtask | Language | Best Exp ID | Model / Context | Metric | Score |
|---|---|---|---|---|---:|
| **Subtask A (Stereotype)** | English (`EN`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.6962** |
| **Subtask A (Stereotype)** | Italian (`IT`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.6356** |
| **Subtask A (Stereotype)** | Dutch (`NL`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.6727** |
| **Subtask B (Hate Speech)** | English (`EN`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.5238** |
| **Subtask B (Hate Speech)** | Italian (`IT`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.5308** |
| **Subtask B (Hate Speech)** | Dutch (`NL`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.5165** |
| **Subtask C (Target Type)** | English (`EN`) | `baseline_v1_comment` | TF-IDF / `comment` | Macro-F1 | **0.7178** |
| **Subtask C (Identities)** | English (`EN`) | `baseline_v1_full_context` | TF-IDF / `full_context` | Micro-F1 | **0.7476** |

---

## 2. Idea Graveyard (Negative Knowledge Base — DO NOT RETRY)

Before proposing any new experiment, agents MUST check this table. Do not repeat failed interventions without explicit human authorization:

| Tested Intervention | Tested In | Observed Impact | Root Cause / Post-Mortem | Directive |
|---|---|---|---|---|
| **Video Description for Subtask A & B** | `baseline_v1_full_context` | Macro-F1 drops: EN A (-0.053), EN B (-0.053), IT A (-0.054), NL A (-0.048). | Long video descriptions dilute comment-level lexical signals and introduce out-of-domain noise. | **Do NOT** use `full_context` for Subtask A or B classification. Use `comment` or `title_comment`. |
| **Unconditioned Subtask C Training** | Pre-experiment analysis | Training target/identity models on non-hateful rows collapsed type precision. | 70%+ of comments are non-hateful and possess `target=none`. | Always filter to `hate_speech != 'no'` for Subtask C (`oracle_hate`). |

---

## 3. Experiment History & Lineage Registry

| Exp ID | Branch | Parent | Date | Model / Backbone | Context | EN Macro A | EN Macro B | EN C Type/Id | IT Macro A | NL Macro A | Artifact Path | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `baseline_v1_comment` | `baseline` | `none` | 2026-09-20 | TF-IDF + LogisticReg | comment | 0.6962 | 0.5238 | 0.7178 / 0.2408 | 0.6356 | 0.6727 | `artifacts/baseline_v1` | Strongest Subtask A & B across all languages for sparse baseline. |
| `baseline_v1_title_comment` | `feat` | `baseline_v1_comment` | 2026-09-20 | TF-IDF + LogisticReg | title_comment | 0.6783 | 0.4873 | 0.6400 / 0.2835 | 0.6234 | 0.6632 | `artifacts/baseline_v1` | Video title improves identity classification at slight cost to A/B. |
| `baseline_v1_full_context` | `feat` | `baseline_v1_comment` | 2026-09-20 | TF-IDF + LogisticReg | full_context | 0.6432 | 0.4709 | 0.6284 / 0.2944 | 0.5814 | 0.6246 | `artifacts/baseline_v1` | Highest identity micro-F1, but degrades A and B significantly. |
