# Failure Diagnostics Report: `exp000_template`

This report details ground-truth failure modes, error distributions, and critical bottlenecks.
Agents MUST use these findings to formulate hypotheses rather than doing pseudo-ensembling.

## 1. Critical Bottlenecks (Action Items for Agent)

- 🔴 **Severe implicit hate speech miss rate: 60.0% of 'yes_implicit' misclassified as 'no'.**
- 🔴 **Zero recall on rare identity classes: ['g', 'lgbtqia+', 'nb'].**

## 2. Subtask B (Hate Speech) Confusion Matrix

| True Label \ Predicted | `no` | `yes_explicit` | `yes_implicit` |
|---|---:|---:|---:|
| **no** | 8 | 5 | 0 |
| **yes_explicit** | 5 | 1 | 1 |
| **yes_implicit** | 6 | 3 | 1 |

> **Implicit Hate Miss Rate**: 60.0% (6 / 10 misclassified as `no`).

## 3. Subtask C Minority Identity Analysis

| Identity | Support | TP | FP | FN | Precision | Recall | F1 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `l` | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | ⚠️ LOW |
| `g` | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | ❌ ZERO RECALL |
| `b` | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | ⚠️ LOW |
| `t` | 8 | 1 | 0 | 7 | 1.0000 | 0.1250 | 0.2222 | ⚠️ LOW |
| `q` | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | ⚠️ LOW |
| `i` | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | ⚠️ LOW |
| `a` | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | ⚠️ LOW |
| `lgbtqia+` | 5 | 0 | 2 | 5 | 0.0000 | 0.0000 | 0.0000 | ❌ ZERO RECALL |
| `nb` | 1 | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | ❌ ZERO RECALL |

## 4. Per-Language Accuracy Breakdown

| Language | Total Samples | Hateful Samples | Subtask A Acc | Subtask B Acc |
|---|---:|---:|---:|---:|
| **EN** | 8 | 4 | 0.8750 | 0.3750 |
| **IT** | 10 | 8 | 0.7000 | 0.0000 |
| **NL** | 12 | 5 | 0.5833 | 0.5833 |
