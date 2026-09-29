# Competition Experiment Memory & Leaderboard (Tier 1 Global Memory)

This document serves as the human- and agent-readable central tracking registry for all cross-validation experiments and hypotheses.
All metrics represent cross-validation performance across the defined validation scheme (`task.yaml`).

---

## 1. Current Best Models (Hall of Fame)

Use these models as the parent baseline anchor for subsequent iterations:

| Rank | Exp ID | Branch | Model Family | Primary Metric | Score | Key Advantage / Breakthrough |
|---|---|---|---|---|---:|---|
| 👑 **1** | `exp001_tfidf_baseline` | `baseline` | `linear` | Macro-F1 | **0.4500** | Canonical linear baseline establishing validation floor. |

---

## 2. Idea Graveyard (Negative Knowledge Base — DO NOT RETRY)

Before proposing any new experiment, agents MUST check this table. Do not repeat failed interventions without novel empirical justification:

| Tested Intervention | Tested In | Observed Impact | Root Cause / Post-Mortem | Directive |
|---|---|---|---|---|
| **Raw Unnormalized Features** | Initial probe | Metric degradation (-0.08) | Large feature magnitude differences dominated gradient updates. | Always apply scaling/normalization before linear or neural training. |
| **Identical Model Stacking** | Synthetic probe | Kappa = 0.98, Disagreement < 1.0% | Constituent models shared identical inductive bias; zero diversity. | Ensembles must satisfy Cohen's Kappa $\kappa \le 0.85$ and Disagreement $\ge 5\%$. |

---

## 3. Experiment History & Lineage Registry

| Exp ID | Branch | Parent ID | Approach | Model Family | Primary Metric | Composite Score | Status | Notes |
|---|---|---|---|---|---|---:|---|---|
| `exp000_template` | `baseline` | `none` | `ml` | `linear` | Macro-F1 | 0.0000 | `ARCHIVED` | Template configuration. |
| `exp001_tfidf_baseline` | `baseline` | `none` | `ml` | `linear` | Macro-F1 | 0.4500 | `COMPLETED` | Canonical baseline reference implementation. |

---

## 4. Lineage Management Instructions for Agents

1. **Before Running**: Look up the current top model in **Hall of Fame** (`journal_tool.py --recommend` or `agent_bridge_cli.py context`).
2. **Atomic Intervention**: Modify exactly one hypothesis / variable at a time (Control vs Treatment).
3. **If Rejected / Degraded**: Immediately log the root cause into **Section 2 (Idea Graveyard)** to prevent circular re-testing.
4. **If Improved by $\ge 2.0\%$**: Record the experiment as the new baseline anchor or candidate pool member.
| `exp001_baseline_tfidf` | `baseline` | `none` | 2026-09-23 | tfidf_logistic_regression |  | 0.2005 | `artifacts/exp001_baseline_tfidf` | Establish a competitive, fast, and reproducible sparse control baseline across all three subtasks. |
