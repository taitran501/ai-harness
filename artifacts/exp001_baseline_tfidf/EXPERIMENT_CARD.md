# Experiment Card: `exp001_baseline_tfidf`

## 1. Lineage & Metadata
- **Experiment ID**: `exp001_baseline_tfidf`
- **Branch / Category**: `baseline`
- **Parent Baseline**: `none`
- **Date**: `2026-09-23`
- **Git Commit**: `dd8043af5dd3cfb335350e47953edab90e467b87-dirty`
- **Problem Addressed**: Establish a competitive, fast, and reproducible sparse control baseline across all three subtasks.

## 2. Hypothesis
> Word (1-2) and char (3-5) n-gram TF-IDF provides a solid lower bound for Subtasks A and B, but will exhibit low Macro-F1 on rare Subtask C identity labels due to lack of semantic generalization.

## 3. Intervention (Control vs Treatment)
- **Controlled Baseline**: `none`
- **Intervention Applied**: Canonical baseline reference implementation across comment, title_comment, and full_context variants.

## 4. Cross-Validation Results

| Language | Input Variant | A Macro-F1 | B Macro-F1 | C Type Macro-F1 | C Id Macro-F1 | C Id Micro-F1 |
|---|---|---:|---:|---:|---:|---:|
| EN | comment | 0.8730 | 0.2222 | 0.2000 | 0.0741 | 0.4000 |
| EN | full_context | 0.2000 | 0.4333 | 0.4000 | 0.0741 | 0.4000 |
| EN | title_comment | 0.3333 | 0.0000 | 0.4286 | 0.0889 | 0.5714 |
| IT | comment | 0.6000 | 0.0000 | 0.1667 | 0.0000 | 0.0000 |
| IT | full_context | 0.4118 | 0.2333 | 0.2000 | 0.0741 | 0.2000 |
| IT | title_comment | 0.3750 | 0.0000 | 0.2000 | 0.0741 | 0.2000 |
| NL | comment | 0.4958 | 0.5556 | 0.4000 | 0.0000 | 0.0000 |
| NL | full_context | 0.3333 | 0.3686 | 0.2500 | 0.0000 | 0.0000 |
| NL | title_comment | 0.4958 | 0.5167 | 0.2000 | 0.0556 | 0.2857 |

## 5. Post-Mortem & Verdict

- **Outcome Status**: `[PENDING REVIEW]` (Update to `[CONFIRMED]`, `[REFUTED]`, or `[INCONCLUSIVE]` upon human inspection).
- **Key Finding**: Document whether the intervention improved the target metrics and whether trade-offs occurred.
- **Next Action**: State whether this branch should be promoted to the Hall of Fame, refined, or deprecated into the Idea Graveyard.
