# Experiment Card: `exp001_baseline_tfidf`

## 1. Lineage & Metadata
- **Experiment ID**: `exp001_baseline_tfidf`
- **Branch / Category**: `baseline`
- **Parent Baseline**: `none`
- **Date**: `2026-09-23`
- **Git Commit**: `764af56ea8c1384c7a7ddbdc95fc83d2ae6d2ad8-dirty`
- **Problem Addressed**: Establish a competitive, fast, and reproducible sparse control baseline across all three subtasks.

## 2. Hypothesis
> Word (1-2) and char (3-5) n-gram TF-IDF provides a solid lower bound for Subtasks A and B, but will exhibit low Macro-F1 on rare Subtask C identity labels due to lack of semantic generalization.

## 3. Intervention (Control vs Treatment)
- **Controlled Baseline**: `none`
- **Intervention Applied**: Canonical baseline reference implementation across comment, title_comment, and full_context variants.

## 4. Cross-Validation Results

| input_variant   |   dev_fold |   train_rows |   dev_rows |   composite_score | language   |   subtask_a_f1 |   subtask_b_f1 |   subtask_c_type_f1 |   subtask_c_type_oracle_f1 |   subtask_c_identity_f1 |
|:----------------|-----------:|-------------:|-----------:|------------------:|:-----------|---------------:|---------------:|--------------------:|---------------------------:|------------------------:|
| full_context    |          0 |           60 |          7 |           0.20048 | IT         |              0 |              0 |                 0.5 |                        0.5 |                       0 |

## 5. Post-Mortem & Verdict

- **Outcome Status**: `[PENDING REVIEW]` (Update to `[CONFIRMED]`, `[REFUTED]`, or `[INCONCLUSIVE]` upon human inspection).
- **Key Finding**: Document whether the intervention improved the target metrics and whether trade-offs occurred.
- **Next Action**: State whether this branch should be promoted to the Hall of Fame, refined, or deprecated into the Idea Graveyard.
