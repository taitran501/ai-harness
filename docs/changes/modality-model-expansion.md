# Modality & model expansion (Agent Usability Phases 5–8)

- **Date**: 2026-10-15
- **PR**: #33 (Phase 7) · #34 (Phase 8) · Phase 5/6 feature branches
- **Status**: merged (squash)
- **Area**: models / evaluation / data strategy / task spec

## 1. Why (motivation + thought)

Expand support so agents never hit missing metrics, improper splits, or
validation barriers across non-NLP paradigms — Tabular, Time-Series/Sequential,
Computer Vision, Audio/Speech, NLP generation & Seq2Seq, Multilabel, and
quantitative finance. Deep transcript audits across
`~/.gemini/antigravity/brain` surfaced the real failure patterns.

## 2. What (concrete changes)

- **Built-in model wrappers** (`harness/models.py`):
  - *Phase 5* — `CatBoostModel` (aliases `catboost`/`cb`/...) with
    `allow_writing_files: False`; `validate_model_features()` diagnostics
    (replaces the opaque feature ValueError); `HarnessProject.__post_init__`
    auto-discovery defaults.
  - *Phase 6* — `RandomForestModel` (aliases `random_forest`/`rf`/...) with
    `n_jobs=-1`; extended `GenericBaselineModel.NAMES` (+ `lasso`);
    `TreeJournal.load_from_file` + flexible instance/class invocation;
    cross-environment dataset path basename fallback.
  - *Phase 7* — `RobustTabularTransformer` (median-imputes numerics,
    ordinal-encodes categoricals, unseen/missing → −1.0), default for
    `RandomForestModel`; `TIME_SERIES`/`TEMPORAL` split modes (leakage-free
    forward validation); `resolve_media_paths()` for CV/audio file resolution.
- **Metrics expansion** (Phase 8, `harness/evaluation.py`): time-series/finance
  (`rmsle`, `smape`, `wape`, `pinball`, `pearson`, `mrr`); CV (`dice`, `iou`,
  `mask_to_rle`/`rle_to_mask`, `box_iou`, `map_50`); audio (`wer`, `cer`); NLP
  generation (`bleu` w/ Laplace smoothing, `exact_match`, `token_f1`,
  `token_accuracy`); classification (`balanced_accuracy`, `weighted_f1`, `mcc`,
  `pr_auc`, `brier_score`); + `default_metric_direction()` inference.
- **Anti-leakage splits** (Phase 8, `harness/data_strategy.py`):
  `PurgedGroupTimeSeriesSplit` (configurable `embargo_pct`) for
  financial/sequential; `MultilabelStratifiedKFold` (greedy, no external deps).
- **Task-spec flexibility**: `TASK_TYPE_NORMALIZATIONS` alias map;
  `SubtaskSpec.from_dict` auto-normalizes + infers direction;
  `SubmissionPreflightValidator` accepts non-numeric predictions (RLE masks,
  ASR text, string class names) when `probability_mode=False`.

## 3. Approach (decisions + rejected alternatives)

Provide built-in domain metrics + anti-leakage splits + string-tolerant
validation so agents tackle these paradigms without fragile custom scoring
loops. Tree-safe numeric encoding with a −1.0 unseen-category sentinel prevents
string-conversion failures while preserving partition boundaries; time-series CV
uses expanding-window splits where validation strictly succeeds training.

## 4. Evidence

- Full suite grows across the phases: 431 → 435 → 440 → **449/449 tests, 100%
  pass** (6/4/5/9 dedicated cases in `test_agent_usability_p5..p8.py`).

## 5. Impact & follow-ups

- **Downstream**: agents can immediately tackle CV, audio, NLP-generation, and
  quantitative-finance tasks with proper splits + metrics + validation.
- Per-phase detail (Phase 5–8) remains in the archived `docs/WORK_LOG.md`.
