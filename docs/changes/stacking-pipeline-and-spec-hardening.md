# Stacking, Pipeline Stage Identity, and Task Spec Fail-Closed Hardening

- **Date:** 2026-11-14
- **Author/Agent:** Sol 6.1 (implemented by Antigravity)
- **PR:** #137
- **Status:** Complete

## Why

During an adversarial first-principles audit by Sol 6.1, six reproducible correctness invariant gaps were uncovered across secondary runtime paths:
1. **Regression stacking truncation & distortion**: `L2LogitStacker.fit_predict` cast `y` to `np.int64` and passed continuous outputs through logit transformations, causing regression targets between 0.0 and 1.0 to truncate to zero and predict constant `0.0`.
2. **Stacking identity gaps**: `align_prediction_frames` checked only set subset, allowing duplicate prediction IDs to inflate output rows and `RankAverager.blend` to return `[nan, nan]` when passed zero-sum weights.
3. **Pipeline stage traversal & name collision**: `_resolve_pipeline_stages` allowed relative path traversals (e.g. `../../outside`) and could resolve duplicate names into colliding identifiers (`a, a_2, a` resolving to `a, a_2, a_2`).
4. **Pipeline CSV reading loses lexical IDs**: Reading predictions via raw `pd.read_csv` without preserving string IDs collapsed IDs like `"001"` into integers, triggering false prediction contract rejections.
5. **Summary.csv placeholder score**: Pipeline and custom script adapters hardcoded `primary_score: 0.0` in summary rows, leaving `summary.csv` conflicting with the evaluated `metrics.json`.
6. **Task config syntax error fallback**: Corrupt or malformed YAML in `configs/task.yaml` was swallowed silently, falling back to default competition metadata instead of failing closed.

## What

1. **`harness/stacking.py`**:
   - Preserves `y` as `float64` for regression tasks, uses raw continuous predictions as meta-features (no logit transform), and applies Ridge regression.
   - Enforces unique, non-null, exact ID set matching in `align_prediction_frames`, rejecting duplicates and mismatches.
   - Rejects zero-sum or non-positive weights in `RankAverager.blend`.
2. **`harness/run_pipeline.py` & `harness/runner.py`**:
   - Validates stage names to prevent directory traversal (`..`, `/`, `\`) and uses an incrementing collision loop to guarantee unique stage names.
   - Uses `read_prediction_frame` with expected IDs to preserve lexical string IDs.
   - Synchronizes evaluated `primary_score` to summary rows before writing `summary.csv`.
3. **`harness/competition_state.py` & `harness/project.py`**:
   - Fails closed with `StateIntegrityError` when `configs/task.yaml` contains invalid YAML syntax or non-mapping root.
4. **`harness/candidate_preflight.py`**:
   - Rejects candidate and incumbent prediction files with duplicate IDs or mismatched ID coverage.

## Verification

- `tests/unit/test_hardened_invariants.py`: 10 new unit tests covering regression fractional labels, duplicate ID rejection, zero-weight rejection, stage name collision/traversal, lexical ID preservation, corrupt YAML fail-closed, and candidate exact coverage.
- `python tools/run_tests.py --smoke`: 84/84 tests passed (24.8s).
- `python -m unittest tests/integration/test_pipeline_execution.py tests/integration/test_candidate_preflight.py`: 27/27 passed.
- `python -m unittest tests/integration/test_submission_manager.py`: 102/102 passed.
- `ruff check harness/ tools/ tests/`: All checks passed.
