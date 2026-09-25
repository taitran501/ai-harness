# AGENTS.md — StereoQueerEval Operating Guidelines

This document defines the core architecture, constraints, and workflow for the **StereoQueerEval** repository (SemEval-2027 Task 8).

---

## 1. Task & Evaluation

- **Subtask A (Stereotype)**: Binary classification (`yes` / `no`). Metric: **Macro-F1**.
- **Subtask B (Hate Speech)**: 3 classes (`yes_explicit`, `yes_implicit`, `no`). Metric: **Macro-F1**.
- **Subtask C (Target & Identity)**: Evaluated **only** on hateful comments (`hate_speech != "no"`):
  - Target Type: `individual` vs `group`. Metric: **Macro-F1**.
  - Target Identities: 9 binary codes (`l`, `g`, `b`, `t`, `q`, `i`, `a`, `lgbtqia+`, `nb`). Metrics: **Macro-F1**, **Micro-F1**.
- **Languages**: English (`EN`), Italian (`IT`), Dutch (`NL`). Metrics must be reported per language.

---

## 2. Hard Constraints

1. **Grouped Cross-Validation**: All models must use `splits/fold_assignments.tsv` (grouped by video to prevent data leakage). Never split randomly.
2. **Immutable Artifacts**: Never overwrite or delete existing folders inside `artifacts/`. Each run requires a unique experiment ID.
3. **Pristine Raw Data**: Files in `dataset/` are strictly read-only.
4. **Reproducibility**: Set fixed seed (`42`), record git commit hash, and save resolved config in `run_config.yaml`.
5. **Hypothesis-First**: Every experiment must declare a `parent_id`, a clear hypothesis, and a single controlled change. Check the **Idea Graveyard** in `EXPERIMENTS.md` before proposing ideas.

---

## 3. Experiment Naming & Lineage

Use the semantic format:
```text
exp<NNN>_<branch>_<descriptor>
```

- **Branches**:
  - `baseline`: Reference benchmarks (e.g. `exp001_baseline_tfidf`)
  - `feat`: Text representations and context handling (e.g. `exp002_feat_title_context`)
  - `model`: Pretrained backbones and architectures (e.g. `exp010_model_xlmr_base`)
  - `loss`: Loss functions and imbalance handlers (e.g. `exp020_loss_focal_identities`)
  - `mtl`: Multi-task learning heads and routing (e.g. `exp030_mtl_hierarchical`)
  - `ens`: Ensembles, blending, and stacking (e.g. `exp050_ens_tfidf_xlmr`)

---

## 4. Two-Tier Memory System

- **Tier 1 (Global — `EXPERIMENTS.md`)**:
  - **Hall of Fame**: Current top-performing models per task and language.
  - **Idea Graveyard**: Documented failed approaches that must not be repeated.
  - **Leaderboard Table**: High-level lineage and scores.
- **Tier 2 (Local — `artifacts/<exp_id>/EXPERIMENT_CARD.md`)**:
  - Complete card per experiment with lineage, hypothesis, intervention, results, and post-mortem verdict.

---

## 5. Experiment Workflow (SOP)

1. **Check Memory**: Read `EXPERIMENTS.md` (check Hall of Fame for parent baseline, verify Idea Graveyard).
2. **Create Config**: Add `configs/exp<NNN>_<branch>_<descriptor>.yaml` specifying `parent_id`, `hypothesis`, and `intervention`.
3. **Smoke Test**: Verify pipeline on a single fold:
   ```powershell
   python train.py --config configs/<exp_config>.yaml --debug --fold 0
   ```
4. **Full Cross-Validation**: Run full 5-fold training:
   ```powershell
   python train.py --config configs/<exp_config>.yaml
   ```
5. **Verify Artifacts**: Ensure `artifacts/<exp_id>/` contains:
   - `run_config.yaml`
   - `run.log`
   - `metrics.json`
   - `summary.md`
   - `oof_predictions.csv`
   - `EXPERIMENT_CARD.md`
6. **Update Central Memory**: Record results in `EXPERIMENTS.md`. Update Hall of Fame or Idea Graveyard accordingly.

---

## 6. Submission & Audit Verification

To maintain rigorous competition integrity:
- **Packaging**: Use `tools/audit_gate.py` to package official submissions:
  ```powershell
  python tools/audit_gate.py --candidate <pred_csv> --baseline <baseline_csv> --output-zip submission.zip
  ```
- **Minimum Effect Size**: Candidate predictions must differ from baseline by at least 2.0% of validation samples.
- **Ensemble Diversity**: When combining models, enforce Cohen's Kappa $\kappa \le 0.85$ and disagreement rate $\ge 5.0\%$.

---

## 7. Command Reference

| Action | Command |
|---|---|
| Unit Tests | `python -m unittest discover -s tests -p "test_*.py"` |
| Smoke Test | `python train.py --config configs/<exp_config>.yaml --debug --fold 0` |
| Full 5-Fold CV | `python train.py --config configs/<exp_config>.yaml` |
| Audit Submission | `python tools/audit_gate.py --candidate <cand.csv> --baseline <base.csv> --output-zip submission.zip` |
| Leaderboard & Graveyard | Inspect `EXPERIMENTS.md` |
