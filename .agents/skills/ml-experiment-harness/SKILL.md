---
name: ml-experiment-harness
description: Manages ML experiments for StereoQueerEval on Codabench via solution tree tracking, physical audit gates, and grounded experiment cards.
---

# ML Experiment Harness — StereoQueerEval

Operational reference and execution guidelines for SemEval-2027 Task 8 (**StereoQueerEval**).

---

## 1. Benchmark & Evaluation

- **Task**: Multi-task text classification on YouTube comments with video metadata.
- **Primary Metric**: Macro-F1 across 5 folds.
- **Languages**: English (`EN`), Italian (`IT`), Dutch (`NL`).

### Subtasks
- **Subtask A (Stereotype)**: Binary (`yes` / `no`). Metric: Macro-F1.
- **Subtask B (Hate Speech)**: 3 classes (`yes_explicit`, `yes_implicit`, `no`). Metric: Macro-F1.
- **Subtask C (Target Type)**: `individual` vs `group`. Evaluated only on hateful comments (`oracle_hate`). Metric: Macro-F1.
- **Subtask C (Identities)**: 9 multi-label codes (`l`, `g`, `b`, `t`, `q`, `i`, `a`, `lgbtqia+`, `nb`). Evaluated on `oracle_hate`. Metrics: Macro-F1, Micro-F1.

### Cross-Validation & Data Rules
- **Split Strategy**: GroupKFold (5 folds, seed 42) grouped strictly by video title/description to prevent leakage.
- **Dataset Path**: `dataset/`
- **Canonical Splits**: `splits/fold_assignments.tsv`
- **Allowed Models**: Local and open weights only. External LLM APIs are disallowed by competition rules.
- **External Data**: Strictly forbidden by competition constraints.
- **Model Size Ceiling**: 150M parameters.

---

## 2. Core Execution Rules

### 2.1 Immutability & Reproducibility
- Folders in `artifacts/` are permanent records. Never overwrite or delete past runs.
- Always fix random seed (`42`) and record git commit hash in `run_config.yaml`.

### 2.2 Hypothesis-First Experimentation
- Every new run must extend a declared `parent_id` baseline.
- Declare a single controlled change (Control vs Treatment).
- Check the **Idea Graveyard** in `EXPERIMENTS.md` before proposing ideas.

### 2.3 Two-Tier Memory
- **Global Memory (`EXPERIMENTS.md`)**: Hall of Fame (top models) and Idea Graveyard (failed approaches).
- **Local Memory (`artifacts/<exp_id>/EXPERIMENT_CARD.md`)**: Detailed hypothesis, results, and post-mortem per run.

### 2.4 Submission Verification Gate
- Package all official submissions through `tools/audit_gate.py`:
  ```powershell
  python tools/audit_gate.py --candidate <pred_csv> --baseline <baseline_csv> --output-zip submission.zip
  ```
- Predictions must differ by at least 2.0% of validation samples from baseline.
- Ensembles must satisfy Cohen's Kappa $\kappa \le 0.85$ and disagreement rate $\ge 5.0\%$.

---

## 3. Standard Operating Procedure (SOP)

1. **Inspect Memory**: Check `EXPERIMENTS.md` for current best baseline and graveyard findings.
2. **Create Config**: Add `configs/exp<NNN>_<branch>_<descriptor>.yaml`.
3. **Smoke Test**: Run on Fold 0 in debug mode:
   ```powershell
   python train.py --config configs/<exp_config>.yaml --debug --fold 0
   ```
4. **Full Training**: Execute 5-fold cross-validation:
   ```powershell
   python train.py --config configs/<exp_config>.yaml
   ```
5. **Verify Artifacts**: Ensure `artifacts/<exp_id>/` contains `metrics.json`, `summary.md`, `oof_predictions.csv`, and `EXPERIMENT_CARD.md`.
6. **Update Leaderboard**: Sync results into `EXPERIMENTS.md`.
