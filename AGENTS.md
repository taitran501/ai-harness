# AGENTS.md — StereoQueerEval Engineering & Operating Contract

Welcome, AI Agent. This document defines the architectural conventions, scientific constraints,
and operational procedures for the **StereoQueerEval** repository (SemEval-2027 Task 8).

Any AI agent interacting with this repository MUST adhere strictly to the rules and workflows
specified herein.

---

## 1. Task Definition & Evaluation Standards

### 1.1 Objectives
StereoQueerEval aims to analyze YouTube comments related to the LGBTQIA+ community, conditioned
on contextual video metadata (`yt_title` and `yt_description`). The task comprises three subtasks:

1. **Subtask A (Stereotype Detection)**:
   - Binary classification: `stereotype` in `{"no", "yes"}`.
   - Primary metric: **Macro-averaged F1**.
2. **Subtask B (Hate Speech Classification)**:
   - 3-class classification: `hate_speech` in `{"no", "yes_explicit", "yes_implicit"}`.
   - Primary metric: **Macro-averaged F1**.
3. **Subtask C (Target Type & Identity Classification)**:
   - **Critical Evaluation Rule (`oracle_hate`)**: Evaluated **ONLY** on comments where `hate_speech != "no"`. Comments labeled `no` default to `target_type="none"` and zero identity vectors.
   - **Target Type**: Binary classification between `individual` and `group`. Primary metric: **Macro-averaged F1**.
   - **Target Identities**: Multi-label classification across 9 identity codes: `["l", "g", "b", "t", "q", "i", "a", "lgbtqia+", "nb"]`. Primary metrics: **Macro-averaged F1** and **Micro-averaged F1**.

### 1.2 Languages
- Supported training languages: **English (`EN`)**, **Italian (`IT`)**, and **Dutch (`NL`)**.
- Persian (`FA`) is planned as an evaluation/zero-shot language.
- **Reporting Requirement**: Because class balances and linguistic patterns diverge significantly across languages, all metrics must be reported per language in addition to overall macro scores.

---

## 2. Hard Scientific Constraints (Immutable Rules)

Agents must never violate the following constraints:

1. **Strict Video-Grouped Cross-Validation**:
   - Comments originating from the same YouTube video share identical title and description metadata. Random comment-level splitting causes data leakage.
   - All models MUST use the canonical precomputed splits defined in `splits/fold_assignments.tsv`. Never generate new splits unless explicitly requested by the user.
2. **Experiment Immutability**:
   - Folders inside `artifacts/` are permanent historical records.
   - Never overwrite, delete, or alter an existing experiment directory.
   - Every modification (e.g., hyperparameter tuning, model backbone switch, input variant change) constitutes a new experiment requiring a unique identifier (e.g., `exp002_...`).
3. **Raw Data Invariance**:
   - Never modify or write to files inside `dataset/`. Raw TSV data must remain pristine and read-only.
4. **Reproducibility Guarantee**:
   - Every run must set fixed random seeds (`seed: 42`).
   - Every run must capture git commit hash, environment versions, and full resolved parameters in `run_config.yaml`.
5. **The Hypothesis-First Rule**:
   - Science is hypothesis-driven, not a series of blind trial-and-error edits.
   - Never modify code or run training without:
     - Formulating a clear scientific hypothesis (what is expected to improve and why).
     - Declaring the `parent_id` baseline being modified.
     - Declaring the single controlled intervention (Control vs Treatment).
   - **Negative Knowledge Defense**: Always check the **Idea Graveyard** in `EXPERIMENTS.md` before proposing an intervention. Never retry an approach already proven detrimental without explicit human instruction.

---

## 3. Semantic Naming Conventions & Branch Taxonomy

Linear version numbers (`v1..v100` or `A1..A100`) obscure scientific intent and cause agent context drift.
All experiments must follow the semantic identifier format:

$$\text{exp}\langle\text{NNN}\rangle\_\langle\text{branch}\rangle\_\langle\text{descriptor}\rangle$$

### Recognized Branches
- **`baseline`**: Foundational benchmarks and reference controls (e.g., `exp001_baseline_tfidf`).
- **`feat`**: Feature engineering, context representations, text concatenations (e.g., `exp002_feat_title_context`).
- **`model`**: Model backbone switches, pretrained weights, architectures (e.g., `exp010_model_xlmr_base`).
- **`loss`**: Loss functions, objective weighting, class imbalance handlers (e.g., `exp020_loss_focal_identities`).
- **`mtl`**: Multi-task learning, shared representations, hierarchical routing (e.g., `exp030_mtl_hierarchical`).
- **`ens`**: Ensembling, stacking, blending, threshold optimization (e.g., `exp050_ens_tfidf_xlmr`).

---

## 4. Two-Tier Memory System for Agents

To prevent context window saturation and hallucinations across dozens or hundreds of runs:

### Tier 1: Global Summary (`EXPERIMENTS.md`)
The agent reads `EXPERIMENTS.md` to understand:
- **The Hall of Fame**: Current top-performing models per task and language to select as parent baselines.
- **The Idea Graveyard**: Approaches that definitively failed and must not be retried.
- **The Lineage Table**: High-level mapping from parent experiments to children.

### Tier 2: Local Experiment Cards (`artifacts/<exp_id>/EXPERIMENT_CARD.md`)
Every experiment bundle MUST generate an `EXPERIMENT_CARD.md` containing:
1. **Lineage**: `parent_id`, `branch`, problem addressed.
2. **Hypothesis**: The scientific assumption and expected quantitative impact.
3. **Intervention**: Control vs Treatment differences.
4. **CV Results Table**: Macro-F1 across A, B, and C.
5. **Post-Mortem & Verdict**: `[CONFIRMED]`, `[REFUTED]`, or `[INCONCLUSIVE]`, plus lessons learned.

The agent only reads detailed experiment cards for the specific parent baseline it is actively extending.

---

## 5. Standard Operating Procedure (SOP) for Experiments

When instructed to evaluate an idea or improve performance:

### Step 1: Consult Global Memory
Inspect `EXPERIMENTS.md`:
- Check the **Hall of Fame** to identify the best parent model.
- Check the **Idea Graveyard** to ensure the proposed intervention was not already refuted.

### Step 2: Formulate Config with Hypothesis Metadata
Create `configs/exp<NNN>_<branch>_<descriptor>.yaml`. Declare:
```yaml
experiment:
  id: "exp014_loss_focal_identities"
  branch: "loss"
  parent_id: "exp010_model_xlmr_base"
  date: "2026-09-24"
  problem_addressed: "Severe class imbalance on rare Subtask C identities."
  hypothesis: "Replacing BCE with Focal Loss (gamma=2.0) will boost rare identity Macro-F1 by +0.03."
  intervention: "Switch Subtask C identity loss to FocalLoss(gamma=2.0, alpha=0.25)."
```

### Step 3: Perform Smoke Testing (Sanity Check)
Always verify pipeline execution on a small sample or single fold before committing GPU/CPU resources:
```powershell
python train.py --config configs/exp<NNN>_<branch>_<descriptor>.yaml --debug --fold 0
```
Verify that data loaders, tensor shapes, loss gradients, and metric calculations execute without warnings or zero-division errors.

### Step 4: Execute Full Grouped Cross-Validation
Run the full 5-fold evaluation across all target languages:
```powershell
python train.py --config configs/exp<NNN>_<branch>_<descriptor>.yaml
```

### Step 5: Verify Generated Artifacts
Confirm that `artifacts/<exp_id>/` contains:
1. `run_config.yaml`: Frozen config with git hash and runtime metadata.
2. `run.log`: Full execution log.
3. `metrics.json`: Detailed per-fold, per-language, and per-class metrics.
4. `summary.csv`: Tabular fold-by-fold results.
5. `summary.md`: Mean ± std markdown table.
6. `oof_predictions.csv`: Out-of-fold predictions with `StereoQueerEval_id`.
7. `EXPERIMENT_CARD.md`: Complete experiment card with lineage, hypothesis, results, and post-mortem.

### Step 6: Update Central Memory
- Verify `EXPERIMENTS.md` was updated with the new run.
- If the model outperforms the existing record, promote it in the **Hall of Fame**.
- If the hypothesis was decisively refuted, document the negative finding in the **Idea Graveyard**.

---

## 6. Command Reference

| Action | Command |
|---|---|
| Run Unit Tests | `python -m unittest discover -s tests -p "test_*.py"` |
| Run Smoke Test | `python train.py --config configs/<exp_config>.yaml --debug --fold 0` |
| Run Full CV | `python train.py --config configs/<exp_config>.yaml` |
| View Tree Summary | `python tools/journal_tool.py --summary` |
| Get Next Branch Recommendation | `python tools/journal_tool.py --recommend --max-params 150000000` |
| Cloud Pre-Flight Check | `python harness/cloud_inspector.py --config configs/<exp_config>.yaml` |
| Dispatch to Cloud GPU | `python harness/cloud_runner.py --backend modal --config configs/<exp_config>.yaml --auto-submit` |
| Register Verified Submission | `python tools/submission_tool.py register --exp-id <id> --candidate <cand_csv> --baseline <base_csv>` |
| List Submissions & CV vs LB | `python tools/submission_tool.py list` |
| Record Official LB Score | `python tools/submission_tool.py record-score --sub-id <sub_id> --macro-a <F1> --macro-b <F1> --macro-c-type <F1>` |
| View Leaderboard & Graveyard | Inspect `EXPERIMENTS.md` |
| View Submission Tracker | Inspect `submissions/SUBMISSIONS.md` |

---

## 7. Known Pitfalls & Edge Cases
- **Support Imbalance in Identity Labels**: Identity labels such as `a`, `i`, and `nb` possess very small support in the training set. Macro-F1 scores on these classes exhibit high variance across folds. Always inspect per-class support alongside macro metrics.
- **Context Noise**: Including video descriptions (`full_context`) introduces noise in Subtask A and B classifiers. As documented in the Idea Graveyard, avoid `full_context` for Subtask A/B.
- **Target Formatting**: The target column uses the compact syntax `target_type_identity1,identity2` (e.g., `group_l,g`). Never treat this string as a single categorical class.

---

## 8. The Deterministic & Statistical Audit Gate Protocol (Strictly Enforced)

To prevent post-hoc rationalization, confabulation, submitting duplicate files, or doing trivial tweaks:
1. **No Direct Submission Creation**: The agent is physically forbidden from declaring a submission ready or manually packaging `submission.zip`.
2. **Mandatory Execution of Audit Gate**: All submissions MUST be packaged through `tools/audit_gate.py`:
   ```powershell
   python tools/audit_gate.py --candidate <candidate_pred_csv> --baseline <baseline_pred_csv> --output-zip submission.zip
   ```
3. **Minimum Meaningful Effect Size (MMES >= 2.0%)**:
   - The candidate must differ from the baseline by at least **2.0% of total validation samples**. Trivial modifications or random seed variations changing only a handful of rows are physically rejected.
4. **Ensemble Submissions Verification**:
   - If submitting an ensemble, the agent MUST provide constituent model predictions:
     ```powershell
     python tools/audit_gate.py --candidate <ens_csv> --baseline <base_csv> --ensemble-components model1.csv model2.csv
     ```
   - The audit gate enforces **Cohen's Kappa ($\kappa \le 0.85$)** and **Disagreement Rate ($\ge 5.0\%$)**. Highly correlated models from the same family are blocked as pseudo-ensembles.
5. **Anti-Confabulation Defense**:
   - If the audit gate reports identical SHA-256 hashes, zero differences, or effect size < 2.0%, the agent MUST NOT invent post-hoc explanations (e.g., claiming zero-loss or already-hit queries without running real code).
   - The agent MUST accept the physical test failure, debug the pipeline, and verify why the intervention produced no behavioral change.
6. **Human Acceptance Criteria**: The human user will reject any submission claim that lacks a valid `AUDIT_RECEIPT.json` generated on disk.

---

## 9. Tree Search & Memory Compression Protocol

When navigating multiple iterations (A1..A50+):
1. **Never dump 50 raw experiment files into prompt context.** Query `python tools/journal_tool.py --summary`.
2. **Never pivot arbitrarily without evidence.** Query `python tools/journal_tool.py --recommend` to retrieve the mathematically optimal branch according to Upper Confidence Bound (UCB) tree search.

---

## 10. The Anti-Degeneration Protocol (Diagnostic-First & Anti-Pseudo-Ensemble)

When agents run experiments over long horizons (A1..A50+), cognitive fatigue tends to cause heuristic degeneration (merging old code, pseudo-ensembling, superficial tweaks). The following rules prevent this:

1. **Strict Prohibition on Pseudo-Ensembling**:
   - An ensemble is **strictly forbidden** if it merely averages models from the same architecture family (e.g. two TF-IDF variants).
   - Ensembles are allowed **ONLY** when combining at least 2 fundamentally distinct model families (e.g., TF-IDF + XLM-RoBERTa) that achieve $\kappa \le 0.85$ and a CV score margin $\ge +0.005$ over the best constituent.
2. **Diagnostic-First Hypotheses**:
   - Before writing any new config, inspect `artifacts/<parent_id>/DIAGNOSTICS.md`.
   - The declared `problem_addressed` MUST directly quote a documented failure mode (e.g. *"Subtask B implicit hate miss rate is 45%"* or *"Identity class 'a' has zero recall"*).
3. **No Version Regressing**:
   - Never 'merge' old versions or copy-paste previous baselines under a new name without a single, controlled atomic modification.

---

## 11. Epistemic Grounding & Assumption Register Protocol (Anti-Outdated Knowledge)

Inspired by Microsoft RD-Agent, AIDE, and Edwards & Schuster (2026) (*Ask or Assume?*):

1. **Mandatory Retrieval-Grounding**:
   - The agent is physically forbidden from proposing model architectures or training interventions based solely on stale parametric memory or "safe guesses" (e.g. Random Forest, simple ensembling, arbitrary hyperparameter tweaks).
   - The agent MUST inspect `docs/KNOWLEDGE_BASE.md` or execute external retrieval (`search_web`, `read_url_content`, Hugging Face Hub) to verify that the proposed model supports the target languages (EN, IT, NL) and addresses the diagnosed failure mode.
2. **Schema Enforcement (`grounding` & `assumption_register`)**:
   - All experiment configurations MUST include explicit `grounding:` and `assumption_register:` blocks.
   - Any config lacking these sections or specifying lazy interventions (*"tune parameters"*, *"try different things"*, *"simple ensemble"*) will be immediately halted by `harness/grounding_gate.py` with exit code 1.
3. **Physical Receipt Generation**:
   - Every run will record `artifacts/<exp_id>/GROUNDING_RECEIPT.json` documenting the empirical source and uncertainty rating. Experiments with `uncertainty_level: high` must execute a verification probe before full 5-fold training.

---

## 12. Submission Lifecycle & Cloud Offloading Protocol

1. **Submission Management**:
   - All competition submission packages are permanently tracked in `submissions/` via `python tools/submission_tool.py`.
   - Never upload a file that does not possess a `LOCAL_VERIFIED` status and a valid `AUDIT_RECEIPT.json`.
   - When official Leaderboard results are released, agents/engineers MUST record scores via:
     ```powershell
     python tools/submission_tool.py record-score --sub-id <sub_id> --macro-a <F1> --macro-b <F1> --macro-c-type <F1>
     ```
   - If $\text{LB} - \text{CV} < -0.05$, the system triggers a **Divergence Warning ⚠️**. The agent must conduct cross-validation leak diagnostics before submitting subsequent models.
2. **Cloud GPU Execution**:
   - Heavy Transformer models (XLM-R, DeBERTa) MUST run pre-flight checks before dispatch:
     ```powershell
     python harness/cloud_inspector.py --config configs/<exp_config>.yaml
     ```
   - Dispatch to Modal or Kaggle using `harness/cloud_runner.py`:
     ```powershell
     python harness/cloud_runner.py --backend modal --config configs/<exp_config>.yaml --auto-submit
     ```
