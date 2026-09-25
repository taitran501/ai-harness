---
name: ml-experiment-harness
description: Manages ML experiments for StereoQueerEval on Codabench via AIDE tree search, enforces SWE-agent deterministic audit gate, and mandates Epistemic Grounding & Assumption Registers.
---

# ML Experiment Harness Skill — StereoQueerEval

> **Active Benchmark**: **StereoQueerEval** | **Platform**: Codabench | **Modality**: Natural Language Processing (NLP)
> **Objective**: Maximize `Macro-F1` across cross-validation folds.

Use this skill whenever planning model architectures, engineering features, formulating hypotheses,
auditing predictions, or registering verified submissions for **StereoQueerEval**.

---

## 1. Benchmark Specification & Score Objectives

- **Task Type**: `Multi-Task Text Classification`
- **Primary Target Metric**: `Macro-F1` (Optimization Direction: `maximize`)
- **Baseline Reference Score**: `0.6459`
- **Target Goal Score**: `0.85`
- **Target Languages**: EN, IT, NL

### Evaluation Criteria & Subtask Breakdown:
| Criteria / Subtask | Metric | Direction | Baseline | Target | Weight | Description |
|---|---|---|---|---|---|---|
| **Stereotype Detection** (`subtask_a`) | `Macro-F1` | `maximize` | `0.6962` | `0.88` | `0.333` | Binary stereotype detection (no vs yes) across EN, IT, NL |
| **Hate Speech Classification** (`subtask_b`) | `Macro-F1` | `maximize` | `0.5238` | `0.8` | `0.333` | 3-class classification (no vs yes_explicit vs yes_implicit) |
| **Target Type** (`subtask_c_type`) | `Macro-F1` | `maximize` | `0.7178` | `0.85` | `0.167` | Evaluated strictly on oracle_hate comments (hate_speech != no) |
| **Target Identities** (`subtask_c_identity`) | `Macro-F1` | `maximize` | `0.2408` | `0.85` | `0.167` | Multi-label classification across 9 identity codes |

### Data & Cross-Validation Strategy:
- **Validation Scheme**: `GroupKFold` (5 folds, seed 42)
- **Anti-Leakage Group Column**: `yt_title`
- **Dataset Paths**: Train: `dataset/` | Test: `dataset/test/`
- **Key Columns**: ID: `StereoQueerEval_id` | Target: `stereotype,hate_speech,target_type,target_identities`

### Competition Rules & Constraints:
- **Allowed LLM / API**: `No (Local/Open models only)`
- **External Data**: `Forbidden`
- **Pretrained Weights**: `Allowed`
- **Parameter Ceiling**: `150M parameters`
- **Daily Submission Quota**: `5 submissions / day`
- **Competition Deadline**: `2027-01-15T23:59:59Z`

### Submission & Audit Specification:
- **File Name / Format**: `submission.zip` (`zip`)
- **MMES Effect Size Threshold**: `>= 2.0%` of validation rows
- **Ensemble Diversity Ceiling**: Cohen's Kappa `kappa <= 0.85`
- **Ensemble Disagreement Floor**: `>= 5.0%`

---

## 2. Core Operational Rules (Enforced & Immutable)

### 2.1 The Context Protection Rule (Anti-Context Saturation)
- **NEVER** attempt to dump raw fold logs or entire `artifacts/` directories into prompt context.
- **ALWAYS** query the solution tree CLI for compressed state:
  ```powershell
  python tools/journal_tool.py --summary
  ```
- To inspect a specific node or parent in depth:
  ```powershell
  python tools/journal_tool.py --node <exp_id>
  ```

### 2.2 The Anti-Pivot & Branch Exploration Rule (AIDE UCB Tree Search)
- **DO NOT** make speculative guesses or pivot arbitrarily without mathematical justification.
- Query the optimal branch recommendation according to Upper Confidence Bound (UCB):
  ```powershell
  python tools/journal_tool.py --recommend
  ```
- Ensure every experiment implements **EXACTLY ONE ATOMIC INTERVENTION** (Control vs Treatment).

### 2.3 Epistemic Grounding & Assumption Register Protocol (Anti-Hallucination)
- **MANDATORY RETRIEVAL**: Before formulating a new model configuration for Natural Language Processing (NLP):
  1. Inspect `docs/KNOWLEDGE_BASE.md` for verified backbones:
     - `microsoft/deberta-v3-base (English / General SOTA encoder)`
     - `microsoft/mdeberta-v3-base (Multilingual cross-lingual transfer)`
     - `FacebookAI/xlm-roberta-base (Robust multilingual tokenization)`
     - `answerdotai/ModernBERT-base (Fast, long-context 8k modern encoder)`
  2. Inspect domain-verified loss functions: Focal Loss (gamma=2.0), Class-Balanced BCE, Asymmetric Loss.
  3. Technique requirements: Pay careful attention to token lengths, special metadata tags, and multilingual subwords.
  4. If proposing an unlisted technique, perform external retrieval (`search_web`, Hugging Face Hub, ArXiv) first.
- **MANDATORY CONFIG SCHEMA**: Declare explicit `grounding:` and `assumption_register:` blocks in `configs/exp<NNN>.yaml`.
- **PRE-FLIGHT VALIDATION GATE**: Validate config before launching training:
  ```powershell
  python harness/grounding_gate.py --config configs/<exp_config>.yaml
  ```

### 2.4 Diagnostic-First Exploration Rule
- Before writing any config, inspect `artifacts/<parent_id>/DIAGNOSTICS.md`.
- Declare a `problem_addressed` that directly targets a documented failure slice or confusion mode.
- **Strictly DO NOT pseudo-ensemble** models of the same family or merge previous versions mechanically.

### 2.5 SWE-Agent Physical & Statistical Audit Gate
- Submissions and candidate predictions CANNOT be approved by natural language alone.
- All submission archives MUST be generated through `tools/audit_gate.py`:
  ```powershell
  python tools/audit_gate.py --candidate <candidate_pred_csv> --baseline <baseline_pred_csv> --output-zip submission.zip
  ```
- **Minimum Meaningful Effect Size (MMES >= 2.0%)**: Candidates differing by < 2% of validation samples are physically rejected.
- **Ensemble Diversity Check**: Constituent models must satisfy **Cohen's Kappa ($\kappa \le 0.85$)** and **Disagreement Rate ($\ge 5.0\%$)**.

### 2.6 Submission Management & Divergence Tracking
- Register all verified packages permanently:
  ```powershell
  python tools/submission_tool.py register --exp-id <id> --candidate <cand_csv> --baseline <base_csv>
  ```
- List registered submissions and comparison against local CV:
  ```powershell
  python tools/submission_tool.py list
  ```
- Record official competition scores when released:
  ```powershell
  python tools/submission_tool.py record-score --sub-id <sub_id> --score <val>
  ```
- If $\Delta (\text{LB} - \text{CV}) < -0.05$, the system triggers a **Divergence Warning ⚠️** requiring CV leak diagnostics.

### 2.7 Cloud Execution & GPU Offloading (Modal.com & Kaggle CLI)
- Run local pre-flight checks before remote dispatch:
  ```powershell
  python harness/cloud_inspector.py --config configs/<exp_config>.yaml
  ```
- Dispatch remote training:
  ```powershell
  # For Modal:
  python harness/cloud_runner.py --backend modal --config configs/<exp_config>.yaml --auto-submit
  # For Kaggle:
  python harness/cloud_runner.py --backend kaggle --config configs/<exp_config>.yaml
  ```

---

## 3. Standard Operating Procedure (SOP) for New Experiments

1. **Consult Central Memory**: Run `python tools/journal_tool.py --recommend` to identify parent node and target branch.
2. **Inspect Parent Diagnostics**: Check `artifacts/<parent_id>/DIAGNOSTICS.md` to pinpoint specific bottlenecks.
3. **Draft Grounded Config**: Create `configs/exp<NNN>_<branch>_<descriptor>.yaml` declaring hypothesis, grounding, and assumptions.
4. **Execute Verification Smoke Test**: Run single-fold probe `python train.py --config configs/<exp_config>.yaml --debug --fold 0`.
5. **Execute Full Cross-Validation**: Run full training `python train.py --config configs/<exp_config>.yaml`.
6. **Verify Artifacts & Audit Gate**: Confirm `metrics.json`, `summary.md`, `oof_predictions.csv`, and run `tools/audit_gate.py`.
7. **Update Central Memory**: Verify solution tree sync via `python tools/journal_tool.py --summary`.
