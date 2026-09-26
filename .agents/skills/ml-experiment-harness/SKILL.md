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

## 2. Three-Layer Architecture (`Antigravity IDE` <-> `AgentBridge` <-> `Harness`)

- **Layer 1 — True Agent Runtime (`Antigravity in IDE`)**: Performs scientific reasoning, hypothesis formulation, and the `Observe -> Decide -> Next Action` loop. The Harness does **not** run a separate internal LLM planner.
- **Layer 2 — Agent Bridge & Context Bridge (`harness/agent_bridge.py` & `tools/agent_bridge_cli.py`)**:
  - Feeds Antigravity a compressed **Context Packet** (`CompetitionState` 8-Phase HUD, Parallel Tracks, `TreeJournal` UCB recommendation, Parent `DIAGNOSTICS` bottlenecks, Idea Graveyard, and active `BridgeSession` budget/usage):
    ```powershell
    python tools/agent_bridge_cli.py context
    ```
  - Tracks stateful **`BridgeSession` (`session_XXXX`)** (`goal`, `experiments_run`, `compute_usage`, `consecutive_rejections`, optional `max_experiments` / `max_gpu_hours` / `max_cloud_runs` budgets) and enforces **`USER_APPROVAL` governance** on sensitive actions (`prepare_submission`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`).
  - Exposes **Typed Harness Actions** returning structured `ActionObservation` JSON payloads (`metric_delta`, `composite_score`, `diagnostics_bottlenecks`, `new_blockers`):
    ```powershell
    python tools/agent_bridge_cli.py start-session --goal "Improve target metric" --max-experiments 3
    python tools/agent_bridge_cli.py act --action <typed_action> --params '{"...": "..."}'
    ```
- **Layer 3 — Competition State Machine & Execution Runtime (`harness/competition_state.py`, `train.py`, `harness/cloud_runner.py`)**:
  - **8-Phase Journey**: `1. Setup` -> `2. Understand` -> `3. Baseline Anchor` -> **Parallel Iteration Tracks** (`4. Experiment Loop` || `5. Candidate Management` || `6. External Evaluation`) -> `7. Freeze (Lock Candidate Portfolio & Mutations)` -> `8. Closure`.
  - **Execution Backends**: Local (`train.py`), Modal Serverless GPU (`--backend modal`), and Kaggle Kernels CLI (`--backend kaggle`).

---

## 3. Core Operational Rules (Enforced & Immutable)

### 3.1 The Context Protection Rule (Anti-Context Saturation)
- **NEVER** attempt to dump raw fold logs or entire `artifacts/` directories into prompt context.
- **ALWAYS** query the Agent Bridge Context Packet or Solution Tree CLI first:
  ```powershell
  python tools/agent_bridge_cli.py context
  python tools/journal_tool.py --summary
  ```

### 3.2 The Anti-Pivot & Branch Exploration Rule (AIDE UCB Tree Search)
- **DO NOT** make speculative guesses or pivot arbitrarily without mathematical justification.
- Query the optimal branch recommendation according to Upper Confidence Bound (UCB):
  ```powershell
  python tools/journal_tool.py --recommend
  ```
- Ensure every experiment implements **EXACTLY ONE ATOMIC INTERVENTION** (Control vs Treatment).

### 3.3 Epistemic Grounding & Assumption Register Protocol (Anti-Hallucination)
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

### 3.4 Diagnostic-First Exploration Rule
- Before writing any config, inspect `artifacts/<parent_id>/DIAGNOSTICS.md` (or call `act --action inspect_diagnostics`).
- Declare a `problem_addressed` that directly targets a documented failure slice or confusion mode.
- **Strictly DO NOT pseudo-ensemble** models of the same family or merge previous versions mechanically.

### 3.5 SWE-Agent Physical & Statistical Audit Gate
- Submissions and candidate predictions CANNOT be approved by natural language alone.
- **EnsembleGuard (`harness/ensemble_guard.py`)**: Checks constituent model diversity before ensembling: Cohen's Kappa ($\kappa \le 0.85$) and Disagreement Rate ($\ge 5.0\%$).
- **AuditGate (`tools/audit_gate.py`)**: Performs pre-submission statistical verification, SHA-256 duplicate detection, Minimum Meaningful Effect Size (MMES >= 2.0%), McNemar significance testing, and signs `AUDIT_RECEIPT.json`.
  ```powershell
  python tools/audit_gate.py --candidate <candidate_pred_csv> --baseline <baseline_pred_csv> --output-zip submission.zip
  ```
- **Minimum Meaningful Effect Size (MMES >= 2.0%)**: Candidates differing by < 2% of validation samples are physically rejected.

### 3.6 Approval Governance (`AUTO` / `AUTO_WITH_LIMIT` / `USER_APPROVAL`)
- **AUTO**: `inspect_data_readiness`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`
- **AUTO_WITH_LIMIT (Session Budget Tracked)**: `establish_baseline_anchor`, `run_full_validation` (`local` | `modal` | `kaggle`), `promote_candidate`, `synthesize_candidate`, `build_ensemble`
- **USER_APPROVAL (Mandatory Human Confirmation)**: `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`

---

## 4. Standard Operating Procedure (SOP) for Antigravity in IDE

1. **Read Bridge Context Packet**: Run `python tools/agent_bridge_cli.py context` to inspect the 8-Phase Competition State, active `BridgeSession`, UCB branch recommendation, and parent `DIAGNOSTICS` bottlenecks.
2. **Start / Configure Session (Optional Budgets)**: Run `python tools/agent_bridge_cli.py start-session --goal "..." --max-experiments 3`.
3. **Create Grounded Experiment**: Run `python tools/agent_bridge_cli.py act --action create_experiment --params '{...}'` (or write `configs/exp<NNN>_<branch>_<descriptor>.yaml` and verify via `harness/grounding_gate.py`).
4. **Execute Smoke Test**: Run `python tools/agent_bridge_cli.py act --action run_smoke_test --params '{"config_path": "configs/<exp>.yaml"}'`.
5. **Execute Full 5-Fold Validation (`local` | `modal` | `kaggle`)**: Run `python tools/agent_bridge_cli.py act --action run_full_validation --params '{"config_path": "configs/<exp>.yaml", "backend": "local"}'`.
6. **Observe & Decide Next Step**: Antigravity analyzes `ActionObservation` (`metric_delta`, `composite_score`, `diagnostics_bottlenecks`, `consecutive_rejections`). Promote candidates via `promote_candidate` (single experiment) OR `synthesize_candidate` (multi-model composition/specialist router).
7. **Human-Gated Submission / Freeze**: Request user confirmation before invoking `prepare_submission` or `freeze_candidate` with `--user-approved`.
