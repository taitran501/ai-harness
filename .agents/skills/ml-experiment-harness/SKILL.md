---
name: ml-experiment-harness
description: Manages ML experiments via AIDE tree search, enforces SWE-agent deterministic audit gate, and mandates Epistemic Grounding & Assumption Registers for SemEval-2027.
---

# ML Experiment Harness Skill

Use this skill whenever working on machine learning experiments, planning next interventions,
analyzing past runs (A1..A50), formulating hypotheses, or preparing competition submissions for StereoQueerEval.

## Core Operational Rules

### 1. The Context Protection Rule (Anti-Context Saturation)
- **NEVER** attempt to read all 50 `artifacts/exp*/` folders or raw log files into context.
- **ALWAYS** query the solution tree via CLI:
  ```powershell
  python tools/journal_tool.py --summary
  ```
- To inspect a specific experiment in depth:
  ```powershell
  python tools/journal_tool.py --node <exp_id>
  ```

### 2. The Anti-Pivot & Branch Exploration Rule
- When asked "what should we do next?", **DO NOT** guess, do superficial EDA, or arbitrarily recommend abandoning a subtask.
- Run the AIDE-guided UCB tree search:
  ```powershell
  python tools/journal_tool.py --recommend --max-params 150000000
  ```
- Base your recommendation strictly on the output branch, parent node, and UCB potential score.
- Ensure every new experiment has **EXACTLY ONE ATOMIC INTERVENTION** (Control vs Treatment).

### 3. The Epistemic Grounding & Assumption Register Protocol (Anti-Outdated Knowledge)
- **NEVER** rely on stale parametric memory or propose classical defaults (e.g. Random Forest, naive ensembling, arbitrary hyperparameter tweaks) as "safe guesses".
- **MANDATORY RETRIEVAL**: Before formulating a new hypothesis or config:
  1. Inspect `docs/KNOWLEDGE_BASE.md` for verified modern backbones (e.g. `microsoft/mdeberta-v3-base`, `FacebookAI/xlm-roberta-base`) and loss functions (Focal Loss, Class-Balanced Loss).
  2. If proposing a technique outside the knowledge base, execute external retrieval first via `search_web` or `read_url_content`.
- **MANDATORY CONFIG SCHEMA**: Every experiment config must include:
  ```yaml
  grounding:
    source_type: "huggingface_hub" # huggingface_hub | arxiv | paperswithcode | empirical_journal
    source_id: "microsoft/mdeberta-v3-base"
    retrieval_summary: "Detailed rationale citing multilingual tokenization for EN/IT/NL and past hate speech benchmark results."
    verified_via: "huggingface_hub"

  assumption_register:
    known_facts:
      - "Baseline exp001 scored 0.62 Subtask A Macro-F1 but 0.00 on rare identities 'a' and 'i'."
    assumptions:
      - "Disentangled attention will capture subtle derogatory nuances better than bag-of-words."
    uncertainty_level: "low" # low | medium | high
    verification_probe: "Run --debug --fold 0 smoke test first" # Required if uncertainty_level is high
  ```
- **PRE-FLIGHT VALIDATION**:
  Run the Grounding Gate CLI before training:
  ```powershell
  python harness/grounding_gate.py --config configs/<exp_config>.yaml
  ```
  If this gate fails, `train.py` will physically refuse to run.

### 4. The Diagnostic-First Exploration Rule
- Before proposing any new intervention or writing configs:
  1. Inspect `artifacts/<parent_id>/DIAGNOSTICS.md`.
  2. Identify the specific bottleneck (e.g., Subtask B implicit hate miss rate, rare identity zero-recall).
  3. Base the new hypothesis strictly on resolving that diagnosed bottleneck.
- **Strictly DO NOT pseudo-ensemble** models of the same family or merge previous versions mechanically.

### 5. The Physical & Statistical Audit Gate Protocol (Anti-Hallucination)
- You **CANNOT** manually create `submission.zip` or claim a submission is ready by natural language alone.
- You **MUST** run the physical audit gate comparing against the baseline:
  ```powershell
  python tools/audit_gate.py --candidate <candidate_pred_csv> --baseline <baseline_pred_csv> --output-zip submission.zip
  ```
- If the candidate changes less than **2.0% of validation rows** (MMES floor), the gate will reject it as random noise.
- If submitting an ensemble, you MUST provide constituent models:
  ```powershell
  python tools/audit_gate.py --candidate <ens_csv> --baseline <base_csv> --ensemble-components model1.csv model2.csv
  ```
  The gate will physically reject pseudo-ensembles with Cohen's Kappa $\kappa > 0.85$.
- You can ONLY present the submission to the user if:
  1. The audit gate returns `PASSED` (exit code 0).
  2. `AUDIT_RECEIPT.json` is generated on disk with documented effect size and statistical metrics.
  3. You report the exact differing sample count directly from the receipt.

### 6. Submission Management & Score Tracking Protocol
- All verified packages MUST be registered and tracked in `submissions/`:
  ```powershell
  python tools/submission_tool.py register --exp-id <id> --candidate <cand_csv> --baseline <base_csv>
  ```
- View registered packages, Local CV vs Leaderboard scores:
  ```powershell
  python tools/submission_tool.py list
  ```
- When competition organizers release evaluation scores, record them immediately:
  ```powershell
  python tools/submission_tool.py record-score --sub-id <sub_id> --macro-a <F1> --macro-b <F1> --macro-c-type <F1>
  ```
- If $\Delta (\text{LB} - \text{CV}) < -0.05$, the system triggers a **Divergence Warning ⚠️**.

### 7. Hard Constraints Pre-Flight Check & Cloud Offloading
Before running any heavy training:
- Verify that `max_params_limit` is respected.
- Ensure `splits/fold_assignments.tsv` is strictly used (no random splitting).
- Run pre-flight inspection before cloud dispatch:
  ```powershell
  python harness/cloud_inspector.py --config <config_path>
  ```
- Offload to cloud GPU if training Transformer backbones (XLM-R, DeBERTa):
  ```powershell
  python harness/cloud_runner.py --backend modal --config <config_path> --auto-submit
  ```
