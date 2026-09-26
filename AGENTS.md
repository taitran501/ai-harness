# AGENTS.md — Universal Agent-Operable Competition Harness

This document defines the 3-layer architecture, constraints, operational state machine, and workflow for the ML Competition Harness.

> **Motto & Operating Boundary**:  
> **“Antigravity thinks. AgentBridge coordinates. Harness verifies and executes.”**

---

## 1. Three-Layer Architecture (`Antigravity IDE` ↔ `AgentBridge` ↔ `Harness`)

```text
Antigravity IDE (Layer 1)
   │
   │ Scientific Reasoning / Hypothesis / Decision Loop
   ▼
AgentBridge (Layer 2)
   │
   │ Context Bridge / Stateful Sessions / Typed Actions
   │ Budget Guards / Soft Stop Advisories / Human Approval
   ▼
Competition & Execution Harness (Layer 3)
   │
   ├─ Competition State Machine (8-Phase Journey)
   ├─ Experiment Memory & TreeJournal (AIDE UCB Tree)
   ├─ Deterministic Guardrails (ExperimentValidator, GroundingGate, AuditGate)
   ├─ Diagnostics & Error Slices
   ├─ Candidate Pool & Composition Strategies
   └─ Execution Backends (Local / Modal / Kaggle)
```

1. **Layer 1 — True Agent Runtime (`Antigravity in IDE`)**:
   - Antigravity inside the IDE owns all scientific reasoning, hypothesis formulation, diagnostics analysis, and the `Observe -> Decide -> Action` loop.
   - The Harness does **not** run an internal LLM planner loop; its runtime is strictly dedicated to state tracking, guardrails, and execution.
2. **Layer 2 — Bidirectional Agent Bridge (`harness/agent_bridge.py` & `tools/agent_bridge_cli.py`)**:
   - **Context Bridge (`context`)**: Synthesizes a high-density **Context Packet** containing the 8-Phase Journey HUD, Parallel Iteration Tracks, `TreeJournal` UCB recommendations, Parent `DIAGNOSTICS` bottlenecks, Idea Graveyard, and active `BridgeSession` telemetry.
   - **Stateful Session & Budget Guards (`BridgeSession`)**: Tracks `session_id`, `goal`, `experiments_run`, compute usage (`local_runs`, `cloud_runs`, `gpu_hours_used`, `wall_seconds_used`), and syncs live activity to `competition_state.json`.
   - **Hard Stop vs. Soft Stop (`REASSESS_RECOMMENDED`)**:
     - *Hard Stop*: Resource exhaustion (`max_experiments`, `max_gpu_hours` $\rightarrow$ `BLOCKED_BY_BUDGET`), frozen competition state (`STOP_POLICY_TRIGGERED`), or pending human sign-off (`PAUSED_FOR_APPROVAL`).
     - *Soft Stop*: When consecutive non-improving runs reach `max_consecutive_rejections`, the bridge returns advisory `REASSESS_RECOMMENDED` without hard-blocking, allowing Antigravity to autonomously pivot strategies.
   - **Typed Action Contract (`act`) & Approval Governance (`ApprovalTier`)**:
     - `AUTO`: `inspect_data_readiness`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`
     - `AUTO_WITH_LIMIT`: `establish_baseline_anchor`, `run_full_validation` (`--backend local|modal|kaggle`), `synthesize_candidate`, `promote_candidate`, `build_ensemble`
     - `USER_APPROVAL` (Mandatory Human Sign-off): `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`
3. **Layer 3 — Universal Competition State Machine & Execution Harness**:
   - **8-Phase Competition Journey**:
     1. `Setup`: Load `configs/task.yaml`, verify environment, dataset availability, split schemas, and execution prerequisites.
     2. `Understand`: Data & schema analysis, class distributions, leakage risk profiling, and evaluation assumptions.
     3. `Baseline Anchor`: Execute benchmark model, evaluate OOF metrics, and lock invariant baseline score.
     4. `Parallel Track 1 — Experiment Loop`: Grounded iterations across exploration branches (`feat`, `model`, `loss`, `mtl`, `ens`).
     5. `Parallel Track 2 — Candidate Management`: Candidate Pool governance and composition strategies (specialist routing, weighted blending, rank averaging).
     6. `Parallel Track 3 — External Evaluation`: External leaderboard tracking, submission registry, and CV-vs-LB alignment.
     7. `Freeze`: Strict code/pipeline lock prior to final submission.
     8. `Closure`: Post-competition audit receipts, code archiving, and technical report generation.
   - **Deterministic Guardrails & Clear Boundaries**:
     - `ExperimentValidator` (`harness/validator.py`): Verifies leakage prevention, group split integrity, schema alignment, and shape consistency.
     - `GroundingGate` (`harness/grounding_gate.py`): Enforces epistemic verification (parent ID, problem addressed, hypothesis, grounded literature/empirical basis, and assumption register) before persisting any config.
     - `AuditGate` (`tools/audit_gate.py`): Performs pre-submission statistical verification, SHA-256 duplicate detection, Minimum Meaningful Effect Size (MMES $\ge$ 2.0%), McNemar significance testing, EnsembleGuard diversity checks (Cohen's $\kappa \le 0.85$, disagreement $\ge 5.0\%$), and signs `AUDIT_RECEIPT.json`.

---

## 2. Universal Operating Principles

1. **Grounded Experiment Creation**:
   - Step 1: Construct candidate config in-memory.
   - Step 2: Validate through `GroundingGate.verify_config()`.
   - Step 3: Persist verified config to `configs/exp<NNN>_<branch>_<descriptor>.yaml`.
2. **Experiment-to-Candidate Governance**:
   - Running `run_full_validation` computes OOF metrics and updates `TreeJournal`. An experiment does **not** automatically become a competition candidate.
   - Antigravity analyzes diagnostics/bottlenecks, then explicitly invokes `promote_candidate` or `synthesize_candidate` to enter the **Candidate Pool**.
3. **Reproducibility & Immutability**:
   - Set fixed seed (`42`), record git commit hash, and write resolved configs.
   - Never overwrite existing folders inside `artifacts/`. Raw data files are strictly read-only.
4. **Two-Tier Memory System**:
   - **Tier 1 (Global)**: `EXPERIMENTS.md` (Hall of Fame & Idea Graveyard) + `artifacts/journal.json` (AIDE UCB Tree).
   - **Tier 2 (Local)**: `artifacts/<exp_id>/EXPERIMENT_CARD.md` & `DIAGNOSTICS.md` (detailed error slices).

---

## 3. Current Competition Adapter: StereoQueerEval (SemEval-2027 Task 8)

*This section captures domain-specific rules configured via `configs/task.yaml` and competition datasets.*

- **Tasks & Evaluation Metrics**:
  - **Subtask A (Stereotype)**: Binary classification (`yes` / `no`). Metric: **Macro-F1**.
  - **Subtask B (Hate Speech)**: 3-class classification (`yes_explicit`, `yes_implicit`, `no`). Metric: **Macro-F1**.
  - **Subtask C (Target & Identity)**: Conditional on hateful comments (`hate_speech != "no"`):
    - Target Type: `individual` vs `group`. Metric: **Macro-F1**.
    - Target Identities: 9 binary codes (`l`, `g`, `b`, `t`, `q`, `i`, `a`, `lgbtqia+`, `nb`). Metrics: **Macro-F1**, **Micro-F1**.
  - **Languages**: English (`EN`), Italian (`IT`), Dutch (`NL`). Metrics reported per language and overall.
- **Cross-Validation Split Strategy**:
  - Grouped by `yt_title` to prevent comment cross-video leakage (`splits/fold_assignments.tsv`).
- **Composition Strategy**:
  - `specialist_router`: Routes Subtask A/B to comment-focused models and Subtask C to context-aware models.
- **External Evaluation Platform**: Codabench.

---

## 4. Standard Operating Procedure (SOP) via Agent Bridge

```powershell
# 1. Inspect Context Packet
python tools/agent_bridge_cli.py context

# 2. Start / Configure Session with Budgets
python tools/agent_bridge_cli.py start-session --goal "Improve Subtask C Macro-F1" --max-experiments 3

# 3. Create Grounded Config (In-memory verification -> Persist)
python tools/agent_bridge_cli.py act --action create_experiment --params '{"branch": "feat", "descriptor": "title_ctx_balanced"}'

# 4. Smoke Test (Fold 0 Probe)
python tools/agent_bridge_cli.py act --action run_smoke_test --params '{"config_path": "configs/<exp_config>.yaml"}'

# 5. Full Validation (Local / Modal / Kaggle)
python tools/agent_bridge_cli.py act --action run_full_validation --params '{"config_path": "configs/<exp_config>.yaml", "backend": "local"}'

# 6. Synthesize Candidate / Specialist Router
python tools/agent_bridge_cli.py act --action synthesize_candidate --params '{"name": "router_v1", "mode": "specialist_router"}'

# 7. Audit & Submission (Requires USER_APPROVAL / Human Sign-off)
python tools/agent_bridge_cli.py act --action prepare_submission --user-approved --params '{"exp_id": "<id>", "candidate_csv": "<cand_csv>", "baseline_csv": "<base_csv>"}'
```

---

## 5. Command Reference

| Action | Command | Governance |
|---|---|---|
| Read Bridge Context Packet | `python tools/agent_bridge_cli.py context` | `Read-only` |
| Execute Typed Bridge Action | `python tools/agent_bridge_cli.py act --action <name> --params '{...}'` | `AUTO / AUTO_WITH_LIMIT` |
| Approve Pending Action | `python tools/agent_bridge_cli.py approve --session-id <session_id>` | `Human Sign-off` |
| Solution Tree UCB Recommend | `python tools/journal_tool.py --recommend` | `Read-only` |
| Direct Smoke Test | `python train.py --config configs/<exp_config>.yaml --debug --fold 0` | `Execution` |
| Direct Full 5-Fold CV | `python train.py --config configs/<exp_config>.yaml` | `Execution` |
| Audit Submission (SHA-256 / MMES) | `python tools/audit_gate.py --candidate <cand.csv> --baseline <base.csv> --output-zip submission.zip` | `AuditGate` |
| Run Test Suite | `python -m unittest discover -s tests -p "test_*.py"` | `Verification` |

