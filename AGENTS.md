# AGENTS.md — StereoQueerEval Operating Guidelines (3-Layer Agentic Harness)

This document defines the core 3-layer architecture, constraints, and workflow for the **StereoQueerEval** repository (SemEval-2027 Task 8).

---

## 1. Three-Layer Architecture (`Antigravity IDE` ↔ `AgentBridge` ↔ `Harness`)

1. **Layer 1 — True Agent Runtime (`Antigravity in IDE`)**:
   - Antigravity inside the IDE owns all scientific reasoning, hypothesis formulation, and the `Observe -> Decide -> Next Action` decision loop.
   - The Harness does **not** run a separate internal LLM planner loop; its runtime is strictly dedicated to state tracking, guardrails, and training execution (`local`, `modal`, `kaggle`).
2. **Layer 2 — Bidirectional Agent Bridge (`harness/agent_bridge.py` & `tools/agent_bridge_cli.py`)**:
   - **Context Bridge (`context`)**: Synthesizes a compact, high-density **Context Packet** containing the 8-Phase Competition State HUD, Parallel Iteration Tracks, `TreeJournal` UCB recommendation, Parent `DIAGNOSTICS` bottlenecks, Idea Graveyard, and active `BridgeSession` telemetry.
   - **Stateful Session & Budget Guards (`BridgeSession`)**: Tracks `session_id`, `goal`, `experiments_run`, `compute_usage` (`local_runs`, `cloud_runs`, `gpu_hours_used`, `wall_seconds_used`), `consecutive_rejections`, optional budgets (`max_experiments`, `max_gpu_hours`, `max_cloud_runs`), and syncs live activity to `competition_state.json` for the Dashboard UI.
   - **Typed Action Contract (`act`) & Approval Governance (`ApprovalTier`)**:
     - `AUTO`: `inspect_data_readiness`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`
     - `AUTO_WITH_LIMIT`: `establish_baseline_anchor`, `run_full_validation` (`--backend local|modal|kaggle`), `synthesize_candidate`, `promote_candidate`, `build_ensemble`
     - `USER_APPROVAL` (Mandatory Human Sign-off): `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`
3. **Layer 3 — Competition State Machine & Execution Harness (`harness/competition_state.py`, `train.py`, `harness/cloud_runner.py`)**:
   - **8-Phase Competition Journey**: `1. Setup` $\rightarrow$ `2. Understand` $\rightarrow$ `3. Baseline Anchor` $\rightarrow$ **Parallel Tracks** (`4. Experiment Loop` $\parallel$ `5. Candidate Management` $\parallel$ `6. External Evaluation`) $\rightarrow$ `7. Freeze` $\rightarrow$ `8. Closure`.
   - **Deterministic Guardrails**: `GroundingGate` (`harness/grounding_gate.py`), `ExperimentValidator` (`harness/validator.py`), `EnsembleGuard` (`harness/ensemble_guard.py`), and `AuditGate` (`tools/audit_gate.py`).

---

## 2. Task & Evaluation

- **Subtask A (Stereotype)**: Binary classification (`yes` / `no`). Metric: **Macro-F1**.
- **Subtask B (Hate Speech)**: 3 classes (`yes_explicit`, `yes_implicit`, `no`). Metric: **Macro-F1**.
- **Subtask C (Target & Identity)**: Evaluated **only** on hateful comments (`hate_speech != "no"`):
  - Target Type: `individual` vs `group`. Metric: **Macro-F1**.
  - Target Identities: 9 binary codes (`l`, `g`, `b`, `t`, `q`, `i`, `a`, `lgbtqia+`, `nb`). Metrics: **Macro-F1**, **Micro-F1**.
- **Languages**: English (`EN`), Italian (`IT`), Dutch (`NL`). Metrics must be reported per language.

---

## 3. Hard Constraints

1. **Grouped Cross-Validation**: All models must use `splits/fold_assignments.tsv` (grouped by video `yt_title` to prevent data leakage). Never split randomly.
2. **Immutable Artifacts**: Never overwrite or delete existing folders inside `artifacts/`. Each run requires a unique experiment ID.
3. **Pristine Raw Data**: Files in `dataset/` are strictly read-only.
4. **Reproducibility**: Set fixed seed (`42`), record git commit hash, and save resolved config in `run_config.yaml`.
5. **Hypothesis-First & Grounded**: Every experiment must declare `parent_id`, `problem_addressed`, `hypothesis`, `intervention`, `grounding`, and `assumption_register`. Check the **Idea Graveyard** in `EXPERIMENTS.md` before proposing ideas.

---

## 4. Experiment Naming & Lineage

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

## 5. Two-Tier Memory System + Bridge Session Registry

- **Tier 1 (Global — `EXPERIMENTS.md` & `artifacts/journal.json`)**:
  - **Hall of Fame**: Current top-performing models per task and language.
  - **Idea Graveyard**: Documented failed approaches that must not be repeated.
  - **TreeJournal (`artifacts/journal.json`)**: AIDE-style solution tree with UCB branch scoring.
- **Tier 2 (Local — `artifacts/<exp_id>/EXPERIMENT_CARD.md` & `DIAGNOSTICS.md`)**:
  - Complete card per experiment with lineage, hypothesis, intervention, results, and ground-truth error slices.
- **Session Registry (`artifacts/agent_sessions/session_XXXX.json` & `artifacts/competition_state.json`)**:
  - Persistent record of Antigravity IDE sessions, actions executed, structured observations, and compute budgets.

---

## 6. Standard Operating Procedure (SOP) via Agent Bridge

1. **Inspect Context Packet**:
   ```powershell
   python tools/agent_bridge_cli.py context
   ```
2. **Start / Configure Session (Optional Budget Limits)**:
   ```powershell
   python tools/agent_bridge_cli.py start-session --goal "Improve Subtask C Macro-F1" --max-experiments 3
   ```
3. **Create Grounded Experiment Config**:
   ```powershell
   python tools/agent_bridge_cli.py act --action create_experiment --params '{"branch": "feat", "descriptor": "title_ctx_balanced"}'
   ```
4. **Smoke Test (Fold 0 Probe)**:
   ```powershell
   python tools/agent_bridge_cli.py act --action run_smoke_test --params '{"config_path": "configs/<exp_config>.yaml"}'
   ```
5. **Full 5-Fold Cross-Validation (`local` | `modal` | `kaggle`)**:
   ```powershell
   python tools/agent_bridge_cli.py act --action run_full_validation --params '{"config_path": "configs/<exp_config>.yaml", "backend": "local"}'
   ```
6. **Synthesize Candidate / Specialist Router**:
   ```powershell
   python tools/agent_bridge_cli.py act --action synthesize_candidate --params '{"name": "router_v1", "mode": "specialist_router"}'
   ```
7. **Audit & Submission (Requires `USER_APPROVAL`)**:
   ```powershell
   python tools/agent_bridge_cli.py act --action prepare_submission --user-approved --params '{"exp_id": "<id>", "candidate_csv": "<cand_csv>", "baseline_csv": "<base_csv>"}'
   ```

---

## 7. Command Reference

| Action | Command |
|---|---|
| Read Bridge Context Packet | `python tools/agent_bridge_cli.py context` |
| Execute Typed Bridge Action | `python tools/agent_bridge_cli.py act --action <name> --params '{...}'` |
| Approve Pending Action | `python tools/agent_bridge_cli.py approve --session-id <session_id>` |
| Tree Summary / UCB Recommend | `python tools/journal_tool.py --summary` / `python tools/journal_tool.py --recommend` |
| Direct Smoke Test | `python train.py --config configs/<exp_config>.yaml --debug --fold 0` |
| Direct Full 5-Fold CV | `python train.py --config configs/<exp_config>.yaml` |
| Audit Submission | `python tools/audit_gate.py --candidate <cand.csv> --baseline <base.csv> --output-zip submission.zip` |
| Unit Tests | `python -m unittest discover -s tests -p "test_*.py"` |
