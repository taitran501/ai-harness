# AGENTS.md — Assistant Integration & Workflow Guide

This document outlines the workflow, conventions, and CLI commands for AI coding assistants (such as Antigravity, OpenAI Codex, or Claude Code) working within ML-Harness.

---

## 1. System Structure

The system separates planning, coordination, and execution into three clean layers:

```text
Assistant / Engineer (Layer 1)
   │ Scientific reasoning, hypothesis formulation, diagnostic review
   ▼
Agent Bridge (Layer 2 - tools/agent_bridge_cli.py)
   │ Context synthesis, session budgets, typed actions, human approval
   ▼
Harness Runtime (Layer 3 - harness/)
   │ Model execution, cross-validation, solution tree tracking, guardrails
```

1. **Layer 1 — Assistant / Engineer**:
   - Analyzes validation metrics, failure slices, and dataset distributions.
   - Formulates atomic hypotheses (changing one variable or intervention at a time).
2. **Layer 2 — Agent Bridge (`harness/agent_bridge.py` & `tools/agent_bridge_cli.py`)**:
   - **Context Packet (`context`)**: Returns a concise JSON/Markdown packet containing current task settings, top leaderboard models, UCB branch recommendations, and recent bottlenecks.
   - **Session & Budgets (`start-session`)**: Tracks experiment counts, GPU runtime, and consecutive non-improving runs against optional user limits.
   - **Action Governance (`act`)**:
     - *Automatic*: `inspect_data_readiness`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`.
     - *Resource-managed*: `establish_baseline_anchor`, `run_full_validation`, `promote_candidate`, `build_ensemble`.
     - *User approval required*: `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`.
3. **Layer 3 — Execution Harness**:
   - **Runner (`harness/runner.py`)**: Executes models across supported paradigms (Classical ML, Transformers, QLoRA, Reasoning LLMs) and validation schemes (holdout, k-fold, debug probe).
   - **Solution Tree (`harness/tree_journal.py`)**: Tracks experiment lineage, scores, and parent-child relations.
   - **Guardrails**: Validates data splits (`harness/validator.py`), checks ensemble diversity (`harness/ensemble_guard.py`), and verifies submissions (`tools/audit_gate.py`).

---

## 2. Operating Principles

1. **Grounded Configurations**:
   - Construct experiment configs using the schema in `configs/exp000_template.yaml`.
   - Specify the parent experiment ID, problem addressed, atomic hypothesis, and verification steps.
2. **Atomic Interventions**:
   - Change one variable at a time (e.g. feature representation, loss function, or model family) so performance deltas can be cleanly attributed.
3. **Fast Probes Before Full Runs**:
   - Run a single fold or debug probe (`--debug --fold 0`) first to verify code syntax and loss convergence before committing full multi-fold or cloud GPU resources.
4. **Reproducibility**:
   - Use the seed declared by the task config. Never overwrite completed experiment artifacts under the attached project's configured artifact directory.

---

## 3. Task Specifications & Competition Presets

The harness is task-agnostic. Competition-specific schemas and metrics are defined declaratively in `configs/task.yaml`:

- **Task Definition (`configs/task.yaml`)**:
  - `name`: Benchmark or competition identifier.
  - `task_type`: `classification`, `regression`, `multi_label`, `ranking`.
  - `primary_metric`: Primary optimization metric (`macro_f1`, `roc_auc`, `rmse`, `logloss`, etc.).
  - `direction`: `maximize` or `minimize`.
  - `subtasks`: Optional array of subtasks with weights, metric formulas, and conditional filtering.
- **Built-in Presets (`configs/tasks/`)**:
  - `configs/tasks/kaggle_house_prices.yaml`
  - `configs/tasks/kaggle_titanic.yaml`
  - Competition-specific presets are distributed by their task plugins.

---

## 4. Standard Workflow via Agent Bridge

```powershell
# 0. Attach the installed harness to this competition workspace
ml-harness attach --project-root . --task-config configs/task.yaml --plugin <installed-plugin-id>

# 1. Inspect current status and recommendations
ml-harness-agent --project-root . context

# 2. Start a work session with a specific goal
ml-harness-agent --project-root . start-session --goal "Improve primary validation metric" --max-experiments 3

# 3. Create a verified experiment configuration
ml-harness-agent --project-root . act --action create_experiment --params '{"branch": "feat", "descriptor": "feature_pipeline_v1"}'

# 4. Run a fast smoke test (Fold 0 or debug probe)
ml-harness-agent --project-root . act --action run_smoke_test --params '{"config_path": "configs/<exp_config>.yaml"}'

# 5. Run full validation (local, modal, or kaggle)
ml-harness-agent --project-root . act --action run_full_validation --params '{"config_path": "configs/<exp_config>.yaml", "backend": "local"}'

# 6. Promote successful experiment to the candidate pool
ml-harness-agent --project-root . act --action promote_candidate --params '{"exp_id": "<exp_id>", "name": "candidate_v1"}'

# 7. Package and audit a submission (requires human approval)
ml-harness-agent --project-root . act --action prepare_submission --user-approved --params '{"exp_id": "<id>", "candidate_csv": "<test_predictions.csv>"}'
```

---

## 5. CLI Command Reference

| Purpose | Command | Notes |
|---|---|---|
| Autonomous Task Onboarding | `ml-harness-bootstrap --project-root . --dataset dataset --context docs/TASK.md` | Auto-extracts data, builds `task.yaml`, generates splits & updates skill |
| View Assistant Context | `ml-harness-agent --project-root . context` | Read-only status overview |
| Execute Typed Action | `ml-harness-agent --project-root . act --action <name> --params '{...}'` | Runs supported harness actions |
| Dashboard | `ml-harness-dashboard --project-root .` | Opens the installed dashboard for this attached task |
| Solution Tree & UCB | `ml-harness-agent --project-root . context` | Includes the current tree and branch recommendation |
| Fast Smoke Test | `ml-harness-run --project-root . --config configs/<config>.yaml --debug --fold 0` | Fast convergence check |
| Full Validation | `ml-harness-run --project-root . --config configs/<config>.yaml` | Runs all configured folds |
| Pre-Submission Audit | `ml-harness-agent --project-root . act --action prepare_submission --user-approved --params '{...}'` | Requires task plugin submission hooks and test-set predictions |
| Run Test Suite | `python -m unittest discover -s tests -p "test_*.py"` | Runs full test suite |


## Correctness contract

Read `docs/CORRECTNESS_CONTRACT.md`. Completed full runs are immutable, including with `--force`. Debug/selected-fold probes never qualify as baseline evidence. Do not promote legacy artifacts, infer missing source IDs, or substitute OOF for test predictions. Always use the task seed and selected-variant full OOF aggregate. Calibration and threshold optimizers are utilities, not built-in estimator names. GPU/remote hard budgets fail closed until allocation and termination are verifiable.
