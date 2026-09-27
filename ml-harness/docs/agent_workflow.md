# AI Assistant Workflow Guide

This guide describes how AI coding assistants (such as Antigravity IDE, Claude Code, or OpenAI Codex) interact with the ML-Harness framework.

---

## 1. Core Operating Principles

1. **Context-Driven**:
   The assistant uses the Agent Bridge (`tools/agent_bridge_cli.py`) to query task settings, the active leaderboard, and search tree recommendations without polluting the context window.

2. **Atomic Interventions**:
   Each experiment config modifies one variable at a time (e.g., feature extraction, loss formulation, or backbone model) so that metric deltas can be cleanly attributed.

3. **Fast Probes Before Full Runs**:
   All new pipelines are verified with a fast single-fold debug probe (`--debug --fold 0`) to catch syntax, data shape, and convergence issues before allocating multi-fold compute.

4. **Deterministic Guardrails**:
   Ensemble candidates and submission files are verified via programmatic audit gates before packaging.

---

## 2. Onboarding a New Competition

When starting a new competition or benchmark, the assistant needs **at least one** of two inputs:
- A dataset location (local archive, folder, or link).
- A task context (competition title, link, or description).

The complete prompt to initiate onboarding is documented in [`docs/prompts/onboarding.md`](prompts/onboarding.md).

---

## 3. Agent Bridge CLI Commands

The assistant interacts with the execution harness through `tools/agent_bridge_cli.py`:

### Viewing Context
```bash
# Retrieve current stage, best CV score, and UCB branch suggestions
python tools/agent_bridge_cli.py context
```

### Starting a Work Session
```bash
# Start a session with experiment and runtime budgets
python tools/agent_bridge_cli.py start-session --goal "Improve primary metric" --max-experiments 3
```

### Executing Actions
```bash
# Create an experiment configuration
python tools/agent_bridge_cli.py act --action create_experiment --params '{"branch": "feat", "descriptor": "target_encoding"}'

# Run a smoke test on fold 0
python tools/agent_bridge_cli.py act --action run_smoke_test --params '{"config_path": "configs/exp002_feat_target_encoding.yaml"}'

# Run full cross-validation
python tools/agent_bridge_cli.py act --action run_full_validation --params '{"config_path": "configs/exp002_feat_target_encoding.yaml"}'

# Promote model to candidate pool
python tools/agent_bridge_cli.py act --action promote_candidate --params '{"exp_id": "exp002", "name": "candidate_v1"}'
```

---

## 4. Solution Tree & Exploration (UCB)

The solution tree (`artifacts/journal.json`) tracks all iterations as a directed acyclic graph (DAG). The assistant queries the next recommended direction using Upper Confidence Bound search:

```bash
# Summary of all runs
python tools/journal_tool.py --summary

# Recommended next intervention branch
python tools/journal_tool.py --recommend
```
