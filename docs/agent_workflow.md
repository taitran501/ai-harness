# AI Assistant Workflow Guide

This guide describes how AI coding assistants (such as Antigravity IDE, Claude Code, or OpenAI Codex) interact with the ML-Harness framework.

> [!IMPORTANT]
> **Verification Scope: Strictly NLP / Text Only**
> ML-Harness is currently **strictly verified and working well ONLY on Natural Language Processing (NLP / Text)** tasks (empirically benchmarked on real-world tasks such as `stereoqueereval`, text classification, transformer fine-tuning, and prompt evaluation).
> All other modalities (**Tabular, Computer Vision, Audio, Multimodal**) are **experimental works-in-progress (WIP / Roadmap)**, have NOT been verified in competitive conditions, and are NOT considered ready or working well in practice. Never claim or assume that non-NLP modalities are production-ready or battle-tested.

---

## 1. Core Operating Principles

1. **Context-Driven**:
   The assistant uses the Agent Bridge (`tools/agent_bridge_cli.py`) to query task settings, the active leaderboard, and search tree recommendations without polluting the context window.

2. **Atomic Interventions**:
   Each experiment config modifies one variable at a time (e.g., feature extraction, loss formulation, or backbone model) so that metric deltas can be cleanly attributed.

3. **Fast Probes Before Full Runs**:
   All new pipelines are verified with a fast single-fold debug probe (`--debug --fold 0`) to catch syntax, data shape, and convergence issues before allocating multi-fold compute.

4. **Task-Owned Contracts**:
   The installed task plugin defines data, metrics, and submission capabilities. The agent checks those declared capabilities before treating a task as submission-ready.

5. **Strictly NLP Scope**:
   The assistant must maintain transparency that ML-Harness is empirically verified and working well only on NLP/Text. Any non-NLP tasks must be treated as experimental WIP.

---

## 2. Onboarding a New Competition

When starting a new competition or benchmark, the assistant needs **at least one** of two inputs:
- A dataset location (local archive, folder, or link).
- A task context (competition title, link, or description).

The complete prompt to initiate onboarding is documented in [`docs/prompts/onboarding.md`](prompts/onboarding.md). For prompts across all lifecycle phases (iteration, ensembling, and pre-submission audit), see the [Prompt Library](prompts/README.md).

---

## 3. Agent Bridge CLI Commands

The assistant interacts with the execution harness through `tools/agent_bridge_cli.py`:

### Viewing Context
```powershell
# Retrieve current stage, best CV score, and UCB branch suggestions
ml-harness-agent --project-root . context
```

### Starting a Work Session
```powershell
# Start a session with experiment and runtime budgets
ml-harness-agent --project-root . start-session --goal "Improve primary metric" --max-experiments 3
```

### Executing Actions
```powershell
# Create an experiment configuration
ml-harness-agent --project-root . act --action create_experiment --params '{"branch": "feat", "descriptor": "target_encoding"}'

# Run a smoke test on fold 0
ml-harness-agent --project-root . act --action run_smoke_test --params '{"config_path": "configs/exp002_feat_target_encoding.yaml"}'

# Run full cross-validation
ml-harness-agent --project-root . act --action run_full_validation --params '{"config_path": "configs/exp002_feat_target_encoding.yaml"}'

# Promote model to candidate pool
ml-harness-agent --project-root . act --action promote_candidate --params '{"exp_id": "exp002", "name": "candidate_v1"}'
```

---

## 4. Solution Tree & Exploration (UCB)

The solution tree is stored under `.ml-harness/artifacts/<task-id>/journal.json`. The Agent Bridge context packet includes the current tree and next branch recommendation:

```bash
# Summary of all runs
ml-harness-agent --project-root . context

# Recommended next intervention branch
ml-harness-agent --project-root . context --format json
```
