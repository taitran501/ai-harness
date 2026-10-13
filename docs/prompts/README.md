# AI Assistant Prompt Library

This directory contains system prompts and execution contracts for AI coding assistants (such as Google Antigravity, Anthropic Claude Code, and OpenAI Codex). Each prompt guides the assistant through specific phases of the **8-Phase Competition Journey**.

---

## 1. Prompt Taxonomy and Phase Alignment

| Lifecycle Phases | Objective | Document | Primary Actions |
|---|---|---|---|
| **Phases 1–3** | **Onboarding, EDA, and Baseline Anchor** | [`onboarding.md`](onboarding.md) | `bootstrap_task`, `establish_baseline_anchor` |
| **Phase 4** | **AIDE UCB Experimentation Loop** | [`experiment_iteration.md`](experiment_iteration.md) | `create_experiment`, `run_smoke_test`, `run_full_validation` |
| **Phases 5–6** | **Candidate Selection and Ensembling** | [`candidate_ensemble.md`](candidate_ensemble.md) | `promote_candidate`, `build_ensemble` |
| **Phases 7–8** | **Pre-Submission Audit and Freeze** | [`submission_audit.md`](submission_audit.md) | `freeze_candidate`, `prepare_submission`, `audit` |

---

## 2. Core Assistant Rules Across All Prompts

1. **Context Economy**:
   Never request or output full raw datasets, multi-thousand-line tables, or complete execution logs. Always use `ai-harness-agent context` to receive a compressed Heads-Up Display (~1,000 tokens).

2. **Grounded Hypotheses**:
   Every experiment proposal must formulate an explicit hypothesis referencing failure slice diagnostics (`.ai-harness/artifacts/<task-id>/<exp_id>/DIAGNOSTICS.md`) from the parent experiment.

3. **Atomic Interventions**:
   Vary exactly **one variable** between parent and child runs (such as a feature pipeline, a loss function, or a backbone model). Multi-variable shotgun iterations are strictly prohibited.

4. **Action Approval Governance**:
   - `AUTO`: Safe inspection, configuration creation, and single-fold smoke tests.
   - `AUTO_WITH_LIMIT`: Session-budgeted full cross-validation and ensemble builds.
   - `USER_APPROVAL`: High-stakes irreversible actions requiring explicit user consent (`freeze_candidate`, `prepare_submission`, `emergency_unfreeze`).

5. **Verification Scope**:
   AI-Harness is strictly verified on Natural Language Processing (NLP / Text) benchmarks. Non-NLP modalities are experimental work-in-progress on the roadmap.
