# AI-Harness AI Assistant Prompt Library

> [!IMPORTANT]
> **Verification Scope: Strictly NLP / Text Only**
> AI-Harness is currently **strictly verified and working well ONLY on Natural Language Processing (NLP / Text)** tasks (e.g. `stereoqueereval`, text classification, transformer fine-tuning, prompt evaluation).
> Support for all other modalities (**Tabular, Computer Vision, Audio, Multimodal**) is **experimental work-in-progress (WIP / Roadmap)**, has NOT been verified in competitive conditions, and is NOT considered ready or working well in practice.
>
> This directory contains system prompts and execution contracts for AI coding assistants (Google Antigravity, Anthropic Claude Code, OpenAI Codex).
> Each prompt corresponds to a specific phase in the **8-Phase AI-Harness Journey**.

---

## 1. Prompt Taxonomy & Phase Alignment

| Phase | Target Objective | Prompt Document | Primary CLI Actions |
|---|---|---|---|
| **Phase 1–3** | **Onboarding & Baseline Anchor** | [`onboarding.md`](onboarding.md) | `bootstrap_task`, `establish_baseline_anchor` |
| **Phase 4** | **AIDE UCB Experiment Loop** | [`experiment_iteration.md`](experiment_iteration.md) | `create_experiment`, `run_smoke_test`, `run_full_validation` |
| **Phase 5–6** | **Ensembling & Synthesis** | [`candidate_ensemble.md`](candidate_ensemble.md) | `promote_candidate`, `synthesize_candidate`, `build_ensemble` |
| **Phase 7–8** | **Audit Gate & Final Submission** | [`submission_audit.md`](submission_audit.md) | `freeze_candidate`, `prepare_submission`, `finalize_competition` |

---

## 2. Core Assistant Rules Across All Prompts

1. **Context Economy**: Never request or dump raw tabular datasets, multi-thousand-line logs, or full training predictions into prompt memory. Use `ai-harness-agent context` to receive compressed HUD telemetry (~1k tokens).
2. **Epistemic Grounding**: Every experiment proposal must formulate an explicit hypothesis referencing failure slice diagnostics (`.ai-harness/artifacts/<task-id>/<exp_id>/DIAGNOSTICS.md`) and verified domain literature.
3. **Atomic Interventions**: Exactly **one variable** must be varied between parent and child experiments (e.g. feature transformation OR loss function OR model architecture). Multi-variable shotgun iterations are prohibited.
4. **Approval Governance**:
   - `AUTO`: Safe exploratory actions (inspection, configuration creation, smoke probes).
   - `AUTO_WITH_LIMIT`: Session-budgeted compute actions (full validation, candidate synthesis).
   - `USER_APPROVAL`: High-stakes irreversible actions (`freeze_candidate`, `prepare_submission`, `emergency_unfreeze`).
5. **Strict Verification Scope**: AI-Harness is strictly verified only on NLP/Text tasks. Never claim or assume that Tabular, Vision, Audio, or Multimodal pipelines are verified or production-ready.
