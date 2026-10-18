# Change Log

Per-change documentation for **framework-repo** changes (code + docs). Each
change/PR gets one file here, written from real context — **never** reconstructed
from a PR description alone (that risks fabricating the "thought" and "approach").
Newer at the top.

## How it works

One file per framework-repo change/PR, named after the change (short kebab-case
slug). Copy [`_TEMPLATE.md`](_TEMPLATE.md), fill it in, and add a row to the
index below — before committing. Newer at the top.

## Changes

| Date | Change | PR | File |
|------|--------|----|------|
| 2026-10-18 | Live adoption verification + Submission field reference (§7) | [#89](https://github.com/taitran501/ai-harness/pull/89) | [live-adoption-verified](live-adoption-verified.md) |
| 2026-10-18 | Anti-hallucination API reference (SKILL.md §7) + dispatch regression tests | [#88](https://github.com/taitran501/ai-harness/pull/88) | [anti-hallucination-api-reference](anti-hallucination-api-reference.md) |
| 2026-10-17 | Harness-native Kaggle submission + kernel dispatch | [#84](https://github.com/taitran501/ai-harness/pull/84) | [kaggle-dispatch-polling](kaggle-dispatch-polling.md) |
| 2026-10-17 | Skill source fix: pipeline section in generated SKILL.md | [#83](https://github.com/taitran501/ai-harness/pull/83) | [skill-source-fix](skill-source-fix.md) |
| 2026-10-17 | Pipeline adoption stack (audit gate, config hardening, discoverability, scaffolds) | [#82](https://github.com/taitran501/ai-harness/pull/82) | [pipeline-adoption-stack](pipeline-adoption-stack.md) |
| 2026-10-17 | Multi-stage pipeline execution (train/blend/stack) | [#81](https://github.com/taitran501/ai-harness/pull/81) | [multi-stage-pipeline-execution](multi-stage-pipeline-execution.md) |
| 2026-10-17 | Single-install architecture (one canonical checkout) | [#78](https://github.com/taitran501/ai-harness/pull/78) | [single-install-architecture](single-install-architecture.md) |
| 2026-10-17 | Adoption stack T3–T5 (session gate, host hooks, adoption health) | [#61](https://github.com/taitran501/ai-harness/pull/61) | [adoption-stack-t3-t5](adoption-stack-t3-t5.md) |
| 2026-10-17 | Agent-bypass audit: status truthfulness, AGENTS.md, bypass detection | [#60](https://github.com/taitran501/ai-harness/pull/60) | [agent-bypass-audit](agent-bypass-audit.md) |
| 2026-10-17 | Ruff zero-noise gate + B10–B16 verified fixes | [#56](https://github.com/taitran501/ai-harness/pull/56) | [ruff-zero-noise-gate](ruff-zero-noise-gate.md) |
| 2026-10-16 | AIMO first live Kaggle dispatch + remote-dispatch hardening | [#64](https://github.com/taitran501/ai-harness/pull/64) · [#65](https://github.com/taitran501/ai-harness/pull/65) · [#66](https://github.com/taitran501/ai-harness/pull/66) | [aimo-live-kaggle-dispatch](aimo-live-kaggle-dispatch.md) |
| 2026-10-15 | Modality & model expansion (Agent Usability Phases 5–8) | [#33](https://github.com/taitran501/ai-harness/pull/33) · [#34](https://github.com/taitran501/ai-harness/pull/34) | [modality-model-expansion](modality-model-expansion.md) |
| 2026-10-13 | Canonical AI-Harness rebranding + decoupled fast testing | rebrand batch | [rebranding-fast-testing](rebranding-fast-testing.md) |
| 2026-10-10 | Research continuity: durable insight traces + human-readable tracking | research-memory series | [research-continuity](research-continuity.md) |

## Older changes (archived)

Changes prior to this change log are recorded in **[`docs/WORK_LOG.md`](../WORK_LOG.md)**
(now a read-only archive). Those entries pre-date the per-file change-doc convention
and are kept as-is; they are not individually split out (reconstructing them from
thin descriptions would risk inaccuracy).
