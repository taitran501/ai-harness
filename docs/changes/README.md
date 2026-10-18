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
| 2026-10-18 | Test isolation (journal leak guard in conftest) + research sync cache integrity (per-file fingerprint + contract identity key) | [#95](https://github.com/taitran501/ai-harness/pull/95) | [test-isolation-and-sync-cache](test-isolation-and-sync-cache.md) |
| 2026-10-18 | Dispatch governance: append-only `dispatch_log.jsonl` + advisory duplicate warning for harness-native Kaggle dispatches | [#97](https://github.com/taitran501/ai-harness/pull/97) | [dispatch-governance](dispatch-governance.md) |
| 2026-10-18 | Repair integrity gate (evidence drift never re-sealed; `--verify`/`--force` + audit log) + recursive artifact sealing (nested `stages/*.log`) | [#96](https://github.com/taitran501/ai-harness/pull/96) | [repair-integrity-and-recursive-sealing](repair-integrity-and-recursive-sealing.md) |
| 2026-10-18 | Import sort (I001) + canonical ruff gate (`ruff.toml`, W605 in gate) | [#94](https://github.com/taitran501/ai-harness/pull/94) | [import-sort-and-ruff-gate](import-sort-and-ruff-gate.md) |
| 2026-10-18 | Test env isolation (conftest sweep) + `kernel_dispatch.main()` CLI tests | [#93](https://github.com/taitran501/ai-harness/pull/93) | [test-env-isolation-dispatch-cli](test-env-isolation-dispatch-cli.md) |
| 2026-10-18 | kagglesdk `list_kernels` routing + `competition_topics` method fix (both-SDK TypeError) | [#92](https://github.com/taitran501/ai-harness/pull/92) | [kagglesdk-list-routing-followup](kagglesdk-list-routing-followup.md) |
| 2026-10-18 | §7 field truthfulness + kagglesdk kernel methods fix (live-run frictions F1/F2) | [#91](https://github.com/taitran501/ai-harness/pull/91) | [api-surface-kagglesdk-fix](api-surface-kagglesdk-fix.md) |
| 2026-10-18 | Journal sync dedup: halve per-action run validation + fix orphan-resurrection bug | [#90](https://github.com/taitran501/ai-harness/pull/90) | [journal-sync-dedupe](journal-sync-dedupe.md) |
| 2026-10-18 | Dead-code trim: remove unreferenced vendored AIDE journal package | [#90](https://github.com/taitran501/ai-harness/pull/90) | [dead-code-trim](dead-code-trim.md) |
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
