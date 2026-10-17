# Pipeline Adoption Stack: Audit Gate, Config Hardening, Discoverability, Scaffolds

- **Date**: 2026-10-17
- **PR**: [#82](https://github.com/taitran501/ai-harness/pull/82)
- **Branch**: `feat/pipeline-adoption-stack`
- **Status**: merged
- **Commits**: `719a1e4`

## Why

`execution.mode: pipeline` (added in PR #81) made multi-stage train/blend/stack runs *possible*, but agents still bypassed it with hand-written orchestration scripts. Root cause: a feature is adopted only when it is **trustworthy, easy, discoverable, and useful** — pipeline execution satisfied "possible" but not the other three. This change adds the adoption stack so `mode: pipeline` becomes the preferred choice, consistent with the adoption-over-prevention strategy.

## What changed

- **Pipeline audit gate** (`harness/runner.py`): new `_verify_pipeline_manifest(expected_count)` verifies `stage_manifest.json` is complete and consistent (stage count, all exit codes 0, index order 0..N-1, every per-stage log present) before finalizing; raises `ValueError` on inconsistency. Called from `_run_pipeline` after the manifest save.
- **Config hardening** (`harness/task_validate.py` + `configs/pipeline_template.yaml`): `validate_task` now warns when `train_path` / `target_column` carry a placeholder token (`YOUR_*`, `CHANGE_ME`, `TODO`, etc.) — the "copied template, ran as-is" failure mode. `configs/pipeline_template.yaml` is a `[CUSTOMIZE]`-marked starting point (not runnable as-is; the grounding gate + placeholder warning force customization).
- **Discoverability** (`docs/agent_workflow.md` + generated SKILL.md): the pipeline section now carries the valid `approach × mode` combo table, template + scaffold references, and the audit-gate guarantee.
- **Scaffolds** (`templates/pipeline/`): `train_template.py` + `blend_template.py` — copy-paste starting points for pipeline stages (read the harness env vars + merged config, write `oof_predictions.csv`), plus `README.md` with the dispatch contract (`--exp-dir`/`--config`, `HARNESS_STAGE_*` env, last stage writes the verified file).

## Approach

Four pillars, each targeting one adoption dimension:

- **Audit gate → trust**: a per-stage manifest verified before finalizing means the agent can rely on `mode: pipeline` not silently skipping a stage.
- **Template + placeholder warning → ease**: a marked, non-runnable starting point plus a validation warning force customization, killing the "copied template, ran as-is" failure.
- **Docs + skill → discoverability**: the combo table and scaffold references live where agents look (agent_workflow + generated skill).
- **Scaffolds → usefulness**: ready-to-copy stage scripts remove the "write the orchestration by hand" cost.

Rejected alternative: a generic orchestrator that auto-derives stages from config. Chose explicit stage scripts + manifest instead — deterministic, auditable, and consistent with non-prescriptive autonomy (the agent authors the stages; the harness audits them).

## Evidence

- Ruff `F,E7` gate clean.
- Pipeline + task_validate tests: 26/26 pass.
- `configs/pipeline_template.yaml` parses as valid YAML (mode=pipeline, approach=custom_script, 3 stages) and the placeholder warning fires on a synthetic task spec.
- Fast suite: 750 tests in 176.5s — OK (skipped=1) (was 743; +7 = 6 pipeline + 1 task_validate).

## Impact & follow-ups

Makes `mode: pipeline` the recommended path over hand-written orchestration. Scaffolds live in the framework `templates/pipeline/` (shared across workspaces). No follow-up PRs required; monitor adoption via `ai-harness status` adoption signals.
