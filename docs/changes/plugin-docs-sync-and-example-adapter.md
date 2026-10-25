# Sync TaskPlugin docs with code and add a reference adapter

- **Date**: 2026-10-25
- **PR**: — (local)
- **Branch**: `docs/plugin-docs-sync`
- **Status**: open
- **Commits**: —

## Why

`docs/plugin_architecture.md` listed hooks (`load_train_data`, `load_test_data`, `run_eda`) that do not exist on `harness.plugins.TaskPlugin` (real names: `load_training_data`, `extend_eda_report`; no test-data hook). An agent following the docs would write methods the runtime never calls — the adapter imports fine but silently lacks the intended behavior. There was also no runnable adapter example.

## What changed

- `docs/plugin_architecture.md`: declared `TaskPlugin` canonical; rewrote the hook table with real names and signatures; documented that `load_test_data`/`run_eda` do not exist; linked the example.
- `examples/task_plugin/example_adapter.py`: new reference adapter (data loader, splits, custom metric, submission builder/validator).
- `tests/test_plugin_docs_and_example.py`: asserts every documented hook exists on `TaskPlugin`, stale names are absent, and the example loads, splits, scores and builds/validates a submission.

## Approach

Docs were fixed to match code, not the reverse: no compatibility hooks were added since no use case needs them. No runtime behavior changed.

## Evidence

New test module and ruff results: see PR description / test run.

## Impact & follow-ups

None for runtime. Follow-up: plugin contract validation (signature checks) as a separate PR.
