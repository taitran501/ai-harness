# Native model asset CLI, bridge actions, and Context HUD (Phase 5)

- **Date**: 2026-11-03
- **PR**: Working tree
- **Branch**: `feat/skills-bridge-submission-and-custom-scorer`
- **Status**: open
- **Commits**: —

## Why

Phases 1–4 established strict model definitions, content-addressed storage,
policy evaluation, backend planning, and lineage. Task workspaces still lacked a
user-facing way to inspect those records, preflight a config, verify local
artifacts, or access the same operations through the typed Agent Bridge.

## What changed

- Added `ai-harness model list`, `inspect`, `register`, `verify`,
  `validate-config`, and `resolve` in `harness/model_assets/cli.py` and
  registered the group in `harness/cli.py`.
- Added typed model registry actions and schemas, routed them through the
  AgentBridge handler mixin, and documented the dynamic action flags in
  `tools/agent_bridge_cli.py`.
- Added a compact Context HUD summary backed by a small rebuildable model index
  and the selected run's model lock/receipt. The HUD does not open model
  definitions, manifests, or weights and does not make network calls.
- Added focused CLI, bridge, verification, backend dry-run, and HUD tests in
  `tests/test_model_cli_and_bridge.py`.

## Approach

The CLI and bridge share service functions in `harness.model_assets.cli` so
policy and storage behavior remain consistent. Registration validates the
existing 16 KiB strict schema and atomically stores only a declaration; its
response keeps artifact verification pending and eligibility unassessed.
`verify` explicitly resolves local resources and checks their manifest sizes
and SHA-256 values. `resolve` always creates a dry-run resource plan and never
dispatches to a provider. The HUD reads aggregate counts from `index.json` and
at most one selected experiment lock and receipt; stale indexes fall back to a
filename count and report every entry as unverified.

## Evidence

Pending requested verification. Results will be recorded here after the
targeted tests, full pytest suite, and Ruff complete.

## Impact & follow-ups

No provider discovery, download, upload, or dispatch was added. The model index
is a rebuildable summary; content-addressed definition objects remain the
source of truth. Weight verification is explicit and local-only.
