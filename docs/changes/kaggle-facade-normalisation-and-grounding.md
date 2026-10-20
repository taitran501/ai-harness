# Kaggle facade normalisation, transient 404 handling, grounding whitelist, and action parameter discovery

- **Date**: 2026-10-20
- **PR**: —
- **Branch**: `feat/agent-friction-remediation`
- **Status**: open
- **Commits**: —

## Why

A forensic audit of 45 Antigravity conversations across real competition workspaces (`rsna-knee`, `arc-prize-2026-arc-agi-3`, `aimo`, `airline_prediction`, `zindi`) revealed three persistent friction points where autonomous agents were blocked:

1. **Kaggle Facade Dictionary Access Crash**: In `rsna-knee` (conv `95c2aed5`, step 1337), querying kernel status via `c.kernels_status(...)` returned raw SDK objects (`ApiGetKernelSessionStatusResponse`) lacking `.get()` or `__getitem__`. Code attempting standard python access (`st.get('status')` or `st['status']`) crashed with `AttributeError`. Furthermore, querying status immediately after push raised transient HTTP 404 from Kaggle's backend before the session status was initialized.
2. **Kaggle Grounding Circuit Breaker**: In `rsna-knee` (conv `95c2aed5`, step 72), agents grounding hypotheses in real Kaggle evidence (`verified_via: kaggle_public_lb`, `kaggle_discussion`, `kaggle_notebook`) were blocked by `EpistemicGroundingError` because `GroundingGate` only whitelisted generic literature sources.
3. **Action Parameter Invisibility**: In `airline_prediction` (conv `101cef92`, steps 652–655, 1385–1388), agents attempting required research memory actions (`record_research_insight`, `plan_research_probe`, `establish_baseline_anchor`) ran `--action <name> --help` but received only generic top-level CLI flags, leading to blind guessing and invalid module imports (`from harness.agents.bridge import ...`).

## What changed

- **`harness/kaggle_compat.py`**:
  - Enhanced `_NormalisedObject` with dict-like methods: `.get(name, default=None)`, `__getitem__`, and `__contains__`.
  - Added support for dict raw payloads and normalized enum values (e.g. `status.name` stringification).
  - Extended `_LEGACY_ALIASES` with `failure_message` $\leftrightarrow$ `failureMessage` and added dynamic camelCase/snake_case bidirectional resolution in `_alias_candidates`.
  - Wrapped `kernels_status()` return values in `_NormalisedObject` and handled transient 404 responses by returning a queued status (`{"status": "QUEUED", "failureMessage": None}`) rather than crashing.
  - Added alias `kernels_list = list_kernels` to `KaggleClient`.
- **`harness/grounding_gate.py`**:
  - Expanded `VALID_SOURCE_TYPES` with `kaggle_public_lb`, `kaggle_discussion`, `kaggle_notebook`, and `kaggle_dataset`.
  - Expanded `VALID_VERIFIED_VIA` with `kaggle_public_lb`, `kaggle_api`, `kaggle_discussion`, and `kaggle_notebook`.
- **`harness/bridge_common.py` & `harness/agent_bridge.py`**:
  - Added `TYPED_ACTION_SCHEMAS` declaring exact required and optional parameter schemas for all 39 typed actions, exposed via `get_action_schema()`.
- **`tools/agent_bridge_cli.py`**:
  - Added `schema` subcommand to inspect action schemas in text or JSON format (`ai-harness-agent schema --action <name>`).
  - Added action-specific help interception: running `ai-harness-agent act --action <name> --help` outputs the action's parameter specification and sample command.
  - Validates action names against `_TYPED_ACTION_NAMES` with actionable guidance.
- **`.agents/skills/ai-harness/SKILL.md`**:
  - Documented primary typed actions and key parameters in §7.
  - Updated Kaggle facade reference with `kernels_list` alias and `kernels_status` dict-like access.
- **`harness/api_reference.py`**:
  - Added `kernels_list` and `kernels_output` aliases to `FACADE_METHODS`.
- **`tests/test_agent_friction_remediation.py`**:
  - 15 comprehensive unit tests verifying dict-like access, 404 handling, aliases, GroundingGate Kaggle sources, and CLI schema discovery.

## Approach

- Maintained complete backward compatibility: `_NormalisedObject` attribute access works as before while adding `Mapping`-like convenience (`.get()`, `[]`, `in`).
- Transient 404 handling in `kernels_status` prevents premature failure in polling loops without suppressing real non-404 API errors.
- Schema discovery is statically self-contained in `bridge_common.py` and usable without an active workspace context, giving agents instant zero-cost documentation.

## Evidence

- `tests/test_agent_friction_remediation.py`: 15/15 tests pass.
- `tests/test_api_reference.py`: 14/14 tests pass.
- `python tools/run_tests.py --smoke`: 47/47 tests pass in 2.4s.
- `ruff check harness/ tools/ tests/`: clean (zero warnings).

## Impact & follow-ups

None.
