# Research memory identity tolerance, action aliases, parameter ergonomics, and agent shims

- **Date**: 2026-10-25
- **PR**: [#131](https://github.com/taitran501/ai-harness/pull/131)
- **Branch**: `feat/research-memory-identity-tolerance-and-param-aliases`
- **Status**: merged
- **Commits**: `a496e90` (squash)

## Why

During real-world autonomous competition runs on 2026-10-24/25 across multiple attached competition workspaces (`airline_prediction` and `arc-prize-2026-arc-agi-3`), agents repeatedly encountered blocking friction and runtime exceptions during research loop interactions:

1. **`scorer_sha256` identity mismatch**:
   `contract_identity` computed `scorer_sha256` by hashing the inspect source of `harness.evaluation` and `harness.task_spec`. Whenever documentation, type hints, or export helpers in these modules were updated, the hash changed. Consequently, legitimate historical baseline artifacts (e.g. `exp000_baseline` in `airline_prediction`, `aimo`, and `rsna-knee`) were rejected with `Run identity mismatch: scorer_sha256` during `establish_baseline_anchor` and silently dropped during `ResearchMemory.sync()`. This caused subsequent actions referencing baseline evidence to fail with `Evidence references must identify verified full runs for the current task`.
2. **`load_active_task_spec` directory path bypass**:
   When passed a project or directory path (e.g. `load_active_task_spec("."` or `load_active_task_spec(project_root)`), `load_active_task_spec` evaluated `if p.is_file()` as `False` and silently fell back to an empty dummy `UniversalTaskSpec(metadata={})`, producing an invalid `task_sha256` digest (`44136fa3...`).
3. **Action and parameter name fragility**:
   - Action guessing: Agents naturally guessed action names like `checkpoint`, `save_checkpoint`, `checkpoint_research_progress`, `record_insight`, `record_decision`, `plan_probe`, `search_memory`, `establish_baseline`, and `get_next_exp_id`, all of which were rejected by the CLI and bridge as `Unknown action`.
   - Parameter KeyErrors in `record_research_insight`: Agents commonly provided `--hypothesis`, `--name`, `--summary`, `--rationale`, or `--plan` instead of exact keys (`title`, `observation`, `mechanism`, `next_action`), crashing with `KeyError: 'title'` or `KeyError: 'observation'`.
   - Parameter KeyErrors and validation errors in `plan_research_probe`: `_handle_research_memory_group` used a strict dict comprehension requiring 7 mandatory keys; missing keys crashed with `KeyError: 'decision_rule'` or `KeyError: 'control'`.
   - Outcome and conclusion errors in `record_research_decision`: Agents passed natural synonyms like `--decision verified_and_ready`, `--outcome accepted`, `--lesson "..."`, or descriptive prose proof for `--evidence` (e.g. Kaggle submission references), crashing with `KeyError: 'outcome'`, `KeyError: 'conclusion'`, `ValueError: outcome must be supported...`, or `Evidence references must identify verified full runs...`.
4. **Missing `harness.agent` shim package**:
   Agents guessing Python imports like `from harness.agent.actions import ...`, `from harness.agent.cli import ...`, or `from harness.agent import AgentBridge` failed with `ModuleNotFoundError: No module named 'harness.agent'`.

## What changed

- **`harness/artifacts.py`**:
  - Introduced `CANONICAL_GENERIC_SCORER_HASHES` containing historical canonical framework generic scorer hashes (`aca79278...`, `8a766ac8...`, `7b33d61f...`, `7d2f454d...`, `d44339a1...`, `13c0aef1...`, `4768719e...`, `bcc7ab19...`).
  - In `validate_run`: when validating `scorer_sha256` for built-in/generic evaluations (`custom_scorer_sha256` is `None` or absent), accepted matches if the manifest scorer hash matches current hash OR is a recognized canonical generic scorer hash.
  - Made `task_spec` acceptable as a 2nd positional argument in `validate_run(directory, task_spec=None, *, plugin=None, expected=None)`.
- **`harness/task_spec.py`**:
  - In `load_active_task_spec`: added directory path resolution checking `p / "configs" / "task.yaml"` and `HarnessProject.discover(p)` before falling back to default spec. Added direct passthrough for `UniversalTaskSpec` instances and `dict` configurations.
- **`harness/bridge_common.py`**:
  - Defined `ACTION_ALIASES` mapping common agent action guesses (`checkpoint`, `save_checkpoint`, `checkpoint_research_progress`, `record_insight`, `add_insight`, `record_decision`, `plan_probe`, `search_memory`, `establish_baseline`, `get_next_exp_id`, `smoke_test`, `full_validation`) to canonical typed actions.
  - Exported `resolve_action_name(name: str) -> str`.
  - Updated `_TYPED_ACTION_NAMES` to include both canonical action names and all action aliases.
  - Updated `get_action_schema` to resolve action aliases automatically.
- **`harness/research_memory.py`**:
  - Added `DECISION_OUTCOME_ALIASES` and `normalize_decision_outcome(val)` supporting synonyms like `accepted`, `keep`, `verified_and_ready`, `passed`, `failed`, `discard`, `postponed`, `uncertain`.
  - Integrated `normalize_decision_outcome` into `ResearchMemory.record_decision`.
- **`harness/bridge_actions.py`**:
  - `_handle_research_memory_group`:
    - `record_research_insight`: Defensive extraction supporting parameter aliases (`hypothesis`, `name`, `summary`, `description`, `content`, `rationale`, `plan`, `todo`). Added graceful handling of descriptive prose evidence.
    - `plan_research_probe`: Defensive extraction with defaults for `control`, `intervention`, `expected_observation`, `alternative_explanation`, `decision_rule`, `budget`, `stop_rule`, and automatic retrieval of latest `insight_id` if omitted.
    - `record_research_decision`: Defensive extraction with outcome alias normalization (`_normalize_decision_outcome`), conclusion/lesson alias fallback, and preservation of descriptive prose evidence as narrative notes instead of rejecting the call.
- **`tools/agent_bridge_cli.py` & `harness/agent_bridge.py`**:
  - Applied `resolve_action_name` before action validation and dispatch.
  - Supported action aliases in `AgentBridge.__getattr__`.
- **`harness/agent/`**:
  - Created package forwarding shims:
    - `harness/agent/__init__.py`: re-exporting `AgentBridge`, `BridgeSession`, `_TYPED_ACTION_NAMES`, etc.
    - `harness/agent/actions.py`: forwarding action mixins, registries, schemas, and typed action names.
    - `harness/agent/cli.py`: forwarding `tools.agent_bridge_cli.main`.
- **`tests/test_agent_session_resilience.py`**:
  - Added unit test suite verifying canonical scorer hash tolerance, directory spec resolution, action alias resolution, outcome normalization, parameter tolerance in bridge actions, and `harness.agent` shim imports.

## Approach

- **Correctness Contract preservation**: Custom scorer hashes (`custom_scorer_sha256`) and manifest artifact file SHA-256 digests remain strictly validated and immutable. Generic built-in scorer tolerance is restricted to known, released canonical generic framework hashes where metric implementations (AUC, RMSE, F1, etc.) are standard.
- **Robustness Principle**: The agent bridge is an ergonomic runtime communication boundary. It accepts semantically valid human/agent variants (`accepted` -> `supported`, `hypothesis` -> `observation`, `checkpoint` -> `record_research_checkpoint`) while preserving strict auditability and schema v2 integrity in the underlying storage.

## Evidence

- `tests/test_agent_session_resilience.py`: 6/6 tests pass.
- Verified on live competition workspace `airline_prediction`: `exp000_baseline` artifact successfully passes `validate_run` and `validate_baseline_artifact` (metric: `roc_auc=0.9214686`).
- Full test suites:
  - `python tools/run_tests.py --smoke`: 47/47 tests pass in 2.5s.
  - `python tools/run_tests.py --fast`: 1004/1004 tests pass.
  - `ruff check harness/ tools/ tests/`: 100% clean.

## Impact & follow-ups

Eliminates repetitive session crashes across all attached competition workspaces when agents interact with research memory, record checkpoints, or reference baseline evidence.
