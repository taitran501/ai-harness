# Developer ergonomics, research memory resilience, Kaggle facade parity, and mount robustness

- **Date**: 2026-10-24
- **PR**: [#130](https://github.com/taitran501/ai-harness/pull/130)
- **Branch**: `feat/developer-ergonomics-research-memory-facade-resilience`
- **Status**: merged
- **Commits**: 1fa0197

## Why

Auditing historical session transcripts across active competition workspaces (`airline_prediction` `8d9fcfe9`, `arc-prize` `ce5361e6`, `01b600ee`, `101cef92`, `5ba70474`, `a9d21987`) revealed recurrent developer-ergonomics friction and runtime crashes where agents attempted standard idioms or intuitive imports:

1. **`ResearchMemory` instantiation and frontier access**:
   - Callers doing `ResearchMemory(".")` crashed with `TypeError: ResearchMemory.__init__() missing 2 required positional arguments: 'artifacts_dir' and 'task_spec'`.
   - Callers attempting `ResearchMemory.frontier` or `memory.frontier()` failed with `AttributeError: type object 'ResearchMemory' has no attribute 'frontier'`, because `context()` computed `"research_frontier"` but no dedicated property/method was exposed.
2. **`CompetitionStateMachine.load()` omission**:
   - Agents instinctively called `CompetitionStateMachine.load(project_root)` after seeing `.save(...)`, failing with `AttributeError: type object 'CompetitionStateMachine' has no attribute 'load'`.
3. **`contract_identity` dictionary fragility**:
   - In multiple sessions, passing a raw dictionary task spec resulted in `AttributeError: 'dict' object has no attribute 'metadata'`.
4. **Intuitive cross-module symbol imports**:
   - `from harness.evaluation import validate_run` crashed with `ImportError`.
   - `from harness.artifacts import repair_run` and `from harness.repair import repair_folder` failed with `AttributeError`.
   - `from harness.kaggle_compat import KaggleLeaderboardTracker` failed with `AttributeError`.
   - `from harness.task_spec import load_task_spec` failed with `ImportError`.
   - `from harness.agent_bridge import HANDLERS, _ACTION_REGISTRY` failed with `ImportError`.
5. **AgentBridge direct action invocation**:
   - Agents probing or calling `bridge._action_record_research_insight(...)` crashed with `AttributeError: 'AgentBridge' object has no attribute '_action_record_research_insight'`.
6. **KaggleClient facade omissions**:
   - `client.leaderboard` was missing (only `get_leaderboard` was present).
   - `client.api` and `client._api` were missing, preventing access to the underlying authenticated `KaggleApi`.
   - Discussion aliases (`topics`, `list_topics`, `discussions`, `list_discussions`, `topic_messages`) were missing.
   - `client.kernels_pull` and `client.get_kernel` were absent, forcing agents into manual low-level SDK calls with raw request constructors that broke.
7. **Kaggle remote dataset mount discovery failure (`find_dataset_mount`)**:
   - In `ce5361e6`, `find_dataset_mount` discarded valid mount paths because `if owned or any(...): candidates = owned` set `candidates = []` when `owned` was empty.
   - Case-sensitivity on Linux caused exact folder matching to fail when Kaggle mounted datasets with different casing.
8. **Outdated API reference in generated skills**:
   - `SKILL.md §7` lacked native subcommands (`competition`, `leaderboard`, `discussion`, `codabench`) and recent facade methods.

## What changed

- **`harness/kaggle_runtime.py`**:
  - Enhanced `find_dataset_mount` with case-insensitive direct match, case-insensitive recursive fallback, and safe owner filtering (`if owned: candidates = owned`).
- **`harness/research_memory.py`**:
  - Made `artifacts_dir` and `task_spec` optional in `ResearchMemory.__init__`, auto-discovering them from `project_root` via `load_active_task_spec` and `.ai-harness/artifacts`.
  - Added `ResearchMemory.load(project_root=".")` classmethod.
  - Implemented `_FrontierDescriptor` providing seamless property and callable access on both instance (`memory.frontier`, `memory.frontier()`, `memory.research_frontier`) and class (`ResearchMemory.frontier()`).
- **`harness/competition_state.py`**:
  - Added `CompetitionStateMachine.load(project_root=".", task_config_path=None)` classmethod.
- **`harness/artifacts.py`**:
  - Updated `contract_identity` to gracefully extract metadata from dictionary task specs (`task_spec.get('metadata')`).
  - Re-exported `repair_run_dir`, `run_repair`, and alias `repair_run = repair_run_dir`.
- **`harness/repair.py`**:
  - Added aliases `repair_run = repair_run_dir` and `repair_folder = repair_run_dir`.
- **`harness/evaluation.py`**:
  - Re-exported `validate_run` from `harness.artifacts`.
- **`harness/task_spec.py`**:
  - Exported alias `load_task_spec = load_active_task_spec`.
- **`harness/agent_bridge.py`**:
  - Exported `HANDLERS = typed_actions` and `_ACTION_REGISTRY = typed_actions`.
  - Added `_ACTION_REGISTRY` on `AgentBridge` class.
  - Updated `AgentBridge.__getattr__` to route `_action_<name>` dynamically to `execute_action`.
- **`harness/kaggle_compat.py`**:
  - Re-exported `KaggleLeaderboardTracker`.
  - Added `@property def api(self)` and `@property def _api(self)` returning `self._legacy_api_instance()`.
  - Added `leaderboard = get_leaderboard`.
  - Added discussion aliases: `list_topics`, `topics`, `discussions`, `list_discussions`, `topic_messages`, `list_topic_messages`.
  - Added `kernels_pull(kernel, path=None, metadata=False, quiet=True)` and aliases `kernel_pull`, `pull_kernel`.
  - Added `get_kernel(kernel)`.
- **`harness/api_reference.py`**:
  - Added `competition`, `leaderboard`, `discussion`, and `codabench` to `CLI_COMMANDS`.
  - Added `get_competition`, `list_competitions`, `get_leaderboard`, `leaderboard`, `competition_topics`, `competition_topic_messages`, `get_topic`, `kernels_pull` to `FACADE_METHODS`.
  - Added `ResearchMemory.load`, `CompetitionStateMachine.load`, and `repair_run_dir` to `PYTHON_SYMBOLS`.
- **`tests/test_ergonomics_and_resilience.py`**:
  - Added 9 unit tests covering all ergonomics and resilience improvements.

## Approach

- Maintained 100% backward compatibility: existing signatures and call paths remain untouched while adding ergonomic aliases and defensive defaults.
- Used descriptor semantics (`_FrontierDescriptor`) to support both property access (`rm.frontier`) and callable invocation (`rm.frontier()`, `RM.frontier()`) without conflicting.
- Kept the verified API reference (`harness/api_reference.py`) strictly verified via automated unit test assertions in `tests/test_api_reference.py`.

## Evidence

- `tests/test_ergonomics_and_resilience.py`: 9/9 unit tests pass.
- `tests/test_api_reference.py`: 14/14 unit tests pass.
- `ruff check harness/ tools/ tests/`: 0 errors.
- `python tools/run_tests.py --smoke`: 47/47 tests pass in 3.0s.
- `python tools/run_tests.py --fast`: 102 test modules pass.

## Impact & follow-ups

None. Workspaces synced via `ai-harness sync`.
