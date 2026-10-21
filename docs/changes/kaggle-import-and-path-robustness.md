# Kaggle remote artifact import critical mismatch policy, safe path resolution, and task spec string compatibility

- **Date**: 2026-10-20
- **PR**: [#114](https://github.com/taitran501/ai-harness/pull/114)
- **Branch**: `feat/agent-friction-and-platform-integrations`
- **Status**: merged
- **Commits**: `966a561` (squash, on `main`)

## Why

Deep forensic analysis across >300 conversation transcripts in 8 competitive ML workspaces (`rsna-knee`, `airline_prediction`, `semeval_2027`, `aimo`, `zindi`, etc.) revealed 4 severe points of failure in live agent operations:
1. **Inverted Boolean in Kaggle Artifact Import**: When pulling output artifacts from Kaggle kernels in `airline_prediction`, imports repeatedly crashed with `RuntimeError: Kaggle output import failed: Downloaded Kaggle artifacts do not match this task/config/job: {'core_version': ...}` because `not _is_critical_identity_mismatch(k, v)` treated tolerated differences (`core_version`) as critical while letting actual critical mismatches (`task_sha256`) pass.
2. **Path relativity mismatch on Windows**: `Path.relative_to` in `competition_state.py` and `candidates.py` threw `ValueError: ... is not in the subpath ... OR one path is relative and the other is absolute` whenever `self.project_root` was relative (`Path(".")`) while artifact paths were `.resolve()`-ed absolute paths (observed in `semeval_2027`).
3. **Task Spec crash on string task name**: In `task_spec.py`, `cfg.get("task", cfg)` forced `task_cfg` into a string whenever an agent configured `task: airline_prediction` at root, causing subsequent `.get("inputs")` to crash with `AttributeError: 'str' object has no attribute 'get'`.
4. **Missing evaluation alias**: Agents attempting `from harness.evaluation import evaluate_predictions` encountered `ImportError`.
5. **Kaggle object dict parity**: Normalised proxy objects lacked `.keys()`, `.values()`, `.items()`.

## What changed

- **`harness/templates/kaggle_template.py`**:
  - Removed erroneous `not` in `critical_mismatches = {k: v for k, v in mismatches.items() if _is_critical_identity_mismatch(k, v)}`. Now accurately classifies `core_version` drift and missing remote `job_id` as non-critical, while strictly blocking real conflicts (`task_sha256`, `config_sha256`).
  - Added resolution for `data_config.kaggle_dataset_slug: "none"` to suppress redundant private dataset upload requirements when competition datasets are already mounted.
- **`harness/tree_journal.py` & `harness/state_store.py` & `harness/import_local.py`**:
  - Hardened `TreeJournal.load()` to parse list-structured files gracefully without throwing `AttributeError`.
  - In `state_store.py`, updated `transaction()` to gracefully handle legacy list-structured payloads by packaging them under `"events"` instead of crashing on revision updates.
  - In `import_local.py`, updated `_append_journal` to seamlessly record audit events into `journal.json` whether it is a list or dictionary, ensuring bidirectional compatibility.
- **`harness/io_utils.py`**:
  - Added `safe_relpath(path, start)` helper function that safely resolves both paths and handles Windows drive letters and relative/absolute mixtures.
- **`harness/competition_state.py` & `harness/candidates.py`**:
  - Replaced fragile `.relative_to(project_root)` calls with `safe_relpath`.
- **`harness/task_spec.py`**:
  - `UniversalTaskSpec.from_dict` now checks `isinstance(cfg.get("task"), dict)`. If `cfg["task"]` is a string, it falls back gracefully and uses the string as the default task `name`.
- **`harness/evaluation.py`**:
  - Added module alias `evaluate_predictions = evaluate`.
- **`harness/kaggle_compat.py`**:
  - Added `.keys()`, `.values()`, and `.items()` methods to `_NormalisedObject`.
- **`tests/test_agent_friction_round2.py`**:
  - Added 14 unit tests covering Kaggle import mismatch policies, `safe_relpath`, string `task` specs, evaluation alias, and normalised object dictionary methods.

## Approach

- Fixed boolean logic so `_is_critical_identity_mismatch(key, mismatch)` returning `True` (critical) correctly populates `critical_mismatches`.
- Implemented `safe_relpath` in `io_utils.py` using fallback cascading: `Path.resolve().relative_to(...)` $\to$ `os.path.relpath(...)` $\to$ `str(path)`.
- Maintained backward compatibility across all task spec configurations.

## Evidence

- `tests/test_agent_friction_round2.py`: 14/14 tests pass.
- `tests/test_agent_friction_remediation.py`: 15/15 tests pass.
- `python tools/run_tests.py --smoke`: 47/47 tests pass in 2.7s.
- `ruff check harness/ tools/ tests/`: All checks passed with 0 errors.

## Impact & follow-ups

None.
