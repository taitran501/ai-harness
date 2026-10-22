# Multi-ref submission watch, submission traceability parity, Codabench auth discovery, and interactive gateway watchdog guideline

- **Date**: 2026-10-22
- **PR**: [#116](https://github.com/taitran501/ai-harness/pull/116)
- **Branch**: `feat/multi-ref-watch-and-submission-traceability`
- **Status**: merged
- **Commits**: `59ef8f9` (squash)

## Why

During real-world competition triage across active assistant sessions (`rsna-knee`, `arc-prize-2026-arc-agi-3`, and `aimo`), four friction points emerged:
1. **Multi-submission polling friction**: When submitting concurrent experiments (e.g. Exp040 ref `57507185` and Exp041 ref `57507198` in `rsna-knee`), `ai-harness submission watch` only accepted a single ref, forcing the assistant to write multiple custom Python polling daemon scripts.
2. **Missing submission metadata & kernel provenance**: In `arc-prize-2026-arc-agi-3` (ref `57510673`), an interactive gateway run timed out after 15 minutes of idle weight loading. The agent had to execute ~10 separate API inspection calls to retrieve the submission kernel URL and error description because `_submission_dict` lacked `url`, `submitted_by`, and `team_name`.
3. **Codabench token discovery & auth friction**: In `aimo`, Codabench session tokens stored in user home directories (`~/.codabench_session`) or scratch probe scripts (`scratch/*.py`) were not automatically discovered by `CodabenchClient`, and there was no CLI command to persist an authenticated cookie without manual file authoring.
4. **Interactive evaluation gateway timeouts**: Competition environments with streaming evaluation (e.g. ARC Prize Hearth gateway) terminate connections with a `0.00` score if model shard downloads/loading take over 15 minutes without sending heartbeat/actions.

## What changed

- **`harness/endgame_submission.py`**:
  - Implemented `wait_for_scores(competition, refs, timeout_s, poll_interval_s, max_poll_interval_s)`: polls multiple submissions concurrently using a single `list_submissions` API call per polling interval to avoid Kaggle API rate-limiting.
  - Extended `_submission_dict` to extract `submitted_by`, `team_name`, `url` (with `/code/owner/kernel?scriptVersionId=...`), and `file_name` fallback.
- **`harness/submission_dispatch.py`**:
  - Upgraded `watch` subcommand to accept one or more refs (`refs, nargs="+"`), supporting both space-separated and comma-separated ref inputs.
  - Automatically delegates to `wait_for_scores` when multiple refs are provided.
- **`harness/codabench_compat.py`**:
  - Upgraded `CodabenchClient._resolve_session_id` into a `@staticmethod` supporting:
    - User home directory lookup (`~/.codabench/session`, `~/.codabench/token`, `~/.codabench_session`).
    - Fallback regex discovery scanning `scratch/*.py` and `scripts/*.py` for `session_id = "..."`.
- **`harness/codabench_cli.py`**:
  - Added native subcommand `ai-harness codabench auth <session_id>` to securely save session cookies directly into `.codabench_session`.
- **`harness/skill_injector.py`**:
  - Added rule guidance for **Interactive Gateway & Idle Watchdog**: mandates background heartbeats or warm-up probes when weights loading exceeds 15 minutes.
  - Updated CLI command documentation for multi-ref watch and `ai-harness codabench auth`.
- **`tests/test_kaggle_discussions_and_submissions.py` & `tests/test_codabench_compat.py`**:
  - Added comprehensive unit tests covering `wait_for_scores`, multi-ref CLI args, submission metadata parity, Codabench token discovery, and `codabench auth`.

## Approach

- **Batched list polling**: Rather than spawning $N$ parallel polling loops each making individual `get_submission` requests, `wait_for_scores` calls `list_submissions` once per interval. This prevents HTTP 429 rate limit errors while tracking multiple submissions concurrently.
- **Subcommand ergonomics**: `ai-harness submission watch <comp> <ref1> [<ref2> ...]` handles both `ai-harness submission watch comp 111 222` and `ai-harness submission watch comp 111,222` transparently, retaining backwards-compatible single-ref output when only 1 ref is provided.

## Evidence

- **Unit tests**:
  - `python -m unittest tests/test_kaggle_discussions_and_submissions.py tests/test_codabench_compat.py` passes 18/18 tests.
  - `python tools/run_tests.py --smoke` passes 47/47 tests in 2.5s.
  - `python tools/run_tests.py --fast` passes 920/920 tests.
- **Static gate**:
  - `ruff check harness/ tools/ tests/` passes with 0 errors.

## Impact & follow-ups

None.
