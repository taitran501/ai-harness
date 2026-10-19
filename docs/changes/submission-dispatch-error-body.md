# Submission dispatch errors surface the Kaggle API response body

- **Date**: 2026-10-19
- **PR**: #TBD (feat/kaggle-error-surface)
- **Branch**: `feat/kaggle-error-surface`
- **Status**: open
- **Commits**: — (pending)

## Why

The arc-prize-2026 forensic audit (Antigravity session `ce5361e6`, step 2500)
found the agent's final `ai-harness submission dispatch` failing with only
`400 Client Error: Bad Request for url: ...` — the raw `requests.HTTPError`
string. Kaggle's explanation of *why* (daily submission limit, bad kernel
slug, rejected file) lives in the response JSON body, which the facade
discarded. The agent spent the rest of the session guessing the cause
(its own insight record concluded "daily limit, wait for UTC reset") and
could not distinguish limit-exhausted from a malformed request — the exact
class of ambiguity a surfaced body eliminates.

## What changed

- **`harness/kaggle_compat.py`** — new module-level
  `_describe_api_error(exc)`:
  - Reads `exc.response` (requests-shaped; verified against the installed
    `kagglesdk.kaggle_http_client`, which raises
    `requests.exceptions.HTTPError(resp["message"], response=...)` on the
    JSON-error path and `http_response.raise_for_status()` otherwise — both
    carry `.response`).
  - Surfaces `response.text` (truncated to 2000 chars) and `response.url`;
    each read is individually guarded so a diagnostic failure can never mask
    the original error.
  - No `.response` → `response_body=<none>` fallback; the original exception
    is always embedded via `raise ... from exc`.
- **`harness/kaggle_compat.py`** — `KaggleClient.submit_submission` wraps the
  legacy `competition_submit` / `competition_submit_code` calls and re-raises
  any dispatch exception as `RuntimeError(_describe_api_error(exc))`. Both
  the standard-file and code-competition paths are covered at this single
  choke point, so `endgame_submission.KaggleLeaderboardTracker.dispatch` and
  the `ai-harness submission dispatch` CLI (which stringifies the exception
  into its JSON payload) now show the body without further changes.
- **`tests/test_kaggle_compat.py`** — `ClientDispatchPathTests` additions:
  - `test_submit_error_surfaces_response_body_and_url` (400 with JSON body →
    body + url present in the raised message, `__cause__` preserved),
  - `test_submit_error_without_response_keeps_fallback` (plain exception →
    `response_body=<none>`),
  - `test_describe_api_error_truncates_long_body` (5000-char body → ≤2000).

## Approach

Wrap at the facade's single choke point (`submit_submission`) rather than in
every caller: the legacy `KaggleApi` is the only submit path, and both
`competition_submit` and `competition_submit_code` funnel through it. The
extractor is shape-tolerant (`getattr` + per-field guards) instead of
importing `requests`, keeping `kaggle_compat` dependency-free as before and
working whether the installed SDK raises `requests.HTTPError` or a
kagglesdk-native wrapper. Body truncation at 2000 chars keeps the one-line
error readable while preserving full Kaggle JSON payloads (typically <300
chars). Deliberately **not** changing: the success path (normalised
`ref`/`message` unchanged), retry semantics (dispatch has none — Kaggle
submissions are not idempotent), or the CLI's JSON error payload shape
(`{"ok": false, "error": ...}`), which now simply carries a better string.

## Evidence

- `tests/test_kaggle_compat.py`: 56/56 pass (53 pre-existing + 3 new) with
  `tests/test_job_state.py`, `tests/test_submission_dispatch.py`,
  `tests/test_kaggle_leaderboard_tracker.py`.
- `python tools/run_tests.py --smoke`: 47 tests OK.
- `ruff check harness/ tools/ tests/`: clean.
- Exception shapes verified against the installed
  `kagglesdk/kaggle_http_client.py` (JSON path: `HTTPError(resp["message"],
  response=http_response)`; fallback path: `raise_for_status()`).

## Impact & follow-ups

- Agents (and users) see the Kaggle error payload on any failed dispatch —
  daily-limit, slug, and file problems become self-diagnosable.
- Message length grows only on failure; no behaviour change on success.
- Follow-up (none blocking): if Kaggle starts raising non-requests-shaped
  exceptions without `.response`, the fallback path still degrades
  gracefully to the original message.
