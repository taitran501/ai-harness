# Anti-Hallucination API Reference + Dispatch Regression Tests

- **Date**: 2026-10-18
- **PR**: [#88](https://github.com/taitran501/ai-harness/pull/88)
- **Branch**: `feat/api-reference-dispatch-tests`
- **Status**: open
- **Commits**: `—` (squash)

## Why

A meta-loop transcript scan of the latest agent conversations (RSNA, Airline, Zindi,
AIMO) showed the top failure signatures were **API hallucinations**, not ML bugs:

- `ModuleNotFoundError` for invented modules (`harness.agents`, `harness.task_config`)
- `AttributeError` / `KeyError` from calling the **raw** Kaggle SDK (`publicScore`,
  `totalVotes`) instead of the `kaggle_compat` facade
- bypass of the dispatch facade to hand-roll raw-SDK calls

Coverage analysis confirmed the correlation: the exact layers where agents
hallucinate/bypass were under-tested — `submission_dispatch.py` at **0%**,
`kaggle_compat.py` at **43%**. `GroundingGate` guards *ML hypotheses* but does
nothing about the *API surface*, and no queryable API map existed for the agent
to look up real names.

## What changed

- **`harness/api_reference.py` (new)** — a curated, *verified* API surface as data
  (CLI commands, entry points, `KaggleClient` facade methods, Python entry-points,
  key modules with their correct names). `validate_reference()` checks that every
  entry actually resolves (CLI command registered, symbol importable, facade
  method present); `render_reference()` emits the Markdown section.
- **`harness/skill_injector.py`** — the generated SKILL.md now appends
  **§7 "Verified API surface — look names up here, do NOT invent them"**,
  rendered from `api_reference`. It carries the anti-bypass message ("call the
  facade, not the raw SDK") and the "NOT `harness.agents`" / "NOT
  `harness.task_config`" name corrections straight from the transcript evidence.
- **`tests/test_api_reference.py` (new)** — pins the reference: every listed
  command/symbol/module/facade-method must resolve; the render includes the key
  names; SKILL.md contains §7; and the hallucinated module names
  (`harness.agents`, `harness.task_config`) are asserted **absent** (so the
  "correct name" hints stay truthful).
- **`tests/test_submission_dispatch.py` (new)** — covers the previously 0%-coverage
  `ai-harness submission` CLI: all four subcommands (`dispatch`/`status`/`watch`/
  `latest`) plus the JSON error path, with the tracker fully mocked (offline).
- **`tests/test_kaggle_compat.py`** — added `ClientDispatchPathTests`: the PR #84
  dispatch routing (file → `competition_submit`, code-comp →
  `competition_submit_code`, `kernels_push`, `_legacy_api_instance`, `get()`),
  exercised via a `KaggleClient` built without a live SDK.

## Approach

- **Reference as data, validated by tests.** The reference is *curated* (one-line
  descriptions have value) but its *correctness* is enforced by
  `validate_reference()` + `tests/test_api_reference.py`, not by the renderer. A
  stale or invented name fails the suite instead of rotting silently. This is the
  anti-hallucination mechanism: the agent gets a real, queryable map, and the map
  is guaranteed honest.
- **Delivery through the existing SKILL.md channel.** The reference is rendered
  into the generated SKILL.md (§7), the same surface the agent already reads — no
  new discovery step, no new tool. Rejected alternative: an `--help` / `api-ref`
  CLI subcommand; it adds a discovery step the agent would not take, whereas the
  SKILL.md is always in context.
- **`_resolve()` walks the longest importable module prefix**, so a dotted path
  like `harness.kaggle_compat.KaggleClient.list_submissions` imports
  `harness.kaggle_compat` then walks the attributes — robust to nested symbols.
- **Dispatch tests use `KaggleClient.__new__` + a `_FakeLegacyApi`** to
  unit-test the exact high-level calls (submit/push) without a live SDK or
  credentials.

## Evidence

- **791 tests, OK (skipped=1)** on the fast tier (+19 new: 8 api-reference,
  6 submission-dispatch, 5 kaggle-compat dispatch-path).
- `ruff check --select F,E7` clean on all changed files.
- `api_reference.validate_reference()` returns `[]` (all entries resolve).
- Generated SKILL.md renders §7 (contains `harness.task_spec`, `list_submissions`,
  the "NOT `harness.agents`" hint, and the facade method table).

## Impact & follow-ups

The agent now receives a verified, in-context map of real command/module/method
names — directly targeting the `AttributeError` / `ModuleNotFoundError` friction
seen in transcripts — and the dispatch/compat layers that were 0% / 43% are now
regression-tested. **Limitations:** the reference is curated (a new CLI command or
symbol must be added to `api_reference.py`; the pin test catches omissions), and
the raw-SDK `publicScore`-style bypass is addressed by the facade message, not a
hard block (the host hook still blocks raw *shell* `kaggle ...` commands). No
follow-up PR required; monitor whether `manual_kaggle` hit counts drop in the
next transcript scan.
