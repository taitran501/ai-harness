# Live Adoption Verification + Submission Field Reference (§7)

- **Date**: 2026-10-18
- **PR**: [#89](https://github.com/taitran501/ai-harness/pull/89)
- **Branch**: `feat/submission-fields-reference`
- **Status**: open
- **Commits**: `—` (squash)

## Why

PR #88 added §7 ("Verified API surface") to the generated SKILL.md to stop the
two failure signatures seen in transcript scans — **API hallucination**
(`ModuleNotFoundError` for invented modules, `AttributeError` from the raw
camelCase SDK) and **raw-SDK bypass** (agents hand-rolling `kaggle ...` / raw
`kagglesdk` calls instead of the harness facade). Before shipping it as proven,
the change needed a live A/B: a *fresh* RSNA agent session running against the
new SKILL.md, scored against the pre-§7 baseline.

**The live run (conv `5116e48a`) confirmed §7 works:**

| Signal | Baseline RSNA (`e0aaeefd`, pre-§7) | Post-§7 run (`5116e48a`) |
|---|---|---|
| Raw-SDK/CLI bypass (executed) | 4016 manual-kaggle calls | **0** |
| Hallucinated-name errors | 108 `AttributeError` (+ invented modules) | **0** tracebacks |
| Adoption | hand-rolled `build_expXXX.py` + 45s poll loops | all via `ai-harness submission` + `KaggleClient` |

**One residual micro-gap** the run exposed: §7 lists the facade *methods* but not
the *fields* on the returned Submission object, so the agent still guessed the
date field (tried `s.date` → `getattr(s, "submittedDate")` → `None`), while
`ref` / `status` / `public_score` / `description` (advertised) worked. The fix
here adds the normalised field names to §7.

## What changed

- **`harness/api_reference.py`** — new `SUBMISSION_FIELDS` curated list
  (`ref`, `status`, `description`, `public_score`, `private_score`,
  `total_votes`, `created_at`). `validate_reference()` now proves each field
  resolves through the `kaggle_compat._NormalisedObject` proxy (via a
  `_submission_fixture()`), and `render_reference()` emits a
  "Submission object — normalised fields" subsection in §7.
- **`tests/test_api_reference.py`** — `SubmissionFieldsTests` (fields resolve on
  the normalised object; `created_at` is the real date field, regression for the
  `.date` guess) + `RenderSubmissionFieldTests` (the render lists the fields).
- **`docs/changes/live-adoption-verified.md`** (this file) + `README.md` index
  row.

## Approach

- **Verify before you ship.** The A/B is a *fresh* conversation post-#88, not a
  re-run of an old one — a re-run would inherit the old context/state and not
  exercise the new SKILL.md. Baseline = the pre-§7 RSNA conv; the control is
  apples-to-apples (same task, same agent, only the §7 SKILL.md differs).
- **Field reference validated by the same mechanism as #88.** Consistent with
  "the reference is data, correctness is enforced by `validate_reference()`":
  the fixture carries the *legacy* response shape (`publicScore`, `createdAt`, …)
  and each *canonical* name must resolve through the proxy — so a typo or a
  missing alias fails the suite. This is stronger than a render smoke-test: it
  proves `created_at` actually maps from `createdAt`.
- **Only list fields the normalisation layer genuinely serves.** The score /
  votes / date fields are described as "if present" because whether the raw
  Kaggle response carries them is competition-dependent; the guarantee is that
  the canonical name is the correct spelling to use (never camelCase).

## Evidence

- **Live A/B (transcript-verified, not self-reported):** parsed the full JSONL of
  conv `5116e48a`. 14 executed commands, **all** `ai-harness submission …` or
  `from harness.kaggle_compat import KaggleClient`; 0 executed raw-SDK/CLI. The
  raw-SDK strings in the transcript are all *file content read*
  (`kaggle_compat.py`) or *SKILL.md policy text*, not executed. 0
  `most recent call last` (the "error" tokens were §7's own quoted examples). The
  17-submission table in the agent's `scratch/adoption_check.md` matches the real
  tool output (best 0.958, refs `57313652` / `57271119`).
- **794 tests, OK (skipped=1)** on the fast tier (+3 new: 2 submission-fields,
  1 render).
- `ruff check --select F,E7` clean on changed files.
- `api_reference.validate_reference()` returns `[]` (all entries incl. the 7
  submission fields resolve).

## Impact & follow-ups

§7 now covers the full read/dispatch surface including the Submission-object
fields, so the agent no longer guesses the date/vote field names.
**Limitation:** the field reference is curated (a new normalised field must be
added to `SUBMISSION_FIELDS`; `validate_reference` + the pin tests catch
omissions), and the fixture models the legacy response shape (the kagglesdk
Pydantic path is covered by the same alias table). **Follow-up:** after this is
live, re-run a transcript scan to confirm the residual `.date`-style guess is
gone (expect 0).
