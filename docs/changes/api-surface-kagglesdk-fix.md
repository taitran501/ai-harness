# §7 field truthfulness + kagglesdk kernel methods fix

**PR:** #91 · **Branch:** `fix/api-surface-kagglesdk` · **Date:** 2026-10-18
**Status:** Implemented

## Why

The post-PR#89 live RSNA run (conversation `95c2aed5`, 2026-10-18) exposed two
harness-side bugs. The agent followed §7 faithfully and still hit
`AttributeError` — the verified reference itself was wrong for the SDK shapes
actually installed:

- **F1 (stale §7 fields):** the live environment runs **kaggle 2.2.4 +
  kagglesdk**, whose `competition_submissions()` returns `ApiSubmission`
  objects in snake_case with a **`date`** field (and **no `total_votes`**).
  §7 advertised `created_at` (from the legacy-1.x `createdAt` fixture) and
  `total_votes` — both unresolvable on the live shape. The PR#89 fixture
  modeled legacy 1.x camelCase, so `validate_reference()` passed while the
  reference lied in production.
- **F2 (broken kagglesdk delegation):** `kernel_output` / `kernels_status` /
  `kernels_logs` delegated flat (`self._client.kernels_status(...)`) on the
  kagglesdk path, but kagglesdk clients are **namespaced**
  (`client.kernels.get_kernel_session_status(...)`). Because the kagglesdk
  client class is also named `KaggleClient`, the resulting
  `AttributeError: 'KaggleClient' object has no attribute 'kernels_status'`
  was misleading (it looked like the facade lacked the method).

## What

- **`harness/kaggle_compat.py`**
  - `_LEGACY_ALIASES` restructured: canonical name → ordered tuple of raw
    candidate names. New entries: `created_at → (createdAt, submittedDate,
    date, creationDate)`, `date → (submittedDate, createdAt)`,
    `total_bytes`, `file_name`, `team_name`.
  - New helper `_alias_candidates(name)` resolves forward (canonical → raw
    candidates) and reverse (legacy request → canonical attr); `__getattr__`
    and `KaggleClient.get` both use it.
  - `kernel_output` / `kernels_status` / `kernels_logs`: kagglesdk path now
    routes through `_legacy_api_instance()` (the flat legacy API, verified
    present in kaggle 2.2.4) — same cross-SDK pattern as
    `list_submissions`/`submit_submission`.
- **`harness/api_reference.py`**
  - `SUBMISSION_FIELDS` re-pinned to the live shape: `date` added, `total_votes`
    removed (does not exist on `ApiSubmission`), `file_name` / `team_name` /
    `total_bytes` added; `created_at` kept as a documented alias of `date`.
  - Dual fixtures: `_submission_fixture()` (live kaggle 2.x/kagglesdk
    snake_case) + `_submission_fixture_legacy()` (legacy 1.x camelCase);
    `validate_reference()` now validates every §7 field against **both**
    shapes, so a future SDK shape change fails the suite instead of rotting
    the reference.

## Approach

Fixes are alias-map + delegation changes only — no public API removals.
`created_at` still works (backward compatible with agents that learned it from
§7 post-PR#89), while `date` becomes the primary documented field matching the
live object. The kernel methods prefer the flat legacy API because it exists
in both SDK generations (kaggle 2.2.4 keeps the classic `KaggleApi` surface),
matching the dispatch path's existing cross-SDK strategy; if the legacy
package were absent, `_legacy_api_instance()` raises an explicit
`ImportError` with install guidance.

## Evidence

- `tests/test_api_reference.py`: `SubmissionFieldsTests` now asserts every
  field resolves on **both** fixtures; `created_at`/`date` value checks on
  both shapes; `total_votes` pinned absent from the reference; render test
  extended (`date`).
- `tests/test_kaggle_compat.py`: new `ClientKagglesdkKernelRoutingTests`
  (status/logs/output all route to the legacy instance on sdk=kagglesdk) +
  `created_at`/`date` resolution tests on kagglesdk and legacy date shapes.
- `validate_reference()` → `[]` on the dev environment (kaggle 2.2.4 +
  kagglesdk installed).
- Targeted: 42 tests pass (test_api_reference + test_kaggle_compat).
- Gates: `ruff check harness/ tools/ tests/ --select F,E7` clean; fast + full
  suite OK (see commit metadata / CI run on this branch).

## Impact

- Agents reading §7 can no longer be sent to a non-existent attribute:
  `list_submissions()[i]` fields resolve on both SDK generations.
- `client.kernels_status(slug)` / `kernels_logs` / `kernel_output` work under
  kagglesdk (previously `AttributeError`), so the live RSNA friction points
  F1/F2 are closed.
- The reference's truthfulness is now enforced against the *live* object
  shape, not only the legacy one — the class of bug where a fixture models a
  shape that no installed SDK produces.
- Follow-up: regenerate `SKILL.md` (via `python -m harness.skill_injector`)
  in all workspaces after merge so §7 ships the corrected field table.
