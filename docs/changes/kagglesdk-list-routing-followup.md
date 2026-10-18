# kagglesdk `list_kernels` routing + `competition_topics` method fix

**PR:** #92 · **Branch:** `fix/kagglesdk-list-routing-and-hygiene` · **Date:** 2026-10-18
**Status:** Implemented

## Why

The post-PR#91 codebase audit found two facade methods still broken on the
live SDK surface (kaggle 2.2.4 + kagglesdk), both advertised in §7:

- **`list_kernels`** — the kagglesdk branch delegated flat
  (`self._client.kernels_list(...)`) but kagglesdk clients are namespaced
  (`client.kernels.*`) with **no** flat method (verified via `dir()`:
  zero flat-ish attributes). Same F2 class fixed in PR #91 for
  `kernels_status/logs/output`.
- **`competition_topics`** — **broken on BOTH SDKs since it shipped (PR #51,
  commit `0cd5e9e`, added from an audit report without live SDK
  verification).** The attribute-probing list tried
  `competition_topic_fields` first; in kaggle ≥ 2.x that name is a
  **class-level constant** (a field-projection list, used internally at
  `kaggle_api_extended.py:3758`), so `hasattr` found it and the call raised
  `TypeError: 'list' object is not callable` — reproduced locally without
  network. The other probed names (`competition_topics_list`,
  `competition_topics`, `list_competition_topics`) do not exist at all. The
  real method is `KaggleApi.competition_list_topics(competition, sort_by,
  page)` (source-verified), which returns
  `ApiListCompetitionTopicsResponse` with `.topics` / `.total_count`.

## What

- **`harness/kaggle_compat.py`**
  - `list_kernels`: both SDKs now route through the legacy flat API
    (`_legacy_api_instance()` on kagglesdk, `self._legacy_api` on kaggle) —
    `KaggleApi.kernels_list(user=..., page, page_size, ...)` verified on
    kaggle 2.2.4; returns snake_case `ApiKernelMetadata` objects, handled by
    the existing normalisation layer. `**kwargs` now passes through on both
    branches.
  - `competition_topics`: direct `api.competition_list_topics(competition,
    **kwargs)` (guarded by `hasattr`) for both SDKs; unwraps
    `response.topics` into normalised objects
    (`id`, `title`, `author_name`, `comment_count`, `post_date`,
    `is_sticky`). The constant-probing loop is gone.
- **Hygiene (3):**
  - `tests/test_challenger_m1_1_empirical.py`: module docstring made raw
    (`r"""`) — math backslashes (`\times`, `\sigma`, …) were W605
    invalid-escape sequences.
  - `docs/research_trace.md`: replaced the stale "maintain WORK_LOG.md"
    instruction with the `docs/changes/` convention (matches AGENTS.md §7;
    WORK_LOG is a read-only archive).
  - `tests/test_kaggle_compat.py` module docstring: removed the false claim
    that `test_kaggle_runtime` covers the facade's SDK contracts (it covers
    the remote runtime, not the facade).

## Approach

Both fixes reuse the proven PR #91 pattern: when the kagglesdk client is
namespaced, the flat legacy `KaggleApi` (kept by kaggle 2.2.4) is the
reliable cross-SDK entry point, with the normalisation layer unchanged above
it. `competition_topics` additionally moves from blind attribute probing to
a single explicit, verified method call — probing against a namespace that
contains field-projection *constants* was the root cause of the TypeError.
No public API removals; §7 method names and the field table are unchanged.

## Evidence

- Verified against the installed SDK (no network):
  - `inspect.signature(KaggleApi.kernels_list)` → `(page, page_size, dataset,
    competition, parent_kernel, search, mine, user, language, kernel_type,
    output_type, sort_by)`.
  - `KaggleApi.competition_topic_fields` → list constant; calling it →
    `TypeError: 'list' object is not callable` (the old facade path,
    reproduced).
  - `inspect` of `competition_list_topics` source → calls
    `kaggle.competitions.competition_api_client.list_competition_topics`;
    `ApiListCompetitionTopicsResponse` → `topics`, `total_count`;
    `ApiCompetitionTopic` → snake_case fields.
  - `dir(kagglesdk.KaggleClient)` → only namespaced services
    (`Kernels`, `Competitions`, …); zero flat attributes.
- New tests: `list_kernels` routing (kagglesdk + kaggle),
  `competition_topics` routing (kagglesdk + kaggle), and
  `ConstantFieldsRegressionTests` proving the `*_fields` constant is never
  treated as callable.
- Gates: ruff F,E7 + W605 clean; targeted 47 tests pass; fast + full suite
  OK (see commit / CI on this branch).

## Impact

- `list_kernels` and `competition_topics` now work under both SDK
  generations — two §7-advertised methods that would have raised
  `AttributeError`/`TypeError` at runtime are closed.
- The constant-probing anti-pattern is removed and regression-pinned, so a
  future upstream rename fails loudly at the `hasattr` guard instead of
  calling a constant.
- Docs now consistently point framework-repo change documentation at
  `docs/changes/`.
