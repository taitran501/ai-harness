# Single-install architecture (PR #78)

- **Date**: 2026-10-17
- **PR**: [#78](https://github.com/taitran501/ai-harness/pull/78)
- **Status**: merged (squash)
- **Area**: framework architecture / toolchain

## 1. Why (motivation + thought)

Root cause, not a symptom: every workspace carried its own framework clone, and
`ai-harness sync` retargeted the ONE global editable install to whichever
workspace it ran in. With N workspaces competing for one global pointer,
`import harness` resolved to whichever workspace synced last — cross-workspace
drift. That drift is what the re-exec pin (PR #63) and the depth circuit
breaker (PR #74) had to *contain*. This change removes the drift at the source.

## 2. What (concrete changes)

- **`harness/toolchain.py`**: new `find_framework_root(project_root)` —
  resolution order `HARNESS_FRAMEWORK_DIR` > legacy per-workspace clone > None.
  `ensure_pinned_toolchain` / `_needs_pin` now pin to the framework root; when
  the loaded origin already IS the framework root, no re-exec happens (the
  re-exec tax disappears). `find_workspace_clone` kept as the legacy fallback.
- **`harness/sync.py`**: canonical-aware `run_sync`. When the framework repo
  lives OUTSIDE the workspace (canonical), sync pulls it but does NOT run
  `pip install -e` per workspace (the shared editable install is retargeted
  once, by hand, during migration); it only verifies the install points at the
  canonical checkout and warns (advisory) if not. Legacy per-workspace clones
  keep the old retarget behavior. Sync records a provenance pointer
  (`framework: {path, commit, branch, updated}`) and reports
  `mode: canonical|workspace-clone`.
- **`harness/project.py`**: `write_framework_pointer` /
  `read_framework_pointer` (manifest provenance helpers).
- **`harness/status_report.py`**: new `framework` section — `single_install`
  flag, resolved framework root, recorded pointer, and a stale-install
  advisory when the import origin != framework root.
- Remote dispatch is unaffected: the Kaggle/Modal bundle copies `harness/` from
  the installed (now canonical) source, so cloud runs pick up canonical code
  automatically.

## 3. Approach (decisions + rejected alternatives)

Remove drift at the source (one canonical checkout) rather than containing it
with per-entry-point pinning. `find_framework_root` prefers the explicit
`HARNESS_FRAMEWORK_DIR` pointer, falling back to a legacy clone only until
migrated. Sync deliberately does NOT retarget the global install per workspace
(that was the bug) — it verifies + warns instead. Rejected: keeping
per-workspace retargeting, which is the drift itself.

## 4. Evidence

- 12 new tests in `tests/test_single_install.py` (find_framework_root
  precedence/fallback/none/invalid; canonical pin + no-pin-when-already-canonical;
  canonical sync skips pip + writes pointer; workspace-clone mode; status
  single-install flag + advisory).
- Full suite green: **808 tests, OK (1 skipped) in ~200s**; ruff F,E7 clean.
- Also hardened `tests/test_sync_command.py` `_make_workspace` (branch-agnostic
  `git push -u origin HEAD`, so the suite is green on any branch).

## 5. Impact & follow-ups

- **Follow-up**: one-time migration (PR #79) — clone the canonical checkout,
  `pip install -e` once, set `HARNESS_FRAMEWORK_DIR`, per workspace delete the
  clone + `ai-harness sync`; verify import origin, status, and a live probe.
- **Downstream**: `docs/DESIGN_single_install.md`; `status` reports a
  `framework` section; `AGENTS.md` point 9 (Single-Install Framework).
