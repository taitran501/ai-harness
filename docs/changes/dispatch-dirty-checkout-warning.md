# Dispatch warning when the canonical framework checkout is dirty

- **Date**: 2026-10-19
- **PR**: [#109](https://github.com/taitran501/ai-harness/pull/109)
- **Branch**: `fix/arc-prize-audit-remediation`
- **Status**: merged
- **Commits**: `abb485b` (squash)

## Why

Remote bundles are built from the installed (canonical) framework
source. During the arc-prize session an agent patched the canonical
checkout **in place** (hand-edited `harness/artifacts.py`,
`harness/project.py`) to work around framework bugs — leaving the
checkout dirty and the framework repo unfixed. The bundle then shipped
uncommitted edits that match no committed framework identity and would
vanish on the next `ai-harness sync`, with no visible signal at
dispatch time.

## What changed

- **`harness/cloud_runner.py`** — new `_warn_if_framework_checkout_dirty()`,
  called from `_prepare_remote_run()` (shared entry for both the Kaggle
  and Modal dispatch paths):
  - Locates the framework checkout via `harness.__file__`; skips
  silently when it is not a git repository (e.g. site-packages install).
  - Runs `git status --porcelain` (15s timeout); any failure is silent.
  - With uncommitted changes, prints
    `[dispatch preflight] WARNING: framework checkout is dirty (N
    uncommitted change(s) at <root>). ...` to stdout so it lands in the
    dispatch log.
- **`tests/test_kaggle_preflight.py` (new, part 3)** —
  `DirtyCheckoutWarningTests`: clean checkout silent; dirty checkout
  warns with the change count; non-git directory silent (no git call);
  git failure silent.

## Approach

Warning, not gate: a dirty checkout does not *prevent* a correct
dispatch (operators legitimately dispatch from WIP trees), but it must
be visible in the log for auditability — especially because the dirty
state in the incident was agent-created. Implemented in
`_prepare_remote_run` so both backends get it with one call; kept
defensive (any probe failure → silence) because a warning feature must
never break dispatch. Alternative considered — hard-failing on a dirty
checkout — rejected: it would block legitimate WIP experimentation and
the sync flow itself operates on dirty trees.

## Evidence

- `tests/test_kaggle_preflight.py::DirtyCheckoutWarningTests`: 4/4 pass.
- Manual check on the framework repo: clean tree → silent; with a temp
  untracked file → warning with correct count (12 → 13 changes).

## Impact & follow-ups

- Dispatch logs now carry an explicit, greppable
  `[dispatch preflight] WARNING` marker whenever uncommitted framework
  edits are being shipped.
- Combined with the SKILL.md §7 "Framework discipline" note (never patch
  the canonical checkout in place; fix in the framework repo +
  `docs/changes/` + `ai-harness sync`), this closes the loop that let
  the in-place patches go unnoticed.
- Follow-up (none blocking): `ai-harness status` could surface the same
  advisory for the local checkout.
