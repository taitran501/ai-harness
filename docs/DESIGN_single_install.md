# Single-Install Architecture — Design

**Status:** Implemented (PR #78) — `find_framework_root` + canonical-aware
`sync` + manifest provenance pointer + `status` framework section are in
place; the one-time operator migration (canonical clone + `pip install -e`
+ `HARNESS_FRAMEWORK_DIR` + delete per-workspace clones) is the remaining
step. Legacy per-workspace clones keep working via the `find_workspace_clone`
fallback until migrated.
**Supersedes:** the per-workspace framework clone model introduced with the
adoption stack. **Relates to:** PR #74 (toolchain re-exec loop breaker),
`docs/architecture.md`, `docs/kaggle_execution.md`.

---

## 1. Overview (TL;DR)

Today each competition workspace ships its **own `ai-harness/` git clone**, and a
**single global editable install is retargeted on every `sync`** to whichever
workspace was synced last. `import harness` therefore resolves to *some* clone —
not necessarily the one the current workspace expects. `ensure_pinned_toolchain`
detects that drift and re-execs the CLI under the workspace clone; that re-exec is
the **~0.8–1.0 s "re-exec tax"** on every non-last-synced workspace, and (before
PR #74) the source of an **infinite re-exec loop** that pegged the CPU.

This design collapses the model to **one canonical framework install**:

- A single canonical repo at a fixed path, installed **once** as the global
  editable package. It is **never retargeted per workspace**.
- Workspaces keep **no framework clone** — only their `.ai-harness/` state and a
  small `framework` pointer (path + commit) for provenance.
- `import harness` always resolves to the canonical install → **no drift, no
  re-exec, no loop**.
- `sync` becomes cheap: `git pull` in the shared canonical repo (idempotent across
  workspaces) + update the workspace pointer + re-attach. **No per-workspace
  `pip install`.**

The remote (Kaggle/Modal) dispatch path is **already clone-independent** and is
unaffected (see §4).

---

## 2. Problem Statement

The N-clones + retargeted-global-install model causes:

1. **Re-exec tax.** Every CLI call in a workspace that is *not* the last one
   synced pays a full process re-exec (~0.8–1.0 s) because the loaded `harness`
   originated from a different clone than the workspace expects.
2. **Loop risk.** The re-exec child could still observe the same drift (e.g. a
   `cwd` holding its own `harness/` package wins `-m` path resolution over the
   pinned `PYTHONPATH`), re-exec again, and so on. PR #74 bounded this with an
   `AI_HARNESS_PIN_DEPTH` circuit breaker — a guard, not a root-cause fix.
3. **N copies of the code.** 5 workspace clones + the dev repo ≈ 6 × ~14 MB, all
   tracking the same GitHub remote, all needing individual `sync`.
4. **Slow `sync`.** Per-workspace `pip install -e` is the dominant cost (minutes,
   sequential).
5. **Version skew.** Clones can sit at different commits depending on when each
   was last synced, so two workspaces can run different framework versions —
   the opposite of what metric reproducibility wants.

---

## 3. Decisions (agreed)

| # | Decision | Choice |
|---|---|---|
| D1 | Canonical location | A **dedicated path separate from the dev repo** — `C:\Users\trant\ai-harness-framework`. Dev work stays in `C:\Users\trant\Documents\ai-harness`. |
| D2 | Versioning policy | **One shared version.** Every workspace runs the same canonical commit. This also improves cross-workspace metric comparability. |
| D3 | Existing clones | **Delete** the 5 workspace `ai-harness/` clones. A workspace keeps only `.ai-harness/` state + a `framework` pointer. |
| D4 | Resolution mechanism | **Global `HARNESS_FRAMEWORK_DIR`** env var pointing at the canonical path. The per-workspace `framework` pointer records provenance but is not the resolver. |

---

## 4. Key Verification Finding (de-risks the change)

**The remote dispatch path is already clone-independent.**

`harness/templates/kaggle_template.py::setup_kernel_package` bundles the
framework core **from the imported package, not from the workspace clone**:

```python
# Bundle the exact installed harness source tree, not a workspace-local fork.
harness_source = Path(__file__).resolve().parents[1]   # -> <installed>/harness
shutil.copytree(harness_source, bundle_base / "harness", ...)
```

`harness/remote_runtime.py` records the core as
`{"distribution": "ai-harness", "source": "bundled"}`, and
`harness/data_bundle.py` stages **only declared competition data** (explicitly
blocking `.ai-harness`, `artifacts`, `.git`, …). So deleting the per-workspace
clones does **not** break Kaggle/Modal execution: the remote kernel receives the
harness source from wherever the dispatching process loaded it — i.e. the
canonical install.

**Therefore the single-install change is local-only** (Python import resolution +
sync/attach bookkeeping). The highest-risk path is already safe.

---

## 5. Target Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │  canonical:  C:\Users\trant\ai-harness-framework
                    │  (git clone, main) — the ONE editable install
                    └───────────────▲─────────────────────────────┘
                                     │  global editable install
                                     │  HARNESS_FRAMEWORK_DIR=<canonical>
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
  ┌─────┴──────┐              ┌──────┴──────┐              ┌──────┴──────┐
  │ workspace  │              │ workspace   │              │ workspace   │
  │ aimo       │              │ rsna-knee   │              │ ... (5 total)│
  │  .ai-harness/  ◄─ pointer (path, commit)│              │  .ai-harness/│
  │  (no clone) │              │  (no clone) │              │  (no clone)  │
  └────────────┘              └─────────────┘              └──────────────┘
```

- **Import resolution.** `import harness` → canonical (via the global editable
  install). `HARNESS_FRAMEWORK_DIR` is the explicit, environment-level pointer
  used by `sync`/attach/toolchain to locate the canonical repo.
- **No drift → no re-exec.** Because every workspace loads the same canonical
  code, `ensure_pinned_toolchain` finds no divergence and does nothing.
- **Provenance.** Each workspace records `framework: {path, commit}` in its
  `.ai-harness/` manifest so an artifact can state *which* framework commit
  produced it, even though the resolver is the shared env var.

### Import-resolution edge cases (handled, not ignored)

- **Dev repo on the path.** The dev repo `C:\Users\trant\Documents\ai-harness`
  also contains `harness/`. When *developing*, tests run with
  `sys.path.insert(0, REPO_ROOT)` so they use dev code explicitly. The global
  `ai-harness` console script uses canonical. These do not conflict.
- **A stray `harness/` in `cwd`.** `python -m harness.cli` from a directory that
  contains its own `harness/` still prefers cwd. This is the pre-existing
  behavior that PR #74's circuit breaker guards. Under single-install the
  *expected* origin is canonical; if cwd diverges, the toolchain either re-execs
  under canonical (depth 1) or fails open with a clear diagnostic (depth ≥ 1) —
  the loop breaker from PR #74 remains as the last line of defense.

---

## 6. Code Changes

All changes are made in the **dev repo** on branch `arch/single-install`, tested,
pushed to GitHub, then the one-time migration (§7) is run.

### 6.1 `harness/toolchain.py`
- Generalize `find_workspace_clone(project_root)` → `find_framework_root(
  project_root)`:
  1. If `HARNESS_FRAMEWORK_DIR` is set and valid → return it (the canonical root).
  2. Else fall back to the current per-workspace clone search (backward
     compatibility during migration).
  3. Else `None`.
- `_needs_pin` / `ensure_pinned_toolchain`: pin to the **canonical root** instead
  of the workspace clone. With no clone and no env var, behavior is unchanged
  (no-op). The `AI_HARNESS_PIN_DEPTH` circuit breaker is kept verbatim.
- Keep `current_origin()` as-is.

### 6.2 `harness/sync.py`
- `find_framework_repo(project_root, framework_dir)`: already honors
  `HARNESS_FRAMEWORK_DIR` / `--framework-dir`; retain, and treat the resolved
  path as the **canonical** repo (external, not a workspace subdirectory).
- `run_sync(...)`:
  - Resolve the canonical repo (env / flag / workspace sub-dir fallback).
  - `git pull --ff-only` in the canonical repo (shared, idempotent — a second
    workspace syncing back-to-back is a no-op pull).
  - **Remove** the per-sync `pip install -e` step. Replace with a **report-only**
    check that the global editable install targets the canonical repo; warn on
    mismatch, never auto-retarget.
  - Write the workspace `framework` pointer (see 6.3).
  - Re-attach in a pinned subprocess; `PYTHONPATH` now pins to the canonical
    root (the global install already points there, so this is belt-and-braces).
  - Report `canonical: {path, commit, branch, dirty}` instead of the per-workspace
    clone state.
- Keep `--list`/JSON output shape backward-compatible (new fields, no removed
  required fields).

### 6.3 `harness/project.py` (attach) — framework pointer
- On `attach_project`, record in the `.ai-harness/` manifest (e.g.
  `.ai-harness/project.yaml`):
  ```yaml
  framework:
    path: C:\Users\trant\ai-harness-framework
    commit: <short sha>
    branch: main
    updated: <iso-timestamp>
  ```
- The pointer is **provenance only** — the resolver is `HARNESS_FRAMEWORK_DIR`.
- `status` / `adoption-scan` read the pointer to report which framework commit a
  workspace last attached against.

### 6.4 `harness/status_report.py`
- Report the harness origin and a `single_install: true/false` flag (true when
  `HARNESS_FRAMEWORK_DIR` is set **and** the loaded origin equals the canonical
  install).
- When the global editable install does not target the canonical repo, surface a
  one-line advisory (do not retarget).

### 6.5 Docs
- `AGENTS.md`: replace the "N workspace clones + retargeted global install"
  notes with the single-install model; update the `sync` description.
- `docs/architecture.md`: add a "Framework installation" section.
- This file (`docs/DESIGN_single_install.md`) becomes the canonical reference.

### 6.6 No change needed
- `harness/templates/kaggle_template.py`, `harness/remote_runtime.py`,
  `harness/data_bundle.py`, `harness/cloud_runner.py` — already clone-independent
  (§4).
- `harness/runner.py`, metrics, audit, research memory — unaffected.

---

## 7. One-Time Migration (after the code is pushed)

> Prerequisite: `arch/single-install` merged to `main` and pushed to GitHub.

| Step | Command | Notes |
|---|---|---|
| M1 | `git clone https://github.com/taitran501/ai-harness.git C:\Users\trant\ai-harness-framework` | Fresh canonical clone. |
| M2 | `pip install -e C:\Users\trant\ai-harness-framework` | Retargets the global editable install to canonical. **Once.** Verify `python -c "import harness; print(harness.__file__)"` → canonical. |
| M3 | `setx HARNESS_FRAMEWORK_DIR C:\Users\trant\ai-harness-framework` | User-level env; **new shells only**. Verify in a fresh shell. |
| M4 | Per workspace (aimo, rsna-knee, airline_prediction, semeval_2027, zindi): verify the clone is clean & at `main` HEAD → **delete** `<ws>\ai-harness\` → run `ai-harness sync` in the workspace | Sync writes the pointer + re-attaches. Deletion is safe because the clone is a clean GitHub clone (rollback §8). |
| M5 | Verify (§11) | Import origin, status, validate-task, one dispatched probe. |

**Ordering matters** because of the chicken-and-egg: the *new* `sync` logic lives
inside the framework, so M2 (install the new canonical code) must precede M4
(running `sync`). Until M2, workspaces keep working on their old clones.

---

## 8. Dev Workflow (post-migration)

```
develop in  C:\Users\trant\Documents\ai-harness   (branch → tests → push GitHub)
            │
            ▼
   ai-harness sync   (any workspace, or standalone)
        = git -C <canonical> pull --ff-only   (idempotent)
        + update workspace framework pointer
        + re-attach
            │
            ▼
   all workspaces now on the shared canonical commit
```

- **Testing dev changes:** run the suite in the dev repo (`python
  tools/run_tests.py`, `--smoke`, etc.) — it uses dev code via `sys.path`.
- **Promoting to workspaces:** `git push` then `ai-harness sync`. The dev repo
  and canonical are **separate checkouts** of the same remote; canonical tracks
  `main`, so workspaces always run released `main`, never in-flight dev.

---

## 9. Rollback

The workspace clones were clean GitHub clones, so rollback is cheap:

1. `git clone <repo> <ws>\ai-harness` (restore a workspace clone).
2. `pip install -e <ws>\ai-harness` (retarget the global install back to that clone).
3. `setx HARNESS_FRAMEWORK_DIR` (delete the variable) and restart shells.

No workspace state (`.ai-harness/`, artifacts, research memory) is touched by the
migration, so rollback never risks experiment evidence.

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Chicken-and-egg (new sync needs new framework installed) | Migration stalls | Staged ordering §7 (M2 before M4); workspaces keep old clones until M2. |
| `setx` only affects new shells | Operator confusion ("why no effect?") | Verify in a fresh shell; document the restart. |
| Global `pip install -e` retarget is machine-wide | Affects other tools using `harness` | Intended; it is the point. Rollback §9 is one `pip install -e`. |
| Dev repo + canonical both on `sys.path` in some invocation | Wrong code loaded | `HARNESS_FRAMEWORK_DIR` is authoritative for sync/attach; toolchain pins to it. Dev tests set `sys.path` explicitly. |
| A stray `harness/` in `cwd` shadows canonical | Drift | PR #74 circuit breaker retained as last line of defense; diagnostic names the cwd. |
| Pointer/env mismatch (workspace records a different commit than the live install) | Confusing provenance | `status` surfaces the mismatch as an advisory; `sync` reconciles it. |

---

## 11. Verification / Test Plan

**Unit (in the dev repo, on the branch):**
- `find_framework_root` returns canonical when `HARNESS_FRAMEWORK_DIR` is set;
  falls back to the workspace clone; returns `None` when neither.
- `ensure_pinned_toolchain` pins to canonical (depth 1) and still fails open at
  depth ≥ 1 (existing PR #74 tests must keep passing).
- `run_sync` does **not** invoke `pip install` when a canonical repo is present;
  writes the pointer; reports `canonical` state.
- `attach_project` writes a valid `framework` pointer; `status` reads it back.
- `status_report.single_install` flag is correct in both set/unset cases.

**Integration (post-migration, live):**
- `python -c "import harness; print(harness.__file__)"` → canonical.
- `ai-harness status` in each of the 5 workspaces → healthy, **no re-exec line**,
  `single_install: true`.
- `ai-harness validate-task` in aimo → `ok: true`.
- One `--smoke`-tier test run (`python tools/run_tests.py --smoke`) from a
  workspace shell → passes, fast.
- One dispatched probe (aimo, Kaggle) → remote kernel bundles harness from
  canonical; job completes; audit gate passes.

**Regression guard:**
- The full suite (`python tools/run_tests.py`) passes in the dev repo before the
  branch is merged; PR #74's loop-breaker tests remain green.

---

## 12. Open Questions / Future Work

- **Per-workspace version pinning (D2 = "no").** If a future need arises for two
  workspaces to run different framework versions simultaneously, the single
  shared install is insufficient. The escape hatch is per-workspace venvs or
  on-demand pinned checkouts — deferred; not designed here.
- **Canonical auto-update policy.** Should `sync` auto-`pull`, or should
  canonical be updated only by an explicit `ai-harness sync-framework`? Current
  plan: `sync` pulls (idempotent); a separate `sync-framework` can be added later
  if "pull only on demand" is preferred.
- **CI pinning.** Independent of this change: `uv.lock`/constraints for
  reproducible dependencies (tracked separately).
