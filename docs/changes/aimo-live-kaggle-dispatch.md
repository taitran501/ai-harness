# AIMO First Live Kaggle Dispatch + Remote-Dispatch Hardening (PR #64, #65, #66)

- **Date**: 2026-10-16
- **PR**: [#64](https://github.com/taitran501/ai-harness/pull/64) · [#65](https://github.com/taitran501/ai-harness/pull/65) · [#66](https://github.com/taitran501/ai-harness/pull/66)
- **Status**: merged (squash, all 3)
- **Area**: cloud dispatch / audit gate / research memory

## 1. Why (motivation + thought)

Implement the AIMO Small Models Track baseline through the full harness flow
(locked Kaggle backend), and fix every framework bug the live dispatch
surfaced. The first real cloud dispatch is worth ten unit tests — it surfaced
five distinct failure modes, four of which were Windows-specific or only
visible under the real provider state machine.

## 2. What (concrete changes)

- **PR #64 — remote dispatch hardening** (`cloud_job_runner.py`, `runner.py`,
  `cloud_runner.py`, `kaggle_inspect.py`, `hygiene.py`, remote templates):
  1. *CRITICAL*: `_pid_running` used `os.kill(pid, 0)` — CPython maps every
     Windows signal (incl. 0) to `TerminateProcess`, so status polling killed
     the live worker (reproduced: exit 3221225794). Now `OpenProcess` +
     `GetExitCodeProcess` on Windows.
  2. *Nested dispatch*: the runner inside a Kaggle/Modal kernel re-applied the
     locked cloud backend and tried to push from inside the kernel. Remote
     templates set `AI_HARNESS_REMOTE_KERNEL=1`; the runner forces local under
     it; `_prepare_remote_run` refuses re-entry.
  3. *Dead workers stuck RUNNING*: `get_job` reconciles dead pid + no terminal
     state → FAILED with an actionable message.
  4. *`kaggle status/log --exp` 404*: the job-state `remote_id` (with the
     job-hash suffix) is now the slug source of truth.
  5. A constant `fold` column error is now actionable (AIMO's v2 data ships
     one); per-workspace hygiene allow-lists via `task.hygiene.allowed_top_dirs`.
- **PR #65 — audit gate OOF support** (`tools/audit_gate.py`): the baseline's
  `predictions.csv` is an OOF table (one row per id × fold × input_variant), so
  `id` repeats by design; the integrity check rejected the verified run.
  Uniqueness is now checked over the composite row identity when
  `fold`/`input_variant`/`split` columns exist; plain tables keep the strict
  check. `compute_diff_metrics` gets the same composite identity.
- **PR #66 — research memory scope stability** (`harness/research_memory.py`):
  editing `configs/task.yaml` changed the scope key (full metadata digest) and
  silently orphaned every previously recorded insight/decision/checkpoint. The
  scope key is now `digest(task_id + scorer_sha256)` (stable across benign
  edits, changes with identity/scorer); `scope_history` is persisted per task;
  legacy files are adopted once on the next append; cross-task isolation
  preserved (per-task map, never a shared flat list).

## 3. Approach (decisions + rejected alternatives)

Fix at the boundary where the bug lives: Windows has no safe pid liveness probe
via the Python API (→ `OpenProcess`/`GetExitCodeProcess`); re-entrancy in
self-dispatching harnesses needs an explicit "I am inside the sandbox" flag;
and any identity hash used for continuity must exclude benign config churn
(task id + scorer, not full metadata).

## 4. Evidence

- 11 new dispatch tests + 4 audit tests + 4 scope tests, all pass.
- **Live E2E in aimo**: baseline `exp001_baseline_tfidf-lr-clean-v2` dispatched
  to Kaggle (embedded bundle + internet); the kernel trained 5-fold GroupKFold
  on train-small-v2 (43 rows); artifacts imported with identity verified=True;
  re-audited PASS after #65 (identical candidate SHA); audit receipt persisted
  to the job record. Result: problem_text acc=0.721 / macro_f1=0.488. Research
  continuity restored (3 insights + decision + checkpoint, verified runs=1).

## 5. Impact & follow-ups

- **Follow-up**: H1 bounded probe in aimo (local Qwen3.5-4B, 12 problems × 3
  perturbations, answer-consistency AUC vs the 0.721 lexical baseline).
- **Downstream**: the first live cloud dispatch is the reference for all
  remote-run correctness (pid liveness, re-entrancy, OOF audit identity, stable
  research scope).
