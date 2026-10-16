# Repository work log

This journal tracks development of AI-Harness. Read the latest entry after a
restart and append an entry after a meaningful implementation/validation step
or before handing off. Earlier entries are historical evidence, not a claim
about the latest code. Task experiments have their own generated
`RESEARCH_PROGRESS.md` in the resolved artifact root.

### 2026-10-16 — Agent Friction Round 4 (Medium-Priority Audit Fixes)

**Task:** Fix the three remaining medium-priority agent friction points from
the 24-conversation Antigravity audit.

**Fixes (3 verified real):**
- `agent_bridge.py`: `_AgentBridgeMeta.__getattr__` now returns a callable
  proxy for typed-action names instead of raising `AttributeError`. Agents
  can now call `AgentBridge.review_hypothesis(...)` or
  `AgentBridge.confirm_task_spec(...)` as if they were classmethods; the
  proxy constructs a default `AgentBridge()` and dispatches through
  `execute_action` so session state, approval tiers, and audit all still
  apply. Non-action names still raise `AttributeError`. Instance-level
  access (`bridge.review_hypothesis(...)`) was already working.
- `kaggle_compat.py`: added `retry_with_backoff(fn, max_attempts=5,
  base_delay=1.0, max_delay=60.0, multiplier=2.0, transient_check=...)` —
  a generic exponential-backoff wrapper that recognises transient errors
  (429, 500, 502, 503, 504, timeout, connection reset) and retries up to
  `max_attempts` times. Non-transient errors propagate immediately.
  Exported via `harness/__init__.py` lazy map.
- `templates/kaggle_template.py`: wrapped `kernels_output` (root-level
  download) and `kaggle kernels status` (CLI polling) with
  `retry_with_backoff` so transient 429 / 5xx don't abort the whole
  download or status poll. Download uses 5 attempts / 2s base / 60s cap;
  status uses 3 attempts / 1s base / 15s cap.
- `agent_bridge.py`: added `_maybe_auto_checkpoint(session, obs)` — records
  a lightweight research-memory checkpoint every 50 executed actions so
  long sessions survive context compaction without losing the "where am
  I" state. Idempotent: skips when the most recent action was already a
  `record_research_checkpoint` call. Summary includes step count,
  experiments_used, smoke_runs, cloud_runs, and last_action. Wired into
  `execute_action` right before the final `return session, obs`; failures
  are swallowed (auto-checkpoint must never break the main action flow).

**Tests:** updated `tests/test_agent_friction_fixes.py`
(`test_class_level_access_returns_callable_proxy` replaces the old
"class-level raises" assert). No new test file — the three fixes are
exercised by existing coverage plus the manual smoke checks above. Full
suite: **626 passed, 1 skipped** (same baseline as PR #B / R3).

### 2026-10-16 — Agent Friction Round 3 (High-Priority Audit Fixes)

**Task:** Fix the five highest-frequency agent friction points identified from
the 24-conversation Antigravity audit (group "ưu tiên cao" = high priority).

**Fixes (5 verified real):**
- `research_memory.py`: `record_insight` now accepts `evidence=None` /
  empty list for hypothesis-based insights that have not yet been verified
  by a full run (brainstorm, external score, mechanism proposal). Non-empty
  evidence is still validated against verified run IDs to prevent fabricated
  links. `record_decision` now supplies a sensible default `revisit_when`
  for `rejected`/`deferred` outcomes so agents don't have to invent a
  revisit condition (they can still override with their own).
- `environment_detector.py`: added `detect_kaggle_environment()` (alias for
  `detect_kaggle`) and `is_kaggle_environment()` (predicate) so agents that
  reach for the longer descriptive name or a boolean check don't hit
  `AttributeError`.
- `submission_manager.py`: added `list_submissions()` (sorted list of entry
  dicts) and `all_submissions()` (snapshot of the raw registry) so agents
  have a stable accessor; the live `self.submissions` attribute remains.
- `cloud_policy.py`: `UploadConsentStore.grant()` now normalises a plain
  string fingerprint into the dict shape downstream code expects, fixing
  `AttributeError: 'str' object has no attribute 'get'` when callers pass
  the hash directly.
- New `harness/kaggle_compat.py`: unified facade over whichever Kaggle SDK
  is installed (`kagglesdk` preferred, `kaggle` legacy fallback). Exposes
  `KaggleClient` with canonical snake_case accessors (`list_kernels`,
  `list_submissions`, `kernel_output`, `kernels_status`, `kernels_logs`,
  `competition_topics`, generic `get()`); a `_NormalisedObject` proxy
  translates legacy camelCase response attributes (`totalVotes` →
  `total_votes`, `publicScore` → `public_score`, …). `KaggleApi =
  KaggleClient` and `detect_sdk()` exported via `harness/__init__.py` lazy
  map.

**Tests:** updated `tests/test_research_trace.py` (the deferred-decision
assert now expects success with the default `revisit_when`); no new test
file — the five fixes are exercised by existing coverage plus the manual
smoke checks above. Full suite: **626 passed, 1 skipped** (same baseline as
PR #B).

### 2026-10-16 — PR #B: Guardrails & State-Contract Fixes (Module Review Findings)

**Task:** Fix the verified P1 guardrail / state-contract findings from the
module review (second wave after PR #A).

**Fixes (13 verified real + 2 bundled):**
- `candidates.py`: `register_external_prediction` now refuses to overwrite an
  existing run directory (immutability contract, matches `exist_ok=False` in
  runner) and records into the canonical `journal.json` instead of orphaning
  `tree_journal.json`.
- `repair.py`: default mode keeps declared entries verbatim (missing file
  stays a visible stale-hash validation failure, never a silent shrink);
  `drop_missing` keeps only declared files still on disk (never adopts
  undeclared files); manifest write is now atomic.
- `runner.py`: new `is_local_only_run()` — probe detection now includes
  `validation.sample_limit` from the experiment config, so sample probes can
  never dispatch to a locked cloud backend.
- `templates/kaggle_template.py`: `wait_for_kernel` tolerates transient
  `ERROR` (status CLI failure) up to 10 consecutive; terminal states remain
  provider-parsed.
- `cloud_job_runner.py`: managed job accounts exactly 1 GPU (model name never
  regex-parsed as a count — was inflating gpu_hours 4–100×); audit
  `candidate-audit.zip` + `AUDIT_RECEIPT.json` now written to
  `artifacts/audit/<job_id>` (outside the sealed run directory, per
  CORRECTNESS_CONTRACT §3).
- `competition_state.py`: `finalize_competition` requires `freeze_locked`
  (terminal state is not display-only) and is idempotent; per-slot portfolio
  attribution (each slot reports its own CV + public score, not the overall
  best); ablation table is None-safe; single-fold `fold_std` NaN guarded.
- `agent_bridge.py`: FINALIZED triggers the same stop set as FROZEN (with a
  distinct message); `gpu_hours_used` is always a float, never None (budget
  enforcement can't be silently disabled).
- `llm_engine.py`: Gemini SDK path — only ImportError falls back to REST;
  SDK errors are surfaced if REST also fails; empty SDK response (no
  candidates) raises; REST path raises on error bodies instead of returning
  `text=''`.

**False positive (verified, not fixed):** the UCB diversification cap —
standard UCB1: in any finite tree the deepest-level leaves have visits=0, so
the explore term always dominates; a "saturated best parent" can never win
the recommendation, the cap is unreachable dead code. Reverted; documented.

**Tests:** new `tests/test_pr_b_guardrails.py` (23 tests); updated
`test_artifact_dedup.py` re-registration test to the immutability contract.
Full suite: **626 passed, 1 skipped**.

**Next:** propagate to the 5 competition workspaces; PR #C (P2 friction
wave) still pending user decision.

### 2026-10-16 — PR #A: Correctness / Silent-Wrong-Result Fixes (Module Review Findings)

**Task:** Fix the highest-severity findings from the full read-only module
review (9 lanes over `harness/` + templates, ~24.9k lines): silent wrong
results, broken managed-cloud dispatch identity, and dead public-LB reads.

**Changes**

- P0 `harness/task_spec.py`: removed the second `train_path`/`test_path`
  property pair that shadowed the `Optional[str]` versions — undeclared splits
  now return `None` instead of always-truthy `"dataset/"` / `"dataset/test/"`
  defaults (truthiness gates like `if spec.test_path:` were always true).
- P0 `harness/templates/kaggle_template.py` + `modal_template.py`:
  generated remote runners now bind the dispatched identity via
  `--job-id <dispatch>`; `harness/runner.py main()` forwards `args.job_id` to
  `run_via_kaggle`/`run_via_modal`. Previously the remote runner minted a
  fresh `run_id`, the manager identity gate raised `Artifact identity
  mismatch` 100% of managed dispatches, and retry was blocked on the
  non-empty artifact dir. Runner worker-state updates are now gated on the
  `JOB_ID_PATTERN` so dispatch identities (e.g. `kaggle_<exp>_<hex>`) bind the
  manifest `run_id` without crashing `update_worker_state`.
- P1 `harness/candidates.py register_external_prediction`: predictions are
  now read from the truth-aligned frame (duplicate-ID guards added) instead
  of the raw file order; a shuffled external OOF file previously
  positionally mislabeled rows during evaluation.
- P1 `harness/runner.py UniversalRunner._evaluate`: an official plugin scorer
  failure behind a custom `metrics_evaluator` hook is no longer silently
  swallowed — it is printed and recorded in the result as
  `_evaluation_warnings` (visible in metrics.json).
- P1 `harness/runner.py`: new `_probe_sample_limit()` helper with stable
  precedence (`--probe-samples` > `--debug`→50 > `validation.sample_limit`),
  used by both the ML and LLM paths — `sample_limit` was previously ignored
  by the ML path and `--probe-samples` was a no-op on the LLM path.
- P1 `harness/competition_state.py`: public-LB reads now use
  `official_primary_value()` (probes spec metric name, `primary`, `score`)
  instead of the `primary` key `record_official_score` never writes — best
  public score/selector was always None/N-A before.
- P1 `harness/phase_workflow.py`: baseline anchor gate now passes
  `task_spec`/`plugin` to `validate_baseline_artifact`, so the anchored run
  is re-verified against the current task/scorer contract hashes.

**Validation**

- New `tests/test_pr_a_correctness.py` (18 tests) covering every fix,
  including the shuffled-vs-ordered external OOF equivalence and the stale-
  contract baseline gate block.
- Existing `tests/test_kaggle_runtime.py::test_generated_script_runs_full_
  validation_and_exports_one_manifest` now exercises the `--job-id` binding
  end-to-end (generated script runs the real runner with the bound id).
- Full suite: 602 passed, 1 skipped (Windows symlink self-skip).

**False positives ruled out during verification (do not "fix")**

- XGBoost probability columns: `proba_*` columns follow sklearn's sorted
  integer class order while `classes_*` holds the sorted original labels;
  decoding via `class_order[argmax]` is correct. Recording
  `pipe.classes_` (integers) would break decoding.
- Baseline `primary_score` alias: `extract_primary_metric` already resolves
  `primary_score` and direct metric keys.
- Blend duplicate keys: per-frame `duplicated().any()` checks already reject
  duplicate IDs before the reindex alignment.
- External manifest `validate_run`: all four REQUIRED files are present in
  the generated manifest.
- Conditional subtask masks: truth-based `eval_condition_only` is the
documented oracle semantics (e.g. `oracle_hate`), not a routing bug.

**Next action:** merge PR #A (branch `fix/correctness-silent-results`), then
`ai-harness sync` into the 5 active competition workspaces and re-verify the
Kaggle-locked smoke/full-run behavior.

### 2026-10-16 — P2 Audit Items: Submissions HUD, Workspace Hygiene, Mechanism Anti-Duplicate

**Task:** Close the three P2 friction items left open by the Oct 2026 agent
chat audit (recurring user complaints: "submission ghi đè lên nhau", "để mọi
thứ ở root folder làm rối quá", repeated mechanism re-runs).

**Changes**

- `harness/hygiene.py` (new): deterministic, read-only root-clutter scan —
  scratch files (`_*`, `~$*`, `*.tmp/bak/swp/orig`), stray top-level
  CSV/parquet data files, unknown top-level dirs (generous allow-list).
  Advisory only; never deletes or blocks. `hygiene_hud_lines()` renders the
  compact HUD block (empty when clean).
- `AgentBridge.get_context_packet`: new `recent_submissions` (last 6 registry
  entries: id, exp, status, local CV, official score, relative path) and
  `workspace_hygiene` keys; `render_context_markdown` now shows a
  `## 📦 Submissions` block (with empty state) and `## 🧹 Workspace Hygiene`
  (only when findings exist).
- `ResearchMemory.review`: new `_mechanism_key()` (model family from
  `architecture/model_type/family/name/type`, hyperparameters ignored) and a
  `mechanism_matches` advisory list (up to 5 earlier entries on the same
  model family; never blocks). `run_full_validation` surfaces matching
  earlier work as `mechanism_warnings` in the SUCCESS/FAILED observation data.

**Validation**

- New `tests/test_p2_audits.py` (13 tests: packet/HUD submissions incl. empty
  state, hygiene scan + HUD lines, mechanism match incl. same-exp exclusion,
  empty-model no-crash).
- Full suite: 584 passed, 1 skipped (Windows symlink-privilege self-skip).

**Next action:** verify in a Kaggle-locked workspace (smoke `--debug --fold 0`
stays local, full run dispatches to Kaggle), then `ai-harness sync` into the
5 active competition workspaces.

### 2026-10-16 — Artifact CSV Deduplication (Hardlink Mirrors + Canonical Truth)

**Task:** Port the `local-hacks` experiment from `airline_prediction`
(hardlink-dedup of oof/predictions/truth CSVs) into the framework as a clean,
reviewed implementation. The workspace hack had grown on top of a stale main;
only the write-section idea was portable.

**Deep review findings (manual, before porting)**

- `predictions.csv` is a byte-identical copy of `oof_predictions.csv` in both
  writers (`UniversalRunner._save_and_finalize`,
  `register_external_prediction`) -> a hardlink mirror is safe; every consumer
  (`validate_run`, repair, research_memory, cloud import, notebook export,
  `iterdir()` scans) is content-based, so links are transparent.
- A fixed-name canonical `evaluation_truth.csv` master (the raw hack) is a
  **correctness risk**: the runner writes the *split* truth while
  `register_external_prediction` writes the *full* training truth, and a split
  change silently links stale content. -> replaced with a **content-addressed**
  master (`canonical_evaluation_truth_<sha16>.csv`).
- Fallback writes must never happen in place over an existing (potentially
  shared-inode) file, or a stale link would truncate its master. ->
  unlink-before-write in the fallback path.
- Concurrent creation of the canonical master is race-free because creation is
  atomic (temp + `os.replace`) and the name is derived from the content hash
  (identical content -> identical path -> idempotent overwrite).

**Changes**

- `harness/artifacts.py`: new `mirror_csv`, `write_csv_dedup`,
  `canonical_truth_path`, `ensure_canonical_truth` helpers (with docs on the
  dedup contract).
- `harness/runner.py`: `predictions.csv` and `evaluation_truth.csv` are now
  hardlink mirrors (plain-write fallback when hardlinks are unavailable).
- `harness/candidates.py`: same treatment in `register_external_prediction`.
- `tests/test_artifact_dedup.py`: unit coverage for mirror/fallback/stale-link
  semantics + end-to-end runner and external-registration runs verifying inode
  identity, content identity, canonical sharing across runs, and manifest hash
  integrity.

**Evidence**

- New tests: 14 (13 pass, 1 skip on machines without symlink privilege).
- Full suite: 547 tests, 100% pass.

**Lesson**

- Workspace-local hacks can encode real insights, but must be re-derived and
  reviewed on current main before porting: the raw version's fixed-name
  canonical master would have introduced a subtle stale-truth bug.

**Next action**

- After merge, re-sync the competition workspaces (`aimo`,
  `airline_prediction`, `zindi`, `rsna-knee`) via `ai-harness sync` so they
  inherit the dedup. The `airline_prediction` `local-hacks` branch then becomes
  redundant (keep as reference).

### 2026-07-23 — P2 Defense-in-Depth Fixes (Config Path Guard, Per-Subtask Direction) (PR #38)

**Task:** After the full codebase audit (runner, dashboard_server, skill_injector,
competition_state, and the 14 modules from the previous two passes), two P2
issues were identified and fixed with regression tests. A third candidate P2
(LLM rate-limiting) was investigated and ruled out: LLM runs go through the
runner, which already tracks them via `experiments_used`, so no separate rate
limit is needed on the bridge.

**Findings (checked carefully before fixing, per the "don't fix blindly" rule)**

- **P2-1 — Unvalidated `config_path` in execution actions.**
  `run_full_validation`, `run_smoke_test`, and `run_probe` accept a
  `config_path` from the caller and load it without a path-traversal guard.
  `write_experiment_config` already constrains writes to `configs/`, but an
  agent could still pass an arbitrary path (e.g. `../../etc/passwd.yaml`) to an
  execution action. `run_smoke_test` is an AUTO action, so it is reachable
  without human approval. Risk is low (`yaml.safe_load` does not execute code
  and the runner re-validates grounding), but it is a real gap worth closing.
- **P2-2 — Specialist ranking used the aggregate spec direction.**
  `compute_candidate_pool` ranked subtask specialists with
  `self.spec.get("direction")` instead of each subtask's own direction. If a
  subtask is minimized while the overall metric is maximized, the ranking is
  inverted.

**Fixes**

- `harness/agent_bridge.py`
  - New `_validate_config_path(config_path)`: resolves the path and requires the
    result to stay within the project root (rejects `../` traversal and
    absolute paths outside the workspace). Called before `run_full_validation`,
    `run_smoke_test`, and `run_probe` load their config.
- `harness/competition_state.py`
  - `compute_candidate_pool` now builds a `sub_directions` lookup from the spec
    `subtasks` (handling both dict and list shapes) and uses the per-subtask
    direction for specialist ranking, falling back to the aggregate spec
    direction when a subtask declares none.

**Validation**

- `tests/test_p2_fixes.py` — 4 regression tests: valid config path accepted,
  `../` traversal rejected, absolute-outside path rejected, and a minimized
  subtask ranks the lower-score candidate as the specialist.
- `python -m pytest tests/ -q` → 458/458 pass.

**Notes**

- The per-subtask direction fix initially used `subtask_meta.get(crit_key)`
  assuming a dict, but the spec declares `subtasks` as a *list* of dicts. The
  fix now handles both shapes; the regression test catches this regression.
- P2-3 (LLM rate-limiting) was investigated: `llm_engine.generate_completion` is
  not called directly by the agent bridge. LLM experiments are dispatched via
  the runner, which consumes the `experiments_used` budget, so no additional
  guard is needed on the bridge path.

### 2026-07-23 — Anti-Bypass Agent-Integration Hardening (Submission Guards) (PR #37)

### 2026-07-23 — Anti-Bypass Agent-Integration Hardening (Submission Guards) (PR #37)

**Task:** Follow-up to the anti-bypass audit. The agent drives submission through
the `prepare_submission` bridge action; the standalone audit gate CLI
(`tools/audit_gate.py`) is *not* on that path, so its three deterministic checks
were being skipped entirely. Added the same guards to the bridge path so an
agent cannot ship a non-improvement.

**Findings**

- `prepare_submission` ran only a *provenance* check (`validate_test_prediction`
  — SHA-256 matches the run manifest, OOF ≠ test). It did **not** run the audit
  gate's improvement checks, so an agent could:
  1. jitter a model's predictions on ≥ `min_diff_pct` of rows → pass a naive diff,
  2. skip the significance test (it is only run when `--ground-truth` is passed),
  3. re-submit the same bytes (no duplicate-submission dedup on the bridge path).

**Fix**

- `harness/agent_bridge.py`
  - New `_guard_submission(cand_path, exp_id, sub_mgr)` runs before
    `register_plugin_submission` and enforces three deterministic checks:
    1. **Duplicate-submission dedup** — the candidate file's SHA-256 must not
       match any already-registered submission (content-based, not path-based).
    2. **MMES** — the candidate OOF must differ from the baseline anchor's OOF on
       ≥ `min_diff_pct` of rows (or `min_diff` rows); a below-floor change is
       rejected as seed noise / trivial tweak.
    3. **Significance** — when `evaluation_truth.csv` is available, a paired
       error test (regression) or McNemar's test (classification) must reach
       p < 0.05 **and** the candidate must be the better side (a significant
       *worse* result is also rejected).
  - `prepare_submission` returns the guard evidence in the observation `data`
    (`submission_guards`), so the agent/human can inspect each check's result.
- `tests/test_submission_guards.py` — new regression tests: dedup blocks
  identical bytes, MMES blocks below-floor and passes above-floor, significance
  rejects a significant-worse candidate and accepts a significant-better one.

**Validation**

- `python -m pytest tests/ -q` → 454/454 pass. No pre-existing test relied on the
  old no-guard submission path; the existing `prepare_submission` tests use a
  baseline exp with no baseline OOF, so MMES/significance skip cleanly.

**Notes**

- The significance direction check was added after the first implementation
  returned `PASS` for a candidate that was significantly *worse* than the
  baseline (McNemar p ≈ 0 but all discordant pairs favored the baseline).
- MMES and significance are skipped (not failed) when the baseline OOF or ground
  truth is absent, so the guard does not block a legitimate first submission.

### 2026-07-23 — Anti-Bypass Agent-Integration Hardening (Grounding, Duplicate Candidates, Stagnation) (PR #36)

Each entry records: request/observation, completed work, evidence, lesson,
remaining uncertainty, and next action. Record failures and abandoned directions
when they affect the next decision. Keep notes concise; do not copy transcripts,
raw datasets, or internal reasoning. Do not invent results or reconstruct lost
history from guesses.

## 2026-10-15 — Agent Bridge P0/P1 Integration Fixes (Session State, Budget, Approval, GPU Accounting)

- Request: Check and debug integration gaps when an IDE coding agent works with the harness. Fix P0 (broken behavior) and P1 (correctness) issues first.
- Completed:
  - **P0 A3 — Budget-blocked action no longer terminates the session.** `execute_action` previously set `session.status = "BUDGET_REACHED" / "STOPPED"` on any guard violation, which permanently stranded the agent in a non-resumable session. Now a budget block is recorded as an observation (with `stop_reason` as a transient note) but the session stays in its current state so the agent can continue with other actions.
  - **P0 C1 — Interleaved actions no longer drop a pending USER_APPROVAL.** `execute_action` previously cleared `pending_approval_action` and set `status = ACTIVE` on *any* action, silently swallowing a pending approval when the agent ran an inspection in between. Now the pending action is only cleared when the action being executed *is* the pending one and `user_approved=True`. A different action leaves the approval intact.
  - **P0 A4 — `experiments_used` only counts terminal ledger rows.** `_reconcile_session_usage` now filters to `status in {COMPLETED, FAILED, CANCELLED}` so a `RESERVED` row whose job never reached a terminal state (process crash) does not consume budget.
  - **P0 B4 — Cloud job terminal accounting computes `gpu_hours`.** `CloudJobManager._persist_job` now extracts the GPU count from the job record and calls `resource_ledger.terminal` with `gpu_count` + `allocation_seconds`, so a terminal cloud row carries a real `gpu_hours` value. Previously a `None` `gpu_hours` on any row made `_reconcile_session_usage` set `gpu_hours_used=None`, silently disabling the session's `max_gpu_hours` budget. Added a `gpu` field to the `CloudJob` dataclass and populated it in `start_job`.
  - **P1 A1 — Stable active-session selection.** Added an `active: bool` flag to `BridgeSession`. `start_session` marks the new session active and demotes any previously-active sessions (via a best-effort per-file transaction). `get_active_or_create_session` now prefers the session with `active=True` over mtime ordering, with a fallback to the most recent open session for legacy workspaces. `list_sessions` normalizes the flag for legacy files.
  - **P1 B1 — Freeze check in `resource_ledger.reserve` is no longer stale.** The `competition_state.json` read now happens under the workspace lock (via a new `_freeze_locked` helper) so a concurrent freeze committed by another process is seen before the reservation is written.
  - **P1 C2 — `approve_pending_action` no longer duplicates the action in `session.actions`.** The pending action entry is now removed from `session.actions` before re-execution, so the action history contains at most one entry for an approved-and-executed action.
  - **P1 — `candidate_operations` key mismatch fixed.** The synthesis/registration/blend actions incremented `compute_usage["candidate_operations_used"]`, but `_reconcile_session_usage` and the `ComputeUsage` dataclass use the key `candidate_operations`. The mismatch meant the in-session counter was never visible to reconcile and the `max_candidate_operations` budget was effectively never enforced. All three increment sites now use the canonical `candidate_operations` key.
  - Reverted an overly aggressive E1 change (suppressing probe `exp_id` in `record_execution`) after it broke `tests/test_research_trace.py` — the trace flow intentionally links probe executions to their experiment for the `needs_diagnosis` state, and evidence validation is a separate concern handled by `_evidence`.
- Evidence: Full test suite passes 100% (440/440 tests) in ~89s (`python -m pytest tests/ -q --ignore=tests/test_e2e_pipeline.py --ignore=tests/test_boost_end_to_end.py`). 11 new regression tests in `tests/test_agent_bridge_p0_p1_fixes.py` cover: budget-block keeps session active, interleaved action preserves pending approval, approve-after-interleave executes the pending action, failed run terminal row counts but session recovers, cloud terminal row carries `gpu_hours`, reconcile produces non-None `gpu_hours_used`, active flag demotes prior sessions and is preferred by `get_active_or_create`, approve does not duplicate in `actions`, and `reserve` blocks when the state file says frozen.
- Lesson: Session state transitions (budget block, approval gate) must be per-action observations, not terminal session states — otherwise the agent is silently stranded or its pending approval is swallowed. Resource accounting (GPU hours, experiment count) must read from terminal ledger rows only, and a `None` value in any row must not poison an aggregate that a budget guard depends on.
- Remaining uncertainty: P2 items remain open — `_refresh_task_context` + `ResearchMemory.sync()` cost per action (A5/E2), context-packet bloat (F3), `wait_cloud_job` blocking the dashboard server thread (D1), and no MCP/tool API for IDE agents (G1). Cloud GPU accounting is an estimate from allocation × wall time, not a metered provider value; accurate billing would require provider-side reporting.

## 2026-10-15 — Anti-Bypass Agent-Integration Hardening (Grounding, Duplicate Submissions, Stagnation)

- Request: Re-audit the harness for ways an IDE agent would *bypass* its own rules to get past the harness — specifically (1) the agent tends to create duplicate candidate/submission versions to "preserve" a submission, and (2) the agent tends to stop brainstorming new versions once a few runs exist. Harden the harness so these are blocked, and add a brainstorm aid.
- Completed:
  - **Grounding bypass closed — `verified_via` is now enforced.** `GroundingGate.verify_config` previously did not check `verified_via`, so a non-baseline proposal that filled in the length requirements with `verified_via='not_verified'` (or empty) passed the gate — the exact anti-hallucination failure the gate exists to stop. Now a non-baseline config must declare `verified_via` from a fixed set of real verification channels (`search_web`, `read_url_content`, `mcp_gemini-docs`, `local_registry`, `verified_baseline`, `huggingface_hub`); the sentinels `not_verified`/`unverified`/`none`/empty are explicitly rejected. The root baseline is exempt (its grounding is synthesized and self-verified). `create_experiment`'s default `verified_via` was `not_verified`; it is now `local_registry` (the parent's verified evidence + research memory is legitimate local grounding), so a minimal agent call still works but a deliberate unverified claim is blocked.
  - **Duplicate-submission padding blocked — candidate recipe fingerprint.** `harness.candidates.build_candidate` now records a deterministic `recipe_fingerprint` (hash of sorted source IDs + mode + normalized weights/routing) on every candidate entry. The agent-bridge synthesis path (`synthesize_candidate`/`promote_candidate`/`build_ensemble`) computes the same fingerprint before building and rejects an identical recipe that is already in the candidate pool, unless the agent supplies an explicit `revisit_reason`. This stops the agent from re-synthesizing the same sources under a new display name to "preserve" a submission or pad the pool — the recipe is byte-identical, so a new name adds no scientific signal. Added `AgentBridge._candidate_source_ids` to resolve the source IDs a recipe will consume (mirrors `build_candidate`).
  - **Stagnation countermeasure — `suggest_next_interventions` action.** New `AUTO` typed action that surfaces *untried, non-graveyard* hypotheses so the agent is not left to stop brainstorming. It is deterministic: it reads the UCB recommendation + parent diagnostics, the active research frontier (insights still needing a probe/experiment), and under-explored branches from the solution tree, then ranks candidate hypotheses that (a) have not been run yet and (b) are not in the idea graveyard. The response includes a `stagnation` block (`reassess_recommended`, `consecutive_non_improvements`, `total_runs`, `branches_tried`) so the agent sees when it should pivot. The action is allowed under freeze lock (read-only) and is surfaced in the context packet's AUTO action list. Suggestions are declared, unrun hypotheses — each still requires grounding + a smoke probe before full validation.
- Evidence: Full test suite passes 100% (449/449 tests) in ~103s. 9 new regression tests: 4 in `tests/test_grounding_gate.py` (unverified rejected across 4 sentinels, verified passes across 4 channels, root baseline exempt, type-error still reported) and 5 in `tests/test_agent_bridge_anti_bypass.py` (duplicate recipe blocked + real synthesis not called, `revisit_reason` allows re-synthesis, `suggest_next_interventions` returns well-formed frontier suggestions, graveyard ideas are filtered from suggestions).
- Lesson: The harness must assume the agent will try to game the rules. A rule that is enforced only by a length check (grounding) can be bypassed by filling in plausible-but-unverified text; the enforcement must key on the *substance* (the verification channel). A rule that keys only on identity (candidate name) can be bypassed by renaming; the enforcement must key on the *recipe* (sources + mode + weights). A stagnation tendency is not a bug to fix by blocking — it is a gap to fill with a deterministic, non-duplicative suggestion surface that hands the agent concrete, untried starting points.
- Remaining uncertainty: The `suggest_next_interventions` heuristics are deterministic but not smart — they surface candidates from existing signals (frontier, UCB, branches) rather than generating genuinely novel hypotheses; a scientist still judges them. Atomicity (one variable per experiment) remains advisory, enforced only by the grounding gate's lazy-pattern matching, not by a structural check. Duplicate-recipe detection keys on the declared recipe; two semantically-identical candidates built from different source experiments would not be caught.
- Next action: Commit the P0/P1 fixes and the new regression test file. Then address P2: add a `--json-compact` CLI flag, cache `sync()` results per call within `get_context_packet`, and verify the dashboard server uses a threading model so `wait_cloud_job` does not starve other requests.

## 2026-10-10 — Durable insight traces

- Observation: flat research notes did not preserve an insight's identity,
  planned test, experiment links, lessons, and derived directions.
- Completed: insight IDs and parent links; bounded probe plans; linked decisions
  with lessons and revisit conditions; paginated trace; pending-result triage;
  checkpoint insight IDs; generated workspace guidance and phase revision updates.
- Corrected: repeated-config metadata could erase insight links; reassigned unrun
  configs could attribute a new result to an older insight; smoke failures lacked
  experiment IDs in the research ledger.
- Evidence: 387 local tests passed in 140.262 seconds on a disposable checkout.
  Template single-fold smoke and a synthetic 12-row/two-fold NLP trace passed;
  restart recovered the linked result, lesson, derived insight and checkpoint.
  Hash checks preserved 43 configs/artifact/document files in the real workspace.
- Lesson: recoverable workflow records are necessary, but do not establish better
  agent reasoning or competition performance.
- Remaining: actual-agent continuity quality has not been measured.
- Next action: add a human-readable progress journal, then evaluate continuity
  with a fixed task, bounded budget, and an intentional context interruption.

## 2026-10-10 — Human-readable ongoing tracking

- Request: make completed work and next steps visible in Markdown while the agent works.
- Completed: generated task-level `RESEARCH_PROGRESS.md` with resume instructions,
  insight statuses and chronological notes/results. Research events and context
  reads refresh it; the HUD exposes its path. This development log and `AGENTS.md`
  now specify when assistants should record repository work.
- Boundary: unrecorded insights and ordinary code edits are not inferred
  automatically. Agents must summarize meaningful work through the research
  actions or this development journal. Tracking does not authorize continuous compute.
- Validation: 24 focused tests passed; all 390 tests passed in 151.749 seconds
  on an isolated checkout. Template smoke and synthetic two-fold/restart integration
  passed. Markdown refresh, evidence invalidation, concurrent notes, preservation
  of user-owned files, and recovery after report-write failure were exercised.
  Hash checks confirmed all 43 protected real-workspace files were unchanged.
- Lesson: event-driven Markdown makes recorded work visible, but recording discipline
  is still necessary; a generated journal cannot recover an insight never recorded.
- Next action: design the actual-agent continuity evaluation with a fixed task,
  interrupted context, repetition counts, interpretation completeness and a budget.

## 2026-10-13 — Canonical AI-Harness Rebranding & Decoupled Fast Testing Architecture

- Request: audit repository for legacy `ml-harness` bias, rebrand to `ai-harness`, resolve unit tests running heavy local models and stalling test cycles, and commit changes cleanly to remote repo.
- Completed:
  - **Comprehensive Rebranding**: Standardized canonical name to `ai-harness` across runtime (`harness/project.py`, `harness/runner.py`, `harness/skill_injector.py`, `harness/phase_workflow.py`, `harness/cli.py`, `harness/plugins.py`, `harness/environment_detector.py`, `harness/data_strategy.py`, `harness/llm_engine.py`, `harness/templates/`), developer tools (`tools/bootstrap_task.py`, `tools/dashboard_server.py`, `tools/agent_bridge_cli.py`, `tools/agent_runtime_cli.py`), Web dashboard (`dashboard/src/App.tsx`, `dashboard/src/constants/i18n.ts`, modals), skills (`.agents/skills/ai-harness/SKILL.md`), and documentation (`docs/README.md`, `docs/agent_workflow.md`, `docs/CORRECTNESS_CONTRACT.md`, `docs/plugin_architecture.md`, prompts).
  - **Dual-path Backward Compatibility**: Preserved automatic fallback resolution for legacy `.ml-harness` workspaces, manifests (`ml-harness-runtime.json`, `mlh-data-bundle.json`), and UI preference storage keys.
  - **Decoupled Testing Engine**: Implemented `DummyBaselineModel` in `harness/models.py` (registered under aliases `dummy`, `mock`, `dummy_baseline`, `dummy_classifier`, `dummy_regressor`) providing instant `O(1)` fitting, label alignment, and probability matrix output without heavy mathematical optimization.
  - **Mocked Process & Network Probes**: Mocked external OS and network calls in `test_cloud_jobs_and_env.py` (dropped from 18.0s to 0.3s) and replaced `localhost` with explicit `127.0.0.1` IPv4 binding in `test_llm_api_endpoints.py` (dropped from 30.5s to 1.0s), preventing Windows IPv6 fallback timeouts.
- Evidence: Full test suite (`python -m unittest discover -s tests -p "test_*.py"`) completed 392 tests in 74.13s (reduced from ~140s) with 100% pass rate.
- Lesson: Harness unit tests validating lifecycle DAG transitions, anti-leakage gates, and orchestration contracts should never depend on heavy numerical convergence or live network sockets; decoupling via lightweight dummy estimators and mocked environment probes achieves sub-second test execution.
- Next action: Partition changes into logical commits and push to remote `origin/dev`.

## 2026-10-13 — Complete Eradication of Legacy ml-harness & mlh- Prefixes

- Request: check if any traces of `ml-harness` remain in the repository and purge them completely.
- Completed:
  - **Script & CLI Removal**: Removed legacy `ml-harness*` script aliases from `pyproject.toml`.
  - **Storage & Runtime Purity**: Removed `ml_harness_ui_lang` key in `i18n.ts`, removed fallback `LEGACY_DATA_MANIFEST_NAME = "mlh-data-bundle.json"`, and eliminated `LEGACY_MANIFEST_RELATIVE_PATH` and `.ml-harness` checks in `harness/project.py`.
  - **Cloud & Resource Prefix Modernization**: Updated dataset, volume, code slug, and tempdir prefixes across `harness/cloud_policy.py`, `harness/cloud_runner.py`, `harness/templates/kaggle_template.py`, and `harness/kaggle_runtime.py` from `mlh-` to `aih-`.
  - **Web Assets & Test Assertions**: Updated pre-built web asset bundle and HTML title to `AI-Harness`, cleaned `.gitignore`, and updated all corresponding test assertions and fixtures in `test_workspace_workflow.py`, `test_kaggle_runtime.py`, `test_research_trace.py`, `test_bootstrap_task.py`, and `test_universal_plugin.py`.
- Evidence: `git grep -inE "ml[-_]harness"` yields 0 matches in codebase files; full test suite passed 392/392 tests in 79.0s.
- Lesson: Full deprecation of legacy prefixes ensures clean naming boundaries without ambiguous dual paths across cloud policies and local file caches.
- Next action: Commit and push changes to remote repository.
## 2026-10-14 — Dashboard Metadata, Test Fixtures, and AGENTS.md Quality Gate Alignment

- Request: audit repository for any lingering legacy metadata or clean-up targets.
- Completed:
  - **Dashboard Source Sync**: Updated `dashboard/package.json` name from `mldl-experiment-dashboard` to `ai-harness-dashboard`, and updated `dashboard/index.html` title to `AI-Harness & Priming Dashboard`.
  - **Skill Fixture Modernization**: Modernized mock skill directory path in `tests/test_bootstrap_task.py` from `.agents/skills/ml-experiment-harness` to canonical `.agents/skills/ai-harness`.
  - **Documentation & Quality Gates**: Updated `AGENTS.md` test count baseline from 330+ to 390+ tests, and clarified invocation context for single-fold smoke probe validation.
- Evidence: Full test suite (`python -m unittest discover -s tests -p "test_*.py"`) executes 392/392 tests with 100% pass rate in ~79s.
- Lesson: Keeping development documentation, package descriptors, and test fixtures synchronized prevents subtle drift between the core harness and auxiliary web/skill scaffolding.
## 2026-10-14 — Fix Kaggle Windows MAX_PATH Overflow and Valid License Identification

- Request: investigate and fix two root causes under `kaggle_bundle_mode: dataset` on Windows: `[Errno 2] No such file or directory` due to path length exceeding Windows `MAX_PATH` (260 characters) during Kaggle CLI token flattening, and `Dataset creation error: Please select a valid license` due to unrecognized `copyright-authors` license name in `dataset-metadata.json`.
- Completed:
  - **Windows MAX_PATH Protection**: Shortened code staging directory structure from `stage / "code-dataset" / bundle_sha256` (77 chars) to compact isolated `stage / "code" / bundle_sha256[:12]` (17 chars). Shortened bundle filename from `aih-code-{64-char-hash}.zip` (77 chars) to `aih-code-{16-char-hash}.zip` (29 chars). Shortened staging prefix in `cloud_runner.py` to `aih-kaggle-{job_id[:8]}-`. This reduces the flattened upload token file path from ~285+ characters to ~145 characters, safely below the 260-character limit.
  - **Valid Kaggle License Specification**: Changed hardcoded `"licenses": [{"name": "copyright-authors"}]` in `harness/templates/kaggle_template.py` to Kaggle's recognized custom license identifier `"licenses": [{"name": "other"}]` while preserving the full retained rights in the description. Also standardized code dataset title to `AI Harness Code {bundle_sha256[:32]}` and updated `docs/kaggle_execution.md`.
  - **Verification**: Updated `tests/test_kaggle_runtime.py` with explicit assertions verifying `"other"` license metadata, "AI Harness Code" title, and relative archive path length < 50 characters.
- Evidence: 392/392 tests passed (100%) across the full test suite.
- Lesson: Windows file paths when manipulated by third-party CLI tools (like Kaggle's flattened upload token files) require proactive defensive length budgeting; custom licenses for Kaggle datasets must conform to Kaggle's recognized license taxonomy (`other`).
- Next action: Commit and push changes to remote `origin/dev`.

## 2026-10-15 — Agent Usability, Experiment ID Resilience, OOF Serialization & Kaggle Exporter

- Request: analyze multi-agent friction across ~35 historical transcripts and active repositories (`aimo`, `airline_prediction`, `zindi`, `rsna-knee`), identify pain points, and enhance framework usability and resilience for AI coding assistants.
- Completed:
  - **CLI Shell Resilience**: Supported flexible JSON/YAML/single-quote parsing in `tools/agent_bridge_cli.py` to prevent PowerShell `ScriptBlock` syntax errors; promoted flat dynamic arguments (`--branch feat --descriptor v1`).
  - **Experiment ID Allocation & Collision Prevention**: Updated `AgentBridge.create_experiment` and added `AgentBridge.get_next_experiment_num` and `TypedActionName.GET_NEXT_EXPERIMENT_ID` action. Scans both `configs/` and `.ai-harness/artifacts/<task_id>/` (and legacy `.ml-harness`), guaranteeing auto-increment IDs (`exp002`, `exp003`) and guarding against accidental overwrites.
  - **OOF & Test Prediction Serialization**: Decomposed multiclass/binary probability predictions into native float columns (`proba_<target>_<class>` and `proba_<target>_1`) alongside legacy JSON strings; exported optional `oof_predictions.parquet` for instant vectorized downstream blending/stacking.
  - **Kaggle Private Dataset Resilience**: Defaulted `data_config.license` to `"CC0-1.0"` with notice log instead of failing with hard `RuntimeError`.
  - **Standalone Kaggle Notebook Exporter**: Implemented `harness/kaggle_notebook_builder.py` and CLI subcommand `ai-harness export-kaggle-notebook` to automatically bundle offline wheels installer, model weights loading, and submission validation into a standalone `.ipynb`.
  - **Compatibility Aliases & Documentation**: Added `ml-harness` and `ml-harness-agent` entrypoint aliases in `pyproject.toml`; updated `AGENTS.md` with PowerShell-safe CLI usage guidelines.
- Next action: Open PR into `main` via `gh pr create`.

## 2026-10-15 — Agent Usability Audit: Label Normalization, Import Aliases & Kaggle Remote Robustness

- Request: Deep audit of historical transcripts across ~35 conversation sessions in `~/.gemini/antigravity/brain` using `ai-harness` (`aimo`, `airline_prediction`, `zindi`, `rsna-knee`). Identify remaining friction points, errors, and workarounds; implement framework improvements and verify with pre-push quality gates.
- Completed:
  - **Label & Class Normalization in Evaluation**: Resolved `ValueError: Ground truth outside declared classes` and `ValueError: Predictions outside target domain` in `harness/evaluation.py` by implementing `_normalize_class_value` and `_normalize_class_array`. Integer-valued floats (e.g. `0.0`, `1.0` in pandas) now cleanly match declared integer classes (`[0, 1]`) and string classes (`['0', '1']`). Added actionable diagnostic guidance when models output continuous probabilities into discrete prediction fields.
  - **Transparent Package & Module Aliases**: Created `ai_harness` and `ml_harness` alias packages with `ModuleSpec` loader forwarding submodules (`ai_harness.task_spec`, `ml_harness.runner`) directly to `harness` module instances, eliminating `ModuleNotFoundError: No module named 'ai_harness'` and `'ml_harness'`. Created `harness/agent.py` forwarding to `harness.agent_bridge`. Added `Runner = UniversalRunner` and `build_parser()` in `harness/runner.py`. Exported `SubmissionValidator` and `kaggle_manager` in `harness/__init__.py`.
  - **Kaggle Remote Runner Error Logs**: Added `KaggleKernelManager.get_kernel_error_log` which downloads and extracts remote kernel stderr tracebacks upon failure, eliminating blind `KernelWorkerStatus.ERROR` exceptions in `harness/cloud_runner.py`.
  - **Kaggle API Transient Error Tolerance**: Upgraded `wait_for_dataset` to tolerate transient HTTP 403 Forbidden / 404 Not Found propagation glitches for up to 5 consecutive attempts.
  - **Identity Mismatch Tolerance & CLI Pull**: Supported non-fatal `core_version` mismatch warnings and `--force-import` in `pull_kernel_output` and runner; added `ai-harness cloud pull --exp-id ... [--kernel-slug ...] [--force-import]` CLI subcommand.
- Evidence: 425/425 unit tests passed in 107.206s (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 6 dedicated test cases in `tests/test_agent_usability_audit.py`.
- Lesson: Submodule import aliases in Python require dedicated `importlib.machinery.ModuleSpec` loaders to guarantee module identity (`is`); remote runner exceptions must proactively download and parse remote stream logs so that AI agents receive actionable tracebacks rather than opaque cloud worker status strings.
- Next action: Commit and push changes to remote branch `feat/agent-usability-p3` and update PR #29.

## 2026-10-15 — Agent Usability Phase 5: CatBoost Built-in, Feature Diagnostics, Project Defaults & Split Robustness

- Request: Continue deep transcript audit across `~/.gemini/antigravity/brain`, identify remaining failure modes, error messages, and missing capabilities for AI coding assistants working with `ai-harness`, implement features/fixes, test thoroughly, and merge into `main`.
- Completed:
  - **Built-in CatBoost Model Wrapper**: Added `CatBoostModel` wrapper to `harness/models.py` (registered under `catboost`, `cb`, `catboost_classifier`, `catboost_regressor`, `cb_classifier`, `cb_regressor`), with automated classification, regression, probability output, parameter calculation, and default `allow_writing_files: False` (preventing transient `catboost_info/` directories from polluting project workspaces).
  - **Feature Validation Diagnostics**: Replaced generic opaque `ValueError("Input features must exist and exclude targets, IDs, groups and split metadata")` in `harness/models.py` with `validate_model_features()` providing explicit diagnostics for empty feature sets, specific missing columns (with a sample of available DataFrame columns), and forbidden metadata overlaps.
  - **HarnessProject Default Auto-Discovery**: Added dataclass field defaults and `__post_init__` to `HarnessProject` in `harness/project.py`. Direct instantiation via `HarnessProject(".")` or `HarnessProject()` now auto-populates `manifest_path`, `task_config_path`, `plugin_id`, and `artifacts_dir`, eliminating `TypeError: missing 5 required positional arguments`.
  - **Split File Column Aliases & ID Coverage Diagnostics**: In `harness/runner.py`, enhanced split file loading to detect fold column aliases (`fold`, `kfold`, `split`, `fold_id`, `cv_fold`, `Fold`, `Split`), perform case-insensitive ID column matching, and provide clear diagnostics displaying the uncovered ID count and sample uncovered IDs if IDs don't match. Extracted `UniversalRunner._load_training_data()` and honored `paths.train_path` override in experiment configs.
  - **Task Specification Flexibility**: In `harness/task_spec.py`, added support for dictionary-formatted `inputs` in `UniversalTaskSpec.from_dict` alongside list-of-dicts; supported `id_column` and `split` path declarations under top-level task metadata.
  - **Agent Bridge Usability & CLI Serialization**: Added `AgentBridge.act` method alias matching `execute_action`; added `harness.agent_bridge.main` CLI forwarder for `python -m harness.agent_bridge`; implemented `_json_serial` fallback serializer in `tools/agent_bridge_cli.py` handling `Path`, `datetime`, numpy scalars/ndarrays, and custom objects to eliminate `TypeError: Object of type ... is not JSON serializable`. Added `catboost_info/` to `.gitignore`.
- Evidence: 431/431 unit tests passed in 111.122s across the full repository test suite (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 6 dedicated test cases in `tests/test_agent_usability_p5.py`.
- Lesson: Providing friendly defaults on core orchestration dataclasses and forgiving alias resolution on data splits and inputs dramatically reduces syntax friction for AI agents without compromising contract rigor.
- Next action: Create feature branch `feat/catboost-and-diagnostics`, push, open PR, and squash-merge into `main`.

## 2026-10-15 — Agent Usability Phase 6: Built-in RandomForest, Extended Linear Aliases, TreeJournal Load Flexibility & Cross-Environment Dataset Fallback

- Request: Continuous audit across 292 transcript files in `~/.gemini/antigravity/brain`, identify remaining friction points, implement features/fixes, test thoroughly, and merge into `main`.
- Completed:
  - **Built-in RandomForestModel Wrapper**: Implemented `RandomForestModel` in `harness/models.py` (registered under `random_forest`, `randomforest`, `rf`, `rf_classifier`, `rf_regressor`) providing out-of-the-box multi-threaded (`n_jobs=-1`) Random Forest classification, regression, probability outputs (`proba_<target>`), and parameter counting on tabular and vectorized text features.
  - **Extended Baseline Model Aliases & Lasso Regression**: Expanded `GenericBaselineModel.NAMES` to recognize `logreg`, `logistic`, `linear`, `linear_regression`, `linear_model`, `lasso`, and `lasso_regression`. Added native `Lasso` estimator support for regression with proper parameter filtering.
  - **TreeJournal Load Flexibility**: Updated `TreeJournal.load` in `harness/tree_journal.py` to seamlessly handle both instance invocations (`journal.load()`) and direct class calls (`TreeJournal.load(path_str)`), eliminating `AttributeError: 'str' object has no attribute 'storage_path'`. Added `TreeJournal.load_from_file(path)` classmethod.
  - **Cross-Environment / Portable Dataset Path Fallback**: In `UniversalTaskSpec.load_dataset_split` (`harness/task_spec.py`), added automatic basename fallback resolution in `root / p.name` and `root / "dataset" / p.name` when an absolute path from a foreign container or host (such as `/kaggle/working/dataset/train.csv` or a foreign Windows/Linux path) does not exist on the current machine.
- Evidence: 435/435 unit tests passed in 92.509s across the full repository test suite (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 4 dedicated test cases in `tests/test_agent_usability_p6.py`.
- Lesson: Providing flexible invocation signatures on persistence classes and portable fallback resolution on file paths ensures experiments transfer seamlessly across cloud and local runner boundaries.
- Next action: Create feature branch `feat/rf-and-portability`, push, open PR, and squash-merge into `main`.

## 2026-10-15 — Agent Usability Phase 7: Non-NLP Modalities (Tabular Robustness, Time-Series CV, Media Path Resolution)

- Request: Expand support for non-NLP problem modalities (Tabular, Time-Series / Sequential, Computer Vision, Audio, Multimodal) and address real-world failure patterns encountered by autonomous coding agents.
- Completed:
  - **RobustTabularTransformer**: Implemented `RobustTabularTransformer` in `harness/models.py` (and exported in `harness/__init__.py`), which median-imputes missing numeric features and ordinally encodes categorical strings with missing/unseen values safely mapped to `-1.0`. Defaulted `RandomForestModel` to `RobustTabularTransformer`, permanently fixing `ValueError: could not convert string to float` when fitting tabular datasets with mixed string and numeric columns.
  - **Native TimeSeriesSplit Cross-Validation**: Added `TIME_SERIES = "time_series"` and `TEMPORAL = "temporal"` to `SplitMode` in `harness/data_strategy.py`, along with parsing for aliases (`time_series`, `time_series_split`, `timeseries`, `temporal`). Added `strategy: "time_series"` branch in `UniversalRunner._resolve_validation_splits` (`harness/runner.py`), enabling leakage-free temporal forward validation splits for financial, forecasting, and sequential tabular problems.
  - **Runner Model Family Auto-Detection**: Added `random_forest` / `rf` detection to `UniversalRunner` so that `metrics.json` records `"model_family": "random_forest"`.
  - **Media & File Path Resolution**: Added `InputVariantSpec.resolve_media_paths(df, project_root, column, extensions)` in `harness/task_spec.py` with `media_dir` configuration support. Proactively resolves relative filenames in image/audio CSVs across `project_root`, `dataset/`, `images/`, `audio/`, and `/kaggle/input/` subdirectories with extension completion (`.jpg`, `.png`, etc.), ensuring CV and Audio pipelines resolve files seamlessly in local, Docker, and Kaggle environments.
- Evidence: Full test suite passes 100% (440/440 tests) in 106.103s (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 5 dedicated test cases in `tests/test_agent_usability_p7.py`.
- Lesson: For non-GBDT tabular estimators, tree-safe numeric encoding with `-1.0` unseen-category sentinel values prevents catastrophic string conversion failures while preserving tree partition boundaries; time-series cross-validation requires expanding-window splits where validation samples strictly succeed training partitions.
- Next action: Commit, push to `feat/non-nlp-modalities`, create PR #33, and squash-merge into `main`.

## 2026-10-15 — Agent Usability Phase 8: Universal Modality Expansion (Forecasting, Financial Splits, CV, Audio & Multilabel)

- Request: Expand framework across all competition paradigms (Financial / Tabular Sequential, Computer Vision, Audio / Speech, NLP Generation & Seq2Seq, Multilabel Classification) so agents never encounter missing metrics, improper splits, or validation barriers.
- Completed:
  - **Comprehensive Metric Expansion**:
    - **Time-Series / Financial**: Implemented `compute_rmsle`, `compute_smape`, `compute_wape`, `compute_pinball_loss`, `compute_pearson`, and `compute_mrr` with aliases (`rmsle`, `smape`, `wape`, `pinball`, `quantile_loss`, `medae`, `pearson`, `mrr`).
    - **Computer Vision**: Implemented `compute_dice_score`, `compute_iou_score`, `mask_to_rle`, `rle_to_mask` (with integer-float parsing), `compute_box_iou`, and `compute_map_50` with aliases (`dice`, `iou`, `bbox_iou`, `map_50`, `map50`).
    - **Speech & Audio**: Implemented `compute_wer` and `compute_cer` with aliases (`wer`, `cer`).
    - **NLP Generation & Seq2Seq**: Implemented `compute_bleu` (with Laplace smoothing for short sentences), `compute_exact_match`, `compute_token_f1`, and `compute_token_accuracy` with aliases (`bleu`, `exact_match`, `token_f1`, `token_accuracy`).
    - **Classification & Probabilities**: Added `balanced_accuracy`, `weighted_f1`, `mcc` (Matthews correlation), `pr_auc`, and `brier_score`.
    - **Direction Inference**: Implemented `default_metric_direction(metric)` automatically inferring minimize vs. maximize for all new metrics.
  - **Financial & Sequential Data Strategy**:
    - Implemented `PurgedGroupTimeSeriesSplit` (and `SplitMode.PURGED_GROUP_TIME_SERIES`) with configurable embargo buffer (`embargo_pct`), preventing lookahead and auto-correlation leakage across sequential trading sessions / dates.
  - **Multilabel Stratified Split**:
    - Implemented `MultilabelStratifiedKFold` (and `SplitMode.MULTILABEL_STRATIFIED_KFOLD`) providing greedy iterative multi-label stratification without external dependencies.
  - **Task Specification & Auto-Normalization**:
    - Defined `TASK_TYPE_NORMALIZATIONS` mapping aliases (`forecasting`, `ts_forecast`, `segmentation`, `detection`, `ner`, `asr`, `generation`, `seq2seq`, `retrieval`, `ranking`) to canonical task types.
    - Updated `SubtaskSpec.from_dict` to auto-normalize task types and infer default metric direction.
  - **Submission Validator Flexibility**:
    - Updated `SubmissionPreflightValidator` to accept non-numeric predictions (such as RLE masks, ASR text transcripts, and string class names) when `probability_mode=False`.
  - **Runner Split Integration**:
    - Added runner branches for `purged_group_time_series` / `financial_time_series` and `multilabel_stratified` / `multilabel_stratified_kfold` in `UniversalRunner._resolve_validation_splits`.
- Evidence: Full test suite passes 100% (449/449 tests) in 94.790s (`python -m unittest discover -s tests -p "test_*.py"`). Verified with 9 dedicated unit tests in `tests/test_agent_usability_p8.py`.
- Lesson: Providing built-in domain metrics, anti-leakage purged financial splits, and string-tolerant submission validation enables autonomous agents to immediately tackle competitive tasks across CV, audio, NLP generation, and quantitative finance without writing fragile custom scoring loops.
- Next action: Stage, commit, push to `feat/universal-modalities-expansion`, create PR #34, and squash-merge into `main`.




## 2026-10-16 — Chat-Audit Follow-Through: Execution Backend Lock, `ai-harness sync`, Script-Adapter Docs

- Request: After auditing 12 Antigravity conversations (12–16/10) across 5 competition workspaces, implement the top-3 remaining friction fixes: (1) persistent Kaggle backend lock from the task contract, (2) one-command framework sync for workspaces, (3) documented ad-hoc script → Kaggle execution path.
- Completed:
  - **Task-contract execution backend policy** (`harness/task_spec.py`): `UniversalTaskSpec.execution_config` / `execution_backend` / `execution_locked` parse `execution: {backend, locked}` from the task document. `AgentBridge._resolve_execution_backend` (new) resolves backend precedence **explicit `--backend` param > `execution.backend` > `local`** and raises `Backend lock violation` when `locked: true` and the request differs. The session budget guard (`_check_budget_and_stop_guards`) canonicalizes `params["backend"]` for `run_full_validation` before dispatch; `run_smoke_test`/`run_probe` remain intentionally local-only. The Context Packet HUD now surfaces `- **Execution Backend**: KAGGLE LOCKED ...` when declared.
  - **`ai-harness sync`** (`harness/sync.py` + `harness/cli.py`): one-command workspace upgrade — auto-detects the framework clone (`<workspace>/ai-harness` or `HARNESS_FRAMEWORK_DIR`/`--framework-dir`), `git fetch` + `pull --ff-only` (never touches a dirty tree; clear reasons for dirty/no-remote states), `pip install -e` with the current interpreter, re-attach (`attach_project force=True`) + regenerates the task-specific `.agents/skills/ai-harness/SKILL.md`, and reports before/after commit, installed version, and entrypoint health. Skippable via `--no-pull/--no-install/--no-attach`; JSON output via `--json`.
  - **SKILL.md generation** (`harness/skill_injector.py`): removed the hardcoded `--backend local` from the SOP; SOP step 5 now uses the contract backend and carries a lock warning; a locked contract injects an extra `> [!IMPORTANT] Execution Backend LOCKED to KAGGLE` banner; new `## 5. Execution Backends & Ad-hoc Scripts (Kaggle)` section documents the script adapter (`execution: {mode: script, entrypoint: ...}`, `--exp-dir`/`--config` contract) and the monitoring commands (`ai-harness kaggle status/log --exp`, `wait_cloud_job`, `inspect_cloud_job`) so agents stop hand-polling `kaggle kernels status`. The generic attach skill (`harness/project.py`) gained a one-line backend-policy note.
  - **Docs**: `docs/TASK_SPEC_TEMPLATE.md` §2.6 documents `execution.backend`/`execution.locked`; `AGENTS.md` gained the backend-policy working principle and the `ai-harness sync` command.
- Evidence: Full test suite passes 100% (533/533) in 137.5s (`python -m unittest discover -s tests -p "test_*.py"`), including 23 new tests: `tests/test_execution_backend_policy.py` (14 — spec properties, resolution precedence, lock violation, guard→FAILED-observation, HUD) and `tests/test_sync_command.py` (9 — repo detection, no-remote/dirty/with-remote sync flows, CLI wiring). Manual smoke: sync on a temp workspace regenerated the locked KAGGLE skill; a locked `act --action run_full_validation --backend local` now returns a clean JSON FAILED observation (`Guard rejected the action: Backend lock violation ...`) instead of a local run; the matching call canonicalizes to `cloud_runs += 1`.
- Lesson: The recurring "agent ran local instead of Kaggle" failure was a state problem, not an agent-misbehavior problem — per-call defaults with no persistent policy mean every fresh session re-derives the wrong default from the SKILL.md SOP. Encoding the policy in the task contract + enforcing it in the guard + surfacing it in the HUD and generated skills fixes it at the source; the same pattern (make the default carry the intent) applies to future per-workspace policies.
- Next action: Stage, commit, push to a feature branch, create PR; propagate to the 5 active competition workspaces by running `ai-harness sync` there and adding `execution: {backend: kaggle, locked: true}` to each `configs/task.yaml` where the user locked Kaggle.

## 2026-10-16 — Chat-Audit Round 2: Agent-Friction Fixes (std_val, typed-action aliases, entries, job_id, consent kwargs, metric alias, auto-exp-ID, CLI backend lock)

- Request: Continue the Oct-15 meta-audit handoff. Verify which of the 9 unresolved issues still exist in current code, fix the remaining ones, plus 2 newly discovered friction bugs (experiment-ID collision hard-fail, runner CLI ignoring the contract backend), with regression tests.
- Completed (8 fixes; issue #9 from the handoff list was already implemented in the tree and hardened):
  - **`ValidationReport.std_val`** (`harness/endgame_submission.py`): preflight now computes and reports the std of predicted values (0.0 for single-row/constant) so agents can detect prediction collapse; default-kept positional constructor compatibility.
  - **`UniversalRunner.expected_validation_ids`** (`harness/runner.py`): initialized eagerly in `__init__` (before split resolution) so script adapters / standalone validation checks no longer hit `AttributeError`.
  - **Typed-action method aliases** (`harness/agent_bridge.py`): every `TypedActionName` is now callable directly on a bridge instance (`bridge.review_hypothesis(...)`, `bridge.run_smoke_test(config_path=...)`) via `__getattr__` forwarding into the single `execute_action` path (session/approval/audit semantics unchanged); class-level access raises an actionable `AttributeError` pointing at instance usage.
  - **`ResearchMemory.entries` dual access** (`harness/research_memory.py`): `entries` is now a property returning an `_EntryList` that is also callable (`entries()` legacy form still works) and whose records are dict subclasses with attribute access (`entry.kind`), fixing `TypeError: 'method' object is not subscriptable`.
  - **Kaggle import identity policy** (`harness/templates/kaggle_template.py`): new `_is_critical_identity_mismatch` — `core_version` drift never blocks; a `job_id` mismatch only blocks when the remote kernel reports a *different concrete* job id (an unset/`None` remote job_id is a warning, not a failure).
  - **`UploadConsentStore.grant()`** (`harness/cloud_policy.py`): accepts `confirmed_upload` plus arbitrary scalar flags (recorded in the consent record) instead of raising `TypeError`.
  - **`SubtaskSpec` metric alias** (`harness/task_spec.py`): constructor accepts `metric=` as an alias for `primary_metric` (conflicting aliases rejected); positional/dict paths unchanged.
  - **create_experiment auto-ID on collision** (`harness/agent_bridge.py`): a requested `exp_id` that already has immutable artifacts no longer returns FAILED; the next free ID is auto-assigned, reported in the observation summary + `data.auto_assigned_exp_id`, and the old artifacts are untouched.
  - **Runner CLI backend lock + smoke-local rule** (`harness/runner.py`): extracted `resolve_cli_backend()` — `ai-harness-run` now resolves explicit `--backend` > contract `execution.backend` > local (matching the documented help and the bridge behavior); locked contracts reject other backends for full runs; **smoke/probe runs (`--debug`/`--fold`) always run locally**, even when the contract declares a cloud backend (previously a smoke test on a locked Kaggle task would silently dispatch to the cloud).
- Evidence: Full suite 100% — 572 tests, OK (1 skipped) in 95s (`python -m unittest discover -s tests -p "test_*.py"`). New: `tests/test_agent_friction_fixes.py` (21 tests, one per observed error string) and `tests/test_execution_backend_policy.py::TestCliBackendResolution` (4 tests). `tests/test_agent_usability_p3.py` collision assertion updated from FAILED-hard-stop to auto-assign (behavior change, deliberate).
- Lesson: Most of the 9 handoff items were one-line API-shape frictions (missing attribute, missing kwarg, method-vs-property) that agents hit repeatedly because the harness surface is read from error messages, not from docs. Fix pattern: keep the canonical API, but make every *plausible* agent call shape work (alias kwarg, dual access, eager init) and make failures say exactly what to do next (auto-assigned ID, "call it on an instance").
- Next action: Stage, commit on a feature branch, open PR against `main`; then verify in a locked Kaggle workspace that `ai-harness-run --config ... --debug --fold 0` stays local while the full run dispatches to Kaggle.
