"""Agent Bridge & Context Bridge for Antigravity IDE Runtime.

Architecture Principle:
- The TRUE Agent Runtime (Reasoning, Hypothesis Formulation, Planning, Decision Loop)
  is **Antigravity inside the IDE** — NOT an internal Python planner or LLM loop inside the Harness.
- The Harness provides:
  1. Task State & Memory (`CompetitionStateMachine`, `TreeJournal`, `Diagnostics`, `Idea Graveyard`).
  2. Deterministic Guardrails (`GroundingGate`, `ExperimentValidator`, `EnsembleGuard`, `AuditGate`).
  3. Execution & Training Backends (`local` CPU/GPU via `ai-harness-run`, remote backends when the selected adapter supports them).
- `AgentBridge` connects Antigravity in the IDE with the Harness and Dashboard UI via:
  - Compact, high-signal `ContextPacket` (JSON & Markdown) so Antigravity can reason without context bloat.
  - Strictly typed Harness Actions (`TypedActionName`) returning structured `ActionObservation` payloads.
  - Stateful `BridgeSession` (`session_0001`, ...) tracking goal, experiments run, compute usage,
    consecutive rejections/failures, optional session budgets (`max_experiments`, `max_gpu_hours`,
    `max_cloud_runs`), and mandatory `USER_APPROVAL` gates for sensitive actions (`submit`, `freeze`,
    `unfreeze`, `finalize`).
"""

from __future__ import annotations

import datetime
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from harness.competition_state import CompetitionStateMachine
from harness.submission_manager import SubmissionManager
from harness.tree_journal import TreeJournal


from harness.bridge_actions import ActionHandlerMixin
from harness.bridge_common import (
    ActionObservation,
    ApprovalTier,
    BridgeSession,
    DEFAULT_ACTION_APPROVAL_TIERS,
    MAX_SESSION_ACTIONS,
    MAX_SESSION_OBSERVATIONS,
    SessionBudget,
    TypedActionCall,
    TypedActionName,
    _TYPED_ACTION_NAMES,
    TYPED_ACTION_SCHEMAS as TYPED_ACTION_SCHEMAS,
    get_action_schema as get_action_schema,
)

typed_actions = [a.value for a in TypedActionName]


class _AgentBridgeMeta(type):
    """Metaclass that routes class-level typed-action lookups to actionable guidance.

    Agents sometimes reach for ``AgentBridge.review_hypothesis`` or
    ``AgentBridge.confirm_task_spec`` as if they were classmethods (mirroring
    the pattern in other harness classes). Instead of a bare ``AttributeError``
    we return a small callable that, when invoked, constructs an
    ``AgentBridge`` and dispatches through the standard instance path so
    session state, approval tiers, and audit all still apply.
    """

    def __getattr__(cls, name: str):
        if name in _TYPED_ACTION_NAMES:
            def _invoke_class(*args, **kwargs):
                params: Dict[str, Any] = {}
                if args and isinstance(args[0], dict):
                    params = dict(args[0])
                params.update(kwargs)
                # A bare class-level call has no project_root; the bridge
                # resolves the active task from CWD or the active attach.
                bridge = AgentBridge()
                return bridge.execute_action(name, params)

            _invoke_class.__name__ = name
            _invoke_class.__doc__ = (
                f"Class-level proxy for the typed action '{name}'. "
                "Equivalent to AgentBridge().execute_action('" + name + "', params). "
                "Prefer an instance (bridge = AgentBridge(project_root)) for explicit scoping."
            )
            return _invoke_class
        raise AttributeError(
            f"type object 'AgentBridge' has no attribute '{name}'"
        )


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _research_continuity_hud_lines(memory: Dict[str, Any]) -> List[str]:
    """Render the research memory payload as compact HUD lines.

    The pre-fix HUD embedded the full payload as one raw ``json.dumps`` line
    (~500 tokens even for an empty workspace, far more with real research
    history) — exactly the context saturation the framework forbids. The full
    payload stays available via ``context --format json`` and the research
    memory actions; the HUD only surfaces what a restarted agent needs to
    resume: counts, latest checkpoint conclusion/next action, recent decisions
    with lessons, unreviewed results, progress path.
    """
    lines: List[str] = []
    streak = memory.get("consecutive_non_improvements") or 0
    lines.append(
        f"- Entries: {memory.get('entry_count', 0)} (insights {memory.get('insight_count', 0)}, "
        f"open {memory.get('open_insight_count', 0)}) | verified runs: {memory.get('verified_run_count', 0)} | "
        f"non-improvement streak: {streak}"
        + (" | REASSESS PLATEAU" if memory.get("reassess_recommended") else "")
    )
    checkpoint = memory.get("latest_checkpoint")
    if checkpoint:
        ids = ", ".join(str(i)[:8] for i in (checkpoint.get("insight_ids") or [])[:3])
        suffix = f" [{ids}]" if ids else ""
        conclusion = _truncate_text(checkpoint.get("conclusion"), 160)
        next_action = _truncate_text(checkpoint.get("next_action"), 140)
        lines.append(f"- Latest checkpoint: {conclusion} -> next: {next_action}{suffix}")
        questions = checkpoint.get("open_questions") or []
        if questions:
            lines.append(
                f"- Open questions: {'; '.join(_truncate_text(q, 120) for q in questions[:2])}"
            )
    for decision in (memory.get("recent_decisions") or [])[:2]:
        outcome = decision.get("outcome") or "recorded"
        conclusion = _truncate_text(decision.get("conclusion"), 120)
        lesson = _truncate_text(decision.get("lesson"), 120)
        lines.append(
            f"- Recent decision ({outcome}): {conclusion}" + (f" | lesson: {lesson}" if lesson else "")
        )
    frontier = memory.get("research_frontier") or []
    frontier_label = (
        "empty"
        if not frontier
        else "; ".join(
            f"{str(state.get('insight_id', '?'))[:8]}={state.get('status')}"
            for state in frontier[:3]
        )
    )
    lines.append(
        f"- Unreviewed results: {memory.get('unreviewed_result_count', 0)} | frontier: {frontier_label}"
    )
    progress = memory.get("progress_path")
    if progress:
        lines.append(f"- Progress: `{progress}`")
    if memory.get("progress_warning"):
        lines.append(f"- \u26a0\ufe0f {memory['progress_warning']}")
    if memory.get("reassess_recommended") or streak >= 2:
        lines.append(
            "- \u26a0\ufe0f STAGNATION / PLATEAU ADVISORY: Do NOT freeze or stop submissions while quota remains. "
            "Run 'ai-harness discussion list <comp>' to harvest competitor insights, explore orthogonal model families, "
            "and execute bounded discriminating probes autonomously."
        )
    return lines


def adoption_gate_lines(
    hygiene: Optional[Dict[str, Any]], total_experiments: int = 0
) -> List[str]:
    """T3 session-start adoption gate (aexp pattern, host-agnostic layer).

    Rendered into the Context HUD AND the start-session observation so the
    bypass evidence / zero-experiment state is in front of the agent at the
    session entry point itself - not only if it happens to call `context`.
    """
    hygiene = hygiene or {}
    bypassed = [
        f for f in hygiene.get("findings", []) if f.get("kind") == "bypassed_run_output"
    ]
    total_experiments = int(total_experiments or 0)
    if not bypassed and total_experiments != 0:
        return []
    lines = ["## ⛔ Adoption protocol — read before any experiment work"]
    if bypassed:
        sample = bypassed[0].get("detail", "")
        n_dirs = max(int(f.get("dir_count") or 0) for f in bypassed)
        scope = (
            f"**{n_dirs} run-output dir(s) lack harness evidence**"
            if n_dirs
            else "**run-output dirs lack harness evidence**"
        )
        lines.append(
            f"- {scope} — dirs created OUTSIDE the harness (manual `kaggle kernels push` / "
            "ad-hoc scripts): no audit trail, no leaderboard record, invisible "
            "to the framework. " + sample[:160]
        )
    if total_experiments == 0:
        lines.append(
            "- **0 harness experiments so far.** Every experiment MUST go "
            "through `create_experiment` → `run_smoke_test` → "
            "`run_full_validation` (Bridge actions below) — there is no "
            "supported alternative path for tracked runs."
        )
    lines.append(
        "- If harness dispatch fails, **report the failure** — do not silently "
        "fall back to a manual run. Manual pushes are the human operator's "
        "privilege (`AI_HARNESS_ALLOW_MANUAL=1`), not the agent's."
    )
    lines.append(
        "- Do not delete or rename run-output dirs to make this warning "
        "disappear; surface them to the user instead."
    )
    return lines


class AgentBridge(ActionHandlerMixin, metaclass=_AgentBridgeMeta):
    """Bidirectional Context & Typed Action Bridge between Antigravity (IDE) and the ML Harness.

    Motto: Antigravity thinks. AgentBridge coordinates. Harness verifies and executes.
    """

    # execute_action keyword arguments that must not be swallowed into `params`
    _EXEC_RESERVED_KEYWORDS = (
        "rationale", "session_id", "goal", "budget", "user_approved",
    )

    def __getattr__(self, name: str):
        """Expose every registered Typed Action as a direct method.

        Agents naturally call ``bridge.review_hypothesis(...)`` or
        ``bridge.run_smoke_test(config_path=...)`` as if they were Python
        methods. This keeps that ergonomic style working while preserving a
        single execution path through :meth:`execute_action` (session state,
        approval tiers, and audit all still apply).
        """
        if name.startswith("__") or name not in _TYPED_ACTION_NAMES:
            raise AttributeError(f"'AgentBridge' object has no attribute '{name}'")

        def _invoke(*args, **kwargs):
            params: Dict[str, Any] = {}
            if args and isinstance(args[0], dict):
                params = dict(args[0])
                args = args[1:]
            if len(args) > 1:
                raise TypeError(
                    f"{name}() takes at most one positional argument (the params dict)"
                )
            params.update(kwargs)
            reserved = {
                key: params.pop(key)
                for key in list(params)
                if key in self._EXEC_RESERVED_KEYWORDS
            }
            _, observation = self.execute_action(name, params=params, **reserved)
            return observation

        _invoke.__name__ = name
        _invoke.__qualname__ = f"AgentBridge.{name}"
        _invoke.__doc__ = (
            f"Direct alias for the typed action '{name}' "
            f"(forwards to execute_action)."
        )
        return _invoke


    def __init__(
        self,
        project_root: Path | str = ".",
        task_spec_path: Optional[str | Path] = None,
    ):
        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.project import HarnessProject
        from harness.task_spec import load_active_task_spec

        self.project = HarnessProject.discover(project_root, task_spec_path)
        self.project_root = self.project.root
        self.artifacts_dir = self.project.artifacts_dir
        self.sessions_dir = self.artifacts_dir / "agent_sessions"
        self.legacy_runs_dir = self.artifacts_dir / "agent_runs"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_runs_dir.mkdir(parents=True, exist_ok=True)
        self.comp_state = CompetitionStateMachine(
            self.project_root, self.project.task_config_path
        )
        self.task_spec = load_active_task_spec(self.project.task_config_path)
        self.plugin_context = TaskPluginContext(
            project_root=self.project_root,
            task_config_path=self.project.task_config_path,
            task_spec=self.task_spec,
            plugin_id=self.project.plugin_id or "default",
        )
        self.task_plugin = load_task_plugin(self.project.plugin_id, self.plugin_context)
        from harness.phase_workflow import PhaseWorkflow

        self.phase_workflow = PhaseWorkflow(
            self.project_root, self.project.task_config_path
        )
        self._session_snapshots = getattr(self, "_session_snapshots", {})

    def _new_journal(self, auto_sync: bool = True) -> TreeJournal:
        return TreeJournal(
            journal_path=self.artifacts_dir / "journal.json",
            artifacts_dir=self.artifacts_dir,
            auto_sync=auto_sync,
            task_spec=self.task_spec,
            task_plugin=self.task_plugin,
        )

    def _new_submission_manager(self) -> SubmissionManager:
        return SubmissionManager(
            self.artifacts_dir / "submissions",
            artifacts_dir=self.artifacts_dir,
            project_root=self.project_root,
            task_config_path=self.project.task_config_path,
        )

    def _research_memory(self):
        from harness.research_memory import ResearchMemory

        return ResearchMemory(self.project_root, self.artifacts_dir, self.task_spec, self.task_plugin)

    def _review_research_config(self, cfg, params, for_execution=False):
        reason = params.get("revisit_reason") or cfg.get("research", {}).get("revisit_reason", "")
        review = self._research_memory().review(cfg, reason, for_execution=for_execution)
        if not review["allowed"]:
            raise ValueError("Repeated declared execution config; use search_research_memory and "
                             "provide revisit_reason for an intentional repeat. Matches: " +
                             json.dumps(review["exact_config_matches"]))
        return review

    def _validate_config_path(self, config_path: str) -> Path:
        """Resolve a config path and reject traversal outside the project root.

        Defense-in-depth: the agent passes `config_path` to `run_full_validation`,
        `run_smoke_test`, and `run_probe`. While `write_experiment_config` already
        constrains writes to `configs/`, an agent could still pass an arbitrary
        path (e.g. `../../etc/passwd.yaml`). Resolving and requiring the result
        to stay within the project root prevents reading files outside the workspace.
        """
        resolved = (self.project_root / config_path).resolve()
        try:
            resolved.relative_to(self.project_root.resolve())
        except ValueError:
            raise ValueError(
                f"Config path '{config_path}' resolves outside the project root; "
                f"only configs within the workspace are allowed."
            )
        return resolved

    def _resolve_execution_backend(self, params: Dict[str, Any]) -> Tuple[str, bool]:
        """Resolve the execution backend for `run_full_validation`.

        Precedence: explicit `backend` param > task contract `execution.backend`
        > "local". When the contract declares `execution.locked: true`, any
        explicit backend different from the locked one is rejected — this is
        what stops agents from silently falling back to local runs on a
        Kaggle-locked workspace (recurring failure mode: weak local machine,
        OOM, wasted compute).

        Returns ``(backend, locked)"""
        contract_backend = getattr(self.task_spec, "execution_backend", None)
        locked = bool(getattr(self.task_spec, "execution_locked", False))
        requested = str(params.get("backend") or "").strip().lower()
        effective = requested or contract_backend or "local"
        if locked and contract_backend and requested and requested != contract_backend:
            raise ValueError(
                f"Backend lock violation: the task contract locks the execution "
                f"backend to '{contract_backend}' (execution.locked: true) but "
                f"'{requested}' was requested. Drop the 'backend' param (the "
                f"contract default applies) or change the contract via "
                f"update_task_contract."
            )
        if effective not in {"local", "kaggle", "modal"}:
            raise ValueError(f"Unsupported execution backend: {effective!r}")
        return effective, locked

    def _refresh_task_context(self) -> None:
        """Reload task config and plugin state after an onboarding action rewrites it."""
        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.project import HarnessProject
        from harness.task_spec import load_active_task_spec

        self.project = HarnessProject.discover(
            self.project_root, self.project.task_config_path
        )
        self.project_root = self.project.root
        self.artifacts_dir = self.project.artifacts_dir
        self.task_spec = load_active_task_spec(
            self.project.task_config_path, force_reload=True
        )
        self.comp_state = CompetitionStateMachine(
            self.project_root, self.project.task_config_path
        )
        self.plugin_context = TaskPluginContext(
            project_root=self.project_root,
            task_config_path=self.project.task_config_path,
            task_spec=self.task_spec,
            plugin_id=self.project.plugin_id or "default",
        )
        self.task_plugin = load_task_plugin(self.project.plugin_id, self.plugin_context)
        from harness.phase_workflow import PhaseWorkflow

        self.phase_workflow = PhaseWorkflow(
            self.project_root, self.project.task_config_path
        )
        self._session_snapshots = getattr(self, "_session_snapshots", {})

    def _get_spec_min_improvement(
        self, session_or_budget: Optional[Any] = None
    ) -> float:
        """Resolve min_meaningful_improvement from SessionBudget or CompetitionSpec (`configs/task.yaml`)."""
        budget_dict = (
            session_or_budget.budget
            if isinstance(session_or_budget, BridgeSession)
            else (session_or_budget if isinstance(session_or_budget, dict) else {})
        )
        if budget_dict and budget_dict.get("min_meaningful_improvement") is not None:
            return float(budget_dict["min_meaningful_improvement"])
        spec = self.comp_state.spec or {}
        if spec.get("min_meaningful_improvement") is not None:
            return float(spec["min_meaningful_improvement"])
        eval_proto = spec.get("evaluation_protocol", {})
        if eval_proto.get("min_meaningful_improvement") is not None:
            return float(eval_proto["min_meaningful_improvement"])
        return 0.002

    def _extract_idea_graveyard(self) -> List[str]:
        """Dynamically parse documented failed interventions from EXPERIMENTS.md."""
        items: List[str] = []
        paths = {
            self.project_root / "EXPERIMENTS.md",
            self.project_root / ".ai-harness/EXPERIMENTS.md",
        }
        base_path = self.project_root / "configs/base.yaml"
        if base_path.exists():
            base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
            location = base.get("paths", {}).get("leaderboard_path")
            if location:
                paths.add(self.project_root / location)
        for exp_md in sorted(paths):
            if not exp_md.exists():
                continue
            try:
                in_graveyard = False
                for line in exp_md.read_text(encoding="utf-8").splitlines():
                    if "Idea Graveyard" in line:
                        in_graveyard = True
                        continue
                    if in_graveyard and line.startswith("## "):
                        break
                    if (
                        in_graveyard
                        and line.strip().startswith("|")
                        and "Tested Intervention" not in line
                        and "---" not in line
                    ):
                        cols = [c.strip() for c in line.strip().strip("|").split("|")]
                        if len(cols) >= 5:
                            items.append(f"{cols[0]}: {cols[4]} ({cols[2]})")
            except Exception:
                pass
        if not items:
            items.append(
                "No legacy graveyard notes found; search structured research memory for decisions."
            )
        return items

    # =========================================================================
    # 1. SESSION MANAGEMENT (STATEFUL TRACKING & BUDGET/APPROVAL GUARDS)
    # =========================================================================
    def get_next_session_id(self) -> str:
        existing = []
        for folder in (self.sessions_dir, self.legacy_runs_dir):
            for f in folder.glob("*.json"):
                try:
                    parts = f.stem.split("_")
                    num = int(parts[-1])
                    existing.append(num)
                except (IndexError, ValueError):
                    pass
        return f"session_{(max(existing, default=0) + 1):04d}"

    @staticmethod
    def _candidate_source_ids(mode: str, source_map: Dict[str, Any]) -> List[str]:
        """Resolve the source experiment IDs a candidate recipe will consume.

        Mirrors the id-resolution in `harness.candidates.build_candidate` so the
        duplicate-recipe fingerprint can be computed before the candidate exists.
        """
        if mode in {"single_model", "checkpoint_selection"}:
            return [source_map.get("source_exp") or source_map.get("model_exp") or ""]
        if mode == "specialist_router":
            routing = source_map.get("routing") or source_map.get("target_routing") or {}
            return list(dict.fromkeys(routing.values()))
        if mode in {"ensemble_vote", "weighted_blend", "probability_blend"}:
            return list(source_map.get("components", []))
        raise ValueError(f"Unsupported candidate mode: {mode}")

    def _guard_submission(self, cand_path: Path, exp_id: str, sub_mgr) -> Dict[str, Any]:
        """Deterministic anti-bypass guards run before a submission is registered.

        The audit gate CLI (`tools/audit_gate.py`) is a standalone tool; the agent
        drives `prepare_submission` through the bridge and would otherwise skip it
        entirely. Enforce here, on the bridge path, the three checks that stop an
        agent from shipping a non-improvement:

        1. Duplicate-submission dedup: the candidate file's SHA-256 must not match
           any previously registered submission (no re-submitting the same file).
        2. Minimum meaningful effect size (MMES): the candidate's OOF must differ
           from the baseline anchor's OOF on at least `min_diff_pct` of rows (or
           `min_diff` rows). A model that only jitters < threshold of rows is not
           a genuine new experiment.
        3. Significance (when ground truth is available): the paired error test
           (regression) / McNemar's test (classification) must reach p < 0.05.

        Returns a small dict of guard evidence; raises ValueError on any hard fail.
        """

        from harness.artifacts import sha256

        evidence: Dict[str, Any] = {"checks": {}}
        cand_sha = sha256(cand_path)
        evidence["candidate_sha256"] = cand_sha

        # ---- 1. Duplicate-submission dedup ---------------------------------
        existing = {
            s.candidate_path: s.id
            for s in sub_mgr.submissions.values()
            if s.candidate_path
        }
        # Also compare by content hash: a previous submission may point at the
        # same bytes even if the path differs. Re-hash prior submissions' files.
        dup_id = None
        for path, sub_id in existing.items():
            p = Path(path)
            if not p.is_absolute():
                p = self.project_root / p
            if not p.is_file():
                continue
            try:
                if sha256(p) == cand_sha:
                    dup_id = sub_id
                    break
            except OSError:
                continue
        if dup_id:
            raise ValueError(
                f"Submission dedup failed: candidate is byte-identical (SHA-256 "
                f"{cand_sha[:12]}...) to already-registered submission '{dup_id}'. "
                f"Re-submitting the same predictions does not advance the "
                f"competition. Produce a genuinely different run first."
            )
        evidence["checks"]["dedup"] = "PASS"

        # ---- 2. Minimum meaningful effect size vs baseline anchor ---------
        baseline_exp_id = self.comp_state.state["gates"].get("baseline_exp_id")
        sub_spec = self.comp_state.spec.get("submission_spec", {}) or {}
        min_diff_pct = float(sub_spec.get("min_diff_pct", 2.0))
        min_diff = int(sub_spec.get("min_diff", 1))
        rel_eps = float(sub_spec.get("rel_epsilon", 0.01))
        mmes = "SKIPPED_NO_BASELINE"
        if baseline_exp_id and baseline_exp_id != exp_id:
            base_oof = self.artifacts_dir / baseline_exp_id / "oof_predictions.csv"
            cand_oof = cand_path.parent / "oof_predictions.csv"
            if base_oof.is_file() and cand_oof.is_file():
                try:
                    cand_df = pd.read_csv(cand_oof)
                    base_df = pd.read_csv(base_oof)
                    key = self.task_spec.id_column
                    if key in cand_df and key in base_df:
                        cand_df = cand_df.set_index(key)
                        base_df = base_df.set_index(key).reindex(cand_df.index)
                        pred_cols = [c for c in cand_df if c.startswith(("pred_", "proba_"))]
                        if pred_cols:
                            changed = pd.Series(False, index=cand_df.index)
                            for col in pred_cols:
                                if col not in base_df.columns:
                                    continue
                                try:
                                    cv = pd.to_numeric(cand_df[col], errors="raise")
                                    bv = pd.to_numeric(base_df[col], errors="raise")
                                    if pd.api.types.is_float_dtype(cv) or pd.api.types.is_float_dtype(bv):
                                        std = float(np.std(bv))
                                        scale = std if std > 1e-8 else 1.0
                                        changed |= (np.abs(cv.to_numpy() - bv.to_numpy()) / scale) >= rel_eps
                                    else:
                                        changed |= cand_df[col] != base_df[col]
                                except (ValueError, TypeError):
                                    changed |= cand_df[col] != base_df[col]
                            n_changed = int(changed.sum())
                            n_rows = int(len(cand_df))
                            pct = (n_changed / n_rows * 100.0) if n_rows else 0.0
                            required = max(min_diff, int(math.ceil(n_rows * min_diff_pct / 100.0)))
                            if n_changed < required:
                                raise ValueError(
                                    f"MMES violation: candidate '{exp_id}' differs from baseline "
                                    f"anchor '{baseline_exp_id}' on only {n_changed}/{n_rows} rows "
                                    f"({pct:.2f}%); required >= {required} ({min_diff_pct}%). "
                                    f"A change below the minimum meaningful effect size is not a "
                                    f"genuine new experiment (likely seed noise or a trivial tweak)."
                                )
                            mmes = f"PASS ({n_changed}/{n_rows} rows, {pct:.2f}%)"
                        else:
                            mmes = "SKIPPED_NO_PRED_COL"
                except ValueError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    mmes = f"SKIPPED ({exc})"
        evidence["checks"]["mmes"] = mmes

        # ---- 3. Significance test (only when ground truth is available) ----
        sig = "SKIPPED_NO_GROUND_TRUTH"
        truth_path = cand_path.parent / "evaluation_truth.csv"
        if truth_path.is_file() and baseline_exp_id and baseline_exp_id != exp_id:
            base_oof = self.artifacts_dir / baseline_exp_id / "oof_predictions.csv"
            cand_oof = cand_path.parent / "oof_predictions.csv"
            if base_oof.is_file() and cand_oof.is_file():
                try:
                    sig = self._significance_test(
                        cand_oof, base_oof, truth_path, self.task_spec.id_column
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    sig = f"SKIPPED ({exc})"
        evidence["checks"]["significance"] = sig

        return evidence

    @staticmethod
    def _significance_test(
        cand_oof: Path, base_oof: Path, truth_path: Path, key: str
    ) -> str:
        """Paired error test (regression) or McNemar (classification); reject p >= 0.05."""
        from scipy.stats import chi2, ttest_rel

        cand = pd.read_csv(cand_oof).set_index(key)
        base = pd.read_csv(base_oof).set_index(key)
        truth = pd.read_csv(truth_path).set_index(key)

        pred_col = next((c for c in cand.columns if c.startswith("pred_")), None)
        if pred_col is None or pred_col not in base.columns or truth.index.intersection(cand.index).empty:
            return "SKIPPED_NO_ALIGNED_COL"

        idx = cand.index.intersection(base.index).intersection(truth.index)
        if len(idx) < 2:
            return "SKIPPED_TOO_FEW_ROWS"
        y = truth.loc[idx, pred_col] if pred_col in truth.columns else None
        if y is None:
            # Fall back to the first truth column
            ycol = truth.columns[0]
            y = truth.loc[idx, ycol]
        cand_v = cand.loc[idx, pred_col]
        base_v = base.loc[idx, pred_col]

        try:
            y_num = pd.to_numeric(y, errors="raise")
            c_num = pd.to_numeric(cand_v, errors="raise")
            b_num = pd.to_numeric(base_v, errors="raise")
            continuous = (
                pd.api.types.is_float_dtype(y_num)
                or pd.api.types.is_float_dtype(c_num)
                or pd.api.types.is_float_dtype(b_num)
            )
        except (ValueError, TypeError):
            continuous = False

        import warnings

        # Determine whether the candidate is genuinely *better* than the baseline;
        # a statistically-significant *worse* result is still rejected.
        if continuous:
            y = y_num.to_numpy(dtype=float)
            c = c_num.to_numpy(dtype=float)
            b = b_num.to_numpy(dtype=float)
            c_err = (c - y) ** 2
            b_err = (b - y) ** 2
            better = float(np.mean(c_err) < np.mean(b_err))
            if np.allclose(c_err, b_err, atol=0.0, rtol=1e-7):
                return "FAIL (p=1.0, no improvement)"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ttest_rel(c_err, b_err)
            p = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
            test = "paired_ttest"
        else:
            y = y.astype(object)
            c = cand_v.astype(object)
            b = base_v.astype(object)
            c_ok = c == y
            b_ok = b == y
            disc = int((c_ok & ~b_ok).sum() + (~c_ok & b_ok).sum())
            if disc == 0:
                return "FAIL (p=1.0, no discordant pairs)"
            b_win = int((c_ok & ~b_ok).sum())  # candidate correct, baseline wrong
            c_win = int((~c_ok & b_ok).sum())  # baseline correct, candidate wrong
            better = b_win > c_win
            stat = ((abs(b_win - c_win) - 1.0) ** 2) / disc
            p = float(1.0 - chi2.cdf(stat, df=1))
            test = "mcnemar"

        if p < 0.05 and better:
            return f"PASS ({test}, p={p:.4f})"
        if p < 0.05 and not better:
            return f"FAIL ({test}, p={p:.4f}, candidate not better than baseline)"
        return f"FAIL ({test}, p={p:.4f})"


    def _suggest_next_interventions(self, limit: int = 5) -> Dict[str, Any]:
        """Deterministic brainstorm aid to counter agent stagnation.

        An agent tends to stop generating new versions once a few runs exist. This
        action surfaces *candidate hypotheses that have not been tested yet* and
        are not in the idea graveyard, ranked against the current bottleneck, so the
        agent has concrete, non-duplicate starting points instead of "thinking from
        nothing". It never fabricates scores: every suggestion is a declared, unrun
        hypothesis whose provenance (frontier insight / branch / bottleneck cue) is
        surfaced so the scientist can judge it.
        """
        journal = self._new_journal(auto_sync=False)
        ucb = journal.recommend_next_branch(
            max_params=self.comp_state.spec.get("competition_rules", {}).get("max_params_limit")
        )
        frontier = self._research_memory().context(
            min_improvement=self._get_spec_min_improvement({}),
            patience=3,
        )
        graveyard = self._extract_idea_graveyard()
        # Branches already tried (from the solution tree) so we can prefer
        # under-explored branches and avoid re-suggesting an identical recipe.
        tried_branches: Dict[str, int] = {}
        tried_hypotheses = set()
        for node in journal.nodes.values():
            tried_branches[node.branch] = tried_branches.get(node.branch, 0) + 1
            if node.atomic_hypothesis:
                tried_hypotheses.add(node.atomic_hypothesis.strip().lower())
        parent_id = ucb.get("parent_id")
        bottlenecks = ucb.get("bottlenecks") or []
        if not bottlenecks and parent_id and parent_id in journal.nodes:
            bottlenecks = (journal.nodes[parent_id].diagnostics_summary or {}).get(
                "critical_bottlenecks", []
            )

        suggestions: List[Dict[str, Any]] = []
        # 1) Active frontier insights that are not yet decided are the strongest
        #    signal: they are hypotheses the scientist already recorded and that
        #    still need a probe or experiment.
        for state in frontier.get("research_frontier", []):
            title = str(state.get("title", "")).strip()
            next_action = str(state.get("next_action", "")).strip()
            required_step = str(state.get("required_step", "")).strip()
            mechanism = str(state.get("mechanism", "")).strip()
            if not title and not next_action:
                continue
            hyp = next_action or title
            if hyp.strip().lower() in tried_hypotheses:
                continue
            if self._in_graveyard(hyp, graveyard):
                continue
            suggestions.append({
                "source": "research_frontier",
                "insight_id": state.get("insight_id"),
                "branch": "probe" if state.get("status") in {"needs_probe_plan", "ready_for_probe"} else "feat",
                "parent_id": parent_id,
                "hypothesis": hyp,
                "rationale": mechanism or required_step or f"Frontier insight '{title}' is {state.get('status')}.",
                "status": state.get("status"),
            })
        # 2) The UCB-recommended branch is the exploration lever: suggest one atomic
        #    intervention on the recommended parent addressing the top bottleneck.
        rec_branch = ucb.get("recommended_branch")
        if rec_branch and not any(s["branch"] == rec_branch for s in suggestions):
            cue = bottlenecks[0] if bottlenecks else ucb.get("diagnostic_cue")
            if cue or rec_branch != "feat":
                hyp = f"Apply a targeted '{rec_branch}' intervention on '{parent_id}' to address: {cue or rec_branch}."
                if hyp.strip().lower() not in tried_hypotheses and not self._in_graveyard(hyp, graveyard):
                    suggestions.append({
                        "source": "ucb_recommendation",
                        "branch": rec_branch,
                        "parent_id": parent_id,
                        "hypothesis": hyp,
                        "rationale": f"UCB recommends branch '{rec_branch}' (score={ucb.get('ucb_score')}); "
                                     f"fewest visits on this branch: {tried_branches.get(rec_branch, 0)}.",
                    })
        # 3) Under-explored branches from the tree journal (diversity lever).
        for branch, visits in sorted(tried_branches.items(), key=lambda kv: kv[1]):
            if len(suggestions) >= limit:
                break
            if visits >= 2 or branch == rec_branch:
                continue
            hyp = f"Explore under-tested branch '{branch}' with one atomic change on '{parent_id}'."
            if hyp.strip().lower() in tried_hypotheses or self._in_graveyard(hyp, graveyard):
                continue
            suggestions.append({
                "source": "underexplored_branch",
                "branch": branch,
                "parent_id": parent_id,
                "hypothesis": hyp,
                "rationale": f"Branch '{branch}' has only {visits} run(s); diversify the search.",
            })

        suggestions = suggestions[: max(1, int(limit))]
        return {
            "stagnation": {
                "reassess_recommended": bool(frontier.get("reassess_recommended")),
                "consecutive_non_improvements": int(frontier.get("consecutive_non_improvements") or 0),
                "total_runs": len(journal.nodes),
                "branches_tried": tried_branches,
                "graveyard_size": len(graveyard) if isinstance(graveyard, (list, tuple, set)) else 0,
            },
            "parent_id": parent_id,
            "current_bottlenecks": list(bottlenecks)[:5],
            "suggestions": suggestions,
            "note": "Suggested hypotheses are unrun and declared; each still requires "
                    "grounding + a smoke probe before full validation. Verify and record "
                    "'verified_via' when you act on one.",
        }

    @staticmethod
    def _in_graveyard(text: str, graveyard: Any) -> bool:
        """True if a candidate hypothesis overlaps an idea already in the graveyard."""
        if not text or not graveyard:
            return False
        norm = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
        if not norm:
            return False
        for item in graveyard:
            g = item if isinstance(item, str) else str(item)
            gnorm = re.sub(r"[^a-z0-9]+", " ", g.lower()).split()
            if not gnorm:
                continue
            overlap = len(set(norm) & set(gnorm)) / max(1, len(set(norm) | set(gnorm)))
            if overlap >= 0.6:
                return True
        return False

    def _claim_active_session(self, session_id: str) -> None:
        """Mark the given session as the workspace's active one, demoting any others.

        Writes are best-effort and idempotent so a crash mid-claim never corrupts
        session files; the winner is whoever wins the per-file transaction.
        """
        from harness.state_store import transaction

        self._session_snapshots = getattr(self, "_session_snapshots", {})
        for path in self.sessions_dir.glob("session_*.json"):
            if path.stem == session_id:
                continue
            try:
                with transaction(path, {"sessions": []}) as current:
                    if current.get("active"):
                        current["active"] = False
            except Exception:
                pass

    def save_session(self, session: BridgeSession) -> None:
        if (
            Path(session.session_id).name != session.session_id
            or "/" in session.session_id
            or "\\" in session.session_id
        ):
            raise ValueError("Invalid session ID")
        session.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Cap unbounded history: long sessions (50+ experiments) otherwise
        # re-serialize MB-scale JSON on every save_session call.
        if len(session.actions) > MAX_SESSION_ACTIONS:
            session.actions = session.actions[-MAX_SESSION_ACTIONS:]
        if len(session.observations) > MAX_SESSION_OBSERVATIONS:
            session.observations = (
                session.observations[-MAX_SESSION_OBSERVATIONS:]
            )
        payload = asdict(session)
        path = self.sessions_dir / f"{session.session_id}.json"

        # Enforce a single active session per workspace. When the session being
        # saved is active, demote any other previously-active sessions so that
        # `get_active_or_create_session` has a stable target to resume.
        if payload.get("active"):
            self._claim_active_session(session.session_id)
            payload["active"] = True
        import copy

        from harness.io_utils import atomic_write_json
        from harness.state_store import save_snapshot

        payload = save_snapshot(
            path, self._session_snapshots.get(session.session_id, {}), payload
        )
        self._session_snapshots[session.session_id] = copy.deepcopy(payload)
        for key, value in payload.items():
            if key in BridgeSession.__dataclass_fields__:
                setattr(session, key, value)

        # Mirror into legacy agent_runs dir (`run_XXXX.json`) for UI/backward compatibility
        run_num = session.session_id.split("_")[-1]
        legacy_id = (
            f"run_{run_num}"
            if not session.session_id.startswith("run_")
            else session.session_id
        )
        legacy_payload = {
            **payload,
            "id": legacy_id,
            "plan": [
                f"[{a.get('approval_tier', 'AUTO')}] {a.get('action')}: {a.get('rationale', '')}"
                for a in session.actions
            ],
            "decisions": [
                {
                    "step_index": o.get("step_index", idx + 1),
                    "verdict": o.get("advisory") or o.get("status", "COMPLETED"),
                    "reasoning": o.get("summary", ""),
                }
                for idx, o in enumerate(session.observations)
            ],
        }
        atomic_write_json(
            self.legacy_runs_dir / f"{legacy_id}.json",
            legacy_payload,
            indent=2,
            ensure_ascii=False,
        )
        self._sync_to_dashboard_state(session)

    def load_session(self, session_id: str) -> BridgeSession:
        if (
            Path(session_id).name != session_id
            or "/" in session_id
            or "\\" in session_id
        ):
            raise ValueError("Invalid session ID")
        candidates = [
            self.sessions_dir / f"{session_id}.json",
            self.legacy_runs_dir / f"{session_id}.json",
        ]
        if session_id.startswith("run_"):
            num = session_id.split("_")[-1]
            candidates.insert(0, self.sessions_dir / f"session_{num}.json")
        elif session_id.startswith("session_"):
            num = session_id.split("_")[-1]
            candidates.append(self.legacy_runs_dir / f"run_{num}.json")

        target = next((p for p in candidates if p.exists()), None)
        if not target:
            raise FileNotFoundError(
                f"BridgeSession '{session_id}' not found in {self.sessions_dir}"
            )

        data = json.loads(target.read_text(encoding="utf-8"))
        valid_fields = BridgeSession.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "session_id" not in filtered:
            filtered["session_id"] = data.get("id", session_id)
        import copy

        self._session_snapshots[filtered["session_id"]] = copy.deepcopy(filtered)
        session = BridgeSession(**filtered)
        self._reconcile_session_usage(session)
        return session

    def _reconcile_session_usage(self, session):
        path = self.artifacts_dir / "resource_ledger.json"
        if not path.is_file():
            return
        ledger = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            r
            for r in ledger.get("reservations", {}).values()
            if r["session_id"] == session.session_id
        ]
        if not rows:
            return
        # Terminal statuses only — a RESERVED row whose job never reached a
        # terminal state (e.g. the process crashed) should not count as used
        # budget, and a FAILED run that the agent is allowed to retry should not
        # consume the experiment budget forever.
        terminal_statuses = {"COMPLETED", "FAILED", "CANCELLED"}
        terminal_rows = [r for r in rows if r.get("status") in terminal_statuses]
        gpu_hours = []
        for r in terminal_rows:
            if r.get("gpu_hours") is not None:
                gpu_hours.append(r["gpu_hours"])
        session.compute_usage.update(
            experiments_used=sum(r["category"] == "training" for r in terminal_rows),
            smoke_runs=sum(r["category"] == "probe" for r in terminal_rows),
            cloud_runs=sum(r["backend"] != "local" for r in terminal_rows),
            candidate_operations=sum(r["category"] == "candidate" for r in terminal_rows),
            wall_seconds_used=round(
                sum(float(r.get("wall_seconds") or 0) for r in terminal_rows), 3
            ),
            gpu_hours_used=(
                round(sum(gpu_hours), 6) if gpu_hours else 0.0
            ),
        )

    # Auto-checkpoint interval: every N executed actions we record a
    # lightweight checkpoint into research memory so that long sessions
    # survive context compaction without losing the "where am I" state.
    _AUTO_CHECKPOINT_INTERVAL = 50

    def _maybe_auto_checkpoint(self, session, obs) -> None:
        """Record a research-memory checkpoint every N steps.

        Idempotent: only fires when ``step_index`` is a positive multiple of
        :data:`_AUTO_CHECKPOINT_INTERVAL` and the most recent action was not
        already a checkpoint (avoids double-recording when the agent
        explicitly called ``record_research_checkpoint`` at the same step).
        """
        step = int(getattr(obs, "step_index", 0) or 0)
        if step <= 0 or step % self._AUTO_CHECKPOINT_INTERVAL != 0:
            return
        # Skip if the agent just recorded a checkpoint manually.
        recent_actions = [a.get("action") for a in session.actions[-3:]]
        if "record_research_checkpoint" in recent_actions:
            return
        # Build a compact summary from the session state.
        usage = session.compute_usage
        actions_run = len(session.actions)
        last_action = session.actions[-1].get("action") if session.actions else None
        stop_reason = session.stop_reason or ""
        summary = (
            f"Auto-checkpoint at step {step}: "
            f"{actions_run} actions executed, "
            f"experiments_used={usage.experiments_used}, "
            f"smoke_runs={usage.smoke_runs}, "
            f"cloud_runs={usage.cloud_runs}, "
            f"last_action={last_action}"
        )
        if stop_reason:
            summary += f", stop_reason={stop_reason[:120]}"
        next_action = (
            f"Continue from step {step}; review research_frontier for the "
            "latest insight and unreviewed results before the next intervention."
        )
        memory = self._research_memory()
        memory.checkpoint(
            summary=summary,
            next_action=next_action,
            session_id=session.session_id,
        )

    def get_active_or_create_session(
        self,
        goal: str = "Iterative competition optimization via Antigravity IDE",
        session_id: Optional[str] = None,
        budget: Optional[Dict[str, Any]] = None,
    ) -> BridgeSession:
        if session_id:
            try:
                sess = self.load_session(session_id)
                if budget:
                    sess.budget.update(
                        {k: v for k, v in budget.items() if v is not None}
                    )
                    self.save_session(sess)
                return sess
            except FileNotFoundError:
                pass

        recent = self.list_sessions(limit=20)
        for item in recent:
            if item.get("active") and item.get("status") in ("ACTIVE", "PAUSED_FOR_APPROVAL"):
                sess = self.load_session(item["session_id"])
                if (
                    goal
                    and goal != "Iterative competition optimization via Antigravity IDE"
                ):
                    sess.goal = goal
                if budget:
                    sess.budget.update(
                        {k: v for k, v in budget.items() if v is not None}
                    )
                sess.active = True
                self.save_session(sess)
                return sess
        # Fallback: legacy sessions without an `active` flag — pick the most recent
        # one in an open state so old workspaces keep working.
        for item in recent:
            if item.get("status") in ("ACTIVE", "PAUSED_FOR_APPROVAL"):
                sess = self.load_session(item["session_id"])
                if (
                    goal
                    and goal != "Iterative competition optimization via Antigravity IDE"
                ):
                    sess.goal = goal
                if budget:
                    sess.budget.update(
                        {k: v for k, v in budget.items() if v is not None}
                    )
                sess.active = True
                self.save_session(sess)
                return sess

        return self.start_session(goal=goal, budget=budget)

    def start_session(
        self,
        goal: str,
        budget: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> BridgeSession:
        journal = self._new_journal(auto_sync=True)
        sub_mgr = self._new_submission_manager()
        snap = self.comp_state.evaluate_journey_snapshot(journal, sub_mgr)

        import uuid

        sid = session_id or "session_" + uuid.uuid4().hex
        if (self.sessions_dir / f"{sid}.json").exists():
            raise ValueError("Session ID already exists; resume the existing session")
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sess_budget = asdict(SessionBudget())
        sess_budget["min_meaningful_improvement"] = self._get_spec_min_improvement(
            budget
        )
        if budget:
            sess_budget.update({k: v for k, v in budget.items() if v is not None})

        session = BridgeSession(
            session_id=sid,
            goal=goal,
            status="ACTIVE",
            created_at=now_iso,
            updated_at=now_iso,
            starting_state=snap.get("hud", {}),
            budget=sess_budget,
            active=True,
        )
        self.save_session(session)
        return session

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        files = sorted(
            list(self.sessions_dir.glob("session_*.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        out: List[Dict[str, Any]] = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                # Normalize: legacy sessions may not have an `active` flag.
                if "active" not in data:
                    data["active"] = False
                out.append(data)
            except Exception:
                pass
        return out

    def get_next_experiment_num(self) -> int:
        """Scan configs, tree journal, and immutable artifact directories to return next available experiment integer."""
        existing_nums: List[int] = []
        configs_dir = self.project_root / "configs"
        if configs_dir.exists():
            for p in configs_dir.glob("exp*.yaml"):
                try:
                    num = int(p.stem.split("_")[0].replace("exp", ""))
                    existing_nums.append(num)
                except Exception:
                    pass

        # Scan artifact directories (.ai-harness and legacy .ml-harness)
        for hidden in [".ai-harness", ".ml-harness"]:
            art_root = self.project_root / hidden / "artifacts"
            if art_root.exists():
                for cand_dir in art_root.iterdir():
                    if cand_dir.is_dir():
                        for exp_folder in cand_dir.iterdir():
                            if exp_folder.is_dir() and exp_folder.name.startswith("exp"):
                                try:
                                    num = int(exp_folder.name.split("_")[0].replace("exp", ""))
                                    existing_nums.append(num)
                                except Exception:
                                    pass

        # Scan tree journal
        try:
            journal = self._load_journal()
            if hasattr(journal, "nodes") and journal.nodes:
                for n_id in journal.nodes.keys():
                    if str(n_id).startswith("exp"):
                        try:
                            num = int(str(n_id).split("_")[0].replace("exp", ""))
                            existing_nums.append(num)
                        except Exception:
                            pass
        except Exception:
            pass

        if not existing_nums:
            return 1
        return max(existing_nums) + 1

    def _sync_to_dashboard_state(self, session: BridgeSession) -> None:
        """Synchronize live BridgeSession telemetry into competition_state.json for the Dashboard UI."""
        merged_budget = {
            **session.budget,
            **session.compute_usage,
        }
        self.comp_state.state["agent_copilot"] = {
            "last_run_id": session.session_id,
            "session_id": session.session_id,
            "last_run_at": session.updated_at,
            "goal": session.goal,
            "status": session.status,
            "stop_reason": session.stop_reason,
            "reassess_recommended": session.reassess_recommended,
            "experiments_run": session.experiments_run,
            "compute_usage": session.compute_usage,
            "consecutive_rejections": session.consecutive_rejections,
            "budget": merged_budget,
            "pending_approval_action": session.pending_approval_action,
            "steps": [
                {
                    "step": f"[{o['action']}] {o['summary']}",
                    "status": o["status"].lower(),
                }
                for o in session.observations
            ],
            "actions": session.actions,
            "observations": session.observations,
        }
        self.comp_state.save()

    # =========================================================================
    # 2. CONTEXT BRIDGE (STRUCTURED CONTEXT PACKET FOR ANTIGRAVITY IN IDE)
    # =========================================================================
    def get_context_packet(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a compressed, high-density Context Packet for Antigravity in the IDE."""
        journal = self._new_journal(auto_sync=True)
        sub_mgr = self._new_submission_manager()
        snap = self.comp_state.evaluate_journey_snapshot(journal, sub_mgr)
        max_p = self.comp_state.spec.get("competition_rules", {}).get(
            "max_params_limit"
        )
        ucb_rec = journal.recommend_next_branch(max_params=max_p)

        active_session = None
        if session_id:
            try:
                active_session = asdict(self.load_session(session_id))
            except Exception:
                pass
        if not active_session:
            sessions = self.list_sessions(limit=1)
            active_session = sessions[0] if sessions else None
        if active_session:
            active_session = {key: active_session.get(key) for key in (
                "session_id", "status", "goal", "budget", "compute_usage", "consecutive_rejections",
                "reassess_recommended", "pending_approval_action", "experiments_run")}
            active_session["goal"] = str(active_session.get("goal", ""))[:600]
            runs = active_session.get("experiments_run") or []
            active_session["experiments_run_count"] = len(runs)
            active_session["experiments_run"] = runs[-8:]
            if active_session.get("pending_approval_action"):
                pending = active_session["pending_approval_action"]
                active_session["pending_approval_action"] = {
                    "action": pending.get("action"), "rationale": str(pending.get("rationale", ""))[:600]}

        parent_id = ucb_rec.get("parent_id") or (
            snap["hud"]["what_is_current_best"]["cv_best"].get("id")
        )
        bottlenecks = ucb_rec.get("bottlenecks") or []
        if not bottlenecks and parent_id and parent_id in journal.nodes:
            diag = journal.nodes[parent_id].diagnostics_summary or {}
            bottlenecks = diag.get("critical_bottlenecks", [])

        graveyard = self._extract_idea_graveyard()

        # P2 audit: submissions visibility — recent registry entries so agents
        # can see what was submitted where without digging through the tree.
        recent_submissions = []
        try:
            for entry in sorted(
                sub_mgr.submissions.values(),
                key=lambda e: e.created_at or "",
                reverse=True,
            )[:6]:
                item: Dict[str, Any] = {
                    "id": entry.id,
                    "exp_id": entry.exp_id,
                    "status": entry.status,
                    "created_at": entry.created_at,
                    "local_cv": {
                        k: round(float(v), 4)
                        for k, v in list(entry.local_cv_scores.items())[:4]
                    },
                    "official": {
                        k: (round(float(v), 4) if v is not None else None)
                        for k, v in list(entry.official_scores.items())[:4]
                    },
                }
                if entry.archive_path:
                    try:
                        item["path"] = str(
                            Path(entry.archive_path).relative_to(self.project_root)
                        ).replace("\\", "/")
                    except ValueError:
                        item["path"] = entry.archive_path
                recent_submissions.append(item)
        except Exception:
            pass

        # P2 audit: workspace hygiene — advisory clutter scan of the root.
        try:
            from harness.hygiene import scan_workspace_hygiene

            workspace_hygiene = scan_workspace_hygiene(self.project_root)
        except Exception:
            workspace_hygiene = {"clean": True, "findings": [], "summary": "unreadable"}

        packet = {
            "bridge_version": "1.1_antigravity_bridge",
            "project_root": str(self.project_root),
            "task_config": str(
                self.project.task_config_path.relative_to(self.project_root)
            ).replace("\\", "/"),
            "task_plugin": {
                "id": self.project.plugin_id or "default",
                "version": self.task_plugin.version,
                "capabilities": self.task_plugin.capabilities(),
            },
            "architecture_motto": "Antigravity thinks. AgentBridge coordinates. Harness verifies and executes.",
            "runtime_owner": "Antigravity (IDE)",
            "active_session": active_session,
            "competition_hud": snap["hud"],
            "active_tracks": snap["active_tracks"],
            "gates": snap["gates"],
            "ucb_recommendation": ucb_rec,
            "parent_diagnostics_bottlenecks": bottlenecks,
            "idea_graveyard": graveyard,
            "recent_submissions": recent_submissions,
            "workspace_hygiene": workspace_hygiene,
            "research_memory": self._research_memory().context(
                min_improvement=self._get_spec_min_improvement((active_session or {}).get("budget", {})),
                patience=int((active_session or {}).get("budget", {}).get("max_consecutive_rejections") or 3),
            ),
            "candidate_pool_summary": {
                "baseline_id": snap["candidate_pool"]["baseline_id"],
                "baseline_score": snap["candidate_pool"]["baseline_score"],
                "overall_best": snap["candidate_pool"]["overall_best"],
                "synthesized_count": len(
                    snap["candidate_pool"]["synthesized_candidates"]
                ),
            },
            "available_typed_actions": DEFAULT_ACTION_APPROVAL_TIERS,
        }
        try:
            packet["phase_workflow"] = self.phase_workflow.phase_status()
        except Exception as exc:
            packet["phase_workflow"] = {"status": "not_configured", "detail": str(exc)}
        return packet

    def render_context_markdown(self, session_id: Optional[str] = None) -> str:
        """Render the Context Packet into a token-efficient Markdown brief for Antigravity."""
        pkt = self.get_context_packet(session_id=session_id)
        hud = pkt["competition_hud"]
        where = hud["where_am_i"]
        best = hud["what_is_current_best"]
        ucb = pkt["ucb_recommendation"]
        sess = pkt["active_session"]

        baseline_score = best["baseline"].get("score")
        current_score = best["cv_best"].get("score")
        current_delta = best["cv_best"].get("delta")
        baseline_label = "N/A" if baseline_score is None else f"{baseline_score:.4f}"
        current_label = "N/A" if current_score is None else f"{current_score:.4f}"
        delta_label = "N/A" if current_delta is None else f"{current_delta:+.4f}"
        exec_cfg = getattr(self.task_spec, "execution_config", {}) or {}
        contract_backend = str(exec_cfg.get("backend") or "").strip().lower()
        lines = [
            "# 🌉 ANTIGRAVITY <-> HARNESS CONTEXT PACKET",
            f"> *{pkt['architecture_motto']}*",
            "",
            f"- **Active Stage**: `{where['phase_title']}` | **Mode**: `{where['mode']}`",
            f"- **Parallel Tracks**: Experimentation=`{where['active_tracks'].get('experimentation')}` | "
            f"Candidates=`{where['active_tracks'].get('candidate_management')}` | "
            f"External Eval=`{where['active_tracks'].get('external_evaluation')}`",
            f"- **Baseline Anchor**: `{best['baseline']['id']}` (`{baseline_label}`)",
            f"- **Current Best CV**: `{best['cv_best']['id']}` (`{current_label}`, Δ `{delta_label}`)",
        ]
        if contract_backend:
            suffix = (
                " **LOCKED** (never run training locally; dispatch with "
                "`run_full_validation`, backend defaults to the contract)"
                if exec_cfg.get("locked")
                else " (default; explicit `--backend` overrides)"
            )
            lines.append(f"- **Execution Backend**: `{contract_backend.upper()}`{suffix}")

        ext_eval = (hud.get("what_am_i_working_on") or {}).get("external_evaluation") or {}
        quota_used = ext_eval.get("quota_used_today")
        daily_quota = ext_eval.get("daily_quota")
        rem_quota = ext_eval.get("remaining_quota_today")
        if quota_used is not None and daily_quota is not None:
            rem_str = f" ({rem_quota} remaining today)" if rem_quota is not None else ""
            lines.append(f"- **Daily Submission Quota**: `{quota_used}/{daily_quota}` used{rem_str}")
            if rem_quota and rem_quota > 0:
                lines.append(
                    f"> ⚡ **QUOTA REMAINING ({rem_quota} slot{'s' if rem_quota != 1 else ''} today)**: "
                    "Per Principle 10 & Rule 3.7, you MUST NOT stop while quota remains. "
                    "Formulate orthogonal probes or test risk-hedging blends!"
                )

        subs = pkt.get("recent_submissions") or []
        lines.extend(["", "## 📦 Submissions (last {})".format(len(subs))])
        if not subs:
            lines.append(
                "- (none yet — `prepare_submission` then `record_external_score`)"
            )
        for s in subs:
            local = s.get("local_cv") or {}
            official = s.get("official") or {}
            local_label = (
                f"{next(iter(local.values())):.4f} ({next(iter(local))})"
                if local
                else "—"
            )
            official_vals = [v for v in official.values() if v is not None]
            official_label = (
                f"{official_vals[0]:.4f}"
                if official_vals
                else ("—" if not official else "pending")
            )
            lines.append(
                f"- `{s['id']}` exp=`{s['exp_id']}` | {s['status']} | "
                f"local CV: {local_label} | official: {official_label}"
                + (f" | `{s['path']}`" if s.get("path") else "")
            )

        if sess:
            cu = sess.get("compute_usage", {})
            bg = sess.get("budget", {})
            max_exp = bg.get("max_experiments") or "∞"
            lines.extend(
                [
                    "",
                    f"## 📋 Active Bridge Session: `{sess['session_id']}` (`{sess['status']}`)",
                    f"- **Goal**: {sess.get('goal')}",
                    f"- **Experiments Run ({cu.get('experiments_used', 0)}/{max_exp})**: `{sess.get('experiments_run', [])}`",
                    f"- **Compute Usage**: Local=`{cu.get('local_runs', 0)}` | Cloud=`{cu.get('cloud_runs', 0)}` | "
                    f"GPU Hours=`{cu.get('gpu_hours_used', 0.0)}h` | Wall=`{cu.get('wall_seconds_used', 0.0)}s`",
                    f"- **Consecutive Rejections**: `{sess.get('consecutive_rejections', 0)}/{bg.get('max_consecutive_rejections', 3)}`",
                ]
            )
            if sess.get("reassess_recommended"):
                lines.append(
                    "- ⚠️ **SOFT ADVISORY (`REASSESS_RECOMMENDED`)**: Multiple consecutive iterations did not reach "
                    "`min_meaningful_improvement`. Re-examine parent `DIAGNOSTICS.md` or pivot branch before running heavy CV."
                )
            if sess.get("pending_approval_action"):
                pa = sess["pending_approval_action"]
                lines.append(
                    f"- ⚠️ **PENDING USER APPROVAL**: Action `{pa.get('action')}` — {pa.get('rationale')} "
                    f"(Resume via `ai-harness-agent approve --session-id {sess['session_id']}`)"
                )

        if hud.get("what_is_blocking_me"):
            lines.extend(["", "## 🚧 Active Blockers"])
            for b in hud["what_is_blocking_me"]:
                lines.append(
                    f"- **[{b['severity'].upper()}] {b['title']}**: {b['message']}"
                )

        lines.extend(
            [
                "",
                "## 🧠 AIDE TreeJournal UCB Recommendation & Diagnostics",
                f"- **Recommended Parent**: `{ucb.get('parent_id')}` (Score: `{ucb.get('parent_score')}`, UCB: `{ucb.get('ucb_score')}`)",
                f"- **Recommended Branch**: `{ucb.get('recommended_branch')}`",
                f"- **Diagnostic Cue**: {ucb.get('diagnostic_cue') or 'None'}",
            ]
        )
        if pkt["parent_diagnostics_bottlenecks"]:
            for bn in pkt["parent_diagnostics_bottlenecks"]:
                lines.append(f"  - 🔴 *Bottleneck*: {bn}")

        lines.extend(
            [
                "",
                "## 🪦 Idea Graveyard (Legacy notes; reassess evidence before revisiting)",
            ]
        )
        for g in pkt["idea_graveyard"]:
            lines.append(f"- ❌ {g}")

        memory = pkt["research_memory"]
        lines.extend(["", "## Research continuity (survives sessions and compaction)"])
        lines.extend(_research_continuity_hud_lines(memory))
        lines.extend(f"- {step}" for step in memory["resume_protocol"])

        from harness.hygiene import hygiene_hud_lines

        if sess and sess.get("consecutive_rejections", 0) >= 2:
            lines.extend(
                [
                    "",
                    "> [!IMPORTANT]",
                    "> 🔍 **AUTONOMOUS DISCUSSION MINING MANDATE (Stagnation Countermeasure)**:",
                    "> Consecutive validation rejections detected. You are strictly forbidden from micro-tuning hyperparameters or passively returning decision lists to the user.",
                    "> You must immediately run `ai-harness discussion list <comp> --sort-by hot`, inspect the top topics with `ai-harness discussion view <id>`, and formulate an orthogonal hypothesis.",
                ]
            )

        hygiene_lines = hygiene_hud_lines(pkt.get("workspace_hygiene"))
        if hygiene_lines:
            lines.extend(["", "## 🧹 Workspace Hygiene"])
            lines.extend(hygiene_lines)

        # T3 session-start adoption gate (shared with the start-session
        # observation): instructions alone are advisory, so bypass evidence
        # is put in front of the agent at session start.
        gate_lines = adoption_gate_lines(
            pkt.get("workspace_hygiene"),
            int((pkt.get("competition_hud") or {}).get("total_experiments") or 0),
        )
        if gate_lines:
            lines.extend(["", gate_lines[0]])
            lines.extend(gate_lines[1:])

        lines.extend(
            [
                "",
                "## 🛠️ Typed Bridge Actions (`ai-harness-agent act --action <name> ...`)",
                "- **AUTO (Research continuity)**: `search_research_memory`, `review_hypothesis`, `record_research_insight`, `plan_research_probe`, `trace_research_insight`, `record_research_decision`, `record_research_checkpoint`, `suggest_next_interventions` (stagnation countermeasure)",
                "- **AUTO**: `inspect_data_readiness`, `run_eda`, `record_phase_output`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`",
                "- **AUTO_WITH_LIMIT (Session Budget Tracked)**: `establish_baseline_anchor`, `run_full_validation` (dispatches a tracked local/Kaggle/Modal job), `synthesize_candidate`, `build_ensemble`",
                "- **AUTO (Job lifecycle)**: `inspect_cloud_job`, `wait_cloud_job`",
                "- **USER_APPROVAL (Requires Human Sign-off / `--user-approved`)**: `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`",
            ]
        )
        return "\n".join(lines) + "\n"

    # =========================================================================
    # 3. TYPED ACTION EXECUTION & GUARDRAIL ENFORCEMENT
    # =========================================================================
    def _check_budget_and_stop_guards(
        self,
        session: BridgeSession,
        action: str,
        params: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        """Enforce Hard Stops (Freeze lock & Resource Budgets).

        Note: `consecutive_rejections` defaults to a SOFT advisory (`REASSESS_RECOMMENDED`)
        rather than a hard block unless `hard_stop_on_consecutive_rejections=True` is set.

        ``run_full_validation`` is also subject to the execution-backend guard:
        the backend is resolved (explicit param > task-contract
        ``execution.backend`` > ``local``) and canonicalized on ``params``; a
        contract with ``execution.locked: true`` rejects other backends by
        raising ``ValueError`` (``execute_action`` converts it into a FAILED
        observation).
        """
        # 1. Hard Stop: Freeze lock guard
        # A FINALIZED competition is terminal too: finalize_competition now
        # requires freeze_locked, and once finalized the same stop set applies
        # (the terminal state must not be display-only).
        frozen = self.comp_state.state["gates"].get("freeze_locked") or (
            self.comp_state.state["gates"].get("finalized")
        )
        if frozen and action not in (
            TypedActionName.EMERGENCY_UNFREEZE.value,
            TypedActionName.FINALIZE_COMPETITION.value,
            TypedActionName.INSPECT_EXPERIMENT.value,
            TypedActionName.INSPECT_DIAGNOSTICS.value,
            TypedActionName.SEARCH_RESEARCH_MEMORY.value,
            TypedActionName.REVIEW_HYPOTHESIS.value,
            TypedActionName.RECORD_RESEARCH_DECISION.value,
            TypedActionName.RECORD_RESEARCH_CHECKPOINT.value,
            TypedActionName.RECORD_RESEARCH_INSIGHT.value,
            TypedActionName.PLAN_RESEARCH_PROBE.value,
            TypedActionName.TRACE_RESEARCH_INSIGHT.value,
            TypedActionName.SUGGEST_NEXT_INTERVENTIONS.value,
        ):
            state_word = (
                "FINALIZED"
                if self.comp_state.state["gates"].get("finalized")
                else "FROZEN"
            )
            return (
                "STOP_POLICY_TRIGGERED",
                f"Competition state is {state_word}. Only inspection, research notes, finalization, or `emergency_unfreeze` (with USER_APPROVAL) is allowed.",
            )

        budget = session.budget
        usage = session.compute_usage

        # 2. Optional Hard Stop on consecutive rejections (only if explicitly enabled in session budget)
        max_rej = int(budget.get("max_consecutive_rejections") or 3)
        if (
            bool(budget.get("hard_stop_on_consecutive_rejections", False))
            and action == TypedActionName.RUN_FULL_VALIDATION.value
            and session.consecutive_rejections >= max_rej
        ):
            return (
                "STOP_POLICY_TRIGGERED",
                f"Hard stop enabled: reached {session.consecutive_rejections}/{max_rej} consecutive rejections.",
            )

        # 3. Hard Stop: Experiment count budget
        max_exp = budget.get("max_experiments")
        if max_exp is not None and action in (
            TypedActionName.RUN_FULL_VALIDATION.value,
        ):
            if int(usage.get("experiments_used", 0)) >= int(max_exp):
                return (
                    "BLOCKED_BY_BUDGET",
                    f"Session experiment budget exhausted ({usage.get('experiments_used')}/{max_exp}).",
                )

        # 4. Hard Stop: Cloud runs budget
        # run_full_validation defaults to the task contract's execution backend
        # (see _resolve_execution_backend), so the guard must resolve the same
        # way the handler will. Smoke tests and probes always run locally, so
        # for those actions the explicit param (default "local") is authoritative.
        if action == TypedActionName.RUN_FULL_VALIDATION.value:
            backend, _locked = self._resolve_execution_backend(params)
            params["backend"] = backend
        else:
            backend = str(params.get("backend", "local")).lower()
        max_cloud = budget.get("max_cloud_runs")
        if (
            max_cloud is not None
            and action == TypedActionName.RUN_FULL_VALIDATION.value
            and backend in ("modal", "kaggle")
        ):
            if int(usage.get("cloud_runs", 0)) >= int(max_cloud):
                return (
                    "BLOCKED_BY_BUDGET",
                    f"Session cloud run budget exhausted ({usage.get('cloud_runs')}/{max_cloud}).",
                )

        execution = action in {"run_full_validation", "run_smoke_test", "run_probe"}
        if execution and (
            budget.get("max_wall_seconds") is not None
            or budget.get("max_gpu_hours") is not None
        ):
            if backend != "local":
                return (
                    "BLOCKED_BY_BUDGET",
                    "Backend cannot prove termination and allocation accounting for a hard runtime budget",
                )
            if budget.get("max_gpu_hours") is not None:
                return (
                    "BLOCKED_BY_BUDGET",
                    "Local backend has no verified GPU allocation measurement; use a wall-time budget",
                )
            remaining = float(budget["max_wall_seconds"]) - float(
                usage.get("wall_seconds_used", 0)
            )
            if remaining <= 0:
                return ("BLOCKED_BY_BUDGET", "Wall-time budget exhausted")
            params["_deadline_seconds"] = remaining
        # 5. Hard Stop: GPU hours budget
        max_gpu = budget.get("max_gpu_hours")
        if max_gpu is not None and float(usage.get("gpu_hours_used") or 0.0) >= float(
            max_gpu
        ):
            return (
                "BLOCKED_BY_BUDGET",
                f"Session GPU hour budget exhausted ({usage.get('gpu_hours_used'):.2f}h / {max_gpu}h).",
            )

        return None

    def _evaluate_oof_dataframe(
        self, oof_csv_path: Path
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate an OOF predictions CSV against the ground-truth training dataset."""
        from harness.artifacts import validate_run

        directory = oof_csv_path.parent
        if (directory / "candidate_manifest.json").is_file():
            from harness.candidates import validate_candidate

            entry = json.loads((directory / "candidate_manifest.json").read_text())
            validate_candidate(self.project, entry)
        else:
            validate_run(directory, task_spec=self.task_spec, plugin=self.task_plugin)
        metrics = json.loads((directory / "metrics.json").read_text())
        return float(metrics["primary_score"]), metrics.get("subtasks", {})

    def _run_subprocess_safe(
        self, cmd: List[str], timeout: float = 1800.0
    ) -> Tuple[bool, int, str, float]:
        """Execute subprocess with strict process tree cleanup on timeout (preventing orphan GPU processes)."""
        t0 = time.time()
        if cmd and cmd[0] == sys.executable and "-u" not in cmd:
            cmd = [cmd[0], "-u", *cmd[1:]]
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=sys.platform != "win32",
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            elapsed = round(time.time() - t0, 3)
            return proc.returncode == 0, proc.returncode or 0, stdout or "", elapsed
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0, 3)
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    check=False,
                    stderr=subprocess.DEVNULL,
                )
            else:
                import os
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
            try:
                stdout, _ = proc.communicate(timeout=5)
            except Exception:
                stdout = ""
            return (
                False,
                -1,
                f"Execution timed out after {timeout}s (PID {proc.pid} terminated).\n{stdout or ''}",
                elapsed,
            )

    def execute_action(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        rationale: str = "",
        session_id: Optional[str] = None,
        goal: Optional[str] = None,
        budget: Optional[Dict[str, Any]] = None,
        user_approved: bool = False,
    ) -> Tuple[BridgeSession, ActionObservation]:
        """Execute a single Typed Action from Antigravity and update the stateful BridgeSession."""
        self._refresh_task_context()
        params = params or {}
        session = self.get_active_or_create_session(
            goal=goal or "Iterative competition optimization via Antigravity IDE",
            session_id=session_id,
            budget=budget,
        )
        journal = self._new_journal(auto_sync=True)
        sub_mgr = self._new_submission_manager()

        step_idx = len(session.actions) + 1
        tier = session.approval_policy.get(
            action, DEFAULT_ACTION_APPROVAL_TIERS.get(action, ApprovalTier.AUTO.value)
        )

        if (
            DEFAULT_ACTION_APPROVAL_TIERS.get(action)
            == ApprovalTier.USER_APPROVAL.value
        ):
            tier = ApprovalTier.USER_APPROVAL.value

        call = TypedActionCall(
            step_index=step_idx,
            action=action,
            approval_tier=tier,
            params=params,
            rationale=rationale or f"Invoked '{action}' via AgentBridge",
            user_approved=user_approved,
        )

        # 1. Check Budget & Stop Guardrails
        try:
            guard_violation = self._check_budget_and_stop_guards(session, action, params)
        except ValueError as guard_error:
            # Contract-level rejection (e.g. execution backend lock violation):
            # record a failed observation instead of raising so the agent loop
            # keeps working and the session history shows the violation.
            obs = ActionObservation(
                step_idx, action, "FAILED", f"Guard rejected the action: {guard_error}"
            )
            session.stop_reason = f"GUARD_REJECTED: {guard_error}"
            session.observations.append(asdict(obs))
            self.save_session(session)
            return session, obs
        if guard_violation is not None:
            v_status, v_msg = guard_violation
            # Do NOT mutate session.status here: a budget block is a per-action
            # observation, not a terminal session state. The session stays in its
            # current state (ACTIVE / PAUSED_FOR_APPROVAL) so the agent can
            # continue with other actions and the budget block is recorded in
            # `stop_reason` only as a transient note. We do record the observation
            # so the agent sees the block in the action history.
            session.stop_reason = f"{v_status}: {v_msg}"
            obs = ActionObservation(
                step_index=step_idx,
                action=action,
                status=v_status,
                summary=v_msg,
                new_blockers=[v_msg],
            )
            # Do NOT append the blocked call to session.actions — the action was
            # not executed. Recording it in actions would inflate step_index for
            # subsequent real actions and confuse the approve flow (C2).
            session.observations.append(asdict(obs))
            self.save_session(session)
            return session, obs

        # 2. Check USER_APPROVAL Gate
        if tier == ApprovalTier.USER_APPROVAL.value and not user_approved:
            session.status = "PAUSED_FOR_APPROVAL"
            session.pending_approval_action = asdict(call)
            session.stop_reason = (
                f"APPROVAL_GATE: Action '{action}' requires explicit USER_APPROVAL."
            )
            obs = ActionObservation(
                step_index=step_idx,
                action=action,
                status="PAUSED_FOR_APPROVAL",
                summary=(
                    f"Action '{action}' is in tier USER_APPROVAL. Execution paused until confirmed by user "
                    f"(pass `--user-approved` or run `ai-harness-agent approve --session-id {session.session_id}`)."
                ),
                data={"pending_action": asdict(call)},
            )
            session.observations.append(asdict(obs))
            self.save_session(session)
            return session, obs

        # Clear pending approval ONLY if the action being executed IS the pending
        # one (i.e. user just approved it). Otherwise a different action is being
        # run while a USER_APPROVAL action is still waiting — leave the pending
        # action and PAUSED_FOR_APPROVAL status intact so the approval is not
        # silently dropped (C1).
        if (
            session.pending_approval_action
            and session.pending_approval_action.get("action") == action
            and user_approved
        ):
            session.pending_approval_action = None
        if session.status == "PAUSED_FOR_APPROVAL" and session.pending_approval_action is None:
            session.status = "ACTIVE"
        session.actions.append(asdict(call))

        resource_actions = {
            "run_full_validation",
            "run_smoke_test",
            "run_probe",
            "promote_candidate",
            "synthesize_candidate",
            "build_ensemble",
            "register_external_prediction",
            "optimize_blend",
            "generate_microdose_probe",
        }
        token = None
        if action == "run_full_validation" and params.get("config_path"):
            cfg_file = self.project_root / params["config_path"]
            if cfg_file.is_file():
                try:
                    self._review_research_config(yaml.safe_load(cfg_file.read_text(encoding="utf-8")), params, for_execution=True)
                except ValueError as error:
                    obs = ActionObservation(step_idx, action, "FAILED", str(error))
                    session.observations.append(asdict(obs))
                    self.save_session(session)
                    return session, obs
        if action in resource_actions:
            from harness.resource_ledger import reserve

            try:
                token, deadline = reserve(
                    self.artifacts_dir / "resource_ledger.json",
                    session.session_id,
                    action,
                    params.get("backend", "local"),
                    session.budget,
                    params.get("_deadline_seconds"),
                )
                if deadline is not None:
                    params["_deadline_seconds"] = deadline
            except ValueError as error:
                obs = ActionObservation(
                    step_index=step_idx,
                    action=action,
                    status="BLOCKED_BY_BUDGET",
                    summary=str(error),
                )
                session.observations.append(asdict(obs))
                self.save_session(session)
                return session, obs
        obs = self._dispatch_typed_action(call, session, journal, sub_mgr)
        if token:
            from harness.resource_ledger import terminal

            if obs.data.get("job_id"):
                from harness.state_store import transaction

                with transaction(self.artifacts_dir / "resource_ledger.json") as ledger:
                    ledger["reservations"][token]["job_id"] = obs.data["job_id"]
                from harness.cloud_job_runner import CloudJobManager

                manager = CloudJobManager(self.project_root)
                job = manager.get_job(obs.data["job_id"])
                if job and job.get("status") in {"COMPLETED", "FAILED", "CANCELLED"}:
                    manager._persist_job(manager._jobs[obs.data["job_id"]])
            else:
                terminal(
                    self.artifacts_dir / "resource_ledger.json",
                    token,
                    "COMPLETED" if obs.status == "SUCCESS" else "FAILED",
                    obs.elapsed_seconds,
                )
        if action in {"run_smoke_test", "run_probe", "run_full_validation"}:
            execution_exp_id = obs.data.get("exp_id")
            execution_insight_id = None
            if params.get("config_path"):
                try:
                    config_file = (self.project_root / params["config_path"]).resolve()
                    config_file.relative_to(self.project_root)
                    execution_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                    execution_exp_id = execution_exp_id or execution_cfg.get("experiment", {}).get("id", config_file.stem)
                    execution_insight_id = (execution_cfg.get("research") or {}).get("insight_id")
                except (OSError, ValueError, TypeError, yaml.YAMLError):
                    pass
            self._research_memory().record_execution(action, obs.status, obs.summary,
                execution_exp_id, session.session_id, insight_id=execution_insight_id)
        if action in {"create_experiment", "write_experiment_config", "run_full_validation",
                      "inspect_cloud_job", "wait_cloud_job", "inspect_diagnostics"}:
            continuity = self._research_memory().context(
                min_improvement=self._get_spec_min_improvement(session),
                patience=int(session.budget.get("max_consecutive_rejections") or 3))
            obs.data["research_continuity"] = continuity
            if continuity["reassess_recommended"]:
                session.reassess_recommended = True
                obs.advisory = "REASSESS_RECOMMENDED"
        session.observations.append(asdict(obs))

        self._reconcile_session_usage(session)

        # Auto-checkpoint every N steps so long sessions don't lose research
        # continuity across compaction. The checkpoint is recorded in the
        # research memory (not the session journal) and is idempotent: it
        # only fires when the step count is a multiple of the interval AND
        # the most recent observation was not already a checkpoint.
        try:
            self._maybe_auto_checkpoint(session, obs)
        except Exception:
            # Auto-checkpoint must never break the main action flow.
            pass

        # Update final snapshot HUD. Reuse the action's journal instance:
        # one idempotent sync pass picks up anything the action just
        # produced (new run dirs), while a brand-new TreeJournal would
        # re-discover the project, reload journal.json and re-validate
        # every run directory a second time. sub_mgr's in-memory state is
        # at least as fresh as the submissions on disk after the action.
        journal.sync_from_artifacts(self.artifacts_dir)
        snap_after = self.comp_state.evaluate_journey_snapshot(journal, sub_mgr)
        session.final_state = snap_after.get("hud", {})
        self.save_session(session)
        return session, obs

    def approve_pending_action(
        self, session_id: str
    ) -> Tuple[BridgeSession, Optional[ActionObservation]]:
        """Approve and execute a paused USER_APPROVAL action in the given BridgeSession."""
        session = self.load_session(session_id)
        if not session.pending_approval_action:
            return session, None

        pending = session.pending_approval_action
        # Remove the PAUSED_FOR_APPROVAL placeholder observation before re-running
        if (
            session.observations
            and session.observations[-1].get("status") == "PAUSED_FOR_APPROVAL"
        ):
            session.observations.pop()
        # Remove the pending action from session.actions so the re-executed
        # action does not appear twice in the action history (C2).
        if session.actions and session.actions[-1].get("action") == pending.get("action"):
            session.actions.pop()
        session.pending_approval_action = None
        session.status = "ACTIVE"
        self.save_session(session)

        return self.execute_action(
            action=pending["action"],
            params=pending.get("params", {}),
            rationale=pending.get("rationale", ""),
            session_id=session.session_id,
            user_approved=True,
        )

    def _dispatch_typed_action(
        self,
        call: TypedActionCall,
        session: BridgeSession,
        journal: TreeJournal,
        sub_mgr: SubmissionManager,
    ) -> ActionObservation:
        """Dispatch a typed agent action to its ``_handle_*`` method.

        The action name maps to a handler via ``_TYPED_ACTION_HANDLER_NAMES``.
        Each handler is a small, independently testable method; adding a new
        action means adding one method + one registry entry, never growing this
        dispatcher. The shared try/except converts any handler exception into a
        FAILED observation so the agent loop never crashes on a bad action.
        """
        t0 = time.time()
        act = call.action
        params = call.params
        try:
            handler_name = self._TYPED_ACTION_HANDLER_NAMES.get(act)
            if handler_name is None:
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="FAILED",
                    summary=f"Unsupported typed action '{act}'.",
                    elapsed_seconds=round(time.time() - t0, 3),
                )
            handler = getattr(self, handler_name)
            return handler(call, session, journal, sub_mgr, act, params, t0)
        except Exception as err:
            return ActionObservation(
                step_index=call.step_index,
                action=act,
                status="FAILED",
                summary=f"Action '{act}' raised exception: {err}",
                elapsed_seconds=round(time.time() - t0, 3),
            )

    # action string -> _handle_* method name. The single source of truth for
    # which typed actions the bridge supports. Multi-action handlers (e.g. the
    # cloud inspect/wait pair) share one method and branch on ``act`` internally.
    _TYPED_ACTION_HANDLER_NAMES = {
        "search_research_memory": "_handle_research_memory_group",
        "review_hypothesis": "_handle_research_memory_group",
        "record_research_decision": "_handle_research_memory_group",
        "record_research_checkpoint": "_handle_research_memory_group",
        "record_research_insight": "_handle_research_memory_group",
        "plan_research_probe": "_handle_research_memory_group",
        "trace_research_insight": "_handle_research_memory_group",
        "suggest_next_interventions": "_handle_suggest_next_interventions",
        "update_workspace_settings": "_handle_update_workspace_settings",
        "update_task_contract": "_handle_update_task_contract",
        "write_experiment_config": "_handle_write_experiment_config",
        "confirm_task_spec": "_handle_confirm_task_spec",
        "run_eda": "_handle_run_eda",
        "record_phase_output": "_handle_record_phase_output",
        "bootstrap_task": "_handle_bootstrap_task",
        "inspect_data_readiness": "_handle_inspect_data_readiness",
        "confirm_data_risks": "_handle_confirm_data_risks",
        "establish_baseline_anchor": "_handle_establish_baseline_anchor",
        "inspect_experiment": "_handle_inspect_experiment",
        "inspect_diagnostics": "_handle_inspect_diagnostics",
        "create_experiment": "_handle_create_experiment",
        "run_smoke_test": "_handle_run_smoke_test",
        "run_probe": "_handle_run_probe",
        "run_full_validation": "_handle_run_full_validation",
        "inspect_cloud_job": "_handle_inspect_cloud_job_or_wait_cloud_job",
        "wait_cloud_job": "_handle_inspect_cloud_job_or_wait_cloud_job",
        "synthesize_candidate": "_handle_synthesize_candidate_or_promote_candidate_or_build_ensemble",
        "promote_candidate": "_handle_synthesize_candidate_or_promote_candidate_or_build_ensemble",
        "build_ensemble": "_handle_synthesize_candidate_or_promote_candidate_or_build_ensemble",
        "register_external_prediction": "_handle_register_external_prediction",
        "optimize_blend": "_handle_optimize_blend",
        "optimize_threshold": "_handle_optimize_threshold",
        "generate_microdose_probe": "_handle_generate_microdose_probe",
        "prepare_submission": "_handle_prepare_submission",
        "record_external_score": "_handle_record_external_score",
        "freeze_candidate": "_handle_freeze_candidate",
        "emergency_unfreeze": "_handle_emergency_unfreeze",
        "finalize_competition": "_handle_finalize_competition",
        "get_next_experiment_id": "_handle_get_next_experiment_id",
    }


    # Alias for execute_action matching CLI 'act' command
    act = execute_action


def main(argv: Optional[List[str]] = None) -> int:
    """Forwarder to tools.agent_bridge_cli.main for module execution."""
    from tools.agent_bridge_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

