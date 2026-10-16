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
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from harness.competition_state import CompetitionStateMachine
from harness.grounding_gate import GroundingGate
from harness.submission_manager import SubmissionManager
from harness.tree_journal import TreeJournal


class ApprovalTier(str, Enum):
    """Governance tiers for Typed Harness Actions."""

    AUTO = "AUTO"
    AUTO_WITH_LIMIT = "AUTO_WITH_LIMIT"
    USER_APPROVAL = "USER_APPROVAL"


class TypedActionName(str, Enum):
    """Canonical Typed Actions exposed by AgentBridge to Antigravity in the IDE."""

    UPDATE_TASK_CONTRACT = "update_task_contract"
    SEARCH_RESEARCH_MEMORY = "search_research_memory"
    REVIEW_HYPOTHESIS = "review_hypothesis"
    SUGGEST_NEXT_INTERVENTIONS = "suggest_next_interventions"
    RECORD_RESEARCH_DECISION = "record_research_decision"
    RECORD_RESEARCH_CHECKPOINT = "record_research_checkpoint"
    RECORD_RESEARCH_INSIGHT = "record_research_insight"
    PLAN_RESEARCH_PROBE = "plan_research_probe"
    TRACE_RESEARCH_INSIGHT = "trace_research_insight"
    WRITE_EXPERIMENT_CONFIG = "write_experiment_config"
    UPDATE_WORKSPACE_SETTINGS = "update_workspace_settings"
    BOOTSTRAP_TASK = "bootstrap_task"
    CONFIRM_TASK_SPEC = "confirm_task_spec"
    INSPECT_DATA_READINESS = "inspect_data_readiness"
    CONFIRM_DATA_RISKS = "confirm_data_risks"
    ESTABLISH_BASELINE_ANCHOR = "establish_baseline_anchor"
    INSPECT_EXPERIMENT = "inspect_experiment"
    INSPECT_DIAGNOSTICS = "inspect_diagnostics"
    CREATE_EXPERIMENT = "create_experiment"
    RUN_EDA = "run_eda"
    RECORD_PHASE_OUTPUT = "record_phase_output"
    INSPECT_CLOUD_JOB = "inspect_cloud_job"
    WAIT_CLOUD_JOB = "wait_cloud_job"
    RUN_SMOKE_TEST = "run_smoke_test"
    RUN_PROBE = "run_probe"
    RUN_FULL_VALIDATION = "run_full_validation"
    SYNTHESIZE_CANDIDATE = "synthesize_candidate"
    PROMOTE_CANDIDATE = "promote_candidate"
    BUILD_ENSEMBLE = "build_ensemble"
    REGISTER_EXTERNAL_PREDICTION = "register_external_prediction"
    OPTIMIZE_BLEND = "optimize_blend"
    OPTIMIZE_THRESHOLD = "optimize_threshold"
    GENERATE_MICRODOSE_PROBE = "generate_microdose_probe"
    PREPARE_SUBMISSION = "prepare_submission"
    RECORD_EXTERNAL_SCORE = "record_external_score"
    FREEZE_CANDIDATE = "freeze_candidate"
    EMERGENCY_UNFREEZE = "emergency_unfreeze"
    FINALIZE_COMPETITION = "finalize_competition"
    GET_NEXT_EXPERIMENT_ID = "get_next_experiment_id"


DEFAULT_ACTION_APPROVAL_TIERS: Dict[str, str] = {
    TypedActionName.SEARCH_RESEARCH_MEMORY.value: ApprovalTier.AUTO.value,
    TypedActionName.REVIEW_HYPOTHESIS.value: ApprovalTier.AUTO.value,
    TypedActionName.SUGGEST_NEXT_INTERVENTIONS.value: ApprovalTier.AUTO.value,
    TypedActionName.RECORD_RESEARCH_DECISION.value: ApprovalTier.AUTO.value,
    TypedActionName.RECORD_RESEARCH_CHECKPOINT.value: ApprovalTier.AUTO.value,
    TypedActionName.RECORD_RESEARCH_INSIGHT.value: ApprovalTier.AUTO.value,
    TypedActionName.PLAN_RESEARCH_PROBE.value: ApprovalTier.AUTO.value,
    TypedActionName.TRACE_RESEARCH_INSIGHT.value: ApprovalTier.AUTO.value,
    TypedActionName.BOOTSTRAP_TASK.value: ApprovalTier.AUTO.value,
    TypedActionName.CONFIRM_TASK_SPEC.value: ApprovalTier.USER_APPROVAL.value,
    TypedActionName.INSPECT_DATA_READINESS.value: ApprovalTier.AUTO.value,
    TypedActionName.CONFIRM_DATA_RISKS.value: ApprovalTier.USER_APPROVAL.value,
    TypedActionName.INSPECT_EXPERIMENT.value: ApprovalTier.AUTO.value,
    TypedActionName.INSPECT_DIAGNOSTICS.value: ApprovalTier.AUTO.value,
    TypedActionName.CREATE_EXPERIMENT.value: ApprovalTier.AUTO.value,
    TypedActionName.GET_NEXT_EXPERIMENT_ID.value: ApprovalTier.AUTO.value,
    TypedActionName.RUN_EDA.value: ApprovalTier.AUTO.value,
    TypedActionName.RECORD_PHASE_OUTPUT.value: ApprovalTier.AUTO.value,
    TypedActionName.INSPECT_CLOUD_JOB.value: ApprovalTier.AUTO.value,
    TypedActionName.WAIT_CLOUD_JOB.value: ApprovalTier.AUTO.value,
    TypedActionName.RUN_SMOKE_TEST.value: ApprovalTier.AUTO.value,
    TypedActionName.RUN_PROBE.value: ApprovalTier.AUTO.value,
    TypedActionName.ESTABLISH_BASELINE_ANCHOR.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.RUN_FULL_VALIDATION.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.SYNTHESIZE_CANDIDATE.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.PROMOTE_CANDIDATE.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.BUILD_ENSEMBLE.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.REGISTER_EXTERNAL_PREDICTION.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.OPTIMIZE_BLEND.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.OPTIMIZE_THRESHOLD.value: ApprovalTier.AUTO.value,
    TypedActionName.GENERATE_MICRODOSE_PROBE.value: ApprovalTier.AUTO_WITH_LIMIT.value,
    TypedActionName.PREPARE_SUBMISSION.value: ApprovalTier.USER_APPROVAL.value,
    TypedActionName.RECORD_EXTERNAL_SCORE.value: ApprovalTier.USER_APPROVAL.value,
    TypedActionName.FREEZE_CANDIDATE.value: ApprovalTier.USER_APPROVAL.value,
    TypedActionName.EMERGENCY_UNFREEZE.value: ApprovalTier.USER_APPROVAL.value,
    TypedActionName.FINALIZE_COMPETITION.value: ApprovalTier.USER_APPROVAL.value,
}


@dataclass
class SessionBudget:
    """Optional session-level resource & stopping limits enforced by AgentBridge."""

    max_experiments: Optional[int] = None
    max_probes: Optional[int] = None
    max_candidate_operations: Optional[int] = None
    max_gpu_hours: Optional[float] = None
    max_cloud_runs: Optional[int] = None
    max_wall_seconds: Optional[float] = None
    max_submissions: Optional[int] = None
    max_consecutive_rejections: int = 3
    hard_stop_on_consecutive_rejections: bool = False
    min_meaningful_improvement: Optional[float] = None


@dataclass
class ComputeUsage:
    """Cumulative compute and execution telemetry for a BridgeSession."""

    experiments_used: int = 0
    local_runs: int = 0
    cloud_runs: int = 0
    smoke_runs: int = 0
    wall_seconds_used: float = 0.0
    gpu_hours_used: Optional[float] = None
    submissions_used: int = 0


@dataclass
class TypedActionCall:
    """A single typed action invoked by Antigravity through AgentBridge."""

    step_index: int
    action: str
    approval_tier: str
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    user_approved: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@dataclass
class ActionObservation:
    """Structured observation returned by AgentBridge to Antigravity after an action."""

    step_index: int
    action: str
    status: str  # "SUCCESS" | "FAILED" | "PAUSED_FOR_APPROVAL" | "BLOCKED_BY_BUDGET" | "STOP_POLICY_TRIGGERED"
    summary: str
    advisory: Optional[str] = None  # e.g. "REASSESS_RECOMMENDED"
    metric_delta: Optional[float] = None
    composite_score: Optional[float] = None
    fold_variance: Optional[float] = None
    regressions: List[str] = field(default_factory=list)
    diagnostics_bottlenecks: List[str] = field(default_factory=list)
    new_blockers: List[str] = field(default_factory=list)
    artifact_path: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class BridgeSession:
    """Stateful session tracking Antigravity's interaction with the Harness."""

    session_id: str
    goal: str
    status: str  # "ACTIVE" | "PAUSED_FOR_APPROVAL" | "BUDGET_REACHED" | "STOPPED" | "COMPLETED"
    created_at: str
    updated_at: str
    active: bool = True
    starting_state: Dict[str, Any] = field(default_factory=dict)
    final_state: Optional[Dict[str, Any]] = None
    experiments_run: List[str] = field(default_factory=list)
    compute_usage: Dict[str, Any] = field(
        default_factory=lambda: asdict(ComputeUsage())
    )
    consecutive_rejections: int = 0
    reassess_recommended: bool = False
    budget: Dict[str, Any] = field(default_factory=lambda: asdict(SessionBudget()))
    approval_policy: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ACTION_APPROVAL_TIERS)
    )
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    pending_approval_action: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None


class AgentBridge:
    """Bidirectional Context & Typed Action Bridge between Antigravity (IDE) and the ML Harness.

    Motto: Antigravity thinks. AgentBridge coordinates. Harness verifies and executes.
    """

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
        import hashlib
        import pandas as pd

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
        import pandas as pd
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

    def get_active_session_id(self) -> Optional[str]:
        """Return the session_id currently marked active, or None."""
        for item in self.list_sessions(limit=20):
            if item.get("active") and item.get("session_id"):
                return item["session_id"]
        return None

    def save_session(self, session: BridgeSession) -> None:
        if (
            Path(session.session_id).name != session.session_id
            or "/" in session.session_id
            or "\\" in session.session_id
        ):
            raise ValueError("Invalid session ID")
        session.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = asdict(session)
        path = self.sessions_dir / f"{session.session_id}.json"

        # Enforce a single active session per workspace. When the session being
        # saved is active, demote any other previously-active sessions so that
        # `get_active_or_create_session` has a stable target to resume.
        if payload.get("active"):
            self._claim_active_session(session.session_id)
            payload["active"] = True
        from harness.io_utils import atomic_write_json
        import copy
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
                sum(gpu_hours) if gpu_hours else (0.0 if not terminal_rows else None)
            ),
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
        lines.extend(["", "## Research continuity (survives sessions and compaction)",
                      json.dumps({key: memory[key] for key in (
                          "entry_count", "verified_run_count", "latest_checkpoint", "recent_decisions",
                          "recent_results", "consecutive_non_improvements", "reassess_recommended",
                          "insight_count", "open_insight_count", "research_frontier",
                          "unreviewed_result_count", "unreviewed_results", "progress_path", "progress_warning")},
                          ensure_ascii=False),
                      *[f"- {step}" for step in memory["resume_protocol"]]])

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
        """
        # 1. Hard Stop: Freeze lock guard
        if self.comp_state.state["gates"].get("freeze_locked") and action not in (
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
            return (
                "STOP_POLICY_TRIGGERED",
                "Competition state is FROZEN. Only inspection, research notes, finalization, or `emergency_unfreeze` (with USER_APPROVAL) is allowed.",
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
                import os, signal

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
        guard_violation = self._check_budget_and_stop_guards(session, action, params)
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

        # Update final snapshot HUD
        snap_after = self.comp_state.evaluate_journey_snapshot(
            self._new_journal(auto_sync=True), self._new_submission_manager()
        )
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
        t0 = time.time()
        act = call.action
        params = call.params

        try:
            if act in {"search_research_memory", "review_hypothesis", "record_research_decision",
                       "record_research_checkpoint", "record_research_insight", "plan_research_probe",
                       "trace_research_insight"}:
                memory = self._research_memory()
                if act == "search_research_memory":
                    result = {"matches": memory.search(params.get("query", ""), params.get("limit", 8))}
                elif act == "record_research_insight":
                    result = memory.record_insight(title=params["title"], observation=params["observation"],
                        mechanism=params["mechanism"], next_action=params["next_action"],
                        evidence=params.get("evidence"), parent_insight_id=params.get("parent_insight_id"),
                        session_id=session.session_id)
                elif act == "plan_research_probe":
                    result = memory.plan_probe(**{key: params[key] for key in (
                        "insight_id", "control", "intervention", "expected_observation", "alternative_explanation",
                        "decision_rule", "budget", "stop_rule")}, session_id=session.session_id)
                elif act == "trace_research_insight":
                    result = memory.trace(params["insight_id"], params.get("offset", 0), params.get("limit", 20))
                elif act == "review_hypothesis":
                    cfg = params.get("config")
                    if cfg is None:
                        path = (self.project_root / params["config_path"]).resolve()
                        path.relative_to(self.project_root)
                        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                    result = memory.review(cfg, params.get("revisit_reason", ""))
                elif act == "record_research_decision":
                    result = memory.record_decision(outcome=params["outcome"], conclusion=params["conclusion"],
                        evidence=params.get("evidence", []), next_action=params.get("next_action", ""),
                        insight_id=params.get("insight_id"), lesson=params.get("lesson", ""),
                        revisit_when=params.get("revisit_when", ""),
                        session_id=session.session_id)
                else:
                    result = memory.checkpoint(summary=params["summary"], next_action=params["next_action"],
                        open_questions=params.get("open_questions", []), session_id=session.session_id,
                        insight_ids=params.get("insight_ids"))
                return ActionObservation(call.step_index, act, "SUCCESS", "Research memory action completed",
                                         data=result, elapsed_seconds=round(time.time() - t0, 3))
            if act == TypedActionName.SUGGEST_NEXT_INTERVENTIONS.value:
                suggestions = self._suggest_next_interventions(limit=params.get("limit", 5))
                n = len(suggestions.get("suggestions", []))
                if suggestions["stagnation"].get("reassess_recommended"):
                    summary = (f"Stagnation detected ({suggestions['stagnation']['consecutive_non_improvements']} "
                               f"consecutive non-improvements). {n} untried intervention(s) suggested.")
                else:
                    summary = f"{n} untried intervention(s) suggested to continue brainstorming."
                return ActionObservation(call.step_index, act, "SUCCESS", summary,
                                         data=suggestions, elapsed_seconds=round(time.time() - t0, 3))
            if act == "update_workspace_settings":
                if params.get("gates"):
                    raise ValueError("Gates require their verified actions")
                for key in ("mode", "manual_phase_override", "agent_context_config"):
                    if key in params:
                        self.comp_state.state[key] = params[key]
                cid = params.get("toggle_pin_candidate")
                if cid:
                    pinned = self.comp_state.state["candidate_pool"][
                        "pinned_candidates"
                    ]
                    if cid in pinned:
                        pinned.remove(cid)
                    else:
                        pinned.append(cid)
                self.comp_state.save()
                return ActionObservation(
                    call.step_index, act, "SUCCESS", "Workspace settings updated"
                )
            if act == "update_task_contract":
                from harness.io_utils import atomic_write_text
                from harness.task_spec import UniversalTaskSpec

                document = params["task"]
                UniversalTaskSpec.from_dict({"task": document})
                atomic_write_text(
                    self.project.task_config_path,
                    yaml.safe_dump(
                        {"task": document}, sort_keys=False, allow_unicode=True
                    ),
                )
                self._refresh_task_context()
                return ActionObservation(
                    call.step_index,
                    act,
                    "SUCCESS",
                    "Task contract updated; evidence will be rechecked",
                )
            if act == "write_experiment_config":
                from harness.io_utils import atomic_write_text

                path = (self.project_root / params["path"]).resolve()
                path.relative_to(self.project_root / "configs")
                if path == self.project.task_config_path:
                    raise ValueError("Use update_task_contract")
                cfg = yaml.safe_load(params["content"])
                GroundingGate.verify_config(
                    cfg, config_path=str(path), raise_on_error=True
                )
                exp = cfg.get("experiment", {}).get("id", path.stem)
                cfg.setdefault("experiment", {})["id"] = exp
                if params.get("insight_id"):
                    cfg.setdefault("research", {})["insight_id"] = params["insight_id"]
                if (self.artifacts_dir / exp / "artifact_manifest.json").is_file():
                    raise ValueError(
                        "Use a new experiment ID to change a run configuration"
                    )
                from harness.state_store import workspace_lock

                with workspace_lock(self.artifacts_dir):
                    review = self._review_research_config(cfg, params)
                    if review["revisit_reason"]:
                        cfg.setdefault("research", {})["revisit_reason"] = review["revisit_reason"]
                    atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
                    self._research_memory().propose(cfg, review["revisit_reason"], session.session_id)
                return ActionObservation(
                    call.step_index,
                    act,
                    "SUCCESS",
                    "Experiment config saved",
                    artifact_path=str(path),
                    data={"novelty_review": review},
                )
            if act == TypedActionName.CONFIRM_TASK_SPEC.value:
                self.comp_state.state["gates"]["spec_confirmed"] = True
                self.comp_state._append_audit_event(
                    "TASK_SPEC_CONFIRMED",
                    "A user approved the attached task specification.",
                    {"task_config": str(self.project.task_config_path)},
                )
                self.comp_state.save()
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary="Task contract marked reviewed and confirmed.",
                    artifact_path=str(
                        self.project.task_config_path.relative_to(self.project_root)
                    ).replace("\\", "/"),
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.RUN_EDA.value:
                from harness.eda import run_eda

                result = run_eda(self.project_root, data_path=params.get("data_path"))
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"EDA profiled {len(result.get('reports', []))} dataset file(s) without copying raw rows.",
                    artifact_path=result["paths"]["markdown"],
                    data={
                        "json_report": result["paths"]["json"],
                        "reports": result.get("reports", []),
                    },
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.RECORD_PHASE_OUTPUT.value:
                from harness.phase_workflow import PhaseWorkflow

                phase_id = params.get("phase_id")
                response = params.get("response")
                response_file = params.get("response_file")
                if response is None and response_file:
                    response_path = (self.project_root / response_file).resolve()
                    response_path.relative_to(self.project_root)
                    response = response_path.read_text(encoding="utf-8")
                if not isinstance(response, str):
                    raise ValueError(
                        "Provide `response` or a workspace-relative `response_file`."
                    )
                workflow = PhaseWorkflow(
                    self.project_root, self.project.task_config_path
                )
                record = workflow.record(
                    str(phase_id or ""),
                    response,
                    sources=params.get("sources", []),
                    artifacts=params.get("artifacts", []),
                    metadata=params.get("metadata", {}),
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Recorded phase `{record['phase_id']}` response with {len(record['sources'])} source(s) and {len(record['artifacts'])} artifact(s).",
                    artifact_path=record["response_path"],
                    data=record,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            if act == TypedActionName.BOOTSTRAP_TASK.value:
                from tools.bootstrap_task import bootstrap_competition_task

                dataset_dir = params.get("dataset_dir", "dataset")
                context_file = params.get("context_file", "docs/TASK.md")
                output_config = params.get("output_config", "configs/task.yaml")
                proj_root = params.get("project_root") or self.project_root
                res = bootstrap_competition_task(
                    dataset_dir=dataset_dir,
                    context_file=context_file,
                    output_config=output_config,
                    project_root=proj_root,
                )
                self._refresh_task_context()
                self.comp_state.state["gates"]["spec_confirmed"] = False
                self.comp_state.state["gates"]["data_risks_confirmed"] = False
                self.comp_state.save()
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Bootstrapped task '{res.get('task_name')}' ({res.get('modality')}, {res.get('task_type')}) with {res.get('samples_count')} samples.",
                    data=res,
                    artifact_path=res.get("config_file"),
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.INSPECT_DATA_READINESS.value:
                report = self.comp_state.run_data_readiness_audit(params.get("config"))
                warns = [w.get("title", "") for w in report.get("warnings", [])]
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Executed {len(report.get('checks_performed', []))} declarative data checks; {len(warns)} warnings logged.",
                    diagnostics_bottlenecks=warns,
                    artifact_path=str(
                        self.comp_state.data_report_file.relative_to(self.project_root)
                    ).replace("\\", "/"),
                    data={
                        "checks_performed": report.get("checks_performed", []),
                        "warnings": warns,
                    },
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.CONFIRM_DATA_RISKS.value:
                if not self.comp_state.state["gates"].get("spec_confirmed"):
                    raise ValueError(
                        "Review and confirm the task specification before confirming data risks."
                    )
                report = self.comp_state.get_data_readiness_report()
                if report.get("status") == "BLOCKED":
                    raise ValueError(
                        "The latest data readiness report is BLOCKED; fix its critical checks first."
                    )
                self.comp_state.state["gates"]["data_risks_confirmed"] = True
                self.comp_state._append_audit_event(
                    "DATA_RISKS_CONFIRMED",
                    "A user reviewed and confirmed the latest data readiness report.",
                    {
                        "report_status": report.get("status"),
                        "report_path": str(self.comp_state.data_report_file),
                    },
                )
                self.comp_state.save()
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Data readiness report ({report.get('status')}) reviewed and confirmed.",
                    artifact_path=str(
                        self.comp_state.data_report_file.relative_to(self.project_root)
                    ).replace("\\", "/"),
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.ESTABLISH_BASELINE_ANCHOR.value:
                gates = self.comp_state.state.get("gates", {})
                if not gates.get("spec_confirmed") or not gates.get(
                    "data_risks_confirmed"
                ):
                    raise ValueError(
                        "Review the task contract and confirm data risks before establishing the baseline."
                    )
                pool = self.comp_state.compute_candidate_pool(journal)
                default_base_id = (
                    self.comp_state.state["gates"].get("baseline_exp_id")
                    or self.comp_state.spec.get("baseline_anchor", {}).get(
                        "default_exp_id"
                    )
                    or "baseline"
                )
                base_id = params.get("baseline_id") or pool.get(
                    "baseline_id", default_base_id
                )
                from harness.baseline import validate_baseline_artifact

                baseline = validate_baseline_artifact(
                    self.artifacts_dir,
                    str(base_id),
                    str(self.comp_state.spec.get("primary_metric") or "score"),
                    task_spec=self.task_spec,
                    plugin=self.task_plugin,
                )
                base_score = float(baseline["score"])
                self.comp_state.state["gates"]["baseline_locked"] = True
                self.comp_state.state["gates"]["baseline_exp_id"] = base_id
                self.comp_state.state["gates"]["baseline_evidence"] = baseline
                self.comp_state.save()
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Locked official baseline anchor '{base_id}' at {base_score:.4f}.",
                    composite_score=base_score,
                    metric_delta=0.0,
                    fold_variance=None,
                    artifact_path=str(
                        Path(baseline["metrics_path"]).relative_to(self.project_root)
                    ).replace("\\", "/"),
                    data={
                        "baseline_id": base_id,
                        "baseline_score": base_score,
                        "evidence": baseline,
                    },
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.INSPECT_EXPERIMENT.value:
                default_exp_id = (
                    (pool.get("overall_best") or {}).get("id")
                    or self.comp_state.state["gates"].get("baseline_exp_id")
                    or "baseline"
                )
                exp_id = params.get("exp_id", default_exp_id)
                pool = self.comp_state.compute_candidate_pool(journal)
                cand = next(
                    (c for c in pool["all_candidates"] if c["id"] == exp_id), None
                ) or (pool["overall_best"] if pool["overall_best"] else {})
                cand_score = float(
                    cand.get("composite_score")
                    or cand.get("score")
                    or self.comp_state.spec.get("primary_metric_baseline")
                    or 0.0
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Inspected '{cand.get('id', exp_id)}': score={cand_score:.4f}, std={cand.get('fold_std', 0.015)}.",
                    composite_score=cand_score,
                    metric_delta=cand.get("delta_vs_baseline", 0.0),
                    fold_variance=cand.get("fold_std", 0.015),
                    data=cand,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.INSPECT_DIAGNOSTICS.value:
                pool = self.comp_state.compute_candidate_pool(journal)
                default_exp_id = (
                    (pool.get("overall_best") or {}).get("id")
                    or self.comp_state.state["gates"].get("baseline_exp_id")
                    or "baseline"
                )
                exp_id = params.get("exp_id", default_exp_id)
                diag_file = self.artifacts_dir / exp_id / "DIAGNOSTICS.md"
                node = journal.nodes.get(exp_id)
                bottlenecks = []
                if node and node.verified and isinstance(node.diagnostics_summary, dict):
                    bottlenecks = node.diagnostics_summary.get(
                        "critical_bottlenecks", []
                    )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Found {len(bottlenecks)} verified diagnostic bottlenecks from '{exp_id}'; missing diagnostics are not evidence of no errors.",
                    diagnostics_bottlenecks=bottlenecks,
                    artifact_path=(
                        str(diag_file.relative_to(self.project_root)).replace("\\", "/")
                        if diag_file.exists()
                        else None
                    ),
                    data={"exp_id": exp_id, "bottlenecks": bottlenecks},
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.CREATE_EXPERIMENT.value:
                pool = self.comp_state.compute_candidate_pool(journal)
                branch = params.get("branch", "feat")
                descriptor = params.get("descriptor", "iteration")
                default_parent = (
                    (pool.get("overall_best") or {}).get("id")
                    or self.comp_state.state["gates"].get("baseline_exp_id")
                    or "baseline"
                )
                parent_id = params.get("parent_id", default_parent)
                configs_dir = self.project_root / "configs"
                configs_dir.mkdir(parents=True, exist_ok=True)

                exp_id = params.get("exp_id")
                if not exp_id:
                    next_num = self.get_next_experiment_num()
                    exp_id = f"exp{next_num:03d}_{branch}_{descriptor}"
                else:
                    # Check if exp_id collides with existing immutable artifacts
                    for hidden in [".ai-harness", ".ml-harness"]:
                        art_root = self.project_root / hidden / "artifacts"
                        if art_root.exists():
                            for cand_dir in art_root.iterdir():
                                if cand_dir.is_dir() and (cand_dir / exp_id).exists():
                                    next_num = self.get_next_experiment_num()
                                    return ActionObservation(
                                        step_index=call.step_index,
                                        action=act,
                                        status="FAILED",
                                        summary=(
                                            f"Experiment artifacts for '{exp_id}' already exist at '{cand_dir / exp_id}'. "
                                            f"Completed runs are immutable under the Correctness Contract. "
                                            f"Use next available ID: exp{next_num:03d}_{branch}_{descriptor}"
                                        ),
                                        elapsed_seconds=round(time.time() - t0, 3),
                                    )

                cfg_path = configs_dir / f"{exp_id}.yaml"
                uncertainty = str(params.get("uncertainty_level", "low")).lower()
                task_name = self.comp_state.spec.get("name", "Competition Task")
                primary_metric = self.comp_state.spec.get("primary_metric", "score")
                parent_node = (
                    journal.get_node(parent_id)
                    if hasattr(journal, "get_node")
                    else journal.nodes.get(parent_id)
                )
                parent_score = (
                    parent_node.get_score()
                    if parent_node
                    else float(
                        self.comp_state.spec.get("primary_metric_baseline") or 0.0
                    )
                )

                modality = (
                    getattr(self.task_spec, "modality", None)
                    or self.comp_state.spec.get("modality", "")
                ).lower()
                task_type = (
                    getattr(self.task_spec, "task_type", None)
                    or self.comp_state.spec.get("task_type", "")
                ).lower()
                is_reg = (
                    task_type in ("regression", "ranking")
                    or "regression" in task_type
                )

                if "input_variants" in params:
                    input_variants = params["input_variants"]
                elif (
                    hasattr(self, "task_spec")
                    and self.task_spec
                    and getattr(self.task_spec, "inputs_dict", None)
                    and self.task_spec.inputs_dict
                ):
                    input_variants = list(self.task_spec.inputs_dict.keys())
                else:
                    input_variants = ["default"]

                # Resolve model configuration dynamically from params or task_spec
                spec_model_cfg = None
                if hasattr(self, "task_spec") and self.task_spec:
                    spec_meta = getattr(self.task_spec, "metadata", {}) or {}
                    spec_model_cfg = (
                        spec_meta.get("model")
                        or spec_meta.get("default_model")
                        or spec_meta.get("baseline_model")
                    )
                if not spec_model_cfg and hasattr(self, "comp_state") and self.comp_state:
                    spec_model_cfg = (
                        self.comp_state.spec.get("model")
                        or self.comp_state.spec.get("default_model")
                        or self.comp_state.spec.get("baseline_model")
                    )

                spec_name = None
                spec_params = {}
                if isinstance(spec_model_cfg, dict):
                    spec_name = spec_model_cfg.get("name")
                    spec_params = spec_model_cfg.get("params", {})
                elif isinstance(spec_model_cfg, str):
                    spec_name = spec_model_cfg

                params_model_cfg = params.get("model")
                if isinstance(params_model_cfg, dict):
                    model_payload = dict(params_model_cfg)
                    if "model_params" in params and isinstance(params["model_params"], dict):
                        model_payload.setdefault("params", {}).update(params["model_params"])
                elif isinstance(params_model_cfg, str):
                    model_payload = {
                        "name": params_model_cfg,
                        "params": dict(params.get("model_params", {})),
                    }
                elif "model_name" in params:
                    model_payload = {
                        "name": params["model_name"],
                        "params": dict(params.get("model_params", {})),
                    }
                elif spec_model_cfg:
                    if isinstance(spec_model_cfg, dict):
                        model_payload = dict(spec_model_cfg)
                    else:
                        model_payload = {"name": spec_model_cfg, "params": {}}
                    if "model_params" in params and isinstance(params["model_params"], dict):
                        model_payload.setdefault("params", {}).update(params["model_params"])
                elif modality == "tabular":
                    model_payload = {
                        "name": "lightgbm",
                        "params": {
                            "n_estimators": 100,
                            "learning_rate": 0.05,
                            "num_leaves": 31,
                        },
                    }
                    if "model_params" in params and isinstance(params["model_params"], dict):
                        model_payload["params"].update(params["model_params"])
                elif modality in {"text", "nlp"}:
                    if is_reg:
                        model_payload = {
                            "name": "tfidf_ridge",
                            "params": {"alpha": 1.0},
                        }
                    else:
                        model_payload = {
                            "name": "tfidf_logistic_regression",
                            "params": {
                                "C": 1.5,
                            },
                        }
                    if "model_params" in params and isinstance(params["model_params"], dict):
                        model_payload["params"].update(params["model_params"])
                else:
                    model_payload = {
                        "name": "baseline",
                        "params": {},
                    }
                    if "model_params" in params and isinstance(params["model_params"], dict):
                        model_payload["params"].update(params["model_params"])

                if "name" not in model_payload and "factory" not in model_payload:
                    model_payload["name"] = spec_name or ("lightgbm" if modality == "tabular" else "baseline")
                if "params" not in model_payload:
                    model_payload["params"] = {}

                # Strict GroundingGate-compliant YAML structure (task-agnostic defaults)
                cfg_payload: Dict[str, Any] = {
                    "experiment": {
                        "id": exp_id,
                        "parent_id": parent_id,
                        "branch": branch,
                        "problem_addressed": params.get(
                            "problem_addressed",
                            f"Test a declared intervention on '{task_name}' ({primary_metric}); inspect verified parent diagnostics before implementation.",
                        ),
                        "hypothesis": params.get(
                            "hypothesis",
                            f"Targeted '{branch}' intervention ({descriptor}) improves '{primary_metric}' over parent '{parent_id}' by >= {self._get_spec_min_improvement(session):.4f}.",
                        ),
                        "intervention": params.get(
                            "intervention",
                            f"Apply '{descriptor}' configuration under branch '{branch}' while keeping CV split alignment fixed.",
                        ),
                        "description": f"Created via AgentBridge session {session.session_id}",
                    },
                    "grounding": {
                        "source_type": params.get("source_type", "empirical_journal"),
                        "source_id": params.get(
                            "source_id", "research_memory:unverified_proposal"
                        ),
                        "retrieval_summary": params.get(
                            "retrieval_summary",
                            f"Declared proposal targeting {primary_metric} on {task_name}; mechanism and source require scientist review.",
                        ),
                        # Anti-hallucination: do NOT fabricate a 'verified' channel.
                        # If the agent did not supply a real verified_via, mark the
                        # source as NOT verified so the GroundingGate / execution gate
                        # can refuse to trust an ungrounded proposal. A fabricated
                        # 'local_registry' here is exactly the bypass the gate exists
                        # to stop (see GroundingGate.VERIFIED_VIA comment).
                        "verified_via": params.get("verified_via", "not_verified"),
                    },
                    "assumption_register": {
                        "known_facts": params.get(
                            "known_facts",
                            [
                                (f"Verified parent '{parent_id}' achieved {primary_metric}={parent_score:.4f}."
                                 if parent_node and parent_node.verified else
                                 f"Task contract declares primary metric '{primary_metric}'; parent performance is not verified."),
                            ],
                        ),
                        "assumptions": params.get(
                            "assumptions",
                            [
                                f"Intervention '{descriptor}' addresses primary error slices without increasing fold variance.",
                            ],
                        ),
                        "uncertainty_level": uncertainty,
                        "verification_probe": params.get(
                            "verification_probe",
                            f"Run single-fold smoke test: ai-harness-run --project-root . --task-config {self.project.task_config_path.relative_to(self.project_root)} --config configs/{exp_id}.yaml --debug --fold 0",
                        ),
                    },
                    "data": {
                        "folds": params.get("folds", [0, 1, 2, 3, 4]),
                        "input_variants": input_variants,
                        "selection_variant": params.get("selection_variant", input_variants[0]),
                    },
                    "model": model_payload,
                }

                # Pre-verify with GroundingGate. B1-a: do NOT raise here so the
                # config is still written (keeping the create_experiment workflow
                # intact); instead we surface the grounding state in the
                # observation. The real circuit breaker is harness.runner.run(),
                # which still calls GroundingGate.verify_config(raise_on_error=True)
                # at the start of _execute_run() and refuses to train on an
                # unverified non-baseline proposal. This way an agent cannot create
                # a config with a fabricated 'verified_via' AND run it.
                grounding_ok, grounding_errors, grounding_receipt = GroundingGate.verify_config(
                    cfg_payload, config_path=cfg_path, raise_on_error=False
                )
                from harness.state_store import workspace_lock
                from harness.io_utils import atomic_write_text

                if params.get("insight_id"):
                    cfg_payload.setdefault("research", {})["insight_id"] = params["insight_id"]
                cfg_path.resolve().relative_to(configs_dir.resolve())
                if cfg_path.exists() or (self.artifacts_dir / exp_id / "artifact_manifest.json").exists():
                    raise ValueError("Experiment ID already exists; choose a new ID")
                with workspace_lock(self.artifacts_dir):
                    if cfg_path.exists():
                        raise ValueError("Experiment ID already exists; choose a new ID")
                    review = self._review_research_config(cfg_payload, params)
                    if review["revisit_reason"]:
                        cfg_payload.setdefault("research", {})["revisit_reason"] = review["revisit_reason"]
                    atomic_write_text(cfg_path, yaml.safe_dump(cfg_payload, sort_keys=False, allow_unicode=True))
                    self._research_memory().propose(cfg_payload, review["revisit_reason"], session.session_id)
                rel_cfg = str(cfg_path.relative_to(self.project_root)).replace(
                    "\\", "/"
                )

                if grounding_ok:
                    grounding_status = "PASSED"
                    grounding_verification = "declared_source"
                    grounding_summary = (
                        f"Created config '{rel_cfg}' with declared grounding (Parent: {parent_id})."
                    )
                else:
                    grounding_status = "UNVERIFIED"
                    grounding_verification = "unverified_placeholder"
                    grounding_summary = (
                        f"Created config '{rel_cfg}' (Parent: {parent_id}) but the grounding source is "
                        f"NOT verified. The runner will REFUSE to train this until you supply a real "
                        f"'grounding.source_id' + 'grounding.verified_via' (verify the source first). "
                        f"Grounding issues: {grounding_errors}"
                    )

                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=grounding_summary,
                    artifact_path=rel_cfg,
                    data={
                        "exp_id": exp_id,
                        "config_path": rel_cfg,
                        "grounding_status": grounding_status,
                        "grounding_verification": grounding_verification,
                        "grounding_errors": grounding_errors,
                        "novelty_review": review,
                    },
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.RUN_SMOKE_TEST.value:
                cfg_path = params.get("config_path")
                if not cfg_path:
                    available = sorted(
                        list((self.project_root / "configs").glob("exp*.yaml"))
                    )
                    cfg_path = (
                        str(available[0].relative_to(self.project_root)).replace(
                            "\\", "/"
                        )
                        if available
                        else "configs/baseline.yaml"
                    )
                fold = str(params.get("fold", 0))
                cmd = [
                    sys.executable,
                    "-m",
                    "harness.runner",
                    "--project-root",
                    str(self.project_root),
                    "--task-config",
                    str(self.project.task_config_path),
                    "--config",
                    cfg_path,
                    "--debug",
                    "--fold",
                    fold,
                    "--skip-leaderboard",
                ]
                passed, retcode, out_text, elapsed = self._run_subprocess_safe(
                    cmd, timeout=min(180.0, float(params.get("_deadline_seconds", 180)))
                )
                session.compute_usage["smoke_runs"] = (
                    int(session.compute_usage.get("smoke_runs", 0)) + 1
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS" if passed else "FAILED",
                    summary=f"Smoke test on Fold {fold} ({cfg_path}): {'PASSED' if passed else 'FAILED'}.",
                    artifact_path=cfg_path,
                    data={"returncode": retcode, "tail": out_text[-500:]},
                    elapsed_seconds=elapsed,
                )

            elif act == TypedActionName.RUN_PROBE.value:
                cfg_path = params.get("config_path")
                if not cfg_path:
                    available = sorted(
                        list((self.project_root / "configs").glob("exp*.yaml"))
                    )
                    cfg_path = (
                        str(available[0].relative_to(self.project_root)).replace(
                            "\\", "/"
                        )
                        if available
                        else "configs/baseline.yaml"
                    )
                probe_samples = str(params.get("probe_samples", 100))
                fold = str(params.get("fold", 0))
                cmd = [
                    sys.executable,
                    "-m",
                    "harness.runner",
                    "--project-root",
                    str(self.project_root),
                    "--task-config",
                    str(self.project.task_config_path),
                    "--config",
                    cfg_path,
                    "--debug",
                    "--fold",
                    fold,
                    "--probe-samples",
                    probe_samples,
                    "--skip-leaderboard",
                ]
                passed, retcode, out_text, elapsed = self._run_subprocess_safe(
                    cmd, timeout=min(120.0, float(params.get("_deadline_seconds", 120)))
                )
                session.compute_usage["smoke_runs"] = (
                    int(session.compute_usage.get("smoke_runs", 0)) + 1
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS" if passed else "FAILED",
                    summary=f"Fast Probe run ({probe_samples} samples, Fold {fold}, {cfg_path}): {'PASSED' if passed else 'FAILED'}.",
                    artifact_path=cfg_path,
                    data={"returncode": retcode, "tail": out_text[-500:]},
                    elapsed_seconds=elapsed,
                )

            elif act == TypedActionName.RUN_FULL_VALIDATION.value:
                cfg_path = params.get("config_path")
                if not cfg_path:
                    raise ValueError(
                        "Parameter 'config_path' is required for 'run_full_validation'."
                    )
                self._validate_config_path(cfg_path)

                backend = str(params.get("backend", "local")).lower()
                force = bool(params.get("force", False))

                full_cfg_file = self.project_root / cfg_path
                raw_cfg = (
                    yaml.safe_load(full_cfg_file.read_text(encoding="utf-8"))
                    if full_cfg_file.exists()
                    else {}
                )
                exp_id = raw_cfg.get("experiment", {}).get("id", Path(cfg_path).stem)
                self._review_research_config(raw_cfg, params, for_execution=True)

                if backend in ("modal", "kaggle"):
                    from harness.cloud_job_runner import CloudJobManager

                    manager = CloudJobManager(self.project_root)
                    job = manager.start_job(
                        backend=backend,
                        config_path=cfg_path,
                        gpu=str(params.get("gpu", "A10")),
                        auto_audit=bool(params.get("auto_audit", True)),
                        auto_sync=True,
                        baseline_csv=params.get("baseline_csv"),
                    )
                    session.compute_usage["experiments_used"] = (
                        int(session.compute_usage.get("experiments_used", 0)) + 1
                    )
                    session.compute_usage["cloud_runs"] = (
                        int(session.compute_usage.get("cloud_runs", 0)) + 1
                    )
                    if exp_id not in session.experiments_run:
                        session.experiments_run.append(exp_id)
                    return ActionObservation(
                        step_index=call.step_index,
                        action=act,
                        status="SUCCESS",
                        summary=f"Dispatched {backend} job `{job.job_id}` for experiment `{exp_id}`; result import waits for a terminal provider state.",
                        artifact_path=job.artifact_dir,
                        data={
                            "job_id": job.job_id,
                            "backend": backend,
                            "status": job.status,
                            "provider_status": job.provider_status,
                            "exp_id": exp_id,
                        },
                        elapsed_seconds=round(time.time() - t0, 3),
                    )

                if backend != "local":
                    raise ValueError(f"Unsupported execution backend: {backend}")
                cmd = [
                    sys.executable,
                    "-m",
                    "harness.runner",
                    "--project-root",
                    str(self.project_root),
                    "--task-config",
                    str(self.project.task_config_path),
                    "--config",
                    cfg_path,
                ]
                if force:
                    cmd.append("--force")

                passed, retcode, out_text, elapsed = self._run_subprocess_safe(
                    cmd,
                    timeout=min(1800.0, float(params.get("_deadline_seconds", 1800))),
                )

                # Update compute usage
                session.compute_usage["experiments_used"] = (
                    int(session.compute_usage.get("experiments_used", 0)) + 1
                )
                session.compute_usage["local_runs"] = (
                    int(session.compute_usage.get("local_runs", 0)) + 1
                )

                if exp_id not in session.experiments_run:
                    session.experiments_run.append(exp_id)

                max_rej = int(session.budget.get("max_consecutive_rejections") or 3)
                if not passed:
                    session.consecutive_rejections += 1
                    session.reassess_recommended = (
                        session.consecutive_rejections >= max_rej
                    )
                    return ActionObservation(
                        step_index=call.step_index,
                        action=act,
                        status="FAILED",
                        summary=f"Full validation ({backend}) failed for '{exp_id}'.",
                        advisory=(
                            "REASSESS_RECOMMENDED"
                            if session.reassess_recommended
                            else None
                        ),
                        artifact_path=str(
                            (self.artifacts_dir / exp_id).relative_to(self.project_root)
                        ).replace("\\", "/"),
                        data={"returncode": retcode, "tail": out_text[-600:]},
                        elapsed_seconds=elapsed,
                    )

                # Reload TreeJournal and evaluate metrics
                journal.sync_from_artifacts(self.artifacts_dir)
                pool = self.comp_state.compute_candidate_pool(journal)
                base_score = float(
                    pool.get("baseline_score")
                    or self.comp_state.spec.get("primary_metric_baseline")
                    or 0.0
                )
                oof_path = self.artifacts_dir / exp_id / "oof_predictions.csv"

                if oof_path.exists():
                    comp_score, detail_scores = self._evaluate_oof_dataframe(oof_path)
                else:
                    node = journal.get_node(exp_id)
                    comp_score = node.get_score() if node else 0.0
                    detail_scores = {"composite_score": comp_score}

                delta = round(
                    (comp_score - base_score)
                    * (1 if self.task_spec.direction == "maximize" else -1),
                    4,
                )
                min_imp = self._get_spec_min_improvement(session)
                verified_runs = self._research_memory().sync()
                current = next((r for r in verified_runs if r["exp_id"] == exp_id), None)
                comparable = [r["utility"] for r in verified_runs if current and
                              r["exp_id"] != exp_id and r["comparison_group"] == current["comparison_group"]]
                incumbent_gain = current["utility"] - max(comparable) if current and comparable else None
                if incumbent_gain is None or (incumbent_gain > 0 and incumbent_gain >= min_imp):
                    session.consecutive_rejections = 0
                else:
                    session.consecutive_rejections += 1
                session.reassess_recommended = session.consecutive_rejections >= max_rej

                diag_json = self.artifacts_dir / exp_id / "diagnostics.json"
                bottlenecks = []
                if diag_json.exists():
                    try:
                        bottlenecks = json.loads(
                            diag_json.read_text(encoding="utf-8")
                        ).get("critical_bottlenecks", [])
                    except Exception:
                        pass

                validation_cfg = self.comp_state.spec.get("validation") or {}
                data_cfg = (
                    self.comp_state.spec.get("data_config")
                    or self.comp_state.spec.get("data")
                    or {}
                )
                validation_strategy = (
                    validation_cfg.get("strategy")
                    or validation_cfg.get("scheme")
                    or data_cfg.get("split_strategy")
                    or "configured validation"
                )
                fold_count = (
                    validation_cfg.get("n_folds")
                    or validation_cfg.get("num_folds")
                    or data_cfg.get("n_folds")
                )
                validation_label = (
                    f"{validation_strategy} ({fold_count} folds)"
                    if fold_count
                    else str(validation_strategy)
                )

                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Completed {validation_label} ({backend}) for '{exp_id}': Composite={comp_score:.4f} (Δ={delta:+.4f} vs baseline).",
                    composite_score=comp_score,
                    metric_delta=delta,
                    diagnostics_bottlenecks=bottlenecks,
                    advisory=(
                        "REASSESS_RECOMMENDED" if session.reassess_recommended else None
                    ),
                    artifact_path=str(oof_path.relative_to(self.project_root)).replace(
                        "\\", "/"
                    ),
                    data={
                        "exp_id": exp_id,
                        "backend": backend,
                        "scores": detail_scores,
                        "incumbent_utility_gain": incumbent_gain,
                    },
                    elapsed_seconds=elapsed,
                )

            elif act in (
                TypedActionName.INSPECT_CLOUD_JOB.value,
                TypedActionName.WAIT_CLOUD_JOB.value,
            ):
                from harness.cloud_job_runner import CloudJobManager

                job_id = params.get("job_id")
                if not job_id:
                    raise ValueError(f"Parameter `job_id` is required for `{act}`.")
                manager = CloudJobManager(self.project_root)
                wait_seconds = min(
                    3600.0, max(0.0, float(params.get("timeout_seconds", 0)))
                )
                poll_seconds = min(
                    30.0, max(0.25, float(params.get("poll_interval", 3)))
                )
                deadline = time.monotonic() + wait_seconds
                job = manager.get_job(str(job_id))
                if not job:
                    raise ValueError(
                        f"Cloud job `{job_id}` was not found in this task workspace."
                    )
                while (
                    act == TypedActionName.WAIT_CLOUD_JOB.value
                    and job.get("status") in {"PENDING", "RUNNING"}
                    and time.monotonic() < deadline
                ):
                    time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
                    job = manager.get_job(str(job_id)) or job
                artifact_data: Dict[str, Any] = {}
                if job.get("status") == "COMPLETED":
                    artifact_dir = self.project_root / str(
                        job.get("artifact_dir") or ""
                    )
                    metrics_file = artifact_dir / "metrics.json"
                    if metrics_file.is_file():
                        try:
                            artifact_data["metrics"] = json.loads(
                                metrics_file.read_text(encoding="utf-8")
                            )
                        except json.JSONDecodeError:
                            artifact_data["metrics_error"] = (
                                "metrics.json is not valid JSON"
                            )
                    artifact_data["artifact_paths"] = (
                        [
                            str(path.relative_to(self.project_root)).replace("\\", "/")
                            for path in artifact_dir.iterdir()
                            if path.is_file()
                        ]
                        if artifact_dir.is_dir()
                        else []
                    )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status=(
                        "SUCCESS"
                        if job.get("status") not in {"FAILED", "CANCELLED"}
                        else "FAILED"
                    ),
                    summary=f"Cloud job `{job_id}` is {job.get('status')}; artifact identity verified={job.get('artifact_identity_verified', False)}.",
                    artifact_path=job.get("artifact_dir"),
                    data={"job": job, **artifact_data},
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act in (
                TypedActionName.SYNTHESIZE_CANDIDATE.value,
                TypedActionName.PROMOTE_CANDIDATE.value,
                TypedActionName.BUILD_ENSEMBLE.value,
            ):
                name = params.get("name", f"{session.session_id}_router")
                mode = params.get("mode", "specialist_router")
                source_map = params.get("source_map")
                if act == TypedActionName.PROMOTE_CANDIDATE.value:
                    if not params.get("exp_id"):
                        raise ValueError("Promotion requires exp_id")
                    mode = "single_model"
                    source_map = {"source_exp": params["exp_id"]}
                if not source_map:
                    raise ValueError("Explicit source_map is required")
                # Anti-duplicate-submission guard: an agent may re-synthesize the
                # same recipe under a new display name to "preserve" a submission
                # or pad the candidate pool. Reject an identical recipe unless the
                # agent explicitly justifies a revisit (the recipe is otherwise
                # byte-identical, so a new name adds no scientific signal).
                from harness.candidates import _candidate_recipe_fingerprint

                new_fp = _candidate_recipe_fingerprint(mode, self._candidate_source_ids(mode, source_map), source_map)
                existing_fps = {
                    c.get("recipe_fingerprint"): c["id"]
                    for c in self.comp_state.state["candidate_pool"].get("synthesized_candidates", [])
                    if c.get("recipe_fingerprint")
                }
                if new_fp in existing_fps and not str(params.get("revisit_reason", "")).strip():
                    raise ValueError(
                        f"Candidate recipe already exists as '{existing_fps[new_fp]}'. "
                        f"Re-synthesizing the same sources/mode/weights under a new name does not "
                        f"create a new submission. Provide 'revisit_reason' to intentionally revisit "
                        f"this recipe, or choose different sources." if act != "promote_candidate" else
                        f"Candidate for this experiment is already in the pool as '{existing_fps[new_fp]}'"
                    )
                entry = self.comp_state.synthesize_candidate(
                    name=name, mode=mode, source_map=source_map
                )
                session.compute_usage["candidate_operations"] = (
                    int(session.compute_usage.get("candidate_operations", 0)) + 1
                )

                oof_abs = self.project_root / entry["oof_path"]
                synth_score, detail_scores = self._evaluate_oof_dataframe(oof_abs)
                base_score = float(
                    self.comp_state.compute_candidate_pool(journal).get(
                        "baseline_score"
                    )
                    or self.comp_state.spec.get("primary_metric_baseline")
                    or 0.0
                )
                delta = round(
                    (synth_score - base_score)
                    * (1 if self.task_spec.direction == "maximize" else -1),
                    4,
                )

                min_imp = self._get_spec_min_improvement(session)
                max_rej = int(session.budget.get("max_consecutive_rejections") or 3)
                if delta >= min_imp:
                    session.consecutive_rejections = 0
                else:
                    session.consecutive_rejections += 1
                session.reassess_recommended = session.consecutive_rejections >= max_rej

                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Synthesized candidate '{entry['id']}' via '{mode}' (Real OOF Composite={synth_score:.4f}, Δ={delta:+.4f}).",
                    composite_score=synth_score,
                    metric_delta=delta,
                    fold_variance=None,
                    advisory=(
                        "REASSESS_RECOMMENDED" if session.reassess_recommended else None
                    ),
                    artifact_path=entry["oof_path"],
                    data={**entry, "scores": detail_scores},
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.REGISTER_EXTERNAL_PREDICTION.value:
                from harness.candidates import register_external_prediction

                name = params.get("name") or params.get("exp_id")
                if not name:
                    raise ValueError("Action 'register_external_prediction' requires 'name' (or 'exp_id').")
                oof_file = params.get("oof_file") or params.get("oof_path")
                if not oof_file:
                    raise ValueError("Action 'register_external_prediction' requires 'oof_file'.")
                test_file = params.get("test_file") or params.get("test_path")

                res = register_external_prediction(
                    project=self.project,
                    name=name,
                    oof_path=oof_file,
                    test_path=test_file,
                    author=params.get("author", "grandmaster"),
                    source=params.get("source", "external_import"),
                    notes=params.get("notes", ""),
                )
                session.compute_usage["candidate_operations"] = (
                    int(session.compute_usage.get("candidate_operations", 0)) + 1
                )
                cand_score = float(res.get("primary_score", 0.0))
                base_score = float(
                    self.comp_state.compute_candidate_pool(journal).get("baseline_score")
                    or self.comp_state.spec.get("primary_metric_baseline")
                    or 0.0
                )
                delta = round(
                    (cand_score - base_score) * (1 if self.task_spec.direction == "maximize" else -1),
                    4,
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Registered external prediction '{name}' ({res.get('metric')}={cand_score:.4f}, Δ={delta:+.4f}, rows={res.get('rows')}).",
                    composite_score=cand_score,
                    metric_delta=delta,
                    artifact_path=res.get("artifact_dir"),
                    data=res,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.OPTIMIZE_BLEND.value:
                from harness.candidates import find_optimal_blend_weights

                components = params.get("components") or params.get("component_ids") or []
                if isinstance(components, str):
                    try:
                        components = json.loads(components)
                    except Exception:
                        components = [c.strip() for c in components.split(",") if c.strip()]
                if len(components) < 2:
                    raise ValueError("Action 'optimize_blend' requires at least 2 experiment IDs in 'components'.")

                blend_mode = params.get("mode") or params.get("blend_mode")
                opt_res = find_optimal_blend_weights(
                    project=self.project,
                    component_ids=components,
                    blend_mode=blend_mode,
                    maxiter=int(params.get("maxiter", 150)),
                )

                create_cand = params.get("create_candidate", True)
                cand_entry = None
                if create_cand and opt_res.get("superiority_satisfied"):
                    name = params.get("name", f"blend_{'_'.join(components[:3])}")
                    cand_entry = self.comp_state.synthesize_candidate(
                        name=name,
                        mode=opt_res["blend_mode"],
                        source_map={
                            "components": components,
                            "weights": opt_res["optimal_weights"],
                        },
                    )
                    session.compute_usage["candidate_operations"] = (
                        int(session.compute_usage.get("candidate_operations", 0)) + 1
                    )

                if cand_entry:
                    summary_msg = f"Optimized blend candidate '{cand_entry['id']}' ({opt_res['blend_mode']}) at {opt_res['best_score']:.4f} (Δ vs best constituent={opt_res['gain_vs_best_constituent']:+.4f})."
                    art_path = cand_entry.get("oof_path")
                else:
                    summary_msg = f"Blend optimization found weights {opt_res['weights_dict']}, best_score={opt_res['best_score']:.4f} (Δ vs best={opt_res['gain_vs_best_constituent']:+.4f})."
                    art_path = None

                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=summary_msg,
                    composite_score=opt_res["best_score"],
                    metric_delta=opt_res["gain_vs_best_constituent"],
                    artifact_path=art_path,
                    data={"optimization": opt_res, "candidate": cand_entry},
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.OPTIMIZE_THRESHOLD.value:
                from harness.endgame_submission import ThresholdOptimizer

                pool = self.comp_state.compute_candidate_pool(journal)
                default_exp = (
                    (pool.get("overall_best") or {}).get("id")
                    or self.comp_state.state["gates"].get("baseline_exp_id")
                    or "baseline"
                )
                exp_id = params.get("exp_id", default_exp)
                exp_dir = self.artifacts_dir / exp_id
                if not exp_dir.is_dir():
                    raise ValueError(f"Experiment artifact folder not found: {exp_dir}")

                oof_csv = exp_dir / "oof_predictions.csv"
                truth_csv = exp_dir / "evaluation_truth.csv"
                if not oof_csv.is_file() or not truth_csv.is_file():
                    raise ValueError(f"OOF predictions or truth missing in {exp_dir}")

                oof_df = pd.read_csv(oof_csv)
                truth_df = pd.read_csv(truth_csv)
                id_col = self.task_spec.id_column

                merged = oof_df.merge(truth_df, on=id_col, suffixes=("_pred", "_truth"))

                target_col = None
                for st_name, st in self.task_spec.subtasks.items():
                    target_col = st.column or st_name
                    break
                if not target_col:
                    target_col = [c for c in truth_df.columns if c != id_col][0]

                y_raw = merged[target_col].values
                unique_vals = np.unique(y_raw)
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    y_true = y_raw.astype(int)
                else:
                    med = np.median(y_raw)
                    y_true = (y_raw > med).astype(int)

                proba_col = f"proba_{target_col}"
                if proba_col in merged.columns:
                    raw_p = merged[proba_col].values
                    if isinstance(raw_p[0], str) and raw_p[0].startswith("["):
                        y_prob = np.asarray([json.loads(p)[1] if len(json.loads(p)) > 1 else json.loads(p)[0] for p in raw_p], float)
                    else:
                        y_prob = raw_p.astype(float)
                elif f"pred_{target_col}" in merged.columns:
                    y_prob = merged[f"pred_{target_col}"].values.astype(float)
                elif target_col + "_pred" in merged.columns:
                    y_prob = merged[target_col + "_pred"].values.astype(float)
                else:
                    raise ValueError(f"Could not locate probability column for target '{target_col}'.")

                metric = params.get("metric", "f1")
                num_th = int(params.get("num_thresholds", 181))
                best_th, best_score = ThresholdOptimizer.optimize_threshold(
                    y_true=y_true,
                    y_prob=y_prob,
                    metric=metric,
                    num_thresholds=num_th,
                )

                preds_05 = (y_prob >= 0.5).astype(int)
                from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
                m_lower = metric.lower()
                if m_lower in {"kappa", "cohen_kappa"}:
                    base_05_score = float(cohen_kappa_score(y_true, preds_05))
                elif m_lower in {"accuracy", "acc"}:
                    base_05_score = float(accuracy_score(y_true, preds_05))
                else:
                    base_05_score = float(f1_score(y_true, preds_05, zero_division=0))
                delta = round(best_score - base_05_score, 4)

                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Optimized decision threshold for '{exp_id}': best_th={best_th:.3f} ({metric}={best_score:.4f}, Δ vs 0.5={delta:+.4f}).",
                    composite_score=best_score,
                    metric_delta=delta,
                    data={
                        "exp_id": exp_id,
                        "metric": metric,
                        "best_threshold": round(best_th, 4),
                        "best_score": round(best_score, 5),
                        "baseline_score_at_05": round(base_05_score, 5),
                        "gain": delta,
                    },
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.GENERATE_MICRODOSE_PROBE.value:
                from harness.endgame_submission import MicroDoseProbeGenerator

                base = params.get("base") or params.get("base_exp_id") or params.get("base_csv")
                probe = params.get("probe") or params.get("probe_exp_id") or params.get("probe_csv")
                if not base or not probe:
                    raise ValueError("Action 'generate_microdose_probe' requires 'base' and 'probe'.")

                base_path = Path(base)
                if not base_path.exists():
                    cand = self.artifacts_dir / base / "test_predictions.csv"
                    if cand.is_file():
                        base_path = cand
                    else:
                        raise FileNotFoundError(f"Base submission file not found: {base}")

                probe_path = Path(probe)
                if not probe_path.exists():
                    cand = self.artifacts_dir / probe / "test_predictions.csv"
                    if cand.is_file():
                        probe_path = cand
                    else:
                        raise FileNotFoundError(f"Probe submission file not found: {probe}")

                alpha = float(params.get("alpha", 0.03))
                if alpha > 0.08:
                    alpha = 0.08

                out_file = params.get("output_csv")
                if not out_file:
                    out_file = self.project_root / "submissions" / f"probe_{base_path.parent.name}_{probe_path.parent.name}_a{int(alpha*100)}.csv"

                report = MicroDoseProbeGenerator.generate_probe(
                    base_csv=base_path,
                    probe_csv=probe_path,
                    output_csv=out_file,
                    alpha=alpha,
                    space=params.get("space", "logit"),
                    id_column=self.task_spec.id_column,
                )

                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Generated micro-dose probe (α={alpha:.2f}, space='{report['space']}'): Spearman r={report['spearman_correlation_vs_base']:.4f}, max_diff={report['max_absolute_diff']:.4f}.",
                    artifact_path=report["output_path"],
                    data=report,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.PREPARE_SUBMISSION.value:
                pool = self.comp_state.compute_candidate_pool(journal)
                best_cand = pool.get("overall_best") or {}
                default_exp = (
                    best_cand.get("id")
                    or self.comp_state.state["gates"].get("baseline_exp_id")
                    or "baseline"
                )
                exp_id = params.get("exp_id", default_exp)
                cand_csv = params.get("candidate_csv")
                if not cand_csv:
                    default_cand = self.artifacts_dir / exp_id / "test_predictions.csv"
                    if default_cand.is_file():
                        cand_csv = str(default_cand)
                    else:
                        raise ValueError(
                            f"Action 'prepare_submission' requires 'candidate_csv' or existing 'test_predictions.csv' in {self.artifacts_dir / exp_id}."
                        )
                cand_path = Path(cand_csv)
                if not cand_path.is_absolute():
                    cand_path = self.project_root / cand_path

                from harness.artifacts import validate_test_prediction

                validate_test_prediction(
                    cand_path,
                    task_spec=self.task_spec,
                    plugin=self.task_plugin,
                )
                # Anti-bypass submission guards: dedup, MMES, significance. The
                # agent drives submission through this bridge action; the standalone
                # audit gate CLI is not on this path, so enforce the core checks here.
                guard_evidence = self._guard_submission(cand_path, exp_id, sub_mgr)
                capabilities = self.task_plugin.capabilities()
                if not capabilities.get("submission_builder") or not capabilities.get(
                    "submission_validator"
                ):
                    raise ValueError("Task plugin must implement submission hooks")
                sub_spec = getattr(self.task_spec, "submission_spec", None) or self.comp_state.spec.get("submission_spec", {})
                file_name = sub_spec.get("file_name", "submission.csv") if isinstance(sub_spec, dict) else "submission.csv"
                entry = sub_mgr.register_plugin_submission(
                    exp_id=exp_id,
                    candidate_csv=cand_path,
                    plugin=self.task_plugin,
                    plugin_context=self.plugin_context,
                    file_name=file_name,
                    notes=params.get("notes", ""),
                )
                session.compute_usage["submissions_used"] = (
                    int(session.compute_usage.get("submissions_used", 0)) + 1
                )
                plugin_label = self.project.plugin_id or "default"
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Built and validated task submission '{entry.id}' with plugin '{plugin_label}'.",
                    artifact_path=entry.archive_path,
                    data={**asdict(entry), "submission_guards": guard_evidence},
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.RECORD_EXTERNAL_SCORE.value:
                sub_id = params.get("sub_id")
                if not sub_id:
                    raise ValueError(
                        "Parameter 'sub_id' is required for 'record_external_score'."
                    )
                entry = sub_mgr.record_official_score(
                    sub_id=sub_id,
                    primary_score=(
                        params["primary_score"]
                        if "primary_score" in params
                        else params.get("score")
                    ),
                    scores=params.get("scores") or params.get("custom_scores"),
                    split_type=params.get("split_type", "public"),
                    notes=params.get("notes", ""),
                )
                blockers = (
                    [
                        "CV vs LB Divergence detected (significant score delta between LB and Local CV)!"
                    ]
                    if entry.divergence_warning
                    else []
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Recorded official score for '{sub_id}': {entry.official_scores} (Divergence={entry.divergence_warning}).",
                    new_blockers=blockers,
                    data=asdict(entry),
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.FREEZE_CANDIDATE.value:
                s1 = params.get("slot_1_id")
                s2 = params.get("slot_2_id")
                if not s1:
                    candidates = self.comp_state.state["candidate_pool"].get(
                        "synthesized_candidates", []
                    )
                    if candidates:
                        s1 = candidates[0].get("id")
                        if len(candidates) > 1 and not s2:
                            s2 = candidates[1].get("id")
                    else:
                        raise ValueError(
                            "Parameter 'slot_1_id' is required for 'freeze_candidate' when candidate pool is empty."
                        )
                lock = self.comp_state.freeze_final_candidates(
                    slot_1_id=s1,
                    slot_2_id=s2,
                    notes=params.get(
                        "notes", f"Frozen via AgentBridge session {session.session_id}"
                    ),
                )
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Frozen final portfolio ({s1}, {s2}) into FROZEN_FINAL_LOCK.json.",
                    artifact_path="submissions/FROZEN_FINAL_LOCK.json",
                    data=lock,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.EMERGENCY_UNFREEZE.value:
                reason = params.get("reason")
                if not reason:
                    raise ValueError("Unfreeze requires an explicit reason")
                res = self.comp_state.emergency_unfreeze(reason=reason)
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Emergency unfreeze executed: {reason}",
                    data=res,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.FINALIZE_COMPETITION.value:
                res = self.comp_state.finalize_competition(params, journal, sub_mgr)
                session.status = "COMPLETED"
                session.stop_reason = "COMPETITION_FINALIZED"
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Generated Competition Closure & Reproducibility Report at '{res['report_path']}'.",
                    artifact_path=res["report_path"],
                    data=res,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            elif act == TypedActionName.GET_NEXT_EXPERIMENT_ID.value:
                next_num = self.get_next_experiment_num()
                branch = params.get("branch", "feat")
                descriptor = params.get("descriptor", "iteration")
                suggested_id = f"exp{next_num:03d}_{branch}_{descriptor}"
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="SUCCESS",
                    summary=f"Next available experiment number is {next_num} (suggested ID: '{suggested_id}').",
                    data={
                        "next_num": next_num,
                        "next_id_prefix": f"exp{next_num:03d}",
                        "suggested_exp_id": suggested_id,
                    },
                    elapsed_seconds=round(time.time() - t0, 3),
                )

            else:
                return ActionObservation(
                    step_index=call.step_index,
                    action=act,
                    status="FAILED",
                    summary=f"Unsupported typed action '{act}'.",
                    elapsed_seconds=round(time.time() - t0, 3),
                )
        except Exception as err:
            return ActionObservation(
                step_index=call.step_index,
                action=act,
                status="FAILED",
                summary=f"Action '{act}' raised exception: {err}",
                elapsed_seconds=round(time.time() - t0, 3),
            )

    # Alias for execute_action matching CLI 'act' command
    act = execute_action


def main(argv: Optional[List[str]] = None) -> int:
    """Forwarder to tools.agent_bridge_cli.main for module execution."""
    from tools.agent_bridge_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

