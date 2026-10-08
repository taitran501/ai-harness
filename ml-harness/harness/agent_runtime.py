"""Backward-Compatible Adapter for `harness/agent_bridge.py`.

Architecture Note:
- The TRUE Agent Runtime (Reasoning, Hypothesis Formulation, Planning, Decision Loop)
  is **Antigravity inside the IDE** — NOT an internal Python planner inside the Harness.
- All session tracking, Context Packet generation, budget/approval guardrails, and Typed
  Action execution now live in `harness/agent_bridge.py` (`AgentBridge` & `BridgeSession`).
- This module wraps `AgentBridge` to preserve 100% backward compatibility with existing
  CLI scripts, Dashboard endpoints, and unit tests.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.agent_bridge import (
    ActionObservation,
    AgentBridge,
    ApprovalTier,
    BridgeSession,
    DEFAULT_ACTION_APPROVAL_TIERS,
    TypedActionCall,
    TypedActionName,
)
from harness.submission_manager import SubmissionManager
from harness.tree_journal import TreeJournal


@dataclass
class RunBudget:
    """Compatibility budget wrapper mapped to SessionBudget & ComputeUsage."""

    max_experiments: int = 3
    experiments_used: int = 0
    max_wall_seconds: float = 1800.0
    wall_seconds_used: float = 0.0
    max_gpu_hours: float = 2.0
    gpu_hours_used: float = 0.0
    max_submissions: int = 1
    submissions_used: int = 0
    max_tokens: int = 100_000
    tokens_used: int = 0


@dataclass
class StoppingPolicy:
    """Compatibility stopping policy configuration."""

    max_consecutive_rejections: int = 3
    min_meaningful_improvement: float = 0.002
    target_score_goal: Optional[float] = None
    stop_on_submission_quota_exhausted: bool = True
    stop_on_freeze_state: bool = True
    pause_on_user_approval_required: bool = True


@dataclass
class PlannerDecision:
    """Compatibility decision record mapped from Bridge observations."""

    step_index: int
    verdict: str
    reasoning: str
    next_action_selected: Optional[str] = None
    stop_reason: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@dataclass
class AgentRun:
    """Backward-compatible projection of a `BridgeSession`."""

    id: str
    goal: str
    status: str
    created_at: str
    updated_at: str
    starting_state: Dict[str, Any] = field(default_factory=dict)
    final_state: Optional[Dict[str, Any]] = None
    budget: Dict[str, Any] = field(default_factory=lambda: asdict(RunBudget()))
    stopping_policy: Dict[str, Any] = field(default_factory=lambda: asdict(StoppingPolicy()))
    approval_policy: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ACTION_APPROVAL_TIERS))
    plan: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    pending_approval_action: Optional[Dict[str, Any]] = None
    consecutive_rejections: int = 0
    stop_reason: Optional[str] = None


class AgentRuntime:
    """Adapter class routing legacy AgentRuntime calls to `AgentBridge`."""

    def __init__(self, project_root: Path | str = "."):
        self.project_root = Path(project_root).resolve()
        self.bridge = AgentBridge(self.project_root)
        self.runs_dir = self.bridge.legacy_runs_dir
        self.comp_state = self.bridge.comp_state

    @staticmethod
    def _session_to_agent_run(sess: BridgeSession) -> AgentRun:
        run_num = sess.session_id.split("_")[-1]
        legacy_id = f"run_{run_num}" if not sess.session_id.startswith("run_") else sess.session_id
        merged_budget = {
            **asdict(RunBudget()),
            **sess.budget,
            **sess.compute_usage,
        }
        decisions = []
        for idx, o in enumerate(sess.observations):
            st = o.get("status", "SUCCESS")
            verdict = (
                "PAUSE_FOR_APPROVAL"
                if st == "PAUSED_FOR_APPROVAL"
                else ("KEEP_AND_PROMOTE" if (o.get("metric_delta") or 0.0) >= 0.002 else "CONTINUE")
            )
            decisions.append(
                asdict(
                    PlannerDecision(
                        step_index=o.get("step_index", idx + 1),
                        verdict=verdict,
                        reasoning=o.get("summary", ""),
                        stop_reason=sess.stop_reason if st == "PAUSED_FOR_APPROVAL" else None,
                    )
                )
            )

        return AgentRun(
            id=legacy_id,
            goal=sess.goal,
            status=sess.status,
            created_at=sess.created_at,
            updated_at=sess.updated_at,
            starting_state=sess.starting_state,
            final_state=sess.final_state,
            budget=merged_budget,
            approval_policy=sess.approval_policy,
            plan=[f"[{a.get('approval_tier', 'AUTO')}] {a.get('action')}: {a.get('rationale', '')}" for a in sess.actions],
            actions=sess.actions,
            observations=sess.observations,
            decisions=decisions,
            pending_approval_action=sess.pending_approval_action,
            consecutive_rejections=sess.consecutive_rejections,
            stop_reason=sess.stop_reason,
        )

    def get_next_run_id(self) -> str:
        sid = self.bridge.get_next_session_id()
        return f"run_{sid.split('_')[-1]}"

    def load_run(self, run_id: str) -> AgentRun:
        sess = self.bridge.load_session(run_id)
        return self._session_to_agent_run(sess)

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        sessions = self.bridge.list_sessions(limit=limit)
        return [asdict(self._session_to_agent_run(BridgeSession(**s))) for s in sessions]

    def execute_typed_action(
        self,
        call: TypedActionCall,
        run: AgentRun,
        journal: TreeJournal,
        sub_mgr: SubmissionManager,
    ) -> ActionObservation:
        sess = self.bridge.get_active_or_create_session(goal=run.goal, session_id=run.id)
        return self.bridge.execute_action(call.action,params=call.params,rationale=call.rationale,session_id=sess.session_id,user_approved=call.user_approved)[1]

    def start_run(
        self,
        goal: str,
        max_experiments: int = 3,
        max_wall_seconds: float = 1800.0,
        auto_approve_all: bool = False,
    ) -> AgentRun:
        """Initialize a BridgeSession and execute initial readiness/synthesis actions via AgentBridge."""
        sess = self.bridge.start_session(
            goal=goal,
            budget={
                "max_experiments": max_experiments,
                "max_wall_seconds": max_wall_seconds,
            },
        )
        gates = self.comp_state.state.get("gates", {})

        if not gates.get("data_risks_confirmed"):
            sess, _ = self.bridge.execute_action(
                TypedActionName.INSPECT_DATA_READINESS.value,
                rationale="Verify dataset readiness and group splits.",
                session_id=sess.session_id,
            )

        if not gates.get("baseline_locked"):
            sess, _ = self.bridge.execute_action(
                TypedActionName.ESTABLISH_BASELINE_ANCHOR.value,
                rationale="Establish official baseline anchor.",
                session_id=sess.session_id,
            )

        default_base = self.bridge.comp_state.state["gates"].get("baseline_exp_id") or "baseline"
        sess, _ = self.bridge.execute_action(
            TypedActionName.INSPECT_DIAGNOSTICS.value,
            params={"exp_id": default_base},
            rationale=f"Extract failure modes for goal: {goal}",
            session_id=sess.session_id,
        )

        sess, _ = self.bridge.execute_action(
            TypedActionName.CREATE_EXPERIMENT.value,
            params={"parent_id": default_base, "branch": "feat", "descriptor": "feature_pipeline"},
            rationale=f"Draft GroundingGate-verified config for goal: {goal}.",
            session_id=sess.session_id,
        )

        sess, _ = self.bridge.execute_action(
            TypedActionName.SYNTHESIZE_CANDIDATE.value,
            params={
                "name": f"{sess.session_id}_router",
                "mode": "specialist_router",
            },
            rationale="Synthesize specialist candidate portfolio.",
            session_id=sess.session_id,
        )

        sess, _ = self.bridge.execute_action(
            TypedActionName.FREEZE_CANDIDATE.value,
            params={},
            rationale="Propose freezing candidate portfolio (gated by USER_APPROVAL).",
            session_id=sess.session_id,
            user_approved=auto_approve_all,
        )

        return self._session_to_agent_run(sess)

    def approve_pending_action(self, run_id: str) -> AgentRun:
        sess, _ = self.bridge.approve_pending_action(run_id)
        return self._session_to_agent_run(sess)
