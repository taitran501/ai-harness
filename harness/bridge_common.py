"""Shared typed-action vocabulary and session types for the Agent Bridge.

Extracted verbatim from ``harness/agent_bridge.py`` (M4 refactoring,
PR-E): the ``ApprovalTier`` / ``TypedActionName`` enums, session caps
(``MAX_SESSION_ACTIONS`` / ``MAX_SESSION_OBSERVATIONS``),
``DEFAULT_ACTION_APPROVAL_TIERS``, the dataclasses (``SessionBudget``,
``ComputeUsage``, ``TypedActionCall``, ``ActionObservation``,
``BridgeSession``), and the ``_TYPED_ACTION_NAMES`` guard set.

This module is the single source of truth for the bridge's typed
vocabulary; ``harness.agent_bridge`` re-exports every name defined
here, so existing importers are unaffected.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


# Session history caps: keep `sessions_dir/*.json` bounded for long
# competition runs. Oldest entries are dropped (the most recent context is
# what the agent re-primes from; full history remains in the journal).
MAX_SESSION_ACTIONS = 200
MAX_SESSION_OBSERVATIONS = 200


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


_TYPED_ACTION_NAMES = frozenset(item.value for item in TypedActionName)
