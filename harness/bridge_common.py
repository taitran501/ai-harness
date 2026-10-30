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
    RECORD_MACRO_REVIEW = "record_macro_review"
    GET_MACRO_HYPOTHESIS_MAP = "get_macro_hypothesis_map"
    CHECK_RETROSPECTIVE_STATUS = "check_retrospective_status"
    PLAN_RESEARCH_PROBE = "plan_research_probe"
    REGISTER_RECOVERY_PROBE = "register_recovery_probe"
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
    CANDIDATE_PREFLIGHT = "candidate_preflight"
    VALIDATE_CANDIDATE_PITCH = "validate_candidate_pitch"
    RECONCILE_SUBMISSIONS = "reconcile_submissions"
    ACKNOWLEDGE_EXTERNAL_SUBMISSIONS = "acknowledge_external_submissions" 


# Session history caps: keep `sessions_dir/*.json` bounded for long
# competition runs. Oldest entries are dropped (the most recent context is
# what the agent re-primes from; full history remains in the journal).
MAX_SESSION_ACTIONS = 200
MAX_SESSION_OBSERVATIONS = 200


DEFAULT_ACTION_APPROVAL_TIERS: Dict[str, str] = {
    "candidate_preflight": ApprovalTier.AUTO.value,
    "validate_candidate_pitch": ApprovalTier.AUTO.value,
    "reconcile_submissions": ApprovalTier.AUTO.value,
    "acknowledge_external_submissions": ApprovalTier.USER_APPROVAL.value,
    "record_macro_review": ApprovalTier.AUTO.value,
    "get_macro_hypothesis_map": ApprovalTier.AUTO.value,
    "check_retrospective_status": ApprovalTier.AUTO.value,
    "register_recovery_probe": ApprovalTier.AUTO.value,
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
    failure: Optional[Dict[str, Any]] = None


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


ACTION_ALIASES: Dict[str, str] = {
    # Checkpoint aliases
    "checkpoint": "record_research_checkpoint",
    "save_checkpoint": "record_research_checkpoint",
    "checkpoint_research_progress": "record_research_checkpoint",
    "record_checkpoint": "record_research_checkpoint",
    "research_checkpoint": "record_research_checkpoint",
    # Insight aliases
    "record_insight": "record_research_insight",
    "add_insight": "record_research_insight",
    "create_insight": "record_research_insight",
    # Decision aliases
    "record_decision": "record_research_decision",
    "add_decision": "record_research_decision",
    # Probe aliases
    "plan_probe": "plan_research_probe",
    "probe_plan": "plan_research_probe",
    # Memory search aliases
    "search_memory": "search_research_memory",
    "search_research": "search_research_memory",
    # Baseline anchor aliases
    "establish_baseline": "establish_baseline_anchor",
    "set_baseline_anchor": "establish_baseline_anchor",
    "set_baseline": "establish_baseline_anchor",
    # Experiment ID aliases
    "get_next_exp_id": "get_next_experiment_id",
    "next_experiment_id": "get_next_experiment_id",
    "next_exp_id": "get_next_experiment_id",
    # Execution aliases
    "smoke_test": "run_smoke_test",
    "full_validation": "run_full_validation",
    "run_validation": "run_full_validation",
}


def resolve_action_name(name: str) -> str:
    """Resolve an action name, mapping aliases to canonical typed actions."""
    return ACTION_ALIASES.get(name, name)


_TYPED_ACTION_NAMES = frozenset(item.value for item in TypedActionName) | frozenset(ACTION_ALIASES.keys())


TYPED_ACTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "record_macro_review": {
        "summary": "Record an evidence-linked macro retrospective and advance its milestone.",
        "required": ["map_digest", "changes_since_last_review", "supported_mechanisms", "rejected_mechanisms",
                     "highest_priority_untouched", "next_probe_rationale", "next_action"],
        "optional": {"review_id": None, "run_count_at_review": None, "task_id": None,
                     "budget_limit": None, "revisit_conditions": None},
        "description": "Use the current map digest. Cite real insight/decision/candidate run IDs. "
                       "Nested objects/lists accept JSON via flat flags or --param-file. Session/task/count are derived.",
    },
    "get_macro_hypothesis_map": {
        "summary": "Project research family coverage from live verified evidence.",
        "required": [], "optional": {}, "description": "Returns families, evidence counts and map_digest for a retrospective.",
    },
    "check_retrospective_status": {
        "summary": "Check five-run cadence and unreviewed stagnation events.",
        "required": [], "optional": {}, "description": "Read-only status; smoke and failed runs never advance the cadence.",
    },
    "register_recovery_probe": {
        "summary": "Register one bounded recovery probe for a paused comparison group.",
        "required": ["comparison_identity_sha256", "rule_id", "diagnosis", "mechanism", "stop_rule"],
        "optional": {"max_runs": 1},
        "description": "Keeps submissions paused until verified promotion of a preregistered recovery candidate.",
    },
    "search_research_memory": {
        "summary": "Search persistent research insights and hypotheses by semantic query.",
        "required": ["query"],
        "optional": {"limit": 8},
        "description": "Searches research_memory.jsonl for matches against past observations, mechanisms, and hypotheses.",
    },
    "record_research_insight": {
        "summary": "Record an empirical research insight into persistent task memory.",
        "required": ["title", "observation", "mechanism", "next_action"],
        "optional": {"evidence": None, "parent_insight_id": None, "family_id": "unclassified", "source_refs": None,
                      "citation_claims": None, "citation_audit_refs": None, "empirical_status": "hypothesis",
                      "decision_evaluation_id": None},
        "description": "Registers a hypothesis or reviewed claim with separate source provenance and local-run evidence status.",
    },
    "plan_research_probe": {
        "summary": "Plan a bounded discriminating probe for a specific insight.",
        "required": [
            "insight_id",
            "control",
            "intervention",
            "expected_observation",
            "alternative_explanation",
            "decision_rule",
            "budget",
            "stop_rule",
        ],
        "optional": {},
        "description": "Predeclare a structured DecisionRuleV1 mapping before execution; legacy text remains an unverified plan.",
    },
    "trace_research_insight": {
        "summary": "Trace the lineage, probe results, and decisions associated with an insight.",
        "required": ["insight_id"],
        "optional": {"offset": 0, "limit": 20},
        "description": "Recovers the full audit history and downstream decisions derived from an insight ID.",
    },
    "record_research_decision": {
        "summary": "Record an evidence-based decision (keep/discard/pivot) after an experiment.",
        "required": ["conclusion"],
        "optional": {"outcome": None, "rule_id": None, "candidate_evidence": None, "claimed_outcome": None,
                     "evidence": None, "next_action": "", "insight_id": None, "lesson": "", "revisit_when": ""},
        "description": "Documents an empirical decision outcome with takeaway lesson and next step.",
    },
    "record_research_checkpoint": {
        "summary": "Record a research checkpoint with current summary, next action, and open questions.",
        "required": ["summary", "next_action"],
        "optional": {"open_questions": [], "insight_ids": None},
        "description": "Saves state to research_frontier for seamless resumption after context compaction or restart.",
    },
    "review_hypothesis": {
        "summary": "Review a proposed experiment config against the research memory before running.",
        "required": [],
        "optional": {"config": None, "config_path": None, "revisit_reason": ""},
        "description": "Validates novelty against past experiments to prevent redundant or circular iterations.",
    },
    "suggest_next_interventions": {
        "summary": "Generate un-tried candidate interventions based on diagnostics and past branches.",
        "required": [],
        "optional": {"limit": 5},
        "description": "Recommends untried hypothesis interventions when search stagnates.",
    },
    "establish_baseline_anchor": {
        "summary": "Lock the official baseline candidate anchor and verify its OOF artifacts.",
        "required": [],
        "optional": {"baseline_id": "exp_id of baseline run (defaults to configured baseline or best run)"},
        "description": "Locks the official baseline anchor in CompetitionState against which all treatments are compared.",
    },
    "create_experiment": {
        "summary": "Create a new scaffolded experiment configuration file under configs/.",
        "required": [],
        "optional": {
            "branch": "feat",
            "descriptor": "iteration",
            "parent_id": None,
            "insight_id": None,
            "hypothesis": None,
            "target_variable": None,
            "model_name": None,
            "model_params": None,
            "input_variants": None,
        },
        "description": "Generates a fully grounded experiment config yaml with linked insight and assumption register.",
    },
    "get_next_experiment_id": {
        "summary": "Compute the next sequential experiment ID for a given branch and descriptor.",
        "required": [],
        "optional": {"branch": "feat", "descriptor": "iteration"},
        "description": "Returns the next available exp_id without creating a file.",
    },
    "run_smoke_test": {
        "summary": "Execute a fast single-fold smoke test on fold 0.",
        "required": ["config_path"],
        "optional": {"fold": 0},
        "description": "Runs a single-fold probe to verify pipeline syntax and grounding before full runs.",
    },
    "run_probe": {
        "summary": "Execute a sub-sampled data probe.",
        "required": ["config_path"],
        "optional": {"fold": 0, "probe_samples": 500},
        "description": "Validates hypothesis on a small subsample of data.",
    },
    "run_full_validation": {
        "summary": "Execute full multi-fold cross-validation (local or remote).",
        "required": ["config_path"],
        "optional": {"backend": "local", "force": False, "auto_audit": True},
        "description": "Runs complete CV evaluation, writes immutable artifacts, and seals hashes.",
    },
    "inspect_experiment": {
        "summary": "Inspect performance, metrics, and manifest of a completed experiment.",
        "required": ["exp_id"],
        "optional": {},
        "description": "Retrieves verified scores and fold breakdown for a given exp_id.",
    },
    "inspect_diagnostics": {
        "summary": "Inspect current error distributions, residual diagnostics, and metric breakdowns.",
        "required": [],
        "optional": {"exp_id": None},
        "description": "Provides diagnostic insights into model weaknesses and error clusters.",
    },
    "optimize_blend": {
        "summary": "Optimize ensemble weights across multiple candidate predictions.",
        "required": [],
        "optional": {"components": None, "blend_mode": "nelder_mead", "name": "blend_v1"},
        "description": "Finds metric-optimal blending weights using out-of-fold predictions.",
    },
    "optimize_threshold": {
        "summary": "Optimize decision thresholds on validation probabilities for F1 / Kappa / etc.",
        "required": [],
        "optional": {"exp_id": None, "metric": None, "num_thresholds": 100},
        "description": "Tunes classification thresholds to maximize the competition metric.",
    },
    "prepare_submission": {
        "summary": "Package candidate predictions into competition-ready submission format.",
        "required": [],
        "optional": {"exp_id": None, "candidate_csv": None, "notes": ""},
        "description": "Validates format, columns, and row counts against submission requirements.",
    },
    "inspect_cloud_job": {
        "summary": "Query status and logs of a remote cloud execution job.",
        "required": ["job_id"],
        "optional": {},
        "description": "Checks remote job execution state and extracts recent logs.",
    },
    "wait_cloud_job": {
        "summary": "Wait and poll until a remote cloud job completes.",
        "required": ["job_id"],
        "optional": {"poll_interval": 30, "timeout_seconds": 3600},
        "description": "Blocks until the remote job finishes, importing artifacts on completion.",
    },
    "confirm_task_spec": {
        "summary": "Confirm task specification and schema alignment.",
        "required": [],
        "optional": {},
        "description": "Marks the task specification gate as confirmed.",
    },
    "inspect_data_readiness": {
        "summary": "Inspect raw data formats, leakage risks, and splits.",
        "required": [],
        "optional": {"config": None},
        "description": "Verifies data files and split integrity.",
    },
    "confirm_data_risks": {
        "summary": "Confirm that data leakages and risks have been audited.",
        "required": [],
        "optional": {},
        "description": "Marks the data risks gate as confirmed.",
    },
    "run_eda": {
        "summary": "Run automated exploratory data analysis.",
        "required": [],
        "optional": {"data_path": None},
        "description": "Executes standard exploratory data analysis on target dataset.",
    },
    "update_task_contract": {
        "summary": "Update task contract fields.",
        "required": ["task"],
        "optional": {},
        "description": "Modifies task contract specification.",
    },
    "update_workspace_settings": {
        "summary": "Update workspace settings and feature flags.",
        "required": [],
        "optional": {"mode": None, "manual_phase_override": None, "agent_context_config": None},
        "description": "Updates workspace configuration.",
    },
    "bootstrap_task": {
        "summary": "Bootstrap a new task workspace.",
        "required": [],
        "optional": {"project_root": None, "dataset_dir": None, "output_config": None, "context_file": None},
        "description": "Scaffolds a new competition workspace.",
    },
    "record_phase_output": {
        "summary": "Record output of a competition phase.",
        "required": [],
        "optional": {"phase_id": None, "response": None, "response_file": None, "artifacts": None, "sources": None, "metadata": None},
        "description": "Records phase output in competition state.",
    },
    "synthesize_candidate": {
        "summary": "Synthesize a new ensemble candidate.",
        "required": ["exp_id"],
        "optional": {"mode": None, "name": None, "revisit_reason": None, "source_map": None},
        "description": "Synthesizes an ensemble candidate from component models.",
    },
    "promote_candidate": {
        "summary": "Promote a candidate run to active pool.",
        "required": ["exp_id"],
        "optional": {},
        "description": "Promotes an experiment to the candidate pool.",
    },
    "build_ensemble": {
        "summary": "Build an ensemble model from verified candidates.",
        "required": ["exp_id"],
        "optional": {},
        "description": "Constructs an ensemble artifact.",
    },
    "register_external_prediction": {
        "summary": "Register external prediction files.",
        "required": [],
        "optional": {"name": None, "oof_path": None, "test_path": None, "source": None, "author": None, "notes": None},
        "description": "Imports external predictions into candidate pool.",
    },
    "generate_microdose_probe": {
        "summary": "Generate a microdose probe blending baseline with candidate.",
        "required": [],
        "optional": {"base_exp_id": None, "probe_exp_id": None, "alpha": None, "space": None},
        "description": "Creates a conservative blend probe.",
    },
    "record_external_score": {
        "summary": "Record external leaderboard score.",
        "required": ["primary_score"],
        "optional": {"sub_id": None, "split_type": None, "notes": None, "custom_scores": None},
        "description": "Records score from external competition platform.",
    },
    "freeze_candidate": {
        "summary": "Freeze final submission candidates.",
        "required": [],
        "optional": {"slot_1_id": None, "slot_2_id": None, "notes": ""},
        "description": "Locks selected candidates for final submission.",
    },
    "emergency_unfreeze": {
        "summary": "Unfreeze candidates in emergency situation.",
        "required": [],
        "optional": {"reason": ""},
        "description": "Unlocks candidate freeze with justification.",
    },
    "finalize_competition": {
        "summary": "Finalize competition and generate endgame report.",
        "required": [],
        "optional": {},
        "description": "Concludes competition lifecycle.",
    },
    "write_experiment_config": {
        "summary": "Write an experiment configuration file.",
        "required": ["path", "content", "insight_id"],
        "optional": {},
        "description": "Writes an experiment configuration with insight tracking.",
    },
    "candidate_preflight": {
        "summary": "Preflight verification of candidate predictions against incumbent baseline.",
        "required": ["candidate_path"],
        "optional": {"incumbent_path": None, "id_col": None, "pred_col": None, "citation_dependencies": []},
        "description": "Computes Spearman rho, rank swaps, and delta distributions. Flags rho >= 0.995; rank similarity alone does not establish score cause.",
    },
    "validate_candidate_pitch": {
        "summary": "Verify a candidate pitch citation against its stored preflight and prediction hash.",
        "required": ["preflight_id"],
        "optional": {
            "candidate_path": None,
            "claim_kind": "submission_candidate",
            "candidate_exp_id": None,
            "incumbent_exp_id": None,
            "paired_oof_evaluation_id": None,
            "citation_dependencies": None,
        },
        "description": "Requires a current preflight citation. Near-duplicates (rho >= 0.995) require an indexed, machine-supported paired OOF gain tied to candidate and incumbent IDs.",
    },
    "reconcile_submissions": {
        "summary": "Reconcile live provider submissions against local ledger.",
        "required": [],
        "optional": {"competition": None},
        "description": "Audits live platform dispatches against local submission events to catch external bypasses.",
    },
    "acknowledge_external_submissions": {
        "summary": "Acknowledge and clear external submission discrepancy warnings.",
        "required": ["refs", "reason"],
        "optional": {},
        "description": "Records operator acknowledgement of external dispatches with justification, unlocking submission gate.",
    },
}


def get_action_schema(action: str) -> Optional[Dict[str, Any]]:
    """Return schema definition for a Typed Action, if registered."""
    return TYPED_ACTION_SCHEMAS.get(resolve_action_name(action))

