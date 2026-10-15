"""ML Experiment Harness - Autonomous Agentic ML/DL Experimentation Platform.

Combines AIDE ML (Tree Search & Memory Compression) with SWE-agent (Deterministic Audit Gate).
"""

from .tree_journal import Node, TreeJournal
from .validator import ExperimentValidator
from .competition_state import CompetitionStateMachine, CompetitionState
from .agent_bridge import AgentBridge
from .ensemble_guard import EnsembleGuard
from .submission_manager import SubmissionManager
from .stacking import L2LogitStacker, RankAverager, align_prediction_frames, to_clipped_logit
from .foundation_models import TeacherLogitPrior, MultiScaleBinner, HierarchicalDiscretizer
from .endgame_submission import (
    SubmissionPreflightValidator,
    SubmissionPreflightValidator as SubmissionValidator,
    KaggleLeaderboardTracker,
    ThresholdOptimizer,
    QuantileMatcher,
    MicroDoseProbeGenerator,
)
from .templates.kaggle_template import KaggleKernelManager, KaggleKernelManager as kaggle_manager
from .models import (
    RobustTabularTransformer,
    NativeCategoricalTransformer,
    RandomForestModel,
)
from .evaluation import (
    metric_name,
    default_metric_direction,
    compute_rmsle,
    compute_smape,
    compute_wape,
    compute_pinball_loss,
    compute_pearson,
    compute_mrr,
    compute_dice_score,
    compute_iou_score,
    mask_to_rle,
    rle_to_mask,
    compute_box_iou,
    compute_wer,
    compute_cer,
    compute_bleu,
    compute_exact_match,
)
from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    json_default_serializer,
    sanitize_for_json,
)

__all__ = [
    "Node",
    "TreeJournal",
    "ExperimentValidator",
    "CompetitionStateMachine",
    "CompetitionState",
    "AgentBridge",
    "EnsembleGuard",
    "SubmissionManager",
    "L2LogitStacker",
    "RankAverager",
    "align_prediction_frames",
    "to_clipped_logit",
    "TeacherLogitPrior",
    "MultiScaleBinner",
    "HierarchicalDiscretizer",
    "SubmissionPreflightValidator",
    "SubmissionValidator",
    "KaggleKernelManager",
    "kaggle_manager",
    "KaggleLeaderboardTracker",
    "ThresholdOptimizer",
    "QuantileMatcher",
    "MicroDoseProbeGenerator",
    "RobustTabularTransformer",
    "NativeCategoricalTransformer",
    "RandomForestModel",
    "metric_name",
    "default_metric_direction",
    "compute_rmsle",
    "compute_smape",
    "compute_wape",
    "compute_pinball_loss",
    "compute_pearson",
    "compute_mrr",
    "compute_dice_score",
    "compute_iou_score",
    "mask_to_rle",
    "rle_to_mask",
    "compute_box_iou",
    "compute_wer",
    "compute_cer",
    "compute_bleu",
    "compute_exact_match",
    "atomic_write_json",
    "atomic_write_text",
    "json_default_serializer",
    "sanitize_for_json",
]

