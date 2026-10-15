"""ML Experiment Harness - Autonomous Agentic ML/DL Experimentation Platform.

Combines AIDE ML (Tree Search & Memory Compression) with SWE-agent (Deterministic Audit Gate).

The public API is exposed lazily (PEP 562 ``__getattr__``) so that importing the
package (which every ``from harness.<submodule> import ...`` and every CLI
subprocess triggers) does not eagerly import the heavy submodules. Eager import
used to pull in scipy.stats, lightgbm, xgboost and the pandas/sklearn stack at
``import harness`` time; deferring it to first attribute access keeps the common
import path fast while remaining fully backwards-compatible: any name in
``__all__`` is imported on demand, and ``from harness import <submodule>`` keeps
working via ``sys.modules`` registration.
"""

from __future__ import annotations

# name -> the harness submodule that defines it.
_LAZY_MAP = {
    # tree_journal
    "Node": ".tree_journal",
    "TreeJournal": ".tree_journal",
    # validator
    "ExperimentValidator": ".validator",
    # competition_state
    "CompetitionStateMachine": ".competition_state",
    "CompetitionState": ".competition_state",
    # agent_bridge
    "AgentBridge": ".agent_bridge",
    # ensemble_guard
    "EnsembleGuard": ".ensemble_guard",
    # submission_manager
    "SubmissionManager": ".submission_manager",
    # stacking
    "L2LogitStacker": ".stacking",
    "RankAverager": ".stacking",
    "align_prediction_frames": ".stacking",
    "to_clipped_logit": ".stacking",
    # foundation_models
    "TeacherLogitPrior": ".foundation_models",
    "MultiScaleBinner": ".foundation_models",
    "HierarchicalDiscretizer": ".foundation_models",
    # endgame_submission
    "SubmissionPreflightValidator": ".endgame_submission",
    "SubmissionValidator": ".endgame_submission",
    "KaggleLeaderboardTracker": ".endgame_submission",
    "ThresholdOptimizer": ".endgame_submission",
    "QuantileMatcher": ".endgame_submission",
    "MicroDoseProbeGenerator": ".endgame_submission",
    # templates.kaggle_template
    "KaggleKernelManager": (".templates.kaggle_template", "KaggleKernelManager"),
    "kaggle_manager": (".templates.kaggle_template", "KaggleKernelManager"),
    # models
    "RobustTabularTransformer": ".models",
    "NativeCategoricalTransformer": ".models",
    "RandomForestModel": ".models",
    # evaluation
    "metric_name": ".evaluation",
    "default_metric_direction": ".evaluation",
    "compute_rmsle": ".evaluation",
    "compute_smape": ".evaluation",
    "compute_wape": ".evaluation",
    "compute_pinball_loss": ".evaluation",
    "compute_pearson": ".evaluation",
    "compute_mrr": ".evaluation",
    "compute_dice_score": ".evaluation",
    "compute_iou_score": ".evaluation",
    "mask_to_rle": ".evaluation",
    "rle_to_mask": ".evaluation",
    "compute_box_iou": ".evaluation",
    "compute_wer": ".evaluation",
    "compute_cer": ".evaluation",
    "compute_bleu": ".evaluation",
    "compute_exact_match": ".evaluation",
    # io_utils
    "atomic_write_json": ".io_utils",
    "atomic_write_text": ".io_utils",
    "json_default_serializer": ".io_utils",
    "sanitize_for_json": ".io_utils",
}

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


def __getattr__(name):
    # Submodule alias access (e.g. `from harness import task_spec` or
    # `import harness.task_spec`): import and register the submodule so the
    # normal `harness.<submodule>` attribute lookup succeeds.
    import importlib
    import sys

    if name in sys.modules:
        return sys.modules[name]
    submodule = "harness." + name
    try:
        module = importlib.import_module(submodule)
    except ImportError:
        module = None
    if module is not None and getattr(module, "__name__", "") == submodule:
        sys.modules[name] = module
        globals()[name] = module
        return module

    # Public API name: resolve through the lazy map.
    target = _LAZY_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'harness' has no attribute {name!r}")
    if isinstance(target, str):
        target = (target, name)
    rel, attr = target
    module = importlib.import_module(rel, "harness")
    value = getattr(module, attr)
    globals()[name] = value  # cache so repeated access is O(1)
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
