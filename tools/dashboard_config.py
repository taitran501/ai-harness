#!/usr/bin/env python3
"""Task-config I/O and priming-prompt helpers for the experimentation dashboard.

Extracted from ``tools/dashboard_server.py`` (M4 refactoring, PR-D): these are
the dashboard's *data* layer — task discovery/switching, task-config
load/save, experiment-ID sequencing, and priming-prompt generation. The HTTP
layer (``DashboardHandler``) imports them from here.

Project discovery and the ``HARNESS_PROJECT_ROOT`` / ``HARNESS_TASK_CONFIG``
environment side-effects live here as the single source of truth; importing
this module is what initializes the dashboard project context.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Ensure harness directory and project root in sys.path
HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from harness.io_utils import get_project_root

from harness.project import HarnessProject

PROJECT_ROOT = get_project_root()
HARNESS_PROJECT = HarnessProject.discover(PROJECT_ROOT)
ARTIFACTS_DIR = HARNESS_PROJECT.artifacts_dir
os.environ["HARNESS_PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["HARNESS_TASK_CONFIG"] = str(HARNESS_PROJECT.task_config_path)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.agent_bridge import AgentBridge
from harness.competition_state import CompetitionStateMachine
from harness.submission_manager import SubmissionManager
from harness.tree_journal import TreeJournal


def get_next_experiment_id(branch: str = "feat", descriptor: str = "run") -> Dict[str, Any]:
    """Find the next consecutive experiment ID in task configs and managed artifacts."""
    exp_numbers: List[int] = []

    # 1. Scan configs directory
    configs_dir = PROJECT_ROOT / "configs"
    if configs_dir.exists():
        for cfg in configs_dir.glob("*.yaml"):
            m = re.search(r"exp(\d+)", cfg.name, re.IGNORECASE)
            if m:
                exp_numbers.append(int(m.group(1)))
            try:
                content = cfg.read_text(encoding="utf-8")
                m2 = re.search(r'id:\s*["\']?exp(\d+)', content, re.IGNORECASE)
                if m2:
                    exp_numbers.append(int(m2.group(1)))
            except Exception:
                pass

    # 2. Scan artifacts directory
    art_dir = ARTIFACTS_DIR
    if art_dir.exists():
        for exp_folder in art_dir.iterdir():
            if exp_folder.is_dir():
                m = re.search(r"exp(\d+)", exp_folder.name, re.IGNORECASE)
                if m:
                    exp_numbers.append(int(m.group(1)))

    next_num = max(exp_numbers, default=0) + 1
    clean_desc = re.sub(r"[^a-zA-Z0-9_]+", "_", (descriptor or "run").strip()).strip("_").lower() or "run"
    clean_branch = (branch or "feat").strip().lower()
    next_id = f"exp{next_num:03d}_{clean_branch}_{clean_desc}"
    suggested_file = f"configs/{next_id}.yaml"

    return {
        "next_id": next_id,
        "next_number": next_num,
        "branch": clean_branch,
        "descriptor": clean_desc,
        "suggested_file": suggested_file,
    }


DEFAULT_STRATEGY_CARDS = [
    {
        "id": "feat_representation",
        "branch": "feat",
        "title": "Input Representation & Feature Pipeline",
        "problem_addressed": "Raw feature representations miss higher-order interactions and domain signals.",
        "hypothesis": "Composing structured feature representations and n-gram/polynomial features improves baseline score.",
        "intervention": "Compose enriched input representation and normalize numerical/text features.",
        "approach_type": "ml",
        "model_family": "linear",
        "model_name": "logistic_regression",
        "descriptor": "feat_pipeline",
        "rationale": "Structured representation engineering exposes predictive feature interactions directly to the classifier.",
        "target_tasks": ["Primary Task Metric"],
        "uncertainty_level": "low",
        "source_type": "verified_baseline",
        "source_id": "feature_pipeline",
        "model_params": {
            "C": 1.0,
            "max_iter": 1000,
        },
    },
    {
        "id": "model_pretrained_encoder",
        "branch": "model",
        "title": "Pretrained Transformer Backbone",
        "problem_addressed": "Classical shallow models lack contextual representations across challenging test slices.",
        "hypothesis": "Fine-tuning a modern pretrained encoder backbone captures deep context and improves validation performance.",
        "intervention": "Replace shallow baseline with fine-tuned pretrained transformer encoder.",
        "approach_type": "dl",
        "model_family": "transformer",
        "model_name": "pretrained_encoder_base",
        "descriptor": "transformer_backbone",
        "rationale": "Pretrained representations transfer rich semantic abstractions directly to the competition dataset.",
        "target_tasks": ["All Prediction Targets"],
        "uncertainty_level": "medium",
        "source_type": "huggingface_hub",
        "source_id": "pretrained_encoder",
        "model_params": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 3,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        },
    },
    {
        "id": "loss_focal_balanced",
        "branch": "loss",
        "title": "Class-Balanced / Focal Loss Optimization",
        "problem_addressed": "Severe class imbalance and label skew degrade recall on minority classes.",
        "hypothesis": "Employing focal loss with dynamically adjusted alpha focuses gradient updates on hard minority examples.",
        "intervention": "Switch cross-entropy objective to Class-Balanced Focal Loss (gamma=2.0, alpha=0.25).",
        "approach_type": "dl",
        "model_family": "transformer",
        "model_name": "pretrained_encoder_base",
        "descriptor": "focal_balanced_loss",
        "rationale": "Focal Loss dynamically scales cross-entropy down for easy negatives, focusing gradients squarely on hard rare minority classes.",
        "target_tasks": ["Minority Target Classes"],
        "uncertainty_level": "medium",
        "source_type": "verified_literature",
        "source_id": "lin2017focal",
        "model_params": {
            "gamma": 2.0,
            "alpha": 0.25,
            "loss_type": "focal",
        },
    },
    {
        "id": "ens_threshold_opt",
        "branch": "ens",
        "title": "Out-of-Fold Threshold Calibration & Ensembling",
        "problem_addressed": "Static decision thresholds produce miscalibrated confidence scores across validation partitions.",
        "hypothesis": "Optimizing decision thresholds per slice on out-of-fold validation probabilities via Nelder-Mead maximizes primary metric.",
        "intervention": "Apply harness ThresholdOptimizer on out-of-fold validation probabilities.",
        "approach_type": "ml",
        "model_family": "ensemble",
        "model_name": "threshold_optimizer",
        "descriptor": "threshold_tuning",
        "rationale": "Threshold search shifts the precision-recall trade-off to maximize Macro-F1 directly on validation folds without retraining.",
        "target_tasks": ["All Targets"],
        "uncertainty_level": "low",
        "source_type": "verified_baseline",
        "source_id": "harness.tuning_engine",
        "model_params": {
            "method": "nelder-mead",
            "metric": "macro_f1",
        },
    },
    {
        "id": "model_unsloth_qwen_7b",
        "branch": "model",
        "title": "Fast QLoRA Fine-Tuning (Qwen2.5 7B)",
        "problem_addressed": "Full fine-tuning large language models exceeds single-GPU memory and compute limits.",
        "hypothesis": "Fine-tuning with 4-bit QLoRA adapts generative knowledge to the competition format with minimal compute.",
        "intervention": "Execute 4-bit QLoRA fine-tuning on quantized base LLM with custom classification formatting.",
        "approach_type": "llm",
        "model_family": "llm_peft",
        "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "descriptor": "unsloth_qwen_7b_qlora",
        "rationale": "4-bit quantized Qwen 7B consumes <9GB VRAM with Unsloth Triton kernels while capturing domain context.",
        "target_tasks": ["Primary Targets"],
        "param_size": "7B",
        "uncertainty_level": "low",
        "source_type": "unsloth_hub",
        "source_id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "model_params": {
            "param_size": "7B",
            "quantization": "4-bit NormalFloat (NF4)",
            "paradigm": "unsloth_qlora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "epochs": 3,
            "learning_rate": 2e-4,
        },
    },
    {
        "id": "model_deepseek_r1_cot_14b",
        "branch": "model",
        "title": "DeepSeek-R1 Distill 14B Chain-of-Thought",
        "problem_addressed": "Complex classification tasks require structured reasoning steps prior to emitting categorical labels.",
        "hypothesis": "Few-shot Chain-of-Thought prompting unpacks subtle nuances and establishes a strong zero-training baseline.",
        "intervention": "Apply structured Chain-of-Thought prompting with JSON classification schema parsing.",
        "approach_type": "llm",
        "model_family": "llm_reasoning",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "descriptor": "deepseek_r1_cot_14b",
        "rationale": "Reasoning models excel at unpacking implicit subtext and domain tropes without task-specific retraining.",
        "target_tasks": ["Target Classifications"],
        "param_size": "14B",
        "uncertainty_level": "medium",
        "source_type": "huggingface_hub",
        "source_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "model_params": {
            "param_size": "14B",
            "quantization": "4-bit AWQ / NF4",
            "paradigm": "zero_shot_cot",
            "use_cot": True,
            "temperature": 0.2,
            "context_length": 8192,
        },
    },
]

def list_available_tasks() -> List[Dict[str, Any]]:
    """List all available competition tasks from configs/tasks/ and mark the active one."""
    tasks_dirs = [PROJECT_ROOT / "configs" / "tasks", HARNESS_DIR / "configs" / "tasks", HARNESS_DIR / "harness" / "defaults" / "tasks"]
    for td in tasks_dirs:
        td.mkdir(parents=True, exist_ok=True)

    active_task = load_task_config()
    active_id = active_task.get("id") or active_task.get("task_name") or active_task.get("name", "general")

    results = []
    found_ids = set()

    # 1. Scan configs/tasks/*.yaml across PROJECT_ROOT and HARNESS_DIR
    for td in tasks_dirs:
        for t_file in sorted(td.glob("*.yaml")):
            try:
                with open(t_file, "r", encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                    t_spec = d.get("task", d)
                    tid = t_spec.get("id", t_file.stem)
                    if tid in found_ids:
                        continue
                    found_ids.add(tid)
                    results.append({
                        "id": tid,
                        "name": t_spec.get("task_name") or t_spec.get("name", tid),
                        "platform": t_spec.get("platform", "generic"),
                        "type": t_spec.get("task_type") or t_spec.get("type", "Predictive Modeling"),
                        "modality": t_spec.get("modality", "generic"),
                        "primary_metric": t_spec.get("primary_metric", "Score"),
                        "direction": t_spec.get("direction", "maximize"),
                        "file_path": str(t_file.relative_to(PROJECT_ROOT) if PROJECT_ROOT in t_file.parents else t_file.name).replace("\\", "/"),
                        "is_active": (tid == active_id or t_spec.get("name") == active_task.get("name")),
                    })
            except Exception:
                pass

    # 2. If configs/tasks was empty or missing active task, add active task
    if not results or not any(r["is_active"] for r in results):
        results.insert(0, {
            "id": active_id,
            "name": active_task.get("task_name") or active_task.get("name", "Active Task"),
            "platform": active_task.get("platform", "custom"),
            "type": active_task.get("task_type") or active_task.get("type", "Predictive Modeling"),
            "modality": active_task.get("modality", "generic"),
            "primary_metric": active_task.get("primary_metric", "Score"),
            "direction": active_task.get("direction", "maximize"),
            "file_path": "configs/task.yaml",
            "is_active": True,
        })

    return results


def switch_task(task_id: str) -> Dict[str, Any]:
    """A dashboard is attached to one immutable task workspace."""
    return {
        "success": False,
        "error": "Each task needs its own workspace. Attach the harness from the task workspace you want to use.",
    }


def load_task_config() -> Dict[str, Any]:
    """Load dynamic task specification from configs/task.yaml or fall back to generic schema."""
    task_file = HARNESS_PROJECT.task_config_path
    if task_file.exists():
        try:
            import yaml
            with open(task_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if "task" in data:
                    return data["task"]
        except Exception:
            pass

    return {
        "id": "general",
        "platform": "custom",
        "name": "General ML/DL Benchmark",
        "type": "Predictive Modeling",
        "modality": "generic",
        "primary_metric": "Score",
        "direction": "maximize",
        "metrics": [],
        "languages": [],
        "description": "Task-agnostic ML/DL experimentation harness.",
    }


def save_task_config(task_data: Dict[str, Any]) -> None:
    """Save updated task specification to configs/task.yaml."""
    _, obs = AgentBridge(PROJECT_ROOT).execute_action("update_task_contract", params={"task": task_data})
    if obs.status != "SUCCESS":
        raise ValueError(obs.summary)


TASK_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "multitask_text_classification",
        "preset_name": "Generic Multi-Task Text Classification",
        "platform": "custom",
        "name": "Multi-Task Text Classification",
        "type": "Multi-Task Text Classification",
        "modality": "text",
        "primary_metric": "Macro-F1",
        "direction": "maximize",
        "metrics": [
            {"id": "target_primary", "name": "Primary Target", "metric": "Macro-F1", "direction": "maximize", "weight": 0.5, "description": "Primary classification target"},
            {"id": "target_secondary", "name": "Secondary Target", "metric": "Macro-F1", "direction": "maximize", "weight": 0.5, "description": "Optional fine-grained target"},
        ],
        "languages": [],
        "description": "Generic multi-task text classification example with configurable validation.",
        "data_config": {
            "train_path": "dataset/",
            "test_path": "dataset/test/",
            "split_strategy": "KFold",
            "n_folds": 5,
            "seed": 42,
            "id_column": "id",
            "target_column": "target_1,target_2",
            "missing_data_policy": "drop_invalid_rows",
        },
    },
    {
        "id": "kaggle_tabular_regression",
        "preset_name": "Kaggle Tabular Regression (e.g. House Prices)",
        "platform": "kaggle",
        "name": "House Prices Regression",
        "type": "Tabular Regression",
        "modality": "tabular",
        "primary_metric": "RMSE",
        "direction": "minimize",
        "primary_metric_baseline": 0.145,
        "primary_metric_target": 0.110,
        "metrics": [
            {"id": "rmse", "name": "Root Mean Squared Error", "metric": "RMSE", "direction": "minimize", "baseline_score": 0.145, "target_score": 0.110, "weight": 1.0, "description": "Log-scale RMSE"},
            {"id": "mae", "name": "Mean Absolute Error", "metric": "MAE", "direction": "minimize", "baseline_score": 0.095, "target_score": 0.075, "weight": 0.0, "description": "L1 error reference"},
            {"id": "r2", "name": "R-Squared", "metric": "R2", "direction": "maximize", "baseline_score": 0.890, "target_score": 0.940, "weight": 0.0, "description": "Variance explained"},
        ],
        "languages": [],
        "description": "Tabular regression benchmark optimizing Root Mean Squared Error (RMSE).",
        "data_config": {
            "train_path": "dataset/train.csv",
            "test_path": "dataset/test.csv",
            "split_strategy": "KFold",
            "n_folds": 5,
            "seed": 42,
            "id_column": "Id",
            "target_column": "SalePrice",
            "missing_data_policy": "impute_median_categorical_missing",
        },
        "competition_rules": {
            "allow_llm": False,
            "allow_api": False,
            "allow_external_data": False,
            "allow_pretrained_models": False,
            "max_params_limit": 50000000,
            "daily_submission_limit": 5,
            "deadline": "2026-12-31T23:59:59Z",
            "reproducibility_required": True,
        },
        "submission_spec": {
            "file_name": "submission.csv",
            "archive_format": "csv",
            "min_diff_pct": 2.0,
            "max_kappa": 0.85,
            "min_disagreement": 5.0,
        },
    },
    {
        "id": "kaggle_binary_classification",
        "preset_name": "Kaggle Binary Classification (e.g. Titanic / Fraud)",
        "platform": "kaggle",
        "name": "Binary Classification Benchmark",
        "type": "Tabular Binary Classification",
        "modality": "tabular",
        "primary_metric": "ROC-AUC",
        "direction": "maximize",
        "primary_metric_baseline": 0.812,
        "primary_metric_target": 0.880,
        "metrics": [
            {"id": "roc_auc", "name": "Area Under ROC Curve", "metric": "ROC-AUC", "direction": "maximize", "baseline_score": 0.812, "target_score": 0.880, "weight": 1.0, "description": "Primary ranking metric"},
            {"id": "accuracy", "name": "Classification Accuracy", "metric": "Accuracy", "direction": "maximize", "baseline_score": 0.795, "target_score": 0.860, "weight": 0.0, "description": "Top-level accuracy"},
            {"id": "logloss", "name": "Logarithmic Loss", "metric": "LogLoss", "direction": "minimize", "baseline_score": 0.440, "target_score": 0.350, "weight": 0.0, "description": "Calibrated probability cross-entropy"},
        ],
        "languages": [],
        "description": "Binary classification benchmark optimizing area under the ROC curve.",
        "data_config": {
            "train_path": "dataset/train.csv",
            "test_path": "dataset/test.csv",
            "split_strategy": "StratifiedKFold",
            "n_folds": 5,
            "seed": 42,
            "id_column": "id",
            "target_column": "target",
            "missing_data_policy": "impute_iterative",
        },
        "competition_rules": {
            "allow_llm": False,
            "allow_api": False,
            "allow_external_data": False,
            "allow_pretrained_models": True,
            "max_params_limit": 50000000,
            "daily_submission_limit": 5,
            "deadline": "2026-11-30T23:59:59Z",
            "reproducibility_required": True,
        },
        "submission_spec": {
            "file_name": "submission.csv",
            "archive_format": "csv",
            "min_diff_pct": 2.0,
            "max_kappa": 0.85,
            "min_disagreement": 5.0,
        },
    },
    {
        "id": "vision_multiclass",
        "preset_name": "Computer Vision Multi-Class (e.g. ImageNet / CIFAR)",
        "platform": "kaggle",
        "name": "Vision Classification Benchmark",
        "type": "Image Classification",
        "modality": "vision",
        "primary_metric": "Top-1-Accuracy",
        "direction": "maximize",
        "primary_metric_baseline": 0.840,
        "primary_metric_target": 0.920,
        "metrics": [
            {"id": "top1_acc", "name": "Top-1 Accuracy", "metric": "Accuracy", "direction": "maximize", "baseline_score": 0.840, "target_score": 0.920, "weight": 1.0, "description": "Top-1 accuracy"},
            {"id": "top5_acc", "name": "Top-5 Accuracy", "metric": "Top-5 Acc", "direction": "maximize", "baseline_score": 0.950, "target_score": 0.985, "weight": 0.0, "description": "Top-5 accuracy"},
            {"id": "loss", "name": "Cross-Entropy Loss", "metric": "Loss", "direction": "minimize", "baseline_score": 0.650, "target_score": 0.320, "weight": 0.0, "description": "Multi-class cross-entropy"},
        ],
        "languages": [],
        "description": "Multi-class image classification benchmark optimizing Top-1 Accuracy.",
        "data_config": {
            "train_path": "dataset/train/",
            "test_path": "dataset/test/",
            "split_strategy": "StratifiedKFold",
            "n_folds": 5,
            "seed": 42,
            "id_column": "image_id",
            "target_column": "label",
            "missing_data_policy": "verify_image_integrity",
        },
        "competition_rules": {
            "allow_llm": False,
            "allow_api": False,
            "allow_external_data": True,
            "allow_pretrained_models": True,
            "max_params_limit": 250000000,
            "daily_submission_limit": 5,
            "deadline": "2026-10-31T23:59:59Z",
            "reproducibility_required": True,
        },
        "submission_spec": {
            "file_name": "submission.csv",
            "archive_format": "csv",
            "min_diff_pct": 2.0,
            "max_kappa": 0.85,
            "min_disagreement": 5.0,
        },
    },
    {
        "id": "custom_blank",
        "preset_name": "Custom / Blank Task",
        "platform": "custom",
        "name": "Custom Experimentation Benchmark",
        "type": "Supervised Learning",
        "modality": "generic",
        "primary_metric": "Score",
        "direction": "maximize",
        "metrics": [
            {"id": "primary_metric", "name": "Primary Metric", "metric": "Score", "direction": "maximize", "weight": 1.0, "description": "Primary benchmark objective"},
        ],
        "languages": [],
        "description": "Custom machine learning / deep learning experimentation task.",
        "data_config": {
            "train_path": "dataset/train.csv",
            "test_path": "dataset/test.csv",
            "split_strategy": "KFold",
            "n_folds": 5,
            "seed": 42,
            "id_column": "id",
            "target_column": "target",
            "missing_data_policy": "standard",
        },
        "competition_rules": {
            "allow_llm": False,
            "allow_api": False,
            "allow_external_data": False,
            "allow_pretrained_models": True,
            "max_params_limit": 150000000,
            "daily_submission_limit": 5,
            "reproducibility_required": True,
        },
        "submission_spec": {
            "file_name": "submission.csv",
            "archive_format": "csv",
            "min_diff_pct": 2.0,
            "max_kappa": 0.85,
            "min_disagreement": 5.0,
        },
    },
]

# Keep the historical Codabench preset ID while sharing the same generic,
# task-configurable multi-task text contract.
TASK_PRESETS.insert(1, {
    **TASK_PRESETS[0],
    "id": "codabench_nlp_multitask",
    "preset_name": "Codabench NLP Multi-Task (generic)",
    "platform": "codabench",
    "name": "Codabench NLP Multi-Task",
})


def generate_prompt_by_template(journal: TreeJournal, template_id: str = "codex") -> Dict[str, Any]:
    """Generate dynamic prompt templates reflecting live solution tree state for various coding agents."""
    task = load_task_config()
    best = journal.get_best_node()
    rec = journal.recommend_next_branch()
    total_runs = len(journal.nodes)

    best_id = best.id if best else "None"
    best_score = f"{best.get_score():.4f}" if best else "0.0000"
    best_type = getattr(best, "approach_type", "ml").upper() if best else "N/A"
    best_family = getattr(best, "model_family", "linear") if best else "N/A"
    best_params = f"{best.param_count / 1e6:.1f}M" if (best and best.param_count) else "N/A"

    task_name = task.get("name", "ML/DL Task")
    task_type = task.get("type", "Machine Learning Task")
    task_modality = task.get("modality", "generic")
    primary_metric = task.get("primary_metric", "Primary Metric")
    direction = task.get("direction", "maximize")
    artifact_rel = ARTIFACTS_DIR.relative_to(PROJECT_ROOT).as_posix()
    data_config = task.get("data_config", {})
    fold_count = data_config.get("n_folds") or task.get("validation", {}).get("num_folds")
    validation_label = f"{fold_count}-fold validation" if fold_count else "the configured validation scheme"
    custom_plugin = HARNESS_PROJECT.plugin_id not in (None, "default")
    if custom_plugin:
        submission_guidance = (
            f"Use task plugin '{HARNESS_PROJECT.plugin_id}' for submission building and validation. "
            "If either hook is missing, stop and report that the task is not submission-ready; "
            "use test-set predictions rather than validation OOF predictions."
        )
    else:
        submission_guidance = (
            "Use the configured generic submission checks and task-declared output format; "
            "keep validation metrics separate from external scores."
        )

    # Dynamic diagnostics extraction
    diagnostics_notes = []
    if best and best.diagnostics_summary:
        diag = best.diagnostics_summary
        for k, v in diag.items():
            if isinstance(v, dict):
                zero_keys = [sub_k for sub_k, sub_v in v.items() if isinstance(sub_v, dict) and sub_v.get("recall", 1.0) == 0.0]
                if zero_keys:
                    diagnostics_notes.append(f"- **Zero-Recall Slices**: Sub-classes {zero_keys} in category `{k}` scored 0.00 recall in `{best_id}`.")
            elif isinstance(v, list) and k in ("critical_bottlenecks", "top_confusions"):
                for item in v[:3]:
                    diagnostics_notes.append(f"- **{k.replace('_', ' ').title()}**: {item}")
    if not diagnostics_notes:
        diagnostics_notes.append(f"- **Performance Frontier**: Target metric `{primary_metric}` stands at `{best_score}`. Focus on generalization on validation splits.")

    graveyard_items = ["- Search research memory below for decisions and evidence; legacy notes are not a permanent ban."]

    rec_parent = rec.get("parent_id") or best_id
    rec_branch = rec.get("branch") or "feat"

    # Extract latest Competition Submissions & Codabench Leaderboard feedback
    submission_feedback_lines = []
    platform_name = str(task.get("platform", "Leaderboard")).title()
    try:
        sub_entries = SubmissionManager().get_all_submissions()
        if sub_entries:
            for s in sub_entries[-5:]:
                cv_val = s.cv_scores.get("primary_metric") or s.cv_scores.get(primary_metric) or next(iter(s.cv_scores.values()), "N/A") if s.cv_scores else "N/A"
                lb_val = f"Pending on {platform_name}"
                if s.official_scores:
                    lb_num = (
                        s.official_scores.get("primary_metric")
                        or s.official_scores.get(primary_metric)
                        or s.official_scores.get("score")
                    )
                    if lb_num is not None:
                        lb_val = f"{float(lb_num):.4f}"
                note_str = f" — Note: {s.notes}" if s.notes else ""
                submission_feedback_lines.append(
                    f"- **`{s.id}` (`{s.exp_id}`)**: Local CV = `{cv_val}` | Official {platform_name} LB = `{lb_val}`{note_str}"
                )
    except Exception:
        pass
    if not submission_feedback_lines:
        submission_feedback_lines.append(f"- No {platform_name} submissions recorded yet.")

    diagnostics_notes.append("")
    diagnostics_notes.append(f"### Competition Leaderboard ({platform_name} LB vs Local CV) Feedback")
    diagnostics_notes.extend(submission_feedback_lines)

    # Normalize template_id / aliases
    tid = template_id.lower().strip()
    if tid in ("antigravity", "antigravity_iterate"):
        tid = "antigravity_iterate"
    elif tid in ("codex", "codex_iterate"):
        tid = "codex_iterate"
    elif tid in ("antigravity_attach", "attach_antigravity"):
        tid = "antigravity_attach"
    elif tid in ("codex_attach", "attach_codex"):
        tid = "codex_attach"
    elif tid == "aide":
        tid = "antigravity_iterate"
    elif tid == "minimal":
        tid = "codex_iterate"

    data_config = task.get("data_config", {})
    split_strategy = data_config.get("split_strategy", "task-declared validation")
    group_col = data_config.get("group_column")
    train_path = data_config.get("train_path", "dataset/")

    if tid == "antigravity_attach":
        prompt_lines = [
            "# SYSTEM DIRECTIVE: ANTIGRAVITY INITIAL TASK SETUP & ATTACH",
            "",
            "> [!IMPORTANT]",
            f"> Benchmark Task: **{task_name}** | Type: **{task_type}** ({task_modality}) | Target Metric: **{primary_metric} ({direction})**",
            "> You are Antigravity, attaching to a fresh workspace or competition repo. Your objective is to ingest the problem, verify data contracts, and establish a rock-solid, leakage-free baseline.",
            "",
            "## 1. REPOSITORY ORIENTATION & CONTRACTS",
            f"- **Dataset Path**: `{train_path}`",
            f"- **Validation Scheme**: `{split_strategy}` on `{group_col}` (Zero data leakage tolerated).",
            "- **Task Specification**: Inspect `configs/task.yaml` and `harness/task_spec.py` for target columns, subtasks, and conditional if-else routing.",
            f"- **Scientific Protocol**: Strictly adhere to `AGENTS.md`. Completed runs under `{artifact_rel}/` are permanent records.",
            "",
            "## 2. ACTIVE TASK METRICS & CONSTRAINTS",
            f"- **Primary Metric**: `{primary_metric}` ({direction})",
            f"- **Target Score Goal**: `{task.get('primary_metric_target', 'not configured')}`",
            "- **Rule Invariants**: Use the task-declared seed and split files; completed experiments are immutable.",
            "",
            "## 3. MANDATORY INITIAL SETUP SEQUENCE",
            "Follow these sequential steps to initialize the competition harness:",
            "1. **Explore Data & Audit Splits**: Inspect declared training inputs, check target distributions, and verify split integrity with `harness/data_strategy.py`.",
            "2. **Declare Baseline Configuration**: Create `configs/exp001_baseline_<name>.yaml` declaring the foundational control model with explicit `grounding:` and `assumption_register:` blocks.",
            "3. **Execute Smoke Test Probe**: Verify execution on Fold 0 with minimal samples:",
            "   ```powershell",
            "   ai-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml --debug --fold 0",
            "   ```",
            f"4. **Run Full Validation**: Execute {validation_label} and verify generated metrics and experiment artifacts:",
            "   ```powershell",
            "   ai-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml",
            "   ```",
            "5. **Register Champion Baseline**: Ensure the result is recorded in `EXPERIMENTS.md` and registered in `TreeJournal`.",
            "",
            "## 4. AGENT ACKNOWLEDGMENT GATE",
            "Before writing code, acknowledge in chat: (1) Target task and metric, (2) Splitting strategy & leakage prevention, (3) Baseline model architecture family.",
        ]
    elif tid == "codex_attach":
        prompt_lines = [
            "# SYSTEM DIRECTIVE: OPENAI CODEX INITIAL PIPELINE SETUP & ATTACH",
            "",
            "> [!IMPORTANT]",
            f"> Target Benchmark: **{task_name}** | Type: **{task_type}** | Primary Target: **{primary_metric} ({direction})**",
            "> You are OpenAI Codex CLI operating in autonomous ML engineer mode. Setup the baseline and verify pipeline integrity.",
            "",
            "## 1. REPOSITORY INVARIANTS & HARD RULES",
            f"1. **Operating Contract**: Strictly obey `AGENTS.md`. Treat `{train_path}` as read-only and never overwrite completed runs under `{artifact_rel}/`.",
            f"2. **Data Strategy**: Use the declared `{split_strategy}` validation strategy" + (f" grouped by `{group_col}`." if group_col else ". Do not invent a grouping column or split rule."),
            f"3. **Submission Contract**: {submission_guidance}",
            "",
            "## 2. INITIALIZATION SEQUENCE (PLAN -> CODE -> SMOKE TEST -> FULL CV)",
            "1. **Inspect Repo**: Read `configs/task.yaml`, `harness/task_spec.py`, and `harness/data_strategy.py`.",
            "2. **Formulate exp001 Baseline**: Create `configs/exp001_baseline_<name>.yaml` with valid `grounding:` and `assumption_register:` blocks.",
            "3. **Run Smoke Test**: Execute `ai-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml --debug --fold 0`.",
            "4. **Run Full Cross-Validation**: Execute `ai-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml`.",
            f"5. **Verify Receipts**: Confirm `{artifact_rel}/exp001_.../` contains the expected metrics and experiment record.",
        ]
    elif tid == "codex_iterate":
        prompt_lines = [
            "# SYSTEM DIRECTIVE: OPENAI CODEX / AUTONOMOUS ML ENGINEER",
            "",
            "> [!IMPORTANT]",
            f"> Repository Task: **{task_name}** | Type: **{task_type}** ({task_modality}) | Target: **{primary_metric} ({direction})**",
            "> You are operating via the Codex CLI in this local codebase. You must follow the operational contract in `AGENTS.md`.",
            "",
            "## 1. REPOSITORY & BENCHMARK STATE",
            f"- **Repository Benchmark Record**: `{best_score}` ({primary_metric})",
            f"- **Current Champion Architecture**: `{best_id}` ({best_type} / {best_family}, {best_params} parameters)",
            f"- **Total Validated Experiments**: {total_runs}",
            f"- **AIDE Tree Search Recommendation**: Explore `{rec_branch}` branch from parent `{rec_parent}`",
            "",
            "## 2. STRICT OPERATIONAL CONSTRAINTS (HARD BOUNDARIES)",
            "1. **Reference Contract**: Read and strictly adhere to `AGENTS.md` before modifying any files.",
            "2. **Single Atomic Intervention**: Exactly ONE independent variable modified per iteration (Control vs Treatment). Never combine multiple speculative changes into a single run.",
            "3. **Ensemble Policy**: Follow the ensemble constraints declared by the task and harness configuration; do not infer undocumented thresholds.",
            f"4. **Submission Contract**: {submission_guidance}",
            "5. **Epistemic Grounding**: Physically cite verified models/papers (Hugging Face / ArXiv) in `configs/exp<NNN>.yaml`. Zero guessing from stale memory.",
            "",
            "## 3. ACTIVE BOTTLENECK & IDEA GRAVEYARD",
            "### Diagnosed Failure Modes",
            *diagnostics_notes,
            "",
            "### The Idea Graveyard (Legacy notes; review evidence before revisiting)",
            *graveyard_items,
            "",
            "## 4. EXECUTION LOOP (PLAN -> CODE -> VERIFY)",
            "1. **Plan & Declare**: Output a 3-bullet plan: (a) parent baseline, (b) single atomic intervention, (c) grounded justification.",
            "2. **Implement**: Create a new config `configs/exp<NNN>_<branch>_<descriptor>.yaml` and necessary `src/` modules.",
            "3. **Validate**: Execute CV validation: `ai-harness-run --project-root . --config configs/exp<NNN>_<branch>_<descriptor>.yaml --debug --fold 0` (smoke test first), then full CV.",
            "4. **Output Format**: Provide a concise summary, git diff, and resulting validation scores.",
        ]
    else:  # antigravity_iterate (default)
        prompt_lines = [
            "# MISSION DIRECTIVE & RUNTIME CONTEXT INJECTION",
            "",
            "> [!IMPORTANT]",
            f"> Task: **{task_name}** ({task_type}) | Primary Metric: **{primary_metric}**",
            "> You must strictly adhere to the empirical solution tree state below. Do NOT assume a blank slate.",
            "",
            "## 1. Current Solution Frontier & Hall of Fame",
            f"- **Total Experiments Completed**: {total_runs}",
            f"- **Reigning Champion**: `{best_id}`",
            f"- **Champion Score**: `{best_score}` ({primary_metric})",
            f"- **Approach Type**: `{best_type}` ({best_family})",
            f"- **Model Parameters**: `{best_params}`",
            "",
            "## 2. AIDE UCB Tree Search Recommendation",
            f"- **Recommended Parent**: `{rec_parent}`",
            f"- **Recommended Branch**: `{rec_branch}`",
            f"- **Search Recommendation**: `{rec.get('recommendation')}`",
            f"- **Scientific Guideline**: {rec.get('guideline')}",
            "",
            "## 3. Diagnosed Empirical Bottlenecks (Target Problems)",
            *diagnostics_notes,
            "",
            "## 4. The Idea Graveyard (Legacy notes; review evidence before revisiting)",
            *graveyard_items,
            "",
            "## 5. Hard Operational Protocols (Physical Circuit Breakers)",
            "1. **Epistemic Grounding Gate**: State evidence and assumptions for each experiment; apply the grounding gate when required by the task configuration.",
            f"2. **Submission Contract**: {submission_guidance}",
            "3. **Atomic Modification**: Exactly ONE controlled variable per experiment (Control vs Treatment).",
            "",
            "## 6. Mandatory Agent Acknowledgment Step (Alignment Gate)",
            "Before proposing code or creating an experiment configuration, you MUST reply with:",
            "1. **Parent Node & Baseline Score**: Confirm the parent node and score you are extending.",
            "2. **Diagnosed Bottleneck**: Name the specific failure mode you are resolving.",
            "3. **Intervention & Grounding Source**: Specify your atomic intervention and its empirical citation.",
            "4. **Explicit Assumptions**: State 1-2 testable assumptions and your uncertainty rating (`low` | `medium` | `high`).",
            "",
            "*Continue authorized experiments within the session budget. Request approval only for governed sensitive actions or missing authority.*",
        ]

    try:
        comp_state = CompetitionStateMachine(PROJECT_ROOT)
        comp_block = comp_state.render_agent_context_block(journal, SubmissionManager(), task)
        if comp_block:
            prompt_lines = [comp_block, "", "---", ""] + prompt_lines
    except Exception as comp_err:
        print(f"⚠️ Warning: Failed to render competition state context: {comp_err}")

    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.research_memory import ResearchMemory
    from harness.task_spec import load_active_task_spec

    project = HarnessProject.discover(PROJECT_ROOT)
    spec = load_active_task_spec(project.task_config_path)
    plugin = load_task_plugin(project.plugin_id, TaskPluginContext(
        project.root, project.task_config_path, spec, project.plugin_id or "default"))
    prompt_lines.extend(["", ResearchMemory(project.root, project.artifacts_dir, spec, plugin).render()])
    prompt_text = "\n".join(prompt_lines)
    return {
        "template_id": template_id,
        "prompt": prompt_text,
        "character_count": len(prompt_text),
    }


