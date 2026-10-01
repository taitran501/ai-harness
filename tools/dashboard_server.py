#!/usr/bin/env python3
"""Task-Agnostic ML/DL Experimentation Dashboard & Antigravity Priming Prompt Server.

Zero-dependency standard library Python HTTP server providing:
1. REST APIs for the AIDE solution tree, experiments, diagnostics, and UCB recommendations.
2. Dynamic Antigravity Priming Prompt generation to eliminate agent amnesia and hallucinated certainty.
3. Clean modern web SPA frontend.
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import ipaddress
import json
import os
import re
import socketserver
import subprocess
import sys
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from harness.tree_journal import Node, TreeJournal
from harness.submission_manager import SubmissionManager
from harness.cloud_inspector import CloudPreflightInspector
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script, push_dataset_to_modal
from harness.environment_detector import EnvironmentDetector
from harness.cloud_job_runner import CloudJobManager, CloudJob
from harness.llm_engine import LLMEngine, LLMConfig
from harness.skill_injector import inject_skill_file
from harness.competition_state import CompetitionStateMachine
from harness.agent_bridge import AgentBridge


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
    tasks_dirs = [PROJECT_ROOT / "configs" / "tasks", HARNESS_DIR / "configs" / "tasks"]
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


def create_and_activate_task(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """A new task must be bootstrapped and attached in its own workspace."""
    return {
        "success": False,
        "error": "Create or bootstrap the task in its own workspace, then run `ml-harness attach`.",
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
    task_file = HARNESS_PROJECT.task_config_path
    task_file.parent.mkdir(parents=True, exist_ok=True)
    content = {"task": task_data}
    with open(task_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


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

    graveyard_items = ["- No rejected approaches are recorded in the current experiment history."]

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
            f"# SYSTEM DIRECTIVE: ANTIGRAVITY INITIAL TASK SETUP & ATTACH",
            "",
            "> [!IMPORTANT]",
            f"> Benchmark Task: **{task_name}** | Type: **{task_type}** ({task_modality}) | Target Metric: **{primary_metric} ({direction})**",
            "> You are Antigravity, attaching to a fresh workspace or competition repo. Your objective is to ingest the problem, verify data contracts, and establish a rock-solid, leakage-free baseline.",
            "",
            "## 1. REPOSITORY ORIENTATION & CONTRACTS",
            f"- **Dataset Path**: `{train_path}`",
            f"- **Validation Scheme**: `{split_strategy}` on `{group_col}` (Zero data leakage tolerated).",
            f"- **Task Specification**: Inspect `configs/task.yaml` and `harness/task_spec.py` for target columns, subtasks, and conditional if-else routing.",
            f"- **Scientific Protocol**: Strictly adhere to `AGENTS.md`. Completed runs under `{artifact_rel}/` are permanent records.",
            "",
            "## 2. ACTIVE TASK METRICS & CONSTRAINTS",
            f"- **Primary Metric**: `{primary_metric}` ({direction})",
            f"- **Target Score Goal**: `{task.get('primary_metric_target', 'not configured')}`",
            f"- **Rule Invariants**: Use the task-declared seed and split files; completed experiments are immutable.",
            "",
            "## 3. MANDATORY INITIAL SETUP SEQUENCE",
            "Follow these sequential steps to initialize the competition harness:",
            "1. **Explore Data & Audit Splits**: Inspect declared training inputs, check target distributions, and verify split integrity with `harness/data_strategy.py`.",
            "2. **Declare Baseline Configuration**: Create `configs/exp001_baseline_<name>.yaml` declaring the foundational control model with explicit `grounding:` and `assumption_register:` blocks.",
            "3. **Execute Smoke Test Probe**: Verify execution on Fold 0 with minimal samples:",
            "   ```powershell",
            "   ml-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml --debug --fold 0",
            "   ```",
            f"4. **Run Full Validation**: Execute {validation_label} and verify generated metrics and experiment artifacts:",
            "   ```powershell",
            "   ml-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml",
            "   ```",
            "5. **Register Champion Baseline**: Ensure the result is recorded in `EXPERIMENTS.md` and registered in `TreeJournal`.",
            "",
            "## 4. AGENT ACKNOWLEDGMENT GATE",
            "Before writing code, acknowledge in chat: (1) Target task and metric, (2) Splitting strategy & leakage prevention, (3) Baseline model architecture family.",
        ]
    elif tid == "codex_attach":
        prompt_lines = [
            f"# SYSTEM DIRECTIVE: OPENAI CODEX INITIAL PIPELINE SETUP & ATTACH",
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
            "3. **Run Smoke Test**: Execute `ml-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml --debug --fold 0`.",
            "4. **Run Full Cross-Validation**: Execute `ml-harness-run --project-root . --config configs/exp001_baseline_<name>.yaml`.",
            f"5. **Verify Receipts**: Confirm `{artifact_rel}/exp001_.../` contains the expected metrics and experiment record.",
        ]
    elif tid == "codex_iterate":
        prompt_lines = [
            f"# SYSTEM DIRECTIVE: OPENAI CODEX / AUTONOMOUS ML ENGINEER",
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
            "### The Idea Graveyard (Permanently Forbidden Interventions)",
            *graveyard_items,
            "",
            "## 4. EXECUTION LOOP (PLAN -> CODE -> VERIFY)",
            "1. **Plan & Declare**: Output a 3-bullet plan: (a) parent baseline, (b) single atomic intervention, (c) grounded justification.",
            "2. **Implement**: Create a new config `configs/exp<NNN>_<branch>_<descriptor>.yaml` and necessary `src/` modules.",
            "3. **Validate**: Execute CV validation: `ml-harness-run --project-root . --config configs/exp<NNN>_<branch>_<descriptor>.yaml --debug --fold 0` (smoke test first), then full CV.",
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
            "## 4. The Idea Graveyard (Permanently Forbidden Interventions)",
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
            "*Wait for user confirmation after completing this acknowledgment before executing training commands.*",
        ]

    try:
        comp_state = CompetitionStateMachine(PROJECT_ROOT)
        comp_block = comp_state.render_agent_context_block(journal, SubmissionManager(), task)
        if comp_block:
            prompt_lines = [comp_block, "", "---", ""] + prompt_lines
    except Exception as comp_err:
        print(f"⚠️ Warning: Failed to render competition state context: {comp_err}")

    prompt_text = "\n".join(prompt_lines)
    return {
        "template_id": template_id,
        "prompt": prompt_text,
        "character_count": len(prompt_text),
    }


def generate_priming_prompt(journal: TreeJournal) -> str:
    """Backward-compatible wrapper for generating Antigravity priming prompt."""
    return generate_prompt_by_template(journal, "antigravity")["prompt"]


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler serving REST API and dashboard frontend."""

    def __init__(self, *args, **kwargs):
        candidates = [
            PROJECT_ROOT / "dashboard" / "dist",
            HARNESS_DIR / "dashboard" / "dist",
            HARNESS_DIR / "tools" / "web",
            PROJECT_ROOT / "tools" / "web",
            HARNESS_DIR / "dashboard",
            PROJECT_ROOT / "dashboard",
        ]
        self.web_dir = next((p for p in candidates if p.exists() and (p / "index.html").exists()), candidates[0])
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    @staticmethod
    def _is_loopback_name(hostname: Optional[str]) -> bool:
        if not hostname:
            return False
        hostname = hostname.rstrip(".").casefold()
        if hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    def _reject_nonlocal_request(self) -> bool:
        """Reject DNS rebinding hosts and cross-origin browser requests before dispatch."""
        host_header = self.headers.get("Host", "")
        try:
            host_parts = urllib.parse.urlsplit("//" + host_header)
            local_host = self._is_loopback_name(host_parts.hostname)
        except ValueError:
            local_host = False
        origin = self.headers.get("Origin")
        valid_origin = True
        if origin:
            try:
                origin_parts = urllib.parse.urlsplit(origin)
                valid_origin = (
                    origin_parts.scheme in {"http", "https"}
                    and self._is_loopback_name(origin_parts.hostname)
                    and origin_parts.netloc.casefold() == host_header.casefold()
                )
            except ValueError:
                valid_origin = False
        if not local_host or not valid_origin:
            self.send_error(403, "Dashboard API is restricted to the local same-origin workspace.")
            return True
        return False

    def do_GET(self) -> None:
        if self._reject_nonlocal_request():
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/tree":
            self.handle_api_tree()
        elif path == "/api/experiments":
            self.handle_api_experiments()
        elif path.startswith("/api/node/"):
            node_id = path.replace("/api/node/", "").strip()
            self.handle_api_node(node_id)
        elif path == "/api/recommend":
            self.handle_api_recommend(parsed_url.query)
        elif path == "/api/priming_prompt":
            self.handle_api_priming_prompt(parsed_url.query)
        elif path == "/api/prompt_templates":
            self.handle_api_prompt_templates()
        elif path == "/api/status":
            self.handle_api_status()
        elif path == "/api/submissions":
            self.handle_api_submissions()
        elif path == "/api/submissions/download":
            params = urllib.parse.parse_qs(parsed_url.query)
            sub_id = params.get("id", [""])[0]
            mgr = SubmissionManager()
            sub = next((s for s in mgr.get_all_submissions() if s.id == sub_id), None)
            rel_file = (
                getattr(sub, "archive_path", None)
                or getattr(sub, "zip_path", None)
                or getattr(sub, "submission_file", None)
                if sub
                else None
            )
            if not rel_file or not (PROJECT_ROOT / rel_file).exists():
                self.send_json({"error": "Submission package file not found on disk."}, status=404)
                return
            fpath = PROJECT_ROOT / rel_file
            data_bytes = fpath.read_bytes()
            self.send_response(200)
            ctype = "application/zip" if fpath.suffix == ".zip" else "text/csv"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{fpath.name}"')
            self.send_header("Content-Length", str(len(data_bytes)))
            self.end_headers()
            self.wfile.write(data_bytes)
        elif path == "/api/task_info":
            self.handle_api_task_info()
        elif path == "/api/tasks":
            self.send_json({"tasks": list_available_tasks()})
        elif path == "/api/task_presets":
            self.send_json({"presets": TASK_PRESETS})
        elif path == "/api/next_exp_id":
            self.handle_api_next_exp_id(parsed_url.query)
        elif path == "/api/environment/status":
            self.send_json(EnvironmentDetector.get_full_status())
        elif path == "/api/environment/kaggle/auth_status":
            self.send_json(EnvironmentDetector.get_kaggle_auth_flow_status())
        elif path == "/api/configs":
            self.handle_api_configs()
        elif path == "/api/strategy_cards":
            self.handle_api_strategy_cards()
        elif path == "/api/config":
            self.handle_api_get_config(parsed_url.query)
        elif path == "/api/cloud/preflight":
            self.handle_api_cloud_preflight(parsed_url.query)
        elif path == "/api/cloud/job":
            self.handle_api_cloud_get_job(parsed_url.query)
        elif path == "/api/cloud/jobs":
            self.send_json({"jobs": CloudJobManager().list_jobs()})
        elif path == "/api/cloud/logs/stream":
            self.handle_api_cloud_logs_stream(parsed_url.query)
        elif path == "/api/sync":
            self.handle_api_sync()
        elif path == "/api/llm/providers":
            self.handle_api_llm_providers()
        elif path == "/api/llm/sample_data":
            self.handle_api_llm_sample_data()
        elif path == "/api/llm/hf_models":
            self.handle_api_llm_hf_models(parsed_url.query)
        elif path == "/api/llm/unsloth_models":
            self.handle_api_llm_unsloth_models(parsed_url.query)
        elif path == "/api/llm/provider_models":
            self.handle_api_llm_provider_models(parsed_url.query)
        elif path == "/api/competition-state":
            comp = CompetitionStateMachine(PROJECT_ROOT)
            snap = comp.evaluate_journey_snapshot(TreeJournal(auto_sync=True), SubmissionManager(), load_task_config())
            self.send_json(snap)
        elif path == "/api/agent-bridge/context":
            params = urllib.parse.parse_qs(parsed_url.query)
            sid = params.get("session_id", [None])[0]
            bridge = AgentBridge(PROJECT_ROOT)
            pkt = bridge.get_context_packet(session_id=sid)
            pkt["markdown"] = bridge.render_context_markdown(session_id=sid)
            self.send_json(pkt)
        elif path == "/api/agent-bridge/sessions":
            bridge = AgentBridge(PROJECT_ROOT)
            self.send_json({"sessions": bridge.list_sessions()})
        elif path == "/api/phases":
            from harness.phase_workflow import PhaseWorkflow
            self.send_json(PhaseWorkflow(PROJECT_ROOT, HARNESS_PROJECT.task_config_path).phase_status())
        elif path == "/api/phases/render":
            from harness.phase_workflow import PhaseWorkflow
            params = urllib.parse.parse_qs(parsed_url.query)
            phase_id = params.get("phase", [None])[0]
            try:
                self.send_json({"phase_id": phase_id, "prompt": PhaseWorkflow(PROJECT_ROOT, HARNESS_PROJECT.task_config_path).render(phase_id)})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/eda/latest":
            eda_dir = ARTIFACTS_DIR / "eda"
            reports = sorted(eda_dir.glob("eda_*.json"), reverse=True) if eda_dir.exists() else []
            if reports:
                try:
                    self.send_json(json.loads(reports[0].read_text(encoding="utf-8")))
                except Exception as exc:
                    self.send_json({"error": f"Could not read EDA report: {exc}"}, status=500)
            else:
                self.send_json({"report": None})
        elif path == "/api/dataset-sources":
            try:
                from harness.dataset_sources import list_dataset_sources
                self.send_json({"sources": list_dataset_sources(HARNESS_PROJECT)})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif path == "/api/datasets/local":
            try:
                from harness.dataset_sources import discover_local_datasets
                self.send_json(discover_local_datasets(HARNESS_PROJECT))
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif path == "/api/datasets/research-prompt":
            try:
                from harness.dataset_sources import render_dataset_research_prompt
                self.send_json({"prompt": render_dataset_research_prompt(HARNESS_PROJECT)})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif path == "/api/cloud/status":
            from harness.cloud_policy import UploadConsentStore, provider_preflight
            self.send_json({
                "providers": {name: provider_preflight(HARNESS_PROJECT, name) for name in ("kaggle", "modal")},
                "upload_consent": UploadConsentStore(HARNESS_PROJECT).status(),
            })
        else:
            # Fallback to serving static frontend
            super().do_GET()

    def do_OPTIONS(self) -> None:
        if self._reject_nonlocal_request():
            return
        self.send_response(200)
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self._reject_nonlocal_request():
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/agent-bridge/"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                bridge = AgentBridge(PROJECT_ROOT)
                sub_action = path.replace("/api/agent-bridge/", "").strip()

                if sub_action == "start-session":
                    sess = bridge.start_session(
                        goal=data.get("goal", "Iterative competition optimization via Antigravity IDE"),
                        budget=data.get("budget"),
                        session_id=data.get("session_id"),
                    )
                    self.send_json({"success": True, "session": asdict(sess)})
                elif sub_action == "act":
                    sess, obs = bridge.execute_action(
                        action=data.get("action", "inspect_diagnostics"),
                        params=data.get("params", {}),
                        rationale=data.get("rationale", ""),
                        session_id=data.get("session_id"),
                        goal=data.get("goal"),
                        budget=data.get("budget"),
                        user_approved=bool(data.get("user_approved", False)),
                    )
                    self.send_json({
                        "success": obs.status == "SUCCESS",
                        "session": asdict(sess),
                        "observation": asdict(obs),
                    })
                elif sub_action == "approve":
                    sid = data.get("session_id")
                    if not sid:
                        sessions = bridge.list_sessions(limit=1)
                        sid = sessions[0]["session_id"] if sessions else None
                    if not sid:
                        self.send_json({"error": "No active session to approve."}, status=400)
                        return
                    sess, obs = bridge.approve_pending_action(sid)
                    self.send_json({
                        "success": True,
                        "session": asdict(sess),
                        "observation": asdict(obs) if obs else None,
                    })
                else:
                    self.send_json({"error": f"Unknown agent-bridge action: {sub_action}"}, status=404)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
            return
        elif path.startswith("/api/competition-state/"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                comp = CompetitionStateMachine(PROJECT_ROOT)
                action = path.replace("/api/competition-state/", "").strip()

                if action == "gate":
                    for gk, gv in data.get("gates", {}).items():
                        if gk in {"baseline_locked", "baseline_evidence", "freeze_locked", "finalized"} and gv:
                            self.send_json({"error": f"Gate `{gk}` can only be changed by its verified harness action."}, status=409)
                            return
                        comp.state["gates"][gk] = gv
                    if "mode" in data:
                        comp.state["mode"] = data["mode"]
                    if "manual_phase_override" in data:
                        comp.state["manual_phase_override"] = data["manual_phase_override"]
                    if "toggle_pin_candidate" in data:
                        cid = data["toggle_pin_candidate"]
                        pinned = comp.state["candidate_pool"].get("pinned_candidates", [])
                        if cid in pinned:
                            pinned.remove(cid)
                        else:
                            pinned.append(cid)
                        comp.state["candidate_pool"]["pinned_candidates"] = pinned
                    comp.save()
                elif action == "data-audit":
                    comp.run_data_readiness_audit(data.get("config"))
                elif action == "agent-context":
                    if "config" in data:
                        comp.state["agent_context_config"].update(data["config"])
                        comp.save()
                elif action == "synthesize-candidate":
                    comp.synthesize_candidate(
                        name=data.get("name", "specialist_router_v1"),
                        mode=data.get("mode", "specialist_router"),
                        source_map=data.get("source_map", {}),
                    )
                elif action == "freeze":
                    comp.freeze_final_candidates(
                        slot_1_id=data.get("slot_1_safe_cv", "baseline"),
                        slot_2_id=data.get("slot_2_aggressive_lb"),
                        notes=data.get("notes", ""),
                    )
                elif action == "unfreeze":
                    comp.emergency_unfreeze(reason=data.get("reason", "Manual unfreeze via UI"))
                elif action == "finalize":
                    comp.finalize_competition(data, TreeJournal(auto_sync=True), SubmissionManager())
                elif action == "agent-run":
                    self.send_json({"error": "The dashboard does not run an internal agent. Use the attached IDE agent with `ml-harness phase render` and the Agent Bridge."}, status=409)
                    return
                elif action == "agent-approve":
                    self.send_json({"error": "Use the IDE agent's Agent Bridge session approval endpoint."}, status=409)
                    return

                snap = comp.evaluate_journey_snapshot(TreeJournal(auto_sync=True), SubmissionManager(), load_task_config())
                self.send_json({"success": True, "state": snap})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
            return
        elif path == "/api/submissions/record-score":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                sub_id = data.get("sub_id")
                official_scores = data.get("official_scores", {})
                notes = data.get("notes", "")

                mgr = SubmissionManager()
                if sub_id not in mgr.submissions:
                    self.send_json({"error": f"Submission '{sub_id}' not found."}, status=404)
                    return

                split_type = str(official_scores.get("split_type", "public"))
                primary_metric = str(load_task_config().get("primary_metric") or "score")
                primary_score = official_scores.get(
                    "primary_metric",
                    official_scores.get("score", official_scores.get(primary_metric)),
                )
                custom_scores = {
                    key: value
                    for key, value in official_scores.items()
                    if key not in ("split_type", "primary_metric", "score", primary_metric)
                }

                mgr.record_official_score(
                    sub_id=sub_id,
                    primary_score=primary_score,
                    split_type=split_type,
                    notes=notes,
                    custom_scores=custom_scores,
                )
                self.send_json({"success": True, "submission": asdict(mgr.submissions[sub_id])})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/task_info":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                active_task = load_task_config()
                for identity_key in ("id", "name", "task_name", "plugin_id", "plugin"):
                    if identity_key in data and data.get(identity_key) != active_task.get(identity_key):
                        self.send_json({"error": "Each task needs its own workspace. Attach another workspace to change task identity or plugin."}, status=409)
                        return
                save_task_config(data)
                # Auto-inject updated agent skill for this task
                try:
                    inject_skill_file(task_config=data)
                except Exception as skill_err:
                    print(f"⚠️ Warning: Auto-injecting skill failed: {skill_err}")
                self.send_json({"success": True, "task": load_task_config(), "skill_injected": True})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/tasks/switch":
            self.send_json({"error": "Each task needs its own workspace. Run `ml-harness attach` from the task workspace you want to use."}, status=409)
        elif path == "/api/tasks/create":
            self.send_json({"error": "Create or bootstrap the task in its own workspace, then run `ml-harness attach`."}, status=409)
        elif path == "/api/environment/modal/activate":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_modal_activate(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/modal/token":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_modal_token(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/modal/launch_auth":
            try:
                res = EnvironmentDetector.launch_modal_auth()
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/kaggle/launch_auth":
            try:
                res = EnvironmentDetector.launch_kaggle_auth()
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/run":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_run(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/consent":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                provider = data.get("provider")
                if provider not in {"kaggle", "modal"} or data.get("confirm_upload") is not True:
                    self.send_json({"error": "Choose Kaggle or Modal and explicitly confirm this task dataset upload."}, status=400)
                    return
                from harness.cloud_policy import UploadConsentStore, dataset_fingerprint
                record = UploadConsentStore(HARNESS_PROJECT).grant(provider, dataset_fingerprint(HARNESS_PROJECT))
                self.send_json({"success": True, "consent": record})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/phases/record":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                from harness.phase_workflow import PhaseWorkflow
                record = PhaseWorkflow(PROJECT_ROOT, HARNESS_PROJECT.task_config_path).record(
                    str(data.get("phase_id") or ""),
                    str(data.get("response") or ""),
                    sources=data.get("sources", []),
                    artifacts=data.get("artifacts", []),
                    metadata=data.get("metadata", {}),
                )
                self.send_json({"success": True, "record": record})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/eda/run":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                from harness.eda import run_eda
                report = run_eda(PROJECT_ROOT, data_path=data.get("data_path"))
                self.send_json({"success": True, "report": report})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/datasets/import-kaggle":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
                data = json.loads(body or "{}")
                from harness.dataset_sources import import_kaggle_dataset
                result = import_kaggle_dataset(HARNESS_PROJECT, data.get("slug", ""))
                self.send_json({"success": True, **result})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
        elif path == "/api/cloud/stop":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_stop(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/audit":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_audit(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/skill/inject":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                task_cfg = data if data else load_task_config()
                p = inject_skill_file(task_config=task_cfg)
                self.send_json({"success": True, "skill_path": str(p.relative_to(PROJECT_ROOT))})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/submissions/register":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                exp_id = data.get("exp_id")
                if not exp_id:
                    self.send_json({"error": "Field 'exp_id' is required."}, status=400)
                    return

                candidate_csv = data.get("candidate_csv")
                baseline_csv = data.get("baseline_csv")
                notes = data.get("notes", "")
                min_diff_pct = float(data.get("min_diff_pct", 2.0))
                ensemble_components = data.get("ensemble_components")

                mgr = SubmissionManager()
                if HARNESS_PROJECT.plugin_id not in (None, "default"):
                    if not candidate_csv:
                        raise ValueError(
                            "Provide test-set predictions. The dashboard will not package OOF validation predictions as a submission."
                        )
                    from harness.plugins import TaskPluginContext, load_task_plugin
                    from harness.task_spec import load_active_task_spec

                    task_spec = load_active_task_spec(HARNESS_PROJECT.task_config_path)
                    plugin_context = TaskPluginContext(
                        project_root=PROJECT_ROOT,
                        task_config_path=HARNESS_PROJECT.task_config_path,
                        task_spec=task_spec,
                        plugin_id=HARNESS_PROJECT.plugin_id,
                    )
                    plugin = load_task_plugin(HARNESS_PROJECT.plugin_id, plugin_context)
                    capabilities = plugin.capabilities()
                    if not capabilities.get("submission_builder") or not capabilities.get("submission_validator"):
                        raise ValueError(
                            f"Task plugin '{HARNESS_PROJECT.plugin_id}' must implement submission_builder and submission_validator."
                        )
                    submission_spec = load_task_config().get("submission_spec", {})
                    entry = mgr.register_plugin_submission(
                        exp_id=exp_id,
                        candidate_csv=candidate_csv,
                        plugin=plugin,
                        plugin_context=plugin_context,
                        file_name=submission_spec.get("file_name", "submission.csv"),
                        notes=notes,
                    )
                else:
                    if not candidate_csv:
                        default_cand = ARTIFACTS_DIR / exp_id / "oof_predictions.csv"
                        if default_cand.exists():
                            candidate_csv = str(default_cand)
                        else:
                            self.send_json({
                                "error": f"Candidate predictions file not found at '{default_cand}'. Please provide valid candidate_csv path."
                            }, status=400)
                            return
                    entry = mgr.register_submission(
                        exp_id=exp_id,
                        candidate_csv=candidate_csv,
                        baseline_csv=baseline_csv,
                        ensemble_components=ensemble_components,
                        notes=notes,
                        min_diff_pct=min_diff_pct,
                    )
                self.send_json({"success": True, "submission": asdict(entry)})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/generate":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_generate(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/push-dataset":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_push_dataset(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/kaggle-status":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_kaggle_status(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/cloud/pull-output":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_cloud_pull_output(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/config":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_save_config(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/config/create":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_create_config(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/llm/test_prompt":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_test_prompt(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/llm/generate_peft_script":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_generate_peft(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/llm/generate_unsloth_script":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_generate_unsloth(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/llm/key":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_save_key(data)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/environment/llm/verify":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                self.handle_api_llm_verify_key(data)
            except Exception as e:
                self.send_json({"error": str(e), "success": False, "valid": False}, status=400)
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_api_tree(self) -> None:
        journal = TreeJournal()
        graph_data = journal.get_graph_data()
        self.send_json(graph_data)

    def handle_api_experiments(self) -> None:
        journal = TreeJournal()
        experiments = []
        best = journal.get_best_node()
        best_id = best.id if best else None

        for n in journal.nodes.values():
            experiments.append({
                "id": n.id,
                "branch": n.branch,
                "parent_id": n.parent_id or "ROOT",
                "approach_type": getattr(n, "approach_type", "ml"),
                "model_family": getattr(n, "model_family", "linear"),
                "score": round(n.get_score(), 4),
                "param_count": n.param_count,
                "param_size": getattr(n, "param_size", None),
                "llm_spec": getattr(n, "llm_spec", None),
                "status": n.status,
                "is_best": (n.id == best_id),
                "is_buggy": n.is_buggy,
                "hypothesis": n.atomic_hypothesis,
                "intervention": n.intervention,
                "timestamp": n.timestamp,
            })
        experiments.sort(key=lambda x: x["timestamp"], reverse=True)
        self.send_json({"total": len(experiments), "experiments": experiments})

    def handle_api_node(self, node_id: str) -> None:
        journal = TreeJournal()
        node = journal.get_node(node_id)
        if not node:
            self.send_json({"error": f"Node '{node_id}' not found."}, status=404)
            return

        # Attempt to load grounding receipt if present
        grounding_data = None
        exp_dir = ARTIFACTS_DIR / node_id
        receipt_file = exp_dir / "GROUNDING_RECEIPT.json"
        if receipt_file.exists():
            try:
                with open(receipt_file, "r", encoding="utf-8") as f:
                    grounding_data = json.load(f)
            except Exception:
                pass

        # Attempt to load target experiment config
        config_data = None
        cand_files = [
            PROJECT_ROOT / "configs" / f"{node.id}.yaml",
            ARTIFACTS_DIR / node.id / "run_config.yaml",
        ]
        configs_dir = PROJECT_ROOT / "configs"
        if configs_dir.exists():
            prefix = node.id.split("_")[0]
            for cfg in configs_dir.glob(f"{prefix}*.yaml"):
                if cfg not in cand_files:
                    cand_files.append(cfg)

        for cand in cand_files:
            if cand.exists():
                try:
                    raw_txt = cand.read_text(encoding="utf-8")
                    config_data = {
                        "path": str(cand.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "content": raw_txt,
                    }
                    break
                except Exception:
                    pass

        # Determine llm_spec and param_size
        llm_spec = getattr(node, "llm_spec", None)
        param_size = getattr(node, "param_size", None)

        if config_data and config_data.get("content"):
            try:
                parsed_cfg = yaml.safe_load(config_data["content"]) or {}
                if not llm_spec and parsed_cfg.get("llm"):
                    llm_spec = parsed_cfg.get("llm")
                if not param_size and llm_spec and isinstance(llm_spec, dict):
                    param_size = llm_spec.get("param_size")
                if not param_size and parsed_cfg.get("model", {}).get("param_size"):
                    param_size = parsed_cfg["model"]["param_size"]
            except Exception:
                pass

        # If node represents an LLM, ensure a sensible spec structure
        node_approach = getattr(node, "approach_type", "ml")
        lower_id = node.id.lower()
        if (node_approach == "llm" or any(k in lower_id for k in ["llm", "qwen", "deepseek", "llama", "unsloth", "cot", "qlora"])) and not llm_spec:
            model_name = node.metrics.get("model") or node.id
            if "deepseek" in lower_id:
                llm_spec = {
                    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    "param_size": "14B",
                    "quantization": "4-bit AWQ / NF4",
                    "paradigm": "zero_shot_cot",
                    "context_length": 8192,
                    "provider": "huggingface",
                    "use_cot": True,
                }
                param_size = param_size or "14B"
            elif "qwen" in lower_id or "unsloth" in lower_id:
                llm_spec = {
                    "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                    "param_size": "7B",
                    "quantization": "4-bit NormalFloat (NF4)",
                    "paradigm": "unsloth_qlora",
                    "context_length": 8192,
                    "provider": "unsloth",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    "use_cot": True,
                }
                param_size = param_size or "7B"
            else:
                llm_spec = {
                    "model_name": model_name,
                    "param_size": param_size or "7B",
                    "quantization": "4-bit NF4",
                    "paradigm": "unsloth_qlora",
                    "context_length": 4096,
                    "provider": "unsloth",
                }

        details = {
            "id": node.id,
            "parent_id": node.parent_id,
            "branch": node.branch,
            "approach_type": getattr(node, "approach_type", "ml"),
            "model_family": getattr(node, "model_family", "linear"),
            "score": round(node.get_score(), 4),
            "param_count": node.param_count,
            "param_size": param_size,
            "llm_spec": llm_spec,
            "status": node.status,
            "is_buggy": node.is_buggy,
            "atomic_hypothesis": node.atomic_hypothesis,
            "intervention": node.intervention,
            "notes": node.notes,
            "timestamp": node.timestamp,
            "metrics": node.metrics,
            "diagnostics_summary": node.diagnostics_summary,
            "grounding_receipt": grounding_data,
            "config": config_data,
        }
        self.send_json(details)

    def handle_api_recommend(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        max_params = None
        if "max_params" in params:
            try:
                max_params = int(params["max_params"][0])
            except ValueError:
                pass

        journal = TreeJournal()
        rec = journal.recommend_next_branch(max_params=max_params)
        self.send_json(rec)

    def handle_api_prompt_templates(self) -> None:
        self.send_json({
            "templates": [
                {
                    "id": "antigravity_attach",
                    "ide": "antigravity",
                    "intent": "attach",
                    "name": "Antigravity — Setup & Attach",
                    "badge": "Setup",
                    "description": "Initial session prompt: Ingest task specification, verify split integrity, and establish foundational baseline.",
                },
                {
                    "id": "antigravity_iterate",
                    "ide": "antigravity",
                    "intent": "iterate",
                    "name": "Antigravity — Continue Iteration",
                    "badge": "New Chat",
                    "description": "Context injection for active iterations: Reigning champion, UCB tree recommendation, diagnosed bottlenecks, and 4-step Alignment Gate.",
                },
                {
                    "id": "codex_attach",
                    "ide": "codex",
                    "intent": "attach",
                    "name": "OpenAI Codex — Setup & Attach",
                    "badge": "CLI Setup",
                    "description": "Terminal instructions for Codex CLI: Inspect repository rules, verify canonical data splits, and execute baseline smoke test.",
                },
                {
                    "id": "codex_iterate",
                    "ide": "codex",
                    "intent": "iterate",
                    "name": "OpenAI Codex — Continue Iteration",
                    "badge": "CLI Iteration",
                    "description": "Autonomous ML engineering loop: Single atomic changes, task-configured validation, and Git diff reporting.",
                },
                # Backward-compatibility aliases
                {
                    "id": "antigravity",
                    "ide": "antigravity",
                    "intent": "iterate",
                    "name": "Antigravity Agent (Default)",
                    "badge": "Active",
                    "description": "Alias for Antigravity Continue Iteration prompt.",
                },
                {
                    "id": "codex",
                    "ide": "codex",
                    "intent": "iterate",
                    "name": "OpenAI Codex CLI (Default)",
                    "badge": "Active",
                    "description": "Alias for Codex Continue Iteration prompt.",
                },
                {
                    "id": "aide",
                    "ide": "antigravity",
                    "intent": "iterate",
                    "name": "AIDE ML Solution Tree (Legacy)",
                    "badge": "Legacy",
                    "description": "Legacy alias routed to Antigravity iteration prompt.",
                },
                {
                    "id": "minimal",
                    "ide": "codex",
                    "intent": "iterate",
                    "name": "Minimal Direct Patch (Legacy)",
                    "badge": "Legacy",
                    "description": "Legacy alias routed to Codex iteration prompt.",
                },
            ]
        })

    def handle_api_priming_prompt(self, query: str = "") -> None:
        params = urllib.parse.parse_qs(query)
        ide = params.get("ide", [None])[0]
        intent = params.get("intent", [None])[0]

        if ide and intent:
            template_id = f"{ide.strip().lower()}_{intent.strip().lower()}"
        else:
            template_id = params.get("template", ["antigravity"])[0]

        journal = TreeJournal()
        res = generate_prompt_by_template(journal, template_id=template_id)
        self.send_json(res)

    def handle_api_status(self) -> None:
        journal = TreeJournal()
        best = journal.get_best_node()
        self.send_json({
            "status": "online",
            "total_nodes": len(journal.nodes),
            "best_node": best.id if best else None,
            "best_score": round(best.get_score(), 4) if best else 0.0,
            "distinct_families": journal.count_distinct_model_families(),
        })

    def handle_api_submissions(self) -> None:
        mgr = SubmissionManager()
        data = [asdict(s) for s in mgr.submissions.values()]
        data.sort(key=lambda x: x.get("registered_at", 0), reverse=True)
        self.send_json({"total": len(data), "submissions": data})

    def handle_api_task_info(self) -> None:
        task = load_task_config()
        artifact_path = ARTIFACTS_DIR.relative_to(PROJECT_ROOT).as_posix()
        plugin_id = HARNESS_PROJECT.plugin_id or "default"
        plugin_info: Dict[str, Any] = {"id": plugin_id, "version": None, "capabilities": {}}
        try:
            from harness.plugins import TaskPluginContext, load_task_plugin
            from harness.task_spec import load_active_task_spec

            task_spec = load_active_task_spec(HARNESS_PROJECT.task_config_path)
            context = TaskPluginContext(
                project_root=PROJECT_ROOT,
                task_config_path=HARNESS_PROJECT.task_config_path,
                task_spec=task_spec,
                plugin_id=plugin_id,
            )
            plugin = load_task_plugin(HARNESS_PROJECT.plugin_id, context)
            plugin_info.update(version=plugin.version, capabilities=plugin.capabilities())
        except Exception as exc:
            plugin_info["error"] = str(exc)
        self.send_json({
            "platform": task.get("platform", "codabench"),
            "task_name": task.get("task_name") or task.get("name", "General ML/DL Benchmark"),
            "name": task.get("name", "General ML/DL Benchmark"),
            "task_type": task.get("task_type") or task.get("type", "Predictive Modeling"),
            "type": task.get("type", "Predictive Modeling"),
            "modality": task.get("modality", "generic"),
            "primary_metric": task.get("primary_metric", "Score"),
            "direction": task.get("direction", "maximize"),
            "primary_metric_baseline": task.get("primary_metric_baseline"),
            "primary_metric_target": task.get("primary_metric_target"),
            "inputs": task.get("inputs", []),
            "subtasks": task.get("subtasks", task.get("metrics", [])),
            "metrics": task.get("metrics", []),
            "languages": task.get("languages", []),
            "description": task.get("description", ""),
            "data_config": task.get("data_config", {}),
            "competition_rules": task.get("competition_rules", {}),
            "submission_spec": task.get("submission_spec", {}),
            "artifacts_dir": artifact_path,
            "task_plugin": plugin_info,
        })

    def handle_api_configs(self) -> None:
        configs_dir = PROJECT_ROOT / "configs"
        cfg_files = []
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                if f.name != "task.yaml":
                    cfg_files.append(f"configs/{f.name}")
        self.send_json({"configs": cfg_files})

    def handle_api_strategy_cards(self) -> None:
        cards_file = PROJECT_ROOT / "configs" / "strategy_cards.yaml"
        if cards_file.exists():
            try:
                raw_cards = yaml.safe_load(cards_file.read_text(encoding="utf-8")) or {}
                if isinstance(raw_cards, list):
                    return self.send_json({"strategies": raw_cards})
                elif isinstance(raw_cards, dict) and "strategies" in raw_cards:
                    return self.send_json({"strategies": raw_cards["strategies"]})
            except Exception:
                pass
        self.send_json({"strategies": DEFAULT_STRATEGY_CARDS})

    def handle_api_get_config(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        cfg_path = params.get("path", [""])[0].strip()
        exp_id = params.get("exp_id", [""])[0].strip()

        target_file = None
        if cfg_path:
            norm_path = (PROJECT_ROOT / cfg_path).resolve()
            if (PROJECT_ROOT in norm_path.parents or norm_path == PROJECT_ROOT) and norm_path.exists():
                target_file = norm_path
        elif exp_id:
            for cand in [
                PROJECT_ROOT / "configs" / f"{exp_id}.yaml",
                ARTIFACTS_DIR / exp_id / "run_config.yaml",
            ]:
                if cand.exists():
                    target_file = cand
                    break
            if not target_file:
                prefix = exp_id.split("_")[0]
                configs_dir = PROJECT_ROOT / "configs"
                if configs_dir.exists():
                    for cand in configs_dir.glob(f"{prefix}*.yaml"):
                        target_file = cand
                        break

        if not target_file or not target_file.exists():
            self.send_json({"error": f"Config not found for path='{cfg_path}' exp_id='{exp_id}'"}, status=404)
            return

        try:
            content = target_file.read_text(encoding="utf-8")
            parsed = yaml.safe_load(content) or {}
            self.send_json({
                "path": str(target_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "content": content,
                "parsed": parsed,
            })
        except Exception as e:
            self.send_json({"error": f"Failed to read/parse config: {e}"}, status=400)

    def handle_api_save_config(self, data: Dict[str, Any]) -> None:
        raw_path = data.get("path", "").strip()
        content = data.get("content", "")

        if not raw_path:
            self.send_json({"error": "Field 'path' is required."}, status=400)
            return

        if not (raw_path.endswith(".yaml") or raw_path.endswith(".yml")):
            self.send_json({"error": "Config path must end with .yaml or .yml."}, status=400)
            return

        dest = (PROJECT_ROOT / raw_path).resolve()
        if PROJECT_ROOT not in dest.parents:
            self.send_json({"error": "Path traversal prohibited."}, status=403)
            return

        try:
            parsed = yaml.safe_load(content)
            if not isinstance(parsed, dict):
                raise ValueError("YAML content must be an object/mapping.")
        except Exception as yaml_err:
            self.send_json({"error": f"Invalid YAML syntax: {yaml_err}"}, status=400)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

        self.send_json({
            "success": True,
            "path": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "parsed": parsed,
        })

    def handle_api_create_config(self, data: Dict[str, Any]) -> None:
        exp_id = data.get("exp_id", "").strip()
        parent_id = data.get("parent_id", "exp000_template").strip()
        branch = data.get("branch", "feat").strip()
        hypothesis = data.get("hypothesis", "").strip()
        intervention = data.get("intervention", "").strip()
        problem = data.get("problem_addressed", "").strip()
        approach_type = data.get("approach_type", "dl" if branch in ("model", "loss") else "ml")
        model_family = data.get("model_family", "transformer" if approach_type == "dl" else "linear")

        if not exp_id:
            desc = data.get("descriptor") or data.get("branch") or "target"
            next_info = get_next_experiment_id(branch, desc)
            exp_id = next_info["next_id"]

        if not exp_id.startswith("exp"):
            exp_id = f"exp_{exp_id}"

        filename = f"{exp_id}.yaml" if not exp_id.endswith(".yaml") else exp_id
        dest = PROJECT_ROOT / "configs" / filename

        task_cfg = load_task_config()
        new_cfg = {
            "experiment": {
                "id": exp_id.replace(".yaml", ""),
                "branch": branch,
                "parent_id": parent_id,
                "date": datetime.date.today().isoformat(),
                "problem_addressed": problem or f"Optimization targeting branch '{branch}'.",
                "hypothesis": hypothesis or f"Hypothesis for {exp_id} on {branch} branch.",
                "intervention": intervention or f"Controlled intervention for {exp_id}.",
            },
            "task": {
                "name": task_cfg.get("name", "Competition Task"),
                "type": task_cfg.get("type", "Standard Classification"),
                "primary_metric": task_cfg.get("primary_metric", "Score"),
                "direction": task_cfg.get("direction", "maximize"),
                "approach_type": approach_type,
                "model_family": model_family,
            },
            "grounding": {
                "source_type": data.get("source_type", "huggingface_hub" if approach_type == "dl" else ("api_provider" if approach_type == "llm" else "verified_baseline")),
                "source_id": data.get("source_id", "gemini-2.0-flash" if approach_type == "llm" else ("microsoft/mdeberta-v3-base" if approach_type == "dl" else "sklearn_tfidf_canonical")),
                "retrieval_summary": data.get("retrieval_summary", f"Model verified for {task_cfg.get('name', 'task')} requirements."),
                "verified_via": data.get("verified_via", "local_registry"),
            },
            "assumption_register": {
                "known_facts": [
                    f"Baseline established under parent {parent_id}.",
                ],
                "assumptions": [
                    hypothesis or f"Branch {branch} yields measurable metric improvement.",
                ],
                "uncertainty_level": data.get("uncertainty_level", "low"),
            },
            "model": {
                "name": data.get("model_name", "gemini-2.0-flash" if approach_type == "llm" else ("mdeberta_v3" if approach_type == "dl" else "logistic_regression")),
                "provider": data.get("provider", "gemini") if approach_type == "llm" else None,
                "params": data.get("model_params", {
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "use_cot": False,
                } if approach_type == "llm" else {
                    "seed": 42,
                    "learning_rate": 2e-5 if approach_type == "dl" else 1.0,
                    "batch_size": 16 if approach_type == "dl" else 64,
                    "epochs": 3 if approach_type == "dl" else 1,
                }),
            },
            "data": {
                "train_path": task_cfg.get("data_config", {}).get("train_path", "dataset/"),
                "test_path": task_cfg.get("data_config", {}).get("test_path", "dataset/test/"),
                "folds": [0, 1, 2, 3, 4],
            },
            "training": {
                "seed": 42,
                "folds": [0, 1, 2, 3, 4],
                "debug": False,
            },
        }

        if approach_type == "llm":
            m_params = data.get("model_params", {})
            param_size = data.get("param_size") or m_params.get("param_size", "7B")
            new_cfg["llm"] = {
                "model_name": data.get("model_name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
                "param_size": param_size,
                "quantization": m_params.get("quantization", "4-bit NormalFloat (NF4)"),
                "paradigm": m_params.get("paradigm", "unsloth_qlora"),
                "context_length": m_params.get("context_length", 8192),
                "provider": m_params.get("provider", "unsloth"),
                "lora_r": m_params.get("lora_r", 16),
                "lora_alpha": m_params.get("lora_alpha", 32),
                "lora_dropout": m_params.get("lora_dropout", 0.05),
                "target_modules": m_params.get("target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
                "use_cot": m_params.get("use_cot", True),
                "system_prompt": "You are an expert classification model specializing in nuance, semantic context, and robust reasoning.",
                "prompt_template": "Classify the following input:\n\n{text}",
            }

        yaml_content = yaml.dump(new_cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml_content, encoding="utf-8")

        self.send_json({
            "success": True,
            "path": f"configs/{dest.name}",
            "content": yaml_content,
            "parsed": new_cfg,
        })

    def handle_api_sync(self) -> None:
        """Discover completed experiment artifacts and persist the solution journal."""
        try:
            journal = TreeJournal(
                journal_path=ARTIFACTS_DIR / "journal.json",
                artifacts_dir=ARTIFACTS_DIR,
            )
            synced_count = journal.sync_from_artifacts(ARTIFACTS_DIR)
            self.send_json({
                "success": True,
                "synced_count": synced_count,
                "total_nodes": len(journal.nodes),
            })
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_api_cloud_preflight(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        backend = params.get("provider", params.get("backend", ["kaggle"]))[0]
        config_path = params.get("config", [None])[0]
        result: Dict[str, Any] = {}
        try:
            from harness.cloud_policy import provider_preflight
            result["provider"] = provider_preflight(HARNESS_PROJECT, backend)
            if config_path:
                full_path = Path(config_path).expanduser()
                if not full_path.is_absolute():
                    full_path = PROJECT_ROOT / full_path
                passed, errors, resolved_cfg = CloudPreflightInspector.run_full_preflight_check(
                    full_path, project_root=PROJECT_ROOT
                )
                result["experiment"] = {"passed": passed, "errors": errors, "resolved_config": resolved_cfg}
                result["errors"] = errors
                result["resolved_config"] = resolved_cfg
                result["passed"] = bool(passed and result["provider"].get("ready"))
            else:
                result["passed"] = bool(result["provider"].get("ready"))
            self.send_json(result)
        except Exception as exc:
            self.send_json({"passed": False, "error": str(exc)}, status=400)

    def handle_api_cloud_generate(self, data: Dict[str, Any]) -> None:
        backend = data.get("backend") or data.get("platform") or "modal"
        config_path = data.get("config_path")
        if not config_path:
            self.send_json({"error": "config_path is required."}, status=400)
            return
        gpu = data.get("gpu", "A10" if backend == "modal" else "T4")
        from harness.cloud_policy import dataset_fingerprint, private_upload_resource_names
        fingerprint = dataset_fingerprint(HARNESS_PROJECT)
        username = KaggleKernelManager.get_authenticated_username()
        resources = private_upload_resource_names(
            HARNESS_PROJECT, fingerprint_data=fingerprint, kaggle_username=username
        )
        task_id = HARNESS_PROJECT.manifest.get("task_id") or resources["task_slug"]
        volume_name = resources["modal_volume_name"]
        task_doc = yaml.safe_load(HARNESS_PROJECT.task_config_path.read_text(encoding="utf-8")) or {}
        task_data = task_doc.get("task", task_doc)
        data_config = task_data.get("data_config") or task_data.get("data") or {}
        dataset_slug = data_config.get("kaggle_dataset_slug") or resources["kaggle_dataset_slug"]
        if backend == "kaggle" and not dataset_slug:
            self.send_json({"error": "Could not determine the active Kaggle username. Connect the account through the dashboard or run `kaggle auth login --force`, then retry."}, status=409)
            return

        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file
        exp_doc = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        exp_id = exp_doc.get("experiment", {}).get("id", config_file.stem)
        job_id = f"preview_{exp_id}"
        try:
            from harness.remote_runtime import resolve_remote_runtime
            runtime = resolve_remote_runtime(HARNESS_PROJECT)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        if backend == "modal":
            stage_dir = PROJECT_ROOT / "tmp" / "modal" / job_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            script_path = stage_dir / f"app_{exp_id}.py"
            code_content = generate_modal_script(
                str(config_file.relative_to(PROJECT_ROOT)), gpu=gpu, exp_id=exp_id,
                volume_name=volume_name, project_root=PROJECT_ROOT,
                task_config_path=HARNESS_PROJECT.task_config_path, task_id=task_id,
                job_id=job_id, runtime_manifest=runtime,
                artifacts_rel=ARTIFACTS_DIR.relative_to(PROJECT_ROOT).as_posix(),
            )
            script_path.write_text(code_content, encoding="utf-8")

            self.send_json({
                "success": True,
                "backend": "modal",
                "script_path": str(script_path.relative_to(PROJECT_ROOT)),
                "code": code_content,
                "commands": {
                    "dataset_cmd": f"ml-harness cloud consent --provider modal --confirm-upload\n# then: python harness/cloud_runner.py --backend modal --project-root . --config {config_path}",
                    "push_cmd": f"python -m harness.cloud_runner --backend modal --project-root . --config {config_path} --gpu {gpu}",
                    "status_cmd": "modal app list",
                    "output_cmd": f"modal volume ls {volume_name}",
                }
            })
        else: # kaggle
            stage_dir = PROJECT_ROOT / "tmp" / "kaggle" / job_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            KaggleKernelManager.setup_kernel_package(
                stage_dir, str(config_file.relative_to(PROJECT_ROOT)), exp_id, dataset_slug,
                project_root=PROJECT_ROOT, task_config_path=HARNESS_PROJECT.task_config_path,
                task_id=task_id, job_id=job_id, runtime_manifest=runtime,
                enable_internet=bool((task_data.get("runtime") or {}).get("enable_internet", False)),
            )
            meta_file = stage_dir / "kernel-metadata.json"
            code_file = stage_dir / "run_kaggle.py"

            meta_json = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            code_content = code_file.read_text(encoding="utf-8") if code_file.exists() else ""

            self.send_json({
                "success": True,
                "backend": "kaggle",
                "script_path": str(code_file.relative_to(PROJECT_ROOT)),
                "code": code_content,
                "metadata": meta_json,
                "commands": {
                    "dataset_cmd": f"ml-harness cloud consent --provider kaggle --confirm-upload\n# private by default; declare data_config.license before uploading a new dataset",
                    "push_cmd": f"python -m harness.cloud_runner --backend kaggle --project-root . --config {config_path}",
                    "kernel_cmd": f"kaggle kernels push -p {stage_dir}",
                    "status_cmd": f"kaggle kernels status {meta_json.get('id', 'antigravity/' + exp_id)}",
                    "output_cmd": f"kaggle kernels output {meta_json.get('id', 'antigravity/' + exp_id)} -p {ARTIFACTS_DIR / exp_id}",
                }
            })

    def handle_api_cloud_push_dataset(self, data: Dict[str, Any]) -> None:
        platform = (data.get("platform") or data.get("backend") or "kaggle").lower()
        if platform not in {"kaggle", "modal"}:
            self.send_json({"error": f"Unsupported provider: {platform}"}, status=400)
            return
        from harness.cloud_policy import UploadConsentStore, dataset_fingerprint
        from harness.cloud_policy import private_upload_resource_names
        fingerprint = dataset_fingerprint(HARNESS_PROJECT)
        if not UploadConsentStore(HARNESS_PROJECT).is_granted(platform, fingerprint["fingerprint"]):
            self.send_json({
                "error": f"Upload consent is required for this task, provider, and dataset fingerprint.",
                "fingerprint": fingerprint["fingerprint"],
                "consent_endpoint": "/api/cloud/consent",
            }, status=409)
            return
        resources = private_upload_resource_names(
            HARNESS_PROJECT,
            fingerprint_data=fingerprint,
            kaggle_username=KaggleKernelManager.get_authenticated_username() if platform == "kaggle" else None,
        )

        if platform == "modal":
            res = push_dataset_to_modal(volume_name=resources["modal_volume_name"], project_root=PROJECT_ROOT)
        else:
            dataset_slug = resources["kaggle_dataset_slug"]
            if not dataset_slug:
                self.send_json({"error": "Could not determine the active Kaggle username. Connect the account through the dashboard or run `kaggle auth login --force`, then retry."}, status=409)
                return
            res = KaggleKernelManager.push_dataset_to_kaggle(dataset_slug=dataset_slug, project_root=PROJECT_ROOT)

        res.setdefault("dataset_fingerprint", fingerprint["fingerprint"])
        res.setdefault("visibility", "private")

        self.send_json(res, status=200 if res.get("success") else 400)

    def handle_api_cloud_kaggle_status(self, data: Dict[str, Any]) -> None:
        config_path = data.get("config_path", "")
        exp_id = data.get("exp_id") or Path(config_path).stem
        kernel_slug = data.get("kernel_slug")
        res = KaggleKernelManager.get_kernel_status(exp_id=exp_id, kernel_slug=kernel_slug)
        self.send_json(res)

    def handle_api_cloud_pull_output(self, data: Dict[str, Any]) -> None:
        platform = (data.get("platform") or data.get("backend") or "").lower()
        job_id = data.get("job_id")
        if platform not in {"kaggle", "modal"}:
            self.send_json({"error": "Choose a tracked Kaggle or Modal job."}, status=400)
            return
        if not job_id:
            self.send_json({"error": "Pull output through the tracked cloud job so task/config/artifact identity can be verified."}, status=409)
            return

        manager = CloudJobManager(PROJECT_ROOT)
        job = manager.get_job(str(job_id))
        if not job or job.get("backend") != platform:
            self.send_json({"error": "The job does not belong to this provider and workspace."}, status=404)
            return
        if job.get("status") != "COMPLETED" or not job.get("artifact_identity_verified"):
            self.send_json({"error": "Cloud job is not completed with a verified artifact identity.", "job": job}, status=409)
            return

        exp_dir = (PROJECT_ROOT / job["artifact_dir"]).resolve()
        try:
            exp_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            self.send_json({"error": "The tracked artifact path is outside this task workspace."}, status=403)
            return
        metrics_path = exp_dir / "metrics.json"
        try:
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Verified job metrics are unavailable: {exc}"}, status=409)
            return
        found_files = [path.name for path in exp_dir.iterdir() if path.is_file()]
        self.send_json({
            "success": True,
            "job_id": job_id,
            "exp_id": job["exp_id"],
            "artifacts_dir": str(exp_dir),
            "migrated_files": found_files,
            "metrics": metrics_data,
            "identity_verified": True,
        })

    def handle_api_cloud_run(self, data: Dict[str, Any]) -> None:
        backend = data.get("backend", "modal")
        config_path = data.get("config_path")
        if not config_path:
            self.send_json({"error": "config_path is required."}, status=400)
            return

        gpu = data.get("gpu", "A10")
        auto_audit = bool(data.get("auto_audit", True))
        auto_sync = bool(data.get("auto_sync", True))
        baseline_csv = data.get("baseline_csv")

        manager = CloudJobManager(PROJECT_ROOT)
        job = manager.start_job(
            backend=backend,
            config_path=config_path,
            gpu=gpu,
            auto_audit=auto_audit,
            auto_sync=auto_sync,
            baseline_csv=baseline_csv,
        )
        self.send_json({"success": True, "job": job.to_dict()})

    def handle_api_cloud_stop(self, data: Dict[str, Any]) -> None:
        job_id = data.get("job_id")
        if not job_id:
            self.send_json({"error": "job_id is required."}, status=400)
            return
        res = CloudJobManager().stop_job(job_id)
        status_code = 200 if res.get("success") else 400
        self.send_json(res, status=status_code)

    def handle_api_cloud_get_job(self, query_str: str) -> None:
        query = urllib.parse.parse_qs(query_str)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_json({"error": "Missing job_id parameter"}, status=400)
            return
        job = CloudJobManager().get_job(job_id)
        if not job:
            self.send_json({"error": f"Job '{job_id}' not found."}, status=404)
            return
        self.send_json({"job": job})

    def handle_api_cloud_logs_stream(self, query_str: str) -> None:
        query = urllib.parse.parse_qs(query_str)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_json({"error": "Missing job_id parameter"}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        manager = CloudJobManager()
        for chunk in manager.stream_logs(job_id):
            try:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def handle_api_cloud_audit(self, data: Dict[str, Any]) -> None:
        exp_id = data.get("exp_id")
        candidate_csv = data.get("candidate_csv")
        if not candidate_csv and exp_id:
            candidate_csv = str(ARTIFACTS_DIR / exp_id / "oof_predictions.csv")

        if not candidate_csv or not Path(candidate_csv).exists():
            self.send_json({"error": f"Candidate predictions not found: '{candidate_csv}'"}, status=400)
            return

        baseline_csv = data.get("baseline_csv")
        if not baseline_csv:
            best = TreeJournal().get_best_node()
            if best and (ARTIFACTS_DIR / best.id / "oof_predictions.csv").exists():
                baseline_csv = str(ARTIFACTS_DIR / best.id / "oof_predictions.csv")

        cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "audit_gate.py"), "--candidate", candidate_csv]
        if baseline_csv and Path(baseline_csv).exists():
            cmd.extend(["--baseline", baseline_csv])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        receipt = None
        receipt_path = PROJECT_ROOT / "AUDIT_RECEIPT.json"
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.send_json({
            "success": proc.returncode == 0,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "receipt": receipt,
        })

    def handle_api_llm_providers(self) -> None:
        try:
            providers = LLMEngine.list_providers()
            self.send_json({"providers": providers})
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_api_llm_sample_data(self) -> None:
        try:
            task_cfg = load_task_config()
            train_path_str = task_cfg.get("data_config", {}).get("train_path", "dataset/")
            train_dir = PROJECT_ROOT / train_path_str

            columns = []
            rows = []

            sample_file = None
            if train_dir.is_file():
                sample_file = train_dir
            elif train_dir.is_dir():
                for ext in ["*.tsv", "*.csv", "*.parquet", "*.jsonl"]:
                    matches = list(train_dir.glob(ext))
                    if matches:
                        sample_file = sorted(matches)[0]
                        break

            if sample_file and sample_file.exists():
                try:
                    import pandas as pd
                    sep = "\t" if sample_file.suffix == ".tsv" else ","
                    df = pd.read_csv(sample_file, sep=sep, nrows=5)
                    columns = [str(c) for c in df.columns]
                    rows = df.fillna("").to_dict(orient="records")
                except Exception:
                    pass

            if not columns:
                columns = [inp.get("name", "text") for inp in task_cfg.get("inputs", [])] or ["text"]
                rows = [{"text": "Sample text comment demonstrating LLM prompt extraction and classification."}]

            self.send_json({
                "columns": columns,
                "rows": rows,
                "total_samples": len(rows),
                "source_file": str(sample_file.relative_to(PROJECT_ROOT)).replace("\\", "/") if sample_file else None,
            })
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_api_llm_test_prompt(self, data: Dict[str, Any]) -> None:
        try:
            provider = data.get("provider", "gemini")
            model_name = data.get("model", "gemini-2.0-flash")
            system_prompt = data.get("system_prompt", "")
            user_prompt_template = data.get("user_prompt_template", "{text}")
            row_data = data.get("input_row", {})
            api_key = data.get("api_key")
            base_url = data.get("base_url")
            temperature = float(data.get("temperature", 0.0))
            max_tokens = int(data.get("max_tokens", 512))
            use_cot = bool(data.get("use_cot", False))
            few_shots = data.get("few_shot_examples", [])
            valid_classes = data.get("valid_classes", [])

            eff_sys, formatted_user = LLMEngine.format_prompt(
                template=user_prompt_template,
                row_data=row_data,
                system_prompt=system_prompt,
                few_shots=few_shots,
                use_cot=use_cot,
                valid_classes=valid_classes,
            )

            resp = LLMEngine.generate_completion(
                provider=provider,
                model_name=model_name,
                system_prompt=eff_sys,
                user_prompt=formatted_user,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            )

            raw_text = resp.get("text", "")
            extracted = LLMEngine.extract_prediction(raw_text, valid_classes=valid_classes)

            self.send_json({
                "success": True,
                "provider": provider,
                "model": model_name,
                "effective_system_prompt": eff_sys,
                "effective_user_prompt": formatted_user,
                "raw_text": raw_text,
                "extracted": extracted,
                "latency_ms": resp.get("latency_ms", 0),
                "usage": resp.get("usage", {}),
            })
        except Exception as e:
            self.send_json({"error": str(e), "success": False}, status=400)

    def handle_api_llm_generate_peft(self, data: Dict[str, Any]) -> None:
        try:
            model_name = data.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
            train_path = data.get("train_path", "dataset/train.csv")
            output_dir = data.get("output_dir", str(ARTIFACTS_DIR / "exp_qlora"))
            text_column = data.get("text_column", "text")
            target_column = data.get("target_column", "target")
            lora_r = int(data.get("lora_r", 16))
            lora_alpha = int(data.get("lora_alpha", 32))
            lora_dropout = float(data.get("lora_dropout", 0.05))
            target_modules = data.get("target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
            num_train_epochs = int(data.get("num_train_epochs", 3))
            batch_size = int(data.get("batch_size", 4))
            gradient_accumulation_steps = int(data.get("gradient_accumulation_steps", 4))
            learning_rate = float(data.get("learning_rate", 2e-4))

            script_code = LLMEngine.generate_qlora_script(
                model_name=model_name,
                train_path=train_path,
                output_dir=output_dir,
                text_column=text_column,
                target_column=target_column,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
            )

            save_to_disk = data.get("save_to_disk", False)
            saved_path = None
            if save_to_disk:
                scripts_dir = PROJECT_ROOT / "scripts" / "llm"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.split("/")[-1].lower())
                saved_file = scripts_dir / f"train_qlora_{clean_name}.py"
                saved_file.write_text(script_code, encoding="utf-8")
                saved_path = str(saved_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

            self.send_json({
                "success": True,
                "script_code": script_code,
                "saved_path": saved_path,
            })
        except Exception as e:
            self.send_json({"error": str(e), "success": False}, status=400)

    def handle_api_llm_save_key(self, data: Dict[str, Any]) -> None:
        try:
            key_name = data.get("key_name")
            key_value = data.get("key_value", "")
            if not key_name:
                self.send_json({"error": "key_name is required."}, status=400)
                return
            res = EnvironmentDetector.save_llm_api_key(key_name, key_value)
            self.send_json(res)
        except Exception as e:
            self.send_json({"error": str(e)}, status=400)

    def handle_api_llm_verify_key(self, data: Dict[str, Any]) -> None:
        try:
            provider = data.get("provider") or data.get("key_name") or ""
            key_value = data.get("key_value") or data.get("api_key")
            if not provider:
                self.send_json({"error": "provider or key_name is required.", "success": False, "valid": False}, status=400)
                return
            res = EnvironmentDetector.verify_llm_api_key(provider, key_value)
            self.send_json(res)
        except Exception as e:
            self.send_json({"error": str(e), "success": False, "valid": False}, status=400)

    def handle_api_llm_hf_models(self, query_str: str) -> None:
        try:
            params = urllib.parse.parse_qs(query_str)
            search = params.get("search", [""])[0]
            pipeline = params.get("pipeline", [None])[0]
            author = params.get("author", [None])[0]
            family = params.get("family", [None])[0]
            param_range = params.get("param_range", [None])[0]
            quantization = params.get("quantization", [None])[0]
            sort = params.get("sort", ["downloads"])[0]
            limit = int(params.get("limit", [30])[0])
            force_refresh = params.get("force_refresh", ["false"])[0].lower() in ("true", "1", "yes")
            models = LLMEngine.fetch_hf_models(
                query=search,
                pipeline_tag=pipeline,
                author=author,
                family=family,
                param_range=param_range,
                quantization=quantization,
                limit=limit,
                sort=sort,
                force_refresh=force_refresh,
            )
            self.send_json({"models": models, "count": len(models)})
        except Exception as e:
            self.send_json({"error": str(e), "models": []}, status=500)

    def handle_api_llm_unsloth_models(self, query_str: str) -> None:
        try:
            params = urllib.parse.parse_qs(query_str)
            search = params.get("search", [""])[0]
            family = params.get("family", [None])[0]
            param_range = params.get("param_range", [None])[0]
            quantization = params.get("quantization", [None])[0]
            sort = params.get("sort", ["downloads"])[0]
            limit = int(params.get("limit", [50])[0])
            force_refresh = params.get("force_refresh", ["false"])[0].lower() in ("true", "1", "yes")
            models = LLMEngine.fetch_unsloth_models(
                query=search,
                family=family,
                param_range=param_range,
                quantization=quantization,
                limit=limit,
                sort=sort,
                force_refresh=force_refresh,
            )
            self.send_json({"models": models, "count": len(models)})
        except Exception as e:
            self.send_json({"error": str(e), "models": []}, status=500)

    def handle_api_llm_provider_models(self, query_str: str) -> None:
        try:
            params = urllib.parse.parse_qs(query_str)
            provider = params.get("provider", [""])[0]
            api_key = params.get("api_key", [None])[0]
            if not provider:
                self.send_json({"error": "provider is required"}, status=400)
                return
            models = LLMEngine.fetch_provider_live_models(provider=provider, api_key=api_key)
            self.send_json({"provider": provider, "models": models, "count": len(models)})
        except Exception as e:
            self.send_json({"error": str(e), "models": []}, status=500)

    def handle_api_llm_generate_unsloth(self, data: Dict[str, Any]) -> None:
        try:
            model_name = data.get("model_name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            train_path = data.get("train_path", "dataset/train.csv")
            output_dir = data.get("output_dir", str(ARTIFACTS_DIR / "exp_unsloth_lora"))
            text_column = data.get("text_column", "text")
            target_column = data.get("target_column", "target")
            max_seq_length = int(data.get("max_seq_length", 2048))
            lora_r = int(data.get("lora_r", 16))
            lora_alpha = int(data.get("lora_alpha", 32))
            lora_dropout = float(data.get("lora_dropout", 0.0))
            num_train_epochs = int(data.get("num_train_epochs", 3))
            batch_size = int(data.get("batch_size", 4))
            gradient_accumulation_steps = int(data.get("gradient_accumulation_steps", 4))
            learning_rate = float(data.get("learning_rate", 2e-4))

            script_code = LLMEngine.generate_unsloth_script(
                model_name=model_name,
                train_path=train_path,
                output_dir=output_dir,
                text_column=text_column,
                target_column=target_column,
                max_seq_length=max_seq_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
            )

            save_to_disk = data.get("save_to_disk", False)
            saved_path = None
            if save_to_disk:
                scripts_dir = PROJECT_ROOT / "scripts" / "llm"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.split("/")[-1].lower())
                saved_file = scripts_dir / f"train_unsloth_{clean_name}.py"
                saved_file.write_text(script_code, encoding="utf-8")
                saved_path = str(saved_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

            self.send_json({
                "success": True,
                "script_code": script_code,
                "saved_path": saved_path,
            })
        except Exception as e:
            self.send_json({"error": str(e), "success": False}, status=400)



def run_server(port: int = 8080) -> None:
    web_dir = Path(__file__).resolve().parent / "web"
    web_dir.mkdir(parents=True, exist_ok=True)

    handler = DashboardHandler
    socketserver.TCPServer.allow_reuse_address = True

    current_port = port
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with socketserver.ThreadingTCPServer(("127.0.0.1", current_port), handler) as httpd:
                print(f"=" * 65)
                print(f"🚀 ML/DL Experiment Dashboard & Priming Server Active")
                print(f"📡 Web URL:  http://localhost:{current_port}")
                print(f"📋 API Root: http://localhost:{current_port}/api/tree")
                print(f"🎯 Priming:  http://localhost:{current_port}/api/priming_prompt")
                print(f"=" * 65)
                httpd.serve_forever()
                break
        except OSError as e:
            if "Address already in use" in str(e) or attempt < max_retries - 1:
                print(f"Port {current_port} is busy, trying {current_port + 1}...")
                current_port += 1
            else:
                raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Start ML/DL Experiment Dashboard & Priming Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not attempt to open a browser window")
    args = parser.parse_args()
    run_server(port=args.port)


if __name__ == "__main__":
    main()
