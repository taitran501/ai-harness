#!/usr/bin/env python3
"""Task-Agnostic ML/DL Experimentation Dashboard & Antigravity Priming Prompt Server.

Zero-dependency standard library Python HTTP server providing:
1. REST APIs for the AIDE solution tree, experiments, diagnostics, and UCB recommendations.
2. Dynamic Antigravity Priming Prompt generation to eliminate agent amnesia and hallucinated certainty.
3. Clean modern web SPA frontend.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.tree_journal import Node, TreeJournal
from harness.submission_manager import SubmissionManager
from harness.cloud_inspector import CloudPreflightInspector
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script
from harness.skill_injector import inject_skill_file


def load_task_config() -> Dict[str, Any]:
    """Load dynamic task specification from configs/task.yaml or fall back to generic schema."""
    task_file = PROJECT_ROOT / "configs" / "task.yaml"
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
        "platform": "codabench",
        "name": "General ML/DL Benchmark",
        "type": "Predictive Modeling",
        "modality": "generic",
        "primary_metric": "Score (CV)",
        "direction": "maximize",
        "metrics": [],
        "languages": [],
        "description": "Task-agnostic ML/DL experimentation harness.",
    }


def save_task_config(task_data: Dict[str, Any]) -> None:
    """Save updated task specification to configs/task.yaml."""
    task_file = PROJECT_ROOT / "configs" / "task.yaml"
    import yaml
    task_file.parent.mkdir(parents=True, exist_ok=True)
    content = {"task": task_data}
    with open(task_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


TASK_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "codabench_nlp_multitask",
        "preset_name": "Codabench / SemEval NLP (Multi-Task)",
        "platform": "codabench",
        "name": "StereoQueerEval",
        "type": "Multi-Task Text Classification",
        "modality": "text",
        "primary_metric": "Macro-F1 (CV)",
        "direction": "maximize",
        "primary_metric_baseline": 0.6459,
        "primary_metric_target": 0.85,
        "metrics": [
            {"id": "subtask_a", "name": "Stereotype Detection", "metric": "Macro-F1", "direction": "maximize", "baseline_score": 0.6962, "target_score": 0.88, "weight": 0.333, "description": "Binary stereotype detection"},
            {"id": "subtask_b", "name": "Hate Speech", "metric": "Macro-F1", "direction": "maximize", "baseline_score": 0.5238, "target_score": 0.80, "weight": 0.333, "description": "3-class classification"},
            {"id": "subtask_c", "name": "Target Type & Identity", "metric": "Macro-F1", "direction": "maximize", "baseline_score": 0.7178, "target_score": 0.85, "weight": 0.334, "description": "Conditional oracle_hate evaluation"},
        ],
        "languages": ["EN", "IT", "NL"],
        "description": "Cross-lingual analysis of YouTube comments conditioned on video metadata.",
        "data_config": {
            "train_path": "dataset/",
            "test_path": "dataset/test/",
            "split_strategy": "GroupKFold",
            "group_column": "yt_title",
            "n_folds": 5,
            "seed": 42,
            "id_column": "StereoQueerEval_id",
            "target_column": "stereotype,hate_speech,target_type,target_identities",
            "missing_data_policy": "drop_invalid_rows",
        },
        "competition_rules": {
            "allow_llm": False,
            "allow_api": False,
            "allow_external_data": False,
            "allow_pretrained_models": True,
            "max_params_limit": 150000000,
            "daily_submission_limit": 5,
            "deadline": "2027-01-15T23:59:59Z",
            "reproducibility_required": True,
        },
        "submission_spec": {
            "file_name": "submission.zip",
            "archive_format": "zip",
            "min_diff_pct": 2.0,
            "max_kappa": 0.85,
            "min_disagreement": 5.0,
        },
    },
    {
        "id": "kaggle_tabular_regression",
        "preset_name": "Kaggle Tabular Regression (e.g. House Prices)",
        "platform": "kaggle",
        "name": "House Prices Regression",
        "type": "Tabular Regression",
        "modality": "tabular",
        "primary_metric": "RMSE (CV)",
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
        "primary_metric": "ROC-AUC (CV)",
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
        "primary_metric": "Top-1 Accuracy (CV)",
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
        "primary_metric": "Score (CV)",
        "direction": "maximize",
        "primary_metric_baseline": 0.700,
        "primary_metric_target": 0.850,
        "metrics": [
            {"id": "primary_metric", "name": "Primary Metric", "metric": "Score", "direction": "maximize", "baseline_score": 0.700, "target_score": 0.850, "weight": 1.0, "description": "Primary benchmark objective"},
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
    primary_metric = task.get("primary_metric", "Macro-F1 (CV)")
    direction = task.get("direction", "maximize")

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

    graveyard_items = [
        "- **[REFUTED] Pseudo-Ensembles**: Combining two models of the same family is blocked by Audit Gate (Cohen's Kappa > 0.85).",
        "- **[REFUTED] Unverified Hyperparameter Tweaks**: Minor learning rate or regularization adjustments changing < 2% of predictions fail MMES.",
        "- **[REFUTED] Unconditioned Auxiliary Noise**: Appending uncurated metadata without feature selection degrades validation performance.",
    ]

    rec_parent = rec.get("parent_id") or best_id
    rec_branch = rec.get("branch") or "feat"

    if template_id == "codex":
        # Industry-standard OpenAI Codex / Coding Agent system prompt
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
            "3. **No Pseudo-Ensembles**: You are strictly forbidden from ensembling models of the same family. Ensembles require distinct architectures with Cohen's Kappa <= 0.85.",
            "4. **Physical Audit Gate**: Submissions and candidate predictions must pass `tools/audit_gate.py` with Minimum Meaningful Effect Size (MMES >= 2.0% row delta).",
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
            "3. **Validate**: Execute CV validation: `python train.py --config configs/exp<NNN>_<branch>_<descriptor>.yaml --debug --fold 0` (smoke test first), then full CV.",
            "4. **Output Format**: Provide a concise summary, git diff, and resulting validation scores.",
        ]
    elif template_id == "aide":
        prompt_lines = [
            f"# AIDE AUTONOMOUS ML SOLUTION TREE SEARCH PRIMING",
            "",
            f"You are navigating the Solution Tree for **{task_name}**.",
            f"Objective: {direction.capitalize()} `{primary_metric}` across cross-validation folds.",
            "",
            "## 1. Current Frontier State",
            f"- Champion Node: `{best_id}` (Score: `{best_score}`)",
            f"- UCB Exploration Target: Branch `{rec_branch}` on parent `{rec_parent}`",
            f"- Search Guideline: {rec.get('guideline')}",
            "",
            "## 2. Empirical Failure Slices",
            *diagnostics_notes,
            "",
            "## 3. Operational Rules",
            "1. Register atomic hypothesis declaring expected quantitative delta.",
            "2. Execute cross-validation on precomputed canonical folds without data leakage.",
            "3. Verify effect size MMES >= 2.0% before packaging submission.",
        ]
    elif template_id == "minimal":
        prompt_lines = [
            f"# DIRECT ML EXECUTION: {task_name.upper()}",
            f"- Task: {task_type} | Target: {primary_metric} ({direction})",
            f"- Baseline Champion: {best_id} ({best_score})",
            f"- Next Branch Target: {rec_branch} (Parent: {rec_parent})",
            "",
            "## Instructions",
            "1. Plan 1 atomic intervention addressing diagnosed bottlenecks.",
            "2. Create configs/exp<NNN>_<branch>_<name>.yaml with explicit grounding block.",
            "3. Run: python train.py --config configs/exp<NNN>_<branch>_<name>.yaml --debug --fold 0",
            "4. Verify CV scores and ensure row delta MMES >= 2.0%.",
        ]
    else:  # Antigravity default
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
            "1. **Epistemic Grounding Gate (`harness/grounding_gate.py`)**: You are physically forbidden from proposing models from stale parametric memory. Every config MUST include explicit `grounding:` (citing Hugging Face Hub / ArXiv) and `assumption_register:`.",
            "2. **Physical Audit Gate (`tools/audit_gate.py`)**: Predictions with less than 2.0% row delta (MMES) or pseudo-ensembles ($\kappa > 0.85$) are physically rejected.",
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
        react_dist = PROJECT_ROOT / "dashboard" / "dist"
        if react_dist.exists() and (react_dist / "index.html").exists():
            self.web_dir = react_dist
        else:
            self.web_dir = PROJECT_ROOT / "dashboard"
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    def do_GET(self) -> None:
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
        elif path == "/api/task_info":
            self.handle_api_task_info()
        elif path == "/api/task_presets":
            self.send_json({"presets": TASK_PRESETS})
        elif path == "/api/configs":
            self.handle_api_configs()
        elif path == "/api/cloud/preflight":
            self.handle_api_cloud_preflight(parsed_url.query)
        elif path == "/api/sync":
            self.handle_api_sync()
        else:
            # Fallback to serving static frontend
            super().do_GET()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/submissions/record-score":
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

                macro_a = float(official_scores["subtask_a_macro"]) if official_scores.get("subtask_a_macro") is not None else None
                macro_b = float(official_scores["subtask_b_macro"]) if official_scores.get("subtask_b_macro") is not None else None
                macro_c_type = float(official_scores["subtask_c_type_macro"]) if official_scores.get("subtask_c_type_macro") is not None else None
                macro_c_id = float(official_scores["subtask_c_id_macro"]) if official_scores.get("subtask_c_id_macro") is not None else None
                split_type = str(official_scores.get("split_type", "public"))

                mgr.record_official_score(
                    sub_id=sub_id,
                    macro_a=macro_a,
                    macro_b=macro_b,
                    macro_c_type=macro_c_type,
                    macro_c_id=macro_c_id,
                    split_type=split_type,
                    notes=notes,
                    custom_scores=official_scores,
                )
                self.send_json({"success": True, "submission": asdict(mgr.submissions[sub_id])})
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        elif path == "/api/task_info":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                save_task_config(data)
                # Auto-inject updated agent skill for this task
                try:
                    inject_skill_file(task_config=data)
                except Exception as skill_err:
                    print(f"⚠️ Warning: Auto-injecting skill failed: {skill_err}")
                self.send_json({"success": True, "task": load_task_config(), "skill_injected": True})
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
                if not candidate_csv:
                    default_cand = PROJECT_ROOT / "artifacts" / exp_id / "oof_predictions.csv"
                    if default_cand.exists():
                        candidate_csv = str(default_cand)
                    else:
                        self.send_json({
                            "error": f"Candidate predictions file not found at '{default_cand}'. Please provide valid candidate_csv path."
                        }, status=400)
                        return

                baseline_csv = data.get("baseline_csv")
                notes = data.get("notes", "")
                min_diff_pct = float(data.get("min_diff_pct", 2.0))
                ensemble_components = data.get("ensemble_components")

                mgr = SubmissionManager()
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
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
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
        exp_dir = PROJECT_ROOT / "artifacts" / node_id
        receipt_file = exp_dir / "GROUNDING_RECEIPT.json"
        if receipt_file.exists():
            try:
                with open(receipt_file, "r", encoding="utf-8") as f:
                    grounding_data = json.load(f)
            except Exception:
                pass

        details = {
            "id": node.id,
            "parent_id": node.parent_id,
            "branch": node.branch,
            "approach_type": getattr(node, "approach_type", "ml"),
            "model_family": getattr(node, "model_family", "linear"),
            "score": round(node.get_score(), 4),
            "param_count": node.param_count,
            "status": node.status,
            "is_buggy": node.is_buggy,
            "atomic_hypothesis": node.atomic_hypothesis,
            "intervention": node.intervention,
            "notes": node.notes,
            "timestamp": node.timestamp,
            "metrics": node.metrics,
            "diagnostics_summary": node.diagnostics_summary,
            "grounding_receipt": grounding_data,
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
                    "id": "codex",
                    "name": "OpenAI Codex CLI",
                    "badge": "Recommended",
                    "description": "Standard coding agent system instructions for OpenAI Codex & autonomous coding CLI agents following AGENTS.md conventions.",
                },
                {
                    "id": "antigravity",
                    "name": "Antigravity Research Agent",
                    "badge": "Alignment Gate",
                    "description": "Scientific context injection enforcing 4-step Acknowledgment Gate and UCB search exploration.",
                },
                {
                    "id": "aide",
                    "name": "AIDE ML Solution Tree",
                    "badge": "Tree Search",
                    "description": "Tailored for iterative MCTS/UCB tree expansion, hypothesis tracking, and MMES audit.",
                },
                {
                    "id": "minimal",
                    "name": "Minimal Direct Patch",
                    "badge": "Fast Probe",
                    "description": "High-density concise prompt for quick debugging, smoke testing, and single-variable probing.",
                },
            ]
        })

    def handle_api_priming_prompt(self, query: str = "") -> None:
        params = urllib.parse.parse_qs(query)
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
        self.send_json({
            "platform": task.get("platform", "codabench"),
            "task_name": task.get("task_name") or task.get("name", "General ML/DL Benchmark"),
            "name": task.get("name", "General ML/DL Benchmark"),
            "task_type": task.get("task_type") or task.get("type", "Predictive Modeling"),
            "type": task.get("type", "Predictive Modeling"),
            "modality": task.get("modality", "generic"),
            "primary_metric": task.get("primary_metric", "Score (CV)"),
            "direction": task.get("direction", "maximize"),
            "primary_metric_baseline": task.get("primary_metric_baseline"),
            "primary_metric_target": task.get("primary_metric_target"),
            "metrics": task.get("metrics", []),
            "languages": task.get("languages", []),
            "description": task.get("description", ""),
            "data_config": task.get("data_config", {}),
            "competition_rules": task.get("competition_rules", {}),
            "submission_spec": task.get("submission_spec", {}),
        })

    def handle_api_configs(self) -> None:
        configs_dir = PROJECT_ROOT / "configs"
        cfg_files = []
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                if f.name != "task.yaml":
                    cfg_files.append(f"configs/{f.name}")
        self.send_json({"configs": cfg_files})

    def handle_api_cloud_preflight(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        config_path = params.get("config", ["configs/exp001_tfidf_baseline.yaml"])[0]
        full_path = PROJECT_ROOT / config_path
        passed, errors, resolved_cfg = CloudPreflightInspector.run_full_preflight_check(full_path)
        self.send_json({
            "passed": passed,
            "errors": errors,
            "resolved_config": resolved_cfg,
        })

    def handle_api_cloud_generate(self, data: Dict[str, Any]) -> None:
        backend = data.get("backend") or data.get("platform") or "modal"
        config_path = data.get("config_path", "configs/exp001_tfidf_baseline.yaml")
        gpu = data.get("gpu", "A10G" if backend == "modal" else "T4 x 2")
        dataset_slug = data.get("dataset_slug", "antigravity/stereoqueer-eval-data")
        volume_name = data.get("volume_name", "stereoqueer-dataset-volume")

        exp_id = Path(config_path).stem

        if backend == "modal":
            stage_dir = PROJECT_ROOT / "tmp" / "modal"
            stage_dir.mkdir(parents=True, exist_ok=True)
            script_path = stage_dir / f"app_{exp_id}.py"
            code_content = generate_modal_script(str(config_path), gpu=gpu, exp_id=exp_id)
            script_path.write_text(code_content, encoding="utf-8")

            self.send_json({
                "success": True,
                "backend": "modal",
                "script_path": str(script_path.relative_to(PROJECT_ROOT)),
                "code": code_content,
                "commands": {
                    "dataset_cmd": f"modal volume create {volume_name}\nmodal volume put {volume_name} dataset/ /data/",
                    "push_cmd": f"modal run tmp/modal/app_{exp_id}.py  # Or: python harness/cloud_runner.py --backend modal --config {config_path}",
                    "status_cmd": "modal app list",
                    "output_cmd": f"modal volume ls {volume_name}",
                }
            })
        else: # kaggle
            stage_dir = PROJECT_ROOT / "tmp" / "kaggle" / exp_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            KaggleKernelManager.setup_kernel_package(stage_dir, config_path, exp_id)
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
                    "dataset_cmd": f"kaggle datasets init -p dataset/\nkaggle datasets create -p dataset/ -r zip --public",
                    "push_cmd": f"kaggle kernels push -p tmp/kaggle/{exp_id}  # Or: python harness/cloud_runner.py --backend kaggle --config {config_path}",
                    "status_cmd": f"kaggle kernels status {meta_json.get('id', 'antigravity/' + exp_id)}",
                    "output_cmd": f"kaggle kernels output {meta_json.get('id', 'antigravity/' + exp_id)} -p artifacts/{exp_id}/",
                }
            })

    def handle_api_sync(self) -> None:
        journal = TreeJournal()
        newly_added = journal.sync_from_artifacts()
        self.send_json({
            "success": True,
            "synced_count": newly_added,
            "total_nodes": len(journal.nodes),
        })


def run_server(port: int = 8080) -> None:
    web_dir = Path(__file__).resolve().parent / "web"
    web_dir.mkdir(parents=True, exist_ok=True)

    handler = DashboardHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    current_port = port
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with socketserver.TCPServer(("", current_port), handler) as httpd:
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
