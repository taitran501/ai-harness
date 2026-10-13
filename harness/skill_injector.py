"""Skill Injector Engine for AI-Harness.

Dynamically adapts and injects the AI Agent skill (.agents/skills/ai-harness/SKILL.md)
to perfectly match any competition or benchmark (Kaggle, Codabench, SemEval, DrivenData, Hugging Face),
parameterizing task modalities, metrics, subtasks, grounding backbones, and audit rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from harness.project import HarnessProject

MODALITY_GROUNDING_SUGGESTIONS: Dict[str, Dict[str, Any]] = {
    "text": {
        "title": "Natural Language Processing (NLP) — Verified Core",
        "backbones": [
            "microsoft/deberta-v3-base (English / General SOTA encoder)",
            "microsoft/mdeberta-v3-base (Multilingual cross-lingual transfer)",
            "FacebookAI/xlm-roberta-base (Robust multilingual tokenization)",
            "answerdotai/ModernBERT-base (Fast, long-context 8k modern encoder)",
        ],
        "losses": ["Focal Loss (gamma=2.0)", "Class-Balanced BCE", "Asymmetric Loss"],
        "tokenization": "Pay careful attention to token lengths, special metadata tags, and multilingual subwords.",
    },
    "tabular": {
        "title": "Tabular & Structured Data (Experimental / WIP — Unverified)",
        "backbones": [
            "LightGBM (Gradient Boosting with histogram binning)",
            "XGBoost (Exact greedy & histogram tree booster)",
            "CatBoost (Symmetric trees with native categorical handling)",
            "TabNet (Attentive transformer for tabular data)",
            "ResNet/MLP for Tabular (Deep embeddings with skip connections)",
        ],
        "losses": ["Binary LogLoss", "Multi-Class Cross-Entropy", "RMSE / Hubber / Fair Loss"],
        "tokenization": "Target encoding, frequency encoding, numerical quantile transforms, and out-of-fold target statistics.",
    },
    "vision": {
        "title": "Computer Vision (Roadmap / WIP — Unverified)",
        "backbones": [
            "timm/convnext_small.fb_in22k_ft_in1k (Modern ConvNet architecture)",
            "timm/swin_base_patch4_window7_224 (Hierarchical Vision Transformer)",
            "timm/eva02_base_patch14_448 (High-resolution ViT)",
        ],
        "losses": ["Cross-Entropy with Label Smoothing (eps=0.1)", "Focal Loss", "Cosine Annealing Schedule"],
        "tokenization": "RandAugment, Mixup (alpha=0.8), CutMix (alpha=1.0), and test-time augmentation (TTA).",
    },
    "time-series": {
        "title": "Time-Series & Forecasting (Roadmap / WIP — Unverified)",
        "backbones": [
            "PatchTST (Patching and Channel-Independence Transformer)",
            "DLinear / N-Linear (Fast, interpretable linear decomposition)",
            "LightGBM / CatBoost (Lag features, rolling statistics, calendar encodings)",
        ],
        "losses": ["MAE (L1)", "MSE (L2)", "Quantile Loss for probabilistic intervals"],
        "tokenization": "Rolling window splits strictly expanding or sliding; prevent forward-looking leakages.",
    },
    "generic": {
        "title": "General Machine Learning / Deep Learning (Experimental / WIP — Unverified)",
        "backbones": [
            "Tree Ensembles (LightGBM, XGBoost, Random Forest)",
            "Deep Neural Networks (PyTorch MLP, Transformers)",
        ],
        "losses": ["Task-appropriate objective function (BCE, CrossEntropy, MSE)"],
        "tokenization": "Standard normalization, missing value imputation, and out-of-fold cross-validation.",
    },
}


def load_task_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the attached task specification or a specified task config."""
    project = HarnessProject.discover()
    path = Path(config_path) if config_path is not None else project.task_config_path
    if not path.is_absolute():
        path = project.root / path
    if not path.exists():
        return {
            "name": "General ML/DL Benchmark",
            "platform": "generic",
            "type": "Supervised Learning",
            "modality": "generic",
            "primary_metric": "Score (CV)",
            "direction": "maximize",
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("task", data)


def generate_skill_content(
    task: Dict[str, Any],
    artifact_path: str = ".ai-harness/artifacts/<task-id>",
    skill_name: str = "ai-harness",
) -> str:
    """Generate Markdown content for SKILL.md dynamically tailored to the target task."""
    task_name = task.get("task_name") or task.get("name") or "ML/DL Benchmark"
    platform = (task.get("platform") or "generic").capitalize()
    task_type = task.get("task_type") or task.get("type") or "Supervised Learning"
    modality = (task.get("modality") or "generic").lower()
    primary_metric = task.get("primary_metric") or "Score (CV)"
    direction = task.get("direction") or "maximize"
    base_score = task.get("primary_metric_baseline")
    target_score = task.get("primary_metric_target")
    metrics = task.get("metrics") or task.get("subtasks") or []
    languages = task.get("languages") or []
    description = task.get("description") or "Iterative ML/DL model development and cross-validation benchmarking."

    # Modality knowledge
    mod_info = MODALITY_GROUNDING_SUGGESTIONS.get(modality, MODALITY_GROUNDING_SUGGESTIONS["generic"])

    lines: List[str] = [
        "---",
        f"name: {skill_name}",
        f"description: Manages ML experiments for {task_name} on {platform} via AIDE tree search, enforces SWE-agent deterministic audit gate, and mandates Epistemic Grounding & Assumption Registers.",
        "---",
        "",
        f"# AI Experiment Harness Skill — {task_name}",
        "",
        f"> **Active Benchmark**: **{task_name}** | **Platform**: {platform} | **Modality**: {mod_info['title']}",
        f"> **Objective**: {direction.capitalize()} `{primary_metric}` across cross-validation folds.",
        ">",
        "> [!IMPORTANT]",
        "> **Scope Notice**: AI-Harness is currently validated and working well ONLY on Natural Language Processing (NLP / Text) tasks. Other modalities (Tabular, Vision, Audio, Multimodal) are experimental works-in-progress (WIP / Roadmap) and NOT yet validated in practice.",
        "",
        f"Use this skill whenever planning model architectures, engineering features, formulating hypotheses,",
        f"auditing predictions, or registering verified submissions for **{task_name}**.",
        "",
        "---",
        "",
        "## 1. Benchmark Specification & Score Objectives",
        "",
        f"- **Task Type**: `{task_type}`",
        f"- **Primary Target Metric**: `{primary_metric}` (Optimization Direction: `{direction}`)",
    ]

    if base_score is not None:
        lines.append(f"- **Baseline Reference Score**: `{base_score}`")
    if target_score is not None:
        lines.append(f"- **Target Goal Score**: `{target_score}`")
    if languages:
        lines.append(f"- **Target Languages**: {', '.join(languages)}")

    if metrics:
        lines.extend([
            "",
            "### Evaluation Criteria & Subtask Breakdown:",
            "| Criteria / Subtask | Metric | Direction | Baseline | Target | Weight | Description |",
            "|---|---|---|---|---|---|---|",
        ])
        for m in metrics:
            m_id = m.get("id", m.get("name", "metric"))
            m_name = m.get("name", m_id)
            m_metric = m.get("metric", primary_metric)
            m_dir = m.get("direction", direction)
            m_base = m.get("baseline_score", "-")
            m_target = m.get("target_score", "-")
            m_weight = m.get("weight", "-")
            m_desc = m.get("description", "-")
            lines.append(f"| **{m_name}** (`{m_id}`) | `{m_metric}` | `{m_dir}` | `{m_base}` | `{m_target}` | `{m_weight}` | {m_desc} |")

    # Data & Validation Strategy
    data_cfg = task.get("data_config", {})
    if data_cfg:
        fold_count = data_cfg.get("n_folds")
        fold_label = f"{fold_count} folds" if fold_count is not None else "fold count not declared"
        lines.extend([
            "",
            "### Data & Cross-Validation Strategy:",
            f"- **Validation Scheme**: `{data_cfg.get('split_strategy', 'Not declared')}` ({fold_label}, seed `{data_cfg.get('seed', 'not declared')}`)",
            f"- **Anti-Leakage Group Column**: `{data_cfg.get('group_column', 'Not declared')}`",
            f"- **Dataset Paths**: Train: `{data_cfg.get('train_path', 'not declared')}` | Test: `{data_cfg.get('test_path', 'not declared')}`",
            f"- **Key Columns**: ID: `{data_cfg.get('id_column', 'not declared')}` | Target: `{data_cfg.get('target_column', 'not declared')}`",
        ])

    # Competition Constraints & Rules
    comp_rules = task.get("competition_rules", {})
    if comp_rules:
        max_p = comp_rules.get("max_params_limit")
        p_str = f"{max_p / 1e6:.0f}M parameters" if max_p else "Not declared"
        def declared_permission(key: str) -> str:
            if key not in comp_rules:
                return "Not declared"
            return "Allowed" if comp_rules[key] else "Not allowed"
        lines.extend([
            "",
            "### Competition Rules & Constraints:",
            f"- **LLM / API**: `{declared_permission('allow_llm')}`",
            f"- **External Data**: `{declared_permission('allow_external_data')}`",
            f"- **Pretrained Weights**: `{declared_permission('allow_pretrained_models')}`",
            f"- **Parameter Ceiling**: `{p_str}`",
            f"- **Daily Submission Quota**: `{comp_rules.get('daily_submission_limit', 'Not declared')}`",
        ])
        if comp_rules.get("deadline"):
            lines.append(f"- **Competition Deadline**: `{comp_rules['deadline']}`")

    # Submission Specification
    sub_spec = task.get("submission_spec", {})
    if sub_spec:
        lines.extend([
            "",
            "### Submission & Audit Specification:",
            f"- **File Name / Format**: `{sub_spec.get('file_name', 'not declared')}` (`{sub_spec.get('archive_format', 'not declared')}`)",
            "- Follow only the audit rules explicitly declared by the task or its selected plugin.",
        ])
        for key, label in (
            ("min_diff_pct", "Minimum prediction difference (%)"),
            ("max_kappa", "Maximum ensemble Cohen's Kappa"),
            ("min_disagreement", "Minimum ensemble disagreement (%)"),
        ):
            if sub_spec.get(key) is not None:
                lines.append(f"- **{label}**: `{sub_spec[key]}`")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Three-Layer Architecture (`Antigravity IDE` <-> `AgentBridge` <-> `Harness`)",
        "",
        "- **Layer 1 — True Agent Runtime (`Antigravity in IDE`)**: Performs scientific reasoning, hypothesis formulation, and the `Observe -> Decide -> Next Action` loop. The Harness does **not** run a separate internal LLM planner.",
        "- **Layer 2 — Agent Bridge & Context Bridge (`harness/agent_bridge.py`, `ai-harness-agent`)**:",
        "  - Feeds Antigravity a compressed **Context Packet** (`CompetitionState` 8-Phase HUD, Parallel Tracks, `TreeJournal` UCB recommendation, Parent `DIAGNOSTICS` bottlenecks, Idea Graveyard, and active `BridgeSession` budget/usage):",
        "    ```powershell",
        "    ai-harness-agent context",
        "    ```",
        "  - Tracks stateful **`BridgeSession` (`session_XXXX`)** (`goal`, `experiments_run`, `compute_usage`, `consecutive_rejections`, optional `max_experiments` / `max_gpu_hours` / `max_cloud_runs` budgets) and enforces **`USER_APPROVAL` governance** on sensitive actions (`prepare_submission`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`).",
        "  - Exposes **Typed Harness Actions** returning structured `ActionObservation` JSON payloads (`metric_delta`, `composite_score`, `diagnostics_bottlenecks`, `new_blockers`):",
        "    ```powershell",
        "    ai-harness-agent start-session --goal \"Improve target metric\" --max-experiments 3",
        "    ai-harness-agent act --action <typed_action> --params '{\"...\": \"...\"}'",
        "    ```",
        "- **Layer 3 — Task State & Execution Runtime (`harness/competition_state.py`, `ai-harness-run`, `harness/cloud_runner.py`)**:",
        "  - **8-Phase Journey**: `1. Setup` -> `2. Understand` -> `3. Baseline Anchor` -> **Parallel Iteration Tracks** (`4. Experiment Loop` || `5. Candidate Management` || `6. External Evaluation`) -> `7. Freeze (Lock Candidate Portfolio & Mutations)` -> `8. Closure`.",
        "  - **Execution Backends**: Local (`ai-harness-run`); remote backends require the selected task adapter to be installed remotely.",
        "",
        "---",
        "",
        "## 3. Core Operational Rules (Enforced & Immutable)",
        "",
        "### 3.1 The Context Protection Rule (Anti-Context Saturation)",
        f"- **NEVER** attempt to dump raw fold logs or the entire `{artifact_path}/` directory into prompt context.",
        "- **ALWAYS** query the Agent Bridge Context Packet first:",
        "  ```powershell",
        "  ai-harness-agent context",
        "  ```",
        "",
        "### 3.2 The Anti-Pivot & Branch Exploration Rule (AIDE UCB Tree Search)",
        "- **DO NOT** make speculative guesses or pivot arbitrarily without mathematical justification.",
        "- Query the optimal branch recommendation according to Upper Confidence Bound (UCB):",
        "  ```powershell",
        "  ai-harness-agent context --format json",
        "  ```",
        "- Ensure every experiment implements **EXACTLY ONE ATOMIC INTERVENTION** (Control vs Treatment).",
        "",
        "### 3.3 Epistemic Grounding & Assumption Register Protocol (Anti-Hallucination)",
        f"- **MANDATORY RETRIEVAL**: Before formulating a new model configuration for {mod_info['title']}:",
        f"  1. Inspect `docs/KNOWLEDGE_BASE.md` for reference backbones (Note: Only NLP is verified; non-NLP modalities are experimental WIP):",
    ])

    for b in mod_info["backbones"][:4]:
        lines.append(f"     - `{b}`")

    lines.extend([
        f"  2. Inspect domain-verified loss functions: {', '.join(mod_info['losses'])}.",
        f"  3. Technique requirements: {mod_info['tokenization']}",
        "  4. If proposing an unlisted technique, perform external retrieval (`search_web`, Hugging Face Hub, ArXiv) first.",
        "- **MANDATORY CONFIG SCHEMA**: Declare explicit `grounding:` and `assumption_register:` blocks in `configs/exp<NNN>.yaml`.",
        "- **PRE-FLIGHT VALIDATION GATE**: Validate config before launching training:",
        "  ```powershell",
        "  python harness/grounding_gate.py --config configs/<exp_config>.yaml",
        "  ```",
        "",
        "### 3.4 Diagnostic-First Exploration Rule",
        f"- Before writing any config, inspect `{artifact_path}/<parent_id>/DIAGNOSTICS.md` (or call `act --action inspect_diagnostics`).",
        "- Declare a `problem_addressed` that directly targets a documented failure slice or confusion mode.",
        "- **Strictly DO NOT pseudo-ensemble** models of the same family or merge previous versions mechanically.",
        "",
        "### 3.5 SWE-Agent Physical & Statistical Audit Gate",
        "- Submissions and candidate predictions CANNOT be approved by natural language alone.",
        "- **Task Contract and Approval Gates**: Apply validation and submission requirements only when the task or selected plugin declares them.",
        "- **Submission checks**: Inspect the selected plugin capabilities and declared task contract. Do not infer scoring or audit thresholds.",
        "  ```powershell",
        "  ai-harness-agent act --action prepare_submission --user-approved --params '{\"exp_id\": \"<experiment-id>\", \"candidate_csv\": \"<test-predictions.csv>\"}'",
        "  ```",
        "",
        "### 3.6 Approval Governance (`AUTO` / `AUTO_WITH_LIMIT` / `USER_APPROVAL`)",
        "- **AUTO**: `inspect_data_readiness`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`",
        "- **AUTO_WITH_LIMIT (Session Budget Tracked)**: `establish_baseline_anchor`, `run_full_validation`, `promote_candidate`, `synthesize_candidate`, `build_ensemble`",
        "- **USER_APPROVAL (Mandatory Human Confirmation)**: `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`",
        "",
        "---",
        "",
        "## 4. Standard Operating Procedure (SOP) for Antigravity in IDE",
        "",
        "1. **Read Bridge Context Packet**: Run `ai-harness-agent context` to inspect the 8-Phase Competition State, active `BridgeSession`, UCB branch recommendation, and parent `DIAGNOSTICS` bottlenecks.",
        "2. **Start / Configure Session (Optional Budgets)**: Run `ai-harness-agent start-session --goal \"...\" --max-experiments 3`.",
        "3. **Create Grounded Experiment**: Run `ai-harness-agent act --action create_experiment --params '{...}'` (or write `configs/exp<NNN>_<branch>_<descriptor>.yaml` and verify via `harness/grounding_gate.py`).",
        "4. **Execute Smoke Test**: Run `ai-harness-agent act --action run_smoke_test --params '{\"config_path\": \"configs/<exp>.yaml\"}'`.",
        "5. **Execute Task-Declared Validation**: Run `ai-harness-agent act --action run_full_validation --params '{\"config_path\": \"configs/<exp>.yaml\", \"backend\": \"local\"}'`.",
        "6. **Observe & Decide Next Step**: Antigravity analyzes `ActionObservation` (`metric_delta`, `composite_score`, `diagnostics_bottlenecks`, `consecutive_rejections`). Promote candidates via `promote_candidate` (single experiment) OR `synthesize_candidate` (multi-model composition/specialist router).",
        "7. **Human-Gated Submission / Freeze**: Request user confirmation before invoking `prepare_submission` or `freeze_candidate` with `--user-approved`.",
        "",
    ])

    return "\n".join(lines)


def inject_skill_file(
    task_config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Inject/write dynamic SKILL.md tailored to the specified task configuration."""
    project = HarnessProject.discover()
    task_path = Path(config_path).expanduser() if config_path is not None else project.task_config_path
    if not task_path.is_absolute():
        task_path = project.root / task_path
    cfg = task_config or load_task_config(task_path)
    try:
        artifact_path = project.artifacts_dir.relative_to(project.root).as_posix()
    except ValueError:
        artifact_path = str(project.artifacts_dir)
    dest = output_path or (project.root / ".agents" / "skills" / "ai-harness" / "SKILL.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = generate_skill_content(cfg, artifact_path=artifact_path, skill_name=dest.parent.name)
    dest.write_text(content, encoding="utf-8")
    for twin_name in ("ai-experiment-harness", "ai-harness", "ml-experiment-harness", "ml-harness"):
        if dest.parent.parent.name == "skills" and dest.parent.name != twin_name:
            twin_dest = dest.parent.parent / twin_name / "SKILL.md"
            twin_dest.parent.mkdir(parents=True, exist_ok=True)
            twin_content = generate_skill_content(cfg, artifact_path=artifact_path, skill_name=twin_name)
            twin_dest.write_text(twin_content, encoding="utf-8")
        elif output_path is None:
            twin_dest = project.root / ".agents" / "skills" / twin_name / "SKILL.md"
            twin_dest.parent.mkdir(parents=True, exist_ok=True)
            twin_content = generate_skill_content(cfg, artifact_path=artifact_path, skill_name=twin_name)
            twin_dest.write_text(twin_content, encoding="utf-8")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inject dynamic competition-specific skill into .agents/skills/")
    parser.add_argument("--config", type=str, default="configs/task.yaml", help="Path to task configuration YAML")
    parser.add_argument("--output", type=str, default=None, help="Output path for SKILL.md")
    args = parser.parse_args()

    out_p = inject_skill_file(
        config_path=Path(args.config),
        output_path=Path(args.output) if args.output else None,
    )
    print(f"✔ Successfully injected dynamic task-specific skill into: {out_p}")
