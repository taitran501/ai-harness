"""Skill Injector Engine for ML-Harness.

Dynamically adapts and injects the AI Agent skill (.agents/skills/ml-experiment-harness/SKILL.md)
to perfectly match any competition or benchmark (Kaggle, Codabench, SemEval, DrivenData, Hugging Face),
parameterizing task modalities, metrics, subtasks, grounding backbones, and audit rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = PROJECT_ROOT / ".agents" / "skills" / "ml-experiment-harness"
SKILL_FILE = SKILL_DIR / "SKILL.md"
TASK_CONFIG_FILE = PROJECT_ROOT / "configs" / "task.yaml"


MODALITY_GROUNDING_SUGGESTIONS: Dict[str, Dict[str, Any]] = {
    "text": {
        "title": "Natural Language Processing (NLP)",
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
        "title": "Tabular & Structured Data",
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
        "title": "Computer Vision",
        "backbones": [
            "timm/convnext_small.fb_in22k_ft_in1k (Modern ConvNet architecture)",
            "timm/swin_base_patch4_window7_224 (Hierarchical Vision Transformer)",
            "timm/eva02_base_patch14_448 (High-resolution ViT)",
        ],
        "losses": ["Cross-Entropy with Label Smoothing (eps=0.1)", "Focal Loss", "Cosine Annealing Schedule"],
        "tokenization": "RandAugment, Mixup (alpha=0.8), CutMix (alpha=1.0), and test-time augmentation (TTA).",
    },
    "time-series": {
        "title": "Time-Series & Forecasting",
        "backbones": [
            "PatchTST (Patching and Channel-Independence Transformer)",
            "DLinear / N-Linear (Fast, interpretable linear decomposition)",
            "LightGBM / CatBoost (Lag features, rolling statistics, calendar encodings)",
        ],
        "losses": ["MAE (L1)", "MSE (L2)", "Quantile Loss for probabilistic intervals"],
        "tokenization": "Rolling window splits strictly expanding or sliding; prevent forward-looking leakages.",
    },
    "generic": {
        "title": "General Machine Learning / Deep Learning",
        "backbones": [
            "Tree Ensembles (LightGBM, XGBoost, Random Forest)",
            "Deep Neural Networks (PyTorch MLP, Transformers)",
        ],
        "losses": ["Task-appropriate objective function (BCE, CrossEntropy, MSE)"],
        "tokenization": "Standard normalization, missing value imputation, and out-of-fold cross-validation.",
    },
}


def load_task_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load task configuration from configs/task.yaml or specified path."""
    path = config_path or TASK_CONFIG_FILE
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


def generate_skill_content(task: Dict[str, Any]) -> str:
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
        "name: ml-experiment-harness",
        f"description: Manages ML experiments for {task_name} on {platform} via AIDE tree search, enforces SWE-agent deterministic audit gate, and mandates Epistemic Grounding & Assumption Registers.",
        "---",
        "",
        f"# ML Experiment Harness Skill — {task_name}",
        "",
        f"> **Active Benchmark**: **{task_name}** | **Platform**: {platform} | **Modality**: {mod_info['title']}",
        f"> **Objective**: {direction.capitalize()} `{primary_metric}` across cross-validation folds.",
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
        lines.extend([
            "",
            "### Data & Cross-Validation Strategy:",
            f"- **Validation Scheme**: `{data_cfg.get('split_strategy', 'KFold')}` ({data_cfg.get('n_folds', 5)} folds, seed {data_cfg.get('seed', 42)})",
            f"- **Anti-Leakage Group Column**: `{data_cfg.get('group_column', 'None (Standard split)')}`",
            f"- **Dataset Paths**: Train: `{data_cfg.get('train_path', 'dataset/')}` | Test: `{data_cfg.get('test_path', 'dataset/test/')}`",
            f"- **Key Columns**: ID: `{data_cfg.get('id_column', 'id')}` | Target: `{data_cfg.get('target_column', 'target')}`",
        ])

    # Competition Constraints & Rules
    comp_rules = task.get("competition_rules", {})
    if comp_rules:
        max_p = comp_rules.get("max_params_limit")
        p_str = f"{max_p / 1e6:.0f}M parameters" if max_p else "No limit"
        lines.extend([
            "",
            "### Competition Rules & Constraints:",
            f"- **Allowed LLM / API**: `{'Yes' if comp_rules.get('allow_llm') else 'No (Local/Open models only)'}`",
            f"- **External Data**: `{'Allowed' if comp_rules.get('allow_external_data') else 'Forbidden'}`",
            f"- **Pretrained Weights**: `{'Allowed' if comp_rules.get('allow_pretrained_models', True) else 'Forbidden'}`",
            f"- **Parameter Ceiling**: `{p_str}`",
            f"- **Daily Submission Quota**: `{comp_rules.get('daily_submission_limit', 'Unlimited')} submissions / day`",
        ])
        if comp_rules.get("deadline"):
            lines.append(f"- **Competition Deadline**: `{comp_rules['deadline']}`")

    # Submission Specification
    sub_spec = task.get("submission_spec", {})
    if sub_spec:
        lines.extend([
            "",
            "### Submission & Audit Specification:",
            f"- **File Name / Format**: `{sub_spec.get('file_name', 'submission.csv')}` (`{sub_spec.get('archive_format', 'csv')}`)",
            f"- **MMES Effect Size Threshold**: `>= {sub_spec.get('min_diff_pct', 2.0)}%` of validation rows",
            f"- **Ensemble Diversity Ceiling**: Cohen's Kappa `kappa <= {sub_spec.get('max_kappa', 0.85)}`",
            f"- **Ensemble Disagreement Floor**: `>= {sub_spec.get('min_disagreement', 5.0)}%`",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 2. Three-Layer Architecture (`Antigravity IDE` <-> `AgentBridge` <-> `Harness`)",
        "",
        "- **Layer 1 — True Agent Runtime (`Antigravity in IDE`)**: Performs scientific reasoning, hypothesis formulation, and the `Observe -> Decide -> Next Action` loop. The Harness does **not** run a separate internal LLM planner.",
        "- **Layer 2 — Agent Bridge & Context Bridge (`harness/agent_bridge.py` & `tools/agent_bridge_cli.py`)**:",
        "  - Feeds Antigravity a compressed **Context Packet** (`CompetitionState` 8-Phase HUD, Parallel Tracks, `TreeJournal` UCB recommendation, Parent `DIAGNOSTICS` bottlenecks, Idea Graveyard, and active `BridgeSession` budget/usage):",
        "    ```powershell",
        "    python tools/agent_bridge_cli.py context",
        "    ```",
        "  - Tracks stateful **`BridgeSession` (`session_XXXX`)** (`goal`, `experiments_run`, `compute_usage`, `consecutive_rejections`, optional `max_experiments` / `max_gpu_hours` / `max_cloud_runs` budgets) and enforces **`USER_APPROVAL` governance** on sensitive actions (`prepare_submission`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`).",
        "  - Exposes **Typed Harness Actions** returning structured `ActionObservation` JSON payloads (`metric_delta`, `composite_score`, `diagnostics_bottlenecks`, `new_blockers`):",
        "    ```powershell",
        "    python tools/agent_bridge_cli.py start-session --goal \"Improve target metric\" --max-experiments 3",
        "    python tools/agent_bridge_cli.py act --action <typed_action> --params '{\"...\": \"...\"}'",
        "    ```",
        "- **Layer 3 — Competition State Machine & Execution Runtime (`harness/competition_state.py`, `train.py`, `harness/cloud_runner.py`)**:",
        "  - **8-Phase Journey**: `1. Setup` -> `2. Understand` -> `3. Baseline Anchor` -> **Parallel Iteration Tracks** (`4. Experiment Loop` || `5. Candidate Management` || `6. External Evaluation`) -> `7. Freeze` -> `8. Closure`.",
        "  - **Execution Backends**: Local (`train.py`), Modal Serverless GPU (`--backend modal`), and Kaggle Kernels CLI (`--backend kaggle`).",
        "",
        "---",
        "",
        "## 3. Core Operational Rules (Enforced & Immutable)",
        "",
        "### 3.1 The Context Protection Rule (Anti-Context Saturation)",
        "- **NEVER** attempt to dump raw fold logs or entire `artifacts/` directories into prompt context.",
        "- **ALWAYS** query the Agent Bridge Context Packet or Solution Tree CLI first:",
        "  ```powershell",
        "  python tools/agent_bridge_cli.py context",
        "  python tools/journal_tool.py --summary",
        "  ```",
        "",
        "### 3.2 The Anti-Pivot & Branch Exploration Rule (AIDE UCB Tree Search)",
        "- **DO NOT** make speculative guesses or pivot arbitrarily without mathematical justification.",
        "- Query the optimal branch recommendation according to Upper Confidence Bound (UCB):",
        "  ```powershell",
        "  python tools/journal_tool.py --recommend",
        "  ```",
        "- Ensure every experiment implements **EXACTLY ONE ATOMIC INTERVENTION** (Control vs Treatment).",
        "",
        "### 3.3 Epistemic Grounding & Assumption Register Protocol (Anti-Hallucination)",
        f"- **MANDATORY RETRIEVAL**: Before formulating a new model configuration for {mod_info['title']}:",
        f"  1. Inspect `docs/KNOWLEDGE_BASE.md` for verified backbones:",
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
        "- Before writing any config, inspect `artifacts/<parent_id>/DIAGNOSTICS.md` (or call `act --action inspect_diagnostics`).",
        "- Declare a `problem_addressed` that directly targets a documented failure slice or confusion mode.",
        "- **Strictly DO NOT pseudo-ensemble** models of the same family or merge previous versions mechanically.",
        "",
        "### 3.5 SWE-Agent Physical & Statistical Audit Gate",
        "- Submissions and candidate predictions CANNOT be approved by natural language alone.",
        "- All submission archives MUST be generated through `tools/audit_gate.py` (or `act --action prepare_submission --user-approved`):",
        "  ```powershell",
        "  python tools/audit_gate.py --candidate <candidate_pred_csv> --baseline <baseline_pred_csv> --output-zip submission.zip",
        "  ```",
        "- **Minimum Meaningful Effect Size (MMES >= 2.0%)**: Candidates differing by < 2% of validation samples are physically rejected.",
        "- **Ensemble Diversity Check**: Constituent models must satisfy **Cohen's Kappa ($\\kappa \\le 0.85$)** and **Disagreement Rate ($\\ge 5.0\\%$)**.",
        "",
        "### 3.6 Approval Governance (`AUTO` / `AUTO_WITH_LIMIT` / `USER_APPROVAL`)",
        "- **AUTO**: `inspect_data_readiness`, `inspect_experiment`, `inspect_diagnostics`, `create_experiment`, `run_smoke_test`",
        "- **AUTO_WITH_LIMIT (Session Budget Tracked)**: `establish_baseline_anchor`, `run_full_validation` (`local` | `modal` | `kaggle`), `synthesize_candidate`, `build_ensemble`",
        "- **USER_APPROVAL (Mandatory Human Confirmation)**: `prepare_submission`, `record_external_score`, `freeze_candidate`, `emergency_unfreeze`, `finalize_competition`",
        "",
        "---",
        "",
        "## 4. Standard Operating Procedure (SOP) for Antigravity in IDE",
        "",
        "1. **Read Bridge Context Packet**: Run `python tools/agent_bridge_cli.py context` to inspect the 8-Phase Competition State, active `BridgeSession`, UCB branch recommendation, and parent `DIAGNOSTICS` bottlenecks.",
        "2. **Start / Configure Session (Optional Budgets)**: Run `python tools/agent_bridge_cli.py start-session --goal \"...\" --max-experiments 3`.",
        "3. **Create Grounded Experiment**: Run `python tools/agent_bridge_cli.py act --action create_experiment --params '{...}'` (or write `configs/exp<NNN>_<branch>_<descriptor>.yaml` and verify via `harness/grounding_gate.py`).",
        "4. **Execute Smoke Test**: Run `python tools/agent_bridge_cli.py act --action run_smoke_test --params '{\"config_path\": \"configs/<exp>.yaml\"}'`.",
        "5. **Execute Full 5-Fold Validation (`local` | `modal` | `kaggle`)**: Run `python tools/agent_bridge_cli.py act --action run_full_validation --params '{\"config_path\": \"configs/<exp>.yaml\", \"backend\": \"local\"}'`.",
        "6. **Observe & Decide Next Step**: Inspect the returned `ActionObservation` (`metric_delta`, `composite_score`, `diagnostics_bottlenecks`, `consecutive_rejections`). Promote winning candidates via `synthesize_candidate` or pivot to the next hypothesis.",
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
    cfg = task_config or load_task_config(config_path)
    content = generate_skill_content(cfg)

    dest = output_path or SKILL_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inject dynamic competition-specific skill into .agents/skills/")
    parser.add_argument("--config", type=str, default="configs/task.yaml", help="Path to task configuration YAML")
    parser.add_argument("--output", type=str, default=str(SKILL_FILE), help="Output path for SKILL.md")
    args = parser.parse_args()

    out_p = inject_skill_file(config_path=Path(args.config), output_path=Path(args.output))
    print(f"✔ Successfully injected dynamic task-specific skill into: {out_p}")
