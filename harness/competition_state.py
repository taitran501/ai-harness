"""Universal Competition State Machine, Spec Adapter & Agent Co-Pilot Engine.

Decouples the universal 8-Phase Competition Journey from benchmark-specific rules:
1. Reads all task rules, subtasks, declarative `data_checks`, `evaluation_protocol`,
   `candidate_strategies`, `final_selection_policy` (N slots), and `closure_outputs`
   from the Competition Spec (`configs/task.yaml`).
2. Supports non-linear parallel iteration tracks (`active_tracks`: `experimentation`,
   `candidate_management`, `external_evaluation`) once the baseline anchor is set.
3. Replaces technical gate codes (`Gate 2->3`) with human-centric action language
   ("Data review required", "Baseline anchor required", "Final selection open").
4. Powers the Agent Co-Pilot ("Agent will: ..." plan + live step-by-step execution
   activity timeline + immutable audit trail for Emergency Unfreeze).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from harness.ensemble_guard import EnsembleGuard
from harness.submission_manager import SubmissionManager
from harness.tree_journal import Node, TreeJournal


def official_primary_value(entry, primary_metric: str):
    """Extract the primary official score from a submission entry.

    ``record_official_score`` writes the primary value under the spec's metric
    name (falling back to ``"score"`` when the spec is unavailable); legacy
    entries may store it under ``"primary"``. Probe all known keys.
    """
    scores = getattr(entry, "official_scores", None) or {}
    for key in (primary_metric, "primary", "score"):
        value = scores.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


@dataclass
class DataAuditConfig:
    """User-configurable settings for Phase 2 Data & Problem Understanding."""

    auto_scan_mode: str = "deep"  # "manual" | "light" | "deep"
    check_label_distribution: bool = True
    check_group_leakage: bool = True
    check_conditional_labels: bool = True
    check_text_length_outliers: bool = True
    check_missing_nulls: bool = True


@dataclass
class AgentContextConfig:
    """User-configurable settings for which Competition State slices are injected into Agent Priming."""

    inject_phase_hud: bool = True
    inject_current_best_anchors: bool = True
    inject_active_blockers: bool = True
    inject_next_action: bool = True
    inject_data_risk_warnings: bool = True
    inject_candidate_pool: bool = True
    inject_submission_quota_and_divergence: bool = True
    inject_idea_graveyard: bool = True


class CompetitionStateMachine:
    """Task-agnostic 8-Phase Competition Journey Engine driven by CompetitionSpec."""

    def __init__(
        self,
        project_root: Path | str = ".",
        task_config_path: Optional[Path | str] = None,
    ):
        from harness.project import HarnessProject

        self.project = HarnessProject.discover(project_root, task_config_path)
        self.project_root = self.project.root
        self.artifacts_dir = self.project.artifacts_dir
        self.submissions_dir = self.artifacts_dir / "submissions"
        self.state_file = self.artifacts_dir / "competition_state.json"
        self.data_report_file = self.artifacts_dir / "DATA_READINESS_REPORT.json"
        self.freeze_lock_file = self.submissions_dir / "FROZEN_FINAL_LOCK.json"
        self.final_report_file = self.artifacts_dir / "FINAL_COMPETITION_REPORT.md"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        self.spec: Dict[str, Any] = self.load_competition_spec()
        self.state: Dict[str, Any] = self._load_or_init_state()
        import copy

        self._state_snapshot = copy.deepcopy(self.state)

    def load_competition_spec(self) -> Dict[str, Any]:
        """Load the declarative Competition Spec from configs/task.yaml."""
        spec_path = self.project.task_config_path
        if spec_path.exists():
            try:
                raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
                return raw.get("task", raw)
            except Exception:
                pass
        return {
            "name": "ML Competition Benchmark",
            "primary_metric": "score",
            "direction": "maximize",
            "primary_metric_baseline": 0.0,
            "primary_metric_target": 1.0,
            "subtasks": [],
            "evaluation_protocol": {
                "evaluation_mode": "public_private_leaderboard",
                "external_eval_label": "Public Leaderboard",
                "final_eval_label": "Private Leaderboard",
                "has_public_leaderboard": True,
                "has_private_leaderboard": True,
                "daily_quota": 5,
            },
            "data_checks": [],
            "candidate_strategies": [
                {
                    "id": "single_model",
                    "label": "Single Model Candidate",
                    "enabled": True,
                },
                {
                    "id": "ensemble_blend",
                    "label": "Diverse Ensemble Blend",
                    "enabled": True,
                },
            ],
            "final_selection_policy": {
                "max_candidates": 2,
                "selection_strategy": "risk_balanced_portfolio",
                "slots": [
                    {
                        "id": "slot_1",
                        "label": "Primary Safe Candidate (Stable Validation)",
                    },
                    {
                        "id": "slot_2",
                        "label": "Secondary Candidate (Ensemble / Specialist)",
                    },
                ],
            },
        }

    def get_phases_meta(self) -> List[Dict[str, Any]]:
        """Build the 8-Phase metadata dynamically using labels from CompetitionSpec."""
        eval_proto = self.spec.get("evaluation_protocol", {})
        ext_label = eval_proto.get("external_eval_label", "External Evaluation")
        final_label = eval_proto.get("final_eval_label", "Private Evaluation")
        max_cands = int(
            self.spec.get("final_selection_policy", {}).get("max_candidates", 2)
        )

        return [
            {
                "index": 1,
                "id": "setup",
                "title": "1. Competition Setup",
                "short_title": "Setup",
                "question": f"Harness đã hiểu quy tắc & cấu trúc của '{self.spec.get('name', 'Competition')}' như thế nào?",
            },
            {
                "index": 2,
                "id": "understand",
                "title": "2. Data & Problem Understanding",
                "short_title": "Understand",
                "question": "Dữ liệu hiện tại có rủi ro rò rỉ hoặc mất cân bằng nào cần xác nhận?",
            },
            {
                "index": 3,
                "id": "baseline",
                "title": "3. Baseline Anchor",
                "short_title": "Baseline",
                "question": f"Điểm xuất phát chuẩn ({self.spec.get('primary_metric', 'Score')}) để làm mỏ neo là gì?",
            },
            {
                "index": 4,
                "id": "experiment",
                "title": "4. Experiment Loop",
                "short_title": "Experiment",
                "question": "Ta đang thử gì? So với cái gì? Kết quả thế nào? Tiếp theo làm gì?",
            },
            {
                "index": 5,
                "id": "select",
                "title": "5. Candidate Management",
                "short_title": "Candidates",
                "question": "Ứng viên đơn lẻ hoặc tổ hợp nào đủ tiêu chuẩn để mang đi đánh giá?",
            },
            {
                "index": 6,
                "id": "public_lb",
                "title": f"6. {ext_label}",
                "short_title": ext_label,
                "question": f"Điểm Validation nội bộ có đồng nhất với {ext_label} không?",
            },
            {
                "index": 7,
                "id": "freeze",
                "title": f"7. Freeze & {final_label}",
                "short_title": "Freeze",
                "question": f"Chốt danh mục tối đa {max_cands} ứng viên nào để khóa cho {final_label}?",
            },
            {
                "index": 8,
                "id": "finalization",
                "title": "8. Competition Closure",
                "short_title": "Closure",
                "question": "Kết quả chung cuộc, phả hệ chiến thắng và gói tái lập (Reproducibility) là gì?",
            },
        ]

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": "3.0_universal",
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": "EXPLORE",  # EXPLORE | OPTIMIZE | FROZEN | FINALIZED
            "manual_phase_override": None,
            "gates": {
                "spec_confirmed": False,
                "data_risks_confirmed": False,
                "baseline_locked": False,
                "baseline_exp_id": self.spec.get("baseline_anchor", {}).get(
                    "default_exp_id"
                )
                or "baseline",
                "baseline_evidence": None,
                "freeze_locked": False,
                "finalized": False,
            },
            "data_audit_config": asdict(DataAuditConfig()),
            "agent_context_config": asdict(AgentContextConfig()),
            "candidate_pool": {
                "pinned_candidates": [],
                "synthesized_candidates": [],
            },
            "freeze_selection": {
                "selected_slots": {},
                "slot_1_safe_cv": None,
                "slot_2_aggressive_lb": None,
                "frozen_at": None,
                "freeze_notes": "",
                "lock_receipt_path": None,
            },
            "audit_trail": [],
            "agent_copilot": {
                "last_run_at": None,
                "status": "IDLE",  # IDLE | RUNNING | COMPLETED
                "current_action_title": None,
                "steps": [],
                "history": [],
            },
            "finalization": {
                "private_score_slot_1": None,
                "private_score_slot_2": None,
                "winning_slot": "slot_1",
                "final_rank": None,
                "total_teams": None,
                "lessons_learned": "",
                "paper_title": f"{self.spec.get('name', 'Benchmark')} System Description & Reproducibility Report",
                "report_generated_at": None,
            },
        }

    def _load_or_init_state(self) -> Dict[str, Any]:
        default = self._default_state()
        if not self.state_file.exists():
            from harness.state_store import transaction

            with transaction(self.state_file, default) as current:
                default = dict(current)
            return default
        try:
            loaded = json.loads(self.state_file.read_text(encoding="utf-8"))
            for k, v in default.items():
                if k not in loaded:
                    loaded[k] = v
                elif isinstance(v, dict) and isinstance(loaded[k], dict):
                    for sk, sv in v.items():
                        if sk not in loaded[k]:
                            loaded[k][sk] = sv
            return loaded
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Required competition state is unreadable") from error

    def _save_raw(self, data: Dict[str, Any]) -> None:
        data["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        from harness.io_utils import atomic_write_json

        atomic_write_json(self.state_file, data, indent=2, ensure_ascii=False)

    def save(self) -> None:
        import copy
        from harness.state_store import save_snapshot

        self.state = save_snapshot(
            self.state_file, getattr(self, "_state_snapshot", {}), self.state
        )
        self._state_snapshot = copy.deepcopy(self.state)

    def _append_audit_event(
        self, event_type: str, summary: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        trail = self.state.setdefault("audit_trail", [])
        trail.insert(
            0,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "event_type": event_type,
                "summary": summary,
                "details": details or {},
            },
        )
        self.state["audit_trail"] = trail

    # =========================================================================
    # PHASE 2: DECLARATIVE SPEC-DRIVEN DATA CHECKS
    # =========================================================================
    def run_data_readiness_audit(
        self, custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run declarative data checks defined in CompetitionSpec (`configs/task.yaml`)."""
        if custom_config:
            self.state["data_audit_config"].update(custom_config)
            self.save()

        cfg_values = self.state.get("data_audit_config", {})
        cfg = DataAuditConfig(
            **{
                key: value
                for key, value in cfg_values.items()
                if key in DataAuditConfig.__dataclass_fields__
            }
        )
        declared_checks = self.spec.get("data_checks", [])
        data_cfg = self.spec.get("data_config", {})
        group_col = data_cfg.get("group_column") or "group"

        report: Dict[str, Any] = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config_used": asdict(cfg),
            "declared_checks_count": len(declared_checks),
            "dataset_overview": {},
            "checks_performed": [],
            "warnings": [],
            "status": "READY",
        }

        if cfg.auto_scan_mode == "manual":
            report["status"] = "MANUAL_MODE"
            report["warnings"].append(
                {
                    "id": "MANUAL_SCAN_MODE",
                    "severity": "info",
                    "title": "Chế độ rà soát dữ liệu thủ công đang bật",
                    "detail": "Bạn có thể chạy kiểm tra tự động theo Competition Spec hoặc bấm xác nhận trực tiếp.",
                }
            )
            from harness.io_utils import atomic_write_json

            atomic_write_json(
                self.data_report_file, report, indent=2, ensure_ascii=False
            )
            return report

        try:
            merged = None
            try:
                from harness.task_spec import load_active_task_spec

                spec = load_active_task_spec(self.project.task_config_path)
                merged = spec.load_dataset_split(
                    "train", project_root=self.project_root
                )
            except Exception:
                pass

            if merged is None:
                data_dir = self.project_root / data_cfg.get("train_path", "dataset")
                if data_dir.is_file():
                    sep = "\t" if data_dir.suffix == ".tsv" else ","
                    merged = pd.read_csv(data_dir, sep=sep)
                elif data_dir.is_dir():
                    dfs = []
                    for f in sorted(data_dir.glob("*.tsv")) + sorted(
                        data_dir.glob("*.csv")
                    ):
                        sep = "\t" if f.suffix == ".tsv" else ","
                        try:
                            dfs.append(pd.read_csv(f, sep=sep))
                        except Exception:
                            pass
                    if dfs:
                        merged = pd.concat(dfs, ignore_index=True)

            if merged is None:
                merged = pd.DataFrame()

            split_path = self.project_root / "splits" / "fold_assignments.tsv"
            if split_path.exists() and "fold" not in merged.columns and len(merged) > 0:
                try:
                    splits_df = pd.read_csv(split_path, sep="\t")
                    common_cols = [
                        c
                        for c in merged.columns
                        if c in splits_df.columns and c != "fold"
                    ]
                    if common_cols:
                        merged = merged.merge(
                            splits_df[common_cols + ["fold"]],
                            on=common_cols[0],
                            how="left",
                        )
                except Exception:
                    pass

            data_cfg = self.spec.get("data_config") or self.spec.get("data", {})
            partition_col = data_cfg.get("partition_column") or self.spec.get(
                "partition_column"
            )
            partition_counts = (
                merged[partition_col].value_counts().to_dict()
                if partition_col and partition_col in merged.columns
                else {}
            )
            fold_counts = (
                merged["fold"].value_counts().sort_index().to_dict()
                if "fold" in merged.columns
                else {}
            )
            report["dataset_overview"] = {
                "total_rows": int(len(merged)),
                "partition_column": partition_col,
                "partitions": {str(k): int(v) for k, v in partition_counts.items()},
                "folds": {str(k): int(v) for k, v in fold_counts.items()},
                "unique_groups": (
                    int(merged[group_col].nunique())
                    if group_col in merged.columns
                    else 0
                ),
                "group_column": group_col,
            }

            # Execute declarative checks from CompetitionSpec
            for chk in declared_checks:
                chk_type = chk.get("type")
                chk_id = chk.get("id", chk_type)

                if chk_type == "group_leakage" and cfg.check_group_leakage:
                    g_key = chk.get("group_key", group_col)
                    if g_key in merged.columns and "fold" in merged.columns:
                        report["checks_performed"].append(chk_id)
                        leaked = 0
                        for f_idx in sorted(merged["fold"].unique()):
                            tr_g = set(
                                merged[merged["fold"] != f_idx][g_key].dropna().unique()
                            )
                            va_g = set(
                                merged[merged["fold"] == f_idx][g_key].dropna().unique()
                            )
                            leaked += len(tr_g.intersection(va_g))
                        report["leakage_audit"] = {
                            "group_column": g_key,
                            "cross_fold_leaked_groups": leaked,
                            "passed": leaked == 0,
                        }
                        if leaked == 0:
                            report["warnings"].append(
                                {
                                    "id": chk_id,
                                    "severity": "info",
                                    "title": f"{chk.get('title', 'Group Split Verification')}: 0% rò rỉ nhóm (`{g_key}`)",
                                    "detail": f"Đã xác minh {report['dataset_overview']['unique_groups']} nhóm `{g_key}` hoàn toàn tách biệt giữa các fold.",
                                }
                            )
                        else:
                            report["status"] = "BLOCKED"
                            report["warnings"].append(
                                {
                                    "id": chk_id,
                                    "severity": "critical",
                                    "title": f"Phát hiện {leaked} nhóm `{g_key}` bị rò rỉ giữa Train và Validation!",
                                    "detail": "Rebuild the validation folds using the task's declared group-aware split strategy before training.",
                                }
                            )

                elif chk_type == "conditional_label" and cfg.check_conditional_labels:
                    if_col = chk.get("if_column")
                    if_val = chk.get("if_equals")
                    req_col = chk.get("require_column")
                    req_val = chk.get("require_value")
                    if if_col in merged.columns and req_col in merged.columns:
                        report["checks_performed"].append(chk_id)
                        mask = merged[if_col] == if_val
                        violations = int((merged[mask][req_col] != req_val).sum())
                        active_pct = round(float((~mask).mean()) * 100, 2)
                        report.setdefault("label_distribution", {})[
                            "active_conditioned_pct"
                        ] = active_pct
                        report["warnings"].append(
                            {
                                "id": chk_id,
                                "severity": "warning",
                                "title": f"{chk.get('title', 'Conditional Target Rule')}: `{if_col} == '{if_val}' → {req_col} = '{req_val}'`",
                                "detail": (
                                    f"Chỉ {active_pct}% mẫu thỏa điều kiện đánh giá của nhánh phụ (`{if_col} != '{if_val}'`). "
                                    f"Số dòng vi phạm trong tập gốc: {violations}."
                                ),
                            }
                        )

                elif chk_type == "label_sparsity" and cfg.check_label_distribution:
                    f_col = chk.get("filter_column")
                    f_neq = chk.get("filter_not_equals")
                    prefix = chk.get("prefix", "label_")
                    thresh = float(chk.get("min_pct_threshold", 5.0))
                    sub_df = (
                        merged[merged[f_col] != f_neq]
                        if (f_col and f_col in merged.columns)
                        else merged
                    )
                    pref_cols = [c for c in sub_df.columns if c.startswith(prefix)]
                    if pref_cols:
                        report["checks_performed"].append(chk_id)
                        rates = {
                            c.replace(prefix, ""): round(
                                float(sub_df[c].mean()) * 100, 2
                            )
                            for c in pref_cols
                        }
                        rare = [k for k, v in rates.items() if v < thresh]
                        if rare:
                            report["warnings"].append(
                                {
                                    "id": chk_id,
                                    "severity": "critical",
                                    "title": f"{chk.get('title', 'Label Sparsity Alert')}: {len(rare)}/{len(pref_cols)} nhãn xuất hiện < {thresh}%",
                                    "detail": (
                                        f"Rare labels {rare} occur below {thresh}%. Check whether the task's official metric is sensitive to this sparsity and choose an intervention supported by the task contract."
                                    ),
                                }
                            )

                elif (
                    chk_type == "context_length_ratio"
                    and cfg.check_text_length_outliers
                    and cfg.auto_scan_mode == "deep"
                ):
                    p_col = chk.get("primary_text_column", "text")
                    c_col = chk.get("context_column", "yt_description")
                    max_r = float(chk.get("max_ratio", 3.0))
                    if p_col in merged.columns and c_col in merged.columns:
                        report["checks_performed"].append(chk_id)
                        p_med = float(
                            merged[p_col].fillna("").astype(str).str.len().median()
                        )
                        c_med = float(
                            merged[c_col].fillna("").astype(str).str.len().median()
                        )
                        if p_med > 0 and (c_med / p_med) > max_r:
                            report["warnings"].append(
                                {
                                    "id": chk_id,
                                    "severity": "warning",
                                    "title": f"{chk.get('title', 'Context Dilution Risk')}: `{c_col}` dài gấp {c_med / p_med:.1f}x `{p_col}`",
                                    "detail": (
                                        f"Độ dài trung vị của `{c_col}` ({int(c_med)} ký tự) lớn hơn nhiều so với `{p_col}` ({int(p_med)} ký tự), "
                                        f"có nguy cơ làm loãng tín hiệu nếu nối trực tiếp cho toàn bộ các subtask."
                                    ),
                                }
                            )

        except Exception as e:
            report["status"] = "WARN"
            report["error"] = str(e)

        from harness.io_utils import atomic_write_json

        atomic_write_json(self.data_report_file, report, indent=2, ensure_ascii=False)
        return report

    def get_data_readiness_report(self) -> Dict[str, Any]:
        if not self.data_report_file.exists():
            return self.run_data_readiness_audit()
        try:
            return json.loads(self.data_report_file.read_text(encoding="utf-8"))
        except Exception:
            return self.run_data_readiness_audit()

    # =========================================================================
    # PHASE 5: UNIVERSAL CANDIDATE MANAGEMENT POOL & STRATEGIES
    # =========================================================================
    def compute_candidate_pool(self, journal: TreeJournal) -> Dict[str, Any]:
        """Classify completed runs and list supported candidate synthesis strategies from Spec."""
        completed = [
            n
            for n in journal.nodes.values()
            if not n.is_buggy and n.status == "COMPLETED" and n.verified and n.eligible
        ]
        baseline_id = (
            self.state["gates"].get("baseline_exp_id")
            or self.spec.get("baseline_anchor", {}).get("default_exp_id")
            or "baseline"
        )
        baseline_node = next((n for n in completed if n.id == baseline_id), None)
        base_score = (
            baseline_node.get_score()
            if baseline_node
            else self.spec.get("primary_metric_baseline")
        )

        candidates_list: List[Dict[str, Any]] = []
        criteria = self.spec.get("evaluation_criteria", {})
        if not criteria:
            declared_subtasks = self.spec.get("subtasks", [])
            if isinstance(declared_subtasks, dict):
                criteria = declared_subtasks
            elif isinstance(declared_subtasks, list):
                criteria = {
                    item.get("id"): item
                    for item in declared_subtasks
                    if isinstance(item, dict) and item.get("id")
                }

        for n in completed:
            score = n.get_score()
            subtasks_scores = {}

            # 1. Dynamically extract declared criteria
            for crit_key, crit_meta in criteria.items():
                if crit_key in n.metrics.get("subtasks", {}):
                    subtasks_scores[crit_key] = float(
                        n.metrics["subtasks"][crit_key]["score"]
                    )
                    continue
                val = None
                if isinstance(n.metrics.get(crit_key), (int, float)):
                    val = float(n.metrics[crit_key])
                elif isinstance(n.metrics.get(crit_key), dict):
                    metric_key = (
                        str(
                            crit_meta.get("metric_key")
                            or crit_meta.get("primary_metric")
                            or "score"
                        )
                        if isinstance(crit_meta, dict)
                        else "score"
                    )
                    metric_value = n.metrics[crit_key].get(metric_key)
                    if metric_value is None:
                        metric_value = next(
                            (
                                n.metrics[crit_key][key]
                                for key in ("score", "macro_f1", "f1", "accuracy")
                                if key in n.metrics[crit_key]
                            ),
                            None,
                        )
                    val = None if metric_value is None else float(metric_value)
                if val is not None:
                    subtasks_scores[crit_key] = round(val, 4)

            fold_std = None
            exp_dir = (
                (self.project_root / n.artifact_dir).resolve()
                if n.artifact_dir
                else (self.artifacts_dir / n.id).resolve()
            )
            summary_csv = exp_dir / "summary.csv"
            oof_csv = exp_dir / "oof_predictions.csv"
            if summary_csv.exists():
                try:
                    sdf = pd.read_csv(summary_csv)
                    metric_cols = [
                        c
                        for c in sdf.columns
                        if c.endswith("_f1")
                        or c.endswith("_macro_f1")
                        or c in {"score", "primary_score"}
                    ]
                    if metric_cols:
                        fold_std = round(float(sdf[metric_cols[0]].std()), 4)
                except Exception:
                    pass

            delta = (
                None
                if base_score is None
                else round(
                    (score - base_score) * (1 if n.direction == "maximize" else -1), 4
                )
            )
            verdict = (
                "UNANCHORED"
                if delta is None
                else ("KEEP" if delta >= -0.005 else "REJECT")
            )

            cand_entry = {
                "id": n.id,
                "branch": n.branch,
                "parent_id": n.parent_id,
                "model": n.metrics.get("model", n.model_family),
                "composite_score": round(score, 4),
                "utility": n.get_utility(),
                "verified": n.verified,
                "eligible": n.eligible,
                "direction": n.direction,
                "delta_vs_baseline": delta,
                "subtasks": subtasks_scores,
                "fold_std": fold_std,
                "has_oof": oof_csv.exists(),
                "oof_path": (
                    str(oof_csv.relative_to(self.project_root)).replace("\\", "/")
                    if oof_csv.exists()
                    else None
                ),
                "verdict": verdict,
                "is_pinned": n.id
                in self.state["candidate_pool"].get("pinned_candidates", []),
            }
            candidates_list.append(cand_entry)

        candidates_list.sort(key=lambda x: x["utility"], reverse=True)

        overall_best = candidates_list[0] if candidates_list else None
        specialists = {}
        subtask_meta = self.spec.get("subtasks", {}) or {}
        # `subtasks` may be a list of dicts (with an `id`/`column` key) or a dict
        # keyed by subtask id. Build a lookup by subtask key regardless of shape.
        sub_directions: Dict[str, str] = {}
        if isinstance(subtask_meta, dict):
            for k, v in subtask_meta.items():
                if isinstance(v, dict):
                    sub_directions[k] = v.get("direction") or self.spec.get("direction", "maximize")
        elif isinstance(subtask_meta, list):
            for item in subtask_meta:
                if isinstance(item, dict):
                    key = item.get("id") or item.get("column") or item.get("name")
                    if key:
                        sub_directions[str(key)] = item.get("direction") or self.spec.get("direction", "maximize")
        default_dir = self.spec.get("direction", "maximize")
        for crit_key in criteria.keys():
            matching = [c for c in candidates_list if crit_key in c.get("subtasks", {})]
            if matching:
                # Use the per-subtask direction when declared; fall back to the
                # aggregate spec direction. This keeps ranking correct when a
                # subtask is minimized while the overall metric is maximized.
                sign = 1 if sub_directions.get(str(crit_key), default_dir) == "maximize" else -1
                specialists[crit_key] = max(
                    matching,
                    key=lambda x: x["subtasks"][crit_key] * sign,
                )
        stable_candidates = sorted(
            [c for c in candidates_list if c["verdict"] == "KEEP"],
            key=lambda x: (
                x["fold_std"] if x["fold_std"] is not None else float("inf"),
                -x["utility"],
            ),
        )[:3]
        rejected = [c for c in candidates_list if c["verdict"] == "REJECT"]

        return {
            "baseline_id": baseline_id,
            "baseline_score": None if base_score is None else round(base_score, 4),
            "available_strategies": self.spec.get(
                "candidate_strategies",
                [
                    {
                        "id": "single_model",
                        "label": "Single Model Promotion",
                        "enabled": True,
                    },
                    {
                        "id": "ensemble_blend",
                        "label": "Diverse Ensemble Blend",
                        "enabled": True,
                    },
                    {
                        "id": "specialist_routing",
                        "label": "Subtask Specialist Routing",
                        "enabled": True,
                    },
                ],
            ),
            "overall_best": overall_best,
            "subtask_specialists": specialists,
            "stable_candidates": stable_candidates,
            "rejected_experiments": rejected,
            "all_candidates": candidates_list,
            "synthesized_candidates": self.state["candidate_pool"].get(
                "synthesized_candidates", []
            ),
        }

    def synthesize_candidate(
        self,
        name: str,
        mode: str,  # "single_model" | "specialist_router" | "ensemble_vote" | "checkpoint_selection"
        source_map: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a verified candidate artifact in Phase 5 according to the selected strategy."""
        from harness.candidates import build_candidate

        entry = build_candidate(self.project, name, mode, source_map)
        self.state["candidate_pool"].setdefault("synthesized_candidates", []).insert(
            0, entry
        )
        self._append_audit_event(
            "CANDIDATE_SYNTHESIZED", f"Created verified candidate {entry['id']}", entry
        )
        self.save()
        return entry

    # =========================================================================
    # PHASE 7: DYNAMIC N-SLOT FINAL SELECTION POLICY & FREEZE AUDIT TRAIL
    # =========================================================================
    def freeze_final_candidates(
        self,
        slot_1_id: str,
        slot_2_id: Optional[str] = None,
        selected_slots: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Lock up to N final candidates according to CompetitionSpec.final_selection_policy."""
        policy = self.spec.get("final_selection_policy", {})
        max_cands = int(policy.get("max_candidates", 2))
        slots_map = selected_slots or {"slot_1": slot_1_id}
        if slot_2_id:
            slots_map["slot_2"] = slot_2_id

        if (
            not slots_map
            or len(slots_map) > max_cands
            or any(not v for v in slots_map.values())
            or len(set(slots_map.values())) != len(slots_map)
        ):
            raise ValueError(
                "Freeze requires unique eligible candidate IDs within the slot limit"
            )
        from harness.candidates import validate_candidate

        entries = {
            c["id"]: c
            for c in self.state["candidate_pool"].get("synthesized_candidates", [])
        }
        for candidate in slots_map.values():
            if candidate not in entries:
                raise ValueError("Unknown candidate: " + candidate)
            validate_candidate(self.project, entries[candidate])
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.project_root,
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_commit = "UNKNOWN_COMMIT"

        lock_payload = {
            "freeze_version": "2.0_policy_driven",
            "status": "FROZEN_FOR_FINAL_EVALUATION",
            "max_candidates_allowed": max_cands,
            "selection_strategy": policy.get(
                "selection_strategy", "risk_balanced_portfolio"
            ),
            "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": git_commit,
            "selected_slots": slots_map,
            "slot_1_safe_cv": {"id": slot_1_id},
            "slot_2_aggressive_lb": {"id": slot_2_id if max_cands >= 2 else None},
            "notes": notes,
        }
        from harness.io_utils import atomic_write_json

        atomic_write_json(
            self.freeze_lock_file, lock_payload, indent=2, ensure_ascii=False
        )

        self.state["mode"] = "FROZEN"
        self.state["gates"]["freeze_locked"] = True
        self.state["freeze_selection"] = {
            "selected_slots": slots_map,
            "slot_1_safe_cv": slot_1_id,
            "slot_2_aggressive_lb": slot_2_id if max_cands >= 2 else None,
            "frozen_at": lock_payload["frozen_at"],
            "freeze_notes": notes,
            "lock_receipt_path": str(
                self.freeze_lock_file.relative_to(self.project_root)
            ).replace("\\", "/"),
            "git_commit": git_commit,
        }
        self._append_audit_event(
            "COMPETITION_FROZEN",
            f"Locked {len(slots_map)}/{max_cands} final candidate slots at commit {git_commit[:8]}",
            lock_payload,
        )
        self.save()
        return lock_payload

    def emergency_unfreeze(
        self, reason: str = "User requested emergency unfreeze"
    ) -> Dict[str, Any]:
        """Unlock Phase 7 Freeze mode and record a permanent cryptographic audit trail entry."""
        if not reason or not reason.strip():
            raise ValueError("Unfreeze requires a reason")
        self.state["mode"] = "OPTIMIZE"
        self.state["gates"]["freeze_locked"] = False
        self.state["gates"]["finalized"] = False
        self.state["freeze_selection"]["unfreeze_reason"] = reason
        if self.freeze_lock_file.exists():
            try:
                self.freeze_lock_file.unlink()
            except Exception:
                pass
        self._append_audit_event(
            "EMERGENCY_UNFREEZE",
            f"Freeze lock broken! Reason: {reason}",
            {
                "reason": reason,
                "previous_freeze": self.state["freeze_selection"].get("frozen_at"),
            },
        )
        self.save()
        return self.state["freeze_selection"]

    # =========================================================================
    # AGENT CO-PILOT EXECUTION ENGINE ("RUN WITH AGENT")
    # =========================================================================
    def execute_agent_copilot_action(
        self,
        journal: TreeJournal,
        sub_mgr: SubmissionManager,
        task_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility entrypoint: all mutations use the canonical action layer."""
        from harness.agent_bridge import AgentBridge

        bridge = AgentBridge(self.project_root)
        snapshot = self.evaluate_journey_snapshot(journal, sub_mgr, task_info)
        recommendation = snapshot["hud"]["what_should_i_do_next"]
        action_id = recommendation.get("action_id")
        action = {
            "CONFIRM_TASK_SPEC": "confirm_task_spec",
            "CONFIRM_DATA_RISKS": "inspect_data_readiness",
            "LOCK_BASELINE_ANCHOR": "establish_baseline_anchor",
        }.get(action_id, "inspect_diagnostics")
        params = {}
        if action == "establish_baseline_anchor":
            params["exp_id"] = self.state["gates"].get("baseline_exp_id") or (
                snapshot.get("candidate_pool", {}).get("baseline_id")
            )
            if not params["exp_id"]:
                raise ValueError("Choose an explicit verified baseline experiment")
        _, observation = bridge.execute_action(action=action, params=params)
        refreshed = CompetitionState(project_root=self.project_root)
        return {
            **refreshed.evaluate_journey_snapshot(journal, sub_mgr, task_info),
            "observation": observation.__dict__,
        }

    # =========================================================================
    # PHASE 8: COMPETITION CLOSURE & REPRODUCIBILITY PACKAGE
    # =========================================================================
    def finalize_competition(
        self, payload: Dict[str, Any], journal: TreeJournal, sub_mgr: SubmissionManager
    ) -> Dict[str, Any]:
        """Record final evaluation outcomes and generate Closure & Reproducibility outputs."""
        fin = self.state["finalization"]
        for k in [
            "private_score_slot_1",
            "private_score_slot_2",
            "winning_slot",
            "final_rank",
            "total_teams",
            "lessons_learned",
            "paper_title",
        ]:
            if k in payload and payload[k] is not None:
                fin[k] = payload[k]

        self.state["mode"] = "FINALIZED"
        self.state["gates"]["finalized"] = True
        fin["report_generated_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        pool = self.compute_candidate_pool(journal)
        best_cand = pool.get("overall_best") or {}
        slot1_id = (
            self.state["freeze_selection"].get("slot_1_safe_cv")
            or best_cand.get("id")
            or self.state["gates"].get("baseline_exp_id")
            or "baseline"
        )
        slot2_id = self.state["freeze_selection"].get("slot_2_aggressive_lb") or "none"

        public_scores = [
            s.official_scores.get("primary")
            for s in sub_mgr.submissions.values()
            if s.official_scores.get("primary") is not None
        ]
        best_pub = max(public_scores) if public_scores else None
        cv_score = best_cand.get(
            "composite_score", self.spec.get("primary_metric_baseline")
        )
        priv_1 = fin.get("private_score_slot_1")
        priv_2 = fin.get("private_score_slot_2")
        rank_str = (
            f"#{fin.get('final_rank')} / {fin.get('total_teams')}"
            if fin.get("final_rank")
            else "Pending Official Release"
        )

        lines = [
            f"# {fin.get('paper_title', 'Competition Closure & Reproducibility Report')}",
            "",
            f"> **Generated At**: `{fin['report_generated_at']}` | **Status**: `FINALIZED`",
            "",
            "## 1. Four-Tier Evaluation Comparison (`Local CV ≠ External/Public ≠ Final/Private ≠ Rank`)",
            "",
            "| Portfolio Slot | Candidate ID | Local Validation | External / Public Score | Final / Private Score | Competition Rank |",
            "|---|---|---:|---:|---:|---|",
            f"| **Slot 1 (Primary Safe)** | `{slot1_id}` | `{cv_score:.4f}` | `{best_pub if best_pub is not None else 'N/A'}` | `{priv_1 if priv_1 is not None else 'Pending'}` | **{rank_str}** |",
            f"| **Slot 2 (Secondary)** | `{slot2_id}` | `{cv_score:.4f}` | `{best_pub if best_pub is not None else 'N/A'}` | `{priv_2 if priv_2 is not None else 'Pending'}` | **{rank_str}** |",
            "",
            "## 2. Full Experiment Ablation Study",
            "",
            "| Experiment ID | Branch | Model | Validation Score | Delta vs Baseline | Verdict |",
            "|---|---|---|---:|---:|---|",
        ]
        for c in pool.get("all_candidates", []):
            lines.append(
                f"| `{c['id']}` | `{c['branch']}` | {c['model']} | **{c['composite_score']:.4f}** | `{c['delta_vs_baseline']:+.4f}` | `{c['verdict']}` |"
            )

        lines.extend(
            [
                "",
                "## 3. Lessons Learned & Post-Mortem",
                "",
                fin.get("lessons_learned")
                or "Recorded via Competition Closure workspace.",
            ]
        )
        report_md = "\n".join(lines)
        self.final_report_file.write_text(report_md, encoding="utf-8")
        self._append_audit_event(
            "COMPETITION_FINALIZED",
            "Generated Competition Closure & Reproducibility Report",
        )
        self.save()
        return {
            "finalization": fin,
            "report_markdown": report_md,
            "report_path": str(
                self.final_report_file.relative_to(self.project_root)
            ).replace("\\", "/"),
        }

    # =========================================================================
    # CORE STATE MACHINE SNAPSHOT (ACTIVE TRACKS + ACTION LANGUAGE + CO-PILOT)
    # =========================================================================
    def evaluate_journey_snapshot(
        self,
        journal: TreeJournal,
        sub_mgr: SubmissionManager,
        task_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute the universal Competition Journey snapshot, Active Tracks, and Agent Co-Pilot Plan."""
        self.spec = self.load_competition_spec()
        phases_meta = self.get_phases_meta()
        gates = self.state["gates"]
        pool = self.compute_candidate_pool(journal)
        all_cands = pool["all_candidates"]
        synth_cands = pool["synthesized_candidates"]
        subs = list(sub_mgr.submissions.values())
        data_report = self.get_data_readiness_report()

        eval_proto = self.spec.get("evaluation_protocol", {})
        has_public_lb = bool(eval_proto.get("has_public_leaderboard", True))
        daily_quota = int(eval_proto.get("daily_quota", 5))
        max_final_slots = int(
            self.spec.get("final_selection_policy", {}).get("max_candidates", 2)
        )

        has_baseline = len(all_cands) >= 1
        has_submissions = len(subs) > 0

        if gates.get("finalized"):
            derived_phase = 8
        elif gates.get("freeze_locked"):
            derived_phase = 7
        elif not gates.get("spec_confirmed"):
            derived_phase = 1
        elif not gates.get("data_risks_confirmed"):
            derived_phase = 2
        elif not gates.get("baseline_locked") or not has_baseline:
            derived_phase = 3
        elif has_submissions:
            derived_phase = 6
        elif (
            len(synth_cands) > 0
            or len(self.state["candidate_pool"].get("pinned_candidates", [])) > 0
        ):
            derived_phase = 5
        else:
            derived_phase = 4

        active_phase_idx = self.state.get("manual_phase_override") or derived_phase
        active_phase_meta = next(
            (p for p in phases_meta if p["index"] == active_phase_idx), phases_meta[3]
        )

        # Non-Linear Parallel Iteration Tracks (Phases 4, 5, 6 run concurrently once Baseline is locked)
        baseline_ready = bool(
            gates.get("data_risks_confirmed") and gates.get("baseline_locked")
        )
        is_frozen = bool(gates.get("freeze_locked") or gates.get("finalized"))
        active_tracks = {
            "experimentation": (
                "frozen" if is_frozen else ("active" if baseline_ready else "locked")
            ),
            "candidate_management": (
                "frozen" if is_frozen else ("active" if baseline_ready else "locked")
            ),
            "external_evaluation": (
                "disabled"
                if not has_public_lb
                else (
                    "frozen"
                    if is_frozen
                    else ("active" if baseline_ready else "locked")
                )
            ),
        }

        today_prefix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        subs_today = sum(1 for s in subs if str(s.created_at).startswith(today_prefix))

        overall_best = pool.get("overall_best")
        primary_metric = str(self.spec.get("primary_metric") or "score")
        scored_subs = [
            s
            for s in subs
            if official_primary_value(s, primary_metric) is not None
        ]
        best_pub_sub = (
            max(
                scored_subs,
                key=lambda s: official_primary_value(s, primary_metric),
            )
            if scored_subs
            else None
        )
        any_divergence = any(s.divergence_warning for s in subs)

        phases_status = []
        for p in phases_meta:
            idx = p["index"]
            # Phases 4, 5, 6 show as active parallel tracks when baseline_ready and not frozen
            if baseline_ready and not is_frozen and idx in (4, 5, 6):
                p_state = "current" if idx == derived_phase else "active_track"
            elif idx < derived_phase:
                p_state = "completed"
            elif idx == derived_phase:
                p_state = "current"
            else:
                p_state = "upcoming"

            gate_locked = False
            if idx == 2 and not gates.get("spec_confirmed"):
                gate_locked = True
            if idx == 3 and not gates.get("data_risks_confirmed"):
                gate_locked = True
            if idx in (4, 5, 6) and not gates.get("baseline_locked"):
                gate_locked = True

            phases_status.append(
                {
                    **p,
                    "status": p_state,
                    "is_active_view": idx == active_phase_idx,
                    "gate_locked": gate_locked,
                }
            )

        # 3. ACTION-ORIENTED BLOCKERS (No developer jargon like 'Gate 2->3')
        blockers: List[Dict[str, Any]] = []
        warn_count = len(data_report.get("warnings", []))
        if not gates.get("spec_confirmed"):
            blockers.append(
                {
                    "code": "TASK_SPEC_REVIEW_REQUIRED",
                    "title": "Task contract review required",
                    "severity": "critical",
                    "phase": 1,
                    "action_label": "Review task contract",
                    "message": "Confirm the selected task config and plugin before relying on their metrics and constraints.",
                }
            )
        if not gates.get("data_risks_confirmed"):
            blockers.append(
                {
                    "code": "DATA_REVIEW_REQUIRED",
                    "title": "Data review required",
                    "severity": "critical",
                    "phase": 2,
                    "action_label": "Review data risks",
                    "message": f"{warn_count} data risks need your confirmation before a baseline can be established.",
                }
            )
        if not gates.get("baseline_locked"):
            blockers.append(
                {
                    "code": "BASELINE_ANCHOR_REQUIRED",
                    "title": "Baseline anchor required",
                    "severity": "critical",
                    "phase": 3,
                    "action_label": "Establish baseline anchor",
                    "message": "Lock the initial validation benchmark so future experiments have a common reference.",
                }
            )
        if any_divergence:
            blockers.append(
                {
                    "code": "VALIDATION_LEAK_OR_DIVERGENCE",
                    "title": "Validation vs. Leaderboard divergence",
                    "severity": "critical",
                    "phase": 6,
                    "action_label": "Inspect divergence",
                    "message": "External evaluation score dropped by > 0.05 compared to local CV.",
                }
            )
        if has_public_lb and subs_today >= daily_quota:
            blockers.append(
                {
                    "code": "DAILY_QUOTA_REACHED",
                    "title": "Daily submission limit reached",
                    "severity": "warning",
                    "phase": 6,
                    "action_label": "View submission history",
                    "message": f"Used {subs_today}/{daily_quota} submissions today. Continue iterating locally.",
                }
            )
        if not gates.get("freeze_locked") and baseline_ready:
            blockers.append(
                {
                    "code": "FINAL_PORTFOLIO_UNFROZEN",
                    "title": "Final selection not yet frozen",
                    "severity": "info",
                    "phase": 7,
                    "action_label": "Select final candidates",
                    "message": f"Up to {max_final_slots} final candidate slots can be locked before competition closure.",
                }
            )

        # 4. NEXT RECOMMENDED ACTION + AGENT CO-PILOT PLAN
        if not gates.get("spec_confirmed"):
            next_action = {
                "target_phase": 1,
                "action_id": "CONFIRM_TASK_SPEC",
                "title": "Review and confirm the selected task contract",
                "detail": f"Task config: `{self.project.task_config_path.relative_to(self.project_root)}`; plugin: `{self.project.plugin_id or 'default'}`.",
                "cta_label": "Confirm task contract",
                "agent_plan": [
                    "Inspect the selected task specification and adapter capabilities",
                    "Confirm the metric, data, and validation contract",
                    "Request explicit user approval before marking it confirmed",
                ],
            }
        elif not gates.get("data_risks_confirmed"):
            next_action = {
                "target_phase": 2,
                "action_id": "CONFIRM_DATA_RISKS",
                "title": "Review & confirm dataset risks before establishing Baseline",
                "detail": f"{warn_count} declarative checks executed from Competition Spec (`configs/task.yaml`).",
                "cta_label": "Review data risks",
                "agent_plan": [
                    "Run declarative data checks from Competition Spec",
                    "Verify group split isolation across folds",
                    "Audit conditional label rules & class sparsity",
                    "Request explicit user approval before marking the dataset ready for baseline work",
                ],
            }
        elif not gates.get("baseline_locked") and pool["baseline_score"] is None:
            next_action = {
                "target_phase": 2,
                "action_id": "ESTABLISH_VERIFIED_BASELINE",
                "title": "Run a complete verified baseline validation",
                "detail": "No verified baseline score is available yet.",
                "cta_label": "Run baseline validation",
                "agent_plan": [
                    "Create a declared baseline experiment",
                    "Run full validation with the active task scorer",
                ],
            }
        elif not gates.get("baseline_locked"):
            next_action = {
                "target_phase": 3,
                "action_id": "LOCK_BASELINE_ANCHOR",
                "title": f"Establish official Baseline anchor ({pool['baseline_id']})",
                "detail": f"Lock reference score ({pool['baseline_score']:.4f}) to activate Experiment, Candidate, and Evaluation tracks.",
                "cta_label": "Establish baseline anchor",
                "agent_plan": [
                    "Verify baseline validation metrics across all folds",
                    "Record immutable git commit & seed receipt",
                    "Set official benchmark anchor and unlock parallel iteration tracks",
                ],
            }
        elif len(synth_cands) == 0:
            next_action = {
                "target_phase": 4,
                "action_id": "BUILD_SPECIALIST_ROUTER",
                "title": "Investigate errors in current best candidate & synthesize next iteration",
                "detail": "Combine top subtask performers or launch a grounded experiment to close the target gap.",
                "cta_label": "Explore & synthesize candidate",
                "agent_plan": [
                    "Inspect diagnostics & worst failure slices of current best model",
                    "Compare recent experiment branches in Solution Tree",
                    "Search domain knowledge base for verified improvements",
                    "Propose one atomic hypothesis & build next candidate",
                ],
            }
        elif not gates.get("freeze_locked"):
            next_action = {
                "target_phase": 7,
                "action_id": "FREEZE_FOR_PRIVATE_LB",
                "title": f"Select & freeze up to {max_final_slots} candidates for Final Evaluation",
                "detail": "Lock your risk-balanced portfolio (Stable CV + Diverse Ensemble) with SHA-256 verification.",
                "cta_label": "Review final selection",
                "agent_plan": [
                    "Rank candidates by fold stability and ensemble diversity",
                    f"Select optimal {max_final_slots}-slot portfolio per Final Selection Policy",
                    "Generate cryptographic freeze lock (FROZEN_FINAL_LOCK.json)",
                ],
            }
        else:
            next_action = {
                "target_phase": 8,
                "action_id": "GENERATE_FINAL_PAPER",
                "title": "Record final evaluation outcomes & export Reproducibility Package",
                "detail": "Compare Local CV vs External vs Final evaluation and generate ablation tables.",
                "cta_label": "Open Competition Closure",
                "agent_plan": [
                    "Trace winning experiment lineage from Baseline to Final Candidate",
                    "Compile 4-tier evaluation comparison table",
                    "Export Markdown & LaTeX reproducibility report",
                ],
            }

        baseline_label = (
            "N/A" if pool["baseline_score"] is None else f"{pool['baseline_score']:.4f}"
        )
        best_score = (
            overall_best.get("composite_score")
            if overall_best
            else pool["baseline_score"]
        )
        best_label = "N/A" if best_score is None else f"{best_score:.4f}"
        hud = {
            "where_am_i": {
                "phase_index": active_phase_idx,
                "derived_phase_index": derived_phase,
                "phase_title": active_phase_meta["title"],
                "phase_question": active_phase_meta["question"],
                "mode": self.state["mode"],
                "active_tracks": active_tracks,
                "total_experiments": len(all_cands),
                "total_submissions": len(subs),
                "banner": (
                    f"{active_phase_meta['title']} ({self.state['mode']}) — "
                    f"Baseline: {baseline_label} | Best CV: {best_label}"
                ),
            },
            "what_is_current_best": {
                "baseline": {
                    "id": pool["baseline_id"],
                    "score": pool["baseline_score"],
                },
                "cv_best": {
                    "id": overall_best["id"] if overall_best else pool["baseline_id"],
                    "score": (
                        overall_best["composite_score"]
                        if overall_best
                        else pool["baseline_score"]
                    ),
                    "delta": (
                        overall_best["delta_vs_baseline"] if overall_best else None
                    ),
                    "subtasks": (
                        overall_best.get("subtasks", {}) if overall_best else {}
                    ),
                },
                "public_lb_best": {
                    "enabled": has_public_lb,
                    "label": eval_proto.get(
                        "external_eval_label", "Public Leaderboard"
                    ),
                    "sub_id": best_pub_sub.id if best_pub_sub else None,
                    "exp_id": best_pub_sub.exp_id if best_pub_sub else None,
                    "score": (
                        official_primary_value(best_pub_sub, primary_metric)
                        if best_pub_sub
                        else None
                    ),
                    "delta_vs_cv": (
                        best_pub_sub.score_deltas.get(
                            primary_metric, best_pub_sub.score_deltas.get("primary_delta")
                        )
                        if best_pub_sub
                        else None
                    ),
                    "quota_used_today": subs_today,
                    "daily_quota": daily_quota,
                },
                "private_frozen": {
                    "label": eval_proto.get("final_eval_label", "Private Evaluation"),
                    "max_candidates": max_final_slots,
                    "is_frozen": bool(gates.get("freeze_locked")),
                    "slot_1_safe_cv": (self.state.get("freeze_selection") or {}).get(
                        "slot_1_safe_cv"
                    ),
                    "slot_2_aggressive_lb": (
                        self.state.get("freeze_selection") or {}
                    ).get("slot_2_aggressive_lb"),
                    "private_score_1": (self.state.get("finalization") or {}).get(
                        "private_score_slot_1"
                    ),
                    "private_score_2": (self.state.get("finalization") or {}).get(
                        "private_score_slot_2"
                    ),
                    "final_rank": (self.state.get("finalization") or {}).get(
                        "final_rank"
                    ),
                },
            },
            "what_is_blocking_me": blockers,
            "what_should_i_do_next": next_action,
        }

        return {
            "spec_summary": {
                "name": self.spec.get("name", "Benchmark"),
                "task_type": self.spec.get("task_type", "Predictive Modeling"),
                "primary_metric": self.spec.get("primary_metric", "Score"),
                "direction": self.spec.get("direction", "maximize"),
                "baseline_goal": self.spec.get("primary_metric_baseline"),
                "target_goal": self.spec.get("primary_metric_target"),
                "subtasks": self.spec.get("subtasks", []),
                "competition_rules": self.spec.get("competition_rules", {}),
                "evaluation_protocol": eval_proto,
                "final_selection_policy": self.spec.get("final_selection_policy", {}),
                "closure_outputs": self.spec.get("closure_outputs", []),
            },
            "hud": hud,
            "active_tracks": active_tracks,
            "agent_copilot": self.state.get("agent_copilot", {}),
            "audit_trail": self.state.get("audit_trail", []),
            "phases": phases_status,
            "gates": gates,
            "mode": self.state["mode"],
            "data_audit_config": self.state["data_audit_config"],
            "agent_context_config": self.state["agent_context_config"],
            "data_readiness_report": data_report,
            "candidate_pool": pool,
            "freeze_selection": self.state["freeze_selection"],
            "finalization": self.state["finalization"],
        }

    def render_agent_context_block(
        self,
        journal: TreeJournal,
        sub_mgr: SubmissionManager,
        task_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the Competition State block to inject into the Agent Priming Prompt."""
        snap = self.evaluate_journey_snapshot(journal, sub_mgr, task_info)
        cfg = AgentContextConfig(**self.state["agent_context_config"])
        hud = snap["hud"]
        sections: List[str] = []

        if cfg.inject_phase_hud:
            w = hud["where_am_i"]
            tracks = w.get("active_tracks", {})
            sections.append(
                "## 🧭 0. COMPETITION JOURNEY STATE & ACTIVE TRACKS\n"
                f"- **Current Stage**: `{w['phase_title']}` (Mode: `{w['mode']}`)\n"
                f"- **Active Parallel Tracks**: Experimentation=`{tracks.get('experimentation')}` | "
                f"Candidate Management=`{tracks.get('candidate_management')}` | "
                f"External Eval=`{tracks.get('external_evaluation')}`\n"
                f"- **Guiding Question**: *{w['phase_question']}*"
            )

        if cfg.inject_current_best_anchors:
            b = hud["what_is_current_best"]
            base_score_str = (
                f"{b['baseline']['score']:.4f}"
                if b["baseline"].get("score") is not None
                else "N/A"
            )
            cv_score_str = (
                f"{b['cv_best']['score']:.4f}"
                if b["cv_best"].get("score") is not None
                else "N/A"
            )
            cv_delta_str = (
                f"Δ `{b['cv_best']['delta']:+.4f}`"
                if b["cv_best"].get("delta") is not None
                else ""
            )
            sections.append(
                "## 🏆 FOUR-TIER EVALUATION ANCHORS (`CV ≠ External ≠ Final ≠ Rank`)\n"
                f"- **Official Baseline**: `{b['baseline']['id']}` (`{base_score_str}`)\n"
                f"- **Current Best Local Validation**: `{b['cv_best']['id']}` (`{cv_score_str}`{', ' + cv_delta_str if cv_delta_str else ''})\n"
                f"- **External Evaluation ({b['public_lb_best']['label']})**: `{b['public_lb_best']['sub_id'] or 'None'}` "
                f"(Score: `{b['public_lb_best']['score'] if b['public_lb_best']['score'] is not None else 'N/A'}`)\n"
                f"- **Final Portfolio ({b['private_frozen']['max_candidates']} slots max)**: "
                f"`{'LOCKED' if b['private_frozen']['is_frozen'] else 'OPEN'}`"
            )

        if cfg.inject_active_blockers and hud["what_is_blocking_me"]:
            blk_lines = ["## 🚧 ACTIVE BLOCKERS Requiring Resolution"]
            for blk in hud["what_is_blocking_me"]:
                blk_lines.append(
                    f"- **[{blk['severity'].upper()}] {blk['title']}**: {blk['message']}"
                )
            sections.append("\n".join(blk_lines))

        if cfg.inject_next_action:
            nxt = hud["what_should_i_do_next"]
            plan_str = "\n".join(f"  - {s}" for s in nxt.get("agent_plan", []))
            sections.append(
                "## 🚀 NEXT RECOMMENDED ACTION & AGENT CO-PILOT PLAN\n"
                f"- **Action**: **{nxt['title']}**\n"
                f"- **Rationale**: {nxt['detail']}\n"
                f"- **Agent Execution Plan**:\n{plan_str}"
            )

        return "\n\n".join(sections)


# Backward-compatible alias
CompetitionState = CompetitionStateMachine
