"""Submission Lifecycle and Score Tracking Manager.

Manages all competition submissions across local validation and remote leaderboard evaluation:
1. Tracks verification status (LOCAL_VERIFIED, SUBMITTED_PENDING, SCORED, DISQUALIFIED).
2. Maintains central registry (submissions/registry.json) and visual table (submissions/SUBMISSIONS.md).
3. Records official remote leaderboard scores (Public / Private LB) and compares against Local CV.
4. Alerts on CV vs LB divergence (direction-aware overfit detection when delta exceeds threshold).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SubmissionEntry:
    id: str
    exp_id: str
    created_at: str
    status: str  # LOCAL_VERIFIED, SUBMITTED_PENDING, SCORED, DISQUALIFIED
    local_cv_scores: Dict[str, float] = field(default_factory=dict)
    official_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    score_deltas: Dict[str, Optional[float]] = field(default_factory=dict)
    candidate_path: str = ""
    baseline_path: Optional[str] = None
    archive_path: str = ""
    receipt_path: str = ""
    notes: str = ""
    divergence_warning: bool = False


class SubmissionManager:
    """Central registry and tracker for competition submissions."""

    def __init__(self, submissions_dir: str = "submissions"):
        self.submissions_dir = Path(submissions_dir)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.submissions_dir / "registry.json"
        self.submissions: Dict[str, SubmissionEntry] = {}
        self.load()

    def load(self) -> None:
        """Load registry from disk."""
        if not self.registry_file.exists():
            return
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("submissions", []):
                entry = SubmissionEntry(**item)
                self.submissions[entry.id] = entry
        except Exception as e:
            print(f"[SubmissionManager] Warning: Failed to load registry: {e}", file=sys.stderr)

    def save(self) -> None:
        """Save registry to disk and regenerate SUBMISSIONS.md."""
        data = {
            "version": "1.0",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_submissions": len(self.submissions),
            "submissions": [asdict(s) for s in self.submissions.values()],
        }
        from harness.io_utils import atomic_write_json
        atomic_write_json(self.registry_file, data, indent=2, ensure_ascii=False)
        self.render_submissions_markdown()

    def get_next_submission_id(self) -> str:
        """Generate monotonically increasing submission ID (sub_001, sub_002, ...)."""
        existing_indices = []
        for sid in self.submissions.keys():
            if sid.startswith("sub_"):
                try:
                    num = int(sid.split("_")[1])
                    existing_indices.append(num)
                except (IndexError, ValueError):
                    pass
        next_idx = max(existing_indices, default=0) + 1
        return f"sub_{next_idx:03d}"

    def register_submission(
        self,
        exp_id: str,
        candidate_csv: str,
        baseline_csv: Optional[str] = None,
        ensemble_components: Optional[List[str]] = None,
        notes: str = "",
        min_diff_pct: float = 2.0,
    ) -> SubmissionEntry:
        """Execute audit gate, package archive, and register submission entry."""
        sub_id = f"{self.get_next_submission_id()}_{exp_id}"
        target_dir = self.submissions_dir / sub_id
        target_dir.mkdir(parents=True, exist_ok=True)

        target_zip = target_dir / "submission.zip"
        target_receipt = target_dir / "AUDIT_RECEIPT.json"

        # 1. Run physical audit gate
        cmd = [
            sys.executable,
            "tools/audit_gate.py",
            "--candidate", str(candidate_csv),
            "--output-zip", str(target_zip),
            "--receipt", str(target_receipt),
            "--min-diff-pct", str(min_diff_pct),
        ]
        if baseline_csv:
            cmd.extend(["--baseline", str(baseline_csv)])
        else:
            cmd.append("--allow-initial")

        if ensemble_components:
            cmd.append("--ensemble-components")
            cmd.extend(ensemble_components)

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise RuntimeError(
                f"Audit Gate rejected candidate!\nStdout: {res.stdout}\nStderr: {res.stderr}"
            )

        # 2. Extract CV scores if summary.csv or metrics.json exists in artifacts
        local_cv: Dict[str, float] = {}
        artifact_dir = Path("artifacts") / exp_id
        summary_csv = artifact_dir / "summary.csv"
        metrics_json = artifact_dir / "metrics.json"

        try:
            from harness.task_spec import load_active_task_spec
            spec = load_active_task_spec()
            primary_metric = spec.primary_metric
        except Exception:
            primary_metric = "macro_f1"

        if summary_csv.exists():
            import pandas as pd
            sdf = pd.read_csv(summary_csv)
            num_cols = sdf.select_dtypes(include=["number"]).columns
            for col in num_cols:
                if col not in ("dev_fold", "train_rows", "dev_rows", "fold"):
                    local_cv[col] = round(float(sdf[col].mean()), 4)

            # Preserve backward-compatible keys
            if "a_macro_f1" in sdf.columns:
                local_cv["subtask_a_macro"] = round(float(sdf["a_macro_f1"].mean()), 4)
            if "b_macro_f1" in sdf.columns:
                local_cv["subtask_b_macro"] = round(float(sdf["b_macro_f1"].mean()), 4)
            if "c_target_type_macro_f1_oracle_hate" in sdf.columns:
                local_cv["subtask_c_type_macro"] = round(
                    float(sdf["c_target_type_macro_f1_oracle_hate"].mean()), 4
                )
            if "c_identity_macro_f1_oracle_hate" in sdf.columns:
                local_cv["subtask_c_id_macro"] = round(
                    float(sdf["c_identity_macro_f1_oracle_hate"].mean()), 4
                )
        elif metrics_json.exists():
            try:
                with open(metrics_json, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                if isinstance(m_data, list) and len(m_data) > 0:
                    m_dict = m_data[0]
                elif isinstance(m_data, dict):
                    m_dict = m_data
                else:
                    m_dict = {}
                for k, v in m_dict.items():
                    if isinstance(v, (int, float)):
                        local_cv[k] = round(float(v), 4)
            except Exception:
                pass

        entry = SubmissionEntry(
            id=sub_id,
            exp_id=exp_id,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            status="LOCAL_VERIFIED",
            local_cv_scores=local_cv,
            candidate_path=str(candidate_csv),
            baseline_path=str(baseline_csv) if baseline_csv else None,
            archive_path=str(target_zip),
            receipt_path=str(target_receipt),
            notes=notes,
        )

        # Write metadata.json in sub_dir atomically
        from harness.io_utils import atomic_write_json
        atomic_write_json(target_dir / "metadata.json", asdict(entry), indent=2)

        self.submissions[sub_id] = entry
        self.save()
        return entry

    def record_official_score(
        self,
        sub_id: str,
        macro_a: Optional[float] = None,
        macro_b: Optional[float] = None,
        macro_c_type: Optional[float] = None,
        macro_c_id: Optional[float] = None,
        split_type: str = "public",
        notes: str = "",
        custom_scores: Optional[Dict[str, Any]] = None,
    ) -> SubmissionEntry:
        """Record ground-truth evaluation returned from organizer / competition portal."""
        entry = self.submissions.get(sub_id)
        if not entry:
            raise KeyError(f"Submission ID '{sub_id}' not found in registry.")

        scores_dict: Dict[str, Any] = {
            "subtask_a_macro": round(macro_a, 4) if macro_a is not None else None,
            "subtask_b_macro": round(macro_b, 4) if macro_b is not None else None,
            "subtask_c_type_macro": round(macro_c_type, 4) if macro_c_type is not None else None,
            "subtask_c_id_macro": round(macro_c_id, 4) if macro_c_id is not None else None,
            "split_type": split_type,
        }
        if custom_scores:
            for k, v in custom_scores.items():
                if v is not None and k not in scores_dict:
                    try:
                        scores_dict[k] = round(float(v), 4)
                    except (ValueError, TypeError):
                        scores_dict[k] = v

        entry.official_scores = scores_dict

        # Calculate deltas (LB - CV) across all matched keys
        deltas = {}
        divergence = False
        opt_direction = "maximize"
        try:
            from harness.task_spec import load_active_task_spec
            spec = load_active_task_spec()
            if hasattr(spec, "primary_metric") and hasattr(spec.primary_metric, "direction"):
                opt_direction = str(spec.primary_metric.direction).lower()
            else:
                opt_direction = str(getattr(spec, "direction", "maximize")).lower()
        except Exception:
            pass

        all_metric_keys = set(entry.local_cv_scores.keys()).union(entry.official_scores.keys())
        for k in all_metric_keys:
            if k == "split_type":
                continue
            cv_val = entry.local_cv_scores.get(k)
            lb_val = entry.official_scores.get(k)
            if isinstance(cv_val, (int, float)) and isinstance(lb_val, (int, float)):
                d = round(float(lb_val) - float(cv_val), 4)
                deltas[k] = d
                if opt_direction == "minimize":
                    if d > 0.05:  # Error increased on LB
                        divergence = True
                else:
                    if d < -0.05:  # Score dropped on LB
                        divergence = True

        entry.score_deltas = deltas
        entry.divergence_warning = divergence
        entry.status = "SCORED"
        if notes:
            entry.notes = (entry.notes + " | " + notes).strip(" | ")

        # Update metadata.json in sub_dir
        sub_dir = self.submissions_dir / sub_id
        if sub_dir.exists():
            with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(asdict(entry), f, indent=2)

        self.save()
        return entry

    def mark_submitted(self, sub_id: str, portal_reference: str = "") -> SubmissionEntry:
        """Mark submission as uploaded to the competition platform (pending evaluation)."""
        entry = self.submissions.get(sub_id)
        if not entry:
            raise KeyError(f"Submission ID '{sub_id}' not found.")
        entry.status = "SUBMITTED_PENDING"
        if portal_reference:
            entry.notes = (entry.notes + f" | Portal Ref: {portal_reference}").strip(" | ")
        self.save()
        return entry

    def render_submissions_markdown(self) -> str:
        """Render high-density submissions tracker table into SUBMISSIONS.md."""
        try:
            from harness.task_spec import load_active_task_spec
            spec = load_active_task_spec()
            spec_name = spec.name
            primary_metric = spec.primary_metric
        except Exception:
            spec_name = "Competition"
            primary_metric = "macro_f1"

        lines = [
            f"# {spec_name} Submissions & Leaderboard Registry",
            "",
            "This registry tracks all verified competition packages, submission statuses,",
            "and compares Local Cross-Validation against official Leaderboard scores.",
            "",
            "> **Status Legend**:",
            "> - 🔒 `LOCAL_VERIFIED`: Passed Deterministic & Statistical Audit Gate.",
            "> - ⏳ `SUBMITTED_PENDING`: Uploaded to evaluation portal, awaiting leaderboard score.",
            "> - 🎯 `SCORED`: Official scores recorded with CV vs LB delta analysis.",
            "> - ⚠️ `DIVERGENCE`: Local CV significantly over-estimated performance (LB - CV < -0.05).",
            "",
            f"| Sub ID | Exp ID | Status | Local CV | Official LB | Delta (LB - CV) | Package | Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]

        if not self.submissions:
            lines.append("| - | - | *No submissions registered yet.* | - | - | - | - | - |")
        else:
            for s in sorted(self.submissions.values(), key=lambda x: x.created_at, reverse=True):
                status_icon = (
                    "🎯 `SCORED`"
                    if s.status == "SCORED"
                    else (
                        "⏳ `PENDING`"
                        if s.status == "SUBMITTED_PENDING"
                        else "🔒 `LOCAL_VERIFIED`"
                    )
                )
                if s.divergence_warning:
                    status_icon += " ⚠️"

                # CV string
                if "subtask_a_macro" in s.local_cv_scores:
                    cv_a = s.local_cv_scores.get("subtask_a_macro", "-")
                    cv_b = s.local_cv_scores.get("subtask_b_macro", "-")
                    cv_c = s.local_cv_scores.get("subtask_c_type_macro", "-")
                    cv_str = f"A:{cv_a} / B:{cv_b} / C:{cv_c}"
                elif primary_metric in s.local_cv_scores:
                    cv_str = f"{primary_metric}:{s.local_cv_scores.get(primary_metric)}"
                elif s.local_cv_scores:
                    first_k, first_v = next(iter(s.local_cv_scores.items()))
                    cv_str = f"{first_k}:{first_v}"
                else:
                    cv_str = "-"

                # LB string
                if s.status == "SCORED":
                    if "subtask_a_macro" in s.official_scores:
                        lb_a = s.official_scores.get("subtask_a_macro", "-")
                        lb_b = s.official_scores.get("subtask_b_macro", "-")
                        lb_c = s.official_scores.get("subtask_c_type_macro", "-")
                        lb_str = f"A:{lb_a} / B:{lb_b} / C:{lb_c}"
                    elif primary_metric in s.official_scores:
                        lb_str = f"{primary_metric}:{s.official_scores.get(primary_metric)}"
                    elif s.official_scores:
                        first_k, first_v = next(iter(s.official_scores.items()))
                        lb_str = f"{first_k}:{first_v}"
                    else:
                        lb_str = "-"
                else:
                    lb_str = "Pending"

                # Delta string
                if "subtask_a_macro" in s.score_deltas and "subtask_b_macro" in s.score_deltas:
                    d_a = s.score_deltas.get("subtask_a_macro")
                    d_b = s.score_deltas.get("subtask_b_macro")
                    delta_str = f"A:{d_a:+.4f} / B:{d_b:+.4f}" if (d_a is not None and d_b is not None) else "-"
                elif primary_metric in s.score_deltas and s.score_deltas[primary_metric] is not None:
                    delta_str = f"{s.score_deltas[primary_metric]:+.4f}"
                else:
                    delta_str = "-"

                pkg_link = f"[`zip`](file:///{Path(s.archive_path).as_posix()})" if s.archive_path else "-"
                notes_clean = s.notes or "-"

                lines.append(
                    f"| `{s.id}` | `{s.exp_id}` | {status_icon} | {cv_str} | {lb_str} | {delta_str} | {pkg_link} | {notes_clean} |"
                )

        content = "\n".join(lines) + "\n"
        out_file = self.submissions_dir / "SUBMISSIONS.md"
        from harness.io_utils import atomic_write_text
        atomic_write_text(out_file, content, encoding="utf-8")
        return content
