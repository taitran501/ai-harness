"""Submission Lifecycle and Score Tracking Manager.

Manages all competition submissions across local validation and remote leaderboard evaluation:
1. Tracks verification status (LOCAL_VERIFIED, SUBMITTED_PENDING, SCORED, DISQUALIFIED).
2. Maintains central registry (submissions/registry.json) and visual table (submissions/SUBMISSIONS.md).
3. Records official remote leaderboard scores (Public / Private LB) and compares against Local CV.
4. Alerts on CV vs LB divergence (direction-aware overfit detection when delta exceeds threshold).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    def __init__(
        self,
        submissions_dir: Optional[str | Path] = None,
        *,
        artifacts_dir: Optional[str | Path] = None,
        project_root: Optional[str | Path] = None,
        task_config_path: Optional[str | Path] = None,
    ):
        from harness.project import HarnessProject

        project = HarnessProject.discover(project_root, task_config_path)
        self.project_root = project.root
        self.task_config_path = project.task_config_path
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else project.artifacts_dir
        )
        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = self.project_root / self.artifacts_dir
        if submissions_dir is None:
            submissions_dir = (
                self.artifacts_dir / "submissions"
                if project.manifest
                else self.project_root / "submissions"
            )
        self.submissions_dir = Path(submissions_dir)
        if not self.submissions_dir.is_absolute():
            self.submissions_dir = self.project_root / self.submissions_dir
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.submissions_dir / "registry.json"
        self.submissions: Dict[str, SubmissionEntry] = {}
        self.load()
        self._snapshot = [asdict(n) for n in self.submissions.values()]

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
            print(
                f"[SubmissionManager] Warning: Failed to load registry: {e}",
                file=sys.stderr,
            )

    def save(self) -> None:
        """Save registry to disk and regenerate SUBMISSIONS.md."""
        data = {
            "version": "1.0",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_submissions": len(self.submissions),
            "submissions": [asdict(s) for s in self.submissions.values()],
        }
        from harness.io_utils import atomic_write_json
        from harness.state_store import save_registry

        records = save_registry(
            self.registry_file,
            "submissions",
            data["submissions"],
            getattr(self, "_snapshot", []),
        )
        self.submissions = {r["id"]: SubmissionEntry(**r) for r in records}
        self._snapshot = records
        if self.submissions_dir != self.project_root / "submissions":
            root_sub_dir = self.project_root / "submissions"
            root_sub_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(root_sub_dir / "registry.json", data, indent=2, ensure_ascii=False)
        self.render_submissions_markdown()

    def get_next_submission_id(self) -> str:
        """Generate monotonically increasing submission ID (sub_001, sub_002, ...)."""
        existing_indices = []
        for sid in self.submissions.keys():
            if sid.startswith("sub_"):
                parts = sid.split("_")
                if len(parts) > 1:
                    try:
                        num = int(parts[1])
                        existing_indices.append(num)
                    except (IndexError, ValueError):
                        pass
        next_idx = max(existing_indices, default=0) + 1
        return f"sub_{next_idx:03d}"

    def list_submissions(self) -> List[Dict[str, Any]]:
        """Return a list of submission entry dicts, sorted by submission ID.

        Agents commonly reach for a ``list_submissions`` method; this is the
        canonical accessor for the registry contents.
        """
        return [asdict(entry) for sid, entry in sorted(self.submissions.items())]

    def all_submissions(self) -> Dict[str, "SubmissionEntry"]:
        """Return a copy of the submissions registry (mapping sub_id -> entry).

        Use this when you need the raw dict (the ``self.submissions`` attribute
        is live and may mutate; this returns a snapshot).
        """
        return dict(self.submissions)

    def register_submission(
        self,
        exp_id: str,
        candidate_csv: Optional[str] = None,
        baseline_csv: Optional[str] = None,
        ensemble_components: Optional[List[str]] = None,
        notes: str = "",
        min_diff_pct: float = 2.0,
    ) -> SubmissionEntry:
        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.project import HarnessProject
        from harness.task_spec import load_active_task_spec

        project = HarnessProject.discover(self.project_root, self.task_config_path)
        spec = load_active_task_spec(project.task_config_path)
        context = TaskPluginContext(
            project.root,
            project.task_config_path,
            spec,
            project.plugin_id or "default",
        )
        plugin = load_task_plugin(project.plugin_id, context)
        caps = plugin.capabilities()
        if not caps.get("submission_builder") or not caps.get("submission_validator"):
            raise ValueError(
                "Submission requires task-plugin submission hooks and verified test evidence"
            )
        if not candidate_csv:
            default_cand = self.artifacts_dir / exp_id / "test_predictions.csv"
            if default_cand.is_file():
                candidate_csv = str(default_cand)
            else:
                raise ValueError(
                    f"Candidate CSV not specified and {default_cand} does not exist"
                )
        sub_spec = getattr(spec, "submission_spec", None) or spec.metadata.get("submission_spec", {})
        file_name = sub_spec.get("file_name", "submission.csv") if isinstance(sub_spec, dict) else "submission.csv"
        return self.register_plugin_submission(
            exp_id=exp_id,
            candidate_csv=candidate_csv,
            plugin=plugin,
            plugin_context=context,
            file_name=file_name,
            notes=notes,
        )

    def register_plugin_submission(
        self,
        *,
        exp_id: str,
        candidate_csv: str | Path,
        plugin: Any,
        plugin_context: Any,
        file_name: str = "submission.csv",
        notes: str = "",
    ) -> SubmissionEntry:
        """Build and validate a task-specific package through the selected plugin."""
        import re

        source = Path(candidate_csv).expanduser()
        if not source.is_absolute():
            source = self.project_root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Prediction file does not exist: {source}")

        from harness.artifacts import validate_test_prediction

        validate_test_prediction(
            source, task_spec=plugin_context.task_spec, plugin=plugin
        )
        normalized_file_name = str(file_name).replace("\\", "/")
        safe_name = Path(normalized_file_name).name
        if safe_name in ("", ".", "..") or "/" in normalized_file_name:
            raise ValueError(
                "Submission file_name must be a simple filename without directories."
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", safe_name):
            raise ValueError(
                f"Submission file_name contains unsupported characters: {safe_name!r}"
            )

        sub_id = f"{self.get_next_submission_id()}_{exp_id}"
        target_dir = self.submissions_dir / sub_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target_path = target_dir / safe_name
        receipt_path = target_dir / "PLUGIN_AUDIT_RECEIPT.json"

        try:
            import pandas as pd

            predictions = pd.read_csv(source)
            built_path = plugin.build_submission(
                plugin_context, predictions, target_path
            )
            final_path = Path(built_path or target_path).expanduser()
            if not final_path.is_absolute():
                final_path = target_dir / final_path
            final_path = final_path.resolve()
            try:
                final_path.relative_to(target_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    "Task plugin wrote its submission outside the managed submission directory."
                ) from exc
            if not final_path.is_file():
                raise FileNotFoundError(
                    f"Task plugin did not create a submission file: {final_path}"
                )

            plugin.validate_submission(plugin_context, final_path)

            metrics = json.loads(
                (source.parent / "metrics.json").read_text(encoding="utf-8")
            )
            if source.parent.name != exp_id:
                raise ValueError(
                    "Submission source experiment does not match requested exp_id"
                )
            local_cv = {metrics["primary_metric"]: float(metrics["primary_score"])}
            for target, values in metrics.get("subtasks", {}).items():
                local_cv[target] = float(values["score"])

            from harness.io_utils import atomic_write_json

            receipt = {
                "status": "PASS",
                "plugin_id": getattr(plugin, "plugin_id", "unknown"),
                "plugin_version": getattr(plugin, "version", "unknown"),
                "source_predictions": str(source),
                "submission_file": str(final_path),
                "validated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }
            atomic_write_json(receipt_path, receipt, indent=2, ensure_ascii=False)

            entry = SubmissionEntry(
                id=sub_id,
                exp_id=exp_id,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                status="LOCAL_VERIFIED",
                local_cv_scores=local_cv,
                candidate_path=str(source),
                archive_path=str(final_path),
                receipt_path=str(receipt_path),
                notes=notes,
            )
            from harness.io_utils import atomic_write_json

            atomic_write_json(
                target_dir / "metadata.json",
                asdict(entry),
                indent=2,
                ensure_ascii=False,
            )
            if self.submissions_dir != self.project_root / "submissions":
                root_target_dir = self.project_root / "submissions" / sub_id
                root_target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final_path, root_target_dir / safe_name)
                shutil.copy2(receipt_path, root_target_dir / "PLUGIN_AUDIT_RECEIPT.json")
                shutil.copy2(target_dir / "metadata.json", root_target_dir / "metadata.json")
            self.submissions[sub_id] = entry
            self.save()
            return entry
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

    def record_official_score(
        self,
        sub_id: str,
        primary_score: Optional[float] = None,
        scores: Optional[Dict[str, Any]] = None,
        split_type: str = "public",
        notes: str = "",
        custom_scores: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SubmissionEntry:
        """Record ground-truth evaluation returned from organizer / competition portal."""
        entry = self.submissions.get(sub_id)
        if not entry:
            raise KeyError(f"Submission ID '{sub_id}' not found in registry.")

        scores_dict: Dict[str, Any] = {
            "split_type": split_type,
        }

        # Resolve primary metric name dynamically from spec or local_cv_scores
        primary_key = "score"
        try:
            from harness.task_spec import load_active_task_spec

            spec = load_active_task_spec(self.task_config_path)
            if getattr(spec, "primary_metric", None):
                primary_key = getattr(
                    spec.primary_metric, "name", str(spec.primary_metric)
                )
        except Exception:
            pass

        if primary_score is not None:
            scores_dict[primary_key] = round(float(primary_score), 4)

        all_custom: Dict[str, Any] = {}
        if custom_scores:
            all_custom.update(custom_scores)
        if scores:
            all_custom.update(scores)
        for k, v in kwargs.items():
            if v is not None:
                all_custom[k] = v

        for k, v in all_custom.items():
            if v is not None:
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

            spec = load_active_task_spec(self.task_config_path)
            if hasattr(spec, "primary_metric") and hasattr(
                spec.primary_metric, "direction"
            ):
                opt_direction = str(spec.primary_metric.direction).lower()
            else:
                opt_direction = str(getattr(spec, "direction", "maximize")).lower()
        except Exception:
            pass

        all_metric_keys = set(entry.local_cv_scores.keys()).union(
            entry.official_scores.keys()
        )
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
        if self.submissions_dir != self.project_root / "submissions":
            root_sub_dir = self.project_root / "submissions" / sub_id
            if root_sub_dir.exists():
                with open(root_sub_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(asdict(entry), f, indent=2)

        self.save()
        return entry

    def mark_submitted(
        self, sub_id: str, portal_reference: str = ""
    ) -> SubmissionEntry:
        """Mark submission as uploaded to the competition platform (pending evaluation)."""
        entry = self.submissions.get(sub_id)
        if not entry:
            raise KeyError(f"Submission ID '{sub_id}' not found.")
        entry.status = "SUBMITTED_PENDING"
        if portal_reference:
            entry.notes = (entry.notes + f" | Portal Ref: {portal_reference}").strip(
                " | "
            )
        self.save()
        return entry

    def render_submissions_markdown(self) -> str:
        """Render high-density submissions tracker table into SUBMISSIONS.md."""
        try:
            from harness.task_spec import load_active_task_spec

            spec = load_active_task_spec(self.task_config_path)
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
            "| Sub ID | Exp ID | Status | Local CV | Official LB | Delta (LB - CV) | Package | Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]

        if not self.submissions:
            lines.append(
                "| - | - | *No submissions registered yet.* | - | - | - | - | - |"
            )
        else:
            for s in sorted(
                self.submissions.values(), key=lambda x: x.created_at, reverse=True
            ):
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
                if primary_metric in s.local_cv_scores:
                    cv_str = f"{primary_metric}:{s.local_cv_scores.get(primary_metric)}"
                elif s.local_cv_scores:
                    items = [f"{k}:{v}" for k, v in list(s.local_cv_scores.items())[:3]]
                    cv_str = " / ".join(items)
                else:
                    cv_str = "-"

                # LB string
                if s.status == "SCORED":
                    if primary_metric in s.official_scores:
                        lb_str = (
                            f"{primary_metric}:{s.official_scores.get(primary_metric)}"
                        )
                    elif s.official_scores:
                        items = [
                            f"{k}:{v}"
                            for k, v in list(s.official_scores.items())[:3]
                            if k != "split_type"
                        ]
                        lb_str = " / ".join(items) if items else "-"
                    else:
                        lb_str = "-"
                else:
                    lb_str = "Pending"

                # Delta string
                if (
                    primary_metric in s.score_deltas
                    and s.score_deltas[primary_metric] is not None
                ):
                    delta_str = f"{s.score_deltas[primary_metric]:+.4f}"
                elif s.score_deltas:
                    items = [
                        f"{k}:{v:+.4f}"
                        for k, v in list(s.score_deltas.items())[:2]
                        if isinstance(v, (int, float))
                    ]
                    delta_str = " / ".join(items) if items else "-"
                else:
                    delta_str = "-"

                pkg_link = (
                    f"[`zip`](file:///{Path(s.archive_path).as_posix()})"
                    if s.archive_path
                    else "-"
                )
                notes_clean = s.notes or "-"

                lines.append(
                    f"| `{s.id}` | `{s.exp_id}` | {status_icon} | {cv_str} | {lb_str} | {delta_str} | {pkg_link} | {notes_clean} |"
                )

        content = "\n".join(lines) + "\n"
        out_file = self.submissions_dir / "SUBMISSIONS.md"
        from harness.io_utils import atomic_write_text

        atomic_write_text(out_file, content, encoding="utf-8")
        if self.submissions_dir != self.project_root / "submissions":
            root_out = self.project_root / "submissions" / "SUBMISSIONS.md"
            root_out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(root_out, content, encoding="utf-8")
        return content
