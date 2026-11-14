"""Submission Lifecycle and Score Tracking Manager.

Manages all competition submissions across local validation and remote leaderboard evaluation:
1. Tracks verification status (LOCAL_VERIFIED, SUBMITTED_PENDING, SCORED, DISQUALIFIED).
2. Maintains central registry (submissions/registry.json) and visual table (submissions/SUBMISSIONS.md).
3. Records official remote leaderboard scores (Public / Private LB) and compares against Local CV.
4. Alerts on CV vs LB divergence (direction-aware overfit detection when delta exceeds threshold).
"""

from __future__ import annotations

import json
import copy
import math
import re
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

RESERVED_SUBMISSION_FILENAMES = frozenset(name.casefold() for name in (
    "metadata.json", "PLUGIN_AUDIT_RECEIPT.json", "run_manifest.json",
    "PRESERVATION_RECEIPT.json", "AUDIT_RECEIPT.json", "registry.json", "SUBMISSIONS.md",
))


def _validate_payload_filename(name: str) -> str:
    name = str(name)
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError("Submission file_name must be a simple filename without directories.")
    if name.endswith((".", " ")) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Submission file_name contains unsupported characters: {name!r}")
    if name.casefold() in RESERVED_SUBMISSION_FILENAMES:
        raise ValueError(f"Submission file name cannot be a reserved metadata filename: {name!r}")
    return name


def _validate_registry(value):
    rows = value.get("submissions")
    if not isinstance(rows, list):
        raise ValueError("submissions must be a list")
    ids = set()
    for row in rows:
        entry = SubmissionEntry(**row)
        if not isinstance(entry.id, str) or entry.id in ids:
            raise ValueError("submission IDs must be unique strings")
        ids.add(entry.id)


@dataclass
class SubmissionEntry:
    id: str
    exp_id: str
    created_at: str
    status: str  # LOCAL_VERIFIED, SUBMITTED_PENDING, SCORED, DISQUALIFIED
    local_cv_scores: Dict[str, float] = field(default_factory=dict)
    official_scores: Dict[str, Any] = field(default_factory=dict)
    score_deltas: Dict[str, Optional[float]] = field(default_factory=dict)
    candidate_path: str = ""
    baseline_path: Optional[str] = None
    archive_path: str = ""
    receipt_path: str = ""
    notes: str = ""
    divergence_warning: bool = False
    gate_receipt_id: Optional[str] = None
    payload_sha256: Optional[str] = None
    prediction_sha256: Optional[str] = None
    dispatch_intent_id: Optional[str] = None
    approval_id: Optional[str] = None
    official_observation_id: Optional[str] = None
    official_score_source: str = "declared"
    reported_scores: Dict[str, Optional[float]] = field(default_factory=dict)


class SubmissionManager:
    """Central registry and tracker for competition submissions."""

    def __init__(
        self,
        submissions_dir: Optional[str | Path] = None,
        *,
        artifacts_dir: Optional[str | Path] = None,
        project_root: Optional[str | Path] = None,
        task_config_path: Optional[str | Path] = None,
        read_only: bool = False,
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
        self.read_only = read_only
        if not read_only:
            self.submissions_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.submissions_dir / "registry.json"
        self.submissions: Dict[str, SubmissionEntry] = {}
        self.load()
        self._snapshot = [asdict(n) for n in self.submissions.values()]

    def load(self) -> None:
        """Load registry from disk."""
        from harness.state_store import read_json_state
        data = read_json_state(self.registry_file, default_factory=lambda: {"submissions": []},
                               validate=_validate_registry)
        self._hydrate(data["submissions"])

    def _hydrate(self, records):
        active = {}
        for record in records:
            entry = self.submissions.get(record["id"]) or SubmissionEntry(**record)
            for key, value in record.items():
                setattr(entry, key, copy.deepcopy(value))
            active[entry.id] = entry
        self.submissions = active
        self._snapshot = copy.deepcopy(records)

    def save(self) -> None:
        """Save registry to disk and regenerate SUBMISSIONS.md."""
        if self.read_only:
            raise ValueError("SubmissionManager is read-only")
        from harness.state_store import save_registry, read_json_state, workspace_lock
        with workspace_lock(self.submissions_dir):
            save_registry(self.registry_file, "submissions",
                          [asdict(s) for s in self.submissions.values()], self._snapshot,
                          validate=_validate_registry)
            data = read_json_state(self.registry_file, validate=_validate_registry)
            self._hydrate(data["submissions"])
            self._publish_registry(data)

    def _publish_registry(self, data):
        """Derived mirrors follow the committed authoritative registry."""
        from harness.io_utils import atomic_write_json
        if self.submissions_dir != self.project_root / "submissions":
            root_sub_dir = self.project_root / "submissions"
            root_sub_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(root_sub_dir / "registry.json", data, indent=2, ensure_ascii=False)
        for record in data["submissions"]:
            for directory in {self.submissions_dir, self.project_root / "submissions"}:
                target = directory / record["id"]
                if target.is_dir():
                    atomic_write_json(target / "metadata.json", record, indent=2, ensure_ascii=False)
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
        for name, value, default in (("baseline_csv", baseline_csv, None),
                                     ("ensemble_components", ensemble_components, None),
                                     ("min_diff_pct", min_diff_pct, 2.0)):
            if value != default:
                warnings.warn(f"{name} is deprecated and ignored; it does not enforce submission gates.",
                              DeprecationWarning, stacklevel=2)
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
        if self.read_only:
            raise ValueError("SubmissionManager is read-only")
        safe_name = _validate_payload_filename(file_name)
        source = Path(candidate_csv).expanduser()
        if not source.is_absolute():
            source = self.project_root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Prediction file does not exist: {source}")

        from harness.artifacts import validate_test_prediction

        validate_test_prediction(
            source, task_spec=plugin_context.task_spec, plugin=plugin, project_root=plugin_context.project_root
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
            returned_name = final_path.name
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

            _validate_payload_filename(returned_name)
            _validate_payload_filename(final_path.name)
            for payload in target_dir.rglob("*"):
                if payload.name.casefold() in RESERVED_SUBMISSION_FILENAMES:
                    raise ValueError(f"Plugin created reserved metadata filename: {payload.name!r}")

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
                "scope": "validation",
                "dispatch_authorized": False,
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
            from harness.artifacts import sha256
            from harness.audit_gate import semantic_prediction_hash
            entry.payload_sha256 = sha256(final_path)
            entry.prediction_sha256 = semantic_prediction_hash(predictions, plugin_context.task_spec)
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

    def preserve_file_submission(
        self,
        *,
        file_path: str | Path,
        notes: str = "",
        reported_score: Optional[float] = None,
        reported_partition: Optional[str] = None,
    ) -> SubmissionEntry:
        """Preserve an arbitrary submission artifact (zip, csv, etc.) for safekeeping.

        Distinct from `register_plugin_submission`:
        - Does NOT require an experiment directory or plugin test predictions.
        - Generates a dedicated PRESERVATION_RECEIPT.json (record_type: preservation_only).
        - Status is PRESERVED_UNVERIFIED (dispatch_authorized: False).
        - Keeps official_scores strictly empty; unverified user scores go to reported_scores.
        - Constrains sources to the project root to prevent reading external host files.
        - Rejects reserved metadata names to prevent payload collisions.
        - Fingerprints the exact payload SHA-256 and records byte count.
        """
        if self.read_only:
            raise ValueError("SubmissionManager is read-only")
        import math

        source = Path(file_path).expanduser()
        _validate_payload_filename(source.name)
        if not source.is_absolute():
            source = self.project_root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Submission artifact does not exist: {source}")

        # Security boundary: constrain source file to the project root
        try:
            source.relative_to(self.project_root.resolve())
        except ValueError:
            raise ValueError(
                f"Preserved file must be located within the project root ({self.project_root}): {source}"
            )

        safe_name = _validate_payload_filename(source.name)

        if reported_score is not None:
            if (
                isinstance(reported_score, bool)
                or not isinstance(reported_score, (int, float))
                or math.isnan(reported_score)
                or math.isinf(reported_score)
            ):
                raise ValueError("reported_score must be a finite number and cannot be a boolean.")

        sub_id = f"{self.get_next_submission_id()}_preserved"
        target_dir = self.submissions_dir / sub_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target_path = target_dir / safe_name
        receipt_path = target_dir / "PRESERVATION_RECEIPT.json"

        try:
            shutil.copy2(source, target_path)
            from harness.artifacts import sha256

            payload_hash = sha256(target_path)
            byte_count = target_path.stat().st_size

            # Check that copy was uncorrupted
            if sha256(source) != payload_hash:
                raise IOError("Preserved file checksum mismatch after copy")

            from harness.io_utils import atomic_write_json

            receipt = {
                "schema_version": "preservation_receipt.v1",
                "record_type": "preservation_only",
                "verification_status": "unverified",
                "dispatch_authorized": False,
                "source_file": str(source),
                "archive_path": str(target_path),
                "payload_sha256": payload_hash,
                "byte_count": byte_count,
                "preserved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "notes": notes,
                "reported_score": (
                    float(reported_score) if reported_score is not None else None
                ),
                "reported_partition": reported_partition,
            }
            atomic_write_json(receipt_path, receipt, indent=2, ensure_ascii=False)

            reported_scores_dict: Dict[str, Optional[float]] = {}
            if reported_score is not None:
                key = (
                    f"reported_{reported_partition}"
                    if reported_partition
                    else "reported_score"
                )
                reported_scores_dict[key] = float(reported_score)

            entry = SubmissionEntry(
                id=sub_id,
                exp_id="preserved",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                status="PRESERVED_UNVERIFIED",
                local_cv_scores={},
                official_scores={},  # Keep official_scores strictly empty for unverified entries
                score_deltas={},
                reported_scores=reported_scores_dict,
                candidate_path=str(source),
                archive_path=str(target_path),
                receipt_path=str(receipt_path),
                notes=notes,
                payload_sha256=payload_hash,
                official_score_source="user_reported_unverified",
            )
            atomic_write_json(
                target_dir / "metadata.json",
                asdict(entry),
                indent=2,
                ensure_ascii=False,
            )
            if self.submissions_dir != self.project_root / "submissions":
                root_target_dir = self.project_root / "submissions" / sub_id
                root_target_dir.mkdir(parents=True, exist_ok=False)
                shutil.copy2(target_path, root_target_dir / safe_name)
                shutil.copy2(
                    receipt_path, root_target_dir / "PRESERVATION_RECEIPT.json"
                )
                shutil.copy2(
                    target_dir / "metadata.json", root_target_dir / "metadata.json"
                )

            # Invariant check: verify target_path hash was not mutated by receipt/metadata writes
            if sha256(target_path) != payload_hash:
                raise IOError("Preserved file was corrupted during metadata generation")

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
        metric_scores = dict(custom_scores or {})
        metric_scores.update(scores or {})
        metric_scores.update({k: v for k, v in kwargs.items() if v is not None})
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
            if isinstance(primary_score, bool):
                raise ValueError("Score must not be boolean")
            metric_scores[primary_key] = float(primary_score)
        return self._commit_score(sub_id, metric_scores=metric_scores,
                                  split_type=split_type, notes=notes)

    def _apply_score_update(self, entry, *, metric_scores, split_type, notes="",
                            observation_id=None, score_source="declared"):
        scores_dict = {"split_type": split_type}
        for key, value in metric_scores.items():
            if value is not None:
                if isinstance(value, bool):
                    raise ValueError("Score must not be boolean")
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Score must be finite")
                scores_dict[key] = value
        entry.official_scores = scores_dict
        entry.official_observation_id = observation_id
        entry.official_score_source = score_source

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
                d = float(lb_val) - float(cv_val)
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

    def _commit_score(self, sub_id, *, expected=None, **updates):
        if self.read_only:
            raise ValueError("SubmissionManager is read-only")
        from harness.state_store import transaction, workspace_lock, StateConflictError
        baseline = expected or next((r for r in self._snapshot if r["id"] == sub_id), None)
        new_entry = baseline is None and sub_id in self.submissions
        if new_entry:
            baseline = asdict(self.submissions[sub_id])
        if baseline is None:
            raise KeyError(f"Submission ID '{sub_id}' not found in registry.")
        baseline = asdict(SubmissionEntry(**baseline))
        protected = ("dispatch_intent_id", "payload_sha256", "prediction_sha256",
                     "official_scores", "official_observation_id", "official_score_source")
        with workspace_lock(self.submissions_dir):
            with transaction(self.registry_file, {"submissions": []}, validate=_validate_registry) as data:
                record = next((r for r in data["submissions"] if r["id"] == sub_id), None)
                if record is None and new_entry:
                    record = copy.deepcopy(baseline)
                    data["submissions"].append(record)
                if record is None:
                    raise KeyError(sub_id)
                entry = SubmissionEntry(**record)
                latest = asdict(entry)
                for key in protected:
                    if latest[key] != baseline[key]:
                        raise StateConflictError((sub_id, key))
                self._apply_score_update(entry, **updates)
                record.update(asdict(entry))
            self._hydrate(data["submissions"])
            self._publish_registry(data)
        return self.submissions[sub_id]

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

    def record_official_observation(self, sub_id, *, client, provider_ref, partition="public"):
        from harness.submission_gate import SubmissionGate
        expected = asdict(self.submissions[sub_id])
        if not expected["dispatch_intent_id"]:
            raise ValueError("Provider observations require a verified dispatch intent")
        observation = SubmissionGate(self.project_root).resolve_official_score(
            expected["dispatch_intent_id"], client=client, provider_ref=provider_ref, partition=partition)
        if (observation.get("intent_id") != expected["dispatch_intent_id"]
                or observation.get("partition") != partition
                or str(observation.get("provider_ref")) != str(provider_ref)
                or not isinstance(observation.get("metric"), str) or not observation["metric"].strip()
                or not isinstance(observation.get("observation_id"), str) or not observation["observation_id"].strip()
                or partition not in {"public", "private"}
                or observation.get("source") != "provider_api"):
            raise ValueError("Invalid provider observation identity or provenance")
        for key in ("payload_sha256", "prediction_sha256"):
            if observation.get(key) is not None and observation[key] != expected[key]:
                raise ValueError(f"Provider observation {key} mismatch")
        score = observation.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Provider score must be a finite number")
        return self._commit_score(sub_id, expected=expected,
            metric_scores={observation["metric"]: score}, split_type=partition,
            observation_id=observation["observation_id"], score_source="provider_api")

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
            "> - 📦 `PRESERVED_UNVERIFIED`: Preserved artifact for safekeeping; not verified or dispatch-authorized.",
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
                if s.status == "SCORED":
                    status_icon = "🎯 `SCORED`"
                elif s.status == "SUBMITTED_PENDING":
                    status_icon = "⏳ `PENDING`"
                elif s.status == "PRESERVED_UNVERIFIED":
                    status_icon = "📦 `PRESERVED`"
                elif s.status == "DISQUALIFIED":
                    status_icon = "❌ `DISQUALIFIED`"
                else:
                    status_icon = "🔒 `LOCAL_VERIFIED`"
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
                elif s.status == "PRESERVED_UNVERIFIED":
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
                if s.status == "PRESERVED_UNVERIFIED" and s.reported_scores:
                    rep_str = ", ".join(f"{k}:{v}" for k, v in s.reported_scores.items())
                    notes_clean = f"Reported: {rep_str} | {s.notes}" if s.notes else f"Reported: {rep_str}"

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
