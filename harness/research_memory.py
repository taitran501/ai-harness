"""Durable research continuity outside immutable experiment evidence.

Configuration equivalence is deterministic; mechanism similarity is an advisory,
not proof that two scientific hypotheses are equivalent. Decisions remain declared
interpretations unless a prospective structured rule is machine-evaluated.
"""
from __future__ import annotations

import copy
import html
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from harness.artifacts import contract_identity, digest, validate_run
from harness.io_utils import atomic_write_text
from harness.state_store import transaction, workspace_lock

_SYNC_CACHE: dict = {}
# Key: (absolute folder path, sync identity digest, per-file (name, mtime_ns,
# size) tuple over every top-level artifact file). The identity digest is the
# task+scorer contract that produced the cached record; the per-file tuple
# invalidates the cache when ANY artifact file changes (not only
# manifest/config/metrics). See ResearchMemory.sync().

UNVERIFIED_AGENT_NOTE = "[UNVERIFIED AGENT NOTE - HYPOTHESIS ONLY]"


def render_markdown_progress(
    scope: str,
    entries: Sequence[Mapping[str, Any]],
    verified_runs: Optional[Sequence[Mapping[str, Any]]] = None,
    verified_decisions: Optional[Sequence[Mapping[str, Any]]] = None,
    validated_incumbent: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render verified research state separately from agent-declared notes.

    ``verified_runs``, ``verified_decisions``, and ``validated_incumbent`` are
    projections supplied by the live validators. Checkpoint prose is always
    rendered as an unverified hypothesis and never contributes to metrics.
    """
    verified_runs = [
        dict(run) for run in (verified_runs or [])
        if run.get("evidence_status") == "verified_full"
    ]
    verified_decisions = [
        dict(decision) for decision in (verified_decisions or [])
        if decision.get("verification_status") == "machine_evaluated"
    ]
    entries = [dict(entry) for entry in (entries or [])]

    def markdown(value):
        text = html.escape(str(value), quote=False)
        return re.sub(r"([\\`*_{}\[\]()#+!|])", r"\\\1", text)

    groups: Dict[str, Dict[str, Any]] = {}
    for run in verified_runs:
        group = str(run.get("comparison_group") or "unknown")
        utility = run.get("utility")
        if not isinstance(utility, (int, float)) or not math.isfinite(float(utility)):
            continue
        previous = groups.get(group)
        if previous is None or float(utility) > float(previous["utility"]):
            groups[group] = run

    checkpoints = [entry for entry in entries if entry.get("kind") == "checkpoint"]
    checkpoint = checkpoints[-1] if checkpoints else None
    lines = [
        "<!-- ai-harness:research-progress:v1 -->",
        "# Research progress",
        "",
        "Generated from research_memory.json and live-validated full runs. "
        "Record notes through the Bridge; manual edits to this generated file are replaced.",
        "",
        f"Task/scorer scope: `{markdown(scope)}`",
        "",
        "## Resume here",
        "",
        "### Verified state",
        "",
        f"- Verified completed full runs: {len(verified_runs)}",
    ]
    if groups:
        best_runs = sorted(groups.items(), key=lambda item: item[0])
        if len(best_runs) == 1:
            group, run = best_runs[0]
            lines.append(
                f"- Best verified CV: `{markdown(run.get('exp_id', 'unknown'))}` "
                f"({float(run.get('score', float('nan'))):.6g}; comparison identity `{markdown(group[:12])}`)"
            )
        else:
            lines.append("- Best verified CV by comparison identity:")
            for group, run in best_runs:
                lines.append(
                    f"  - `{markdown(run.get('exp_id', 'unknown'))}` "
                    f"({float(run.get('score', float('nan'))):.6g}; `{markdown(group[:12])}`)"
                )
    else:
        lines.append("- Best verified CV: unknown (no live-validated full run)")

    lines.append(f"- Machine-verified decisions: {len(verified_decisions)}")
    for decision in verified_decisions[-3:]:
        lines.append(
            f"  - `{markdown(decision.get('evaluation_id', 'evaluation'))}`: "
            f"{markdown(decision.get('outcome', 'recorded'))}"
        )
    if validated_incumbent:
        ref = validated_incumbent.get("evidence_ref") or {}
        score = validated_incumbent.get("score")
        score_text = f" ({float(score):.6g})" if isinstance(score, (int, float)) else ""
        lines.append(
            f"- Validated incumbent: `{markdown(ref.get('id', 'unknown'))}`{score_text} "
            f"(evaluation `{markdown(validated_incumbent.get('evaluation_id', 'unknown'))}`)"
        )
    else:
        lines.append("- Validated incumbent: unknown (no current machine-verified incumbent state)")

    considered = {
        exp
        for entry in entries
        if entry.get("kind") == "decision"
        for exp in (entry.get("evidence") or [])
    }
    pending = [run.get("exp_id") for run in verified_runs if run.get("exp_id") not in considered]
    lines.append(f"- Uninterpreted verified full results: {markdown(', '.join(pending) or 'none')}")

    lines.extend([
        "",
        f"### Unverified Agent Notes {UNVERIFIED_AGENT_NOTE}",
        "",
        f"> [!NOTE] {UNVERIFIED_AGENT_NOTE}",
    ])
    if checkpoint:
        lines.extend([
            f"> **Checkpoint summary**: {markdown(checkpoint.get('conclusion', ''))}",
            f"> **Next action**: {markdown(checkpoint.get('next_action', ''))}",
        ])
        for question in checkpoint.get("open_questions") or []:
            lines.append(f"> **Open question**: {markdown(question)}")
    else:
        lines.append("> No checkpoint recorded.")

    insights = [entry for entry in entries if entry.get("kind") == "insight"]
    lines.extend([
        "",
        "## Insight status",
        "",
        "| Insight | Status | Required step | Next action |",
        "| --- | --- | --- | --- |",
    ])
    for insight in insights:
        state = insight.get("progress_state") or {}
        cells = [
            f"{insight.get('title', 'Untitled')} ({insight.get('id', '')})",
            state.get("status", "needs evidence review"),
            state.get("required_step", "Review verified results"),
            state.get("next_action", "Plan a bounded probe"),
        ]
        lines.append("| " + " | ".join(markdown(cell).replace("\n", " ") for cell in cells) + " |")
    if not insights:
        lines.append("| No insights recorded | needs notes | Record an observation and mechanism | Plan a bounded probe |")

    lines.extend(["", "## Chronological journal", ""])
    history = sorted(
        [*entries, *verified_runs],
        key=lambda item: (item.get("timestamp", 0), item.get("id", item.get("exp_id", ""))),
    )
    labels = (
        "title", "parent_insight_id", "insight_id", "probe_plan_id", "exp_id", "action",
        "family_id", "source_refs", "review_id", "map_digest", "run_count_at_review",
        "changes_since_last_review", "supported_mechanisms", "rejected_mechanisms",
        "highest_priority_untouched", "next_probe_rationale", "budget_limit", "revisit_conditions",
        "observation", "mechanism", "hypothesis", "control", "intervention", "expected_observation",
        "alternative_explanation", "decision_rule", "budget", "stop_rule", "outcome", "score",
        "conclusion", "lesson", "revisit_when", "next_action", "evidence", "open_questions", "insight_ids",
        "claimed_outcome", "effective_outcome", "verification_status", "rule_id", "evaluation_id", "reason_codes",
    )
    verified_ids = {run.get("exp_id") for run in verified_runs}
    for item in history:
        timestamp = item.get("timestamp", 0)
        stamp = datetime.fromtimestamp(float(timestamp or 0), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        is_checkpoint = item.get("kind") == "checkpoint"
        kind = "note [unverified]" if is_checkpoint else markdown(item.get("kind", "verified_full_result"))
        identity = item.get("id", item.get("exp_id", ""))
        lines.extend([f"### {stamp} — {kind} `{markdown(identity)}`", ""])
        if is_checkpoint:
            lines.append(f"> [!NOTE] {UNVERIFIED_AGENT_NOTE}")
        for key in labels:
            value = item.get(key)
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                value = "; ".join(str(part) for part in value)
            text = markdown(value).replace("\n", "\n> ")
            lines.append(f"> **{key}**: {text}\n")
        evidence = item.get("evidence", [])
        evidence_status = item.get("evidence_status", "declared")
        if evidence:
            evidence_status += "; evidence currently verified=" + str(all(exp in verified_ids for exp in evidence)).lower()
        lines.extend([f"> **evidence_status**: {markdown(evidence_status)}", ""])
    if not history:
        lines.append("No research events or verified full results recorded for this scope.")
    return "\n".join(lines) + "\n"


class _EntryRecord(dict):
    """A ledger entry that is a real dict but also exposes fields as attributes.

    Keeping it a dict subclass means every existing consumer (``entry["kind"]``,
    ``entry.get(...)``, ``json.dumps(entry)``) keeps working, while agents that
    reach for ``entry.kind`` also get a natural answer.
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value):
        self[name] = value


class _EntryList(list):
    """List of :class:`_EntryRecord` that also supports the legacy callable form.

    ``memory.entries`` and ``memory.entries()`` therefore both yield the same
    list, so code written against either convention keeps working.
    """

    def __call__(self) -> "_EntryList":
        return self


class _FrontierDescriptor:
    """Descriptor enabling both instance property and classmethod frontier access."""

    def __get__(self, instance, owner=None):
        if instance is None:
            def _class_frontier(project_root: str | Path = ".") -> _EntryList:
                return owner.load(project_root).frontier
            return _class_frontier
        ctx = instance.context()
        return _EntryList(ctx.get("research_frontier", []))


DECISION_OUTCOME_ALIASES = {
    # Supported
    "supported": "supported",
    "support": "supported",
    "accepted": "supported",
    "accept": "supported",
    "keep": "supported",
    "adopt": "supported",
    "success": "supported",
    "passed": "supported",
    "pass": "supported",
    "positive": "supported",
    "verified": "supported",
    "verified_and_ready": "supported",
    # Rejected
    "rejected": "rejected",
    "reject": "rejected",
    "discarded": "rejected",
    "discard": "rejected",
    "drop": "rejected",
    "failed": "rejected",
    "fail": "rejected",
    "negative": "rejected",
    # Inconclusive
    "inconclusive": "inconclusive",
    "uncertain": "inconclusive",
    "unknown": "inconclusive",
    "mixed": "inconclusive",
    "flat": "inconclusive",
    # Deferred
    "deferred": "deferred",
    "defer": "deferred",
    "postponed": "deferred",
    "postpone": "deferred",
    "later": "deferred",
    "pause": "deferred",
    "paused": "deferred",
}


def normalize_decision_outcome(val: Any) -> str:
    s = str(val or "").strip().lower()
    return DECISION_OUTCOME_ALIASES.get(s, s)


class ResearchMemory:
    frontier = _FrontierDescriptor()
    research_frontier = _FrontierDescriptor()

    def __init__(
        self,
        project_root: str | Path = ".",
        artifacts_dir: Optional[str | Path] = None,
        task_spec: Optional[Dict[str, Any]] = None,
        plugin: Any = None,
    ):
        self.root = Path(project_root).resolve()
        if task_spec is None:
            task_path = self.root / "configs" / "task.yaml"
            if task_path.exists():
                try:
                    from harness.task_spec import load_task_spec

                    task_spec = load_task_spec(task_path)
                except Exception:
                    task_spec = {}
            else:
                task_spec = {}
        if artifacts_dir is None:
            meta = getattr(task_spec, "metadata", None) if isinstance(task_spec, dict) else getattr(task_spec, "metadata", None)
            task_id = ""
            if isinstance(meta, dict):
                task_id = str(meta.get("id") or meta.get("task_name") or "").strip()
            art_base = self.root / ".ai-harness" / "artifacts"
            found = None
            if task_id and (art_base / task_id).is_dir():
                found = art_base / task_id
            elif art_base.is_dir():
                for child in art_base.iterdir():
                    if child.is_dir() and (child / "research_memory.json").exists():
                        found = child
                        break
                    elif child.is_dir() and not found:
                        found = child
            artifacts_dir = found or (art_base / task_id if task_id else art_base)

        self.artifacts = Path(artifacts_dir)
        self.path = self.artifacts / "research_memory.json"
        self.progress_path = self.artifacts / "RESEARCH_PROGRESS.md"
        self.spec = task_spec
        self.plugin = plugin
        self.identity = contract_identity(task_spec, plugin)
        # Scope key: stable across benign task.yaml edits (data paths,
        # runtime, hygiene allow-lists) because it uses only the task's
        # declared identity and the scorer implementation. Hashing the full
        # task metadata (the legacy behavior) orphaned every recorded
        # insight/decision/checkpoint whenever the task config was touched.
        meta = getattr(task_spec, "metadata", None)
        task_id = ""
        if isinstance(meta, dict):
            task_id = str(meta.get("id") or meta.get("task_name") or "").strip()
        self._task_key = task_id or self.identity["task_sha256"]
        if task_id:
            self.scope = digest(
                {"task_id": task_id, "scorer_sha256": self.identity["scorer_sha256"]}
            )
        else:
            self.scope = digest(self.identity)
        # sync() cache-invalidation key: the task+scorer contract identity
        # under which each cached record was validated. A second spec/plugin
        # in the same process (same artifacts dir) must never be served the
        # first contract's cached "verified_full" claim.
        self._sync_identity_key = digest(self.identity)

    @classmethod
    def load(
        cls,
        project_root: str | Path = ".",
        artifacts_dir: Optional[str | Path] = None,
        task_spec: Optional[Dict[str, Any]] = None,
        plugin: Any = None,
    ) -> "ResearchMemory":
        """Load research memory directly from a project root or artifacts directory."""
        return cls(project_root, artifacts_dir=artifacts_dir, task_spec=task_spec, plugin=plugin)

    def _read(self):
        if not self.path.exists():
            return {"entries": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _valid_scopes(self, data) -> set:
        """Current scope plus scopes this task's memory file has known.

        The file is already task-scoped by path; the scope field distinguishes
        scope-key versions over time. Research continuity must survive
        scope-key migrations and benign contract edits, so entries from
        previously-known scopes stay addressable instead of being silently
        orphaned (the scope value remains on each entry for provenance).

        Legacy files (written before scope_history existed) get one-time
        adoption when the task writing them next appends an entry: the scopes
        already present are attributed to that task's bucket (a workspace's
        artifacts directory belongs to one task lineage), so pre-migration
        entries stay addressable while different tasks sharing one artifacts
        directory remain isolated. The scope value stays on each entry for
        provenance.
        """
        valid = {self.scope}
        history = data.get("scope_history")
        if history is None:
            for entry in data.get("entries", []):
                if entry.get("scope"):
                    valid.add(entry["scope"])
        elif isinstance(history, dict):
            valid.update(history.get(self._task_key) or [])
        else:
            # Defensive: a flat list cannot be attributed to a task, so it is
            # only honored for the task currently writing the file.
            valid.update(history)
        return valid

    @property
    def entries(self) -> _EntryList:
        """All entries for this task, newest last.

        Works as a property (``memory.entries[-3:]``) and as the legacy callable
        (``memory.entries()``). Each element supports both attribute access
        (``entry.kind``) and dict access (``entry["kind"]``).
        """
        data = self._read()
        valid = self._valid_scopes(data)
        return _EntryList(
            _EntryRecord(e)
            for e in data.get("entries", [])
            if e.get("scope") in valid
        )

    def _append(self, kind, **fields):
        entry = {"id": uuid.uuid4().hex, "scope": self.scope, "kind": kind,
                 "timestamp": time.time(), **fields}
        with workspace_lock(self.path.parent):
            with transaction(self.path, {"schema_version": 1, "entries": []}) as state:
                history = state.get("scope_history")
                if history is None:
                    # Migrate the legacy file: attribute every scope present
                    # so far to the task writing it (workspace = one task
                    # lineage) so pre-migration entries stay addressable.
                    history = {
                        self._task_key: [
                            e["scope"] for e in state.get("entries", []) if e.get("scope")
                        ]
                    }
                    state["scope_history"] = history
                if not isinstance(history, dict):
                    history = state["scope_history"] = {self._task_key: list(history)}
                bucket = history.setdefault(self._task_key, [])
                if self.scope not in bucket:
                    bucket.append(self.scope)
                state.setdefault("entries", []).append(entry)
            self._publish_progress(self.sync())
        return entry

    @staticmethod
    def _markdown(value):
        text = html.escape(str(value), quote=False)
        return re.sub(r"([\\`*_{}\[\]()#+!|])", r"\\\1", text)

    def _validated_progress_decisions(self, entries):
        """Return only decisions backed by indexed, intact evaluation receipts."""
        from harness.decision_rules import policy_directory, read_decision_evaluation
        from harness.project import HarnessProject

        directory = policy_directory(HarnessProject.discover(self.root))
        index_path = directory / "evaluations.json"
        try:
            indexed = set(json.loads(index_path.read_text(encoding="utf-8")).get("evaluations", {}).values())
        except (OSError, ValueError, TypeError):
            return []

        verified = []
        for entry in entries:
            if entry.get("kind") != "decision" or entry.get("verification_status") != "machine_evaluated":
                continue
            evaluation_id = entry.get("evaluation_id")
            if not evaluation_id or evaluation_id not in indexed:
                continue
            try:
                receipt = read_decision_evaluation(directory / f"{evaluation_id}.json")
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (receipt.get("outcome") != entry.get("outcome")
                    or receipt.get("evaluation_id") != evaluation_id
                    or dict(entry.get("evaluation") or {}) != receipt):
                continue
            verified.append(dict(entry))
        return verified

    def _validated_progress_incumbent(self):
        """Read the current incumbent only from replayed machine evaluation state."""
        try:
            from harness.decision_rules import policy_directory, read_decision_evaluation
            from harness.project import HarnessProject
            from harness.stagnation import StagnationController

            project = HarnessProject.discover(self.root)
            states = StagnationController(project).reconcile_results()
            candidates = [
                state for state in states.values()
                if state.get("incumbent_evaluation_id") and state.get("incumbent_evidence_ref")
            ]
            if not candidates:
                return None
            state = max(candidates, key=lambda item: item.get("updated_at", ""))
            evaluation_id = state["incumbent_evaluation_id"]
            directory = policy_directory(project)
            receipt = read_decision_evaluation(directory / f"{evaluation_id}.json")
            if (receipt.get("outcome") != "supported"
                    or receipt.get("candidate") != state["incumbent_evidence_ref"]):
                return None
            return {
                "evidence_ref": dict(state["incumbent_evidence_ref"]),
                "evaluation_id": evaluation_id,
                "score": receipt.get("candidate_score"),
            }
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _publish_progress(self, runs):
        """Refresh a derived human-readable journal; JSON remains authoritative."""
        marker = "<!-- ai-harness:research-progress:v1 -->"
        try:
            with workspace_lock(self.path.parent):
                previous = self.progress_path.read_text(encoding="utf-8") if self.progress_path.exists() else None
                if previous is not None and not previous.startswith(marker):
                    return {"progress_path": None,
                            "progress_warning": "RESEARCH_PROGRESS.md is user-owned; it was preserved."}
                # Re-read under the lock so a concurrent decision is not erased
                # by publishing an older ledger snapshot.
                entries = self.entries()
                verified_runs = [run for run in runs if run.get("evidence_status") == "verified_full"]
                render_entries = []
                for entry in entries:
                    rendered = dict(entry)
                    if entry.get("kind") == "insight":
                        rendered["progress_state"] = self._trace_state(entry, entries, verified_runs)
                    render_entries.append(rendered)
                content = render_markdown_progress(
                    self.scope,
                    render_entries,
                    verified_runs=verified_runs,
                    verified_decisions=self._validated_progress_decisions(entries),
                    validated_incumbent=self._validated_progress_incumbent(),
                )
                if content != previous:
                    atomic_write_text(self.progress_path, content)
                return {"progress_path": str(self.progress_path), "progress_warning": None}
        except (OSError, UnicodeError) as exc:
            # Report failures must not turn a committed note into a failed action
            # that the assistant retries, duplicating the canonical event.
            return {"progress_path": None, "progress_warning": f"Could not refresh Markdown progress: {exc}"}

    def fingerprint(self, cfg):
        """Compare resolved declared execution, ignoring names and narrative metadata.

        This is NOT a data identity or a semantic-equivalence guarantee. An explicit
        revisit reason permits replication, changed data, or changed implementation.
        """
        base_path = self.root / "configs" / "base.yaml"
        if not base_path.is_file():
            base_path = Path(__file__).parent / "defaults" / "base.yaml"
        if "resolved_runtime" in cfg:
            # A completed runner config already contains its historical defaults.
            # Do not inject today's new base settings into immutable old execution.
            # Package defaults also normalize direct UniversalRunner API calls.
            base_path = Path(__file__).parent / "defaults" / "base.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

        def merge(left, right):
            result = copy.deepcopy(left)
            for key, value in right.items():
                result[key] = merge(result[key], value) if (
                    isinstance(value, dict) and isinstance(result.get(key), dict)
                ) else copy.deepcopy(value)
            return result

        resolved = merge(base, cfg)
        for key in ("experiment", "grounding", "assumption_register", "research", "resolved_runtime"):
            resolved.pop(key, None)
        resolved.setdefault("runtime", {})["seed"] = self.spec.resolve_seed(
            resolved.get("runtime", {}).get("seed", 42))
        # Leaderboard location is reporting, not the scientific intervention.
        resolved.get("paths", {}).pop("leaderboard_path", None)
        def normalize(value):
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return value
        return digest({"scope": self.scope, "execution": normalize(resolved)})

    @staticmethod
    def _narrative(cfg):
        exp = cfg.get("experiment", {})
        return {key: str(exp.get(key, ""))[:2000] for key in
                ("hypothesis", "intervention", "problem_addressed")}

    def _folder_file_key(self, folder: Path):
        """Per-file (name, mtime_ns, size) over every top-level artifact file.

        The previous cache key covered only manifest/config/metrics mtimes, so
        tampering any other sealed artifact (e.g. OOF predictions) between two
        syncs left a cached ``verified_full`` record that a fresh
        ``validate_run`` rejects. mtime+size mirrors the
        ``artifacts._HASH_CACHE`` idiom: a tamper preserving both is exactly
        the same tamper-strength the hash cache itself relies on, and any
        other change forces a fresh ``validate_run``. An OSError here
        propagates to the per-folder handler (the run is skipped, not cached).
        """
        return tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(folder.iterdir())
            if p.is_file()
        )

    def sync(self):
        """Revalidate artifacts on every read; a cached claim never verifies evidence.

        A cached record is reused only when the run folder's entire file
        surface (per-file mtime+size) is unchanged AND the validating
        contract (task+scorer identity) is the same one that produced the
        record; anything else forces a fresh ``validate_run``.
        """
        runs = []
        for folder in sorted(self.artifacts.iterdir()) if self.artifacts.exists() else []:
            manifest_path = folder / "artifact_manifest.json"
            cfg_path = folder / "run_config.yaml"
            metrics_path = folder / "metrics.json"
            if not folder.is_dir() or not manifest_path.is_file():
                continue
            try:
                cache_key = (
                    str(folder.resolve()),
                    self._sync_identity_key,
                    self._folder_file_key(folder),
                )
                if cache_key in _SYNC_CACHE:
                    runs.append(_SYNC_CACHE[cache_key])
                    continue

                manifest = validate_run(folder, task_spec=self.spec, plugin=self.plugin)
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                score = float(metrics["primary_score"])
                value = score if metrics["direction"] == "maximize" else -score
                if not math.isfinite(value):
                    continue
                diag = {}
                if "diagnostics.json" in manifest["files"]:
                    diag = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
                record = {
                    "exp_id": manifest["exp_id"],
                    "path": str(folder),
                    "fingerprint": self.fingerprint(cfg),
                    **self._narrative(cfg),
                    "model": cfg.get("model", {}),
                    "insight_id": (cfg.get("research") or {}).get("insight_id"),
                    "score": score,
                    "utility": value,
                    "comparison_group": digest({
                        key: manifest.get(key)
                        for key in (
                            "task_sha256",
                            "scorer_sha256",
                            "custom_scorer_sha256",
                            "data_sha256",
                            "split_sha256",
                        )
                    }),
                    "bottlenecks": diag.get("critical_bottlenecks", [])[:5],
                    "timestamp": manifest_path.stat().st_mtime,
                    "evidence_status": "verified_full",
                    "evidence_sha256": digest(manifest),
                }
                _SYNC_CACHE[cache_key] = record
                runs.append(record)
            except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError):
                continue
        # Imported/remote completions are evaluated on the canonical workspace,
        # where their preregistration and fixed control remain available.
        for run in runs:
            cfg_path = Path(run["path"]) / "run_config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            rule_id = cfg.get("research", {}).get("decision_rule_id")
            if rule_id:
                from harness.decision_rules import evaluate_decision_rule, load_registration
                registration = load_registration(self.root, rule_id)
                evaluate_decision_rule(registration["rule"],
                    {"kind": "full_run", "id": run["exp_id"], "partition": cfg.get("validation", {}).get("partition", "oof")},
                    project=self.root, task_spec=self.spec, plugin=self.plugin, registration=registration)
        return sorted(runs, key=lambda r: (r["timestamp"], r["exp_id"]))

    @staticmethod
    def _compact(record):
        return {key: record[key][:600] if isinstance(record[key], str) else record[key] for key in
                ("id", "kind", "exp_id", "hypothesis", "intervention", "score",
                 "conclusion", "next_action", "outcome", "evidence_status", "path",
                 "insight_id", "parent_insight_id", "probe_plan_id", "title", "observation", "mechanism", "lesson", "revisit_when",
                 "control", "expected_observation", "alternative_explanation",
                 "decision_rule", "budget", "stop_rule", "claimed_outcome", "effective_outcome", "verification_status",
                 "rule_id", "evaluation_id", "reason_codes", "family_id", "source_refs",
                 "review_id", "map_digest", "run_count_at_review", "changes_since_last_review",
                 "supported_mechanisms", "rejected_mechanisms", "highest_priority_untouched",
                 "next_probe_rationale", "budget_limit", "revisit_conditions") if key in record}

    @staticmethod
    def _text(value, field):
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError(f"{field} requires non-empty text of at most 4000 characters")
        return value.strip()

    def _insight(self, insight_id, entries=None):
        insight = next((entry for entry in (self.entries() if entries is None else entries)
                        if entry["kind"] == "insight" and entry["id"] == insight_id), None)
        if insight is None:
            raise ValueError("insight_id must identify an insight in the current task/scorer scope")
        return insight

    def _evidence(self, evidence):
        if evidence is None:
            return []
        if isinstance(evidence, str):
            # Agents often pass a single ID (or comma-separated IDs) instead of
            # a list; accept that shape instead of forcing a retry.
            evidence = [part.strip() for part in evidence.replace(";", ",").split(",") if part.strip()]
        if not isinstance(evidence, list) or any(not isinstance(exp, str) for exp in evidence):
            raise ValueError("evidence must be a list of verified experiment IDs (single or comma-separated IDs also accepted)")
        available = {run["exp_id"] for run in self.sync()}
        if any(exp not in available for exp in evidence):
            raise ValueError("Evidence references must identify verified full runs for the current task")
        return list(dict.fromkeys(evidence))

    def record_insight(self, *, title, observation, mechanism, next_action, evidence=None,
                       parent_insight_id=None, session_id=None, family_id: Optional[str] = "unclassified",
                       source_refs: Optional[List[str]] = None):
        from harness.macro_research import family_id as normalize_family
        if source_refs is not None and (not isinstance(source_refs, list) or
                any(not isinstance(ref, str) or not ref.strip() for ref in source_refs)):
            raise ValueError("source_refs must be a list of non-empty source references")
        fields = {key: self._text(value, key) for key, value in {
            "title": title, "observation": observation, "mechanism": mechanism,
            "next_action": next_action}.items()}
        with workspace_lock(self.path.parent):
            if parent_insight_id:
                self._insight(parent_insight_id)
            # Allow empty evidence for hypothesis/brainstorm insights that have
            # not yet been verified by a full run. Non-empty evidence is still
            # validated against verified run IDs to prevent fabricated links.
            evidence_list = self._evidence(evidence) if evidence else []
            return self._append("insight", **fields, evidence=evidence_list,
                parent_insight_id=parent_insight_id, session_id=session_id,
                family_id=normalize_family(family_id), source_refs=list(source_refs or []),
                evidence_status="declared_insight")

    def plan_probe(self, *, insight_id, control, intervention, expected_observation,
                   alternative_explanation, decision_rule, budget, stop_rule, session_id=None):
        from harness.decision_rules import DecisionRuleV1, register_decision_rule
        structured = isinstance(decision_rule, (dict, DecisionRuleV1))
        fields = {key: self._text(value, key) for key, value in {
            "control": control, "intervention": intervention, "expected_observation": expected_observation,
            "alternative_explanation": alternative_explanation,
            "budget": budget, "stop_rule": stop_rule}.items()}
        with workspace_lock(self.path.parent):
            self._insight(insight_id)
            if structured:
                registration = register_decision_rule(decision_rule, project=self.root, task_spec=self.spec, plugin=self.plugin)
                fields.update(decision_rule=registration["rule"], registration=registration,
                              rule_id=registration["rule_id"], entry_schema_version=2, verification_status="preregistered")
            else:
                fields.update(decision_rule=self._text(decision_rule, "decision_rule"), decision_rule_text=str(decision_rule),
                              verification_status="legacy_unstructured")
            plan = self._append("probe_plan", insight_id=insight_id, **fields,
                session_id=session_id, evidence_status="declared_plan_not_execution")
            if structured:
                from harness.decision_rules import policy_directory
                from harness.io_utils import atomic_write_json
                registration.update(probe_plan_id=plan["id"], insight_id=insight_id)
                atomic_write_json(policy_directory(self.root) / (registration["rule_id"] + ".json"), registration)
            return plan

    def _linked_records(self, insight, entries, runs):
        linked = [entry for entry in entries if entry.get("insight_id") == insight["id"]]
        def belongs(record):
            # A sealed config or captured execution link is authoritative. An unrun
            # config may have been rewritten for another insight using the same ID.
            if record.get("insight_id"):
                return record["insight_id"] == insight["id"]
            proposals = [entry for entry in entries if entry["kind"] == "proposal" and
                         entry["exp_id"] == record.get("exp_id") and
                         entry["timestamp"] <= record["timestamp"]]
            return bool(proposals and proposals[-1].get("insight_id") == insight["id"])
        results = [run for run in runs if belongs(run)]
        executions = [entry for entry in entries if entry["kind"] == "execution" and
                      belongs(entry)]
        return linked, results, executions

    def _trace_state(self, insight, entries, runs):
        linked, results, executions = self._linked_records(insight, entries, runs)
        proposals = {entry["exp_id"] for entry in linked if entry["kind"] == "proposal"}
        decisions = [entry for entry in linked if entry["kind"] == "decision"]
        plans = [entry for entry in linked if entry["kind"] == "probe_plan"]
        decision, plan = (decisions[-1] if decisions else None), (plans[-1] if plans else None)
        considered = {exp for item in decisions for exp in item.get("evidence", [])}
        unreviewed = [run for run in results if run["exp_id"] not in considered]
        latest = max((entry["timestamp"] for entry in [*results, *executions, *plans]), default=0)
        missing_evidence = [exp for exp in (decision.get("evidence", []) if decision else [])
                            if exp not in {run["exp_id"] for run in runs}]
        if missing_evidence:
            status, action = "needs_evidence_review", "Revalidate missing evidence before relying on the previous lesson."
        elif unreviewed:
            status, action = "needs_interpretation", "Inspect linked results and record a decision, lesson and next_action."
        elif decision and decision["timestamp"] >= latest:
            from harness.decision_rules import effective_outcome
            status, action = effective_outcome(decision), decision["next_action"]
        elif executions and (not plan or executions[-1]["timestamp"] >= plan["timestamp"]):
            status, action = "needs_diagnosis", "Inspect execution/probe output; it does not reject the scientific mechanism."
        elif plan:
            status, action = "ready_for_probe", "Implement the recorded atomic probe within its declared budget and stop rule."
        else:
            status, action = "needs_probe_plan", "Record a bounded probe that separates the mechanism from an alternative explanation."
        return {"insight_id": insight["id"], "title": insight["title"][:160], "status": status,
            "required_step": action[:300],
            "next_action": (decision["next_action"] if decision and status == decision["outcome"] else
                            plan["intervention"] if plan and status == "ready_for_probe" else insight["next_action"])[:300],
            "observation": insight["observation"][:200], "mechanism": insight["mechanism"][:200],
            "latest_plan_id": plan["id"] if plan else None,
            "lesson": decision.get("lesson", "")[:300] if decision else "",
            "revisit_when": decision.get("revisit_when", "")[:300] if decision else "",
            "unreviewed_exp_ids": [run["exp_id"] for run in unreviewed][-5:],
            "unreviewed_result_count": len(unreviewed),
            "missing_evidence": missing_evidence[:5],
            "exp_ids": sorted(proposals | {run["exp_id"] for run in results})}

    def trace(self, insight_id, offset=0, limit=20):
        entries, runs = self.entries(), self.sync()
        insight = self._insight(insight_id, entries)
        state = self._trace_state(insight, entries, runs)
        linked, results, executions = self._linked_records(insight, entries, runs)
        execution_ids = {entry["id"] for entry in executions}
        events = [entry for entry in entries if entry["id"] == insight_id or
                  entry.get("insight_id") == insight_id or
                  entry["id"] in execution_ids]
        events.sort(key=lambda entry: entry["timestamp"])
        offset, limit = max(0, int(offset)), max(1, min(int(limit), 30))
        verified = {run["exp_id"] for run in runs}
        def checked(entry):
            record = copy.deepcopy(entry)
            record.pop("scope", None)
            evidence = entry.get("evidence", [])
            record["evidence_currently_verified"] = bool(evidence) and all(exp in verified for exp in evidence)
            return record
        visible_state = {**state, "exp_ids": state["exp_ids"][-30:], "exp_count": len(state["exp_ids"])}
        children = [entry for entry in entries if entry.get("parent_insight_id") == insight_id]
        return {"insight": checked(insight), "state": visible_state,
            "events": [checked(entry) for entry in events[offset:offset + limit]],
            "event_count": len(events), "next_offset": offset + limit if offset + limit < len(events) else None,
            "results": [self._compact(run) for run in results[-8:]],
            "derived_insights": [{"insight_id": entry["id"], "title": entry["title"][:160]}
                                 for entry in children[-30:]], "derived_insight_count": len(children)}

    def search(self, query="", limit=8):
        # Search the entire ledger, not just the HUD's recent window.
        records = [*self.entries(), *self.sync()]
        words = set(re.findall(r"\w+", str(query).lower()))
        ranked = []
        for record in records:
            haystack = json.dumps(record, ensure_ascii=False).lower()
            hits = sum(word in haystack for word in words)
            if not words or hits:
                ranked.append((hits, record.get("timestamp", 0), record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [self._compact(r) for _, _, r in ranked[:max(1, min(int(limit), 30))]]

    @staticmethod
    def _mechanism_key(model):
        """Normalized model-family identifier for advisory mechanism matching.

        Deliberately coarse: the family (architecture / model type / name) is
        compared, hyperparameters are not — different C on the same logistic
        model is the same mechanism. Returns "" when the model block is empty
        (no family signal, no advisory).
        """
        if not isinstance(model, dict) or not model:
            return ""
        for key in ("architecture", "model_type", "family", "name", "type"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return ""

    def review(self, cfg, revisit_reason="", for_execution=False):
        fp = self.fingerprint(cfg)
        exp_id = cfg.get("experiment", {}).get("id")
        records = [*self.entries(), *self.sync()]
        exact, neighbors, mechanism_matches = [], [], []
        model = cfg.get("model", {})
        mechanism_key = self._mechanism_key(model)
        tokens = set(re.findall(r"\w+", " ".join(self._narrative(cfg).values()).lower()))
        for record in records:
            if for_execution and record.get("kind") == "proposal":
                continue
            # A saved proposal may proceed to its own run; a completed run may not
            # overwrite immutable evidence, independently enforced by the runner.
            if record.get("kind") == "proposal" and record.get("exp_id") == exp_id:
                continue
            if record.get("fingerprint") == fp:
                exact.append(self._compact(record))
                continue
            other = set(re.findall(r"\w+", " ".join(str(record.get(k, "")) for k in
                         ("hypothesis", "intervention", "problem_addressed", "observation", "mechanism", "lesson")).lower()))
            overlap = len(tokens & other) / max(1, len(tokens | other))
            if overlap >= 0.45 or (model and model == record.get("model")):
                neighbors.append({**self._compact(record), "token_similarity": round(overlap, 3)})
            # Advisory mechanism match: same model family as an earlier entry
            # (verified run, rejected decision, ...). Never blocks; lets the
            # agent compare mechanisms and scores before re-running.
            if (
                mechanism_key
                and record.get("exp_id") != exp_id
                and self._mechanism_key(record.get("model")) == mechanism_key
            ):
                compact = self._compact(record)
                compact["mechanism"] = mechanism_key
                if record.get("score") is not None:
                    compact.setdefault("score", record["score"])
                mechanism_matches.append(compact)
        insight_id = (cfg.get("research") or {}).get("insight_id")
        trace_state = None
        if insight_id:
            trace_state = self.trace(insight_id)["state"]
            if not trace_state["latest_plan_id"]:
                raise ValueError("Linked insight needs plan_research_probe before implementation or execution")
        return {"fingerprint": fp, "exact_config_matches": exact[:8], "research_trace": trace_state,
                "related_approaches": neighbors[:8], "mechanism_matches": mechanism_matches[:5],
                "revisit_reason": str(revisit_reason).strip(),
                "allowed": not exact or bool(str(revisit_reason).strip()),
                "interpretation": "Exact means same declared execution config; related is lexical advisory only; "
                    "mechanism_matches lists earlier work on the same model family (advisory). "
                    "Explain changed mechanism or provide revisit_reason for replication/data/code changes."}

    def propose(self, cfg, revisit_reason="", session_id=None):
        # Serialize check-and-register to prevent two agents claiming a new config.
        with workspace_lock(self.path.parent):
            review = self.review(cfg, revisit_reason)
            if not review["allowed"]:
                raise ValueError("Repeated declared execution config. Review research memory and supply "
                                 "revisit_reason if intentional: " + json.dumps(review["exact_config_matches"]))
            self._append("proposal", exp_id=cfg["experiment"]["id"],
                         insight_id=(cfg.get("research") or {}).get("insight_id"),
                         probe_plan_id=(review.get("research_trace") or {}).get("latest_plan_id"),
                         fingerprint=review["fingerprint"], model=cfg.get("model", {}),
                         **self._narrative(cfg), revisit_reason=str(revisit_reason), session_id=session_id,
                         evidence_status="unrun_proposal")
        return review

    def record_decision(self, *, outcome=None, conclusion, evidence=None, next_action="", session_id=None,
                        insight_id=None, lesson="", revisit_when="", rule_id=None, candidate_evidence=None, claimed_outcome=None):
        from harness.decision_rules import (DecisionRuleNotSatisfied, evaluate_decision_rule, load_registration)
        evaluation = None
        requested = normalize_decision_outcome(outcome) if outcome else None
        if rule_id:
            registration = load_registration(self.root, rule_id)
            plans = [entry for entry in self.entries() if entry.get("rule_id") == rule_id and entry["kind"] == "probe_plan"]
            if not plans or (insight_id and plans[-1]["insight_id"] != insight_id):
                raise ValueError("Rule must belong to the selected insight's probe plan")
            if not candidate_evidence:
                raise ValueError("Structured decision requires typed candidate_evidence")
            evaluation = evaluate_decision_rule(registration["rule"], candidate_evidence, project=self.root,
                task_spec=self.spec, plugin=self.plugin, registration=registration)
            if requested == "supported" and evaluation["outcome"] != "supported":
                raise DecisionRuleNotSatisfied(evaluation["reason_codes"])
            outcome = evaluation["outcome"]
        elif requested == "supported":
            raise DecisionRuleNotSatisfied(["NOT_PREREGISTERED"])
        else:
            outcome = requested or "inconclusive"
        outcome = normalize_decision_outcome(outcome)
        if outcome not in {"supported", "rejected", "inconclusive", "deferred"}:
            hint = (
                " - that value looks like prose; put the narrative in 'conclusion' "
                "and use one of the four outcomes here"
                if len(str(outcome)) > 40 else ""
            )
            raise ValueError(f"outcome must be supported, rejected, inconclusive, or deferred{hint}")
        if not str(conclusion).strip():
            raise ValueError("A decision requires a conclusion")
        if insight_id:
            self._insight(insight_id)
            next_action = self._text(next_action, "next_action")
            lesson = self._text(lesson, "lesson")
            if outcome in {"rejected", "deferred"}:
                # Provide a sensible default so agents don't have to invent a
                # revisit condition; they can still override with their own.
                if not str(revisit_when).strip():
                    revisit_when = "When a fundamentally different mechanism with plausible improvement is available."
                revisit_when = self._text(revisit_when, "revisit_when")
        if evaluation:
            evidence = list(evidence or []) if not isinstance(evidence, str) else [evidence]
            evidence += [ref["id"] for ref in (evaluation["candidate"], evaluation["control"]) if ref["kind"] == "full_run" and ref["id"] not in evidence]
        evidence = self._evidence(evidence)
        return self._append("decision", outcome=outcome, conclusion=str(conclusion),
            evidence=evidence, next_action=str(next_action), session_id=session_id,
            insight_id=insight_id, lesson=str(lesson), revisit_when=str(revisit_when),
            evidence_status="machine_evaluated" if evaluation else "declared_interpretation",
            entry_schema_version=2, claimed_outcome=claimed_outcome or requested,
            effective_outcome=outcome, verification_status="machine_evaluated" if evaluation else "legacy_unstructured",
            rule_id=rule_id, evaluation_id=evaluation["evaluation_id"] if evaluation else None,
            reason_codes=evaluation["reason_codes"] if evaluation else [], evaluation=dict(evaluation) if evaluation else None)

    def checkpoint(self, *, summary, next_action, open_questions=None, session_id=None, insight_ids=None):
        if not str(summary).strip() or not str(next_action).strip():
            raise ValueError("Checkpoint requires summary and next_action")
        if insight_ids is not None and (not isinstance(insight_ids, list) or len(insight_ids) > 30 or
                                       any(not isinstance(value, str) for value in insight_ids)):
            raise ValueError("insight_ids must be a list of at most 30 current-scope insight IDs")
        for insight_id in insight_ids or []:
            self._insight(insight_id)
        return self._append("checkpoint", conclusion=str(summary), next_action=str(next_action),
            insight_ids=list(insight_ids or []),
            open_questions=list(open_questions or []), session_id=session_id,
            evidence_status="declared_checkpoint")

    def record_execution(self, action, status, summary, exp_id=None, session_id=None, insight_id=None):
        return self._append("execution", action=action, outcome=status, conclusion=summary,
            exp_id=exp_id, session_id=session_id, insight_id=insight_id,
            evidence_status="execution_observation_only")

    def context(self, min_improvement=0.0, patience=3):
        entries, runs = self.entries(), self.sync()
        decisions = [e for e in entries if e["kind"] == "decision"]
        checkpoints = [e for e in entries if e["kind"] == "checkpoint"]
        # Compare within each data/split/scorer identity, across sessions. Failed
        # execution and single-fold probes never become negative scientific evidence.
        groups = {}
        for run in runs:
            key = run["comparison_group"]
            best, stalled = groups.get(key, (None, 0))
            gain = None if best is None else run["utility"] - best
            stalled = 0 if gain is None or (gain > 0 and gain >= min_improvement) else stalled + 1
            groups[key] = (max(best, run["utility"]) if best is not None else run["utility"], stalled)
        stalled = groups[runs[-1]["comparison_group"]][1] if runs else 0
        checkpoint = copy.deepcopy(checkpoints[-1]) if checkpoints else None
        if checkpoint:
            checkpoint.pop("scope", None)
            checkpoint.pop("session_id", None)
            checkpoint["conclusion"] = checkpoint["conclusion"][:1000]
            checkpoint["next_action"] = checkpoint["next_action"][:600]
            checkpoint["open_questions"] = [str(q)[:300] for q in checkpoint["open_questions"][:5]]
        # Resolve evidence links live: a prior interpretation can outlive its evidence.
        verified = {r["exp_id"] for r in runs}
        recent_decisions = []
        from harness.decision_rules import effective_outcome
        for decision in decisions[-4:]:
            recent_decisions.append({**self._compact(decision),
                "effective_outcome": effective_outcome(decision),
                "verification_status": decision.get("verification_status", "legacy_unverified"),
                "evidence": decision.get("evidence", []),
                "evidence_currently_verified": bool(decision.get("evidence")) and
                    all(exp in verified for exp in decision["evidence"])})
        interpreted = {exp for decision in decisions for exp in decision.get("evidence", [])}
        unreviewed = [run for run in runs if run["exp_id"] not in interpreted]
        insights = [entry for entry in entries if entry["kind"] == "insight"]
        frontier = [self._trace_state(insight, entries, runs) for insight in insights]
        active = [state for state in frontier if state["status"] not in {"supported", "rejected", "deferred"}]
        order = {"needs_evidence_review": 0, "needs_interpretation": 0, "needs_diagnosis": 1, "inconclusive": 2,
                 "needs_probe_plan": 3, "ready_for_probe": 4}
        active.sort(key=lambda state: order.get(state["status"], 5))
        compact_frontier = [{key: value for key, value in state.items() if key != "exp_ids"}
                            for state in active[:3]]
        progress = self._publish_progress(runs)
        from harness.decision_rules import _project
        from harness.stagnation import StagnationController
        project = _project(self.root)
        states = StagnationController(project).reconcile_results() if project.artifacts_dir.resolve() == self.artifacts.resolve() else {}
        paused = [state for state in states.values() if state["state"] == "paused_for_stagnation"]
        from harness.macro_research import compute_macro_hypothesis_map, retrospective_status
        macro_map = compute_macro_hypothesis_map(self, runs)
        # Context is bounded independently of campaign length. Full evidence IDs
        # and comparison groups are available through get_macro_hypothesis_map.
        macro_map = {"map_digest": macro_map["map_digest"], "verified_run_count": macro_map["verified_run_count"],
            "families": {fid: {key: row[key] for key in ("state", "run_count", "supported", "rejected", "active")}
                for fid, row in macro_map["families"].items()}}
        retrospective = retrospective_status(self, runs, states)
        macro_reviews = [entry for entry in entries if entry["kind"] == "macro_review"]
        latest_macro_review = None
        if macro_reviews:
            last = macro_reviews[-1]
            latest_macro_review = {"review_id": last["review_id"], "run_count_at_review": last["run_count_at_review"],
                "highest_priority_untouched": {key: str(last["highest_priority_untouched"][key])[:180]
                    for key in ("family_id", "proposed_direction")}}
        return {"ledger_path": str(self.path), **progress, "entry_count": len(entries), "verified_run_count": len(runs),
            "macro_hypothesis_map": macro_map, "retrospective_status": retrospective,
            "latest_macro_review": latest_macro_review,
            "insight_count": len(insights), "open_insight_count": len(active), "research_frontier": compact_frontier,
            "unreviewed_result_count": len(unreviewed),
            "unreviewed_results": [self._compact(run) for run in unreviewed[-4:]],
            "latest_checkpoint": checkpoint, "recent_decisions": recent_decisions,
            "recent_results": [self._compact(r) for r in runs[-4:]],
            "consecutive_non_improvements": stalled, "reassess_recommended": stalled >= max(1, patience),
            "submission_pause": bool(paused), "stagnation_states": paused[:3],
            "resume_protocol": ["Read checkpoint and search_research_memory before brainstorming.",
                "Use review_hypothesis before implementation; explain related approaches and intentional repeats.",
                "Record each insight with record_research_insight; plan_research_probe before linked implementation. "
                "Link config.research.insight_id; use trace_research_insight for its history and derived directions.",
                "Interpret unreviewed results before another brainstorm. Record insight-linked decisions with lesson, "
                "next_action and revisit_when for rejected/deferred directions; checkpoint active insight_ids.",
                "Quota is a ceiling. Honor the task-scoped stagnation pause after three failed comparable promotion results. "
                "Submissions and ordinary full exploration stay paused until a registered bounded recovery probe is verified supported. "
                "Proactively mine community discussions ('ai-harness discussion list <comp>') for fresh signals, representations, or external insights. "
                "Inspect verified diagnostics, retrieve old lessons, enumerate alternative orthogonal mechanisms (different model families, novel representations, risk-hedging ensembles), "
                "and choose a bounded discriminating probe with an explicit promotion/stop criterion. "
                "A plateau is not evidence of a capability ceiling and does not authorize freeze."]}

    def render(self, **kwargs):
        context = self.context(**kwargs)
        from harness.macro_research import macro_hud_lines
        return ("\n".join(macro_hud_lines(context["macro_hypothesis_map"], context["retrospective_status"],
                context["latest_macro_review"])) + "\n## Research continuity\n" +
                json.dumps(context, ensure_ascii=False, indent=2) + "\n")
