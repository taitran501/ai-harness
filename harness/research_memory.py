"""Durable research continuity outside immutable experiment evidence.

Configuration equivalence is deterministic; mechanism similarity is an advisory,
not proof that two scientific hypotheses are equivalent. Decisions remain declared
interpretations, even when their referenced measurements are verified.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
import uuid
from pathlib import Path

import yaml

from harness.artifacts import contract_identity, digest, validate_run
from harness.state_store import transaction, workspace_lock


class ResearchMemory:
    def __init__(self, project_root, artifacts_dir, task_spec, plugin=None):
        self.root = Path(project_root)
        self.artifacts = Path(artifacts_dir)
        self.path = self.artifacts / "research_memory.json"
        self.spec = task_spec
        self.plugin = plugin
        self.identity = contract_identity(task_spec, plugin)
        self.scope = digest(self.identity)

    def _read(self):
        if not self.path.exists():
            return {"entries": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def entries(self):
        return [e for e in self._read().get("entries", []) if e["scope"] == self.scope]

    def _append(self, kind, **fields):
        entry = {"id": uuid.uuid4().hex, "scope": self.scope, "kind": kind,
                 "timestamp": time.time(), **fields}
        with transaction(self.path, {"schema_version": 1, "entries": []}) as state:
            state.setdefault("entries", []).append(entry)
        return entry

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

    def sync(self):
        """Revalidate artifacts on every read; a cached claim never verifies evidence."""
        runs = []
        for folder in sorted(self.artifacts.iterdir()) if self.artifacts.exists() else []:
            manifest_path = folder / "artifact_manifest.json"
            if not folder.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = validate_run(folder, task_spec=self.spec, plugin=self.plugin)
                cfg = yaml.safe_load((folder / "run_config.yaml").read_text(encoding="utf-8"))
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                score = float(metrics["primary_score"])
                value = score if metrics["direction"] == "maximize" else -score
                if not math.isfinite(value):
                    continue
                diag = {}
                if "diagnostics.json" in manifest["files"]:
                    diag = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
                runs.append({"exp_id": manifest["exp_id"], "path": str(folder),
                    "fingerprint": self.fingerprint(cfg), **self._narrative(cfg),
                    "model": cfg.get("model", {}),
                    "score": score, "utility": value,
                    "comparison_group": digest({key: manifest.get(key) for key in
                        ("task_sha256", "scorer_sha256", "custom_scorer_sha256",
                         "data_sha256", "split_sha256")}),
                    "bottlenecks": diag.get("critical_bottlenecks", [])[:5],
                    "timestamp": manifest_path.stat().st_mtime, "evidence_status": "verified_full"})
            except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError):
                continue
        return sorted(runs, key=lambda r: (r["timestamp"], r["exp_id"]))

    @staticmethod
    def _compact(record):
        return {key: record[key][:600] if isinstance(record[key], str) else record[key] for key in
                ("id", "kind", "exp_id", "hypothesis", "intervention", "score",
                 "conclusion", "next_action", "outcome", "evidence_status", "path") if key in record}

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

    def review(self, cfg, revisit_reason="", for_execution=False):
        fp = self.fingerprint(cfg)
        exp_id = cfg.get("experiment", {}).get("id")
        records = [*self.entries(), *self.sync()]
        exact, neighbors = [], []
        model = cfg.get("model", {})
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
                         ("hypothesis", "intervention", "problem_addressed")).lower()))
            overlap = len(tokens & other) / max(1, len(tokens | other))
            if overlap >= 0.45 or (model and model == record.get("model")):
                neighbors.append({**self._compact(record), "token_similarity": round(overlap, 3)})
        return {"fingerprint": fp, "exact_config_matches": exact[:8],
                "related_approaches": neighbors[:8], "revisit_reason": str(revisit_reason).strip(),
                "allowed": not exact or bool(str(revisit_reason).strip()),
                "interpretation": "Exact means same declared execution config; related is lexical advisory only. "
                    "Explain changed mechanism or provide revisit_reason for replication/data/code changes."}

    def propose(self, cfg, revisit_reason="", session_id=None):
        # Serialize check-and-register to prevent two agents claiming a new config.
        with workspace_lock(self.path.parent):
            review = self.review(cfg, revisit_reason)
            if not review["allowed"]:
                raise ValueError("Repeated declared execution config. Review research memory and supply "
                                 "revisit_reason if intentional: " + json.dumps(review["exact_config_matches"]))
            self._append("proposal", exp_id=cfg["experiment"]["id"],
                         fingerprint=review["fingerprint"], model=cfg.get("model", {}),
                         **self._narrative(cfg), revisit_reason=str(revisit_reason), session_id=session_id,
                         evidence_status="unrun_proposal")
        return review

    def record_decision(self, *, outcome, conclusion, evidence=None, next_action="", session_id=None):
        if outcome not in {"supported", "rejected", "inconclusive", "deferred"}:
            raise ValueError("outcome must be supported, rejected, inconclusive, or deferred")
        if not str(conclusion).strip():
            raise ValueError("A decision requires a conclusion")
        evidence = list(evidence or [])
        available = {r["exp_id"] for r in self.sync()}
        if any(exp not in available for exp in evidence):
            raise ValueError("Evidence references must identify verified full runs for the current task")
        return self._append("decision", outcome=outcome, conclusion=str(conclusion),
            evidence=evidence, next_action=str(next_action), session_id=session_id,
            evidence_status="declared_interpretation")

    def checkpoint(self, *, summary, next_action, open_questions=None, session_id=None):
        if not str(summary).strip() or not str(next_action).strip():
            raise ValueError("Checkpoint requires summary and next_action")
        return self._append("checkpoint", conclusion=str(summary), next_action=str(next_action),
            open_questions=list(open_questions or []), session_id=session_id,
            evidence_status="declared_checkpoint")

    def record_execution(self, action, status, summary, exp_id=None, session_id=None):
        return self._append("execution", action=action, outcome=status, conclusion=summary,
            exp_id=exp_id, session_id=session_id, evidence_status="execution_observation_only")

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
        for decision in decisions[-4:]:
            recent_decisions.append({**self._compact(decision),
                "evidence": decision.get("evidence", []),
                "evidence_currently_verified": bool(decision.get("evidence")) and
                    all(exp in verified for exp in decision["evidence"])})
        return {"ledger_path": str(self.path), "entry_count": len(entries), "verified_run_count": len(runs),
            "latest_checkpoint": checkpoint, "recent_decisions": recent_decisions,
            "recent_results": [self._compact(r) for r in runs[-4:]],
            "consecutive_non_improvements": stalled, "reassess_recommended": stalled >= max(1, patience),
            "resume_protocol": ["Read checkpoint and search_research_memory before brainstorming.",
                "Use review_hypothesis before implementation; explain related approaches and intentional repeats.",
                "After a result, record_research_decision and record_research_checkpoint.",
                "On plateau: inspect verified diagnostics, retrieve old lessons, enumerate alternative mechanisms, "
                "choose a bounded discriminating probe and record its promotion/stop criterion. "
                "A plateau is not evidence of a capability ceiling and does not authorize freeze."]}

    def render(self, **kwargs):
        context = self.context(**kwargs)
        return "## Research continuity\n" + json.dumps(context, ensure_ascii=False, indent=2) + "\n"
