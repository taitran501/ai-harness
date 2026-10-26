"""Evidence-derived family coverage and task-scoped retrospective milestones.

The research ledger is authoritative. Maps are projections, never editable state;
review milestones retain run IDs so lost evidence cannot replenish the cadence.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Tuple

from harness.artifacts import digest
from harness.policy_schema import validate_document
from harness.state_store import workspace_lock

FAMILIES = {
    "data_and_splits": "Data and Splits",
    "representation": "Representation",
    "architecture": "Architecture",
    "target": "Target",
    "optimization": "Optimization",
    "ensemble": "Ensemble",
    "post_processing": "Post-Processing",
    "domain_signals": "Domain Signals",
    "unclassified": "Unclassified",
}
CADENCE = 5


class GateViolationError(ValueError):
    code = "RETROSPECTIVE_REQUIRED"
    blockers = [code]
    receipt = None


def family_id(value):
    return value if isinstance(value, str) and value in FAMILIES else "unclassified"


def load_memory(project_root):
    """Resolve the attached task and plugin, avoiding legacy artifact guessing."""
    from harness.decision_rules import _project
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.research_memory import ResearchMemory
    from harness.task_spec import load_active_task_spec
    project = _project(project_root)
    spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(project.root, project.task_config_path, spec, project.plugin_id or "default")
    plugin = load_task_plugin(project.plugin_id, context)
    return ResearchMemory(project.root, project.artifacts_dir, spec, plugin)


def _entries(memory):
    return memory.get("entries", []) if isinstance(memory, dict) else memory.entries()


def _verified_runs(history):
    return {r["exp_id"]: r for r in history if r.get("evidence_status") == "verified_full"
            and r.get("run_kind", "full") == "full" and r.get("status", "COMPLETED") == "COMPLETED"}


def _machine_decision(memory, decision, runs):
    """Require the machine receipt and its live full-run dependencies."""
    evaluation = decision.get("evaluation")
    if decision.get("verification_status") != "machine_evaluated" or not evaluation:
        return False
    if (evaluation.get("evaluation_id") != decision.get("evaluation_id") or
            evaluation.get("outcome") != decision.get("outcome") or
            evaluation.get("rule_id") != decision.get("rule_id")):
        return False
    refs = [evaluation.get("candidate", {}), evaluation.get("control", {})]
    if any(ref.get("kind") != "full_run" or ref.get("id") not in runs for ref in refs):
        return False
    if not isinstance(memory, dict):
        from harness.decision_rules import policy_directory, read_decision_evaluation
        try:
            receipt = read_decision_evaluation(policy_directory(memory.root) / (decision["evaluation_id"] + ".json"))
            if receipt != evaluation:
                return False
        except (ValueError, OSError, KeyError):
            return False
        from harness.artifacts import sha256
        from harness.decision_rules import resolve_evidence
        try:
            for name in ("candidate", "control"):
                live = resolve_evidence(evaluation[name], project=memory.root, task_spec=memory.spec, plugin=memory.plugin)
                if live["evidence_sha256"] != evaluation["evidence_sha256"][name]:
                    return False
                if name == "candidate" and sha256(live["directory"] / "evaluation_truth.csv") != evaluation["evidence_sha256"]["truth"]:
                    return False
        except (ValueError, OSError, KeyError):
            return False
    return True


def compute_macro_hypothesis_map(research_memory, run_history) -> Dict[str, Any]:
    entries, runs = _entries(research_memory), _verified_runs(run_history)
    insights = {e["id"]: e for e in entries if e.get("kind") == "insight"}
    families = {fid: dict(family_id=fid, label=label, state="untried", insight_count=0,
        run_count=0, supported=0, rejected=0, active=0, insight_ids=[], run_ids=[], comparison_groups=[])
        for fid, label in FAMILIES.items()}
    linked_runs = {}
    for rid, run in runs.items():
        iid = run.get("insight_id")
        if not iid:
            proposals = [e for e in entries if e.get("kind") == "proposal" and e.get("exp_id") == rid
                         and e.get("timestamp", 0) <= run.get("timestamp", 0)]
            iid = proposals[-1].get("insight_id") if proposals else None
        linked_runs[rid] = iid
        fid = family_id(insights.get(iid, {}).get("family_id"))
        families[fid]["run_ids"].append(rid)
        group = run.get("comparison_group")
        if group and group not in families[fid]["comparison_groups"]:
            families[fid]["comparison_groups"].append(group)
    for iid, insight in insights.items():
        row = families[family_id(insight.get("family_id"))]
        row["insight_ids"].append(iid)
        linked = [e for e in entries if e.get("insight_id") == iid]
        decisions = [e for e in linked if e.get("kind") == "decision"]
        plans = [e for e in linked if e.get("kind") == "probe_plan"]
        results = [r for rid, r in runs.items() if linked_runs[rid] == iid]
        decision = decisions[-1] if decisions else None
        plan = plans[-1] if plans else None
        machine = {d["id"]: _machine_decision(research_memory, d, runs) for d in decisions}
        supported = any(d.get("outcome") == "supported" and machine[d["id"]] for d in decisions)
        row["supported"] += int(supported)
        row["rejected"] += int(any(d.get("outcome") == "rejected" and machine[d["id"]] for d in decisions))
        latest = max((e.get("timestamp", 0) for e in [*plans, *results,
            *(e for e in linked if e.get("kind") in {"execution", "proposal"})]), default=0)
        interpreted = set(decision.get("evidence", [])) if decision else set()
        terminal = bool(decision and decision.get("outcome") in {"supported", "rejected"}
            and machine[decision["id"]] and plan
            and str(plan.get("stop_rule", "")).strip() and str(plan.get("budget", "")).strip()
            and str(decision.get("revisit_when", "")).strip()
            and decision.get("timestamp", 0) >= latest
            and all(r["exp_id"] in interpreted for r in results))
        # Supported mechanisms may coexist with open directions in the family.
        closed_support = bool(supported and decision and decision.get("outcome") == "supported"
            and machine[decision["id"]]
            and decision.get("timestamp", 0) >= latest and all(r["exp_id"] in interpreted for r in results))
        row["active"] += int(not terminal and not closed_support)
    for row in families.values():
        row["insight_count"], row["run_count"] = len(row["insight_ids"]), len(row["run_ids"])
        row["run_ids"].sort()
        row["comparison_groups"].sort()
        row["state"] = ("active" if row["active"] else "supported" if row["supported"] else
            "exhausted" if row["insight_count"] else "active" if row["run_count"] else "untried")
    result = dict(schema_version=1, families=families, verified_run_count=len(runs),
        evidence_digest=digest(dict(events=[e for e in entries if e.get("kind") in
            {"insight", "probe_plan", "decision", "proposal", "execution"}],
            runs=[{k: v for k, v in runs[rid].items() if k not in {"path", "timestamp"}} for rid in sorted(runs)])))
    result["map_digest"] = digest(result)
    validate_document("macro_hypothesis_map", result)
    return result


def _stagnation_events(states):
    if states.get("state"):
        states = {"state": states}
    events = set()
    for s in states.values():
        identity = s.get("comparison_identity_sha256")
        gen = s.get("generation", 0)
        if s.get("state") == "paused_for_stagnation":
            events.add(digest({k: s.get(k) for k in ("comparison_identity_sha256", "generation", "paused_at")}))
        for g in range(1, gen + 1):
            events.add(digest(dict(comparison_identity_sha256=identity, pause_generation=g)))
    return sorted(events)


def retrospective_status(memory, run_history, stagnation_state):
    runs = _verified_runs(run_history)
    reviews = [e for e in _entries(memory) if e.get("kind") == "macro_review"]
    last = reviews[-1] if reviews else {}
    new_ids = set(runs) - set(last.get("verified_run_ids", []))
    events = _stagnation_events(stagnation_state)
    unreviewed_pause = bool(set(events) - set(last.get("stagnation_events", [])))
    due = len(new_ids) >= CADENCE or unreviewed_pause
    reason = ("stagnation pause has not been evaluated by a macro review" if unreviewed_pause else
              f"{len(new_ids)}/{CADENCE} verified full-run results since last review")
    return dict(due=due, reason=reason, runs_since_review=len(new_ids), cadence=CADENCE,
                verified_run_count=len(runs), last_review_id=last.get("review_id"))


def is_retrospective_due(project_root, run_history, stagnation_state) -> Tuple[bool, str]:
    memory = project_root if hasattr(project_root, "entries") else load_memory(project_root)
    status = retrospective_status(memory, run_history, stagnation_state)
    return status["due"], status["reason"]


def record_macro_review(research_memory, payload=None, **fields):
    memory = research_memory if hasattr(research_memory, "entries") else load_memory(research_memory)
    from harness.stagnation import StagnationController
    with workspace_lock(memory.path.parent):
        runs = memory.sync()
        macro_map = compute_macro_hypothesis_map(memory, runs)
        states = StagnationController(memory.root).reconcile_results()
        value = dict(payload or {}, **fields)
        value.setdefault("review_id", "rev_" + uuid.uuid4().hex)
        value.setdefault("task_id", memory._task_key)
        value.setdefault("run_count_at_review", len(_verified_runs(runs)))
        validate_document("macro_review", value)
        if value["task_id"] != memory._task_key or value["run_count_at_review"] != macro_map["verified_run_count"]:
            raise ValueError("Review must reference the current task and verified run count")
        if value["map_digest"] != macro_map["map_digest"]:
            raise ValueError("Stale map_digest: retrieve get_macro_hypothesis_map and review current evidence")
        entries = memory.entries()
        if any(e.get("review_id") == value["review_id"] for e in entries):
            raise ValueError("review_id already recorded")
        decisions = {e["id"]: e for e in entries if e.get("kind") == "decision"}
        insights = {e["id"]: e for e in entries if e.get("kind") == "insight"}
        verified = _verified_runs(runs)
        seen_supported = set()
        for key, outcome in (("supported_mechanisms", "supported"), ("rejected_mechanisms", "rejected")):
            for claim in value[key]:
                did = claim.get("decision_id")
                if key == "supported_mechanisms":
                    if did in seen_supported:
                        raise ValueError(f"Duplicate supported claim for decision_id {did}")
                    seen_supported.add(did)
                    summary = str(claim.get("mechanism_summary", "")).strip()
                    if len(summary) < 5:
                        raise ValueError(f"Supported claim {did} requires a substantive mechanism_summary")
                decision = decisions.get(did, {})
                insight = insights.get(claim["insight_id"], {})
                candidate = (decision.get("evaluation") or {}).get("candidate", {})
                if (not insight or decision.get("insight_id") != insight["id"] or
                    family_id(insight.get("family_id")) != claim["family_id"] or
                    decision.get("outcome") != outcome or candidate.get("id") != claim["run_id"] or
                    not _machine_decision(memory, decision, verified)):
                    raise ValueError("Mechanism claim requires linked, live machine-evaluated decision evidence")
        lead = value["highest_priority_untouched"]
        if any(row["state"] == "untried" for row in macro_map["families"].values()) and macro_map["families"][lead["family_id"]]["state"] != "untried":
            raise ValueError("highest_priority_untouched must identify an untried family while coverage gaps remain")
        return memory._append("macro_review", **value, verified_run_ids=sorted(verified),
            stagnation_events=_stagnation_events(states), evidence_status="validated_macro_review")


def enforce_retrospective(project_root, *, cfg=None, recovery_probe=False):
    memory = load_memory(project_root)
    runs = memory.sync()
    from harness.stagnation import StagnationController
    controller = StagnationController(project_root)
    states = controller.reconcile_results()
    due, reason = is_retrospective_due(memory, runs, states)
    if not due:
        return
    research = (cfg or {}).get("research", {})
    rid = research.get("recovery_probe_id")
    candidate = (cfg or {}).get("experiment", {}).get("id")
    if recovery_probe and rid and candidate:
        from harness.decision_rules import policy_directory
        path = policy_directory(controller.project) / "recovery_probes.json"
        probes = json.loads(path.read_text(encoding="utf-8"))["probes"] if path.exists() else {}
        probe = probes.get(rid)
        if (probe and research.get("decision_rule_id") == probe["rule_id"]
            and any(s.get("state") == "paused_for_stagnation" and s["comparison_identity_sha256"] == probe["comparison_identity_sha256"]
                    and s["generation"] == probe["generation"] for s in states.values())
            and controller.check_action("run_full_validation", recovery_probe_id=rid, candidate_id=candidate) is None):
            return
    raise GateViolationError("RETROSPECTIVE_REQUIRED: 5 verified runs completed since last review. "
        "You must call record_macro_review before launching costly runs. Trigger: " + reason)


def macro_hud_lines(macro_map, status, latest_review=None):
    lines = ["## Macro Hypothesis Map (Coverage & Gaps)"]
    for fid, row in macro_map["families"].items():
        lines.append(f"- {FAMILIES[fid]}: {row['state'].title()} ({row['supported']} supported, "
                     f"{row['rejected']} rejected, {row['active']} active; {row['run_count']} runs)")
    untouched = [FAMILIES[fid] for fid, row in macro_map["families"].items() if row["state"] == "untried"]
    lead = (latest_review or {}).get("highest_priority_untouched")
    if lead and (not untouched or macro_map["families"][lead["family_id"]]["state"] == "untried"):
        lines.append("- High-Priority Untouched Lead: " + lead["proposed_direction"][:180])
    elif untouched:
        lines.append("- Untouched families (priority not yet reviewed): " + ", ".join(untouched))
    lines.append(f"- Retrospective Status: {'DUE' if status['due'] else 'OK'} "
                 f"({status['runs_since_review']}/{status['cadence']} runs completed; {status['reason']})")
    if status["due"]:
        lines.append("RETROSPECTIVE REVIEW DUE: Step back and evaluate the macro map with "
                     "record_macro_review before launching the next full run.")
    return lines
