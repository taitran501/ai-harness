"""Task-scoped reversible submission pause derived from completed evaluations."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from harness.artifacts import digest
from harness.decision_rules import _project, policy_directory, resolve_research_policy, utc_now
from harness.io_utils import atomic_write_json
from harness.policy_schema import validate_document
from harness.state_store import transaction, workspace_lock


@dataclass(frozen=True)
class StagnationStateV1:
    """Schema-validated state projection; the evaluation ledger is authoritative."""
    value: dict

    def __post_init__(self):
        validate_document("stagnation_state", self.value)


class StagnationController:
    def __init__(self, project):
        self.project = _project(project)
        self.directory = policy_directory(self.project)

    def reconcile_results(self, results=None):
        """Replay canonical completion sequence; repeated notes never add failures."""
        from harness.task_spec import load_active_task_spec
        spec = load_active_task_spec(self.project.task_config_path)
        policy = resolve_research_policy(self.project, spec)
        index_path = self.directory / "evaluations.json"
        with workspace_lock(self.directory):
            if index_path.exists():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                except Exception as e:
                    raise ValueError(f"CORRUPT_STAGNATION_STATE: {index_path} cannot be parsed: {e}")
            else:
                index = {"evaluations": {}}
                for p in sorted(self.directory.glob("eval-*.json")):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if "evaluation_id" in data:
                            index["evaluations"][data["evaluation_id"]] = data["evaluation_id"]
                    except Exception:
                        pass
            evaluations = []
            for evaluation_id in index["evaluations"].values():
                from harness.decision_rules import read_decision_evaluation
                value = read_decision_evaluation(self.directory / (evaluation_id + ".json"))
                evaluations.append(value)
            states, seen = {}, set()
            recovery_path = self.directory / "recovery_probes.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8")) if recovery_path.exists() else {"probes": {}}
            for result in evaluations:
                identity = result["comparison_identity_sha256"]
                if not identity or result["checks"]["identity"] != "pass" or result["gain"] is None:
                    continue
                key = digest({k: result[k] for k in ("evidence_sha256", "rule_sha256", "comparison_identity_sha256")})
                if key in seen:
                    continue
                seen.add(key)
                from harness.decision_rules import load_registration
                registration = load_registration(self.project, result["rule_id"]) if result["rule_id"].startswith("rule-") else None
                bound_policy = registration.get("policy") if registration else None
                if bound_policy is not None and digest(bound_policy) != result["policy_sha256"]:
                    raise ValueError("Registered evaluation policy changed")
                patience = bound_policy["stagnation_patience"] if bound_policy else policy.stagnation_patience
                state = states.setdefault(identity, dict(schema_version=1, comparison_identity_sha256=identity,
                    policy_sha256=result["policy_sha256"], generation=0, state="active", patience=patience,
                    consecutive_failures=0, incumbent_evidence_ref=result["control"], incumbent_evaluation_id=None,
                    last_processed_result_key=None, trigger_evaluation_ids=[], paused_at=None, recovery_probe_id=None,
                    resume_evaluation_id=None, override_id=None, updated_at=result["evaluated_at"]))
                # A result against an obsolete baseline cannot reset or alter the current streak.
                if result["control"] != state["incumbent_evidence_ref"]:
                    continue
                state.update(last_processed_result_key=key, updated_at=result["evaluated_at"])
                probe_id = next((pid for pid, p in recovery["probes"].items()
                    if p["comparison_identity_sha256"] == identity and p["rule_id"] == result["rule_id"]
                    and result["candidate"]["id"] in p.get("candidates", [])), None)
                if result["outcome"] == "supported":
                    if state["state"] == "paused_for_stagnation" and probe_id is None:
                        continue
                    state.update(consecutive_failures=0, state="active", incumbent_evidence_ref=result["candidate"],
                                 incumbent_evaluation_id=result["evaluation_id"], trigger_evaluation_ids=[], paused_at=None,
                                 patience=patience, policy_sha256=result["policy_sha256"])
                    if probe_id:
                        state.update(recovery_probe_id=probe_id, resume_evaluation_id=result["evaluation_id"], generation=state["generation"] + 1)
                else:
                    state["consecutive_failures"] += 1
                    state["trigger_evaluation_ids"].append(result["evaluation_id"])
                    if state["consecutive_failures"] >= state["patience"] and state["state"] != "paused_for_stagnation":
                        state.update(state="paused_for_stagnation", paused_at=result["evaluated_at"], generation=state["generation"] + 1)
            for identity, state in states.items():
                validate_document("stagnation_state", state)
                atomic_write_json(self.directory / "stagnation" / (identity + ".json"), state)
            if states:
                latest = max(states.values(), key=lambda s: s["updated_at"])
                atomic_write_json(self.project.artifacts_dir / "stagnation_state.json", latest)
            return states

    def check_action(self, action, *, recovery_probe_id=None, candidate_id=None):
        states = self.reconcile_results()
        paused = [s for s in states.values() if s["state"] == "paused_for_stagnation"]
        if not paused or action not in {"dispatch", "prepare_submission", "run_full_validation", "synthesize_candidate",
                                      "build_ensemble", "optimize_blend", "promote_candidate", "register_external_prediction"}:
            return None
        if action == "run_full_validation" and recovery_probe_id and candidate_id:
            with transaction(self.directory / "recovery_probes.json", {"probes": {}}) as value:
                probe = value["probes"].get(recovery_probe_id)
                if probe and any(probe["comparison_identity_sha256"] == s["comparison_identity_sha256"] for s in paused):
                    candidates = probe.setdefault("candidates", [])
                    if candidate_id in candidates:
                        return None
                    if len(candidates) < probe["max_runs"]:
                        candidates.append(candidate_id)
                        return None
        return {"code": "STAGNATION_PAUSE", "message": "Submissions and ordinary full exploration are paused. Register a bounded recovery probe; diagnostics and smoke tests remain available.", "states": paused}

    def register_recovery_probe(self, *, comparison_identity_sha256, rule_id, diagnosis, mechanism, stop_rule, max_runs=1):
        from harness.decision_rules import load_registration
        states = self.reconcile_results()
        state = states.get(comparison_identity_sha256)
        if not state or state["state"] != "paused_for_stagnation":
            raise ValueError("Recovery requires a paused comparison group")
        rule = load_registration(self.project, rule_id)
        if rule["comparison_identity_sha256"] != comparison_identity_sha256 or rule["rule"]["control"] != state["incumbent_evidence_ref"]:
            raise ValueError("Recovery rule must use the fixed incumbent")
        if type(max_runs) is not int or not 1 <= max_runs <= 3 or not all(str(x).strip() for x in (diagnosis, mechanism, stop_rule)):
            raise ValueError("Recovery requires diagnosis, mechanism, stop rule and 1-3 runs")
        probe = dict(id="recovery-" + uuid.uuid4().hex, comparison_identity_sha256=comparison_identity_sha256,
                     rule_id=rule_id, diagnosis=diagnosis, mechanism=mechanism, stop_rule=stop_rule, max_runs=max_runs,
                     candidates=[], registered_at=utc_now())
        with transaction(self.directory / "recovery_probes.json", {"probes": {}}) as value:
            # Renaming a recovery probe cannot replenish this pause generation.
            if any(p["comparison_identity_sha256"] == comparison_identity_sha256 and p.get("generation") == state["generation"] for p in value["probes"].values()):
                raise ValueError("Recovery budget already registered for this pause generation")
            probe["generation"] = state["generation"]
            value["probes"][probe["id"]] = probe
        return probe

    def resume_if_qualified(self):
        return self.reconcile_results()
