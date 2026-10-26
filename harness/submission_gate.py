"""One pre-dispatch authorization path for the tracker, CLI and Kaggle facade.

Permits are ephemeral single-use Python capabilities. Receipts and intents are
durable evidence, and cannot be loaded as permits. Unknown sends retain quota.
"""
from __future__ import annotations

import getpass
import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from harness.artifacts import digest, sha256
from harness.audit_gate import evaluate_submission_candidate, semantic_prediction_hash
from harness.decision_rules import (evaluate_decision_rule, load_registration, policy_directory,
                                    resolve_evidence, resolve_research_policy, utc_now)
from harness.dispatch_audit import append_dispatch_event, read_submission_events
from harness.io_utils import atomic_write_json
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.policy_schema import validate_document
from harness.project import HarnessProject
from harness.stagnation import StagnationController
from harness.state_store import workspace_lock
from harness.task_spec import load_active_task_spec

_ISSUER = object()
_PERMITS = {}


class SubmissionGateError(ValueError):
    def __init__(self, code, message=None, *, receipt=None):
        self.code = code
        self.blockers = [code]
        self.receipt = receipt
        super().__init__(code + (": " + message if message else ""))


class DispatchPermit:
    __slots__ = ("_token",)

    def __init__(self, issuer, token):
        if issuer is not _ISSUER:
            raise SubmissionGateError("INVALID_DISPATCH_PERMIT")
        self._token = token

    def __reduce__(self):
        raise TypeError("Dispatch permits cannot be serialized")


def consume_dispatch_permit(permit, *, file_name, competition, kernel=None, kernel_version=None):
    """Facade boundary: consume before constructing an SDK request."""
    if not isinstance(permit, DispatchPermit):
        raise SubmissionGateError("DISPATCH_PERMIT_REQUIRED")
    binding = _PERMITS.pop(permit._token, None)
    if not binding or time.monotonic() > binding["deadline"]:
        raise SubmissionGateError("EXPIRED_OR_CONSUMED_PERMIT")
    if (str(Path(file_name).resolve()) != binding["file_name"] or competition != binding["competition"] or
        kernel != binding["kernel"] or kernel_version != binding["kernel_version"] or sha256(file_name) != binding["payload_sha256"]):
        raise SubmissionGateError("PERMIT_BINDING_MISMATCH")
    return binding


def _intent_states(events, competition):
    latest = {}
    for event in events:
        if event["competition"] == competition:
            latest[event["intent_id"]] = event
    return latest


class SubmissionGate:
    def __init__(self, project_root=None):
        explicit = project_root or os.environ.get("HARNESS_PROJECT_ROOT")
        root = Path(explicit or Path.cwd()).resolve()
        # Never fall back to the installed framework when the caller has no workspace.
        if not (root / ".ai-harness/project.yaml").is_file() and not (root / ".ml-harness/project.yaml").is_file():
            raise SubmissionGateError("WORKSPACE_REQUIRED", "Attach a task workspace and pass --project-root.")
        self.project = HarnessProject.discover(root)
        self.spec = load_active_task_spec(self.project.task_config_path)
        self.context = TaskPluginContext(self.project.root, self.project.task_config_path, self.spec, self.project.plugin_id or "default")
        self.plugin = load_task_plugin(self.project.plugin_id, self.context)
        self.directory = policy_directory(self.project)

    def _history(self, competition):
        events = read_submission_events(self.project.root)
        latest = _intent_states(events, competition)
        if any(e["event"] in {"sending", "unknown"} for e in latest.values()):
            raise SubmissionGateError("UNRESOLVED_DISPATCH", "Reconcile the uncertain provider send before retrying.")
        refs, payloads, accounted_refs = [], [], set()
        for intent_id, event in latest.items():
            if event["event"] not in {"submitted", "reconciled", "prepared"}:
                continue
            intent_path = self.project.root / ".ai-harness/submission_intents" / (intent_id + ".json")
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if intent["competition"] != competition or intent["payload_sha256"] != event["payload_sha256"]:
                raise SubmissionGateError("HISTORY_INCOMPLETE")
            if event["event"] == "prepared":
                # Outstanding authorized sends reserve quota but are not comparison history yet.
                continue
            frame_path = Path(intent["prediction_snapshot"])
            if not frame_path.is_file() or sha256(frame_path) != intent["prediction_file_sha256"]:
                raise SubmissionGateError("HISTORY_INCOMPLETE", "A prior submitted prediction snapshot is missing or changed.")
            refs.append(dict(id=intent_id, path=str(frame_path), source="dispatched", exp_id=intent["exp_id"], provider_ref=event["provider_ref"]))
            payloads.append(intent["payload_sha256"])
            if event["provider_ref"] is not None:
                accounted_refs.add(str(event["provider_ref"]))
        # Import available legacy sends, failing closed for missing bytes or malformed lines.
        legacy = self.project.root / ".ai-harness/dispatch_log.jsonl"
        legacy_entries = []
        if legacy.exists():
            for line in legacy.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except ValueError as exc:
                    raise SubmissionGateError("HISTORY_INCOMPLETE") from exc
                if entry.get("kind") == "submission" and entry.get("competition") == competition and str(entry.get("ref")) not in accounted_refs:
                    legacy_entries.append(entry)
        # A legacy registered remote score is also evidence of prior dispatch.
        registry = self.project.artifacts_dir / "submissions/registry.json"
        if registry.exists():
            records = json.loads(registry.read_text(encoding="utf-8")).get("submissions", [])
            for entry in records:
                if entry.get("status") in {"SCORED", "SUBMITTED_PENDING"} and not entry.get("dispatch_intent_id"):
                    legacy_entries.append(entry)
        for entry in legacy_entries:
            prediction_path = entry.get("candidate_path")
            payload_hash = entry.get("file_sha256") or entry.get("sha256") or entry.get("payload_sha256")
            if payload_hash:
                payloads.append(payload_hash)
            if not prediction_path or not Path(prediction_path).is_file():
                raise SubmissionGateError("HISTORY_INCOMPLETE", "Materialize/reconcile legacy submission predictions before another send.")
            refs.append(dict(id=str(entry.get("id") or entry.get("ref")), path=prediction_path, source="dispatched", exp_id=entry.get("exp_id"), provider_ref=entry.get("ref")))
        return events, refs, payloads, digest({"events": events, "legacy": legacy_entries})

    def _resolve_payload(self, file_name, exp_id):
        path = Path(file_name)
        path = (self.project.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise SubmissionGateError("PAYLOAD_MISSING")
        registry_path = self.project.artifacts_dir / "submissions/registry.json"
        entries = json.loads(registry_path.read_text(encoding="utf-8")).get("submissions", []) if registry_path.exists() else []
        matches = [e for e in entries if e.get("archive_path") and Path(e["archive_path"]).resolve() == path]
        if not exp_id and matches:
            exp_id = matches[-1]["exp_id"]
        if not exp_id:
            raise SubmissionGateError("EXPERIMENT_EVIDENCE_REQUIRED", "Pass --exp-id for a sealed run with test predictions.")
        run_dir = (self.project.artifacts_dir / exp_id).resolve()
        run_dir.relative_to(self.project.artifacts_dir.resolve())
        cfg_path = run_dir / "run_config.yaml"
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        partition = cfg.get("validation", {}).get("partition", "oof")
        evidence = resolve_evidence(dict(kind="full_run", id=exp_id, partition=partition), project=self.project, task_spec=self.spec, plugin=self.plugin)
        predictions = run_dir / "test_predictions.csv"
        if "test_predictions.csv" not in evidence["manifest"]["files"] or not predictions.is_file():
            raise SubmissionGateError("TEST_EVIDENCE_REQUIRED")
        # Bind the actual file to plugin-built output from sealed test predictions.
        with tempfile.TemporaryDirectory(prefix="aih-payload-") as staging:
            built = self.plugin.build_submission(self.context, pd.read_csv(predictions), Path(staging) / path.name)
            built = Path(built or Path(staging) / path.name)
            self.plugin.validate_submission(self.context, built)
            if sha256(built) != sha256(path):
                raise SubmissionGateError("PAYLOAD_EVIDENCE_MISMATCH")
        self.plugin.validate_submission(self.context, path)
        return path, exp_id, predictions, evidence

    def evaluate(self, competition, file_name, *, exp_id=None, rule_id=None, user_approved=False,
                 approval_id=None, kernel=None, kernel_version=None, quota=None, max_spearman=None, provider_refs=None):
        if kernel or kernel_version:
            raise SubmissionGateError("UNVERIFIED_KERNEL_OUTPUT", "Exact completed kernel version/output binding is required; a local same-named file is insufficient.")
        declared_competition = self.spec.metadata.get("competition") or self.spec.metadata.get("competition_slug")
        if isinstance(declared_competition, dict):
            declared_competition = declared_competition.get("slug")
        if declared_competition and declared_competition != competition:
            raise SubmissionGateError("COMPETITION_MISMATCH")
        policy = resolve_research_policy(self.project, self.spec)
        threshold = policy.max_spearman if max_spearman is None else float(max_spearman)
        if not 0 < threshold <= policy.max_spearman:
            raise SubmissionGateError("POLICY_WEAKENING", "Outbound Spearman flags may strengthen the active policy only.")
        blocker = StagnationController(self.project).check_action("dispatch")
        from harness.macro_research import enforce_retrospective
        enforce_retrospective(self.project)
        if blocker:
            raise SubmissionGateError(blocker["code"], blocker["message"])
        state_path = self.project.artifacts_dir / "competition_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            gates = state.get("gates", {})
            if gates.get("freeze_locked") or gates.get("finalized"):
                raise SubmissionGateError("COMPETITION_FROZEN")
        path, exp_id, predictions, evidence = self._resolve_payload(file_name, exp_id)
        events, refs, payload_hashes, history_hash = self._history(competition)
        known_refs = {str(r["provider_ref"]) for r in refs if r.get("provider_ref") is not None}
        if provider_refs is not None and set(provider_refs) - known_refs:
            raise SubmissionGateError("HISTORY_INCOMPLETE", "Provider submissions are missing local prediction snapshots; reconcile them before sending.")
        if quota and type(quota.get("num_total")) is int and quota["num_total"] > len(known_refs):
            raise SubmissionGateError("HISTORY_INCOMPLETE", "The provider reports more submissions than the verified local history.")
        payload_hash = sha256(path)
        if payload_hash in payload_hashes:
            raise SubmissionGateError("DUPLICATE_PAYLOAD")
        diversity = evaluate_submission_candidate(predictions, refs, spec=self.spec, threshold=threshold)
        if any(r["reference_prediction_sha256"] == diversity["candidate_prediction_sha256"] for r in diversity["comparisons"]):
            raise SubmissionGateError("SEMANTIC_DUPLICATE")
        if diversity["status"] == "inconclusive" or (refs and diversity["status"] == "not_applicable"):
            raise SubmissionGateError("DIVERSITY_INCONCLUSIVE")
        evaluation_ids, reasons = [], []
        if refs:
            rule_id = rule_id or evidence["cfg"].get("research", {}).get("decision_rule_id")
            if not rule_id:
                raise SubmissionGateError("PROMOTION_EVIDENCE_REQUIRED")
            registration = load_registration(self.project, rule_id)
            evaluation = evaluate_decision_rule(registration["rule"], evidence["reference"], project=self.project,
                                               task_spec=self.spec, plugin=self.plugin, registration=registration)
            if evaluation["outcome"] != "supported":
                raise SubmissionGateError("PROMOTION_RULE_NOT_SATISFIED", ", ".join(evaluation["reason_codes"]))
            # The control must be the current verified incumbent, not an obsolete anchor.
            states = StagnationController(self.project).reconcile_results()
            state = states.get(evaluation["comparison_identity_sha256"])
            control_id = registration["rule"]["control"]["id"]
            prior_exp_ids = {r["exp_id"] for r in refs}
            if control_id not in prior_exp_ids:
                raise SubmissionGateError("INCUMBENT_COMPARISON_REQUIRED")
            prior_evidence = []
            for prior_exp in sorted(prior_exp_ids):
                if not prior_exp:
                    raise SubmissionGateError("HISTORY_INCOMPLETE", "A historical submission has no sealed experiment binding.")
                prior = resolve_evidence({"kind": "full_run", "id": prior_exp, "partition": evidence["reference"]["partition"]},
                    project=self.project, task_spec=self.spec, plugin=self.plugin)
                if prior["identity"] != evidence["identity"]:
                    raise SubmissionGateError("INCUMBENT_COMPARISON_REQUIRED", "Prior submissions need comparable validation evidence.")
                prior_evidence.append((prior["score"] if self.spec.direction == "maximize" else -prior["score"], prior_exp))
            best_prior_score = max(value for value, _ in prior_evidence)
            if control_id not in {prior_exp for value, prior_exp in prior_evidence if value == best_prior_score}:
                raise SubmissionGateError("INCUMBENT_COMPARISON_REQUIRED", "The predeclared control is worse than the submitted incumbent.")
            if state and state["incumbent_evidence_ref"]["id"] not in {control_id, exp_id}:
                raise SubmissionGateError("INCUMBENT_COMPARISON_REQUIRED")
            if diversity["status"] == "near_duplicate":
                near_ids = {r["reference_id"] for r in diversity["comparisons"] if r["status"] == "near_duplicate"}
                nearest_exp_ids = {r["exp_id"] for r in refs if r["id"] in near_ids}
                if not evaluation["independent_confirmation"] or nearest_exp_ids - {control_id}:
                    raise SubmissionGateError("INDEPENDENT_CONFIRMATION_REQUIRED")
            evaluation_ids.append(evaluation["evaluation_id"])
        subject = digest(dict(competition=competition, payload_sha256=payload_hash, exp_id=exp_id))
        approval = None
        if approval_id:
            if not approval_id.startswith("approval-") or not approval_id[9:].isalnum():
                raise SubmissionGateError("INVALID_APPROVAL")
            approval = json.loads((self.directory / (approval_id + ".json")).read_text(encoding="utf-8"))
            if approval["subject_sha256"] != subject or approval["action"] != "dispatch":
                raise SubmissionGateError("APPROVAL_BINDING_MISMATCH")
        elif user_approved:
            approval = dict(id="approval-" + uuid.uuid4().hex, action="dispatch", subject_sha256=subject,
                            actor=getpass.getuser(), time=utc_now())
            atomic_write_json(self.directory / (approval["id"] + ".json"), approval)
        else:
            reasons.append("DISPATCH_APPROVAL_REQUIRED")
        latest = _intent_states(events, competition)
        today = datetime.now(timezone.utc).date().isoformat()
        pending = sum(e["event"] in {"prepared", "sending", "unknown"} for e in latest.values())
        sent_today = sum(e["event"] in {"submitted", "reconciled"} and e["timestamp"].startswith(today) for e in latest.values())
        quota_snapshot = None
        if quota and type(quota.get("daily_quota")) is int and type(quota.get("num_today")) is int:
            limit, used, source = quota["daily_quota"], max(quota["num_today"], sent_today), "provider"
            if quota.get("limited_by_total") or quota.get("has_remaining_quota") is False:
                reasons.append("QUOTA_EXHAUSTED")
        else:
            quota_path = self.project.root / ".ai-harness/policies/submission_quota.json"
            local = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.exists() else {}
            if local.get("competition") != competition or not local.get("history_reconciled") or not local.get("operator_approved"):
                reasons.append("QUOTA_UNKNOWN")
                limit, used, source = None, None, None
            else:
                limit, used, source = local.get("daily_limit"), sent_today + int(local.get("external_used_today", 0)), "operator_reconciled_local_cap"
                if local.get("date") != today:
                    reasons.append("QUOTA_UNKNOWN")
        if limit is not None:
            if type(limit) is not int or limit < 1 or used < 0:
                raise SubmissionGateError("INVALID_QUOTA_POLICY")
            quota_snapshot = dict(limit=limit, used=used, reserved=pending, observed_at=utc_now(), source=source)
            if used + pending >= limit:
                reasons.append("QUOTA_EXHAUSTED")
        states = StagnationController(self.project).reconcile_results()
        receipt = dict(schema_version=1, receipt_id="gate-" + uuid.uuid4().hex, phase="dispatch",
            purpose="promotion" if refs else "initial", status="fail" if reasons else "pass", workspace_id=digest(str(self.project.root)),
            task_id=self.spec.metadata.get("id") or self.spec.metadata.get("name", "task"), competition=competition, session_id=None,
            policy_sha256=digest(policy.to_dict()), comparison_identity_sha256=digest(evidence["identity"]),
            payload=dict(kind="file", sha256=payload_hash, bytes=path.stat().st_size, filename=path.name, kernel=None, kernel_version=None, output_binding=None),
            prediction_sha256=diversity["candidate_prediction_sha256"], evidence_sha256=evidence["evidence_sha256"],
            decision_evaluation_ids=evaluation_ids, diversity=diversity, history_snapshot_sha256=history_hash, approval=approval,
            stagnation_state_sha256=digest(states), quota_snapshot=quota_snapshot,
            checks=dict(integrity="pass", diversity="pass", promotion="pass" if refs else "not_required",
                        approval="pass" if approval else "fail", quota="fail" if any(r.startswith("QUOTA") for r in reasons) else "pass"),
            reason_codes=list(dict.fromkeys(reasons)), created_at=utc_now(), expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            framework_commit="working-tree")
        validate_document("submission_gate_receipt", receipt)
        atomic_write_json(self.directory / (receipt["receipt_id"] + ".json"), receipt)
        return receipt, path, exp_id, predictions

    def authorize_dispatch(self, competition, file_name, *, dry_run=False, **options):
        receipt, path, exp_id, predictions = self.evaluate(competition, file_name, **options)
        if dry_run:
            return {"receipt": receipt, "dry_run": True}
        if receipt["status"] != "pass":
            raise SubmissionGateError(receipt["reason_codes"][0], receipt=receipt)
        # Content-addressed immutable snapshots outside completed evidence folders.
        stage = self.project.root / ".ai-harness/submission_payloads" / receipt["payload"]["sha256"]
        stage.mkdir(parents=True, exist_ok=True)
        staged_payload, staged_prediction = stage / path.name, stage / "prediction_snapshot.csv"
        if not staged_payload.exists():
            shutil.copyfile(path, staged_payload)
        if not staged_prediction.exists():
            shutil.copyfile(predictions, staged_prediction)
        if sha256(staged_payload) != receipt["payload"]["sha256"] or semantic_prediction_hash(pd.read_csv(staged_prediction), self.spec) != receipt["prediction_sha256"]:
            raise SubmissionGateError("STAGED_PAYLOAD_MISMATCH")
        intent_id = "intent-" + uuid.uuid4().hex
        binding = dict(intent_id=intent_id, receipt_id=receipt["receipt_id"], competition=competition,
                       payload_sha256=receipt["payload"]["sha256"], prediction_sha256=receipt["prediction_sha256"],
                       history_snapshot_sha256=receipt["history_snapshot_sha256"], session_id=None,
                       file_name=str(staged_payload.resolve()), prediction_snapshot=str(staged_prediction.resolve()),
                       prediction_file_sha256=sha256(staged_prediction), exp_id=exp_id, approval_id=receipt["approval"]["id"],
                       kernel=None, kernel_version=None)
        with workspace_lock(self.project.root / ".ai-harness"):
            from harness.macro_research import enforce_retrospective
            enforce_retrospective(self.project)
            _, _, _, current_hash = self._history(competition)
            if current_hash != receipt["history_snapshot_sha256"]:
                raise SubmissionGateError("HISTORY_CHANGED", "Dispatch history changed; evaluate again.")
            if digest(resolve_research_policy(self.project, self.spec).to_dict()) != receipt["policy_sha256"]:
                raise SubmissionGateError("POLICY_CHANGED")
            if digest(StagnationController(self.project).reconcile_results()) != receipt["stagnation_state_sha256"]:
                raise SubmissionGateError("STAGNATION_STATE_CHANGED")
            state_path = self.project.artifacts_dir / "competition_state.json"
            if state_path.is_file():
                gates = json.loads(state_path.read_text(encoding="utf-8")).get("gates", {})
                if gates.get("freeze_locked") or gates.get("finalized"):
                    raise SubmissionGateError("COMPETITION_FROZEN")
            latest = _intent_states(read_submission_events(self.project.root), competition)
            for old_id in latest:
                old = json.loads((self.project.root / ".ai-harness/submission_intents" / (old_id + ".json")).read_text())
                if old["approval_id"] == binding["approval_id"]:
                    raise SubmissionGateError("APPROVAL_ALREADY_USED")
            atomic_write_json(self.project.root / ".ai-harness/submission_intents" / (intent_id + ".json"), binding)
            self._event(binding, "prepared")
            self._event(binding, "sending")
        token = uuid.uuid4().hex
        _PERMITS[token] = {**binding, "deadline": time.monotonic() + 300}
        return {"receipt": receipt, "intent": binding, "permit": DispatchPermit(_ISSUER, token)}

    def _event(self, intent, event, *, provider_ref=None, error_code=None):
        return append_dispatch_event(self.project.root, dict(schema_version=1, event_id="event-" + uuid.uuid4().hex,
            intent_id=intent["intent_id"], event=event, payload_sha256=intent["payload_sha256"], prediction_sha256=intent["prediction_sha256"],
            history_snapshot_sha256=intent["history_snapshot_sha256"], competition=intent["competition"], session_id=intent["session_id"],
            receipt_id=intent["receipt_id"], timestamp=utc_now(), provider_ref=provider_ref, error_code=error_code))

    def finish_submission_intent(self, authorization, *, provider_ref=None, error=None):
        _PERMITS.pop(authorization["permit"]._token, None)
        intent = authorization["intent"]
        if isinstance(error, SubmissionGateError):
            return self._event(intent, "failed", error_code=error.code)
        if error is not None or provider_ref is None:
            # A transport exception or missing ref is not evidence of rejection.
            return self._event(intent, "unknown", error_code=type(error).__name__ if error else "MISSING_PROVIDER_REF")
        event = self._event(intent, "submitted", provider_ref=provider_ref)
        from harness.submission_manager import SubmissionManager
        registry_path = self.project.artifacts_dir / "submissions/registry.json"
        if registry_path.exists():
            manager = SubmissionManager(project_root=self.project.root)
            for entry in manager.submissions.values():
                if entry.exp_id == intent["exp_id"] and entry.payload_sha256 == intent["payload_sha256"]:
                    entry.dispatch_intent_id = intent["intent_id"]
                    entry.gate_receipt_id = intent["receipt_id"]
                    entry.approval_id = intent["approval_id"]
                    entry.status = "SUBMITTED_PENDING"
            manager.save()
        return event

    def reconcile_submission_intent(self, intent_id, *, client, provider_ref):
        if not intent_id.startswith("intent-") or not intent_id[7:].isalnum():
            raise SubmissionGateError("INVALID_INTENT")
        intent = json.loads((self.project.root / ".ai-harness/submission_intents" / (intent_id + ".json")).read_text())
        matches = [s for s in client.list_submissions(intent["competition"]) if str(s.ref) == str(provider_ref)]
        # The outbound request carries an unpredictable durable intent marker.
        # A filename or manually supplied score alone never identifies that request.
        if (not matches or getattr(matches[0], "file_name", None) != Path(intent["file_name"]).name or
            "[aih:" + intent_id + "]" not in str(getattr(matches[0], "description", ""))):
            raise SubmissionGateError("RECONCILIATION_UNVERIFIED")
        with workspace_lock(self.project.root / ".ai-harness"):
            states = _intent_states(read_submission_events(self.project.root), intent["competition"])
            if states.get(intent_id, {}).get("event") not in {"unknown", "sending", "submitted", "reconciled"}:
                raise SubmissionGateError("INVALID_RECONCILIATION_STATE")
            return self._event(intent, "reconciled", provider_ref=provider_ref)

    def resolve_official_score(self, intent_id, *, client, provider_ref, partition="public"):
        if partition not in {"public", "private"}:
            raise SubmissionGateError("INVALID_SCORE_PARTITION")
        self.reconcile_submission_intent(intent_id, client=client, provider_ref=provider_ref)
        intent = json.loads((self.project.root / ".ai-harness/submission_intents" / (intent_id + ".json")).read_text())
        submission = next(s for s in client.list_submissions(intent["competition"]) if str(s.ref) == str(provider_ref))
        score = getattr(submission, partition + "_score", None)
        import math
        if "complete" not in str(getattr(submission, "status", "")).lower() or score is None or not math.isfinite(float(score)):
            raise SubmissionGateError("OFFICIAL_SCORE_NOT_TERMINAL")
        observation = dict(observation_id="official-" + uuid.uuid4().hex, intent_id=intent_id,
            competition=intent["competition"], provider_ref=str(provider_ref), partition=partition,
            metric=self.spec.primary_metric, score=float(score), retrieved_at=utc_now(), source="provider_api",
            payload_sha256=intent["payload_sha256"], prediction_sha256=intent["prediction_sha256"],
            raw_response_sha256=digest({"status": str(submission.status), "score": score, "description": submission.description}))
        atomic_write_json(self.directory / (observation["observation_id"] + ".json"), observation)
        return observation
