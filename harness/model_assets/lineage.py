"""Candidate model lineage and aggregate competition-policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from harness.model_assets.policy import (
    evaluate_composition_policy,
    evaluate_model_policy,
    normalize_competition_rules,
)
from harness.model_assets.types import (
    MODEL_PARAMETER_COUNT_UNKNOWN,
    MODEL_PARAMETER_LIMIT_EXCEEDED,
    CompetitionRules,
    ModelDefinition,
    PolicyAssessment,
)


MODEL_LINEAGE_UNVERIFIED = "MODEL_LINEAGE_UNVERIFIED"
MODEL_POLICY_STALE = "MODEL_POLICY_STALE"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def model_policy_digest(
    policy: Mapping[str, Any] | CompetitionRules,
    governance: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash the declared policy using the same canonical form as model locks."""
    if isinstance(policy, CompetitionRules):
        from dataclasses import asdict

        value: Mapping[str, Any] = asdict(policy)
    elif isinstance(policy, Mapping):
        value = policy
    else:
        raise TypeError("model policy must be a mapping or CompetitionRules")
    if governance is not None:
        if not isinstance(governance, Mapping):
            raise TypeError("model governance must be a mapping")
        value = {"competition_rules": value, "model_governance": governance}
    return _canonical_digest(value)


def semantic_candidate_prediction_hash(frame: Any, task_spec: Any) -> str:
    """Hash task predictions while excluding run-only fold/split annotations."""
    from harness.audit_gate import semantic_prediction_hash

    normalized = frame.drop(columns=["fold", "input_variant", "split"], errors="ignore")
    return semantic_prediction_hash(normalized, task_spec)


@dataclass(frozen=True)
class CandidateSubject:
    """Content-addressed identity for one candidate and its model evidence."""

    task_id: str
    recipe: Mapping[str, Any]
    source_manifest_digests: Sequence[str]
    model_lock_digests: Sequence[str]
    semantic_prediction_hash: str
    candidate_subject_digest: str = ""
    model_policy_digest: str = ""
    model_lineage_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("Candidate subject task_id must be a non-empty string")
        if not isinstance(self.recipe, Mapping):
            raise ValueError("Candidate subject recipe must be a mapping")
        for label, values in (
            ("source_manifest_digests", self.source_manifest_digests),
            ("model_lock_digests", self.model_lock_digests),
        ):
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in values
            ):
                raise ValueError(f"{label} must contain SHA-256 digests")
        for label, value in (("semantic_prediction_hash", self.semantic_prediction_hash),
                             ("model_policy_digest", self.model_policy_digest),
                             ("model_lineage_digest", self.model_lineage_digest)):
            if value and (not isinstance(value, str) or len(value) != 64):
                raise ValueError(f"{label} must be a SHA-256 digest")

        identity = self._identity_dict()
        expected = _canonical_digest(identity)
        if self.candidate_subject_digest and self.candidate_subject_digest != expected:
            raise ValueError("Candidate subject digest does not match its bindings")
        object.__setattr__(self, "candidate_subject_digest", expected)
        object.__setattr__(self, "recipe", json.loads(_canonical_json(dict(self.recipe))))
        object.__setattr__(self, "source_manifest_digests", tuple(self.source_manifest_digests))
        object.__setattr__(self, "model_lock_digests", tuple(self.model_lock_digests))

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id.strip(),
            "model_policy_digest": self.model_policy_digest,
            "recipe": dict(self.recipe),
            "source_manifest_digests": list(self.source_manifest_digests),
            "model_lock_digests": list(self.model_lock_digests),
            "model_lineage_digest": self.model_lineage_digest,
            "semantic_prediction_hash": self.semantic_prediction_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self._identity_dict(),
            "candidate_subject_digest": self.candidate_subject_digest,
        }


@dataclass
class CandidateModelClosure:
    """Resolved inference members and evidence status for a candidate."""

    models: List[ModelDefinition] = field(default_factory=list)
    components: List[Dict[str, Any]] = field(default_factory=list)
    source_manifest_digests: List[str] = field(default_factory=list)
    model_lock_digests: List[str] = field(default_factory=list)
    source_policy_digests: List[str] = field(default_factory=list)
    model_lock_digest: str = ""
    model_lineage_digest: str = ""
    verified: bool = True
    governance_required: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": list(self.components),
            "source_manifest_digests": list(self.source_manifest_digests),
            "model_lock_digests": list(self.model_lock_digests),
            "source_policy_digests": list(self.source_policy_digests),
            "model_lock_digest": self.model_lock_digest,
            "model_lineage_digest": self.model_lineage_digest,
            "verified": self.verified,
            "governance_required": self.governance_required,
            "reasons": list(self.reasons),
        }


def _load_mapping(value: Any, label: str) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        try:
            loaded = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} could not be read: {exc}") from exc
        if isinstance(loaded, Mapping):
            return loaded
    if value is None:
        return None
    raise ValueError(f"{label} must be a mapping or JSON path")


def _manifest_digest(source: Mapping[str, Any]) -> str:
    for key in ("source_manifest_digest", "manifest_sha256", "artifact_manifest_sha256"):
        value = source.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    manifest = source.get("artifact_manifest")
    if isinstance(manifest, Mapping):
        return _canonical_digest(manifest)
    excluded = {
        "model_lock",
        "model_lock_data",
        "model_lock_path",
        "model_policy_receipt",
        "model_policy_receipt_path",
        "source_manifest_digest",
        "manifest_sha256",
        "artifact_manifest_sha256",
    }
    return _canonical_digest({key: value for key, value in source.items() if key not in excluded})


def _lock_payload_digest(lock: Mapping[str, Any]) -> str:
    return _canonical_digest({key: value for key, value in lock.items() if key != "model_lock_digest"})


def _source_id(source: Mapping[str, Any], index: int) -> str:
    for key in ("exp_id", "run_id", "source_id", "id"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"source-{index}"


def _required_from_lock(lock: Mapping[str, Any], receipt: Optional[Mapping[str, Any]]) -> bool:
    governance = str(lock.get("governance_mode", "")).strip().lower()
    return governance == "required" or bool(receipt and receipt.get("required") is True)


def resolve_candidate_model_closure(
    source_manifests: Sequence[Mapping[str, Any]], recipe: Mapping[str, Any]
) -> CandidateModelClosure:
    """Resolve every deployed model root from all candidate constituent runs.

    Each source mapping may contain its validated artifact manifest plus
    ``model_lock`` and ``model_policy_receipt`` mappings loaded from that run.
    Missing or inconsistent locks remain visible as an unverified closure.
    Components are intentionally not deduplicated across runs: two runs in a
    blend are two deployed inference members even when their definitions match.
    """
    if not isinstance(source_manifests, Sequence) or isinstance(
        source_manifests, (str, bytes)
    ):
        raise ValueError("source_manifests must be a sequence of mappings")
    if not isinstance(recipe, Mapping):
        raise ValueError("recipe must be a mapping")

    closure = CandidateModelClosure()
    source_ids: List[str] = []
    lock_digest_values: List[Optional[str]] = []
    for index, raw_source in enumerate(source_manifests):
        if not isinstance(raw_source, Mapping):
            raise ValueError("Every source manifest must be a mapping")
        source = raw_source
        source_id = _source_id(source, index)
        source_ids.append(source_id)
        closure.source_manifest_digests.append(_manifest_digest(source))

        lock_value = source.get("model_lock", source.get("model_lock_data"))
        lock_path = source.get("model_lock_path")
        if lock_value is None and lock_path:
            lock_value = lock_path
        lock = _load_mapping(lock_value, f"model lock for {source_id}")
        receipt_value = source.get("model_policy_receipt")
        if receipt_value is None and source.get("model_policy_receipt_path"):
            receipt_value = source.get("model_policy_receipt_path")
        receipt = _load_mapping(receipt_value, f"model policy receipt for {source_id}")

        if lock is None:
            closure.verified = False
            closure.reasons.append(f"Model lock is missing for source {source_id}.")
            lock_digest_values.append(None)
            continue

        declared_digest = lock.get("model_lock_digest")
        manifest_digest = source.get("model_lock_digest")
        computed_digest = _lock_payload_digest(lock)
        if (
            not isinstance(declared_digest, str)
            or declared_digest != computed_digest
            or (manifest_digest is not None and manifest_digest != declared_digest)
        ):
            closure.verified = False
            closure.reasons.append(f"Model lock digest is inconsistent for source {source_id}.")
        else:
            closure.model_lock_digests.append(declared_digest)
        sealed_files = source.get("files")
        if lock_path and isinstance(sealed_files, Mapping):
            expected_file_digest = sealed_files.get("model_lock.json")
            try:
                actual_file_digest = hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()
            except OSError:
                actual_file_digest = None
            if not expected_file_digest or actual_file_digest != expected_file_digest:
                closure.verified = False
                closure.reasons.append(
                    f"Sealed model lock file differs for source {source_id}."
                )
        receipt_path = source.get("model_policy_receipt_path")
        if receipt_path and isinstance(sealed_files, Mapping):
            expected_receipt_file_digest = sealed_files.get("model_policy_receipt.json")
            try:
                actual_receipt_file_digest = hashlib.sha256(
                    Path(receipt_path).read_bytes()
                ).hexdigest()
            except OSError:
                actual_receipt_file_digest = None
            if not expected_receipt_file_digest or actual_receipt_file_digest != expected_receipt_file_digest:
                closure.verified = False
                closure.reasons.append(
                    f"Sealed model policy receipt differs for source {source_id}."
                )
        lock_digest_values.append(declared_digest if isinstance(declared_digest, str) else None)

        policy_digest = lock.get("policy_digest")
        if isinstance(policy_digest, str) and len(policy_digest) == 64:
            closure.source_policy_digests.append(policy_digest)
        else:
            closure.verified = False
            closure.reasons.append(f"Model policy digest is missing for source {source_id}.")

        closure.governance_required = closure.governance_required or _required_from_lock(lock, receipt)
        if receipt is not None:
            if receipt.get("model_lock_digest") != declared_digest:
                closure.verified = False
                closure.reasons.append(f"Model policy receipt is not bound to source {source_id}'s lock.")
            if receipt.get("policy_digest") != policy_digest:
                closure.verified = False
                closure.reasons.append(f"Model policy receipt digest differs for source {source_id}.")
            if receipt.get("status") == "blocked" or receipt.get("eligible") is False:
                closure.verified = False
                closure.reasons.append(f"Model policy receipt blocks source {source_id}.")
            if receipt.get("status") != "eligible" or receipt.get("eligible") is not True:
                closure.verified = False
                if not any(f"source {source_id}" in reason for reason in closure.reasons):
                    closure.reasons.append(
                        f"Model policy receipt is unverified for source {source_id}."
                    )
            if closure.governance_required and (
                receipt.get("status") != "eligible" or receipt.get("eligible") is not True
            ):
                closure.verified = False
                closure.reasons.append(
                    f"Required model policy receipt is not eligible for source {source_id}."
                )
        else:
            closure.verified = False
            if closure.governance_required:
                closure.reasons.append(f"Required model policy receipt is missing for source {source_id}.")
            else:
                closure.reasons.append(f"Model policy receipt is missing for source {source_id}.")

        roots = lock.get("roots")
        if not isinstance(roots, list):
            closure.verified = False
            closure.reasons.append(f"Model lock roots are malformed for source {source_id}.")
            continue
        if closure.governance_required and not roots:
            closure.verified = False
            closure.reasons.append(f"Required model lock has no inference roots for source {source_id}.")

        for root_index, root in enumerate(roots):
            if not isinstance(root, Mapping):
                closure.verified = False
                closure.reasons.append(f"Model lock root {root_index} is malformed for source {source_id}.")
                continue
            definition_raw = root.get("definition")
            if not isinstance(definition_raw, Mapping):
                closure.verified = False
                closure.reasons.append(f"Model definition is missing for source {source_id} root {root_index}.")
                continue
            try:
                definition = ModelDefinition.from_dict(definition_raw)
            except (TypeError, ValueError) as exc:
                closure.verified = False
                closure.reasons.append(f"Model definition is invalid for source {source_id}: {exc}")
                continue
            expected_definition_digest = root.get("definition_digest")
            if expected_definition_digest and definition.compute_digest() != expected_definition_digest:
                closure.verified = False
                closure.reasons.append(
                    f"Model definition digest is inconsistent for {definition.model_id} in source {source_id}."
                )
            if root.get("model_id") not in (None, definition.model_id):
                closure.verified = False
                closure.reasons.append(f"Model root identity differs for source {source_id}.")
            closure.models.append(definition)
            closure.components.append(
                {
                    "source_id": source_id,
                    "slot": str(root.get("slot", root_index)),
                    "role": str(root.get("role", root.get("slot", root_index))),
                    "model_id": definition.model_id,
                    "definition_digest": definition.compute_digest(),
                    "model_lock_digest": declared_digest,
                }
            )

    declared_source_ids = recipe.get("source_ids") or recipe.get("constituent_ids")
    if declared_source_ids is not None:
        if not isinstance(declared_source_ids, list) or any(
            not isinstance(value, str) for value in declared_source_ids
        ):
            raise ValueError("recipe source_ids must be a list of strings")
        if set(declared_source_ids) != set(source_ids):
            closure.verified = False
            closure.reasons.append("Recipe source IDs do not match the supplied source manifests.")

    closure.model_lock_digest = _canonical_digest(
        {"source_model_lock_digests": lock_digest_values}
    )
    closure.model_lineage_digest = _canonical_digest(
        {
            "recipe": dict(recipe),
            "components": closure.components,
            "source_manifest_digests": closure.source_manifest_digests,
            "model_lock_digests": lock_digest_values,
        }
    )
    return closure


def _policy_restricts_models(policy: Mapping[str, Any] | CompetitionRules) -> bool:
    if isinstance(policy, CompetitionRules):
        return any(
            value is not None
            for value in (
                policy.max_total_parameters,
                policy.max_llm_parameters,
                policy.model_weight_release_cutoff,
                policy.allowed_licenses,
            )
        ) or not policy.allow_pretrained_models or policy.allow_api or policy.allow_external_data
    return any(
        policy.get(key) is not None
        for key in (
            "max_total_parameters",
            "max_llm_parameters",
            "model_weight_release_cutoff",
            "allowed_licenses",
        )
    ) or policy.get("allow_pretrained_models") is False or policy.get("allow_api") is True or policy.get(
        "allow_external_data"
    ) is True


def evaluate_candidate_model_policy(
    model_closure: CandidateModelClosure | Sequence[ModelDefinition],
    policy: Mapping[str, Any] | CompetitionRules,
) -> PolicyAssessment:
    """Re-evaluate component rules and aggregate deployment limits for a candidate."""
    if isinstance(model_closure, CandidateModelClosure):
        closure = model_closure
        models = closure.models
        required = closure.governance_required
        verified = closure.verified
        lineage_facts = closure.to_dict()
    else:
        if not isinstance(model_closure, Sequence) or any(
            not isinstance(model, ModelDefinition) for model in model_closure
        ):
            raise ValueError("model_closure must be a CandidateModelClosure or ModelDefinition sequence")
        closure = None
        models = list(model_closure)
        required = False
        verified = True
        lineage_facts = {}

    restricted = _policy_restricts_models(policy)
    if closure is not None and isinstance(policy, Mapping):
        current_policy_digest = model_policy_digest(policy)
        stale_sources = [
            value for value in closure.source_policy_digests if value != current_policy_digest
        ]
        if stale_sources and (required or restricted or models):
            return PolicyAssessment(
                eligible=False,
                status="blocked",
                failure_code=MODEL_POLICY_STALE,
                reasons=["A source model lock was assessed under a different competition policy."],
                facts_evaluated={
                    "lineage": lineage_facts,
                    "current_model_policy_digest": current_policy_digest,
                    "source_policy_digests": list(closure.source_policy_digests),
                },
            )
    if not verified and (required or restricted):
        return PolicyAssessment(
            eligible=False,
            status="blocked",
            failure_code=MODEL_LINEAGE_UNVERIFIED,
            reasons=list(closure.reasons if closure else [])
            or ["Candidate model closure is unverified under the active governance policy."],
            facts_evaluated={"lineage": lineage_facts, "model_count": len(models)},
        )
    if not models:
        if required or restricted:
            return PolicyAssessment(
                eligible=False,
                status="blocked",
                failure_code=MODEL_PARAMETER_COUNT_UNKNOWN,
                reasons=["The active competition policy requires a verified model closure."],
                facts_evaluated={"lineage": lineage_facts, "model_count": 0},
            )
        return PolicyAssessment(
            eligible=True,
            status="not_required",
            failure_code=None,
            reasons=["No model restrictions apply and no model closure was declared."],
            facts_evaluated={"lineage": lineage_facts, "model_count": 0},
        )

    if not verified:
        return PolicyAssessment(
            eligible=True,
            status="unverified",
            failure_code=None,
            reasons=list(closure.reasons if closure else [])
            or ["Candidate models are declared but their complete lineage is unverified."],
            facts_evaluated={"lineage": lineage_facts, "model_count": len(models)},
        )

    rules = policy if isinstance(policy, CompetitionRules) else normalize_competition_rules(dict(policy))
    for model in models:
        individual = evaluate_model_policy(model, rules)
        if not individual.eligible:
            facts = dict(individual.facts_evaluated)
            facts["lineage"] = lineage_facts
            return PolicyAssessment(
                eligible=False,
                status="blocked",
                failure_code=individual.failure_code,
                reasons=list(individual.reasons),
                facts_evaluated=facts,
            )

    aggregate = (
        evaluate_model_policy(models[0], rules)
        if len(models) == 1
        else evaluate_composition_policy(models, rules)
    )
    aggregate_facts = dict(aggregate.facts_evaluated)
    aggregate_facts["lineage"] = lineage_facts
    if not aggregate.eligible:
        return PolicyAssessment(
            eligible=False,
            status="blocked",
            failure_code=aggregate.failure_code or MODEL_PARAMETER_LIMIT_EXCEEDED,
            reasons=list(aggregate.reasons),
            facts_evaluated=aggregate_facts,
        )
    return PolicyAssessment(
        eligible=True,
        status="eligible",
        failure_code=None,
        reasons=list(aggregate.reasons),
        facts_evaluated=aggregate_facts,
    )

