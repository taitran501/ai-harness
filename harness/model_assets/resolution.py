"""Resolve effective experiment configs and immutable model admission plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import yaml

from harness.model_assets.policy import (
    evaluate_composition_policy,
    evaluate_model_policy,
    normalize_competition_rules,
)
from harness.model_assets.store import MAX_MODEL_DEFINITION_BYTES, ModelStore
from harness.model_assets.types import (
    MODEL_GOVERNANCE_MODE_REQUIRED,
    MODEL_PARAMETER_COUNT_UNKNOWN,
    MODEL_PLAN_INVALID,
    MODEL_REFERENCE_MISSING,
    ModelDefinition,
    PolicyAssessment,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
def _has_governance_restrictions(rules: Mapping[str, Any]) -> bool:
    return any(
        (
            rules.get("max_total_parameters") is not None,
            rules.get("max_llm_parameters") is not None,
            rules.get("model_weight_release_cutoff") is not None,
            rules.get("allow_pretrained_models") is False,
            rules.get("allow_api") is False,
            rules.get("allow_external_data") is False,
            rules.get("allowed_licenses") is not None,
            rules.get("parameter_accounting") is not None,
        )
    )


class ModelAdmissionError(ValueError):
    """A model plan was not eligible to enter an execution path."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}")


@dataclass(frozen=True)
class ModelPlan:
    """Immutable admission result with portable lock and policy receipt data."""

    task_id: str
    governance_mode: str
    required: bool
    declared: bool
    eligible: Optional[bool]
    status: str
    failure_code: Optional[str]
    message: str
    lock_json: str
    receipt_json: str
    lock_digest: str

    @property
    def model_lock(self) -> Dict[str, Any]:
        """Return a fresh JSON-compatible copy of the portable model lock."""
        return json.loads(self.lock_json)

    @property
    def policy_receipt(self) -> Dict[str, Any]:
        """Return a fresh JSON-compatible copy of the policy decision."""
        return json.loads(self.receipt_json)

    @property
    def receipt(self) -> Dict[str, Any]:
        """Compatibility alias for :attr:`policy_receipt`."""
        return self.policy_receipt

    def to_lock_dict(self) -> Dict[str, Any]:
        return self.model_lock

    def to_receipt_dict(self) -> Dict[str, Any]:
        return self.policy_receipt


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _project_root(project: Any) -> Path:
    root = getattr(project, "root", project)
    return Path(root or os.environ.get("HARNESS_PROJECT_ROOT") or Path.cwd()).expanduser().resolve()


def _load_yaml_mapping(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {path}")
    return loaded


def resolve_effective_config(
    project: Any,
    config_path_or_dict: str | Path | Mapping[str, Any],
    args: Any = None,
) -> Dict[str, Any]:
    """Load an experiment config and merge the same base config for every backend.

    The workspace ``configs/base.yaml`` takes precedence over the packaged
    default. ``args.base_config`` selects an explicit base config. A mapping
    may be supplied by callers that already loaded the experiment document.
    """
    root = _project_root(project)
    if isinstance(config_path_or_dict, Mapping):
        experiment_config = deepcopy(dict(config_path_or_dict))
    else:
        config_path = Path(config_path_or_dict).expanduser()
        if not config_path.is_absolute():
            config_path = root / config_path
        experiment_config = _load_yaml_mapping(config_path.resolve(), "Experiment config")

    explicit_base = _arg(args, "base_config")
    if explicit_base:
        if isinstance(explicit_base, Mapping):
            base_config = deepcopy(dict(explicit_base))
        else:
            base_path = Path(explicit_base).expanduser()
            if not base_path.is_absolute():
                base_path = root / base_path
            base_config = _load_yaml_mapping(base_path.resolve(), "Base config")
    else:
        workspace_base = root / "configs" / "base.yaml"
        if workspace_base.is_file():
            base_config = _load_yaml_mapping(workspace_base, "Base config")
        else:
            packaged_base = Path(__file__).resolve().parents[1] / "defaults" / "base.yaml"
            base_config = _load_yaml_mapping(packaged_base, "Packaged base config")
    return _deep_merge(base_config, experiment_config)


def _task_governance(task_spec: Any) -> Dict[str, Any]:
    if isinstance(task_spec, Mapping):
        metadata = task_spec.get("metadata", task_spec.get("task", task_spec))
        value = metadata.get("model_governance", {}) if isinstance(metadata, Mapping) else {}
        return dict(value) if isinstance(value, Mapping) else {}
    value = getattr(task_spec, "model_governance_config", {})
    if callable(value):
        value = value()
    return dict(value) if isinstance(value, Mapping) else {}


def _task_rules(task_spec: Any) -> Dict[str, Any]:
    if isinstance(task_spec, Mapping):
        metadata = task_spec.get("metadata", task_spec.get("task", task_spec))
        value = metadata.get("competition_rules", {}) if isinstance(metadata, Mapping) else {}
        return dict(value) if isinstance(value, Mapping) else {}
    value = getattr(task_spec, "competition_rules", {})
    if callable(value):
        value = value()
    return dict(value) if isinstance(value, Mapping) else {}


def _task_id(task_spec: Any, project: Any) -> str:
    manifest = getattr(project, "manifest", None) or {}
    task_id = manifest.get("task_id") if isinstance(manifest, Mapping) else None
    if isinstance(task_spec, Mapping):
        metadata = task_spec.get("metadata", task_spec.get("task", task_spec)) or {}
        fallback_name = metadata.get("name", "task") if isinstance(metadata, Mapping) else "task"
    else:
        metadata = getattr(task_spec, "metadata", {}) or {}
        fallback_name = getattr(task_spec, "name", "task")
    task_id = task_id or (metadata.get("id") if isinstance(metadata, Mapping) else None)
    return str(task_id or fallback_name)


def _model_assets_config(config: Mapping[str, Any]) -> Any:
    selected = config.get("model_assets")
    if selected is not None:
        return selected
    model = config.get("model")
    if isinstance(model, Mapping):
        return model.get("model_assets", model.get("assets"))
    return None


def _read_catalog(path: Path) -> Dict[str, Any]:
    from harness.model_assets.store import _parse_document, _read_bounded

    payload = _read_bounded(path)
    if len(payload) > MAX_MODEL_DEFINITION_BYTES:
        raise ValueError(f"Model catalog exceeds {MAX_MODEL_DEFINITION_BYTES} bytes: {path}")
    document = _parse_document(path, payload)
    if set(document) - {"schema_version", "models"}:
        unknown = ", ".join(sorted(set(document) - {"schema_version", "models"}))
        raise ValueError(f"Unknown model catalog field(s): {unknown}")
    version = document.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Model catalog schema_version must be 1")
    models = document.get("models", {})
    if not isinstance(models, Mapping) or any(not isinstance(key, str) for key in models):
        raise ValueError("Model catalog 'models' must be a mapping of aliases")
    return dict(models)


def _safe_workspace_path(root: Path, value: str, *, relative_to: Optional[Path] = None) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (relative_to or root) / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Model definition path escapes the task workspace: {value}") from exc
    return resolved


def _resolve_definition(
    ref: Any,
    *,
    store: ModelStore,
    root: Path,
    catalog: Mapping[str, Any],
) -> ModelDefinition:
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("Each model_assets root must declare a non-empty 'ref'")
    reference = ref.strip()
    if reference in catalog:
        entry = catalog[reference]
        if isinstance(entry, str):
            reference = entry
        elif isinstance(entry, Mapping) and set(entry) == {"definition"}:
            reference = str(entry["definition"])
        else:
            raise ValueError(f"Invalid model catalog entry for alias {reference!r}")
    if _SHA256.fullmatch(reference):
        definition = store.load_definition(reference, root)
        if definition is None:
            raise ValueError(f"Registered model definition is missing for digest {reference}")
        return definition
    path = _safe_workspace_path(root, reference)
    if not path.is_file():
        raise ValueError(
            f"Model reference {reference!r} is neither a catalog alias, a registered digest, nor a definition file"
        )
    return store.load_catalog_file(path)


def _failed_plan(
    *,
    task_id: str,
    mode: str,
    required: bool,
    declared: bool,
    code: str,
    message: str,
    policy_digest: str,
    roots: list[Dict[str, Any]],
) -> ModelPlan:
    lock = {
        "schema_version": 1,
        "task_id": task_id,
        "governance_mode": mode,
        "roots": roots,
        "policy_digest": policy_digest,
    }
    lock_bytes = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    lock_digest = hashlib.sha256(lock_bytes.encode("utf-8")).hexdigest()
    lock["model_lock_digest"] = lock_digest
    receipt = {
        "schema_version": 1,
        "task_id": task_id,
        "governance_mode": mode,
        "required": required,
        "declared": declared,
        "status": "blocked",
        "eligible": False,
        "failure_code": code,
        "reasons": [message],
        "policy_digest": policy_digest,
        "model_lock_digest": lock_digest,
    }
    return ModelPlan(
        task_id=task_id,
        governance_mode=mode,
        required=required,
        declared=declared,
        eligible=False,
        status="blocked",
        failure_code=code,
        message=message,
        lock_json=json.dumps(lock, sort_keys=True, ensure_ascii=False),
        receipt_json=json.dumps(receipt, sort_keys=True, ensure_ascii=False),
        lock_digest=lock_digest,
    )


def prepare_or_validate_model_plan(
    task_spec: Any,
    effective_config: Mapping[str, Any],
    model_store: Optional[ModelStore] = None,
    project: Any = None,
) -> ModelPlan:
    """Resolve selected definitions, evaluate competition rules, and freeze a lock."""
    if not isinstance(effective_config, Mapping):
        raise ValueError("Effective experiment config must be a mapping")
    root = _project_root(project)
    store = model_store or ModelStore()
    governance = _task_governance(task_spec)
    rules_raw = _task_rules(task_spec)
    task_id = _task_id(task_spec, project)
    mode_value = governance.get("mode", "legacy")
    mode = str(mode_value).strip().lower()
    if mode not in {"legacy", "optional", "required"}:
        return _failed_plan(
            task_id=task_id,
            mode=mode,
            required=True,
            declared=False,
            code=MODEL_PLAN_INVALID,
            message="model_governance.mode must be 'optional' or 'required'.",
            policy_digest=_digest(rules_raw),
            roots=[],
        )
    declared_config = _model_assets_config(effective_config)
    required = mode == "required"
    has_restrictions = _has_governance_restrictions(rules_raw)
    if mode == "legacy" and has_restrictions:
        return _failed_plan(
            task_id=task_id,
            mode="migration_required",
            required=True,
            declared=False,
            code=MODEL_GOVERNANCE_MODE_REQUIRED,
            message=(
                "Competition model restrictions are declared without "
                "model_governance.mode: required; declare governed model assets before execution."
            ),
            policy_digest=_digest(rules_raw),
            roots=[],
        )

    if declared_config is None:
        if required:
            return _failed_plan(
                task_id=task_id,
                mode=mode,
                required=True,
                declared=False,
                code=MODEL_REFERENCE_MISSING,
                message="Required model governance has no model_assets roots declaration.",
                policy_digest=_digest(rules_raw),
                roots=[],
            )
        return _build_plan(
            task_id=task_id,
            mode=mode,
            required=False,
            declared=False,
            eligible=None,
            assessment=None,
            policy_digest=_digest(rules_raw),
            roots=[],
        )

    try:
        if isinstance(declared_config, list):
            declared_config = {"schema_version": 1, "roots": declared_config}
        if not isinstance(declared_config, Mapping):
            raise ValueError("model_assets must be a mapping or a roots list")
        if set(declared_config) - {"schema_version", "roots"}:
            unknown = ", ".join(sorted(set(declared_config) - {"schema_version", "roots"}))
            raise ValueError(f"Unknown model_assets field(s): {unknown}")
        if declared_config.get("schema_version", 1) != 1:
            raise ValueError("model_assets.schema_version must be 1")
        roots_spec = declared_config.get("roots", [])
        if not isinstance(roots_spec, list) or not roots_spec:
            raise ValueError("model_assets.roots must be a non-empty list")
        if any(not isinstance(item, Mapping) for item in roots_spec):
            raise ValueError("Every model_assets root must be a mapping")
        catalog: Dict[str, Any] = {}
        catalog_value = governance.get("catalog")
        if catalog_value:
            catalog_path = _safe_workspace_path(root, str(catalog_value))
            catalog = _read_catalog(catalog_path)
        models = []
        lock_roots = []
        slots = set()
        for root_spec in roots_spec:
            if set(root_spec) - {"slot", "ref", "phases", "role"}:
                unknown = ", ".join(sorted(set(root_spec) - {"slot", "ref", "phases", "role"}))
                raise ValueError(f"Unknown model_assets root field(s): {unknown}")
            slot = str(root_spec.get("slot", "")).strip()
            ref = root_spec.get("ref")
            if not slot:
                raise ValueError("Each model_assets root must declare a non-empty 'slot'")
            phases = root_spec.get("phases", [])
            if not isinstance(phases, list) or any(not isinstance(item, str) for item in phases):
                raise ValueError(f"model_assets root {slot!r} phases must be a list of strings")
            role = root_spec.get("role", slot)
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"model_assets root {slot!r} role must be a non-empty string")
            if slot in slots:
                raise ValueError(f"Duplicate model_assets slot: {slot}")
            slots.add(slot)
            definition = _resolve_definition(ref, store=store, root=root, catalog=catalog)
            digest = definition.compute_digest()
            models.append(definition)
            lock_roots.append(
                {
                    "slot": slot,
                    "role": role.strip(),
                    "ref": str(ref).strip(),
                    "phases": list(phases),
                    "model_id": definition.model_id,
                    "definition_digest": digest,
                    "artifact_bundle_digest": definition.artifacts.bundle_digest,
                    "definition": definition.to_dict(),
                    "parents": list(definition.parents),
                    "verification_refs": list(definition.verification_refs),
                }
            )
        rules = normalize_competition_rules(rules_raw)
        _validate_accounting_semantics(rules_raw)
        assessment = (
            evaluate_model_policy(models[0], rules)
            if len(models) == 1
            else evaluate_composition_policy(models, rules)
        )
        return _build_plan(
            task_id=task_id,
            mode=mode,
            required=required,
            declared=True,
            eligible=assessment.eligible,
            assessment=assessment,
            policy_digest=_digest(rules_raw),
            roots=lock_roots,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _failed_plan(
            task_id=task_id,
            mode=mode,
            required=required,
            declared=True,
            code=MODEL_PLAN_INVALID,
            message=str(exc),
            policy_digest=_digest(rules_raw),
            roots=[],
        )


def prepare_model_plan(
    task_spec: Any,
    effective_config: Mapping[str, Any],
    model_store: Optional[ModelStore] = None,
    project: Any = None,
) -> ModelPlan:
    """Compatibility name used in the Phase 2 implementation dossier."""
    return prepare_or_validate_model_plan(task_spec, effective_config, model_store, project)


def _validate_accounting_semantics(rules: Mapping[str, Any]) -> None:
    accounting = rules.get("parameter_accounting")
    if accounting is None:
        return
    if not isinstance(accounting, Mapping):
        raise ValueError("competition_rules.parameter_accounting must be a mapping")
    unsupported = set(accounting) - {"basis", "aggregation"}
    if unsupported:
        raise ValueError(
            "Unsupported parameter_accounting field(s): " + ", ".join(sorted(unsupported))
        )
    if accounting.get("basis", "logical_total") != "logical_total":
        raise ValueError("Only logical_total parameter accounting is currently supported")
    if accounting.get("aggregation", "sum_inference_members") != "sum_inference_members":
        raise ValueError("Only sum_inference_members composition accounting is currently supported")


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_plan(
    *,
    task_id: str,
    mode: str,
    required: bool,
    declared: bool,
    eligible: Optional[bool],
    assessment: Optional[PolicyAssessment],
    policy_digest: str,
    roots: list[Dict[str, Any]],
) -> ModelPlan:
    status = assessment.status if assessment else "unverified"
    failure_code = assessment.failure_code if assessment else None
    message = "; ".join(assessment.reasons) if assessment else "No model asset declaration; legacy execution is unverified."
    lock = {
        "schema_version": 1,
        "task_id": task_id,
        "governance_mode": mode,
        "roots": roots,
        "policy_digest": policy_digest,
    }
    lock_digest = _digest(lock)
    lock["model_lock_digest"] = lock_digest
    receipt = {
        "schema_version": 1,
        "task_id": task_id,
        "governance_mode": mode,
        "required": required,
        "declared": declared,
        "status": status,
        "eligible": eligible,
        "failure_code": failure_code,
        "reasons": list(assessment.reasons) if assessment else [message],
        "facts_evaluated": dict(assessment.facts_evaluated) if assessment else {},
        "policy_digest": policy_digest,
        "model_lock_digest": lock_digest,
    }
    return ModelPlan(
        task_id=task_id,
        governance_mode=mode,
        required=required,
        declared=declared,
        eligible=eligible,
        status=status,
        failure_code=failure_code,
        message=message,
        lock_json=json.dumps(lock, sort_keys=True, ensure_ascii=False),
        receipt_json=json.dumps(receipt, sort_keys=True, ensure_ascii=False),
        lock_digest=lock_digest,
    )


def enforce_model_admission(plan: ModelPlan) -> ModelPlan:
    """Raise before execution when required governance does not admit the plan."""
    if not isinstance(plan, ModelPlan):
        raise TypeError("plan must be a ModelPlan")
    if plan.required and plan.eligible is not True:
        raise ModelAdmissionError(
            plan.failure_code or MODEL_PARAMETER_COUNT_UNKNOWN,
            plan.message or "The required model plan is inconclusive.",
        )
    return plan

