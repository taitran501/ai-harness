"""Pure normalization and evaluation of model governance rules."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Mapping, Optional

from harness.model_assets.types import (
    MODEL_API_PROHIBITED,
    MODEL_LICENSE_UNRESOLVED,
    MODEL_PARAMETER_COUNT_UNKNOWN,
    MODEL_PARAMETER_LIMIT_EXCEEDED,
    MODEL_PRETRAINED_PROHIBITED,
    MODEL_REFERENCE_MISSING,
    MODEL_RELEASE_CUTOFF_VIOLATION,
    MODEL_RELEASE_EVIDENCE_MISSING,
    MODEL_REVISION_UNPINNED,
    CompetitionRules,
    ModelDefinition,
    PolicyAssessment,
)


_MUTABLE_REVISIONS = {"default", "dev", "head", "latest", "main", "master"}


def _optional_limit(raw: Mapping[str, Any], key: str) -> Optional[int]:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer or null")
    return value


def _optional_date(raw: Mapping[str, Any], key: str) -> Optional[str]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO date string or null")
    normalized = value.strip()
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{key} must use YYYY-MM-DD format") from exc
    if len(normalized) != 10:
        raise ValueError(f"{key} must use YYYY-MM-DD format")
    return normalized


def _boolean(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def normalize_competition_rules(raw: Dict[str, Any]) -> CompetitionRules:
    """Parse supported rule fields without modifying the caller's mapping."""
    if not isinstance(raw, dict):
        raise ValueError("competition rules must be a dictionary")

    allowed_licenses = raw.get("allowed_licenses")
    if allowed_licenses is not None:
        if not isinstance(allowed_licenses, list) or any(
            not isinstance(item, str) for item in allowed_licenses
        ):
            raise ValueError("allowed_licenses must be a list of strings or null")
        allowed_licenses = [item.strip() for item in allowed_licenses]

    return CompetitionRules(
        max_total_parameters=_optional_limit(raw, "max_total_parameters"),
        max_llm_parameters=_optional_limit(raw, "max_llm_parameters"),
        model_weight_release_cutoff=_optional_date(raw, "model_weight_release_cutoff"),
        allow_pretrained_models=_boolean(raw, "allow_pretrained_models", True),
        allow_api=_boolean(raw, "allow_api", False),
        allow_external_data=_boolean(raw, "allow_external_data", False),
        allowed_licenses=allowed_licenses,
        weight_cutoff_inclusive=_boolean(raw, "weight_cutoff_inclusive", True),
    )


def _model_kind(model_def: ModelDefinition) -> str:
    value = model_def.kind.value if hasattr(model_def.kind, "value") else model_def.kind
    return str(value).strip().lower()


def _is_llm(model_def: ModelDefinition) -> bool:
    return any(str(modality).strip().lower() == "text" for modality in model_def.modalities) or (
        "llm" in model_def.family.strip().lower()
    )


def _assessment(
    *,
    failure_code: str,
    reason: str,
    facts: Dict[str, Any],
) -> PolicyAssessment:
    return PolicyAssessment(
        eligible=False,
        status="blocked",
        failure_code=failure_code,
        reasons=[reason],
        facts_evaluated=facts,
    )


def evaluate_model_policy(
    model_def: ModelDefinition, rules: CompetitionRules
) -> PolicyAssessment:
    """Evaluate declared model facts against one normalized competition policy."""
    kind = _model_kind(model_def)
    source = model_def.source
    source_reference = bool(
        source.repository.strip()
        or source.source_uri.strip()
        or model_def.artifacts.manifest_ref
        or model_def.artifacts.bundle_digest
    )
    exact_count = model_def.parameter_facts.exact_param_count
    facts: Dict[str, Any] = {
        "model_id": model_def.model_id,
        "kind": kind,
        "source_reference_present": source_reference,
        "source_revision": source.revision.strip(),
        "exact_param_count": exact_count,
        "count_basis": getattr(model_def.parameter_facts.count_basis, "value", model_def.parameter_facts.count_basis),
        "max_total_parameters": rules.max_total_parameters,
        "max_llm_parameters": rules.max_llm_parameters,
    }

    if kind == "pretrained" and not rules.allow_pretrained_models:
        return _assessment(
            failure_code=MODEL_PRETRAINED_PROHIBITED,
            reason="The competition rules prohibit pretrained models.",
            facts=facts,
        )

    if kind == "service" and not rules.allow_api:
        return _assessment(
            failure_code=MODEL_API_PROHIBITED,
            reason="The competition rules prohibit API or hosted model services.",
            facts=facts,
        )

    if not source_reference:
        return _assessment(
            failure_code=MODEL_REFERENCE_MISSING,
            reason="The model has no source URI, repository, or artifact reference.",
            facts=facts,
        )

    if (
        source.provider.strip().lower() not in {"local", "workspace"}
        and kind not in {"service", "composite"}
        and source.revision.strip().lower() in {"", *_MUTABLE_REVISIONS}
    ):
        return _assessment(
            failure_code=MODEL_REVISION_UNPINNED,
            reason="The external model source must use an immutable revision.",
            facts=facts,
        )

    needs_total_count = rules.max_total_parameters is not None
    needs_llm_count = rules.max_llm_parameters is not None and _is_llm(model_def)
    if needs_total_count or needs_llm_count:
        if exact_count is None or exact_count <= 0:
            return _assessment(
                failure_code=MODEL_PARAMETER_COUNT_UNKNOWN,
                reason="The exact logical parameter count is unknown; unknown is not zero.",
                facts=facts,
            )
        if needs_total_count and exact_count > rules.max_total_parameters:
            return _assessment(
                failure_code=MODEL_PARAMETER_LIMIT_EXCEEDED,
                reason=(
                    f"The model has {exact_count} parameters, exceeding the total limit "
                    f"of {rules.max_total_parameters}."
                ),
                facts=facts,
            )
        if needs_llm_count and exact_count > rules.max_llm_parameters:
            return _assessment(
                failure_code=MODEL_PARAMETER_LIMIT_EXCEEDED,
                reason=(
                    f"The LLM has {exact_count} parameters, exceeding the LLM limit "
                    f"of {rules.max_llm_parameters}."
                ),
                facts=facts,
            )

    cutoff = rules.model_weight_release_cutoff
    if cutoff is not None and kind == "pretrained":
        release_date = model_def.release.public_release_date
        if not release_date:
            return _assessment(
                failure_code=MODEL_RELEASE_EVIDENCE_MISSING,
                reason="The pretrained model has no public weight release date.",
                facts=facts,
            )
        try:
            released = date.fromisoformat(release_date)
            cutoff_date = date.fromisoformat(cutoff)
        except ValueError:
            return _assessment(
                failure_code=MODEL_RELEASE_EVIDENCE_MISSING,
                reason="The model release date or policy cutoff is not a valid ISO date.",
                facts=facts,
            )
        after_cutoff = released > cutoff_date if rules.weight_cutoff_inclusive else released >= cutoff_date
        facts["public_release_date"] = released.isoformat()
        facts["model_weight_release_cutoff"] = cutoff_date.isoformat()
        if after_cutoff:
            return _assessment(
                failure_code=MODEL_RELEASE_CUTOFF_VIOLATION,
                reason=(
                    f"The model weights were released on {released.isoformat()}, outside "
                    f"the allowed cutoff of {cutoff_date.isoformat()}."
                ),
                facts=facts,
            )

    if rules.allowed_licenses is not None:
        expression = (model_def.license.expression or "").strip()
        allowed = {item.strip().lower() for item in rules.allowed_licenses}
        if not expression or expression.lower() not in allowed:
            return _assessment(
                failure_code=MODEL_LICENSE_UNRESOLVED,
                reason="The model license is missing or is not in the allowed license list.",
                facts=facts,
            )

    return PolicyAssessment(
        eligible=True,
        status="eligible",
        failure_code=None,
        reasons=["Declared model facts satisfy the applicable competition rules."],
        facts_evaluated=facts,
    )


def evaluate_composition_policy(
    models: List[ModelDefinition], rules: CompetitionRules
) -> PolicyAssessment:
    """Evaluate total deployed-component parameters for a composite system."""
    component_facts = [
        {
            "model_id": model.model_id,
            "exact_param_count": model.parameter_facts.exact_param_count,
            "is_llm": _is_llm(model),
        }
        for model in models
    ]
    known_total = sum(
        model.parameter_facts.exact_param_count
        for model in models
        if model.parameter_facts.exact_param_count is not None
        and model.parameter_facts.exact_param_count > 0
    )
    unknown_total = any(
        model.parameter_facts.exact_param_count is None
        or model.parameter_facts.exact_param_count <= 0
        for model in models
    )
    llm_models = [model for model in models if _is_llm(model)]
    known_llm_total = sum(
        model.parameter_facts.exact_param_count
        for model in llm_models
        if model.parameter_facts.exact_param_count is not None
        and model.parameter_facts.exact_param_count > 0
    )
    unknown_llm_total = any(
        model.parameter_facts.exact_param_count is None
        or model.parameter_facts.exact_param_count <= 0
        for model in llm_models
    )
    facts: Dict[str, Any] = {
        "components": component_facts,
        "total_parameters": None if unknown_total else known_total,
        "llm_parameters": None if unknown_llm_total else known_llm_total,
        "max_total_parameters": rules.max_total_parameters,
        "max_llm_parameters": rules.max_llm_parameters,
    }

    if rules.max_total_parameters is not None:
        if unknown_total:
            return _assessment(
                failure_code=MODEL_PARAMETER_COUNT_UNKNOWN,
                reason="At least one deployed component has an unknown parameter count.",
                facts=facts,
            )
        if known_total > rules.max_total_parameters:
            return _assessment(
                failure_code=MODEL_PARAMETER_LIMIT_EXCEEDED,
                reason=(
                    f"The deployed composition has {known_total} parameters, exceeding "
                    f"the total limit of {rules.max_total_parameters}."
                ),
                facts=facts,
            )

    if rules.max_llm_parameters is not None:
        if unknown_llm_total:
            return _assessment(
                failure_code=MODEL_PARAMETER_COUNT_UNKNOWN,
                reason="At least one deployed LLM component has an unknown parameter count.",
                facts=facts,
            )
        if known_llm_total > rules.max_llm_parameters:
            return _assessment(
                failure_code=MODEL_PARAMETER_LIMIT_EXCEEDED,
                reason=(
                    f"The deployed LLM components have {known_llm_total} parameters, "
                    f"exceeding the LLM limit of {rules.max_llm_parameters}."
                ),
                facts=facts,
            )

    return PolicyAssessment(
        eligible=True,
        status="eligible",
        failure_code=None,
        reasons=["Deployed component parameter totals satisfy the applicable limits."],
        facts_evaluated=facts,
    )
