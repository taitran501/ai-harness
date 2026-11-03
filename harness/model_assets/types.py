"""Typed, content-addressed records for governed model assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional


class ModelKind(str, Enum):
    """Kinds of model components understood by the asset registry."""

    PRETRAINED = "pretrained"
    CHECKPOINT = "checkpoint"
    ADAPTER = "adapter"
    COMPOSITE = "composite"
    SERVICE = "service"


class ParameterBasis(str, Enum):
    """Meaning of a parameter count recorded in a model definition."""

    LOGICAL_TOTAL = "logical_total"
    TRAINABLE = "trainable"
    ACTIVE = "active"


MODEL_REFERENCE_MISSING = "MODEL_REFERENCE_MISSING"
MODEL_REVISION_UNPINNED = "MODEL_REVISION_UNPINNED"
MODEL_PARAMETER_COUNT_UNKNOWN = "MODEL_PARAMETER_COUNT_UNKNOWN"
MODEL_PARAMETER_LIMIT_EXCEEDED = "MODEL_PARAMETER_LIMIT_EXCEEDED"
MODEL_RELEASE_EVIDENCE_MISSING = "MODEL_RELEASE_EVIDENCE_MISSING"
MODEL_RELEASE_CUTOFF_VIOLATION = "MODEL_RELEASE_CUTOFF_VIOLATION"
MODEL_LICENSE_UNRESOLVED = "MODEL_LICENSE_UNRESOLVED"
MODEL_API_PROHIBITED = "MODEL_API_PROHIBITED"
MODEL_PRETRAINED_PROHIBITED = "MODEL_PRETRAINED_PROHIBITED"


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return value


def _check_keys(
    value: Mapping[str, Any], label: str, allowed: set[str], required: set[str] | None = None
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown field(s) in {label}: {', '.join(sorted(unknown))}")
    missing = (required or set()) - set(value)
    if missing:
        raise ValueError(f"Missing required field(s) in {label}: {', '.join(sorted(missing))}")


def _string(value: Any, label: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        suffix = " or null" if optional else ""
        raise ValueError(f"{label} must be a string{suffix}")
    return value.strip()


def _string_list(value: Any, label: str) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return [item.strip() for item in value]


def _optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer or null")
    return value


def _enum_value(value: Any, enum_type: type[Enum], label: str) -> str:
    if isinstance(value, enum_type):
        return str(value.value)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip().lower()
    choices = {str(item.value) for item in enum_type}
    if normalized not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return normalized


def _date_string(value: Any, label: str) -> Optional[str]:
    parsed = _string(value, label, optional=True)
    if parsed is None or parsed == "":
        return parsed
    try:
        date.fromisoformat(parsed)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format") from exc
    if len(parsed) != 10:
        raise ValueError(f"{label} must use YYYY-MM-DD format")
    return parsed


@dataclass
class ParameterFacts:
    """Parameter counts and evidence for one declared model component."""

    exact_param_count: Optional[int]
    count_basis: ParameterBasis | str
    active_param_count: Optional[int] = None
    trainable_param_count: Optional[int] = None
    evidence_refs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> "ParameterFacts":
        value = _as_mapping(raw, "parameter_facts")
        _check_keys(
            value,
            "parameter_facts",
            {
                "exact_param_count",
                "count_basis",
                "active_param_count",
                "trainable_param_count",
                "evidence_refs",
            },
            {"count_basis"},
        )
        return cls(
            exact_param_count=_optional_int(value.get("exact_param_count"), "exact_param_count"),
            count_basis=_enum_value(value["count_basis"], ParameterBasis, "count_basis"),
            active_param_count=_optional_int(value.get("active_param_count"), "active_param_count"),
            trainable_param_count=_optional_int(
                value.get("trainable_param_count"), "trainable_param_count"
            ),
            evidence_refs=_string_list(value.get("evidence_refs", []), "evidence_refs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_param_count": self.exact_param_count,
            "count_basis": _enum_value(self.count_basis, ParameterBasis, "count_basis"),
            "active_param_count": self.active_param_count,
            "trainable_param_count": self.trainable_param_count,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class SourceSpec:
    provider: str
    repository: str
    revision: str
    source_uri: str

    @classmethod
    def from_dict(cls, raw: Any) -> "SourceSpec":
        value = _as_mapping(raw, "source")
        fields = {"provider", "repository", "revision", "source_uri"}
        _check_keys(value, "source", fields, fields)
        return cls(**{key: _string(value[key], f"source.{key}") for key in fields})

    def to_dict(self) -> Dict[str, str]:
        return {
            "provider": self.provider.strip(),
            "repository": self.repository.strip(),
            "revision": self.revision.strip(),
            "source_uri": self.source_uri.strip(),
        }


@dataclass
class ArtifactsSpec:
    manifest_ref: Optional[str] = None
    bundle_digest: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Any) -> "ArtifactsSpec":
        value = _as_mapping(raw, "artifacts")
        _check_keys(value, "artifacts", {"manifest_ref", "bundle_digest"})
        return cls(
            manifest_ref=_string(value.get("manifest_ref"), "artifacts.manifest_ref", optional=True),
            bundle_digest=_string(
                value.get("bundle_digest"), "artifacts.bundle_digest", optional=True
            ),
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "manifest_ref": self.manifest_ref.strip() if self.manifest_ref else None,
            "bundle_digest": self.bundle_digest.strip() if self.bundle_digest else None,
        }


@dataclass
class ReleaseFacts:
    public_release_date: Optional[str]
    evidence_refs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> "ReleaseFacts":
        value = _as_mapping(raw, "release")
        _check_keys(value, "release", {"public_release_date", "evidence_refs"})
        return cls(
            public_release_date=_date_string(
                value.get("public_release_date"), "release.public_release_date"
            ),
            evidence_refs=_string_list(value.get("evidence_refs", []), "release.evidence_refs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "public_release_date": self.public_release_date.strip()
            if self.public_release_date
            else self.public_release_date,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class LicenseFacts:
    expression: Optional[str]
    evidence_refs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> "LicenseFacts":
        value = _as_mapping(raw, "license")
        _check_keys(value, "license", {"expression", "evidence_refs"})
        return cls(
            expression=_string(value.get("expression"), "license.expression", optional=True),
            evidence_refs=_string_list(value.get("evidence_refs", []), "license.evidence_refs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression.strip() if self.expression else self.expression,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class ModelDefinition:
    model_id: str
    kind: ModelKind | str
    family: str
    task_types: List[str]
    modalities: List[str]
    source: SourceSpec
    artifacts: ArtifactsSpec
    parameter_facts: ParameterFacts
    release: ReleaseFacts
    license: LicenseFacts
    parents: List[str] = field(default_factory=list)
    verification_refs: List[str] = field(default_factory=list)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, raw: Any) -> "ModelDefinition":
        value = _as_mapping(raw, "model definition")
        fields = {
            "schema_version",
            "model_id",
            "kind",
            "family",
            "task_types",
            "modalities",
            "source",
            "artifacts",
            "parameter_facts",
            "release",
            "license",
            "parents",
            "verification_refs",
        }
        required = {
            "model_id",
            "kind",
            "family",
            "task_types",
            "modalities",
            "source",
            "artifacts",
            "parameter_facts",
            "release",
            "license",
        }
        _check_keys(value, "model definition", fields, required)
        version = value.get("schema_version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError("schema_version must be integer 1")
        model_id = _string(value["model_id"], "model_id")
        family = _string(value["family"], "family")
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not family:
            raise ValueError("family must not be empty")
        return cls(
            model_id=model_id,
            kind=_enum_value(value["kind"], ModelKind, "kind"),
            family=family,
            task_types=_string_list(value["task_types"], "task_types"),
            modalities=_string_list(value["modalities"], "modalities"),
            source=SourceSpec.from_dict(value["source"]),
            artifacts=ArtifactsSpec.from_dict(value["artifacts"]),
            parameter_facts=ParameterFacts.from_dict(value["parameter_facts"]),
            release=ReleaseFacts.from_dict(value["release"]),
            license=LicenseFacts.from_dict(value["license"]),
            parents=_string_list(value.get("parents", []), "parents"),
            verification_refs=_string_list(
                value.get("verification_refs", []), "verification_refs"
            ),
            schema_version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id.strip(),
            "kind": _enum_value(self.kind, ModelKind, "kind"),
            "family": self.family.strip(),
            "task_types": [item.strip() for item in self.task_types],
            "modalities": [item.strip() for item in self.modalities],
            "source": self.source.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "parameter_facts": self.parameter_facts.to_dict(),
            "release": self.release.to_dict(),
            "license": self.license.to_dict(),
            "parents": [item.strip() for item in self.parents],
            "verification_refs": [item.strip() for item in self.verification_refs],
        }

    def compute_digest(self) -> str:
        """Return SHA-256 of the normalized, key-sorted JSON definition."""
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass
class CompetitionRules:
    max_total_parameters: Optional[int] = None
    max_llm_parameters: Optional[int] = None
    model_weight_release_cutoff: Optional[str] = None
    allow_pretrained_models: bool = True
    allow_api: bool = False
    allow_external_data: bool = False
    allowed_licenses: Optional[List[str]] = None
    weight_cutoff_inclusive: bool = True


@dataclass
class PolicyAssessment:
    eligible: bool
    status: str
    failure_code: Optional[str]
    reasons: List[str] = field(default_factory=list)
    facts_evaluated: Dict[str, Any] = field(default_factory=dict)
