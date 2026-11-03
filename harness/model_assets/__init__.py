"""Native model asset definitions, policy evaluation, and bounded storage."""

from harness.model_assets.policy import (
    evaluate_composition_policy,
    evaluate_model_policy,
    normalize_competition_rules,
)
from harness.model_assets.store import MAX_MODEL_DEFINITION_BYTES, ModelStore
from harness.model_assets.types import (
    MODEL_API_PROHIBITED,
    MODEL_LICENSE_UNRESOLVED,
    MODEL_PARAMETER_COUNT_UNKNOWN,
    MODEL_PARAMETER_LIMIT_EXCEEDED,
    MODEL_PLAN_INVALID,
    MODEL_GOVERNANCE_MODE_REQUIRED,
    MODEL_PRETRAINED_PROHIBITED,
    MODEL_REFERENCE_MISSING,
    MODEL_RELEASE_CUTOFF_VIOLATION,
    MODEL_RELEASE_EVIDENCE_MISSING,
    MODEL_REVISION_UNPINNED,
    ArtifactsSpec,
    CompetitionRules,
    LicenseFacts,
    ModelDefinition,
    ModelKind,
    ParameterBasis,
    ParameterFacts,
    PolicyAssessment,
    ReleaseFacts,
    SourceSpec,
)

__all__ = [
    "ArtifactsSpec",
    "CompetitionRules",
    "LicenseFacts",
    "MAX_MODEL_DEFINITION_BYTES",
    "MODEL_API_PROHIBITED",
    "MODEL_LICENSE_UNRESOLVED",
    "MODEL_PARAMETER_COUNT_UNKNOWN",
    "MODEL_PARAMETER_LIMIT_EXCEEDED",
    "MODEL_PLAN_INVALID",
    "MODEL_GOVERNANCE_MODE_REQUIRED",
    "MODEL_PRETRAINED_PROHIBITED",
    "MODEL_REFERENCE_MISSING",
    "MODEL_RELEASE_CUTOFF_VIOLATION",
    "MODEL_RELEASE_EVIDENCE_MISSING",
    "MODEL_REVISION_UNPINNED",
    "ModelDefinition",
    "ModelKind",
    "ModelStore",
    "ParameterBasis",
    "ParameterFacts",
    "PolicyAssessment",
    "ReleaseFacts",
    "SourceSpec",
    "evaluate_composition_policy",
    "evaluate_model_policy",
    "normalize_competition_rules",
]

from harness.model_assets.resolution import (
    ModelAdmissionError,
    ModelPlan,
    enforce_model_admission,
    prepare_model_plan,
    prepare_or_validate_model_plan,
    resolve_effective_config,
)

__all__ += [
    "ModelAdmissionError",
    "ModelPlan",
    "enforce_model_admission",
    "prepare_model_plan",
    "prepare_or_validate_model_plan",
    "resolve_effective_config",
]

from harness.model_assets.resolver import BackendResourcePlan, ModelResolver

__all__ += ["BackendResourcePlan", "ModelResolver"]
