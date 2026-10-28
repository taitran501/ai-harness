"""Prediction provenance classification and submission eligibility helpers."""
from __future__ import annotations

from enum import Enum
from pathlib import Path


class PredictionProvenance(str, Enum):
    LOCAL_RUN = "local_run"
    REPRODUCED_EXTERNAL_CODE = "reproduced_external_code"
    EXTERNAL_PREDICTION = "external_prediction"


_ALLOWED = {PredictionProvenance.LOCAL_RUN.value, PredictionProvenance.REPRODUCED_EXTERNAL_CODE.value}


def validate_provenance_record(record: dict) -> None:
    if not isinstance(record, dict) or record.get("provenance") not in {item.value for item in PredictionProvenance}:
        raise ValueError("PROVENANCE_MISSING_OR_INVALID")
    provenance = record["provenance"]
    if provenance == PredictionProvenance.REPRODUCED_EXTERNAL_CODE.value:
        details = record.get("provenance_record") or {}
        if not details.get("source_snapshot_sha256") or not details.get("execution_receipt_sha256"):
            raise ValueError("EXTERNAL_CODE_EXECUTION_EVIDENCE_REQUIRED")


def resolve_run_provenance(project, exp_id: str, *, require_execution_receipt: bool = True) -> dict:
    from harness.artifacts import validate_run

    folder = Path(project.artifacts_dir) / exp_id
    manifest = validate_run(folder, expected={"exp_id": exp_id})
    validate_provenance_record(manifest)
    if require_execution_receipt and manifest["provenance"] == PredictionProvenance.REPRODUCED_EXTERNAL_CODE.value:
        receipt = manifest.get("provenance_record", {}).get("execution_receipt_id")
        receipt_path = Path(project.root) / ".ai-harness" / "execution_receipts" / f"{receipt}.json"
        if not receipt or not receipt_path.is_file():
            raise ValueError("EXECUTION_RECEIPT_MISSING")
    return manifest


def require_allowed_provenance(records: list[dict], *, purpose: str) -> None:
    for record in records:
        validate_provenance_record(record)
        if record["provenance"] not in _ALLOWED:
            raise ValueError(f"EXTERNAL_PREDICTION_NOT_SUBMITTABLE: External prediction artifacts cannot be used for {purpose}")


def derive_submission_eligibility(*, provenance_valid: bool, validation_valid: bool,
                                  test_predictions_valid: bool, promotion_valid: bool) -> dict:
    checks = {"provenance_valid": provenance_valid, "validation_valid": validation_valid,
              "test_predictions_valid": test_predictions_valid, "promotion_valid": promotion_valid}
    reasons = [name.upper() for name, passed in checks.items() if not passed]
    eligible = not reasons
    return {"submission_eligible": eligible, "eligible": eligible, "eligibility_reason_codes": reasons}
