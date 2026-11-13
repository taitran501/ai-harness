"""Prediction transport shared by script and pipeline adapters."""

import json

from harness.evaluation import align_prediction_frame, validate_prediction_batch, PredictionContractError


def align_prediction_ids(frame, expected_ids, id_column, *, kind="Test"):
    """Require an exact one-to-one ID set and return rows in dataset order."""
    return align_prediction_frame(frame, expected_ids=expected_ids, id_column=id_column, purpose=kind)


def extract_test_predictions(frame, task_spec, *, expected_ids=None):
    """Keep model class order alongside labels and probability matrices."""
    if task_spec.id_column not in frame:
        raise PredictionContractError("Test predictions require explicit IDs")
    if expected_ids is not None:
        frame = align_prediction_ids(frame, expected_ids, task_spec.id_column)
    result = {"ids": frame[task_spec.id_column].tolist()}
    for target in task_spec.subtasks:
        column = f"pred_{target}" if f"pred_{target}" in frame else target
        if column in frame:
            result[f"pred_{target}"] = [json.loads(v) if isinstance(v, str) and v.startswith("[") else v for v in frame[column]]
        if f"proba_{target}" in frame:
            result[f"proba_{target}"] = [json.loads(v) if isinstance(v, str) else v for v in frame[f"proba_{target}"]]
        class_column = f"classes_{target}"
        if class_column in frame:
            if frame[class_column].isna().any() or frame[class_column].nunique() != 1:
                raise PredictionContractError("Class order changed within test predictions")
            value = frame[class_column].iloc[0]
            result[class_column] = json.loads(value) if isinstance(value, str) else value
    validate_prediction_batch(task_spec, result, expected_ids=frame[task_spec.id_column], purpose="test")
    return result
