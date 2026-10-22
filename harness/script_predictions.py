"""Prediction transport shared by script and pipeline adapters."""

import json


def extract_test_predictions(frame, task_spec):
    """Keep model class order alongside labels and probability matrices."""
    result = {}
    for target in task_spec.subtasks:
        column = f"pred_{target}" if f"pred_{target}" in frame else target
        if column in frame:
            result[f"pred_{target}"] = frame[column].values
        if f"proba_{target}" in frame:
            result[f"proba_{target}"] = frame[f"proba_{target}"].values
        class_column = f"classes_{target}"
        if class_column in frame:
            if frame[class_column].isna().any() or frame[class_column].nunique() != 1:
                raise ValueError("Class order changed within test predictions")
            value = frame[class_column].iloc[0]
            result[class_column] = json.loads(value) if isinstance(value, str) else value
    return result
