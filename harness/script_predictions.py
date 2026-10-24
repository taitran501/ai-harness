"""Prediction transport shared by script and pipeline adapters."""

import json

import pandas as pd


def align_prediction_ids(frame, expected_ids, id_column, *, kind="Test"):
    """Require an exact one-to-one ID set and return rows in dataset order."""
    expected = pd.Series(list(expected_ids), name=id_column)
    if expected.isna().any() or expected.duplicated().any():
        raise ValueError(f"{kind} dataset requires unique, non-null IDs")
    if id_column not in frame or frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"{kind} predictions require unique, non-null IDs")
    if set(frame[id_column]) != set(expected):
        raise ValueError(f"{kind} prediction IDs do not match the declared dataset")
    return expected.to_frame().merge(frame, on=id_column, how="left", validate="one_to_one")


def extract_test_predictions(frame, task_spec, *, expected_ids=None):
    """Keep model class order alongside labels and probability matrices."""
    if expected_ids is not None:
        frame = align_prediction_ids(frame, expected_ids, task_spec.id_column)
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
