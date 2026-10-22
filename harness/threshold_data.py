"""Read verified, selected binary OOF evidence for threshold recommendations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from harness.artifacts import validate_run
from harness.candidates import predictions
from harness.evaluation import _normalize_class_value
from harness.script_predictions import align_prediction_ids


def load_binary_threshold_data(project, task_spec, plugin, exp_id):
    """Return labels, declared-positive-class probabilities and source metadata.

    The last declared class is positive. Its probability column is selected by
    label, independently of the model's serialized class order. This function
    reads completed evidence and never changes it or applies a threshold.
    """
    if len(task_spec.subtasks) != 1:
        raise ValueError("Threshold optimization requires a single binary classification target")
    target, subtask = next(iter(task_spec.subtasks.items()))
    declared = [_normalize_class_value(value) for value in subtask.classes]
    if subtask.task_type != "binary_classification" or len(declared) != 2 or len(set(declared)) != 2:
        raise ValueError("Threshold optimization requires a binary target with two declared classes")
    if not exp_id or Path(exp_id).name != exp_id:
        raise ValueError("Threshold optimization requires an exact experiment ID")
    folder = project.artifacts_dir / exp_id
    validate_run(folder, task_spec=task_spec, plugin=plugin, expected={"exp_id": exp_id})
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(folder / "oof_predictions.csv")
    selection = metrics.get("selection_variant")
    if not selection or "input_variant" not in oof:
        raise ValueError("Threshold source must declare its selected input variant")
    selected = oof[oof.input_variant == selection]
    truth = pd.read_csv(folder / "evaluation_truth.csv")
    selected = align_prediction_ids(selected, truth[task_spec.id_column], task_spec.id_column, kind="OOF")
    column = subtask.column or target
    if column not in truth or truth[column].isna().any():
        raise ValueError("Threshold source is missing binary ground truth")
    labels = np.asarray([_normalize_class_value(value) for value in truth[column]])
    if not set(labels).issubset(declared):
        raise ValueError("Threshold truth labels do not match the declared binary classes")
    batch = predictions(selected)
    if "proba_" + target not in batch or "classes_" + target not in batch:
        raise ValueError("Threshold optimization requires probabilities and class order for " + target)
    order = [_normalize_class_value(value) for value in batch["classes_" + target]]
    probability = np.asarray(batch["proba_" + target], dtype=float)
    if len(order) != 2 or set(order) != set(declared) or probability.shape != (len(truth), 2):
        raise ValueError("Threshold probabilities must have one column per declared binary class")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Threshold probabilities must be finite and lie in [0, 1]")
    positive = declared[-1]
    return (labels == positive).astype(int), probability[:, order.index(positive)], {
        "target": target, "positive_class": subtask.classes[-1],
        "selection_variant": selection, "n_samples": len(truth),
    }
