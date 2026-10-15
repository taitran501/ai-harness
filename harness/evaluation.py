"""Strict prediction and metric contracts shared by runners and consumers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import re
import numpy as np
import pandas as pd
from sklearn import metrics


def metric_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", value.lower())
    aliases = {
        "macrof1": "macro_f1",
        "f1macro": "macro_f1",
        "microf1": "micro_f1",
        "f1micro": "micro_f1",
        "rocauc": "roc_auc",
        "aucroc": "roc_auc",
        "auc": "roc_auc",
        "logloss": "logloss",
        "accuracy": "accuracy",
        "rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
        "rsquared": "r2",
        "acc": "accuracy",
        "rootmeansquarederror": "rmse",
        "meanabsoluteerror": "mae",
        "qwk": "qwk",
        "quadraticweightedkappa": "qwk",
        "cohenkappa": "qwk",
        "cohenkappascore": "qwk",
        "kappa": "qwk",
        "ndcg": "ndcg",
        "ndcgscore": "ndcg",
        "normalizeddiscountedcumulativegain": "ndcg",
        "spearman": "spearman",
        "spearmanr": "spearman",
        "spearmanrho": "spearman",
        "spearmancorrelation": "spearman",
        "spearmanrankcorrelation": "spearman",
        "mape": "mape",
        "meanabsolutepercentageerror": "mape",
    }
    return aliases.get(name, value)


def utility(score: float, direction: str) -> float:
    if direction not in {"maximize", "minimize"} or not np.isfinite(score):
        raise ValueError("A finite score and maximize/minimize direction are required")
    return float(score) if direction == "maximize" else -float(score)


@dataclass
class PredictionBatch:
    ids: list[Any]
    labels: dict[str, Any] = field(default_factory=dict)
    probabilities: dict[str, Any] = field(default_factory=dict)
    class_order: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, predictions: Any, ids: Any) -> "PredictionBatch":
        if isinstance(predictions, cls):
            if list(predictions.ids) != list(ids):
                raise ValueError("Prediction IDs do not match evaluation IDs")
            return predictions
        if not isinstance(predictions, dict):
            raise TypeError("Predictions must be a mapping or PredictionBatch")
        if "labels" in predictions and isinstance(predictions["labels"], dict):
            if "ids" in predictions and list(predictions["ids"]) != list(ids):
                raise ValueError("Prediction IDs mismatch")
            return cls(
                list(ids),
                predictions["labels"],
                predictions.get("probabilities", {}),
                predictions.get("class_order", {}),
            )
        return cls(
            list(ids),
            {
                k: v
                for k, v in predictions.items()
                if not k.startswith(("proba_", "classes_"))
            },
            {k[6:]: v for k, v in predictions.items() if k.startswith("proba_")},
            {
                k[8:]: list(v)
                for k, v in predictions.items()
                if k.startswith("classes_")
            },
        )


@dataclass
class EvaluationResult:
    primary_metric: str
    primary_score: float
    direction: str
    utility: float
    subtasks: dict[str, Any]
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {**self.subtasks, **asdict(self), "composite_score": self.primary_score}


def resolve_evaluation(spec: Any, result: dict, n_samples: int) -> dict:
    """Normalize plugin output without guessing a different optimization metric."""
    if not isinstance(result, dict):
        raise TypeError("Scorer must return a metric mapping")
    name = metric_name(spec.primary_metric)
    score = result.get("primary_score")
    if score is None:
        score = next(
            (
                v
                for k, v in result.items()
                if metric_name(k) == name and isinstance(v, (int, float))
            ),
            None,
        )
    if score is None and len(spec.subtasks) == 1:
        nested = result.get(next(iter(spec.subtasks)), {})
        score = nested.get(name) if isinstance(nested, dict) else None
    if score is None:
        score = result.get("composite_score")
    if score is None:
        raise ValueError(f"Scorer omitted primary metric {name}")
    return {
        **result,
        **EvaluationResult(
            name,
            float(score),
            spec.direction,
            utility(float(score), spec.direction),
            result.get(
                "subtasks", {k: v for k, v in result.items() if isinstance(v, dict)}
            ),
            n_samples,
        ).to_dict(),
    }

def _normalize_class_value(val: Any) -> str:
    """Normalize a class label so integer-like floats (e.g. 0.0, 1.0) match integer classes."""
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "nan"
        if val.is_integer():
            return str(int(val))
        return str(val)
    s = str(val).strip()
    if re.match(r"^-?\d+\.0+$", s):
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    return s


def _normalize_class_array(arr: Any) -> np.ndarray:
    """Normalize array of class labels element-wise into strings."""
    flat = np.asarray(arr)
    if np.issubdtype(flat.dtype, np.integer):
        return flat.astype(str)
    normalized = [_normalize_class_value(x) for x in flat.ravel()]
    return np.array(normalized, dtype=str).reshape(flat.shape)


def evaluate(spec: Any, truth: pd.DataFrame, predictions: Any) -> dict:
    if truth.empty or not spec.subtasks:
        raise ValueError("Evaluation requires rows and declared targets")
    if (
        spec.id_column not in truth
        or truth[spec.id_column].isna().any()
        or truth[spec.id_column].duplicated().any()
    ):
        raise ValueError("Evaluation IDs must be present, non-null and unique")
    batch = PredictionBatch.from_mapping(predictions, truth[spec.id_column])
    active = [s for s in spec.subtasks.values() if s.weight > 0]
    if not active:
        raise ValueError("At least one positive target weight is required")
    if len({(metric_name(s.primary_metric), s.direction) for s in active}) > 1:
        raise ValueError("Mixed metrics/directions require a custom scorer")
    if any(not np.isfinite(s.weight) or s.weight < 0 for s in spec.subtasks.values()):
        raise ValueError("Invalid target weights")
    if (metric_name(active[0].primary_metric), active[0].direction) != (
        metric_name(spec.primary_metric),
        spec.direction,
    ):
        raise ValueError("Primary metric/direction must match the generic aggregate")
    masks = spec.compute_evaluation_masks(truth)
    results, weighted, total = {}, 0.0, 0.0
    for key, st in spec.subtasks.items():
        column = st.column or key
        candidates = [key, column, f"pred_{column}", f"pred_{key}"]
        label = next((batch.labels[k] for k in candidates if k in batch.labels), None)
        if label is None or len(label) != len(truth):
            raise ValueError(f"Missing or incomplete predictions for {key}")
        mask = masks[key]
        if not mask.any():
            if st.weight > 0:
                raise ValueError(f"No evaluation support for {key}")
            continue
        if st.column_prefix:
            cols = [st.column_prefix + str(x) for x in st.labels]
            if not all(c in truth for c in cols):
                raise ValueError(f"Missing targets for {key}")
            y = truth[cols].to_numpy()[mask]
        else:
            if column not in truth or truth[column].isna().any():
                raise ValueError(f"Missing targets for {key}")
            y = truth[column].to_numpy()[mask]
        p = np.asarray(label)[mask]
        name = metric_name(st.primary_metric)
        if st.task_type in {"regression", "ranking"}:
            y, p = np.asarray(y, float), np.asarray(p, float)
            if not np.isfinite(y).all() or not np.isfinite(p).all():
                raise ValueError("Regression values must be finite")
            available = {
                "rmse": float(np.sqrt(metrics.mean_squared_error(y, p))),
                "mae": float(metrics.mean_absolute_error(y, p)),
                "r2": float(metrics.r2_score(y, p)) if len(y) > 1 else 0.0,
            }
            try:
                available["mape"] = float(metrics.mean_absolute_percentage_error(y, p))
            except Exception:
                available["mape"] = 0.0
            try:
                import scipy.stats as stats

                s_res = stats.spearmanr(y, p)
                s_val = getattr(s_res, "statistic", s_res[0])
                available["spearman"] = 0.0 if np.isnan(s_val) else float(s_val)
            except Exception:
                available["spearman"] = 0.0
            try:
                y_nonneg = y - np.min(y) if len(y) and np.min(y) < 0 else y
                if len(y) > 1:
                    available["ndcg"] = float(
                        metrics.ndcg_score(y_nonneg[None, :], p[None, :])
                    )
                else:
                    available["ndcg"] = 0.0
            except Exception:
                available["ndcg"] = 0.0
            try:
                val = float(
                    metrics.cohen_kappa_score(
                        np.rint(y).astype(int),
                        np.rint(p).astype(int),
                        weights="quadratic",
                    )
                )
                available["qwk"] = 0.0 if np.isnan(val) else val
            except Exception:
                available["qwk"] = 0.0
        elif st.task_type in {
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        }:
            if st.task_type != "multilabel_classification":
                y = _normalize_class_array(y)
                p = _normalize_class_array(p)
                classes = [_normalize_class_value(c) for c in (st.classes or [])] or sorted(set(y))
                if not set(y).issubset(classes):
                    raise ValueError(f"Ground truth outside declared classes for {key}: found {set(y) - set(classes)} not in {classes}")
                if not set(p).issubset(classes):
                    raise ValueError(f"Invalid predicted class for {key}: found {set(p) - set(classes)} not in {classes}")
            else:
                if (
                    p.shape != y.shape
                    or not np.isin(p, [0, 1]).all()
                    or not np.isin(y, [0, 1]).all()
                ):
                    raise ValueError(f"Invalid multilabel shape/values for {key}")
                classes = None
            available = {
                "accuracy": float(metrics.accuracy_score(y, p)),
                "macro_f1": float(
                    metrics.f1_score(
                        y, p, labels=classes, average="macro", zero_division=0
                    )
                ),
                "micro_f1": float(
                    metrics.f1_score(
                        y, p, labels=classes, average="micro", zero_division=0
                    )
                ),
            }
            if st.task_type != "multilabel_classification":
                try:
                    val = float(
                        metrics.cohen_kappa_score(
                            y, p, labels=classes, weights="quadratic"
                        )
                    )
                    available["qwk"] = 0.0 if np.isnan(val) else val
                except Exception:
                    available["qwk"] = 0.0
            else:
                try:
                    val = float(metrics.ndcg_score(y, p))
                    available["ndcg"] = 0.0 if np.isnan(val) else val
                except Exception:
                    available["ndcg"] = 0.0
            if name in {"roc_auc", "logloss"}:
                raw = next(
                    (
                        batch.probabilities[k]
                        for k in candidates
                        if k in batch.probabilities
                    ),
                    None,
                )
                order = next(
                    (
                        batch.class_order[k]
                        for k in candidates
                        if k in batch.class_order
                    ),
                    None,
                )
                if raw is None or order is None or len(raw) != len(truth):
                    raise ValueError(
                        f"{name} requires probabilities and class order for {key}"
                    )
                prob = np.asarray(raw, float)[mask]
                if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
                    raise ValueError("Invalid probabilities")
                order = [_normalize_class_value(c) for c in order]
                expected_classes = (
                    [_normalize_class_value(c) for c in (st.labels or [])]
                    if st.task_type == "multilabel_classification"
                    else classes
                )
                if len(set(order)) != len(order) or (
                    expected_classes is not None
                    and len(expected_classes) > 0
                    and set(order) != set(expected_classes)
                ):
                    raise ValueError(
                        "Probability class order must match declared classes"
                    )
                if prob.ndim != 2 or prob.shape[1] != len(order):
                    raise ValueError(
                        "Probability matrix must have one column per class"
                    )
                if st.task_type == "multiclass_classification" and not np.allclose(
                    prob.sum(axis=1), 1
                ):
                    raise ValueError(
                        "Probability matrix must have one column per class and sum to one"
                    )
                if st.task_type == "multilabel_classification":
                    target_order = [str(c) for c in (st.labels or order)]
                    reordered = prob[:, [order.index(c) for c in target_order]]
                    if name == "logloss":
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            available[name] = float(metrics.log_loss(y, reordered))
                    else:
                        try:
                            val = float(metrics.roc_auc_score(y, reordered, average="macro"))
                            available[name] = 0.5 if np.isnan(val) else val
                        except Exception:
                            available[name] = 0.5
                else:
                    sorted_order = sorted(order)
                    reordered = prob[:, [order.index(c) for c in sorted_order]]
                    available[name] = float(
                        metrics.log_loss(y, reordered, labels=sorted_order)
                        if name == "logloss"
                        else (
                            metrics.roc_auc_score(y == order[1], prob[:, 1])
                            if len(order) == 2
                            else metrics.roc_auc_score(
                                y,
                                reordered,
                                labels=sorted_order,
                                multi_class="ovr",
                                average="macro",
                            )
                        )
                    )
        else:
            raise ValueError(f"Unsupported task type {st.task_type}")
        if name not in available:
            raise ValueError(f"Unsupported metric {name} for {st.task_type}")
        results[key] = {
            **available,
            "score": available[name],
            "n_samples": int(mask.sum()),
        }
        if st.condition and st.condition.eval_condition_only:
            results[f"{key}_oracle"] = results[key]
        weighted += available[name] * st.weight
        total += st.weight
    score = weighted / total
    return EvaluationResult(
        metric_name(spec.primary_metric),
        score,
        spec.direction,
        utility(score, spec.direction),
        results,
        len(truth),
    ).to_dict()


def validate_prediction_contract(spec, truth, predictions):
    if (
        truth.empty
        or spec.id_column not in truth
        or truth[spec.id_column].isna().any()
        or truth[spec.id_column].duplicated().any()
    ):
        raise ValueError(
            "Evaluation requires unique non-null IDs and nonempty coverage"
        )
    batch = PredictionBatch.from_mapping(predictions, truth[spec.id_column])
    for key, st in spec.subtasks.items():
        col = st.column or key
        columns = (
            [st.column_prefix + str(c) for c in st.labels]
            if st.column_prefix
            else [col]
        )
        if not all(c in truth and not truth[c].isna().any() for c in columns):
            raise ValueError("Missing target: " + key)
        values = next(
            (
                batch.labels[k]
                for k in (key, col, "pred_" + key, "pred_" + col)
                if k in batch.labels
            ),
            None,
        )
        if values is None or len(values) != len(truth):
            raise ValueError("Missing prediction coverage: " + key)
        array = np.asarray(values)
        if st.task_type in ("regression", "ranking"):
            if (
                array.shape != (len(truth),)
                or not np.isfinite(array.astype(float)).all()
            ):
                raise ValueError("Invalid regression predictions")
        elif st.task_type == "multilabel_classification":
            width = len(st.labels or st.classes)
            if array.shape != (len(truth), width) or not np.isin(array, [0, 1]).all():
                raise ValueError("Invalid multilabel predictions")
        elif st.classes:
            normalized_classes = set(_normalize_class_value(c) for c in st.classes)
            normalized_array = _normalize_class_array(array)
            if not set(normalized_array).issubset(normalized_classes):
                if (
                    st.task_type == "binary_classification"
                    and np.issubdtype(array.dtype, np.floating)
                    and len(array) > 0
                    and (array >= 0.0).all()
                    and (array <= 1.0).all()
                ):
                    raise ValueError(
                        f"Predictions for target '{key}' contain continuous probabilities "
                        f"(range [{array.min():.4f}, {array.max():.4f}]) instead of discrete class labels {st.classes}. "
                        f"For probabilistic evaluation (roc_auc, logloss), provide probabilities as 'proba_{key}' "
                        f"or threshold predictions into discrete class labels."
                    )
                raise ValueError(
                    f"Predictions outside target domain for '{key}': "
                    f"found {set(normalized_array) - normalized_classes} not in {normalized_classes}"
                )
    return batch

