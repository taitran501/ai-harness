"""Strict prediction and metric contracts shared by runners and consumers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn import metrics


def metric_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", value.lower())
    aliases = {
        # Classification
        "macrof1": "macro_f1",
        "f1macro": "macro_f1",
        "microf1": "micro_f1",
        "f1micro": "micro_f1",
        "weightedf1": "weighted_f1",
        "f1weighted": "weighted_f1",
        "accuracy": "accuracy",
        "acc": "accuracy",
        "balancedaccuracy": "balanced_accuracy",
        "balancedacc": "balanced_accuracy",
        "rocauc": "roc_auc",
        "aucroc": "roc_auc",
        "auc": "roc_auc",
        "prauc": "pr_auc",
        "auprc": "pr_auc",
        "averageprecision": "pr_auc",
        "logloss": "logloss",
        "brierscore": "brier_score",
        "brier": "brier_score",
        "mcc": "mcc",
        "matthewscorrcoef": "mcc",
        "matthewscorrelationcoefficient": "mcc",
        "qwk": "qwk",
        "quadraticweightedkappa": "qwk",
        "cohenkappa": "qwk",
        "cohenkappascore": "qwk",
        "kappa": "qwk",

        # Regression & Forecasting
        "rmse": "rmse",
        "rootmeansquarederror": "rmse",
        "mae": "mae",
        "meanabsoluteerror": "mae",
        "r2": "r2",
        "rsquared": "r2",
        "rmsle": "rmsle",
        "rootmeansquaredlogerror": "rmsle",
        "rootmeansquaredlogarithmicerror": "rmsle",
        "mape": "mape",
        "meanabsolutepercentageerror": "mape",
        "smape": "smape",
        "symmetricmeanabsolutepercentageerror": "smape",
        "wape": "wape",
        "weightedabsolutepercentageerror": "wape",
        "medae": "medae",
        "medianabsoluteerror": "medae",
        "pinballloss": "pinball_loss",
        "quantileloss": "pinball_loss",
        "pinball": "pinball_loss",
        "spearman": "spearman",
        "spearmanr": "spearman",
        "spearmanrho": "spearman",
        "spearmancorrelation": "spearman",
        "spearmanrankcorrelation": "spearman",
        "pearson": "pearson",
        "pearsonr": "pearson",
        "pearsoncorrelation": "pearson",

        # Ranking, Recommendation & Information Retrieval
        "ndcg": "ndcg",
        "ndcgscore": "ndcg",
        "normalizeddiscountedcumulativegain": "ndcg",
        "mrr": "mrr",
        "meanreciprocalrank": "mrr",
        "map": "map",
        "meanaverageprecision": "map",
        "mapatk": "map",

        # Computer Vision: Segmentation & Detection
        "dice": "dice",
        "dicecoef": "dice",
        "dicecoefficient": "dice",
        "iou": "iou",
        "jaccard": "iou",
        "jaccardscore": "iou",
        "bboxiou": "bbox_iou",
        "map50": "map_50",

        # Audio, Speech & NLP Generation
        "wer": "wer",
        "worderrorrate": "wer",
        "cer": "cer",
        "charactererrorrate": "cer",
        "bleu": "bleu",
        "bleuscore": "bleu",
        "rouge": "rouge_l",
        "rougel": "rouge_l",
        "exactmatch": "exact_match",
        "em": "exact_match",
        "tokenf1": "token_f1",
        "tokenaccuracy": "token_accuracy",
    }
    return aliases.get(name, value)


MINIMIZE_METRICS = {
    "rmse",
    "mae",
    "rmsle",
    "mape",
    "smape",
    "wape",
    "medae",
    "pinball_loss",
    "logloss",
    "brier_score",
    "wer",
    "cer",
}

MAXIMIZE_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "f1",
    "roc_auc",
    "pr_auc",
    "qwk",
    "mcc",
    "ndcg",
    "spearman",
    "r2",
    "dice_score",
    "iou_score",
    "exact_match",
    "bleu",
    "mrr",
    "pearson",
    "token_accuracy",
}

#: Stock task types the generic aggregate can score, and the metric names
#: available for each. This table is the single source of truth and is kept
#: in sync with the ``available`` dicts computed in ``evaluate()``: it lists
#: exactly the metric names the generic path can score for each stock task
#: type. Task types not listed here (segmentation, ASR, detection, NER,
#: interactive/agent tasks, ...) are not generically scorable; a custom
#: ``paths.metrics_evaluator`` is required for them.
#: All of these task types share the regression scoring branch in
#: ``evaluate()`` and therefore score the same metric set.
_REGRESSION_METRICS = {
    "rmse",
    "mae",
    "r2",
    "rmsle",
    "smape",
    "wape",
    "pinball_loss",
    "medae",
    "pearson",
    "mape",
    "spearman",
    "ndcg",
    "mrr",
    "qwk",
}
#: Classification metrics the generic path can score. Binary and multiclass
#: additionally support ``qwk`` and ``brier_score`` (the latter needs a
#: probability column); multilabel swaps those for ``ndcg``.
_CLASSIFICATION_METRICS = {
    "accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "logloss",
    "pr_auc",
    "brier_score",
    "qwk",
}
GENERIC_ALLOWED_METRICS = {
    "regression": _REGRESSION_METRICS,
    "ranking": _REGRESSION_METRICS,
    "time_series_forecasting": _REGRESSION_METRICS,
    "forecasting": _REGRESSION_METRICS,
    "time_series": _REGRESSION_METRICS,
    "multi_output_regression": _REGRESSION_METRICS,
    "binary_classification": _CLASSIFICATION_METRICS,
    "multiclass_classification": _CLASSIFICATION_METRICS,
    "multilabel_classification": _CLASSIFICATION_METRICS - {"qwk", "brier_score"} | {"ndcg"},
}


def default_metric_direction(metric: str) -> str:
    """Infer whether an evaluation metric should be minimized or maximized.

    Unknown/custom metrics default to ``maximize``; only the canonical sets
    above carry a strong prior, so direction-consistency checks should treat
    any other metric as declared-direction-is-trusted.
    """
    m = metric_name(metric)
    return "minimize" if m in MINIMIZE_METRICS else "maximize"


def canonical_direction_conflicts(metric: str, direction: str) -> Optional[str]:
    """Return a human-readable conflict note when a canonical metric is
    declared with the opposite direction, else ``None``.

    Metrics outside both canonical sets (custom scorers) never conflict:
    the declared direction is trusted.
    """
    m = metric_name(metric)
    if direction not in {"maximize", "minimize"}:
        return None
    if m in MINIMIZE_METRICS and direction == "maximize":
        return (
            f"{m} is canonically minimized but direction is 'maximize' — "
            "check the declared direction"
        )
    if m in MAXIMIZE_METRICS and direction == "minimize":
        return (
            f"{m} is canonically maximized but direction is 'minimize' — "
            "check the declared direction"
        )
    return None


def compute_rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Logarithmic Error with zero clipping."""
    y_t = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_t) - np.log1p(y_p)) ** 2)))


def compute_smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_t) + np.abs(y_p)
    mask = denom > 1e-8
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_t[mask] - y_p[mask]) / denom[mask]))


def compute_wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    sum_t = np.sum(np.abs(y_t))
    if sum_t == 0:
        return 0.0
    return float(np.sum(np.abs(y_t - y_p)) / sum_t)


def compute_pinball_loss(y_true, y_pred, alpha: float = 0.5) -> float:
    """Quantile / Pinball loss for probabilistic forecasting."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    diff = y_t - y_p
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def compute_pearson(y_true, y_pred) -> float:
    """Pearson correlation coefficient."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) < 2 or np.std(y_t) == 0 or np.std(y_p) == 0:
        return 0.0
    val = np.corrcoef(y_t, y_p)[0, 1]
    return 0.0 if np.isnan(val) else float(val)


def compute_mrr(y_true, y_pred) -> float:
    """Mean Reciprocal Rank."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if y_t.ndim == 1:
        order = np.argsort(-y_p)
        sorted_t = y_t[order]
        pos = np.where(sorted_t > 0)[0]
        return float(1.0 / (pos[0] + 1)) if len(pos) > 0 else 0.0
    rr_list = []
    for t_row, p_row in zip(y_t, y_p):
        order = np.argsort(-p_row)
        pos = np.where(t_row[order] > 0)[0]
        rr_list.append(1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0)
    return float(np.mean(rr_list)) if rr_list else 0.0


def compute_dice_score(y_true, y_pred, smooth: float = 1e-6) -> float:
    """Dice / F1 coefficient on binary masks."""
    y_t = (np.asarray(y_true) > 0.5).astype(float).ravel()
    y_p = (np.asarray(y_pred) > 0.5).astype(float).ravel()
    intersection = np.sum(y_t * y_p)
    return float((2.0 * intersection + smooth) / (np.sum(y_t) + np.sum(y_p) + smooth))


def compute_iou_score(y_true, y_pred, smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard) on binary masks."""
    y_t = (np.asarray(y_true) > 0.5).astype(float).ravel()
    y_p = (np.asarray(y_pred) > 0.5).astype(float).ravel()
    intersection = np.sum(y_t * y_p)
    union = np.sum(y_t) + np.sum(y_p) - intersection
    return float((intersection + smooth) / (union + smooth))


def mask_to_rle(mask: np.ndarray) -> str:
    """Convert binary 2D mask array to 1-indexed Run-Length Encoded (RLE) string."""
    dots = mask.T.flatten()
    runs = np.where(dots[1:] != dots[:-1])[0] + 2
    runs = np.concatenate([[1] if dots[0] else [], runs, [len(dots) + 1] if dots[-1] else []])
    lengths = runs[1::2] - runs[::2]
    starts = runs[::2]
    return " ".join(f"{int(start)} {int(length)}" for start, length in zip(starts, lengths))


def rle_to_mask(rle_str: str, shape: tuple[int, int]) -> np.ndarray:
    """Convert Run-Length Encoded string back to 2D binary numpy mask (height, width)."""
    if not rle_str or pd.isna(rle_str):
        return np.zeros(shape, dtype=np.uint8)
    s = [int(float(x)) for x in str(rle_str).split()]
    starts, lengths = s[0::2], s[1::2]
    starts = [x - 1 for x in starts]
    ends = [x + y for x, y in zip(starts, lengths)]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape, order="F")


def compute_box_iou(box1: np.ndarray | list, box2: np.ndarray | list) -> float:
    """Compute IoU between two bounding boxes in [xmin, ymin, xmax, ymax] format."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    b1_area = max(0.0, (b1_x2 - b1_x1)) * max(0.0, (b1_y2 - b1_y1))
    b2_area = max(0.0, (b2_x2 - b2_x1)) * max(0.0, (b2_y2 - b2_y1))
    union_area = b1_area + b2_area - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_edit_distance(seq1: list | str, seq2: list | str) -> int:
    """Levenshtein distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return int(dp[n][m])


def compute_wer(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> float:
    """Word Error Rate for speech recognition and transcription."""
    total_dist, total_words = 0, 0
    for ref, hyp in zip(y_true, y_pred):
        ref_words = str(ref).strip().split()
        hyp_words = str(hyp).strip().split()
        total_dist += compute_edit_distance(ref_words, hyp_words)
        total_words += max(len(ref_words), 1)
    return float(total_dist / total_words) if total_words > 0 else 0.0


def compute_cer(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> float:
    """Character Error Rate for OCR and speech recognition."""
    total_dist, total_chars = 0, 0
    for ref, hyp in zip(y_true, y_pred):
        ref_chars = list(str(ref).strip())
        hyp_chars = list(str(hyp).strip())
        total_dist += compute_edit_distance(ref_chars, hyp_chars)
        total_chars += max(len(ref_chars), 1)
    return float(total_dist / total_chars) if total_chars > 0 else 0.0


def compute_exact_match(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> float:
    """Exact Match ratio for QA or code generation."""
    matches = [1.0 if str(yt).strip() == str(yp).strip() else 0.0 for yt, yp in zip(y_true, y_pred)]
    return float(np.mean(matches)) if matches else 0.0


def compute_bleu(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray, max_n: int = 4) -> float:
    """Corpus-level BLEU approximation with Chen-Cherry smoothing for short sentences."""
    precisions = []
    total_ref_len, total_hyp_len = 0, 0
    max_ref_len = max((len(str(r).strip().split()) for r in y_true), default=1)
    max_hyp_len = max((len(str(h).strip().split()) for h in y_pred), default=1)
    effective_n = min(max_n, max(1, min(max_ref_len, max_hyp_len)))

    for n in range(1, effective_n + 1):
        match_count, hyp_count = 0, 0
        for ref, hyp in zip(y_true, y_pred):
            ref_tokens = str(ref).strip().split()
            hyp_tokens = str(hyp).strip().split()
            if n == 1:
                total_ref_len += len(ref_tokens)
                total_hyp_len += len(hyp_tokens)
            ref_ngrams = {}
            for i in range(len(ref_tokens) - n + 1):
                ng = tuple(ref_tokens[i : i + n])
                ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
            hyp_ngrams = {}
            for i in range(len(hyp_tokens) - n + 1):
                ng = tuple(hyp_tokens[i : i + n])
                hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1
            for ng, cnt in hyp_ngrams.items():
                match_count += min(cnt, ref_ngrams.get(ng, 0))
                hyp_count += cnt
        if hyp_count > 0:
            if match_count == 0 and n > 1:
                precisions.append(1.0 / (2.0 * hyp_count))
            else:
                precisions.append(max(match_count, 1e-4) / hyp_count)
        else:
            precisions.append(1e-4)

    geo_mean = np.exp(np.mean(np.log(precisions)))
    bp = 1.0 if total_hyp_len >= total_ref_len else np.exp(1.0 - (total_ref_len / max(total_hyp_len, 1e-8)))
    return float(bp * geo_mean)


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
    pairs = {(metric_name(s.primary_metric), s.direction) for s in active}
    if len(pairs) > 1:
        raise ValueError(
            "Mixed metrics/directions require a custom scorer: "
            f"subtasks declare {sorted(pairs)} — the generic aggregate can only "
            "score a single (metric, direction) pair. Declare one consistent "
            "subtask metric matching the spec, or configure "
            "paths.metrics_evaluator in the run config."
        )
    if any(not np.isfinite(s.weight) or s.weight < 0 for s in spec.subtasks.values()):
        raise ValueError("Invalid target weights")
    sub_pair = (metric_name(active[0].primary_metric), active[0].direction)
    spec_pair = (metric_name(spec.primary_metric), spec.direction)
    if sub_pair != spec_pair:
        raise ValueError(
            f"Primary metric/direction must match the generic aggregate: "
            f"subtask declares {sub_pair} but the spec declares {spec_pair}. "
            "Align the subtask primary_metric/direction with the spec's "
            "primary_metric/direction, or configure paths.metrics_evaluator "
            "in the run config so a custom scorer is authoritative."
        )
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
        if st.task_type in {
            "regression",
            "ranking",
            "time_series_forecasting",
            "forecasting",
            "time_series",
            "multi_output_regression",
        }:
            y, p = np.asarray(y, float), np.asarray(p, float)
            if not np.isfinite(y).all() or not np.isfinite(p).all():
                raise ValueError("Regression values must be finite")
            available = {
                "rmse": float(np.sqrt(metrics.mean_squared_error(y, p))),
                "mae": float(metrics.mean_absolute_error(y, p)),
                "r2": float(metrics.r2_score(y, p)) if len(y) > 1 else 0.0,
                "rmsle": compute_rmsle(y, p),
                "smape": compute_smape(y, p),
                "wape": compute_wape(y, p),
                "pinball_loss": compute_pinball_loss(y, p),
                "medae": float(metrics.median_absolute_error(y, p)),
                "pearson": compute_pearson(y, p),
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
                available["mrr"] = compute_mrr(y, p)
            except Exception:
                available["mrr"] = 0.0
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
                "weighted_f1": float(
                    metrics.f1_score(
                        y, p, labels=classes, average="weighted", zero_division=0
                    )
                ),
            }
            try:
                available["balanced_accuracy"] = float(metrics.balanced_accuracy_score(y, p))
            except Exception:
                available["balanced_accuracy"] = 0.0
            try:
                available["mcc"] = float(metrics.matthews_corrcoef(y, p))
            except Exception:
                available["mcc"] = 0.0

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
            if name in {"roc_auc", "logloss", "pr_auc", "brier_score"}:
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
                    elif name == "pr_auc":
                        try:
                            val = float(metrics.average_precision_score(y, reordered, average="macro"))
                            available[name] = 0.0 if np.isnan(val) else val
                        except Exception:
                            available[name] = 0.0
                    else:
                        try:
                            val = float(metrics.roc_auc_score(y, reordered, average="macro"))
                            available[name] = 0.5 if np.isnan(val) else val
                        except Exception:
                            available[name] = 0.5
                else:
                    sorted_order = sorted(order)
                    reordered = prob[:, [order.index(c) for c in sorted_order]]
                    if name == "logloss":
                        available[name] = float(metrics.log_loss(y, reordered, labels=sorted_order))
                    elif name == "brier_score":
                        if len(order) == 2:
                            available[name] = float(metrics.brier_score_loss(y == order[1], prob[:, 1]))
                        else:
                            try:
                                dum = pd.get_dummies(y)[sorted_order].to_numpy()
                                available[name] = float(np.mean(np.sum((dum - reordered) ** 2, axis=1)))
                            except Exception:
                                available[name] = 0.0
                    elif name == "pr_auc":
                        if len(order) == 2:
                            available[name] = float(metrics.average_precision_score(y == order[1], prob[:, 1]))
                        else:
                            try:
                                dum = pd.get_dummies(y)[sorted_order].to_numpy()
                                available[name] = float(metrics.average_precision_score(dum, reordered, average="macro"))
                            except Exception:
                                available[name] = 0.0
                    else:
                        available[name] = float(
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
        elif st.task_type in {"segmentation", "semantic_segmentation", "instance_segmentation"}:
            dice_scores = [compute_dice_score(yt, yp) for yt, yp in zip(y, p)]
            iou_scores = [compute_iou_score(yt, yp) for yt, yp in zip(y, p)]
            available = {
                "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
                "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
            }
        elif st.task_type in {
            "speech_recognition",
            "asr",
            "transcription",
            "text_generation",
            "generation",
            "seq2seq",
            "summarization",
            "translation",
        }:
            y_str = [str(x) for x in y]
            p_str = [str(x) for x in p]
            available = {
                "wer": compute_wer(y_str, p_str),
                "cer": compute_cer(y_str, p_str),
                "bleu": compute_bleu(y_str, p_str),
                "exact_match": compute_exact_match(y_str, p_str),
            }
        elif st.task_type in {"object_detection", "detection"}:
            box_ious = [compute_box_iou(b1, b2) for b1, b2 in zip(y, p)]
            available = {
                "bbox_iou": float(np.mean(box_ious)) if box_ious else 0.0,
                "map_50": float(np.mean([1.0 if iou >= 0.5 else 0.0 for iou in box_ious])) if box_ious else 0.0,
            }
        elif st.task_type in {"token_classification", "ner"}:
            try:
                flat_y = np.concatenate(y) if hasattr(y[0], "__iter__") and not isinstance(y[0], str) else y
                flat_p = np.concatenate(p) if hasattr(p[0], "__iter__") and not isinstance(p[0], str) else p
                available = {
                    "token_accuracy": float(metrics.accuracy_score(flat_y, flat_p)),
                    "token_f1": float(metrics.f1_score(flat_y, flat_p, average="macro", zero_division=0)),
                }
            except Exception:
                available = {
                    "token_accuracy": float(metrics.accuracy_score(y, p)),
                    "token_f1": float(metrics.f1_score(y, p, average="macro", zero_division=0)),
                }
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

