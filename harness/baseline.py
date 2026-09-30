"""Evidence gates for establishing a real, artifact-backed baseline anchor."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _normalized(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def extract_primary_metric(metrics: Any, primary_metric: str, exp_id: str) -> Tuple[Optional[float], str]:
    """Extract the task's declared score from common runner metrics.json shapes."""
    records = metrics if isinstance(metrics, list) else [metrics]
    normalized_metric = _normalized(primary_metric)
    for record in records:
        if not isinstance(record, dict):
            continue
        recorded_id = record.get("experiment_id") or record.get("exp_id")
        if recorded_id is not None and str(recorded_id) != exp_id:
            continue
        candidates = [
            (key, value) for key, value in record.items()
            if _normalized(str(key)) in {normalized_metric, "primarymetric", "primaryscore"}
        ]
        # Multi-output tasks report the scorer's official aggregate under composite_score.
        if not candidates and "composite_score" in record:
            candidates = [("composite_score", record["composite_score"])]
        for key, value in candidates:
            score = _number(value)
            if score is not None:
                return score, str(key)
    return None, ""


def validate_baseline_artifact(
    artifacts_dir: Path | str,
    exp_id: str,
    primary_metric: str,
) -> Dict[str, Any]:
    """Require a run artifact and finite primary metric; config defaults never qualify."""
    exp_dir = Path(artifacts_dir).resolve() / exp_id
    metrics_path = exp_dir / "metrics.json"
    if not exp_dir.is_dir() or not metrics_path.is_file():
        raise ValueError(
            f"Baseline `{exp_id}` has no completed run artifact. Expected `{metrics_path}`; run a validation first."
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Baseline metrics are unreadable at `{metrics_path}`: {exc}") from exc
    score, metric_key = extract_primary_metric(metrics, primary_metric, exp_id)
    if score is None:
        raise ValueError(
            f"Baseline `{exp_id}` has no finite `{primary_metric}` in metrics.json; configured baseline values do not count as run evidence."
        )
    return {
        "exp_id": exp_id,
        "score": score,
        "primary_metric": primary_metric,
        "metric_key": metric_key,
        "metrics_path": str(metrics_path),
        "run_config_path": str(exp_dir / "run_config.yaml") if (exp_dir / "run_config.yaml").exists() else None,
    }
