"""Model registry and factory for StereoQueerEval."""

from __future__ import annotations

from typing import Any

from src.models.baseline_tfidf import BaselineTfidfModel

MODEL_REGISTRY = {
    "tfidf_logistic_regression": BaselineTfidfModel,
}


def build_model(model_name: str, **kwargs: Any):
    """Instantiate a model registered in MODEL_REGISTRY."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
