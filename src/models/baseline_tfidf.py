"""TF-IDF and Logistic Regression baseline model for StereoQueerEval."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from src.data import IDENTITIES, TARGET_TYPES, build_text


def make_vectorizer() -> FeatureUnion:
    """Create word and character n-gram TF-IDF feature extractor."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=40_000, sublinear_tf=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def make_classifier(y: pd.Series | np.ndarray, c_param: float = 2.0, seed: int = 42):
    """Instantiate a balanced LogisticRegression classifier or fallback dummy."""
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy="most_frequent")
    return LogisticRegression(
        C=c_param,
        class_weight="balanced",
        max_iter=1_000,
        solver="lbfgs",
        random_state=seed,
    )


class BaselineTfidfModel:
    """End-to-end TF-IDF baseline for Subtasks A, B, and C."""

    def __init__(self, c_param: float = 2.0, seed: int = 42):
        self.c_param = c_param
        self.seed = seed
        self.featurizer = make_vectorizer()
        self.model_a = None
        self.model_b = None
        self.model_c_type = None
        self.models_c_identity = {}

    def fit(self, train_df: pd.DataFrame, variant: str) -> BaselineTfidfModel:
        """Fit feature union and all subtask classifiers on training data."""
        text_train = build_text(train_df, variant)
        x_train = self.featurizer.fit_transform(text_train)

        # Subtask A
        self.model_a = make_classifier(train_df["stereotype"], self.c_param, self.seed)
        self.model_a.fit(x_train, train_df["stereotype"])

        # Subtask B
        self.model_b = make_classifier(train_df["hate_speech"], self.c_param, self.seed)
        self.model_b.fit(x_train, train_df["hate_speech"])

        # Subtask C (Trained on hateful rows)
        train_hateful_mask = (train_df["hate_speech"] != "no").to_numpy()
        if train_hateful_mask.sum() > 0:
            x_train_c = x_train[train_hateful_mask]
            train_c = train_df[train_hateful_mask].reset_index(drop=True)

            self.model_c_type = make_classifier(train_c["target_type"], self.c_param, self.seed)
            self.model_c_type.fit(x_train_c, train_c["target_type"])

            self.models_c_identity = {}
            for identity in IDENTITIES:
                clf = make_classifier(train_c[f"identity_{identity}"], self.c_param, self.seed)
                clf.fit(x_train_c, train_c[f"identity_{identity}"])
                self.models_c_identity[identity] = clf

        return self

    def predict(self, dev_df: pd.DataFrame, variant: str) -> Dict[str, Any]:
        """Generate predictions for dev_df across all subtasks."""
        text_dev = build_text(dev_df, variant)
        x_dev = self.featurizer.transform(text_dev)

        pred_a = self.model_a.predict(x_dev)
        pred_b = self.model_b.predict(x_dev)

        # For Subtask C, default to none / zeros
        n_samples = len(dev_df)
        pred_c_type = np.full(n_samples, "none", dtype=object)
        pred_identities = np.zeros((n_samples, len(IDENTITIES)), dtype=int)

        # Condition Subtask C predictions strictly on predicted hateful comments (pred_hate_speech != 'no')
        hateful_mask = pred_b != "no"
        if hateful_mask.sum() > 0:
            x_dev_hateful = x_dev[hateful_mask]
            if self.model_c_type is not None:
                pred_c_type[hateful_mask] = self.model_c_type.predict(x_dev_hateful)

            for idx, identity in enumerate(IDENTITIES):
                if identity in self.models_c_identity:
                    pred_identities[hateful_mask, idx] = (
                        self.models_c_identity[identity].predict(x_dev_hateful).astype(int)
                    )

        return {
            "pred_stereotype": pred_a,
            "pred_hate_speech": pred_b,
            "pred_target_type": pred_c_type,
            "pred_identities": pred_identities,
        }
