"""Explicit, leakage-safe sklearn baseline implementations."""

from __future__ import annotations
import inspect
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression


def call_model(model, method, frame, variant):
    fn = getattr(model, method)
    signature = inspect.signature(fn)
    supports = "variant" in signature.parameters or any(
        p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
    )
    return fn(frame, variant=variant) if supports else fn(frame)


class GenericBaselineModel:
    NAMES = {
        "baseline",
        "ridge",
        "tfidf_ridge",
        "logistic_regression",
        "tfidf_logistic_regression",
    }

    def __init__(self, task_spec, model_name="baseline", **kwargs):
        if model_name not in self.NAMES:
            raise ValueError(
                f"Unsupported built-in model {model_name}; configure a plugin/factory"
            )
        self.task_spec, self.model_name, self.kwargs = task_spec, model_name, kwargs
        self.fitted_models, self.estimator_metadata = {}, {}
        self.seed = task_spec.resolve_seed()

    def fit(self, df, variant=None):
        if variant not in self.task_spec.inputs_dict:
            raise ValueError("Baseline requires an explicitly declared input variant")
        inp = self.task_spec.inputs_dict[variant]
        targets = {s.column or s.id for s in self.task_spec.subtasks.values()}
        targets.update(
            c
            for s in self.task_spec.subtasks.values()
            if s.column_prefix
            for c in df
            if c.startswith(s.column_prefix)
        )
        forbidden = targets | {
            self.task_spec.id_column,
            self.task_spec.group_column,
            "fold",
            "split",
            "input_variant",
            "partition",
            self.task_spec.data_config.get("partition_column"),
            self.task_spec.validation.split_column,
        }
        features = list(inp.source_columns)
        if (
            not features
            or set(features) & forbidden
            or not all(c in df for c in features)
        ):
            raise ValueError(
                "Input features must exist and exclude targets, IDs, groups and split metadata"
            )
        for key, st in self.task_spec.subtasks.items():
            if st.task_type == "multilabel_classification":
                raise ValueError(
                    "Built-in multilabel training requires a task model plugin"
                )
            col = st.column or key
            reg = st.task_type == "regression"
            if self.model_name != "baseline" and (reg != ("ridge" in self.model_name)):
                raise ValueError("Requested model does not match target task type")
            cls = Ridge if reg else LogisticRegression
            valid = inspect.signature(cls).parameters
            if set(self.kwargs) - set(valid):
                raise ValueError(
                    f"Unsupported estimator parameters: {sorted(set(self.kwargs)-set(valid))}"
                )
            params = (
                {"solver": "lsqr"}
                if reg
                else {"max_iter": 500, "class_weight": "balanced"}
            )
            params.update(self.kwargs)
            if "random_state" in valid:
                params["random_state"] = self.seed
            estimator = cls(**params)
            if inp.type in {"text", "text_template"}:
                transform = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
                x = inp.compose_text(df)
            elif inp.type == "tabular":
                numeric = [c for c in features if df[c].dtype.kind in "biuf"]
                categorical = [c for c in features if c not in numeric]
                transform = ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                        (
                            "category",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical,
                        ),
                    ]
                )
                x = df[features]
            else:
                raise ValueError(f"Unsupported baseline input type {inp.type}")
            pipe = Pipeline([("features", transform), ("clf", estimator)])
            pipe.fit(x, df[col])
            self.fitted_models[key] = (pipe, inp, features)
            self.estimator_metadata[key] = {
                "class": type(estimator).__name__,
                "params": estimator.get_params(),
                "features": features,
            }
        return self

    def predict(self, df, variant=None):
        result = {}
        for key, (pipe, inp, features) in self.fitted_models.items():
            x = (
                inp.compose_text(df)
                if inp.type in {"text", "text_template"}
                else df[features]
            )
            result[key] = pipe.predict(x)
            if hasattr(pipe, "predict_proba"):
                result[f"proba_{key}"] = pipe.predict_proba(x)
                result[f"classes_{key}"] = list(pipe.classes_)
        return result

    def num_parameters(self):
        return sum(
            pipe.named_steps["clf"].coef_.size
            + np.asarray(pipe.named_steps["clf"].intercept_).size
            for pipe, _, _ in self.fitted_models.values()
        )
