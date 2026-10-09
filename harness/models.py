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
import warnings

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    LGBMClassifier = LGBMRegressor = None
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    XGBClassifier = XGBRegressor = None
    HAS_XGBOOST = False



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


class LightGBMModel:
    """Built-in LightGBM gradient boosting wrapper for classification and regression."""

    NAMES = {
        "lightgbm",
        "lgbm",
        "lightgbm_classifier",
        "lightgbm_regressor",
        "lgbm_classifier",
        "lgbm_regressor",
    }

    def __init__(self, task_spec, model_name="lightgbm", **kwargs):
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM is not installed. Run `pip install lightgbm` to use this model family."
            )
        clean_name = str(model_name).lower()
        if clean_name not in self.NAMES:
            raise ValueError(f"Unsupported LightGBM model name: {model_name}")
        self.task_spec = task_spec
        self.model_name = clean_name
        self.kwargs = kwargs
        self.fitted_models = {}
        self.estimator_metadata = {}
        self.seed = task_spec.resolve_seed()

    def fit(self, df, variant=None):
        if variant not in self.task_spec.inputs_dict:
            raise ValueError("LightGBM requires an explicitly declared input variant")
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
            cls = LGBMRegressor if reg else LGBMClassifier
            valid = inspect.signature(cls).parameters
            if set(self.kwargs) - set(valid):
                raise ValueError(
                    f"Unsupported estimator parameters: {sorted(set(self.kwargs)-set(valid))}"
                )
            params = {"verbosity": -1, "verbose": -1}
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
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
        total = 0
        for pipe, _, _ in self.fitted_models.values():
            clf = pipe.named_steps["clf"]
            n_trees = getattr(clf, "n_estimators_", getattr(clf, "n_estimators", None)) or 100
            if hasattr(clf, "booster_") and hasattr(clf.booster_, "num_trees"):
                try:
                    n_trees = clf.booster_.num_trees()
                except Exception:
                    pass
            leaves = getattr(clf, "num_leaves", None) or 31
            total += int(n_trees * leaves)
        return max(total, 1)


class XGBoostModel:
    """Built-in XGBoost gradient boosting wrapper for classification and regression."""

    NAMES = {
        "xgboost",
        "xgb",
        "xgboost_classifier",
        "xgboost_regressor",
        "xgb_classifier",
        "xgb_regressor",
    }

    def __init__(self, task_spec, model_name="xgboost", **kwargs):
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Run `pip install xgboost` to use this model family."
            )
        clean_name = str(model_name).lower()
        if clean_name not in self.NAMES:
            raise ValueError(f"Unsupported XGBoost model name: {model_name}")
        self.task_spec = task_spec
        self.model_name = clean_name
        self.kwargs = kwargs
        self.fitted_models = {}
        self.label_encoders = {}
        self.estimator_metadata = {}
        self.seed = task_spec.resolve_seed()

    def fit(self, df, variant=None):
        from sklearn.preprocessing import LabelEncoder

        if variant not in self.task_spec.inputs_dict:
            raise ValueError("XGBoost requires an explicitly declared input variant")
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
            cls = XGBRegressor if reg else XGBClassifier
            valid = inspect.signature(cls).parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid.values())
            if not has_var_kw and (set(self.kwargs) - set(valid)):
                raise ValueError(
                    f"Unsupported estimator parameters: {sorted(set(self.kwargs)-set(valid))}"
                )
            params = {"verbosity": 0}
            params.update(self.kwargs)
            if "random_state" in valid or has_var_kw:
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

            if not reg:
                le = LabelEncoder()
                y = le.fit_transform(df[col])
                self.label_encoders[key] = le
            else:
                y = df[col]

            pipe = Pipeline([("features", transform), ("clf", estimator)])
            pipe.fit(x, y)
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
            raw_pred = pipe.predict(x)
            if key in self.label_encoders:
                result[key] = self.label_encoders[key].inverse_transform(raw_pred)
                if hasattr(pipe, "predict_proba"):
                    result[f"proba_{key}"] = pipe.predict_proba(x)
                    result[f"classes_{key}"] = list(self.label_encoders[key].classes_)
            else:
                result[key] = raw_pred
                if hasattr(pipe, "predict_proba"):
                    result[f"proba_{key}"] = pipe.predict_proba(x)
                    result[f"classes_{key}"] = list(pipe.classes_)
        return result

    def num_parameters(self):
        total = 0
        for key, (pipe, _, _) in self.fitted_models.items():
            clf = pipe.named_steps["clf"]
            n_trees = getattr(clf, "n_estimators", None) or 100
            depth = getattr(clf, "max_depth", None) or 6
            try:
                booster = clf.get_booster()
                dump = booster.get_dump()
                n_trees = len(dump)
            except Exception:
                pass
            total += int(n_trees * (2 ** depth))
        return max(total, 1)


def create_model(task_spec, model_name="baseline", **kwargs):
    """Unified factory creating built-in models across linear, LightGBM, and XGBoost families."""
    name_clean = str(model_name).lower()
    if name_clean in GenericBaselineModel.NAMES:
        return GenericBaselineModel(task_spec, model_name, **kwargs)
    if name_clean in LightGBMModel.NAMES:
        return LightGBMModel(task_spec, model_name, **kwargs)
    if name_clean in XGBoostModel.NAMES:
        return XGBoostModel(task_spec, model_name, **kwargs)
    raise ValueError(
        f"Unsupported built-in model {model_name}; configure a plugin/factory"
    )
