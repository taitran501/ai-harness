"""Explicit, leakage-safe sklearn baseline implementations."""

from __future__ import annotations
import inspect
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


class NativeCategoricalTransformer(BaseEstimator, TransformerMixin):
    """Passes numeric features directly and casts categorical columns to category dtype for native GBDT handling.

    Allows GBDT models (LightGBM, XGBoost) to utilize native categorical splitting and avoids
    forcing StandardScaler or arbitrary OneHotEncoder limits on tabular features.
    """

    def __init__(self):
        self.categories_ = {}
        self.cat_dtypes_ = {}

    def fit(self, X, y=None):
        self.categories_ = {}
        self.cat_dtypes_ = {}
        if hasattr(X, "columns"):
            for c in X.columns:
                if X[c].dtype.kind not in "biuf":
                    if hasattr(X[c], "cat"):
                        cats = list(X[c].cat.categories)
                        ordered = bool(X[c].cat.ordered)
                        cat_dtype = pd.CategoricalDtype(categories=cats, ordered=ordered)
                    else:
                        non_null = X[c].dropna()
                        str_series = non_null.astype(str)
                        cats = sorted(str_series.unique().tolist())
                        cat_dtype = pd.CategoricalDtype(categories=cats, ordered=False)
                    self.categories_[c] = cats
                    self.cat_dtypes_[c] = cat_dtype
        return self

    def transform(self, X):
        if not hasattr(X, "copy"):
            return X
        X_out = X.copy()
        dtypes = getattr(self, "cat_dtypes_", {})
        if dtypes and hasattr(X_out, "columns"):
            for c, cat_dtype in dtypes.items():
                if c in X_out.columns:
                    col = X_out[c]
                    cats = cat_dtype.categories
                    ordered = cat_dtype.ordered
                    if hasattr(col, "cat"):
                        X_out[c] = col.cat.set_categories(cats, ordered=ordered)
                    else:
                        s_str = col.astype(str).where(col.notna(), other=np.nan)
                        s_str = s_str.where(s_str.isin(cats), other=np.nan)
                        X_out[c] = pd.Categorical(s_str, categories=cats, ordered=ordered)
        return X_out



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
        "ridge_regression",
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
            reg = st.task_type in ("regression", "ranking")
            if self.model_name != "baseline" and (reg != ("ridge" in self.model_name)):
                raise ValueError("Requested model does not match target task type")
            cls = Ridge if reg else LogisticRegression
            valid = inspect.signature(cls).parameters
            model_kwargs = dict(self.kwargs)
            feat_transform = model_kwargs.pop("feature_transformer", None) or model_kwargs.pop("transform", None)
            if set(model_kwargs) - set(valid):
                raise ValueError(
                    f"Unsupported estimator parameters: {sorted(set(model_kwargs)-set(valid))}"
                )
            params = (
                {"solver": "lsqr"}
                if reg
                else {"max_iter": 500, "class_weight": "balanced"}
            )
            params.update(model_kwargs)
            if "random_state" in valid:
                params["random_state"] = self.seed
            estimator = cls(**params)
            if inp.type in {"text", "text_template"}:
                transform = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
                x = inp.compose_text(df)
            elif inp.type in {"tabular", "features", "passthrough", "multimodal"}:
                if feat_transform == "passthrough":
                    transform = "passthrough"
                elif feat_transform is not None:
                    transform = feat_transform
                else:
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
                                        (
                                            "encode",
                                            OneHotEncoder(
                                                handle_unknown="ignore",
                                                max_categories=30,
                                            ),
                                        ),
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
            reg = st.task_type in ("regression", "ranking")
            cls = LGBMRegressor if reg else LGBMClassifier
            valid = inspect.signature(cls).parameters
            model_kwargs = dict(self.kwargs)
            feat_transform = model_kwargs.pop("feature_transformer", None) or model_kwargs.pop("transform", None)
            if set(model_kwargs) - set(valid):
                raise ValueError(
                    f"Unsupported estimator parameters: {sorted(set(model_kwargs)-set(valid))}"
                )
            params = {"verbosity": -1, "verbose": -1}
            params.update(model_kwargs)
            if "random_state" in valid:
                params["random_state"] = self.seed
            estimator = cls(**params)

            if inp.type in {"text", "text_template"}:
                transform = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
                x = inp.compose_text(df)
            elif inp.type in {"tabular", "features", "passthrough", "multimodal"}:
                if feat_transform == "passthrough":
                    transform = "passthrough"
                elif feat_transform is not None:
                    transform = feat_transform
                else:
                    transform = NativeCategoricalTransformer()
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
            reg = st.task_type in ("regression", "ranking")
            cls = XGBRegressor if reg else XGBClassifier
            valid = inspect.signature(cls).parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid.values())
            model_kwargs = dict(self.kwargs)
            feat_transform = model_kwargs.pop("feature_transformer", None) or model_kwargs.pop("transform", None)
            if not has_var_kw and (set(model_kwargs) - set(valid)):
                raise ValueError(
                    f"Unsupported estimator parameters: {sorted(set(model_kwargs)-set(valid))}"
                )
            params = {"verbosity": 0, "enable_categorical": True}
            params.update(model_kwargs)
            if "random_state" in valid or has_var_kw:
                params["random_state"] = self.seed
            estimator = cls(**params)

            if inp.type in {"text", "text_template"}:
                transform = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
                x = inp.compose_text(df)
            elif inp.type in {"tabular", "features", "passthrough", "multimodal"}:
                if feat_transform == "passthrough":
                    transform = "passthrough"
                elif feat_transform is not None:
                    transform = feat_transform
                else:
                    transform = NativeCategoricalTransformer()
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


class DummyBaselineModel:
    """Fast, zero-overhead dummy estimator for testing, debugging, and rapid orchestration prototyping."""

    NAMES = {
        "dummy",
        "mock",
        "dummy_baseline",
        "dummy_classifier",
        "dummy_regressor",
    }

    def __init__(self, task_spec, model_name="dummy", **kwargs):
        self.task_spec = task_spec
        self.model_name = str(model_name).lower()
        self.kwargs = kwargs
        self.fitted_targets = {}
        self.estimator_metadata = {}
        self.seed = getattr(task_spec, "resolve_seed", lambda: 42)()

    def fit(self, df, variant=None):
        subtasks = getattr(self.task_spec, "subtasks", {})
        if not subtasks:
            target_col = (
                getattr(self.task_spec, "primary_target", None)
                or getattr(self.task_spec, "target_column", None)
                or (self.task_spec.metadata.get("task", {}).get("target_column") if hasattr(self.task_spec, "metadata") else None)
                or "target"
            )
            task_type = str(getattr(self.task_spec, "task_type", "")).lower()
            is_reg = any(r in task_type for r in ("regression", "ranking")) or (target_col in df and df[target_col].dtype.kind in "fc")
            if not is_reg:
                classes = sorted(df[target_col].dropna().unique().tolist()) if target_col in df else [0, 1]
                if not classes:
                    classes = [0, 1]
                self.fitted_targets[target_col] = {
                    "type": "clf",
                    "classes": classes,
                    "default": classes[0],
                }
            else:
                default_val = float(df[target_col].dropna().mean()) if (target_col in df and not df[target_col].dropna().empty) else 0.0
                self.fitted_targets[target_col] = {
                    "type": "reg",
                    "default": default_val,
                }
            self.estimator_metadata[target_col] = {
                "class": "DummyBaselineModel",
                "params": self.kwargs,
                "features": [],
            }
            return self

        for key, st in subtasks.items():
            col = getattr(st, "column", None) or key
            task_type = getattr(st, "task_type", "regression")
            is_reg = task_type in ("regression", "ranking")
            if not is_reg:
                classes = list(getattr(st, "classes", []) or [])
                if not classes and col in df:
                    try:
                        classes = sorted(df[col].dropna().unique().tolist())
                    except Exception:
                        classes = [0, 1]
                if not classes:
                    classes = [0, 1]
                self.fitted_targets[key] = {
                    "type": "clf",
                    "classes": classes,
                    "default": classes[0],
                }
            else:
                default_val = 0.0
                if col in df:
                    try:
                        s = pd.to_numeric(df[col], errors="coerce").dropna()
                        if not s.empty:
                            default_val = float(s.mean())
                    except Exception:
                        default_val = 0.0
                self.fitted_targets[key] = {
                    "type": "reg",
                    "default": default_val,
                }
            self.estimator_metadata[key] = {
                "class": "DummyBaselineModel",
                "params": self.kwargs,
                "features": [],
            }
        return self

    def predict(self, df, variant=None):
        n = len(df)
        result = {}
        for key, info in self.fitted_targets.items():
            if info["type"] == "clf":
                classes = info["classes"]
                k = len(classes)
                result[key] = np.array([info["default"]] * n)
                if k > 1:
                    proba = np.zeros((n, k), dtype=float)
                    proba[:, 0] = 0.6
                    proba[:, 1:] = 0.4 / (k - 1)
                    result[f"proba_{key}"] = proba
                    result[f"classes_{key}"] = list(classes)
            else:
                base = info["default"]
                result[key] = (
                    np.linspace(base, base + 0.01, n, dtype=float)
                    if n > 1
                    else np.full(n, base, dtype=float)
                )
        return result

    def num_parameters(self):
        return max(len(self.fitted_targets), 1)


def create_model(task_spec, model_name="baseline", **kwargs):
    """Unified factory creating built-in models across linear, GBDT, and dummy testing families."""
    name_clean = str(model_name).lower()
    if name_clean in DummyBaselineModel.NAMES:
        return DummyBaselineModel(task_spec, model_name, **kwargs)
    if name_clean in GenericBaselineModel.NAMES:
        return GenericBaselineModel(task_spec, model_name, **kwargs)
    if name_clean in LightGBMModel.NAMES:
        return LightGBMModel(task_spec, model_name, **kwargs)
    if name_clean in XGBoostModel.NAMES:
        return XGBoostModel(task_spec, model_name, **kwargs)
    raise ValueError(
        f"Unsupported built-in model {model_name}; configure a plugin/factory"
    )
