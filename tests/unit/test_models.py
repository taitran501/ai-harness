"""Consolidated unit test suite for estimators, baseline models, LLMEngine, and model feature validation.

Consolidates:
- tests/test_llm_engine.py (TestLLMEngine)
- Model tests from tests/agent_bridge/test_usability_phase5.py
- Model tests from tests/agent_bridge/test_usability_phase6.py
- Model tests from tests/agent_bridge/test_usability_phase7.py
- tests/test_harness_universal.py (TestDummyBaselineModel)
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from harness.llm_engine import LLMEngine
from harness.models import (
    HAS_CATBOOST,
    CatBoostModel,
    DummyBaselineModel,
    GenericBaselineModel,
    RandomForestModel,
    create_model,
    validate_model_features,
)
from harness.task_spec import SubtaskSpec, UniversalTaskSpec


class TestLLMEngine(unittest.TestCase):
    def test_list_providers(self):
        with patch.object(LLMEngine, "check_local_endpoint", return_value=False):
            providers = LLMEngine.list_providers()
        self.assertTrue(len(providers) >= 6)
        provider_ids = [p["id"] for p in providers]
        self.assertIn("gemini", provider_ids)
        self.assertIn("openai", provider_ids)
        self.assertIn("anthropic", provider_ids)
        self.assertIn("deepseek", provider_ids)
        self.assertIn("groq", provider_ids)
        self.assertIn("ollama", provider_ids)

    def test_format_prompt(self):
        template = "Comment: {comment} | Title: {title}"
        row = {"comment": "Great video!", "title": "Pride Parade 2026"}
        sys_prompt, user_prompt = LLMEngine.format_prompt(
            template=template,
            row_data=row,
            system_prompt="You are a classifier.",
            use_cot=True,
            valid_classes=["yes", "no"],
        )
        self.assertIn("Great video!", user_prompt)
        self.assertIn("Pride Parade 2026", user_prompt)
        self.assertIn("Think step by step", sys_prompt)
        self.assertIn("['yes', 'no']", sys_prompt)

    def test_extract_prediction_json(self):
        response_text = '```json\n{"prediction": "yes", "reasoning": "Explicit stereotype found"}\n```'
        result = LLMEngine.extract_prediction(response_text, valid_classes=["yes", "no"])
        self.assertEqual(result["parsed_label"], "yes")
        self.assertEqual(result["reasoning"], "Explicit stereotype found")
        self.assertTrue(result["is_valid"])

    def test_extract_prediction_fallback(self):
        response_text = "Based on the input context, the answer is clearly no."
        result = LLMEngine.extract_prediction(response_text, valid_classes=["yes", "no"])
        self.assertEqual(result["parsed_label"], "no")
        self.assertTrue(result["is_valid"])

    def test_generate_qlora_script(self):
        script = LLMEngine.generate_qlora_script(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            train_path="dataset/train.csv",
            lora_r=16,
        )
        self.assertIn("BitsAndBytesConfig", script)
        self.assertIn("LoraConfig", script)
        self.assertIn("SFTTrainer", script)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", script)

    def test_generate_unsloth_script(self):
        script = LLMEngine.generate_unsloth_script(
            model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            train_path="dataset/train.csv",
            lora_r=16,
            lora_alpha=32,
        )
        self.assertIn("FastLanguageModel", script)
        self.assertIn("unsloth", script)
        self.assertIn("SFTTrainer", script)
        self.assertIn("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", script)
        self.assertIn("is_bfloat16_supported", script)

    def test_fetch_provider_live_models_fallback(self):
        models = LLMEngine.fetch_provider_live_models("gemini")
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn("id", models[0])

    def test_fetch_unsloth_models_structure(self):
        models = LLMEngine.fetch_unsloth_models(limit=5)
        self.assertIsInstance(models, list)
        if models:
            self.assertIn("id", models[0])
            self.assertIn("family", models[0])


class TestModelFeaturesAndBaselines(unittest.TestCase):
    def test_feature_validation_diagnostics(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "feat_a": [0.1, 0.2, 0.3],
            "target": [0, 1, 0],
        })
        forbidden = {"id", "target", "fold"}

        # Case 1: Empty features
        with self.assertRaises(ValueError) as ctx:
            validate_model_features([], df, forbidden, variant="tab")
        self.assertIn("has no declared source_columns", str(ctx.exception))

        # Case 2: Missing features in DataFrame
        with self.assertRaises(ValueError) as ctx:
            validate_model_features(["feat_a", "feat_b", "feat_c"], df, forbidden, variant="tab")
        self.assertIn("missing in DataFrame", str(ctx.exception))
        self.assertIn("feat_b", str(ctx.exception))
        self.assertIn("feat_c", str(ctx.exception))

        # Case 3: Forbidden features declared
        with self.assertRaises(ValueError) as ctx:
            validate_model_features(["feat_a", "target", "id"], df, forbidden, variant="tab")
        self.assertIn("forbidden columns", str(ctx.exception))
        self.assertIn("target", str(ctx.exception))

        # Case 4: Valid features should succeed without error
        validate_model_features(["feat_a"], df, forbidden, variant="tab")

    @unittest.skipUnless(HAS_CATBOOST, "CatBoost is required for this test")
    def test_catboost_classification_and_regression(self):
        clf_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "label": {
                        "task_type": "binary_classification",
                        "column": "label",
                        "classes": [0, 1],
                    }
                },
                "inputs": {
                    "tabular_raw": {
                        "type": "tabular",
                        "source_columns": ["feat_1", "feat_2"],
                    }
                },
            }
        })
        clf_df = pd.DataFrame({
            "id": list(range(20)),
            "feat_1": np.linspace(0, 1, 20),
            "feat_2": np.linspace(1, 0, 20),
            "label": [0, 1] * 10,
        })

        model = create_model(clf_spec, "catboost", iterations=5, depth=3)
        self.assertIsInstance(model, CatBoostModel)
        model.fit(clf_df, variant="tabular_raw")
        preds = model.predict(clf_df, variant="tabular_raw")

        self.assertIn("label", preds)
        self.assertIn("proba_label", preds)
        self.assertEqual(len(preds["label"]), 20)
        self.assertEqual(preds["proba_label"].shape, (20, 2))
        self.assertGreater(model.num_parameters(), 0)

        # Also test regression
        reg_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "rmse",
                "direction": "minimize",
                "subtasks": {
                    "target": {
                        "task_type": "regression",
                        "column": "target",
                    }
                },
                "inputs": {
                    "tabular_raw": {
                        "type": "tabular",
                        "source_columns": ["feat_1", "feat_2"],
                    }
                },
            }
        })
        reg_df = pd.DataFrame({
            "id": list(range(20)),
            "feat_1": np.linspace(0, 1, 20),
            "feat_2": np.linspace(1, 0, 20),
            "target": np.linspace(10, 20, 20),
        })
        reg_model = create_model(reg_spec, "cb_regressor", iterations=5, depth=3)
        reg_model.fit(reg_df, variant="tabular_raw")
        reg_preds = reg_model.predict(reg_df, variant="tabular_raw")
        self.assertIn("target", reg_preds)
        self.assertEqual(len(reg_preds["target"]), 20)

    def test_random_forest_classification_and_regression(self):
        clf_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "label": {
                        "task_type": "binary_classification",
                        "column": "label",
                        "classes": [0, 1],
                    }
                },
                "inputs": {
                    "tabular_raw": {
                        "type": "tabular",
                        "source_columns": ["feat_1", "feat_2"],
                    }
                },
            }
        })
        clf_df = pd.DataFrame({
            "id": list(range(20)),
            "feat_1": np.linspace(0, 1, 20),
            "feat_2": np.linspace(1, 0, 20),
            "label": [0, 1] * 10,
        })

        model = create_model(clf_spec, "random_forest", n_estimators=10, max_depth=3)
        self.assertIsInstance(model, RandomForestModel)
        model.fit(clf_df, variant="tabular_raw")
        preds = model.predict(clf_df, variant="tabular_raw")

        self.assertIn("label", preds)
        self.assertIn("proba_label", preds)
        self.assertEqual(len(preds["label"]), 20)
        self.assertEqual(preds["proba_label"].shape, (20, 2))
        self.assertGreater(model.num_parameters(), 0)

        # Also test regression with alias 'rf'
        reg_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "rmse",
                "direction": "minimize",
                "subtasks": {
                    "target": {
                        "task_type": "regression",
                        "column": "target",
                    }
                },
                "inputs": {
                    "tabular_raw": {
                        "type": "tabular",
                        "source_columns": ["feat_1", "feat_2"],
                    }
                },
            }
        })
        reg_df = pd.DataFrame({
            "id": list(range(20)),
            "feat_1": np.linspace(0, 1, 20),
            "feat_2": np.linspace(1, 0, 20),
            "target": np.linspace(10, 20, 20),
        })
        reg_model = create_model(reg_spec, "rf", n_estimators=10, max_depth=3)
        reg_model.fit(reg_df, variant="tabular_raw")
        reg_preds = reg_model.predict(reg_df, variant="tabular_raw")
        self.assertIn("target", reg_preds)
        self.assertEqual(len(reg_preds["target"]), 20)

    def test_extended_baseline_aliases_and_lasso(self):
        clf_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "subtasks": {"y": {"task_type": "binary_classification", "column": "y", "classes": [0, 1]}},
                "inputs": {"raw": {"type": "tabular", "source_columns": ["x"]}},
            }
        })
        df = pd.DataFrame({"id": [1, 2, 3, 4], "x": [0.1, 0.2, 0.3, 0.4], "y": [0, 1, 0, 1]})

        # Test 'logreg' alias
        m_logreg = create_model(clf_spec, "logreg")
        self.assertIsInstance(m_logreg, GenericBaselineModel)
        m_logreg.fit(df, variant="raw")
        preds = m_logreg.predict(df, variant="raw")
        self.assertIn("y", preds)

        # Test 'lasso' and 'linear_regression' for regression
        reg_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "rmse",
                "direction": "minimize",
                "subtasks": {"target": {"task_type": "regression", "column": "target"}},
                "inputs": {"raw": {"type": "tabular", "source_columns": ["x"]}},
            }
        })
        reg_df = pd.DataFrame({"id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0], "target": [10.0, 20.0, 30.0, 40.0]})

        m_lasso = create_model(reg_spec, "lasso")
        m_lasso.fit(reg_df, variant="raw")
        lasso_preds = m_lasso.predict(reg_df, variant="raw")
        self.assertIn("target", lasso_preds)

        m_linreg = create_model(reg_spec, "linear_regression")
        m_linreg.fit(reg_df, variant="raw")
        lin_preds = m_linreg.predict(reg_df, variant="raw")
        self.assertIn("target", lin_preds)

    def test_random_forest_model_with_string_categories(self):
        train_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "brand": ["Toyota", "Ford", "Toyota", "BMW", "Ford", "BMW"],
            "mileage": [10000, 25000, np.nan, 5000, 18000, 30000],
            "price_cat": ["mid", "low", "mid", "high", "low", "high"],
        })
        spec_dict = {
            "task": {
                "name": "car_pricing",
                "task_type": "multiclass_classification",
                "primary_metric": "macro_f1",
                "id_column": "id",
                "inputs": [
                    {
                        "name": "tabular_feats",
                        "type": "tabular",
                        "source_columns": ["brand", "mileage"],
                    }
                ],
                "subtasks": {
                    "price_cat": {
                        "name": "price_cat",
                        "task_type": "multiclass_classification",
                        "classes": ["low", "mid", "high"],
                    }
                },
            }
        }
        spec = UniversalTaskSpec.from_dict(spec_dict)
        model = RandomForestModel(spec, n_estimators=10)
        model.fit(train_df, variant="tabular_feats")

        test_df = pd.DataFrame({
            "id": [7, 8],
            "brand": ["Audi", "Ford"],
            "mileage": [12000, np.nan],
        })
        preds = model.predict(test_df, variant="tabular_feats")
        self.assertIn("price_cat", preds)
        self.assertIn("proba_price_cat", preds)
        self.assertEqual(len(preds["price_cat"]), 2)
        self.assertEqual(len(preds["proba_price_cat"]), 2)


class TestDummyBaselineModel(unittest.TestCase):
    def test_dummy_model_classification(self):
        spec = UniversalTaskSpec(
            name="TestClf",
            task_type="binary_classification",
            primary_metric="macro_f1",
            subtasks={
                "target": SubtaskSpec(
                    id="target",
                    name="target",
                    task_type="binary_classification",
                    classes=["no", "yes"],
                    primary_metric="macro_f1",
                )
            },
        )
        model = create_model(spec, "dummy")
        self.assertIsInstance(model, DummyBaselineModel)
        df_train = pd.DataFrame({"id": [1, 2, 3, 4], "feature": [10, 20, 30, 40], "target": ["no", "yes", "no", "yes"]})
        model.fit(df_train)
        preds = model.predict(df_train)
        self.assertIn("target", preds)
        self.assertIn("proba_target", preds)
        self.assertIn("classes_target", preds)
        self.assertEqual(len(preds["target"]), 4)
        self.assertEqual(preds["proba_target"].shape, (4, 2))
        self.assertEqual(model.num_parameters(), 1)

    def test_dummy_model_regression(self):
        spec = UniversalTaskSpec(
            name="TestReg",
            task_type="regression",
            primary_metric="rmse",
            subtasks={
                "price": SubtaskSpec(
                    id="price",
                    name="price",
                    task_type="regression",
                    primary_metric="rmse",
                )
            },
        )
        model = create_model(spec, "mock")
        self.assertIsInstance(model, DummyBaselineModel)
        df_train = pd.DataFrame({"id": [1, 2], "feature": [1.0, 2.0], "price": [100.0, 200.0]})
        model.fit(df_train)
        preds = model.predict(df_train)
        self.assertIn("price", preds)
        self.assertEqual(len(preds["price"]), 2)
        self.assertNotEqual(preds["price"][0], preds["price"][1])
        self.assertEqual(model.num_parameters(), 1)


if __name__ == "__main__":
    unittest.main()


class FittedPredictionContractTests(unittest.TestCase):
    def test_shared_wrappers_preserve_input_and_actual_class_order(self):
        from types import SimpleNamespace
        from sklearn.preprocessing import LabelEncoder
        from harness.models import CatBoostModel, RandomForestModel
        class FakePipe:
            def __init__(self, classes, probabilities=True):
                self.classes = classes
                self.named_steps = {"clf": SimpleNamespace(classes_=classes)}
                if probabilities:
                    self.predict_proba = lambda x: np.tile([.7, .3], (len(x), 1))
            def predict(self, x):
                self.seen = x
                if self.classes is not None and len(self.classes) == 2:
                    return np.array([[self.classes[0]], [self.classes[1]]])
                return np.array([[1], [0]])
        frame = pd.DataFrame({"a": [2, 1], "b": [4, 3]})
        for wrapper in (CatBoostModel, RandomForestModel):
            for text in (False, True):
                for encoded in (False, True):
                    model = wrapper.__new__(wrapper)
                    encoder = LabelEncoder().fit(["a", "b", "missing"])
                    pipe = FakePipe([1, 0] if encoded else ["b", "a"])
                    inp = SimpleNamespace(type="text" if text else "tabular", compose_text=lambda df: ["row2", "row1"])
                    model.fitted_models = {"target": (pipe, inp, ["b", "a"])}
                    model.label_encoders = {"target": encoder} if encoded else {}
                    result = model.predict(frame)
                    self.assertEqual(result["target"].ndim, 1)
                    self.assertEqual(result["target"].tolist(), ["b", "a"])
                    self.assertEqual(result["classes_target"], ["b", "a"])
                    self.assertEqual(result["proba_target"].shape, (2, 2))
                    if text:
                        self.assertEqual(pipe.seen, ["row2", "row1"])
                    else:
                        pd.testing.assert_frame_equal(pipe.seen, frame[["b", "a"]])
            model = wrapper.__new__(wrapper)
            model.fitted_models = {"target": (FakePipe(None, probabilities=False), inp, ["a"])}
            model.label_encoders = {}
            self.assertEqual(set(model.predict(frame)), {"target"})
            for classes in (None, [0], [0, 0], [[0], [1]]):
                model.fitted_models = {"target": (FakePipe(classes), inp, ["a"])}
                with self.assertRaises(ValueError):
                    model.predict(frame)
