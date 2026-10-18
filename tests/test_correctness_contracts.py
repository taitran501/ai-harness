import unittest

import numpy as np
import pandas as pd

from harness.task_spec import UniversalTaskSpec


def task(kind="regression", metric="rmse", direction="minimize"):
    return UniversalTaskSpec.from_dict(
        {
            "task": {
                "name": "fixture",
                "modality": "tabular",
                "primary_metric": metric,
                "direction": direction,
                "data": {"id_column": "id", "group_column": "group", "seed": 13},
                "inputs": [
                    {
                        "name": "features",
                        "type": "tabular",
                        "source_columns": ["x", "category"],
                    }
                ],
                "subtasks": [
                    {
                        "id": "target",
                        "column": "target",
                        "task_type": kind,
                        "primary_metric": metric,
                        "direction": direction,
                        "classes": [] if kind == "regression" else ["0", "1"],
                    }
                ],
            }
        }
    )


class EvaluationContracts(unittest.TestCase):
    def test_raw_regression_and_utility(self):
        result = task().evaluate_predictions(
            pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}), {"target": [0.0, 3.0]}
        )
        self.assertAlmostEqual(result["primary_score"], np.sqrt(2))
        self.assertAlmostEqual(result["utility"], -np.sqrt(2))

    def test_probability_metric(self):
        spec = task("binary_classification", "ROC-AUC", "maximize")
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            spec.evaluate_predictions(truth, {"target": [0, 0, 1, 1]})
        result = spec.evaluate_predictions(
            truth,
            {
                "target": [0, 0, 1, 1],
                "proba_target": [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
                "classes_target": [0, 1],
            },
        )
        self.assertEqual(result["primary_score"], 1)

    def test_multilabel_probability_metric_does_not_require_sum_to_one(self):
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "name": "multilabel_fixture",
                "modality": "tabular",
                "primary_metric": "roc_auc",
                "direction": "maximize",
                "data": {"id_column": "id"},
                "inputs": [{"name": "features", "type": "tabular", "source_columns": ["x"]}],
                "subtasks": [{
                    "id": "target",
                    "column_prefix": "lbl_",
                    "task_type": "multilabel_classification",
                    "primary_metric": "roc_auc",
                    "direction": "maximize",
                    "labels": ["a", "b"],
                }],
            }
        })
        truth = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "lbl_a": [0, 0, 1, 1],
            "lbl_b": [1, 0, 1, 0],
        })
        # Probabilities do NOT sum to 1 across columns (e.g. [0.8, 0.9] sums to 1.7)
        result = spec.evaluate_predictions(
            truth,
            {
                "target": [[0, 1], [0, 0], [1, 1], [1, 0]],
                "proba_target": [[0.1, 0.8], [0.2, 0.3], [0.9, 0.9], [0.8, 0.2]],
                "classes_target": ["a", "b"],
            },
        )
        self.assertGreater(result["primary_score"], 0.5)

    def test_multiclass_strictly_enforces_sum_to_one(self):
        spec = task("multiclass_classification", "logloss", "minimize")
        truth = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
        # Probabilities sum to 1.5 instead of 1.0 -> must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            spec.evaluate_predictions(
                truth,
                {
                    "target": [0, 1],
                    "proba_target": [[0.5, 1.0], [0.8, 0.7]],
                    "classes_target": [0, 1],
                },
            )
        self.assertIn("sum to one", str(ctx.exception))

    def test_binary_proba_does_not_require_sum_to_one(self):
        spec = task("binary_classification", "roc_auc", "maximize")
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
        # Positive class calibrated probabilities [0.1, 0.2, 0.8, 0.9] with arbitrary negative class column
        result = spec.evaluate_predictions(
            truth,
            {
                "target": [0, 0, 1, 1],
                "proba_target": [[0.0, 0.1], [0.0, 0.2], [0.0, 0.8], [0.0, 0.9]],
                "classes_target": [0, 1],
            },
        )
        self.assertEqual(result["primary_score"], 1.0)

    def test_native_categorical_transformer_edge_cases(self):
        from harness.models import NativeCategoricalTransformer
        df = pd.DataFrame({
            "num": [1.0, 2.5, 3.8],
            "int_cat": pd.Series([10, 20, 30], dtype="category"),
            "str_cat": pd.Series(["a", "b", "c"], dtype="category"),
            "mixed_obj": [1, "1", 2],
            "unhashable_obj": [[1, 2], [3], [1, 2]],
        })
        t = NativeCategoricalTransformer()
        out = t.fit_transform(df)
        # 1. Numeric untouched
        self.assertEqual(out["num"].tolist(), [1.0, 2.5, 3.8])
        # 2. Integer categories preserved (not converted to all-NaN!)
        self.assertEqual(out["int_cat"].tolist(), [10, 20, 30])
        # 3. String categories preserved
        self.assertEqual(out["str_cat"].tolist(), ["a", "b", "c"])
        # 4. Mixed object cleanly handled without duplicate categories error
        self.assertEqual(out["mixed_obj"].tolist(), ["1", "1", "2"])
        # 5. Unhashable objects handled without TypeError
        self.assertEqual(out["unhashable_obj"].tolist(), ["[1, 2]", "[3]", "[1, 2]"])

        # 6. Novel categories in transform become NaN gracefully
        test_df = pd.DataFrame({
            "num": [4.0],
            "int_cat": pd.Series([99], dtype="category"),
            "str_cat": pd.Series(["z"], dtype="category"),
            "mixed_obj": ["novel"],
            "unhashable_obj": ["[novel]"],
        })
        out_test = t.transform(test_df)
        self.assertTrue(pd.isna(out_test["int_cat"].iloc[0]))
        self.assertTrue(pd.isna(out_test["str_cat"].iloc[0]))
        self.assertTrue(pd.isna(out_test["mixed_obj"].iloc[0]))

    def test_contract_properties_and_incomplete(self):
        spec = task()
        self.assertEqual(spec.group_column, "group")
        self.assertEqual(spec.resolve_seed(), 13)
        self.assertIn("features", spec.inputs_dict)
        with self.assertRaises(ValueError):
            spec.evaluate_predictions(
                pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}), {"target": [0.0]}
            )

    def test_unsupported_metric(self):
        with self.assertRaises(ValueError):
            task(metric="nonsense").evaluate_predictions(
                pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}),
                {"target": [0.0, 1.0]},
            )


class MetricReferenceTests(unittest.TestCase):
    def test_regression_metrics_match_sklearn(self):
        from sklearn import metrics

        truth = pd.DataFrame({"id": [1, 2, 3], "target": [1.0, 2.0, 4.0]})
        predicted = [1.2, 1.8, 3.0]
        for name, expected, direction in [
            ("MAE", metrics.mean_absolute_error(truth.target, predicted), "minimize"),
            ("R_squared", metrics.r2_score(truth.target, predicted), "maximize"),
            (
                "RMSE",
                np.sqrt(metrics.mean_squared_error(truth.target, predicted)),
                "minimize",
            ),
        ]:
            result = task(metric=name, direction=direction).evaluate_predictions(
                truth, {"target": predicted}
            )
            self.assertAlmostEqual(result["primary_score"], expected)

    def test_declared_non_sorted_probability_order(self):
        truth = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
        result = task(
            "binary_classification", "logloss", "minimize"
        ).evaluate_predictions(
            truth,
            {
                "target": [0, 1],
                "proba_target": [[0.1, 0.9], [0.8, 0.2]],
                "classes_target": [1, 0],
            },
        )
        self.assertAlmostEqual(result["primary_score"], -np.log(0.9 * 0.8) / 2)

    def test_multi_target_missing_is_not_dropped(self):
        spec = task("binary_classification", "accuracy", "maximize")
        from harness.task_spec import SubtaskSpec

        spec.subtasks["second"] = SubtaskSpec(
            "second",
            "Second",
            "binary_classification",
            column="second",
            classes=[0, 1],
            primary_metric="accuracy",
        )
        truth = pd.DataFrame({"id": [1, 2], "target": [0, 1], "second": [1, 0]})
        with self.assertRaises(ValueError):
            spec.evaluate_predictions(truth, {"target": [0, 1]})
        result = spec.evaluate_predictions(truth, {"target": [0, 1], "second": [0, 0]})
        self.assertEqual(result["primary_score"], 0.75)
