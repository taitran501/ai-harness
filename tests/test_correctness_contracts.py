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
