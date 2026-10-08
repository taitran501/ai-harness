import unittest
import numpy as np
import pandas as pd
from harness.task_spec import UniversalTaskSpec


def task(kind="regression", metric="rmse", direction="minimize"):
    return UniversalTaskSpec.from_dict({"task": {"name":"fixture", "modality":"tabular", "primary_metric":metric,
        "direction":direction, "data":{"id_column":"id", "group_column":"group", "seed":13},
        "inputs":[{"name":"features","type":"tabular","source_columns":["x","category"]}],
        "subtasks":[{"id":"target","column":"target","task_type":kind,"primary_metric":metric,"direction":direction,
                      "classes":[] if kind=="regression" else ["0","1"]}]}})


class EvaluationContracts(unittest.TestCase):
    def test_raw_regression_and_utility(self):
        result=task().evaluate_predictions(pd.DataFrame({"id":[1,2],"target":[0.,1.]}),{"target":[0.,3.]})
        self.assertAlmostEqual(result["primary_score"],np.sqrt(2))
        self.assertAlmostEqual(result["utility"],-np.sqrt(2))

    def test_probability_metric(self):
        spec=task("binary_classification","ROC-AUC","maximize")
        truth=pd.DataFrame({"id":[1,2,3,4],"target":[0,0,1,1]})
        with self.assertRaises(ValueError): spec.evaluate_predictions(truth,{"target":[0,0,1,1]})
        result=spec.evaluate_predictions(truth,{"target":[0,0,1,1],"proba_target":[[.9,.1],[.8,.2],[.1,.9],[.2,.8]],"classes_target":[0,1]})
        self.assertEqual(result["primary_score"],1)

    def test_contract_properties_and_incomplete(self):
        spec=task()
        self.assertEqual(spec.group_column,"group")
        self.assertEqual(spec.resolve_seed(),13)
        self.assertIn("features",spec.inputs_dict)
        with self.assertRaises(ValueError): spec.evaluate_predictions(pd.DataFrame({"id":[1,2],"target":[0.,1.]}),{"target":[0.]})

    def test_unsupported_metric(self):
        with self.assertRaises(ValueError): task(metric="nonsense").evaluate_predictions(pd.DataFrame({"id":[1,2],"target":[0.,1.]}),{"target":[0.,1.]})
