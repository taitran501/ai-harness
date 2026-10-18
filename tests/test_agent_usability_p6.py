"""Unit tests for Phase 6 agent usability enhancements.

Covers:
1. Built-in RandomForestModel wrapper (create_model, classification, regression, proba, num_parameters).
2. Extended model aliases in GenericBaselineModel (logreg, logistic, linear, linear_regression, lasso).
3. TreeJournal.load flexibility: supports instance.load(), class invocation TreeJournal.load(path), and TreeJournal.load_from_file(path).
4. Cross-environment / portable dataset path resolution in UniversalTaskSpec.load_dataset_split (resolving nonexistent foreign absolute paths to local basename).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.models import GenericBaselineModel, RandomForestModel, create_model
from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import TreeJournal


class TestAgentUsabilityP6(unittest.TestCase):
    """Test suite covering usability enhancements in Phase 6."""

    def test_random_forest_classification_and_regression(self):
        """Verify RandomForestModel fits, predicts probabilities/classes, and counts parameters."""
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
        """Verify GenericBaselineModel accepts extended aliases (logreg, linear_regression, lasso)."""
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

    def test_tree_journal_load_flexibility(self):
        """Verify TreeJournal.load supports instance call, class call, and load_from_file."""
        with tempfile.TemporaryDirectory() as td:
            j_path = Path(td) / "journal.json"
            dummy_data = {
                "nodes": [
                    {
                        "id": "exp001_baseline",
                        "status": "COMPLETED",
                        "metrics": {"primary_score": 0.85},
                    }
                ]
            }
            j_path.write_text(json.dumps(dummy_data), encoding="utf-8")

            # 1. Instance call
            j1 = TreeJournal(storage_path=j_path)
            self.assertIn("exp001_baseline", j1.nodes)
            j1.nodes.clear()
            self.assertEqual(len(j1.nodes), 0)
            j1.load()
            self.assertIn("exp001_baseline", j1.nodes)

            # 2. Class call TreeJournal.load(str_path)
            j2 = TreeJournal.load(str(j_path))
            self.assertIsInstance(j2, TreeJournal)
            self.assertIn("exp001_baseline", j2.nodes)

            # 3. Classmethod TreeJournal.load_from_file(path)
            j3 = TreeJournal.load_from_file(j_path)
            self.assertIsInstance(j3, TreeJournal)
            self.assertIn("exp001_baseline", j3.nodes)

    def test_cross_environment_dataset_path_fallback(self):
        """Verify load_dataset_split resolves basename if a foreign absolute path is missing."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create a local train.csv in root/dataset/train.csv
            ds_dir = root / "dataset"
            ds_dir.mkdir()
            local_train = ds_dir / "train.csv"
            local_train.write_text("id,val\n1,10\n2,20\n", encoding="utf-8")

            # Declare an absolute foreign path that does not exist on this machine
            foreign_abs_path = "/kaggle/working/dataset/train.csv"
            spec = UniversalTaskSpec.from_dict({
                "task": {
                    "id_column": "id",
                    "primary_metric": "accuracy",
                    "data": {
                        "train": foreign_abs_path,
                    },
                }
            })

            # Must resolve to local root/dataset/train.csv via portable fallback
            df = spec.load_dataset_split("train", project_root=root)
            self.assertEqual(len(df), 2)
            self.assertEqual(df["id"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
