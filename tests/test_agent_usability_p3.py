import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge, TypedActionName
from tools.agent_bridge_cli import _parse_json_arg, _parse_dynamic_kv_args
from harness.kaggle_notebook_builder import build_kaggle_submission_notebook
from harness.templates.kaggle_template import KaggleKernelManager


class AgentUsabilityP3Tests(unittest.TestCase):
    def test_json_parsing_resilience_powershell_compat(self):
        # Standard double-quoted JSON
        self.assertEqual(_parse_json_arg('{"branch": "feat", "num": 1}'), {"branch": "feat", "num": 1})
        # Single-quoted Python/YAML style (common PowerShell output)
        self.assertEqual(_parse_json_arg("{'branch': 'feat', 'num': 2}"), {"branch": "feat", "num": 2})
        # Unquoted keys (YAML style)
        self.assertEqual(_parse_json_arg("{branch: feat, num: 3}"), {"branch": "feat", "num": 3})

    def test_dynamic_cli_kv_arguments(self):
        args = ["--branch", "feat", "--descriptor", "v1", "--weight", "0.75", "--flag", "--config-path", "configs/test.yaml"]
        res = _parse_dynamic_kv_args(args)
        self.assertEqual(res["branch"], "feat")
        self.assertEqual(res["descriptor"], "v1")
        self.assertEqual(res["weight"], 0.75)
        self.assertEqual(res["flag"], True)
        self.assertEqual(res["config_path"], "configs/test.yaml")

    def test_get_next_experiment_num_and_collision_prevention(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            configs_dir = root / "configs"
            configs_dir.mkdir(parents=True)
            # Create task.yaml
            task_yaml = root / "configs" / "task.yaml"
            task_yaml.write_text(
                "task:\n  name: TestTask\n  id: test_task\n  primary_metric: roc_auc\n  data:\n    train: train.csv\n",
                encoding="utf-8"
            )
            # Create train.csv
            (root / "train.csv").write_text("id,text,target\n1,a,0\n2,b,1\n", encoding="utf-8")

            bridge = AgentBridge(root, task_spec_path=task_yaml)
            self.assertEqual(bridge.get_next_experiment_num(), 1)

            # Create exp001.yaml in configs
            (configs_dir / "exp001_baseline.yaml").write_text("dummy", encoding="utf-8")
            self.assertEqual(bridge.get_next_experiment_num(), 2)

            # Simulate existing artifact folder for exp002 in .ai-harness
            art_dir = root / ".ai-harness" / "artifacts" / "test_task" / "exp002_feat_v1"
            art_dir.mkdir(parents=True)
            # Next num must jump to 3 because exp002 already has artifacts!
            self.assertEqual(bridge.get_next_experiment_num(), 3)

            # Test action get_next_experiment_id
            sess = bridge.start_session(goal="Test next id")
            sess, obs = bridge.execute_action(
                action=TypedActionName.GET_NEXT_EXPERIMENT_ID.value,
                params={"branch": "model", "descriptor": "lgbm"},
                session_id=sess.session_id,
            )
            self.assertEqual(obs.status, "SUCCESS")
            self.assertEqual(obs.data["next_num"], 3)
            self.assertEqual(obs.data["suggested_exp_id"], "exp003_model_lgbm")

            # Test collision prevention: trying to create_experiment with exp_id that already has artifacts
            sess, obs_collide = bridge.execute_action(
                action=TypedActionName.CREATE_EXPERIMENT.value,
                params={"exp_id": "exp002_feat_v1", "branch": "feat", "descriptor": "v1"},
                session_id=sess.session_id,
            )
            self.assertEqual(obs_collide.status, "FAILED")
            self.assertIn("already exist", obs_collide.summary)
            self.assertIn("exp003", obs_collide.summary)

    def test_kaggle_submission_notebook_builder(self):
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td) / "exp001_baseline"
            exp_dir.mkdir(parents=True)
            nb_out = Path(td) / "submission.ipynb"

            res = build_kaggle_submission_notebook(
                exp_dir=exp_dir,
                output_path=nb_out,
                id_column="id",
                target_column="satisfaction",
            )
            self.assertEqual(res["status"], "SUCCESS")
            self.assertTrue(nb_out.exists())

            nb_data = json.loads(nb_out.read_text(encoding="utf-8"))
            self.assertEqual(nb_data["nbformat"], 4)
            self.assertGreaterEqual(len(nb_data["cells"]), 5)
            # Verify code contains offline wheel and submission checks
            cell_sources = ["".join(c["source"]) for c in nb_data["cells"]]
            self.assertTrue(any("offline_wheels" in s or "pip install" in s for s in cell_sources))
            self.assertTrue(any("submission.csv" in s for s in cell_sources))

    def test_kaggle_license_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "task.yaml").write_text(
                "task:\n  name: TestKaggle\n  id: test_kaggle\n  data:\n    train: train.csv\n",
                encoding="utf-8"
            )
            (root / "dataset").mkdir()
            (root / "dataset" / "train.csv").write_text("id,val\n1,10\n", encoding="utf-8")
            from harness.project import attach_project
            attach_project(project_root=root, task_config=cfg_dir / "task.yaml", force=True)

            # When license is not in task.yaml, KaggleKernelManager should not fail with license error
            res = KaggleKernelManager.push_dataset_to_kaggle(project_root=root)
            # Since consent is not granted, it should fail on consent, NOT on license!
            self.assertEqual(res["success"], False)
            self.assertIn("consent", res["error"].lower())
            self.assertNotIn("license", res["error"].lower())

    def test_oof_probability_float_decomposition(self):
        from harness.runner import UniversalRunner
        from unittest.mock import MagicMock

        runner = UniversalRunner.__new__(UniversalRunner)
        runner.task_spec = MagicMock()
        runner.task_spec.id_column = "id"

        dev_df = pd.DataFrame({"id": [101, 102]})
        preds = {
            "proba_label": np.array([[0.2, 0.8], [0.6, 0.4]]),
            "classes_label": ["class_A", "class_B"],
        }
        records = runner._record_predictions(dev_df, preds, 0, "default")
        self.assertEqual(len(records), 2)
        # Check first record has both json string and decomposed floats
        rec0 = records[0]
        self.assertEqual(rec0["id"], 101)
        self.assertIn("proba_label", rec0)
        self.assertIsInstance(rec0["proba_label"], str)  # legacy JSON string
        self.assertEqual(rec0["proba_label_class_A"], 0.2)
        self.assertEqual(rec0["proba_label_class_B"], 0.8)
        self.assertEqual(rec0["proba_label_1"], 0.8)  # binary convenience float


if __name__ == "__main__":
    unittest.main()
