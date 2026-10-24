"""Real subprocess output must retain its prediction identity and probability metadata."""

import copy
import json
import math
import shutil
import unittest
from types import SimpleNamespace

import pandas as pd

from harness.artifacts import validate_test_prediction
from harness.runner import UniversalRunner
from harness.script_predictions import align_prediction_ids, extract_test_predictions
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


PREDICTION_SCRIPT = '''
import argparse, json, os
from pathlib import Path
import pandas as pd
p = argparse.ArgumentParser(); p.add_argument('--exp-dir'); p.add_argument('--config'); args = p.parse_args()
root = Path(os.environ['HARNESS_PROJECT_ROOT'])
for split, filename in [('train', 'oof_predictions.csv'), ('test', 'test_predictions.csv')]:
    df = pd.read_csv(root / 'dataset' / (split + '.csv'))
    p1 = df.x.to_numpy()
    result = pd.DataFrame({'id': df.id, 'pred_target': (p1 > .5).astype(int),
        'proba_target': [json.dumps([v, 1-v]) for v in p1], 'classes_target': json.dumps([1, 0])})
    result.to_csv(Path(args.exp_dir) / filename, index=False)
'''


class ScriptPredictionTransportTests(unittest.TestCase):
    def make_runner(self, mode, metric="roc_auc", script=PREDICTION_SCRIPT):
        spec = task("binary_classification", metric, "minimize" if metric == "logloss" else "maximize")
        doc = {"task": copy.deepcopy(spec.metadata)}
        doc["task"]["data"].update(train_path="dataset/train.csv", test_path="dataset/test.csv")
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        frame = pd.DataFrame({"id": range(12), "group": range(12), "x": [.1, .9] * 6,
                              "category": ["a", "b"] * 6, "target": [0, 1] * 6})
        root, _ = make_workspace(task=doc, train_csv=frame.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        pd.DataFrame({"id": [100, 101], "x": [.1, .9], "category": ["a", "b"]}).to_csv(
            root / "dataset/test.csv", index=False)
        (root / "model.py").write_text(script, encoding="utf-8")
        cfg = {"experiment": {"id": "exp000_baseline", "branch": "baseline", "parent_id": None,
                              "approach": "custom_script", "problem_addressed": "Verify probability transport.",
                              "hypothesis": "The declared class order preserves the correct metric.",
                              "intervention": "Run the prediction adapter."},
               "execution": {"mode": mode, **({"entrypoint": "model.py"} if mode == "script"
                                               else {"pipeline": ["model.py"]})}}
        args = SimpleNamespace(project_root=str(root), task_config=None, debug=False, fold=None, probe_samples=None)
        return UniversalRunner(cfg, "config.yaml", args)

    def test_probability_metrics_and_test_labels_keep_declared_class_order(self):
        for mode in ("script", "pipeline"):
            for metric in ("roc_auc", "logloss"):
                with self.subTest(mode=mode, metric=metric):
                    runner = self.make_runner(mode, metric)
                    result = runner.run()
                    expected = 1.0 if metric == "roc_auc" else -math.log(.9)
                    self.assertAlmostEqual(result["primary_score"], expected)
                    oof = pd.read_csv(runner.exp_dir / "oof_predictions.csv")
                    self.assertEqual(json.loads(oof.classes_target.iloc[0]), [1, 0])
                    test = pd.read_csv(runner.exp_dir / "test_predictions.csv")
                    self.assertEqual(test.pred_target.tolist(), [0, 1])
                    self.assertEqual(json.loads(test.classes_target.iloc[0]), [1, 0])
                    validate_test_prediction(runner.exp_dir / "test_predictions.csv",
                                             task_spec=runner.task_spec, plugin=runner.task_plugin)

    def test_inconsistent_test_class_order_is_rejected(self):
        frame = pd.DataFrame({"classes_target": ["[0, 1]", "[1, 0]"]})
        with self.assertRaisesRegex(ValueError, "Class order changed"):
            extract_test_predictions(frame, task())

    def test_reordered_test_predictions_are_joined_by_id(self):
        script = PREDICTION_SCRIPT.replace("    p1 = df.x.to_numpy()", "    if split == 'test': df = df.iloc[::-1]\n    p1 = df.x.to_numpy()")
        for mode in ("script", "pipeline"):
            with self.subTest(mode=mode):
                runner = self.make_runner(mode, script=script)
                runner.run()
                output = pd.read_csv(runner.exp_dir / "test_predictions.csv")
                self.assertEqual(output.id.tolist(), [100, 101])
                self.assertEqual(output.pred_target.tolist(), [0, 1])
                self.assertEqual([json.loads(v)[0] for v in output.proba_target], [.1, .9])
                validate_test_prediction(runner.exp_dir / "test_predictions.csv",
                                         task_spec=runner.task_spec, plugin=runner.task_plugin)

    def test_invalid_prediction_id_sets_fail_closed(self):
        cases = [pd.DataFrame({"pred_target": [1, 2]}),
                 pd.DataFrame({"id": [1, 1]}), pd.DataFrame({"id": [1, None]}),
                 pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [1, 2, 3]}),
                 pd.DataFrame({"id": [1, 3]})]
        for frame in cases:
            with self.subTest(frame=frame.to_dict()):
                with self.assertRaises(ValueError):
                    align_prediction_ids(frame, [1, 2], "id")

    def test_duplicate_test_ids_cannot_be_sealed_as_completed_evidence(self):
        script = PREDICTION_SCRIPT.replace("    p1 = df.x.to_numpy()", "    if split == 'test': df['id'] = 100\n    p1 = df.x.to_numpy()")
        for mode in ("script", "pipeline"):
            with self.subTest(mode=mode):
                runner = self.make_runner(mode, script=script)
                with self.assertRaisesRegex(ValueError, "unique, non-null IDs"):
                    runner.run()
                self.assertEqual(runner.manifest["status"], "FAILED")
                with self.assertRaises(ValueError):
                    validate_test_prediction(runner.exp_dir / "test_predictions.csv",
                                             task_spec=runner.task_spec, plugin=runner.task_plugin)
