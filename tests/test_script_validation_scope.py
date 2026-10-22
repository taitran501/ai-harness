"""Exercise the same child script under full CV, probes, holdout and temporal CV."""

import copy
import json
import shutil
import unittest
from types import SimpleNamespace

import pandas as pd

from harness.artifacts import validate_run
from harness.runner import UniversalRunner
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


SCOPED_SCRIPT = '''
import argparse, os
from pathlib import Path
import pandas as pd
from harness.script_validation import load_validation_manifest
p = argparse.ArgumentParser(); p.add_argument('--exp-dir'); p.add_argument('--config'); a = p.parse_args()
root = Path(os.environ['HARNESS_PROJECT_ROOT'])
df = pd.read_csv(root / 'dataset/train.csv')
if (root / 'dataset/dev.csv').exists():
    df = pd.concat([df, pd.read_csv(root / 'dataset/dev.csv')])
scope = load_validation_manifest()
records = []
for part in scope['partitions']:
    train = df.set_index('id').loc[part['train_ids']]
    dev = df.set_index('id').loc[part['validation_ids']]
    assert len(train) > 0 and len(dev) > 0
    if scope['sample_limit'] is not None:
        assert max(len(train), len(dev)) <= scope['sample_limit']
    value = float(train.target.mean())
    records.extend({'id': sample, 'pred_target': value} for sample in dev.index)
pd.DataFrame(records).to_csv(Path(a.exp_dir) / 'oof_predictions.csv', index=False)
'''


class ScriptValidationScopeTests(unittest.TestCase):
    def make_runner(self, mode, strategy="kfold", *, fold=None, debug=False, limit=None, preset=False):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": strategy, "num_folds": 2}
        if strategy == "holdout":
            doc["task"]["validation"]["holdout_ratio"] = .25
        if strategy == "external_holdout":
            doc["task"]["validation"] = {"strategy": "holdout", "dev_split_name": "dev"}
            doc["task"]["data"]["dev_path"] = "dataset/dev.csv"
        train = pd.DataFrame({"id": range(12), "group": range(12), "x": range(12),
                              "category": ["a", "b"] * 6, "target": [2.0 * i for i in range(12)]})
        if preset:
            train["fold"] = [0, 1] * 6
        root, _ = make_workspace(task=doc, train_csv=train.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        if strategy == "external_holdout":
            dev = train.iloc[:3].copy()
            dev["id"] += 100
            dev["group"] += 100
            dev.to_csv(root / "dataset/dev.csv", index=False)
        (root / "model.py").write_text(SCOPED_SCRIPT, encoding="utf-8")
        cfg = {"experiment": {"id": "exp000_scope", "branch": "baseline", "parent_id": None,
                              "approach": "custom_script", "problem_addressed": "Verify execution scope.",
                              "hypothesis": "Only selected validation partitions are evaluated.",
                              "intervention": "Use the declared partition manifest."},
               "execution": {"mode": mode, **({"entrypoint": "model.py"} if mode == "script"
                                               else {"pipeline": ["model.py"]})}}
        args = SimpleNamespace(project_root=str(root), task_config=None, fold=fold, debug=debug, probe_samples=limit)
        return UniversalRunner(cfg, "config.yaml", args)

    def test_full_and_probe_scopes_match_the_selected_partitions(self):
        cases = [("kfold", {}, 12), ("kfold", {"fold": 0, "debug": True}, 6),
                 ("kfold", {"fold": 0, "debug": True, "preset": True}, 6),
                 ("kfold", {"fold": 0, "limit": 2}, 2), ("holdout", {}, 3),
                 ("external_holdout", {}, 3), ("time_series", {}, 8)]
        for mode in ("script", "pipeline"):
            for strategy, options, expected_rows in cases:
                with self.subTest(mode=mode, strategy=strategy, options=options):
                    runner = self.make_runner(mode, strategy, **options)
                    result = runner.run()
                    self.assertEqual(result["n_samples"], expected_rows)
                    self.assertEqual(result["fold_metrics"][0]["dev_rows"], expected_rows)
                    scope = json.loads((runner.exp_dir / "validation_partitions.json").read_text())
                    oof = pd.read_csv(runner.exp_dir / "oof_predictions.csv")
                    self.assertEqual(set(oof.id), {i for p in scope["partitions"] for i in p["validation_ids"]})
                    if "fold" in options:
                        self.assertEqual(oof.fold.unique().tolist(), [0])
                    if runner.run_kind == "full":
                        validate_run(runner.exp_dir, task_spec=runner.task_spec, plugin=runner.task_plugin)
                    else:
                        self.assertFalse((runner.artifacts_dir / runner.exp_id).exists())

    def test_a_child_cannot_return_full_cv_as_a_single_fold_probe(self):
        for mode in ("script", "pipeline"):
            with self.subTest(mode=mode):
                runner = self.make_runner(mode, fold=0, debug=True)
                (runner.project_root / "model.py").write_text(SCOPED_SCRIPT.replace(
                    "records = []", "records = [{'id': i, 'pred_target': 0.0} for i in range(100, 102)]"))
                with self.assertRaisesRegex(ValueError, "OOF prediction IDs"):
                    runner.run()
                self.assertEqual(runner.manifest["status"], "FAILED")
