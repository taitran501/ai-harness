"""Integration tests for UniversalRunner runtime contracts, adapters, and prediction transport.

Consolidates:
- tests/test_correctness_runtime.py (RuntimeContracts)
- tests/test_runtime_extensions.py (FactoryContracts, RuntimeIOContracts)
- tests/test_script_adapter.py (TestScriptAdapterProtocol)
- tests/test_script_prediction_transport.py (ScriptPredictionTransportTests)
- tests/test_script_validation_scope.py (ScriptValidationScopeTests)
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from harness.artifacts import sha256, validate_run, validate_test_prediction
from harness.candidates import build_candidate, validate_candidate
from harness.models import GenericBaselineModel, call_model
from harness.runner import UniversalRunner
from harness.script_predictions import align_prediction_ids, extract_test_predictions
from tests.fixtures import make_workspace, task, text_task_document as task_document
from tools.agent_bridge_cli import _parse_json_arg, main as bridge_cli_main


PREDICTION_SCRIPT = """
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
"""


SCOPED_SCRIPT = """
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
"""


class RuntimeContracts(unittest.TestCase):
    def test_features_parameters_and_unknown_model(self):
        spec = task()
        model = GenericBaselineModel(spec, "ridge", alpha=0.25)
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "group": [1, 2, 3, 4],
                "x": [1, 2, 3, 4],
                "category": ["a", "b", "a", "b"],
                "target": [2.0, 4.0, 6.0, 8.0],
                "other_label": [0, 0, 1, 1],
            }
        )
        model.fit(frame, variant="features")
        self.assertEqual(model.estimator_metadata["target"]["params"]["alpha"], 0.25)
        self.assertEqual(
            model.estimator_metadata["target"]["features"], ["x", "category"]
        )
        self.assertEqual(len(model.predict(frame, "features")["target"]), 4)
        with self.assertRaises(ValueError):
            GenericBaselineModel(spec, "xgboost")

    def test_model_typeerror_is_not_retried(self):
        class Broken:
            calls = 0

            def fit(self, frame, variant=None):
                self.calls += 1
                raise TypeError("internal failure")

        model = Broken()
        with self.assertRaisesRegex(TypeError, "internal failure"):
            call_model(model, "fit", None, "x")
        self.assertEqual(model.calls, 1)

    def test_end_to_end_regression_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            spec = task()
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
            frame = pd.DataFrame(
                {
                    "id": range(12),
                    "group": range(12),
                    "x": range(12),
                    "category": ["a", "b"] * 6,
                    "target": [x * 2.0 for x in range(12)],
                }
            )
            frame.to_csv(root / "dataset/train.csv", index=False)
            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "exp001",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Establish verified regression baseline",
                    "hypothesis": "Linear features provide a measurable regression control",
                    "intervention": "Train a regularized linear baseline",
                },
                "model": {"name": "ridge", "params": {"alpha": 0.1}},
            }
            runner = UniversalRunner(cfg, "config.yaml", args)
            result = runner.run()
            self.assertEqual(len(result["fold_metrics"]), 2)
            self.assertEqual(result["n_samples"], 12)
            self.assertGreaterEqual(result["primary_score"], 0)
            self.assertEqual(result["utility"], -result["primary_score"])
            self.assertEqual(
                len(pd.read_csv(runner.exp_dir / "oof_predictions.csv")), 12
            )

            validate_run(
                runner.exp_dir, task_spec=runner.task_spec, plugin=runner.task_plugin
            )

            candidate = build_candidate(
                runner.project, "control", "single_model", {"source_exp": "exp001"}
            )
            validate_candidate(runner.project, candidate)
            with self.assertRaises((ValueError, FileNotFoundError)):
                build_candidate(
                    runner.project, "missing", "single_model", {"source_exp": "missing"}
                )

            original = sha256(runner.exp_dir / "metrics.json")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                UniversalRunner(cfg, "config.yaml", args)
            args.debug = True
            probe = UniversalRunner(cfg, "config.yaml", args)
            probe.run()
            self.assertEqual(original, sha256(runner.exp_dir / "metrics.json"))
            self.assertEqual(probe.run_kind, "probe")
            args.debug = False
            limited_cfg = {
                **cfg,
                "validation": {**cfg.get("validation", {}), "sample_limit": 2},
            }
            limited = UniversalRunner(limited_cfg, "config.yaml", args)
            self.assertEqual(limited.run_kind, "probe")
            with self.assertRaises(ValueError):
                validate_run(probe.exp_dir)
            (runner.exp_dir / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_run(runner.exp_dir)

    def test_metadata_feature_capability_is_wired(self):
        from harness.feature_pipeline import UniversalFeaturePipeline

        pipeline = UniversalFeaturePipeline(
            use_sparse_ngrams=False, use_tabular_metadata=True
        )
        texts = pd.Series(["one short text", "another longer text with words"])
        transformed = pipeline.fit(texts).transform(texts)
        self.assertEqual(transformed.shape[0], 2)
        self.assertGreater(transformed.shape[1], 0)

    def test_text_probabilities_and_variant_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            spec = task("binary_classification", "roc_auc", "maximize")
            doc = {"task": spec.metadata}
            doc["task"]["data"].pop("group_column")
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 3}
            doc["task"]["inputs"] = [
                {"name": "text", "type": "text", "source_columns": ["text"]},
                {"name": "context", "type": "text", "source_columns": ["context"]},
            ]
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
            pd.DataFrame(
                {
                    "id": range(24),
                    "target": [0, 1] * 12,
                    "text": ["cold negative", "warm positive"] * 12,
                    "context": ["sun bright", "rain dark"] * 12,
                }
            ).to_csv(root / "dataset/train.csv", index=False)
            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "text",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Establish text probability baseline",
                    "hypothesis": "Text features distinguish declared classes",
                    "intervention": "Fit logistic text estimator",
                },
                "model": {"name": "baseline"},
                "data": {
                    "input_variants": ["text", "context"],
                    "selection_variant": "context",
                },
            }
            runner = UniversalRunner(cfg, "config", args)
            result = runner.run()
            self.assertEqual(len(result["fold_metrics"]), 6)
            self.assertEqual(set(result["variants"]), {"text", "context"})
            self.assertEqual(result["n_samples"], 24)
            self.assertEqual(
                result["primary_score"], result["variants"]["context"]["primary_score"]
            )
            from harness.tree_journal import TreeJournal

            journal = TreeJournal(
                journal_path=runner.artifacts_dir / "journal.json",
                task_spec=runner.task_spec,
            )
            self.assertEqual(
                journal.get_best_node().get_score(), result["primary_score"]
            )

    def test_llm_false_label_and_invalid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            spec = task("binary_classification", "accuracy", "maximize")
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "holdout", "holdout_ratio": 0.5}
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
            pd.DataFrame(
                {"id": range(8), "group": range(8), "x": range(8), "target": [0] * 8}
            ).to_csv(root / "dataset/train.csv", index=False)
            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {"id": "llm", "approach": "llm_prompt", "branch": "baseline", "parent_id": None,
                               "problem_addressed": "Verify LLM output contracts", "hypothesis": "Declared labels are validated",
                               "intervention": "Use mock completion"},
                "llm": {
                    "prompt_template": "{x}",
                    "model_name": "fixture",
                    "provider": "openai",
                },
            }
            runner = UniversalRunner(cfg, "config", args)
            with patch(
                "harness.llm_engine.LLMEngine.generate_completion",
                return_value={"text": '{"prediction":0}'},
            ) as complete:
                frame = pd.read_csv(root / "dataset/train.csv")
                result = runner.run()
            self.assertEqual(result["primary_score"], 1)
            self.assertEqual(complete.call_args.kwargs["model_name"], "fixture")
            for response in ("{}", "not json", '{"prediction":"invalid"}'):
                cfg["experiment"]["id"] = "invalid_" + str(len(response))
                broken = UniversalRunner(cfg, "config", args)
                with patch(
                    "harness.llm_engine.LLMEngine.generate_completion",
                    return_value={"text": response},
                ):
                    with self.assertRaises(ValueError):
                        broken._run_llm_prompting(frame, "test")


class FactoryContracts(unittest.TestCase):
    def setUp(self):
        self.runner = UniversalRunner.__new__(UniversalRunner)
        self.runner.task_spec = object()
        self.runner.plugin_context = object()
        self.runner.cfg = {"model": {"factory": "factory_fixture.build"}}

    def invoke(self, fn, **params):
        module = ModuleType("factory_fixture")
        module.build = fn
        with patch.dict(sys.modules, {"factory_fixture": module}):
            return self.runner._get_model_factory()("custom", **params)

    def test_explicit_task_plugin_and_keyword_context(self):
        def task_factory(task_spec, name, **params):
            return task_spec, name, params
        def plugin_factory(plugin_context, name, **params):
            return plugin_context, name, params
        def keyword_factory(name, *, task_spec, **params):
            return task_spec, name, params
        self.assertEqual(self.invoke(task_factory, C=2), (self.runner.task_spec, "custom", {"C": 2}))
        self.assertEqual(self.invoke(plugin_factory)[0], self.runner.plugin_context)
        self.assertEqual(self.invoke(keyword_factory)[0], self.runner.task_spec)

    def test_legacy_and_configured_context_name(self):
        def legacy(name, **params):
            return name, params
        self.assertEqual(self.invoke(legacy, C=3), ("custom", {"C": 3}))
        def named_context(ctx, name, **params):
            return ctx, name, params
        self.runner.cfg["model"]["factory_context"] = "plugin_context"
        self.assertEqual(self.invoke(named_context)[0], self.runner.plugin_context)
        self.runner.cfg["model"]["factory_context"] = "none"
        self.assertEqual(self.invoke(legacy), ("custom", {}))

    def test_factory_internal_typeerror_is_not_retried_and_context_cannot_be_overridden(self):
        calls = []
        def broken(task_spec, name, **params):
            calls.append(task_spec)
            raise TypeError("factory implementation error")
        with self.assertRaisesRegex(TypeError, "implementation error"):
            self.invoke(broken)
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, "override injected"):
            self.invoke(broken, task_spec="wrong task")
        self.assertEqual(len(calls), 1)


class RuntimeIOContracts(unittest.TestCase):
    def test_param_file_preserves_quotes_unicode_and_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()), encoding="utf-8")
            params = {
                "summary": 'Bài học: "class weighting" còn chưa rõ.',
                "next_action": 'Kiểm tra "minority" slices',
                "open_questions": ["Label noise?"],
            }
            (root / "checkpoint.json").write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8-sig")
            output = io.StringIO()
            with patch.dict(os.environ), contextlib.redirect_stdout(output):
                self.assertEqual(
                    bridge_cli_main([
                        "--project-root", str(root),
                        "act",
                        "--action", "record_research_checkpoint",
                        "--param-file", "checkpoint.json",
                    ]),
                    0,
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["observation"]["status"], "SUCCESS")
            self.assertEqual(payload["observation"]["data"]["conclusion"], params["summary"])

    def test_invalid_json_object_and_mutually_exclusive_parameters(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for raw in ("[]", "42", "not json"):
                with self.assertRaises(SystemExit):
                    _parse_json_arg(raw)
            with self.assertRaises(SystemExit):
                bridge_cli_main(["act", "--action", "search_research_memory", "--params", "{}", "--param-file", "p.json"])

    def test_progress_is_visible_in_redirected_file_before_process_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout_path = root / "stdout.log"
            script = "\n".join([
                "from pathlib import Path",
                "from harness.runner import UniversalRunner",
                "import sys",
                "runner = UniversalRunner.__new__(UniversalRunner)",
                "runner.exp_id = 'progress'",
                "runner.exp_dir = Path(sys.argv[1])",
                "runner._progress('Fold 0 started')",
                "sys.stdin.readline()",
            ])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)
            with stdout_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-c", script, str(root)],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                try:
                    deadline = time.monotonic() + 12
                    while "Fold 0 started" not in stdout_path.read_text(encoding="utf-8") and time.monotonic() < deadline:
                        if process.poll() is not None:
                            break
                        time.sleep(0.02)
                    self.assertIsNone(process.poll())
                    self.assertIn("Fold 0 started", stdout_path.read_text(encoding="utf-8"))
                    self.assertIn("Fold 0 started", (root / "run.log").read_text(encoding="utf-8"))
                finally:
                    process.communicate("\n", timeout=10)


class TestScriptAdapterProtocol(unittest.TestCase):
    def test_script_adapter_success_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            (root / "scripts").mkdir()

            spec = task()
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

            n = 12
            frame = pd.DataFrame({
                "id": range(n),
                "group": range(n),
                "x": range(n),
                "category": ["a", "b"] * 6,
                "target": [float(x * 2.0) for x in range(n)],
            })
            frame.to_csv(root / "dataset/train.csv", index=False)

            script_code = """
import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

project_root = os.environ.get("HARNESS_PROJECT_ROOT", ".")
train_df = pd.read_csv(os.path.join(project_root, "dataset", "train.csv"))

oof = train_df[["id"]].copy()
oof["pred_target"] = train_df["target"].values + 0.1
oof["fold"] = 0
oof.to_csv(os.path.join(args.exp_dir, "oof_predictions.csv"), index=False)
print("Dummy script finished successfully")
"""
            script_file = root / "scripts" / "dummy_train.py"
            script_file.write_text(script_code, encoding="utf-8")

            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "exp000_baseline_script",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Establish verified script-adapter baseline anchor",
                    "hypothesis": "Custom python script adapter integrates cleanly with harness gates",
                    "intervention": "Execute external training script producing valid OOF predictions",
                },
                "execution": {
                    "mode": "script",
                    "entrypoint": "scripts/dummy_train.py",
                },
            }

            runner = UniversalRunner(cfg, "config.yaml", args)
            result = runner.run()

            self.assertEqual(runner.manifest["status"], "COMPLETED")
            self.assertEqual(result["experiment_id"], "exp000_baseline_script")
            self.assertIn("primary_score", result)
            self.assertEqual(result["n_samples"], 12)
            self.assertTrue((runner.exp_dir / "artifact_manifest.json").exists())
            self.assertTrue((runner.exp_dir / "metrics.json").exists())
            self.assertTrue((runner.exp_dir / "oof_predictions.csv").exists())

    def test_script_adapter_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            (root / "scripts").mkdir()

            spec = task()
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

            frame = pd.DataFrame({
                "id": range(6),
                "group": range(6),
                "x": range(6),
                "category": ["a", "b"] * 3,
                "target": [float(x) for x in range(6)],
            })
            frame.to_csv(root / "dataset/train.csv", index=False)

            broken_script = """
import sys
sys.stderr.write("FATAL ERROR: Missing GPU device\\n")
sys.exit(42)
"""
            (root / "scripts" / "fail.py").write_text(broken_script, encoding="utf-8")

            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "exp_baseline_fail",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Test custom script failure detection in baseline mode",
                    "hypothesis": "Subprocess failure is detected and reported accurately",
                    "intervention": "Execute failing python script",
                },
                "execution": {
                    "mode": "script",
                    "entrypoint": "scripts/fail.py",
                },
            }

            runner = UniversalRunner(cfg, "config.yaml", args)
            with self.assertRaises(RuntimeError) as ctx:
                runner.run()
            self.assertIn("FATAL ERROR", str(ctx.exception))


class ScriptPredictionTransportTests(unittest.TestCase):
    def make_runner(self, mode, metric="roc_auc", script=PREDICTION_SCRIPT):
        spec = task("binary_classification", metric, "minimize" if metric == "logloss" else "maximize")
        doc = {"task": copy.deepcopy(spec.metadata)}
        doc["task"]["subtasks"][0]["classes"] = [1, 0]
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
        frame = pd.DataFrame({"id": [1, 2], "classes_target": ["[0, 1]", "[1, 0]"]})
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
                    scope = json.loads((runner.exp_dir / "validation_partitions.json").read_text(encoding="utf-8"))
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
                    "records = []", "records = [{'id': i, 'pred_target': 0.0} for i in range(100, 102)]"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "OOF prediction IDs"):
                    runner.run()
                self.assertEqual(runner.manifest["status"], "FAILED")

    def test_configured_fold_limits_full_script_manifest(self):
        runner = self.make_runner("script")
        runner.cfg["data"] = {"folds": [0]}
        result = runner.run()
        self.assertEqual(result["n_samples"], 6)
        scope = json.loads((runner.exp_dir / "validation_partitions.json").read_text(encoding="utf-8"))
        self.assertEqual([p["fold"] for p in scope["partitions"]], [0])

    def test_cli_fold_overrides_configured_folds(self):
        runner = self.make_runner("script", fold=0, debug=True)
        runner.cfg["data"] = {"folds": [1]}
        runner.run()
        scope = json.loads((runner.exp_dir / "validation_partitions.json").read_text(encoding="utf-8"))
        self.assertEqual([p["fold"] for p in scope["partitions"]], [0])

    def test_invalid_configured_folds_fail(self):
        for value in ([], [0, 0], [99], "0"):
            with self.subTest(value=value):
                runner = self.make_runner("script")
                runner.cfg["data"] = {"folds": value}
                with self.assertRaisesRegex(ValueError, "fold"):
                    runner._resolve_validation_splits(runner._load_training_data(), "kfold")


class TestRunnerUsabilityExtensions(unittest.TestCase):
    def test_oof_probability_float_decomposition(self):
        from unittest.mock import MagicMock
        from harness.runner import UniversalRunner

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
        rec0 = records[0]
        self.assertEqual(rec0["id"], 101)
        self.assertIn("proba_label", rec0)
        self.assertIsInstance(rec0["proba_label"], str)
        self.assertEqual(rec0["proba_label_class_A"], 0.2)
        self.assertEqual(rec0["proba_label_class_B"], 0.8)
        self.assertEqual(rec0["proba_label_1"], 0.8)

    def test_runner_split_loading_aliases_and_errors(self):
        import argparse
        from harness.runner import Runner

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir(parents=True)
            task_yaml = cfg_dir / "task.yaml"
            task_yaml.write_text(
                """
task:
  name: SplitTest
  id: split_test
  id_column: row_id
  primary_metric: accuracy
  validation:
    split_file: splits.csv
  subtasks:
    y:
      task_type: binary_classification
      column: y
      classes: [0, 1]
  inputs:
    raw:
      type: tabular
      source_columns: [x]
""",
                encoding="utf-8",
            )
            train_df = pd.DataFrame({"row_id": [101, 102, 103, 104], "x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
            train_df.to_csv(root / "train.csv", index=False)

            splits_df = pd.DataFrame({"ROW_ID": [101, 102, 103, 104], "kfold": [0, 1, 0, 1]})
            splits_df.to_csv(root / "splits.csv", index=False)

            exp_cfg = {
                "experiment": {"approach": "ml_classical"},
                "model": {"name": "baseline"},
                "input_variant": "raw",
                "paths": {"train_path": "train.csv", "split_path": "splits.csv"},
            }
            exp_file = cfg_dir / "exp001.yaml"
            exp_file.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                project_root=str(root),
                task_config=str(task_yaml),
                debug=False,
                probe_samples=None,
                fold=None,
                job_id="test_run",
                force_import=False,
            )
            runner = Runner(exp_cfg, exp_file, args)
            loaded = runner._load_training_data()
            self.assertIn("fold", loaded.columns)
            self.assertEqual(loaded["fold"].tolist(), [0, 1, 0, 1])

            bad_splits = pd.DataFrame({"row_id": [101, 102], "fold": [0, 1]})
            bad_splits.to_csv(root / "bad_splits.csv", index=False)
            exp_cfg_bad = dict(exp_cfg, paths={"train_path": "train.csv", "split_path": "bad_splits.csv"})
            exp_bad_file = cfg_dir / "exp002.yaml"
            exp_bad_file.write_text("{}", encoding="utf-8")
            runner_bad = Runner(exp_cfg_bad, exp_bad_file, args)
            with self.assertRaises(ValueError) as ctx:
                runner_bad._load_training_data()
            self.assertIn("does not cover 2/4 training IDs", str(ctx.exception))

            no_fold_splits = pd.DataFrame({"row_id": [101, 102, 103, 104], "category": ["a", "b", "c", "d"]})
            no_fold_splits.to_csv(root / "no_fold.csv", index=False)
            exp_cfg_nofold = dict(exp_cfg, paths={"train_path": "train.csv", "split_path": "no_fold.csv"})
            exp_nofold_file = cfg_dir / "exp003.yaml"
            exp_nofold_file.write_text("{}", encoding="utf-8")
            runner_nofold = Runner(exp_cfg_nofold, exp_nofold_file, args)
            with self.assertRaises(ValueError) as ctx:
                runner_nofold._load_training_data()
            self.assertIn("missing required columns", str(ctx.exception))

    def test_runner_time_series_strategy(self):
        import argparse
        from harness.runner import Runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_df = pd.DataFrame({"id": list(range(20)), "feature": list(range(20)), "target": [0, 1] * 10})
            train_df.to_csv(root / "train.csv", index=False)

            task_yaml = root / "task.yaml"
            task_dict = {
                "task": {
                    "name": "ts_task",
                    "task_type": "binary_classification",
                    "primary_metric": "macro_f1",
                    "id_column": "id",
                    "data": {"train_path": "train.csv"},
                    "inputs": [{"name": "feat", "type": "tabular", "source_columns": ["feature"]}],
                    "subtasks": {"target": {"task_type": "binary_classification", "classes": [0, 1]}},
                    "validation": {"strategy": "time_series", "num_folds": 3},
                }
            }
            task_yaml.write_text(yaml.safe_dump(task_dict), encoding="utf-8")

            exp_yaml = root / "exp.yaml"
            exp_dict = {
                "task": {"name": "ts_task"},
                "model": {"name": "random_forest"},
                "validation": {"strategy": "time_series", "num_folds": 3},
            }
            exp_yaml.write_text(yaml.safe_dump(exp_dict), encoding="utf-8")

            args = argparse.Namespace(
                project_root=str(root),
                task_config=str(task_yaml),
                debug=False,
                probe_samples=None,
                fold=None,
                job_id="test_ts",
                force_import=False,
            )
            runner = Runner(exp_dict, exp_yaml, args)
            splits = runner._resolve_validation_splits(train_df, "time_series")
            self.assertEqual(len(splits), 3)
            for fold, tr, val in splits:
                self.assertFalse(tr.empty)
                self.assertFalse(val.empty)
                self.assertEqual(len(set(tr["id"]).intersection(set(val["id"]))), 0)


if __name__ == "__main__":
    unittest.main()
