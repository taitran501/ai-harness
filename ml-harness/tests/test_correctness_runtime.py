import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import yaml
from harness.models import GenericBaselineModel, call_model
from harness.runner import UniversalRunner
from tests.test_correctness_contracts import task


class RuntimeContracts(unittest.TestCase):
    def test_features_parameters_and_unknown_model(self):
        spec=task(); model=GenericBaselineModel(spec,"ridge",alpha=.25)
        frame=pd.DataFrame({"id":[1,2,3,4],"group":[1,2,3,4],"x":[1,2,3,4],"category":["a","b","a","b"],"target":[2.,4.,6.,8.],"other_label":[0,0,1,1]})
        model.fit(frame,variant="features")
        self.assertEqual(model.estimator_metadata["target"]["params"]["alpha"],.25)
        self.assertEqual(model.estimator_metadata["target"]["features"],["x","category"])
        self.assertEqual(len(model.predict(frame,"features")["target"]),4)
        with self.assertRaises(ValueError): GenericBaselineModel(spec,"xgboost")

    def test_model_typeerror_is_not_retried(self):
        class Broken:
            calls=0
            def fit(self, frame, variant=None):
                self.calls+=1; raise TypeError("internal failure")
        model=Broken()
        with self.assertRaisesRegex(TypeError,"internal failure"): call_model(model,"fit",None,"x")
        self.assertEqual(model.calls,1)

    def test_end_to_end_regression_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"configs").mkdir(); (root/"dataset").mkdir()
            spec=task(); doc={"task":spec.metadata}; doc["task"]["data"]["train_path"]="dataset/train.csv"
            doc["task"]["validation"]={"strategy":"kfold","num_folds":2}
            (root/"configs/task.yaml").write_text(yaml.safe_dump(doc))
            frame=pd.DataFrame({"id":range(12),"group":range(12),"x":range(12),"category":["a","b"]*6,"target":[x*2. for x in range(12)]})
            frame.to_csv(root/"dataset/train.csv",index=False)
            args=SimpleNamespace(project_root=str(root),task_config=None,fold=None,debug=False,probe_samples=None)
            cfg={"experiment":{"id":"exp001","branch":"baseline","parent_id":"none","problem_addressed":"Establish verified regression baseline","hypothesis":"Linear features provide a measurable regression control","intervention":"Train a regularized linear baseline"},"model":{"name":"ridge","params":{"alpha":.1}}}
            runner=UniversalRunner(cfg,"config.yaml",args)
            result=runner.run()
            self.assertEqual(len(result["fold_metrics"]),2)
            self.assertEqual(result["n_samples"],12)
            self.assertGreaterEqual(result["primary_score"],0)
            self.assertEqual(result["utility"],-result["primary_score"])
            self.assertEqual(len(pd.read_csv(runner.exp_dir/"oof_predictions.csv")),12)
            from harness.artifacts import validate_run, sha256
            validate_run(runner.exp_dir,task_spec=runner.task_spec,plugin=runner.task_plugin)
            from harness.candidates import build_candidate,validate_candidate
            candidate=build_candidate(runner.project,"control","single_model",{"source_exp":"exp001"})
            validate_candidate(runner.project,candidate)
            with self.assertRaises((ValueError,FileNotFoundError)):
                build_candidate(runner.project,"missing","single_model",{"source_exp":"missing"})

            original=sha256(runner.exp_dir/"metrics.json")
            with self.assertRaisesRegex(ValueError,"overwrite"):
                UniversalRunner(cfg,"config.yaml",args)
            args.debug=True
            probe=UniversalRunner(cfg,"config.yaml",args)
            probe.run()
            self.assertEqual(original,sha256(runner.exp_dir/"metrics.json"))
            self.assertEqual(probe.run_kind,"probe")
            with self.assertRaises(ValueError): validate_run(probe.exp_dir)
            (runner.exp_dir/"metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"hash"): validate_run(runner.exp_dir)

    def test_llm_false_label_and_invalid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"configs").mkdir(); (root/"dataset").mkdir()
            spec=task("binary_classification","accuracy","maximize")
            doc={"task":spec.metadata}; doc["task"]["data"]["train_path"]="dataset/train.csv"
            doc["task"]["validation"]={"strategy":"holdout","holdout_ratio":.5}
            (root/"configs/task.yaml").write_text(yaml.safe_dump(doc))
            pd.DataFrame({"id":range(8),"group":range(8),"x":range(8),"target":[0]*8}).to_csv(root/"dataset/train.csv",index=False)
            args=SimpleNamespace(project_root=str(root),task_config=None,fold=None,debug=False,probe_samples=None)
            cfg={"experiment":{"id":"llm"},"llm":{"prompt_template":"{x}","model_name":"fixture","provider":"openai"}}
            runner=UniversalRunner(cfg,"config",args)
            with patch("harness.llm_engine.LLMEngine.generate_completion",return_value={"text":'{"prediction":0}'}) as complete:
                frame=pd.read_csv(root/"dataset/train.csv")
                result=runner._run_llm_prompting(frame,"test")
            self.assertEqual(result["primary_score"],1)
            self.assertEqual(complete.call_args.kwargs["model_name"],"fixture")
