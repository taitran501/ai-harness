"""Universal Experiment Execution Runner for ML/DL/LLM Competition Harness.

Provides standardized headless execution across:
1. Classical Machine Learning (Tabular / NLP text features with GBDT / Linear models).
2. Deep Learning Transformers (PyTorch / Hugging Face).
3. Modern LLM Prompting & In-Context Learning (via LLMEngine APIs or local endpoints).
4. Validation schemes: K-Fold Cross-Validation, Single Holdout Split, and Sample Probing.
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import yaml

from harness.diagnostics import DiagnosticEngine
from harness.grounding_gate import GroundingGate
from harness.io_utils import (
    get_git_commit_hash,
    get_project_root,
    load_yaml_config,
    render_experiment_card,
    save_json,
    set_seed,
    setup_logger,
    update_experiments_leaderboard,
)
from harness.task_spec import UniversalTaskSpec, load_active_task_spec
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.project import HarnessProject
from harness.tree_journal import Node, TreeJournal
from harness.validator import ExperimentValidator


def merge_configs(base: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge experiment configuration over base configuration."""
    merged = base.copy()
    for k, v in exp.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


from harness.models import GenericBaselineModel, call_model
from harness.evaluation import resolve_evaluation
import json
import importlib
import inspect


class UniversalRunner:
    """One execution contract for local and packaged remote runs."""
    def __init__(self, cfg, config_path, args):
        self.cfg, self.config_path, self.args = cfg, config_path, args
        self.project = HarnessProject.discover(getattr(args,"project_root",None),getattr(args,"task_config",None))
        self.project_root = self.project.root
        self.task_spec = load_active_task_spec(self.project.task_config_path)
        self.exp_id = str(cfg.get("experiment",{}).get("id",Path(config_path).stem))
        if Path(self.exp_id).name != self.exp_id or self.exp_id in {".",".."} or "/" in self.exp_id or "\\" in self.exp_id:
            raise ValueError("Experiment ID must be a simple directory name")
        self.artifacts_dir = self.project.artifacts_dir
        self.exp_dir = self.artifacts_dir / self.exp_id
        self.plugin_context = TaskPluginContext(self.project_root,self.project.task_config_path,self.task_spec,self.project.plugin_id or "default",cfg)
        self.task_plugin = load_task_plugin(self.project.plugin_id,self.plugin_context)
        self.exp_dir.mkdir(parents=True,exist_ok=True)
        self.logger = __import__("logging").getLogger("ml-harness.runner")
        (self.exp_dir/"run.log").write_text("Starting run\n", encoding="utf-8")
        self.seed = self.task_spec.resolve_seed(cfg.get("runtime",{}).get("seed",42))
        self.estimator_metadata = []

    def _evaluate(self, truth, predictions):
        result = self.task_plugin.evaluate_predictions(self.plugin_context,truth,predictions)
        hook = self.cfg.get("paths",{}).get("metrics_evaluator")
        if hook:
            module,name = hook.rsplit(".",1)
            result.update(getattr(importlib.import_module(module),name)(truth,predictions))
        return resolve_evaluation(self.task_spec,result,len(truth))

    def run(self):
        _,_,receipt = GroundingGate.verify_config(self.cfg,config_path=self.config_path,raise_on_error=True)
        GroundingGate.write_receipt(receipt,self.exp_dir/"GROUNDING_RECEIPT.json")
        set_seed(self.seed)
        self.cfg.setdefault("runtime",{})["seed"] = self.seed
        git_hash = get_git_commit_hash(self.project_root)
        self.cfg["resolved_runtime"] = {"git_commit_hash":git_hash,"python_version":platform.python_version(),"seed":self.seed}
        (self.exp_dir/"run_config.yaml").write_text(yaml.safe_dump(self.cfg,sort_keys=False),encoding="utf-8")
        frame = self.task_plugin.load_training_data(self.plugin_context)
        if isinstance(frame,tuple): frame = frame[0]
        if frame is None: frame = self.task_spec.load_dataset_split("train",project_root=self.project_root)
        self.task_plugin.validate_training_data(self.plugin_context,frame)
        id_col = self.task_spec.id_column
        if id_col not in frame or frame[id_col].isna().any() or frame[id_col].duplicated().any():
            raise ValueError("Training requires declared unique, non-null IDs")
        group = self.task_spec.group_column
        if group and (group not in frame or frame[group].isna().any()):
            raise ValueError("Declared group column must exist and contain no nulls")
        split_value = self.cfg.get("paths",{}).get("split_path") or self.task_spec.validation.split_file or self.task_spec.data_config.get("split_path")
        if split_value and "fold" not in frame:
            path = self.project_root / split_value
            splits = pd.read_csv(path,sep="\t" if path.suffix==".tsv" else ",")
            if id_col not in splits or "fold" not in splits: raise ValueError("Split file requires ID and fold")
            frame = frame.merge(splits[[id_col,"fold"]],on=id_col,how="left",validate="one_to_one")
            if frame["fold"].isna().any(): raise ValueError("Split file does not cover training IDs")
        approach = self.cfg.get("experiment",{}).get("approach","ml_classical")
        if approach not in {"ml_classical","ml","dl","llm_peft","llm_prompt"}: raise ValueError(f"Unsupported approach {approach}")
        if approach in {"dl","llm_peft"} and not (self.cfg.get("model",{}).get("factory") or self.task_plugin.capabilities()["custom_model_factory"]):
            raise ValueError("This approach requires an implemented task model factory")
        self.data_frame = frame
        strategy = self.cfg.get("validation",{}).get("strategy",self.task_spec.validation.strategy)
        return self._run_llm_prompting(frame,git_hash) if approach=="llm_prompt" else self._run_model_training(frame,strategy,git_hash)

    def _resolve_validation_splits(self, df, strategy):
        config = {**self.task_spec.metadata.get("validation",{}),**self.cfg.get("validation",{})}
        result = self.task_plugin.build_validation_splits(self.plugin_context,df,strategy,config)
        if result is None:
            if strategy == "kfold":
                if "fold" not in df:
                    from harness.data_strategy import DataStrategyEngine, SplitMode
                    mode = SplitMode.GROUP_KFOLD if self.task_spec.group_column else (SplitMode.KFOLD if all(s.task_type=="regression" for s in self.task_spec.subtasks.values()) else SplitMode.STRATIFIED_KFOLD)
                    indices = DataStrategyEngine.generate_splits(df,mode=mode,n_splits=self.task_spec.validation.num_folds,group_col=self.task_spec.group_column,stratify_col=self.task_spec.primary_target,seed=self.seed)
                    result = [(i,df.iloc[t].copy(),df.iloc[v].copy()) for i,(t,v) in enumerate(indices)]
                else:
                    if df["fold"].isna().any(): raise ValueError("Null validation folds")
                    result = [(f,df[df.fold!=f].copy(),df[df.fold==f].copy()) for f in sorted(df.fold.unique())]
            elif strategy in {"holdout","zero_shot_eval"}:
                ratio = config.get("holdout_ratio")
                if ratio is not None:
                    if not 0<float(ratio)<1: raise ValueError("holdout_ratio must lie between zero and one")
                    from harness.data_strategy import DataStrategyEngine, SplitMode
                    [(t,v)] = DataStrategyEngine.generate_splits(df,mode=SplitMode.HOLDOUT_TEST,group_col=self.task_spec.group_column,holdout_ratio=float(ratio),seed=self.seed)
                    result = [(0,df.iloc[t].copy(),df.iloc[v].copy())]
                else:
                    dev = self.task_spec.load_dataset_split(config.get("dev_split_name",self.task_spec.validation.dev_split_name),project_root=self.project_root)
                    result = [(0,df,dev)]
            else: raise ValueError(f"Unsupported validation strategy {strategy}")
        if not result: raise ValueError("Validation returned no splits")
        requested = getattr(self.args,"fold",None)
        if requested is not None:
            result = [x for x in result if str(x[0])==str(requested)]
            if not result: raise ValueError("Requested fold does not exist")
        for fold,train,dev in result:
            if train.empty or dev.empty: raise ValueError("Empty training or validation partition")
            key = self.task_spec.id_column
            if key not in dev or dev[key].isna().any() or dev[key].duplicated().any(): raise ValueError("Invalid validation IDs")
            if set(train[key]) & set(dev[key]): raise ValueError("Training/validation ID overlap")
            ExperimentValidator.assert_no_group_data_leakage(train,dev,self.task_spec.group_column)
            self.task_plugin.validate_split(self.plugin_context,fold,train,dev)
        return result

    def _get_model_factory(self):
        path = self.cfg.get("model",{}).get("factory")
        if path:
            mod,name = path.rsplit(".",1)
            return getattr(importlib.import_module(mod),name)
        return lambda name,**params: GenericBaselineModel(self.task_spec,name,**params)

    def _record_predictions(self, dev, predictions, fold, variant):
        records=[]
        for i in range(len(dev)):
            record={self.task_spec.id_column:dev[self.task_spec.id_column].iloc[i],"fold":fold,"input_variant":variant}
            for key,value in predictions.items():
                if key.startswith("classes_"):
                    record[key]=json.dumps(list(value),default=str)
                else:
                    if len(value)!=len(dev): raise ValueError(f"Prediction coverage mismatch for {key}")
                    item=value[i]
                    column=key if key.startswith(("pred_","proba_")) else "pred_"+key
                    record[column]=json.dumps(np.asarray(item).tolist()) if isinstance(item,(list,np.ndarray)) else item
            records.append(record)
        return records

    def _run_model_training(self, df, val_strategy, git_hash):
        variants=self.cfg.get("data",{}).get("input_variants") or list(self.task_spec.inputs_dict)
        if not variants: raise ValueError("No input variants declared")
        selected=self.cfg.get("data",{}).get("selection_variant")
        if len(variants)>1 and selected not in variants: raise ValueError("Multiple variants require data.selection_variant")
        self.selection_variant=selected or variants[0]
        details,rows,records,truths=[],[],[],[]
        for fold,train,dev in self._resolve_validation_splits(df,val_strategy):
            partitions=[("all",train,dev)]
            partition=self.cfg.get("data",{}).get("partition_column")
            if partition:
                if partition not in df: raise ValueError("Partition column is missing")
                partitions=[(p,train[train[partition]==p],dev[dev[partition]==p]) for p in sorted(df[partition].dropna().unique()) if len(dev[dev[partition]==p])]
            for part,tr,dv in partitions:
                if tr.empty: raise ValueError("Partition has no training data")
                limit=getattr(self.args,"probe_samples",None)
                if getattr(self.args,"debug",False) or limit:
                    tr=tr.sample(n=min(len(tr),limit or 50),random_state=self.seed)
                    dv=dv.head(limit or 50)
                truths.append(dv)
                for variant in variants:
                    config=self.cfg.get("model",{})
                    model=self.task_plugin.build_model(self.plugin_context,config.get("name","baseline"),config.get("params",{}))
                    if model is None: model=self._get_model_factory()(config.get("name","baseline"),**config.get("params",{}))
                    start=time.monotonic()
                    call_model(model,"fit",tr,variant)
                    ExperimentValidator.assert_parameter_limit(model,max_params=self.cfg.get("constraints",{}).get("max_params_limit"))
                    pred=call_model(model,"predict",dv,variant)
                    pred=self.task_plugin.normalize_predictions(self.plugin_context,pred)
                    pred=self.task_spec.apply_routing(pred,len(dv))
                    evaluation=self._evaluate(dv,pred)
                    detail={"experiment_id":self.exp_id,"input_variant":variant,"dev_fold":fold,"partition":part,"train_rows":len(tr),"dev_rows":len(dv),"fit_time_seconds":time.monotonic()-start,**evaluation}
                    details.append(detail); rows.append({k:v for k,v in detail.items() if isinstance(v,(str,int,float))})
                    records.extend(self._record_predictions(dv,pred,fold,variant))
                    self.estimator_metadata.append(getattr(model,"estimator_metadata",{"class":type(model).__name__}))
        truth=pd.concat(truths).drop_duplicates(self.task_spec.id_column)
        return self._save_and_finalize(truth,details,rows,records,git_hash)

    def _run_llm_prompting(self, df, git_hash):
        from harness.llm_engine import LLMEngine
        config=self.cfg.get("llm",{})
        records,details,truths=[],[],[]
        self.selection_variant="llm_prompt"
        strategy=self.cfg.get("validation",{}).get("strategy",self.task_spec.validation.strategy)
        targets={s.column or key for key,s in self.task_spec.subtasks.items()}
        for fold,train,dev in self._resolve_validation_splits(df,strategy):
            limit=self.cfg.get("validation",{}).get("sample_limit")
            if getattr(self.args,"debug",False): limit=getattr(self.args,"probe_samples",None) or 50
            if limit: dev=dev.head(int(limit))
            truths.append(dev)
            predictions={key:[] for key in self.task_spec.subtasks}
            for _,row in dev.iterrows():
                features={k:v for k,v in row.to_dict().items() if k not in targets}
                system,user=LLMEngine.format_prompt(config.get("prompt_template","{text}"),features,config.get("system_prompt",""),config.get("few_shot_examples",[]),config.get("use_cot",False))
                if any("{"+c+"}" in user for c in targets): raise ValueError("LLM prompt references ground-truth labels")
                response=LLMEngine.generate_completion(provider=config.get("provider","gemini"),model_name=config.get("model_name","gemini-2.0-flash"),system_prompt=system,user_prompt=user,temperature=config.get("temperature",0),max_tokens=config.get("max_tokens",512),top_p=config.get("top_p",.95),base_url=config.get("api_base_url"))
                text=response.get("text","").strip()
                if text.startswith("```"): text=text.split("\n",1)[1].rsplit("```",1)[0].strip()
                try: parsed=json.loads(text)
                except json.JSONDecodeError: raise ValueError("LLM response is not valid JSON")
                if not isinstance(parsed,dict): raise ValueError("LLM output must be an object")
                for key,st in self.task_spec.subtasks.items():
                    value=next((parsed[k] for k in [key,st.column] if k and k in parsed),None)
                    if value is None and len(predictions)==1: value=parsed.get("prediction",parsed.get("label"))
                    if value is None: raise ValueError(f"LLM output omitted {key}")
                    predictions[key].append(value)
            predictions=self.task_spec.apply_routing(predictions,len(dev))
            evaluation=self._evaluate(dev,predictions)
            details.append({"experiment_id":self.exp_id,"input_variant":"llm_prompt","dev_fold":fold,**evaluation})
            records.extend(self._record_predictions(dev,predictions,fold,"llm_prompt"))
        return self._save_and_finalize(pd.concat(truths).drop_duplicates(self.task_spec.id_column),details,[],records,git_hash)

    def _save_and_finalize(self, truth, details, rows, records, git_hash):
        oof=pd.DataFrame(records)
        variants={}
        for variant in oof.input_variant.unique():
            selected=oof[oof.input_variant==variant].copy()
            key=self.task_spec.id_column
            if selected[key].duplicated().any(): raise ValueError("Duplicate OOF IDs within input variant")
            merged=selected[[key]].merge(truth,on=key,how="left",validate="one_to_one")
            predictions={}
            for col in selected:
                if col.startswith("pred_"):
                    values=selected[col].tolist()
                    predictions[col]=[json.loads(v) if isinstance(v,str) and v.startswith("[") else v for v in values]
                elif col.startswith("proba_"): predictions[col]=[json.loads(v) for v in selected[col]]
                elif col.startswith("classes_"):
                    if selected[col].nunique()!=1: raise ValueError("Class order changed across folds")
                    predictions[col]=json.loads(selected[col].iloc[0])
            variants[variant]=self._evaluate(merged,predictions)
        aggregate=variants[self.selection_variant]
        result={"schema_version":2,"experiment_id":self.exp_id,"selection_variant":self.selection_variant,**aggregate,"variants":variants,"fold_metrics":details}
        save_json(self.exp_dir/"metrics.json",result)
        summary=pd.DataFrame(rows or [{"input_variant":d["input_variant"],"dev_fold":d["dev_fold"],"primary_score":d["primary_score"]} for d in details])
        summary.to_csv(self.exp_dir/"summary.csv",index=False)
        oof.to_csv(self.exp_dir/"oof_predictions.csv",index=False)
        oof.to_csv(self.exp_dir/"predictions.csv",index=False)
        truth.to_csv(self.exp_dir/"evaluation_truth.csv",index=False)
        (self.exp_dir/"EXPERIMENT_CARD.md").write_text(f"# {self.exp_id}\n\n{aggregate['primary_metric']}: {aggregate['primary_score']} ({aggregate['direction']})\n",encoding="utf-8")
        try:
            DiagnosticEngine.save_diagnostics_bundle(truth,oof,self.exp_dir,self.exp_id)
            result["diagnostics_status"]="COMPLETED"
        except Exception as error:
            result["diagnostics_status"]="FAILED"; result["diagnostics_error"]=str(error)
        save_json(self.exp_dir/"metrics.json",result)
        journal=TreeJournal(journal_path=self.artifacts_dir/"journal.json",artifacts_dir=self.artifacts_dir,task_spec=self.task_spec)
        journal.add_node(self.exp_id,parent_id=self.cfg.get("experiment",{}).get("parent_id"),score=aggregate["primary_score"],metrics=aggregate)
        return result


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for universal training runner."""
    parser = argparse.ArgumentParser(description="Universal Competition Experiment Runner")
    parser.add_argument("--project-root", type=str, default=None, help="Competition workspace root")
    parser.add_argument("--task-config", type=str, default=None, help="Task specification YAML")
    parser.add_argument("--job-id", type=str, default=None, help="Persistent dashboard/Agent Bridge job ID")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--base-config", type=str, default=None, help="Path to base config YAML")
    parser.add_argument("--fold", type=int, default=None, help="Target fold index to train (smoke test / probe)")
    parser.add_argument("--debug", action="store_true", help="Debug mode with subsampled data")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing experiment artifacts")
    parser.add_argument("--probe-samples", type=int, default=None, help="Probe sample count limit")
    parser.add_argument("--backend", type=str, default="local", choices=["local", "modal", "kaggle"], help="Execution backend")
    parser.add_argument("--skip-leaderboard", action="store_true", help="Do not write the project leaderboard")
    args = parser.parse_args(argv)

    project = HarnessProject.discover(args.project_root, args.task_config)
    os.environ["HARNESS_PROJECT_ROOT"] = str(project.root)
    os.environ["HARNESS_TASK_CONFIG"] = str(project.task_config_path)
    if str(project.root) not in sys.path:
        sys.path.insert(0, str(project.root))
    os.chdir(project.root)

    def resolve_project_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (project.root / path).resolve()

    config_path = resolve_project_path(args.config)
    if not config_path.is_file():
        parser.error(f"Experiment config does not exist: {config_path}")

    # If backend is modal or kaggle, delegate to cloud runner
    if args.backend == "modal":
        from harness.cloud_runner import run_via_modal
        run_via_modal(
            config_path=str(config_path),
            project_root=project.root,
        )
        return 0
    elif args.backend == "kaggle":
        from harness.cloud_runner import run_via_kaggle
        run_via_kaggle(
            config_path=str(config_path),
            project_root=project.root,
        )
        return 0

    # Local execution
    package_base = Path(__file__).resolve().parent / "defaults" / "base.yaml"
    workspace_base = project.root / "configs" / "base.yaml"
    base_path = resolve_project_path(args.base_config) if args.base_config else (
        workspace_base if workspace_base.is_file() else package_base
    )
    base_cfg = load_yaml_config(base_path) if base_path.exists() else {}
    exp_cfg = load_yaml_config(config_path)
    cfg = merge_configs(base_cfg, exp_cfg)

    # Immutability check
    exp_id = cfg.get("experiment", {}).get("id", config_path.stem)
    paths = cfg.get("paths", {})
    artifacts_root = project.artifacts_dir if project.manifest else Path(
        paths.get("artifacts_dir") or project.artifacts_dir
    )
    if not artifacts_root.is_absolute():
        artifacts_root = project.root / artifacts_root
    exp_dir = artifacts_root / exp_id
    if exp_dir.exists() and (exp_dir / "metrics.json").exists() and not args.force:
        if args.job_id:
            from harness.job_state import update_worker_state
            update_worker_state(
                project.root, args.job_id, status="FAILED", provider_status="FAILED",
                error_message=f"Experiment directory `{exp_dir}` already has completed metrics.",
                exp_id=str(exp_id), config_sha256=__import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
            )
        print(
            f"ERROR: Experiment directory '{exp_dir}' already exists with completed metrics.\n"
            f"Pass --force or define a new experiment ID in configs/.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.project_root = str(project.root)
    args.task_config = str(project.task_config_path)
    try:
        runner = UniversalRunner(cfg=cfg, config_path=str(config_path), args=args)
        runner.run()
    except Exception as exc:
        if args.job_id:
            from harness.job_state import update_worker_state
            update_worker_state(
                project.root, args.job_id, status="FAILED", provider_status="FAILED",
                error_message=str(exc), exp_id=str(exp_id),
                config_sha256=__import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
            )
        raise
    if args.job_id:
        import json
        from harness.job_state import update_worker_state
        metrics_path = exp_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        files = sorted(path for path in exp_dir.iterdir() if path.is_file()) if exp_dir.is_dir() else []
        update_worker_state(
            project.root,
            args.job_id,
            status="COMPLETED",
            provider_status="COMPLETED",
            result_metrics=metrics if isinstance(metrics, dict) else {"folds": metrics},
            artifact_paths=[path.relative_to(project.root).as_posix() for path in files],
            exp_id=str(exp_id),
            config_sha256=__import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
