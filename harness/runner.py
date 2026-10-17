"""Universal Experiment Execution Runner for ML/DL/LLM Competition Harness.

Provides standardized headless execution across:
1. Classical Machine Learning (Tabular / NLP text features with GBDT / Linear models).
2. Deep Learning Transformers (PyTorch / Hugging Face).
3. Modern LLM Prompting & In-Context Learning (via LLMEngine APIs or local endpoints).
4. Validation schemes: K-Fold Cross-Validation, Single Holdout Split, and Sample Probing.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from harness.diagnostics import DiagnosticEngine
from harness.grounding_gate import GroundingGate
from harness.io_utils import (
    get_git_commit_hash,
    load_yaml_config,
    save_json,
    set_seed,
)
from harness.task_spec import load_active_task_spec
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.project import HarnessProject
from harness.tree_journal import TreeJournal
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


from harness.models import call_model
from harness.evaluation import resolve_evaluation
import json
import importlib
import inspect
import uuid
from harness.artifacts import (
    contract_identity,
    digest,
    ensure_canonical_truth,
    frame_hash,
    mirror_csv,
    sha256,
)


class UniversalRunner:
    """One execution contract for local and packaged remote runs."""

    def __init__(self, cfg, config_path, args):
        self.cfg, self.config_path, self.args = cfg, config_path, args
        self.project = HarnessProject.discover(
            getattr(args, "project_root", None), getattr(args, "task_config", None)
        )
        self.project_root = self.project.root
        self.task_spec = load_active_task_spec(self.project.task_config_path)
        self.exp_id = str(cfg.get("experiment", {}).get("id", Path(config_path).stem))
        if (
            Path(self.exp_id).name != self.exp_id
            or self.exp_id in {".", ".."}
            or "/" in self.exp_id
            or "\\" in self.exp_id
        ):
            raise ValueError("Experiment ID must be a simple directory name")
        self.artifacts_dir = self.project.artifacts_dir
        self.run_id = getattr(args, "job_id", None) or uuid.uuid4().hex
        self.run_kind = (
            "probe"
            if (
                getattr(args, "debug", False)
                or getattr(args, "probe_samples", None)
                or getattr(args, "fold", None) is not None
                or cfg.get("validation", {}).get("sample_limit") is not None
            )
            else "full"
        )
        self.exp_dir = (
            self.artifacts_dir / self.exp_id
            if self.run_kind == "full"
            else self.artifacts_dir / "probes" / self.exp_id / self.run_id
        )
        self.plugin_context = TaskPluginContext(
            self.project_root,
            self.project.task_config_path,
            self.task_spec,
            self.project.plugin_id or "default",
            cfg,
        )
        self.task_plugin = load_task_plugin(self.project.plugin_id, self.plugin_context)
        self.exp_dir.parent.mkdir(parents=True, exist_ok=True)
        from harness.state_store import workspace_lock

        with workspace_lock(self.artifacts_dir):
            state_path = self.artifacts_dir / "competition_state.json"
            if state_path.is_file() and json.loads(state_path.read_text()).get(
                "gates", {}
            ).get("freeze_locked"):
                raise ValueError(
                    "Competition is frozen; mutation requires an approved unfreeze"
                )
        try:
            self.exp_dir.mkdir(exist_ok=False)
        except FileExistsError:
            raise ValueError(
                "Refusing to overwrite an existing run; use a new experiment ID. --force cannot overwrite artifacts"
            )
        # Initialize eagerly so any code path (script adapters, independent
        # validation checks) can reference it before split resolution runs.
        self.expected_validation_ids: set = set()
        self.manifest = {
            "schema_version": 2,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "exp_id": self.exp_id,
            "status": "RUNNING",
            "plugin_id": self.project.plugin_id or "default",
            **contract_identity(self.task_spec, self.task_plugin),
        }
        save_json(self.exp_dir / "artifact_manifest.json", self.manifest)
        self.logger = __import__("logging").getLogger("ai-harness.runner")
        (self.exp_dir / "run.log").write_text("Starting run\n", encoding="utf-8")
        if not self.task_plugin.capabilities().get("custom_metrics") and not cfg.get(
            "paths", {}
        ).get("metrics_evaluator"):
            from harness.evaluation import metric_name

            allowed = {
                "regression": {
                    "rmse",
                    "mae",
                    "r2",
                    "mape",
                    "spearman",
                    "ndcg",
                    "qwk",
                },
                "binary_classification": {
                    "accuracy",
                    "macro_f1",
                    "micro_f1",
                    "roc_auc",
                    "logloss",
                    "qwk",
                },
                "multiclass_classification": {
                    "accuracy",
                    "macro_f1",
                    "micro_f1",
                    "roc_auc",
                    "logloss",
                    "qwk",
                },
                "multilabel_classification": {
                    "accuracy",
                    "macro_f1",
                    "micro_f1",
                    "ndcg",
                },
                "ranking": {
                    "ndcg",
                    "spearman",
                    "rmse",
                    "mae",
                },
            }
            if not self.task_spec.subtasks:
                raise ValueError("Task must declare targets")
            for st in self.task_spec.subtasks.values():
                if metric_name(st.primary_metric) not in allowed.get(
                    st.task_type, set()
                ):
                    raise ValueError("Unsupported task metric at preflight")
        self.seed = self.task_spec.resolve_seed(cfg.get("runtime", {}).get("seed", 42))
        self.estimator_metadata = []
        hook = cfg.get("paths", {}).get("metrics_evaluator")
        if hook:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            module, name = hook.split(":", 1) if ":" in hook else hook.rsplit(".", 1)
            self.manifest["custom_scorer_sha256"] = digest(
                inspect.getsource(getattr(importlib.import_module(module), name))
            )

    def _probe_sample_limit(self):
        """Effective subsample size for probe runs (None = full data).

        Precedence: explicit ``--probe-samples`` > 50 when ``--debug`` is set >
        ``validation.sample_limit`` from the experiment config. A single-fold
        run is not subsampled on its own.
        """
        cli = getattr(self.args, "probe_samples", None)
        if cli:
            return int(cli)
        if getattr(self.args, "debug", False):
            return 50
        limit = self.cfg.get("validation", {}).get("sample_limit")
        return int(limit) if limit else None

    def _evaluate(self, truth, predictions):
        from harness.evaluation import validate_prediction_contract

        validate_prediction_contract(self.task_spec, truth, predictions)
        hook = self.cfg.get("paths", {}).get("metrics_evaluator")
        if hook:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            module, name = hook.split(":", 1) if ":" in hook else hook.rsplit(".", 1)
            result = getattr(importlib.import_module(module), name)(truth, predictions)
            if not isinstance(result, dict):
                raise TypeError("Custom metrics_evaluator must return a metric mapping")
            try:
                base_result = self.task_plugin.evaluate_predictions(
                    self.plugin_context, truth, predictions
                )
                result = {**base_result, **result}
            except Exception as exc:
                # A custom metrics_evaluator is configured but the official
                # plugin scorer failed. Keep the custom result, but surface
                # the failure — it must not be silently swallowed.
                result.setdefault("_evaluation_warnings", []).append(
                    f"official evaluation failed: {exc.__class__.__name__}: {exc}"
                )
                print(
                    f"[ai-harness] WARNING: official plugin evaluation failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
        else:
            result = self.task_plugin.evaluate_predictions(
                self.plugin_context, truth, predictions
            )
        return resolve_evaluation(self.task_spec, result, len(truth))

    def run(self):
        try:
            return self._execute_run()
        except Exception as error:
            if self.manifest.get("status") != "COMPLETED":
                self.manifest.update(status="FAILED", error=str(error))
                save_json(self.exp_dir / "artifact_manifest.json", self.manifest)
            raise

    def _execute_run(self):
        _, _, receipt = GroundingGate.verify_config(
            self.cfg, config_path=self.config_path, raise_on_error=True
        )
        GroundingGate.write_receipt(receipt, self.exp_dir / "GROUNDING_RECEIPT.json")
        set_seed(self.seed)
        self.cfg.setdefault("runtime", {})["seed"] = self.seed
        git_hash = get_git_commit_hash(self.project_root)
        self.cfg["resolved_runtime"] = {
            "git_commit_hash": git_hash,
            "python_version": platform.python_version(),
            "seed": self.seed,
        }
        (self.exp_dir / "run_config.yaml").write_text(
            yaml.safe_dump(self.cfg, sort_keys=False), encoding="utf-8"
        )
        frame = self._load_training_data()
        approach = self.cfg.get("experiment", {}).get("approach", "ml_classical")
        execution_mode = self.cfg.get("execution", {}).get("mode") or ("script" if approach in {"script", "custom_script"} else "built_in")
        if execution_mode == "pipeline":
            if approach not in {"script", "custom_script"}:
                raise ValueError(
                    "Pipeline execution requires experiment.approach 'script' or 'custom_script'"
                )
        elif execution_mode != "script" and approach not in {"ml_classical", "ml", "dl", "llm_peft", "llm_prompt"}:
            raise ValueError(f"Unsupported approach {approach}")
        if approach in {"dl", "llm_peft"} and execution_mode not in {"script", "pipeline"} and not (
            self.cfg.get("model", {}).get("factory")
            or self.task_plugin.capabilities()["custom_model_factory"]
        ):
            raise ValueError("This approach requires an implemented task model factory")
        self.data_frame = frame
        self.manifest["data_sha256"] = frame_hash(frame)
        self.manifest["resolved_config_sha256"] = sha256(
            self.exp_dir / "run_config.yaml"
        )
        strategy = self.cfg.get("validation", {}).get(
            "strategy", self.task_spec.validation.strategy
        )
        if execution_mode == "script":
            return self._run_custom_script(frame, strategy, git_hash)
        if execution_mode == "pipeline":
            return self._run_pipeline(frame, strategy, git_hash)
        return (
            self._run_llm_prompting(frame, git_hash)
            if approach == "llm_prompt"
            else self._run_model_training(frame, strategy, git_hash)
        )

    def _load_training_data(self) -> pd.DataFrame:
        frame = self.task_plugin.load_training_data(self.plugin_context)
        if isinstance(frame, tuple):
            frame = frame[0]
        if frame is None:
            train_path_override = self.cfg.get("paths", {}).get("train_path")
            if train_path_override:
                from harness.task_spec import load_dataframe_file

                p = Path(train_path_override)
                if not p.is_absolute():
                    p = self.project_root / p
                if p.is_file():
                    frame = load_dataframe_file(p)
            if frame is None:
                frame = self.task_spec.load_dataset_split(
                    "train", project_root=self.project_root
                )
        self.task_plugin.validate_training_data(self.plugin_context, frame)
        id_col = self.task_spec.id_column
        if (
            id_col not in frame
            or frame[id_col].isna().any()
            or frame[id_col].duplicated().any()
        ):
            raise ValueError("Training requires declared unique, non-null IDs")
        group = self.task_spec.group_column
        if group and (group not in frame or frame[group].isna().any()):
            raise ValueError("Declared group column must exist and contain no nulls")
        split_value = (
            self.cfg.get("paths", {}).get("split_path")
            or self.task_spec.validation.split_file
            or self.task_spec.data_config.get("split_path")
        )
        if split_value and "fold" not in frame:
            path = self.project_root / split_value
            if not path.is_file():
                raise FileNotFoundError(f"Split file does not exist: {path}")
            splits = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
            fold_aliases = ["fold", "kfold", "split", "fold_id", "cv_fold", "Fold", "Split"]
            actual_fold_col = None
            for alias in fold_aliases:
                if alias in splits.columns:
                    actual_fold_col = alias
                    break

            actual_id_col = None
            if id_col in splits.columns:
                actual_id_col = id_col
            else:
                for col in splits.columns:
                    if str(col).strip().lower() == str(id_col).strip().lower():
                        actual_id_col = col
                        break

            if not actual_id_col or not actual_fold_col:
                missing = []
                if not actual_id_col:
                    missing.append(f"id column '{id_col}'")
                if not actual_fold_col:
                    missing.append(f"fold column (checked aliases: {fold_aliases})")
                raise ValueError(
                    f"Split file '{path.name}' is missing required columns: {', '.join(missing)}. "
                    f"Found columns: {list(splits.columns)}"
                )

            split_df = splits[[actual_id_col, actual_fold_col]].copy()
            if actual_id_col != id_col:
                split_df = split_df.rename(columns={actual_id_col: id_col})
            if actual_fold_col != "fold":
                split_df = split_df.rename(columns={actual_fold_col: "fold"})

            frame = frame.merge(
                split_df[[id_col, "fold"]], on=id_col, how="left", validate="one_to_one"
            )
            if frame["fold"].isna().any():
                missing_mask = frame["fold"].isna()
                missing_cnt = int(missing_mask.sum())
                sample_missing = frame.loc[missing_mask, id_col].head(5).tolist()
                raise ValueError(
                    f"Split file does not cover {missing_cnt}/{len(frame)} training IDs. "
                    f"Sample uncovered IDs: {sample_missing}"
                )
        return frame

    def _resolve_validation_splits(self, df, strategy):
        config = {
            **self.task_spec.metadata.get("validation", {}),
            **self.cfg.get("validation", {}),
        }
        result = self.task_plugin.build_validation_splits(
            self.plugin_context, df, strategy, config
        )
        if result is None:
            if strategy == "kfold":
                if "fold" not in df:
                    from harness.data_strategy import DataStrategyEngine, SplitMode

                    mode = (
                        SplitMode.GROUP_KFOLD
                        if self.task_spec.group_column
                        else (
                            SplitMode.KFOLD
                            if all(
                                s.task_type == "regression"
                                for s in self.task_spec.subtasks.values()
                            )
                            else SplitMode.STRATIFIED_KFOLD
                        )
                    )
                    indices = DataStrategyEngine.generate_splits(
                        df,
                        mode=mode,
                        n_splits=int(
                            config.get("num_folds", self.task_spec.validation.num_folds)
                        ),
                        group_col=self.task_spec.group_column,
                        stratify_col=self.task_spec.primary_target,
                        seed=self.seed,
                    )
                    result = [
                        (i, df.iloc[t].copy(), df.iloc[v].copy())
                        for i, (t, v) in enumerate(indices)
                    ]
                else:
                    if df["fold"].isna().any():
                        raise ValueError(
                            "Null validation folds: the dataset carries a 'fold' "
                            "column with missing values. Remove it so the "
                            "harness can generate splits, or fill every fold index."
                        )
                    if df.fold.nunique() <= 1:
                        raise ValueError(
                            f"Dataset carries a constant 'fold' column (value "
                            f"{df.fold.iloc[0]!r}) which cannot define validation "
                            "splits. Remove the column so the harness generates "
                            "its CV splits (e.g. GroupKFold on the task "
                            "group_column), or provide distinct fold indices."
                        )
                    if df.fold.nunique() != int(
                        config.get("num_folds", self.task_spec.validation.num_folds)
                    ):
                        raise ValueError(
                            "Declared fold count differs from split file: the "
                            "'fold' column defines "
                            f"{df.fold.nunique()} splits but the config/task "
                            f"declares {config.get('num_folds', self.task_spec.validation.num_folds)}. "
                            "Align num_folds or fix the split file."
                        )
                    result = [
                        (
                            int(f) if isinstance(f, (int, np.integer)) else f,
                            df[df.fold != f].copy(),
                            df[df.fold == f].copy(),
                        )
                        for f in sorted(df.fold.unique())
                    ]
            elif strategy in {"holdout", "zero_shot_eval"}:
                ratio = config.get("holdout_ratio")
                if ratio is not None:
                    if not 0 < float(ratio) < 1:
                        raise ValueError("holdout_ratio must lie between zero and one")
                    from harness.data_strategy import DataStrategyEngine, SplitMode

                    [(t, v)] = DataStrategyEngine.generate_splits(
                        df,
                        mode=SplitMode.HOLDOUT_TEST,
                        group_col=self.task_spec.group_column,
                        holdout_ratio=float(ratio),
                        seed=self.seed,
                    )
                    result = [(0, df.iloc[t].copy(), df.iloc[v].copy())]
                else:
                    dev = self.task_spec.load_dataset_split(
                        config.get(
                            "dev_split_name", self.task_spec.validation.dev_split_name
                        ),
                        project_root=self.project_root,
                    )
                    result = [(0, df, dev)]
            elif strategy in {"time_series", "time_series_split", "timeseries", "temporal"}:
                from harness.data_strategy import DataStrategyEngine, SplitMode

                n_splits = int(
                    config.get(
                        "num_folds", getattr(self.task_spec.validation, "num_folds", 5)
                    )
                )
                indices = DataStrategyEngine.generate_splits(
                    df,
                    mode=SplitMode.TIME_SERIES,
                    n_splits=n_splits,
                )
                result = [
                    (i, df.iloc[t].copy(), df.iloc[v].copy())
                    for i, (t, v) in enumerate(indices)
                ]
            elif strategy in {
                "purged_group_time_series",
                "purged_time_series",
                "purged_kfold",
                "financial_time_series",
            }:
                from harness.data_strategy import DataStrategyEngine, SplitMode

                n_splits = int(
                    config.get(
                        "num_folds", getattr(self.task_spec.validation, "num_folds", 5)
                    )
                )
                embargo = float(
                    config.get(
                        "embargo_pct", getattr(self.task_spec.validation, "embargo_pct", 0.01)
                    )
                )
                indices = DataStrategyEngine.generate_splits(
                    df,
                    mode=SplitMode.PURGED_GROUP_TIME_SERIES,
                    n_splits=n_splits,
                    group_col=self.task_spec.group_column,
                    holdout_ratio=embargo,
                )
                result = [
                    (i, df.iloc[t].copy(), df.iloc[v].copy())
                    for i, (t, v) in enumerate(indices)
                ]
            elif strategy in {
                "multilabel_stratified",
                "multilabel_stratified_kfold",
                "multilabel_kfold",
            }:
                from harness.data_strategy import DataStrategyEngine, SplitMode

                n_splits = int(
                    config.get(
                        "num_folds", getattr(self.task_spec.validation, "num_folds", 5)
                    )
                )
                target_cols = [
                    (s.column_prefix + label) if s.column_prefix else (s.column or s.id)
                    for s in self.task_spec.subtasks.values()
                    for label in (s.labels or [s.id])
                ]
                indices = DataStrategyEngine.generate_splits(
                    df,
                    mode=SplitMode.MULTILABEL_STRATIFIED_KFOLD,
                    n_splits=n_splits,
                    target_cols=target_cols,
                    seed=self.seed,
                )
                result = [
                    (i, df.iloc[t].copy(), df.iloc[v].copy())
                    for i, (t, v) in enumerate(indices)
                ]
            else:
                raise ValueError(f"Unsupported validation strategy {strategy}")
        if not result:
            raise ValueError("Validation returned no splits")
        result = [
            (int(f) if isinstance(f, (int, np.integer)) else f, train, dev)
            for f, train, dev in result
        ]
        if strategy == "kfold":
            held = [x for _, _, dev in result for x in dev[self.task_spec.id_column]]
            if len(held) != len(set(held)) or set(held) != set(
                df[self.task_spec.id_column]
            ):
                raise ValueError(
                    "Validation does not cover each training ID exactly once"
                )
        requested = getattr(self.args, "fold", None)
        if requested is not None:
            result = [x for x in result if str(x[0]) == str(requested)]
            if not result:
                raise ValueError("Requested fold does not exist")
        for fold, train, dev in result:
            if train.empty or dev.empty:
                raise ValueError("Empty training or validation partition")
            key = self.task_spec.id_column
            if key not in dev or dev[key].isna().any() or dev[key].duplicated().any():
                raise ValueError("Invalid validation IDs")
            if set(train[key]) & set(dev[key]):
                raise ValueError("Training/validation ID overlap")
            if self.task_spec.group_column:
                ExperimentValidator.assert_no_group_data_leakage(
                    train, dev, self.task_spec.group_column
                )
            self.task_plugin.validate_split(self.plugin_context, fold, train, dev)
        self.expected_validation_ids = {
            x for _, _, dev in result for x in dev[self.task_spec.id_column]
        }
        self.manifest["split_sha256"] = digest(
            [
                (
                    str(f),
                    t[self.task_spec.id_column].tolist(),
                    v[self.task_spec.id_column].tolist(),
                )
                for f, t, v in result
            ]
        )
        return result

    def _get_model_factory(self):
        path = self.cfg.get("model", {}).get("factory")
        if path:
            mod, name = path.rsplit(".", 1)
            factory = getattr(importlib.import_module(mod), name)
            signature = inspect.signature(factory)
            declared = self.cfg.get("model", {}).get("factory_context")
            if declared not in {None, "none", "task_spec", "plugin_context"}:
                raise ValueError("factory_context must be none, task_spec, or plugin_context")
            context_key = declared if declared != "none" else None
            if declared is None:
                context_key = next((key for key in ("task_spec", "plugin_context")
                                    if key in signature.parameters), None)
            if context_key is None:
                return factory  # legacy factory(name, **params)

            value = self.task_spec if context_key == "task_spec" else self.plugin_context
            parameter = signature.parameters.get(context_key)
            first = next(iter(signature.parameters), None)
            positional = (parameter is None or
                          (first == context_key and parameter.kind in
                           (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)))

            def with_context(model_name, **params):
                if context_key in params:
                    raise ValueError(f"Model params cannot override injected {context_key}")
                args = (value, model_name) if positional else (model_name,)
                kwargs = params if positional else {**params, context_key: value}
                signature.bind(*args, **kwargs)
                # Do not catch TypeError from inside the factory or retry it.
                return factory(*args, **kwargs)

            return with_context
        from harness.models import create_model

        return lambda name, **params: create_model(
            self.task_spec, name, **params
        )

    def _progress(self, message):
        line = f"[ai-harness:{self.exp_id}] {message}"
        print(line, flush=True)
        with (self.exp_dir / "run.log").open("a", encoding="utf-8") as log:
            log.write(line + "\n")

    def _record_predictions(self, dev, predictions, fold, variant):
        fold_id = int(fold) if isinstance(fold, (int, np.integer)) else fold
        records = []
        for i in range(len(dev)):
            record = {
                self.task_spec.id_column: dev[self.task_spec.id_column].iloc[i],
                "fold": fold_id,
                "input_variant": variant,
            }
            for key, value in predictions.items():
                if key.startswith("classes_"):
                    record[key] = json.dumps(list(value), default=str)
                else:
                    if len(value) != len(dev):
                        raise ValueError(f"Prediction coverage mismatch for {key}")
                    item = value[i]
                    column = (
                        key if key.startswith(("pred_", "proba_")) else "pred_" + key
                    )
                    record[column] = (
                        json.dumps(np.asarray(item).tolist())
                        if isinstance(item, (list, np.ndarray))
                        else item
                    )
                    # Float decomposition for downstream agent stacking/blending ergonomics
                    if column.startswith("proba_") and isinstance(item, (list, np.ndarray)):
                        item_arr = np.asarray(item)
                        target_name = column.replace("proba_", "")
                        class_key = f"classes_{target_name}"
                        c_order = predictions.get(class_key)
                        if c_order and len(c_order) == len(item_arr):
                            for c_idx, c_label in enumerate(c_order):
                                clean_lbl = str(c_label).replace(" ", "_")
                                record[f"{column}_{clean_lbl}"] = float(item_arr[c_idx])
                        else:
                            for c_idx in range(len(item_arr)):
                                record[f"{column}_{c_idx}"] = float(item_arr[c_idx])
                        if len(item_arr) == 2 and f"{column}_1" not in record:
                            record[f"{column}_1"] = float(item_arr[1])
            records.append(record)
        return records

    def _run_custom_script(self, df, val_strategy, git_hash):
        """Execute user/agent-defined script (PyTorch, DL, custom pipeline) while enforcing harness audit gates."""
        import subprocess

        exec_cfg = self.cfg.get("execution", {})
        script_path_str = exec_cfg.get("entrypoint") or self.cfg.get("experiment", {}).get("entrypoint")
        if not script_path_str:
            raise ValueError("Script execution requires 'execution.entrypoint' in experiment config")

        script_path = Path(script_path_str)
        if not script_path.is_absolute():
            script_path = self.project_root / script_path
        if not script_path.exists():
            raise FileNotFoundError(f"Custom script entrypoint '{script_path}' not found")

        # 1. Resolve validation splits and audit expected validation IDs
        splits = self._resolve_validation_splits(df, val_strategy)
        if "fold" not in df.columns:
            df = df.copy()
            df["fold"] = 0
            for fold, _, dev in splits:
                df.loc[dev.index, "fold"] = fold

        # 2. Resolve test dataframe if test_path declared
        test_df = None
        if self.task_spec.test_path:
            tp = Path(self.task_spec.test_path)
            if not tp.is_absolute():
                tp = self.project_root / tp
            if tp.exists():
                try:
                    test_df = self.task_spec.load_dataset_split("test", project_root=self.project_root)
                except Exception:
                    test_df = None

        # 3. Export splits
        splits_file = self.exp_dir / "splits.tsv"
        id_col = self.task_spec.id_column
        df[[id_col, "fold"]].to_csv(splits_file, sep="\t", index=False)

        # 3. Environment & command setup
        child_env = os.environ.copy()
        python_bin = str(Path(sys.executable).parent)
        child_env["PATH"] = python_bin + os.pathsep + child_env.get("PATH", "")
        child_env["HARNESS_EXP_ID"] = self.exp_id
        child_env["HARNESS_EXP_DIR"] = str(self.exp_dir)
        child_env["HARNESS_PROJECT_ROOT"] = str(self.project_root)
        child_env["HARNESS_RUN_CONFIG"] = str(self.exp_dir / "run_config.yaml")
        child_env["HARNESS_TASK_CONFIG"] = str(self.project.task_config_path)
        child_env["HARNESS_SEED"] = str(self.seed)
        if splits_file.exists():
            child_env["HARNESS_SPLIT_FILE"] = str(splits_file)

        cmd = [
            sys.executable,
            str(script_path),
            "--exp-dir", str(self.exp_dir),
            "--config", str(self.exp_dir / "run_config.yaml"),
        ]
        extra_args = exec_cfg.get("args") or []
        if isinstance(extra_args, list):
            cmd.extend([str(a) for a in extra_args])

        self._progress(f"Launching custom script adapter: {' '.join(cmd)}")
        start_time = time.monotonic()
        res = subprocess.run(
            cmd,
            cwd=str(self.project_root),
            env=child_env,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            err_msg = res.stderr or res.stdout or f"Exited with code {res.returncode}"
            raise RuntimeError(f"Custom script execution failed:\n{err_msg}")

        # 4. Verify output artifacts
        oof_path = self.exp_dir / "oof_predictions.csv"
        if not oof_path.exists():
            pred_fallback = self.exp_dir / "predictions.csv"
            if pred_fallback.exists():
                oof_df = pd.read_csv(pred_fallback)
                oof_df.to_csv(oof_path, index=False)
            else:
                raise FileNotFoundError(f"Custom script must generate '{oof_path.name}' in target experiment dir")

        oof_df = pd.read_csv(oof_path)
        if id_col not in oof_df.columns:
            raise ValueError(f"OOF predictions file missing ID column '{id_col}'")

        # 5. Extract predictions and align with evaluation truth
        truth = df.drop_duplicates(id_col).copy()
        merged_oof = truth[[id_col]].merge(oof_df, on=id_col, how="inner")
        if len(merged_oof) != len(truth):
            raise ValueError(f"OOF predictions cover {len(merged_oof)} rows, expected {len(truth)}")

        variant = "script"
        self.selection_variant = variant
        records = []
        for i in range(len(merged_oof)):
            row = merged_oof.iloc[i]
            rec = {
                id_col: row[id_col],
                "fold": int(row.get("fold", 0)) if "fold" in row and pd.notna(row["fold"]) else 0,
                "input_variant": variant,
            }
            for target in self.task_spec.subtasks.keys():
                col_name = f"pred_{target}" if f"pred_{target}" in row else target
                if col_name in row:
                    rec[f"pred_{target}"] = row[col_name]
                if f"proba_{target}" in row:
                    rec[f"proba_{target}"] = row[f"proba_{target}"]
            records.append(rec)

        elapsed = time.monotonic() - start_time
        details = [{
            "input_variant": variant,
            "dev_fold": "all",
            "train_rows": len(df),
            "dev_rows": len(df),
            "fit_time_seconds": elapsed,
        }]
        rows = [{
            "input_variant": variant,
            "dev_fold": "all",
            "primary_score": 0.0,
        }]

        fold_test_preds = []
        test_pred_path = self.exp_dir / "test_predictions.csv"
        if test_pred_path.exists() and test_df is not None:
            t_df = pd.read_csv(test_pred_path)
            cand_test = {}
            for target in self.task_spec.subtasks.keys():
                col_name = f"pred_{target}" if f"pred_{target}" in t_df.columns else target
                if col_name in t_df.columns:
                    cand_test[f"pred_{target}"] = t_df[col_name].values
                if f"proba_{target}" in t_df.columns:
                    cand_test[f"proba_{target}"] = t_df[f"proba_{target}"].values
            fold_test_preds.append(cand_test)

        self.estimator_metadata.append({
            "class": "CustomScriptAdapter",
            "script": str(script_path),
            "args": extra_args,
        })

        return self._save_and_finalize(
            truth, details, rows, records, git_hash, test_df=test_df, fold_test_preds=fold_test_preds
        )

    def _resolve_pipeline_stages(self, exec_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve sequential pipeline stages from ``execution.pipeline`` (named stages)
        or ``execution.entrypoints`` (simple list of scripts).

        Returns a list of ``{"name": str, "entrypoint": str, "args": list}`` dicts in
        execution order. Stage names are derived from the entrypoint stem when not
        provided; collisions are disambiguated with the stage index.
        """
        # Variant 2: named stages (execution.pipeline)
        pipeline = exec_cfg.get("pipeline")
        if pipeline:
            if not isinstance(pipeline, list):
                raise ValueError("'execution.pipeline' must be a list of stages")
            stages: List[Dict[str, Any]] = []
            seen: set = set()
            for i, stage in enumerate(pipeline):
                if isinstance(stage, str):
                    entrypoint = stage
                    name = Path(stage).stem or f"stage_{i}"
                    args: List[Any] = []
                elif isinstance(stage, dict):
                    entrypoint = stage.get("entrypoint")
                    if not entrypoint:
                        raise ValueError(
                            f"Pipeline stage at index {i} is missing 'entrypoint'"
                        )
                    name = stage.get("name") or Path(entrypoint).stem or f"stage_{i}"
                    args = list(stage.get("args") or [])
                else:
                    raise ValueError(
                        f"Pipeline stage at index {i} must be a string or a dict with 'entrypoint'"
                    )
                if name in seen:
                    name = f"{name}_{i}"
                seen.add(name)
                stages.append({
                    "name": name,
                    "entrypoint": entrypoint,
                    "args": [str(a) for a in args],
                })
            return stages

        # Variant 1: simple entrypoints list (execution.entrypoints)
        entrypoints = exec_cfg.get("entrypoints")
        if entrypoints:
            if not isinstance(entrypoints, list):
                raise ValueError("'execution.entrypoints' must be a list of script paths")
            stages = []
            seen = set()
            for i, entrypoint in enumerate(entrypoints):
                if not isinstance(entrypoint, str):
                    raise ValueError(f"Pipeline entrypoint at index {i} must be a string")
                name = Path(entrypoint).stem or f"stage_{i}"
                if name in seen:
                    name = f"{name}_{i}"
                seen.add(name)
                stages.append({"name": name, "entrypoint": entrypoint, "args": []})
            return stages

        return []

    def _run_pipeline(self, df, val_strategy, git_hash):
        """Execute a sequential multi-stage pipeline (train/blend/stack) with per-stage audit.

        Each stage is a separate subprocess that shares the experiment directory, so later
        stages can read artifacts produced by earlier stages. Every stage's stdout/stderr is
        captured to ``stages/<name>.log`` and its exit code/duration recorded in
        ``stage_manifest.json``. Execution is fail-fast: the first non-zero exit stops the
        pipeline. The final stage must produce ``oof_predictions.csv`` (and optionally
        ``test_predictions.csv``), which the harness verifies before finalizing.
        """
        import subprocess

        exec_cfg = self.cfg.get("execution", {})
        stages = self._resolve_pipeline_stages(exec_cfg)
        if not stages:
            raise ValueError(
                "Pipeline execution requires 'execution.pipeline' (list of stages) or "
                "'execution.entrypoints' (list of scripts) in the experiment config"
            )

        # 1. Resolve validation splits (mirrors _run_custom_script)
        splits = self._resolve_validation_splits(df, val_strategy)
        if "fold" not in df.columns:
            df = df.copy()
            df["fold"] = 0
            for fold, _, dev in splits:
                df.loc[dev.index, "fold"] = fold

        # 2. Resolve test dataframe
        test_df = None
        if self.task_spec.test_path:
            tp = Path(self.task_spec.test_path)
            if not tp.is_absolute():
                tp = self.project_root / tp
            if tp.exists():
                try:
                    test_df = self.task_spec.load_dataset_split("test", project_root=self.project_root)
                except Exception:
                    test_df = None

        # 3. Export splits
        splits_file = self.exp_dir / "splits.tsv"
        id_col = self.task_spec.id_column
        df[[id_col, "fold"]].to_csv(splits_file, sep="\t", index=False)

        # 4. Run each stage in sequence
        stage_root = self.exp_dir / "stages"
        stage_root.mkdir(parents=True, exist_ok=True)
        stage_manifest: List[Dict[str, Any]] = []
        total = len(stages)
        for idx, stage in enumerate(stages):
            name = stage["name"]
            script_path_str = stage["entrypoint"]
            args = stage["args"]

            script_path = Path(script_path_str)
            if not script_path.is_absolute():
                script_path = self.project_root / script_path
            if not script_path.exists():
                raise FileNotFoundError(
                    f"Pipeline stage '{name}' entrypoint '{script_path}' not found"
                )

            child_env = os.environ.copy()
            python_bin = str(Path(sys.executable).parent)
            child_env["PATH"] = python_bin + os.pathsep + child_env.get("PATH", "")
            child_env["HARNESS_EXP_ID"] = self.exp_id
            child_env["HARNESS_EXP_DIR"] = str(self.exp_dir)
            child_env["HARNESS_PROJECT_ROOT"] = str(self.project_root)
            child_env["HARNESS_RUN_CONFIG"] = str(self.exp_dir / "run_config.yaml")
            child_env["HARNESS_TASK_CONFIG"] = str(self.project.task_config_path)
            child_env["HARNESS_SEED"] = str(self.seed)
            if splits_file.exists():
                child_env["HARNESS_SPLIT_FILE"] = str(splits_file)
            child_env["HARNESS_STAGE_NAME"] = name
            child_env["HARNESS_STAGE_INDEX"] = str(idx)
            child_env["HARNESS_STAGE_TOTAL"] = str(total)
            child_env["HARNESS_STAGE_DIR"] = str(stage_root)

            cmd = [
                sys.executable,
                str(script_path),
                "--exp-dir", str(self.exp_dir),
                "--config", str(self.exp_dir / "run_config.yaml"),
            ]
            cmd.extend(args)

            self._progress(f"Pipeline stage {idx + 1}/{total} '{name}': {script_path.name}")
            start_time = time.monotonic()
            res = subprocess.run(
                cmd, cwd=str(self.project_root), env=child_env, capture_output=True, text=True
            )
            duration = time.monotonic() - start_time

            log_file = stage_root / f"{name}.log"
            with open(log_file, "w", encoding="utf-8") as handle:
                handle.write(f"# stage: {name}\n# index: {idx}/{total}\n")
                handle.write(f"# entrypoint: {script_path}\n# duration_seconds: {duration:.3f}\n")
                handle.write(f"# exit_code: {res.returncode}\n\n")
                handle.write("=== STDOUT ===\n")
                handle.write(res.stdout or "")
                handle.write("\n\n=== STDERR ===\n")
                handle.write(res.stderr or "")

            stage_manifest.append({
                "name": name,
                "entrypoint": str(script_path),
                "index": idx,
                "total": total,
                "exit_code": res.returncode,
                "duration_seconds": round(duration, 3),
                "log_file": log_file.relative_to(self.exp_dir).as_posix(),
            })

            if res.returncode != 0:
                err_msg = res.stderr or res.stdout or f"Exited with code {res.returncode}"
                save_json(self.exp_dir / "stage_manifest.json", stage_manifest)
                raise RuntimeError(
                    f"Pipeline stage '{name}' (index {idx + 1}/{total}) failed "
                    f"(exit {res.returncode}):\n{err_msg}"
                )

        # 5. Persist stage manifest
        save_json(self.exp_dir / "stage_manifest.json", stage_manifest)

        # 6. Verify output artifacts (mirrors _run_custom_script)
        oof_path = self.exp_dir / "oof_predictions.csv"
        if not oof_path.exists():
            pred_fallback = self.exp_dir / "predictions.csv"
            if pred_fallback.exists():
                oof_df = pd.read_csv(pred_fallback)
                oof_df.to_csv(oof_path, index=False)
            else:
                raise FileNotFoundError(
                    f"Pipeline must generate '{oof_path.name}' in target experiment dir"
                )

        oof_df = pd.read_csv(oof_path)
        if id_col not in oof_df.columns:
            raise ValueError(f"OOF predictions file missing ID column '{id_col}'")

        truth = df.drop_duplicates(id_col).copy()
        merged_oof = truth[[id_col]].merge(oof_df, on=id_col, how="inner")
        if len(merged_oof) != len(truth):
            raise ValueError(f"OOF predictions cover {len(merged_oof)} rows, expected {len(truth)}")

        variant = "pipeline"
        self.selection_variant = variant
        records = []
        for i in range(len(merged_oof)):
            row = merged_oof.iloc[i]
            rec = {
                id_col: row[id_col],
                "fold": int(row.get("fold", 0)) if "fold" in row and pd.notna(row["fold"]) else 0,
                "input_variant": variant,
            }
            for target in self.task_spec.subtasks.keys():
                col_name = f"pred_{target}" if f"pred_{target}" in row else target
                if col_name in row:
                    rec[f"pred_{target}"] = row[col_name]
                if f"proba_{target}" in row:
                    rec[f"proba_{target}"] = row[f"proba_{target}"]
            records.append(rec)

        total_duration = sum(s["duration_seconds"] for s in stage_manifest)
        details = [{
            "input_variant": variant,
            "dev_fold": "all",
            "train_rows": len(df),
            "dev_rows": len(df),
            "fit_time_seconds": total_duration,
        }]
        rows = [{
            "input_variant": variant,
            "dev_fold": "all",
            "primary_score": 0.0,
        }]

        fold_test_preds = []
        test_pred_path = self.exp_dir / "test_predictions.csv"
        if test_pred_path.exists() and test_df is not None:
            t_df = pd.read_csv(test_pred_path)
            cand_test = {}
            for target in self.task_spec.subtasks.keys():
                col_name = f"pred_{target}" if f"pred_{target}" in t_df.columns else target
                if col_name in t_df.columns:
                    cand_test[f"pred_{target}"] = t_df[col_name].values
                if f"proba_{target}" in t_df.columns:
                    cand_test[f"proba_{target}"] = t_df[f"proba_{target}"].values
            fold_test_preds.append(cand_test)

        self.estimator_metadata.append({
            "class": "PipelineAdapter",
            "stages": stage_manifest,
        })

        return self._save_and_finalize(
            truth, details, rows, records, git_hash, test_df=test_df, fold_test_preds=fold_test_preds
        )

    def _run_model_training(self, df, val_strategy, git_hash):
        variants = self.cfg.get("data", {}).get("input_variants") or list(
            self.task_spec.inputs_dict
        )
        if not variants:
            raise ValueError("No input variants declared")
        selected = self.cfg.get("data", {}).get("selection_variant")
        if len(variants) > 1 and selected not in variants:
            raise ValueError("Multiple variants require data.selection_variant")
        self.selection_variant = selected or variants[0]

        test_df = None
        test_path_val = self.task_spec.test_path
        if test_path_val:
            tp = Path(test_path_val)
            if not tp.is_absolute():
                tp = self.project_root / tp
            if tp.exists():
                try:
                    test_df = self.task_spec.load_dataset_split("test", project_root=self.project_root)
                except Exception:
                    test_df = None

        if test_df is not None:
            id_col = self.task_spec.id_column
            if (
                id_col not in test_df
                or test_df[id_col].isna().any()
                or test_df[id_col].duplicated().any()
            ):
                raise ValueError("Test set requires declared unique, non-null IDs")

        details, rows, records, truths = [], [], [], []
        fold_test_preds = []
        for fold, train, dev in self._resolve_validation_splits(df, val_strategy):
            self._progress(f"Fold {fold} started: train={len(train)}, dev={len(dev)}")
            partitions = [("all", train, dev)]
            partition = self.cfg.get("data", {}).get("partition_column")
            if partition:
                if partition not in df:
                    raise ValueError("Partition column is missing")
                partitions = [
                    (
                        int(p) if isinstance(p, (int, np.integer)) else p,
                        train[train[partition] == p],
                        dev[dev[partition] == p],
                    )
                    for p in sorted(df[partition].dropna().unique())
                    if len(dev[dev[partition] == p])
                ]
            fold_partition_test_preds = []
            for part, tr, dv in partitions:
                if tr.empty:
                    raise ValueError("Partition has no training data")
                limit = self._probe_sample_limit()
                if limit:
                    tr = tr.sample(n=min(len(tr), limit), random_state=self.seed)
                    dv = dv.head(limit)
                truths.append(dv)
                for variant in variants:
                    config = self.cfg.get("model", {})
                    model = self.task_plugin.build_model(
                        self.plugin_context,
                        config.get("name", "baseline"),
                        config.get("params", {}),
                    )
                    if model is None:
                        model = self._get_model_factory()(
                            config.get("name", "baseline"), **config.get("params", {})
                        )
                    if hasattr(model, "seed"):
                        model.seed = self.seed
                    start = time.monotonic()
                    call_model(model, "fit", tr, variant)
                    ExperimentValidator.assert_parameter_limit(
                        model,
                        max_params=self.cfg.get("constraints", {}).get(
                            "max_params_limit"
                        ),
                    )
                    pred = call_model(model, "predict", dv, variant)
                    from harness.evaluation import PredictionBatch

                    if isinstance(pred, PredictionBatch):
                        if pred.ids != dv[self.task_spec.id_column].tolist():
                            raise ValueError("Prediction IDs mismatch")
                        pred = {
                            **pred.labels,
                            **{"proba_" + k: v for k, v in pred.probabilities.items()},
                            **{"classes_" + k: v for k, v in pred.class_order.items()},
                        }
                    pred = self.task_plugin.normalize_predictions(
                        self.plugin_context, pred
                    )
                    pred = self.task_spec.apply_routing(pred, len(dv))

                    if test_df is not None and variant == self.selection_variant:
                        if part == "all":
                            t_df_part = test_df
                        elif partition and partition in test_df:
                            t_df_part = test_df[test_df[partition] == part]
                        else:
                            t_df_part = None

                        if t_df_part is not None and not t_df_part.empty:
                            t_pred = call_model(model, "predict", t_df_part, variant)
                            if isinstance(t_pred, PredictionBatch):
                                if t_pred.ids != t_df_part[self.task_spec.id_column].tolist():
                                    raise ValueError("Test prediction IDs mismatch")
                                t_pred = {
                                    **t_pred.labels,
                                    **{"proba_" + k: v for k, v in t_pred.probabilities.items()},
                                    **{"classes_" + k: v for k, v in t_pred.class_order.items()},
                                }
                            t_pred = self.task_plugin.normalize_predictions(
                                self.plugin_context, t_pred
                            )
                            t_pred = self.task_spec.apply_routing(t_pred, len(t_df_part))
                            if part == "all":
                                fold_test_preds.append(t_pred)
                            else:
                                fold_partition_test_preds.append((part, t_df_part.index, t_pred))

                    evaluation = self._evaluate(dv, pred)
                    part_id = int(part) if isinstance(part, (int, np.integer)) else part
                    fold_id = int(fold) if isinstance(fold, (int, np.integer)) else fold
                    detail = {
                        "experiment_id": self.exp_id,
                        "input_variant": variant,
                        "dev_fold": fold_id,
                        "partition": part_id,
                        "train_rows": len(tr),
                        "dev_rows": len(dv),
                        "fit_time_seconds": time.monotonic() - start,
                        **evaluation,
                    }
                    details.append(detail)
                    self._progress(f"Fold {fold}, partition={part}, input={variant} completed: "
                                   f"{self.task_spec.primary_metric}={evaluation['primary_score']:.6f}")
                    rows.append(
                        {
                            k: (
                                int(v)
                                if isinstance(v, (int, np.integer))
                                else float(v)
                                if isinstance(v, (float, np.floating))
                                else v
                            )
                            for k, v in detail.items()
                            if isinstance(v, (str, int, float, np.integer, np.floating))
                        }
                    )
                    records.extend(self._record_predictions(dv, pred, fold_id, variant))
                    self.estimator_metadata.append(
                        getattr(
                            model, "estimator_metadata", {"class": type(model).__name__}
                        )
                    )
            if fold_partition_test_preds and test_df is not None:
                combined_fold_test = {}
                for target, st in self.task_spec.subtasks.items():
                    if any("proba_" + target in p[2] for p in fold_partition_test_preds):
                        sample_proba = next(p[2]["proba_" + target] for p in fold_partition_test_preds if "proba_" + target in p[2])
                        n_cls = sample_proba.shape[1] if hasattr(sample_proba, "shape") and len(sample_proba.shape) > 1 else len(sample_proba[0])
                        full_proba = np.zeros((len(test_df), n_cls), dtype=float)
                        for _, idxs, p_dict in fold_partition_test_preds:
                            if "proba_" + target in p_dict:
                                locs = test_df.index.get_indexer(idxs)
                                full_proba[locs] = p_dict["proba_" + target]
                        combined_fold_test["proba_" + target] = full_proba
                    if any("classes_" + target in p[2] for p in fold_partition_test_preds):
                        combined_fold_test["classes_" + target] = next(p[2]["classes_" + target] for p in fold_partition_test_preds if "classes_" + target in p[2])

                    full_preds = [None] * len(test_df)
                    for _, idxs, p_dict in fold_partition_test_preds:
                        sub_preds = p_dict.get("pred_" + target, p_dict.get(target))
                        if sub_preds is not None:
                            locs = test_df.index.get_indexer(idxs)
                            for loc, val in zip(locs, sub_preds):
                                full_preds[loc] = val
                    combined_fold_test["pred_" + target] = full_preds
                fold_test_preds.append(combined_fold_test)
        truth = pd.concat(truths).drop_duplicates(self.task_spec.id_column)
        return self._save_and_finalize(
            truth, details, rows, records, git_hash, test_df=test_df, fold_test_preds=fold_test_preds
        )

    def _run_llm_prompting(self, df, git_hash):
        from harness.llm_engine import LLMEngine

        config = self.cfg.get("llm", {})
        records, details, truths = [], [], []
        self.selection_variant = "llm_prompt"
        strategy = self.cfg.get("validation", {}).get(
            "strategy", self.task_spec.validation.strategy
        )
        targets = {s.column or key for key, s in self.task_spec.subtasks.items()}
        targets.update((s.column_prefix or "") + label for s in self.task_spec.subtasks.values()
                       if s.task_type == "multilabel_classification" for label in s.labels)
        targets.update(c for s in self.task_spec.subtasks.values() if s.column_prefix
                       for c in df.columns if c.startswith(s.column_prefix))
        inputs = {column for variant in self.task_spec.inputs_dict.values()
                  for column in variant.source_columns}
        if inputs & targets:
            raise ValueError("Declared LLM inputs overlap ground-truth target columns")
        from string import Formatter

        template_fields = {field for template in (config.get("prompt_template", "{text}"),
                           config.get("system_prompt", "")) for _, field, _, _ in Formatter().parse(template)
                           if field}
        if template_fields & targets:
            raise ValueError("LLM prompt references ground-truth labels")
        for fold, train, dev in self._resolve_validation_splits(df, strategy):
            self._progress(f"LLM fold {fold} started: dev={len(dev)}")
            limit = self._probe_sample_limit()
            if limit:
                dev = dev.head(limit)
            truths.append(dev)
            predictions = {key: [] for key in self.task_spec.subtasks}
            for _, row in dev.iterrows():
                features = {k: row[k] for k in inputs if k in row and k not in targets}
                system, user = LLMEngine.format_prompt(
                    config.get("prompt_template", "{text}"),
                    features,
                    config.get("system_prompt", ""),
                    config.get("few_shot_examples", []),
                    config.get("use_cot", False),
                )
                if any("{" + c + "}" in user for c in targets):
                    raise ValueError("LLM prompt references ground-truth labels")
                response = LLMEngine.generate_completion(
                    provider=config.get("provider", "gemini"),
                    model_name=config.get("model_name", "gemini-2.0-flash"),
                    system_prompt=system,
                    user_prompt=user,
                    temperature=config.get("temperature", 0),
                    max_tokens=config.get("max_tokens", 512),
                    top_p=config.get("top_p", 0.95),
                    base_url=config.get("api_base_url"),
                )
                text = response.get("text", "").strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    raise ValueError("LLM response is not valid JSON")
                if not isinstance(parsed, dict):
                    raise ValueError("LLM output must be an object")
                for key, st in self.task_spec.subtasks.items():
                    value = next(
                        (parsed[k] for k in [key, st.column] if k and k in parsed), None
                    )
                    if value is None and len(predictions) == 1:
                        value = parsed.get("prediction", parsed.get("label"))
                    if value is None:
                        raise ValueError(f"LLM output omitted {key}")
                    predictions[key].append(value)
            predictions = self.task_spec.apply_routing(predictions, len(dev))
            evaluation = self._evaluate(dev, predictions)
            self._progress(f"LLM fold {fold} completed: {self.task_spec.primary_metric}={evaluation['primary_score']:.6f}")
            fold_id = int(fold) if isinstance(fold, (int, np.integer)) else fold
            details.append(
                {
                    "experiment_id": self.exp_id,
                    "input_variant": "llm_prompt",
                    "dev_fold": fold_id,
                    **evaluation,
                }
            )
            records.extend(
                self._record_predictions(dev, predictions, fold_id, "llm_prompt")
            )
        return self._save_and_finalize(
            pd.concat(truths).drop_duplicates(self.task_spec.id_column),
            details,
            [],
            records,
            git_hash,
        )

    def _save_and_finalize(self, truth, details, rows, records, git_hash, test_df=None, fold_test_preds=None):
        oof = pd.DataFrame(records)
        variants = {}
        for variant in oof.input_variant.unique():
            selected = oof[oof.input_variant == variant].copy()
            key = self.task_spec.id_column
            if (
                self.run_kind == "full"
                and set(selected[key]) != self.expected_validation_ids
            ):
                raise ValueError("Full validation prediction coverage is incomplete")
            if selected[key].duplicated().any():
                raise ValueError("Duplicate OOF IDs within input variant")
            merged = selected[[key]].merge(
                truth, on=key, how="left", validate="one_to_one"
            )
            predictions = {}
            for col in selected:
                if col.startswith("pred_"):
                    values = selected[col].tolist()
                    predictions[col] = [
                        json.loads(v) if isinstance(v, str) and v.startswith("[") else v
                        for v in values
                    ]
                elif col.startswith("proba_"):
                    if any(f"proba_{target}" == col for target in self.task_spec.subtasks.keys()):
                        predictions[col] = [
                            json.loads(v) if isinstance(v, str) and (v.startswith("[") or v.startswith("{")) else v
                            for v in selected[col]
                        ]
                elif col.startswith("classes_"):
                    if selected[col].nunique() != 1:
                        raise ValueError("Class order changed across folds")
                    predictions[col] = json.loads(selected[col].iloc[0])
            variants[variant] = self._evaluate(merged, predictions)
        aggregate = variants[self.selection_variant]
        model_name = self.cfg.get("model", {}).get("name", "baseline")
        model_family = self.cfg.get("model", {}).get("family")
        if not model_family:
            m_lower = str(model_name).lower()
            if any(k in m_lower for k in ["lightgbm", "lgbm", "xgboost", "xgb", "catboost"]):
                model_family = "gbdt"
            elif any(k in m_lower for k in ["random_forest", "randomforest", "rf"]):
                model_family = "random_forest"
            elif any(k in m_lower for k in ["roberta", "deberta", "bert"]):
                model_family = "transformer"
            elif any(k in m_lower for k in ["llm", "llama", "qwen"]):
                model_family = "llm"
            else:
                model_family = self.cfg.get("task", {}).get("model_family") or "linear"

        result = {
            "schema_version": 2,
            "experiment_id": self.exp_id,
            "selection_variant": self.selection_variant,
            "model": model_name,
            "model_family": model_family,
            **aggregate,
            "variants": variants,
            "fold_metrics": details,
        }
        save_json(self.exp_dir / "metrics.json", result)
        summary = pd.DataFrame(
            rows
            or [
                {
                    "input_variant": d["input_variant"],
                    "dev_fold": (
                        int(d["dev_fold"])
                        if isinstance(d["dev_fold"], (int, np.integer))
                        else d["dev_fold"]
                    ),
                    "primary_score": d["primary_score"],
                }
                for d in details
            ]
        )
        summary.to_csv(self.exp_dir / "summary.csv", index=False)
        # predictions.csv is a byte-identical copy of the OOF frame: store it as
        # a hardlink mirror (0 extra bytes), falling back to a plain write when
        # hardlinks are unavailable.
        oof_path = self.exp_dir / "oof_predictions.csv"
        oof.to_csv(oof_path, index=False)
        if not mirror_csv(oof_path, self.exp_dir / "predictions.csv"):
            oof.to_csv(self.exp_dir / "predictions.csv", index=False)
        # evaluation_truth.csv is shared by every run: mirror a content-addressed
        # canonical master so the truth bytes are stored once per truth content.
        canonical_truth = ensure_canonical_truth(self.artifacts_dir, truth)
        if not mirror_csv(canonical_truth, self.exp_dir / "evaluation_truth.csv"):
            truth.to_csv(self.exp_dir / "evaluation_truth.csv", index=False)
        try:
            oof.to_parquet(self.exp_dir / "oof_predictions.parquet", index=False)
        except Exception:
            pass

        if test_df is not None and fold_test_preds:
            id_col = self.task_spec.id_column
            final_test_pred = {}
            for target, st in self.task_spec.subtasks.items():
                class_order = None
                orders = [
                    (json.loads(f["classes_" + target]) if isinstance(f.get("classes_" + target), str) else f.get("classes_" + target))
                    for f in fold_test_preds
                    if "classes_" + target in f
                ]
                if orders:
                    class_order = orders[0]
                    final_test_pred["classes_" + target] = class_order
                elif st.classes:
                    class_order = list(st.classes)
                    final_test_pred["classes_" + target] = class_order

                all_have_proba = all("proba_" + target in f for f in fold_test_preds)
                if all_have_proba:
                    probas = [
                        np.asarray([json.loads(v) if isinstance(v, str) else v for v in f["proba_" + target]], dtype=float)
                        for f in fold_test_preds
                    ]
                    avg_proba = np.mean(probas, axis=0)
                    final_test_pred["proba_" + target] = avg_proba
                    if st.task_type in ("regression", "ranking"):
                        preds = [np.asarray(f.get("pred_" + target, f.get(target)), dtype=float) for f in fold_test_preds]
                        final_test_pred["pred_" + target] = np.mean(preds, axis=0).tolist()
                    elif st.task_type == "multilabel_classification":
                        final_test_pred["pred_" + target] = (avg_proba > 0.5).astype(int).tolist()
                    else:
                        if class_order and len(class_order) == avg_proba.shape[1]:
                            final_test_pred["pred_" + target] = np.asarray(class_order)[avg_proba.argmax(axis=1)].tolist()
                        else:
                            final_test_pred["pred_" + target] = avg_proba.argmax(axis=1).tolist()
                else:
                    if st.task_type in ("regression", "ranking"):
                        preds = [np.asarray(f.get("pred_" + target, f.get(target)), dtype=float) for f in fold_test_preds]
                        final_test_pred["pred_" + target] = np.mean(preds, axis=0).tolist()
                    elif st.task_type == "multilabel_classification":
                        preds = [
                            np.asarray([json.loads(v) if isinstance(v, str) else v for v in f.get("pred_" + target, f.get(target))], dtype=float)
                            for f in fold_test_preds
                        ]
                        avg = np.mean(preds, axis=0)
                        final_test_pred["pred_" + target] = (avg > 0.5).astype(int).tolist()
                    else:
                        preds = [f.get("pred_" + target, f.get(target)) for f in fold_test_preds]
                        order = list(class_order) if class_order else []
                        n_samples = len(test_df)
                        voted = []
                        for i in range(n_samples):
                            s_preds = [p[i] for p in preds]
                            if order:
                                counts = {c: s_preds.count(c) for c in order}
                                voted.append(max(order, key=lambda c: counts[c]))
                            else:
                                from collections import Counter
                                voted.append(Counter(s_preds).most_common(1)[0][0])
                        final_test_pred["pred_" + target] = voted

            test_out = pd.DataFrame({id_col: test_df[id_col].values})
            for target, st in self.task_spec.subtasks.items():
                if "proba_" + target in final_test_pred:
                    test_out["proba_" + target] = [json.dumps(np.asarray(x).tolist()) for x in final_test_pred["proba_" + target]]
                    avg_p = np.asarray(final_test_pred["proba_" + target])
                    c_order = final_test_pred.get("classes_" + target)
                    if hasattr(avg_p, "ndim") and avg_p.ndim > 1:
                        if c_order and len(c_order) == avg_p.shape[1]:
                            for c_idx, c_label in enumerate(c_order):
                                clean_lbl = str(c_label).replace(" ", "_")
                                test_out[f"proba_{target}_{clean_lbl}"] = avg_p[:, c_idx]
                        else:
                            for c_idx in range(avg_p.shape[1]):
                                test_out[f"proba_{target}_{c_idx}"] = avg_p[:, c_idx]
                        if avg_p.shape[1] == 2 and f"proba_{target}_1" not in test_out.columns:
                            test_out[f"proba_{target}_1"] = avg_p[:, 1]
                if "classes_" + target in final_test_pred:
                    test_out["classes_" + target] = json.dumps(np.asarray(final_test_pred["classes_" + target]).tolist())
                p_val = final_test_pred["pred_" + target]
                if isinstance(p_val, (list, np.ndarray)) and st.task_type == "multilabel_classification":
                    test_out["pred_" + target] = [json.dumps(np.asarray(x).tolist()) for x in p_val]
                elif isinstance(p_val, (list, np.ndarray)):
                    test_out["pred_" + target] = np.asarray(p_val).tolist()
                else:
                    test_out["pred_" + target] = p_val.item() if hasattr(p_val, "item") else p_val
                raw_col = st.column or target
                if raw_col not in test_out:
                    test_out[raw_col] = test_out["pred_" + target]

            test_out.to_csv(self.exp_dir / "test_predictions.csv", index=False)
            self.manifest["prediction_splits"] = {"test_predictions.csv": "test"}

        (self.exp_dir / "EXPERIMENT_CARD.md").write_text(
            f"# {self.exp_id}\n\n{aggregate['primary_metric']}: {aggregate['primary_score']} ({aggregate['direction']})\n",
            encoding="utf-8",
        )
        try:
            DiagnosticEngine.save_diagnostics_bundle(
                truth,
                oof[oof.input_variant == self.selection_variant],
                self.exp_dir,
                self.exp_id,
                task_spec=self.task_spec,
            )
            result["diagnostics_status"] = "COMPLETED"
        except Exception as error:
            result["diagnostics_status"] = "FAILED"
            result["diagnostics_error"] = str(error)
        save_json(self.exp_dir / "metrics.json", result)
        self.manifest.update(
            status="COMPLETED",
            estimator=self.estimator_metadata,
            files={
                p.name: sha256(p)
                for p in self.exp_dir.iterdir()
                if p.is_file() and p.name != "artifact_manifest.json"
            },
        )
        save_json(self.exp_dir / "artifact_manifest.json", self.manifest)
        if self.run_kind == "full":
            journal = TreeJournal(
                journal_path=self.artifacts_dir / "journal.json",
                artifacts_dir=self.artifacts_dir,
                task_spec=self.task_spec,
                task_plugin=self.task_plugin,
            )
            journal.sync_from_artifacts()
        return result


Runner = UniversalRunner


def build_parser() -> argparse.ArgumentParser:
    """Build and return CLI argument parser for universal training runner."""
    parser = argparse.ArgumentParser(
        description="Universal Competition Experiment Runner"
    )
    parser.add_argument("--unbuffered", action="store_true",
                        help="Flush redirected stdout/stderr immediately; fold progress always flushes")
    parser.add_argument(
        "--project-root", type=str, default=None, help="Competition workspace root"
    )
    parser.add_argument(
        "--task-config", type=str, default=None, help="Task specification YAML"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Persistent dashboard/Agent Bridge job ID",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--base-config", type=str, default=None, help="Path to base config YAML"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Target fold index to train (smoke test / probe)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode with subsampled data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Accepted for backwards compatibility; sealed experiment "
            "artifacts are NEVER overwritten. Use a new experiment ID "
            "(see --force-import for the separate import-mismatch override)."
        ),
    )
    parser.add_argument(
        "--force-import",
        action="store_true",
        help="Force import Kaggle artifacts even if non-critical identity mismatches occur",
    )
    parser.add_argument(
        "--probe-samples", type=int, default=None, help="Probe sample count limit"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["local", "modal", "kaggle"],
        help=(
            "Execution backend. Omit to use the task contract default "
            "(task.execution.backend) when declared, else 'local'. A locked "
            "contract (task.execution.locked: true) rejects other backends."
        ),
    )
    parser.add_argument(
        "--skip-leaderboard",
        action="store_true",
        help="Do not write the project leaderboard",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(cloud backends only) Build and print the dispatch plan without pushing or consuming cloud quota",
    )
    return parser


def resolve_cli_backend(
    requested: Optional[str],
    contract_backend: Optional[str],
    backend_locked: bool,
    is_smoke_run: bool,
) -> str:
    """Resolve the effective execution backend for the runner CLI.

    Explicit ``--backend`` wins; otherwise the task contract's declared
    backend applies; otherwise local. Smoke tests and bounded probes always
    run locally. A locked contract rejects a different backend for FULL runs
    (raises ValueError; the CLI translates that into a parser error).
    """
    if requested is None:
        if is_smoke_run:
            return "local"
        return (contract_backend or "local").strip().lower()
    requested = requested.strip().lower()
    if (
        backend_locked
        and contract_backend
        and requested != contract_backend
        and not is_smoke_run
    ):
        raise ValueError(
            f"Backend lock violation: the task contract locks the execution "
            f"backend to '{contract_backend}' (execution.locked: true) but "
            f"'{requested}' was requested. Drop the --backend flag "
            f"(the contract default applies) or change the contract."
        )
    return requested


def is_local_only_run(args, cfg_doc: Optional[Dict[str, Any]] = None) -> bool:
    """Whether this CLI invocation is a sample probe that must stay local.

    ``--debug``, ``--fold``, ``--probe-samples``, or a ``validation.sample_limit``
    in the experiment config all make the run a sample probe. Probes must never
    dispatch to the cloud backend: CORRECTNESS_CONTRACT §3 (sample probes never
    enter the official solution tree or candidate pool) and AGENTS.md §2.4
    (smoke/probes always run locally) — regardless of the contract backend lock.
    """
    if bool(getattr(args, "debug", False)) or getattr(args, "fold", None) is not None:
        return True
    if getattr(args, "probe_samples", None):
        return True
    if cfg_doc:
        limit = (cfg_doc.get("validation") or {}).get("sample_limit")
        if limit:
            return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for universal training runner."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.unbuffered:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(line_buffering=True, write_through=True)

    project = HarnessProject.discover(args.project_root, args.task_config)
    from harness.toolchain import ensure_pinned_toolchain

    _pinned = ensure_pinned_toolchain(project.root, "harness.runner", sys.argv[1:])
    if _pinned is not None:
        return int(_pinned.returncode)
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

    # Execution-backend contract: the task spec may declare a default backend
    # (task.execution.backend) and lock it (task.execution.locked). Enforcing
    # here — not only in the agent bridge — stops silent local training when
    # the runner CLI is invoked directly (agent terminal calls, scripts).
    contract_backend = None
    backend_locked = False
    try:
        from harness.task_spec import load_active_task_spec

        _spec = load_active_task_spec(project.task_config_path)
        contract_backend = getattr(_spec, "execution_backend", None)
        backend_locked = bool(getattr(_spec, "execution_locked", False))
    except Exception:
        pass  # spec load failure surfaces in UniversalRunner with full context
    requested_backend = args.backend
    # Probe detection must see validation.sample_limit, which lives in the
    # experiment config; load it here (UniversalRunner re-validates it).
    cfg_doc: Optional[Dict[str, Any]] = None
    try:
        cfg_doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg_doc, dict):
            cfg_doc = None
    except Exception:
        cfg_doc = None  # config errors surface in UniversalRunner with context
    is_smoke_run = is_local_only_run(args, cfg_doc)
    # Re-entrancy guard: when this process was started by a dispatched remote
    # kernel (Kaggle/Modal template sets AI_HARNESS_REMOTE_KERNEL=1), we ARE
    # the training environment. The task contract's cloud backend must not be
    # re-applied here — it would dispatch a nested kernel from inside the
    # kernel (and fail the upload-consent check, which only exists locally).
    inside_remote_kernel = os.environ.get("AI_HARNESS_REMOTE_KERNEL") == "1"
    if inside_remote_kernel:
        requested_backend = None
        contract_backend = None
        backend_locked = False
        print(
            "[ai-harness] inside a dispatched remote kernel "
            "(AI_HARNESS_REMOTE_KERNEL=1); executing locally instead of "
            "re-dispatching to the contract backend."
        )
    try:
        args.backend = resolve_cli_backend(
            requested_backend, contract_backend, backend_locked, is_smoke_run
        )
    except ValueError as exc:
        parser.error(str(exc))
    if requested_backend is None and contract_backend and not is_smoke_run:
        print(
            f"[ai-harness] backend from task contract: {args.backend}"
            + (" (execution.locked: true)" if backend_locked else "")
        )
    elif is_smoke_run and backend_locked and contract_backend:
        print(
            f"[ai-harness] smoke run staying local despite backend lock "
            f"(contract backend: {contract_backend})"
        )

    # If backend is modal or kaggle, delegate to cloud runner
    if args.backend == "modal":
        from harness.cloud_runner import run_via_modal

        run_via_modal(
            config_path=str(config_path),
            project_root=project.root,
            job_id=getattr(args, "job_id", None),
        )
        return 0
    elif args.backend == "kaggle":
        from harness.cloud_runner import run_via_kaggle

        run_via_kaggle(
            config_path=str(config_path),
            project_root=project.root,
            job_id=getattr(args, "job_id", None),
            force_import=getattr(args, "force_import", False),
            dry_run=getattr(args, "dry_run", False),
        )
        return 0

    # Local execution
    package_base = Path(__file__).resolve().parent / "defaults" / "base.yaml"
    workspace_base = project.root / "configs" / "base.yaml"
    base_path = (
        resolve_project_path(args.base_config)
        if args.base_config
        else (workspace_base if workspace_base.is_file() else package_base)
    )
    base_cfg = load_yaml_config(base_path) if base_path.exists() else {}
    exp_cfg = load_yaml_config(config_path)
    cfg = merge_configs(base_cfg, exp_cfg)

    exp_id = cfg.get("experiment", {}).get("id", config_path.stem)
    args.project_root = str(project.root)
    args.task_config = str(project.task_config_path)
    # Worker state belongs to dashboard job IDs (job_...). Remote dispatch
    # identities (e.g. kaggle_<exp>_<hex>) only bind the artifact run_id and
    # must not touch worker state.
    worker_job_id = None
    if args.job_id:
        from harness.job_state import JOB_ID_PATTERN

        if JOB_ID_PATTERN.fullmatch(str(args.job_id)):
            worker_job_id = str(args.job_id)
        else:
            print(
                f"[ai-harness] job-id '{args.job_id}' is not a dashboard worker "
                "job id; skipping worker state update"
            )
    try:
        runner = UniversalRunner(cfg=cfg, config_path=str(config_path), args=args)
        runner.run()
    except Exception as exc:
        if worker_job_id:
            from harness.job_state import update_worker_state

            update_worker_state(
                project.root,
                worker_job_id,
                status="FAILED",
                provider_status="FAILED",
                error_message=str(exc),
                exp_id=str(exp_id),
                config_sha256=__import__("hashlib")
                .sha256(config_path.read_bytes())
                .hexdigest(),
            )
        raise
    if worker_job_id:
        import json
        from harness.job_state import update_worker_state

        exp_dir = runner.exp_dir
        metrics_path = exp_dir / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else {}
        )
        files = (
            sorted(path for path in exp_dir.iterdir() if path.is_file())
            if exp_dir.is_dir()
            else []
        )
        update_worker_state(
            project.root,
            worker_job_id,
            status="COMPLETED",
            provider_status="COMPLETED",
            result_metrics=metrics if isinstance(metrics, dict) else {"folds": metrics},
            artifact_paths=[
                path.relative_to(project.root).as_posix() for path in files
            ],
            exp_id=str(exp_id),
            config_sha256=__import__("hashlib")
            .sha256(config_path.read_bytes())
            .hexdigest(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
