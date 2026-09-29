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


class GenericBaselineModel:
    """Task-agnostic baseline model implementing sklearn Pipeline for text or tabular data."""

    def __init__(self, task_spec: UniversalTaskSpec, model_name: str = "baseline", **kwargs: Any):
        self.task_spec = task_spec
        self.model_name = model_name
        self.kwargs = kwargs
        self.fitted_models: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, variant: Optional[str] = None) -> GenericBaselineModel:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        subtasks = self.task_spec.subtasks
        is_text = self.task_spec.modality == "text" or any(inp.type in ("text", "text_template") for inp in self.task_spec.inputs)

        items_to_fit = subtasks.items() if subtasks else [("target", None)]
        for st_id, st in items_to_fit:
            target_col = st.column if st and st.column else (st_id if st_id in df.columns else None)
            if not target_col or target_col not in df.columns:
                continue

            valid_df = df.dropna(subset=[target_col])
            if len(valid_df) == 0:
                continue
            y = valid_df[target_col]

            if is_text:
                if variant and variant in self.task_spec.inputs_dict:
                    X_text = self.task_spec.inputs_dict[variant].compose_text(valid_df)
                else:
                    text_cols = [c for c in valid_df.columns if c != target_col and valid_df[c].dtype == object]
                    X_text = valid_df[text_cols[0]].fillna("").astype(str) if text_cols else valid_df.iloc[:, 0].astype(str)

                is_reg = getattr(st, "task_type", "") == "regression"
                clf = Ridge() if is_reg else LogisticRegression(max_iter=500, class_weight="balanced")
                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                    ("clf", clf),
                ])
                pipe.fit(X_text, y)
                self.fitted_models[target_col] = (pipe, "text")
            else:
                num_cols = [c for c in valid_df.columns if c != target_col and c not in ("fold", getattr(self.task_spec, "id_column", "id")) and valid_df[c].dtype != object]
                if not num_cols:
                    num_cols = [c for c in valid_df.columns if c != target_col and c not in ("fold", getattr(self.task_spec, "id_column", "id"))]
                is_reg = getattr(st, "task_type", "") == "regression"
                clf = Ridge() if is_reg else LogisticRegression(max_iter=500)
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ])
                pipe.fit(valid_df[num_cols], y)
                self.fitted_models[target_col] = (pipe, "tabular", num_cols)

        return self

    def predict(self, df: pd.DataFrame, variant: Optional[str] = None) -> Dict[str, Any]:
        preds = {}
        for target_col, meta in self.fitted_models.items():
            pipe = meta[0]
            mod_type = meta[1]
            if mod_type == "text":
                if variant and variant in self.task_spec.inputs_dict:
                    X_text = self.task_spec.inputs_dict[variant].compose_text(df)
                else:
                    text_cols = [c for c in df.columns if c != target_col and df[c].dtype == object]
                    X_text = df[text_cols[0]].fillna("").astype(str) if text_cols else df.iloc[:, 0].astype(str)
                preds[f"pred_{target_col}"] = pipe.predict(X_text)
            else:
                num_cols = meta[2]
                preds[f"pred_{target_col}"] = pipe.predict(df[num_cols])
        return preds


class UniversalRunner:
    """Orchestrates model execution, validation, metrics calculation, and artifact logging."""

    def __init__(self, cfg: Dict[str, Any], config_path: str, args: argparse.Namespace):
        self.cfg = cfg
        self.config_path = config_path
        self.args = args
        task_config = cfg.get("paths", {}).get("task_config") or getattr(args, "task_config", None)
        self.project = HarnessProject.discover(getattr(args, "project_root", None), task_config)
        self.project_root = self.project.root
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        exp_info = cfg.get("experiment", {})
        self.exp_id = exp_info.get("id", Path(config_path).stem)
        paths = cfg.get("paths", {})
        # An attached workspace owns all run output paths. This prevents an
        # experiment config copied from another harness checkout from writing
        # into a shared `artifacts/` directory or the installed package.
        artifacts_root = self.project.artifacts_dir if self.project.manifest else Path(
            paths.get("artifacts_dir") or self.project.artifacts_dir
        )
        if not artifacts_root.is_absolute():
            artifacts_root = self.project_root / artifacts_root
        self.artifacts_dir = artifacts_root.resolve()
        self.exp_dir = artifacts_root / self.exp_id

        # Task specification
        self.task_spec = load_active_task_spec(self.project.task_config_path)
        self.plugin_context = TaskPluginContext(
            project_root=self.project_root,
            task_config_path=self.project.task_config_path,
            task_spec=self.task_spec,
            plugin_id=self.project.plugin_id or "default",
            run_config=self.cfg,
        )
        self.task_plugin = load_task_plugin(self.project.plugin_id, self.plugin_context)

        # Logger
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.exp_dir / "run.log"
        self.logger = setup_logger(self.log_file)

    def run(self) -> Dict[str, Any]:
        """Execute full experiment pipeline."""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting Experiment: {self.exp_id}")
        self.logger.info(f"Task: {self.task_spec.name} (Modality: {self.task_spec.metadata.get('modality', 'generic')})")
        self.logger.info(f"Config path: {self.config_path}")
        self.logger.info("=" * 60)

        # 1. Epistemic Grounding Check
        passed, grounding_errors, receipt = GroundingGate.verify_config(
            self.cfg, config_path=self.config_path, raise_on_error=True
        )
        GroundingGate.write_receipt(receipt, self.exp_dir / "GROUNDING_RECEIPT.json")

        # 2. Reproducibility metadata
        seed = self.cfg.get("runtime", {}).get("seed", 42)
        set_seed(seed)
        git_hash = get_git_commit_hash()
        self.logger.info(f"Git commit hash: {git_hash} | Seed: {seed}")

        runtime_meta = {
            "git_commit_hash": git_hash,
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
        }
        self.cfg["resolved_runtime"] = runtime_meta
        with open(self.exp_dir / "run_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

        # 3. Load Datasets via task adapter or task spec
        df = None
        adapter_fn = self.cfg.get("paths", {}).get("data_loader")
        if adapter_fn:
            try:
                import importlib
                mod_name, fn_name = adapter_fn.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                df_res = getattr(mod, fn_name)(str(self.project_root / "dataset"))
                df = df_res[0] if isinstance(df_res, tuple) else df_res
            except Exception as e:
                raise RuntimeError(f"Configured task data loader '{adapter_fn}' failed: {e}") from e

        if df is None:
            df = self.task_plugin.load_training_data(self.plugin_context)
            if isinstance(df, tuple):
                df = df[0]

        if df is None:
            df = self.task_spec.load_dataset_split("train", project_root=self.project_root)
            split_cand = self.cfg.get("paths", {}).get("split_path")
            split_path = self.project_root / split_cand if split_cand else None
            if not split_path or not split_path.exists():
                for cand in ["splits/fold_assignments.tsv", "splits/folds.csv", "splits/splits.csv"]:
                    if (self.project_root / cand).exists():
                        split_path = self.project_root / cand
                        break
            if split_path and split_path.exists() and "fold" not in df.columns:
                try:
                    sep = "\t" if split_path.suffix.lower() in (".tsv", ".tab") else ","
                    splits_df = pd.read_csv(split_path, sep=sep)
                    id_cand = [c for c in ["id", self.task_spec.id_column] if c and c in df.columns and c in splits_df.columns]
                    if id_cand:
                        df = df.merge(splits_df[[id_cand[0], "fold"]], on=id_cand[0], how="left")
                    else:
                        common_cols = [c for c in df.columns if c in splits_df.columns and c != "fold"]
                        if common_cols:
                            df = df.merge(splits_df[common_cols + ["fold"]], on=common_cols[0], how="left")
                except Exception as e:
                    self.logger.warning(f"Could not merge splits from {split_path}: {e}")

        self.logger.info(f"Loaded {len(df):,} training instances.")
        self.task_plugin.validate_training_data(self.plugin_context, df)

        # Subsample for debug/probe (group-aware to guarantee anti-leakage invariant)
        if self.args.debug or getattr(self.args, "probe_samples", None):
            sample_n = getattr(self.args, "probe_samples", None) or 50
            self.logger.warning(f"Probe/Debug mode enabled: sampling {sample_n} instances.")
            group_col = self.task_spec.group_column
            partition_col = (
                self.cfg.get("data", {}).get("partition_column")
                or self.task_spec.metadata.get("data_config", {}).get("partition_column")
            )
            if group_col and group_col in df.columns:
                sampled_groups = set()
                if partition_col and partition_col in df.columns:
                    for _, partition_df in df.groupby(partition_col, dropna=False):
                        for grp in partition_df[group_col].dropna().unique():
                            sampled_groups.add(grp)
                            if len(partition_df[partition_df[group_col].isin(sampled_groups)]) >= sample_n:
                                break
                else:
                    for grp in df[group_col].dropna().unique():
                        sampled_groups.add(grp)
                        if len(df[df[group_col].isin(sampled_groups)]) >= sample_n:
                            break
                df = df[df[group_col].isin(sampled_groups)].reset_index(drop=True)
            elif partition_col and partition_col in df.columns:
                df = df.groupby(partition_col, dropna=False).head(sample_n).reset_index(drop=True)
            else:
                df = df.head(sample_n).reset_index(drop=True)

        # 4. Resolve Approach and Validation Strategy
        exp_approach = self.cfg.get("experiment", {}).get("approach", "ml_classical")
        val_cfg = self.cfg.get("validation", {})
        val_strategy = val_cfg.get("strategy") or getattr(self.task_spec.validation, "strategy", "kfold")

        if exp_approach in ("llm_prompt", "llm_peft"):
            # LLMs default to holdout unless explicitly requested otherwise
            if "strategy" not in val_cfg:
                val_strategy = "holdout"

        self.logger.info(f"Approach: {exp_approach} | Validation Strategy: {val_strategy}")

        # Execute according to approach
        if exp_approach == "llm_prompt":
            return self._run_llm_prompting(df, git_hash)
        else:
            return self._run_model_training(df, val_strategy, git_hash)

    def _get_model_factory(self) -> Any:
        """Resolve an explicitly configured factory or the generic built-in."""
        factory_path = self.cfg.get("model", {}).get("factory")
        if factory_path:
            import importlib
            mod_name, fn_name = factory_path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, fn_name)

        return lambda name, **kwargs: GenericBaselineModel(self.task_spec, model_name=name, **kwargs)

    def _run_model_training(self, df: pd.DataFrame, val_strategy: str, git_hash: str) -> Dict[str, Any]:
        """Execute classical ML or Deep Learning training loop."""
        model_factory = self._get_model_factory()

        model_cfg = self.cfg.get("model", {})
        model_name = model_cfg.get("name", "tfidf_logistic_regression")
        model_params = model_cfg.get("params", {})
        input_variants = self.cfg.get("data", {}).get("input_variants")
        if not input_variants:
            input_variants = [item.name for item in self.task_spec.inputs] or ["default"]

        id_col = self.task_spec.id_column
        if id_col not in df.columns:
            id_col = df.columns[0]

        group_col = self.task_spec.group_column
        max_params = self.cfg.get("constraints", {}).get("max_params_limit")

        detailed_results: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        oof_records: List[Dict[str, Any]] = []

        partition_col = self.cfg.get("data", {}).get("partition_column")

        # Setup folds or holdout splits
        if val_strategy == "holdout" or "fold" not in df.columns:
            # Single train / dev split
            dev_split_name = self.cfg.get("validation", {}).get("dev_split_name", "dev")
            try:
                dev_df = self.task_spec.load_dataset_split(dev_split_name, project_root=self.project_root)
                train_df = df
            except Exception:
                shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                split_idx = int(len(shuffled) * 0.8)
                train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
                dev_df = shuffled.iloc[split_idx:].reset_index(drop=True)

            splits = [(0, train_df, dev_df)]
        else:
            # K-Fold Cross-Validation
            target_folds = [self.args.fold] if self.args.fold is not None else self.cfg.get("data", {}).get("folds", [0, 1, 2, 3, 4])
            splits = []
            for f in target_folds:
                train_part = df[df["fold"] != f].reset_index(drop=True)
                dev_part = df[df["fold"] == f].reset_index(drop=True)
                splits.append((f, train_part, dev_part))

        # Iteration Loop
        for fold_idx, train_fold, dev_fold in splits:
            self.logger.info(f"\n--- Processing Fold / Partition: {fold_idx} ---")
            if len(dev_fold) == 0:
                continue

            partitions = [("all", train_fold, dev_fold)]
            if partition_col and partition_col in df.columns:
                partitions = []
                for p_val in sorted(df[partition_col].unique()):
                    tr = train_fold[train_fold[partition_col] == p_val].reset_index(drop=True)
                    dv = dev_fold[dev_fold[partition_col] == p_val].reset_index(drop=True)
                    if len(dv) > 0:
                        partitions.append((p_val, tr, dv))

            for part_key, train_part, dev_part in partitions:
                for variant in input_variants:
                    model = self.task_plugin.build_model(self.plugin_context, model_name, model_params)
                    if model is None:
                        model = model_factory(model_name, **model_params)
                    ExperimentValidator.assert_parameter_limit(model, max_params=max_params)

                    if group_col and group_col in train_part.columns and group_col in dev_part.columns:
                        ExperimentValidator.assert_no_group_data_leakage(train_part, dev_part, group_column=group_col)

                    # Train
                    t0 = time.time()
                    try:
                        model.fit(train_part, variant=variant)
                    except TypeError:
                        model.fit(train_part)
                    fit_sec = time.time() - t0

                    # Predict
                    try:
                        preds = model.predict(dev_part, variant=variant)
                    except TypeError:
                        preds = model.predict(dev_part)

                    # Route predictions through UniversalTaskSpec if needed
                    normalized_preds = self.task_plugin.normalize_predictions(self.plugin_context, preds)
                    routed_preds = self.task_spec.apply_routing(normalized_preds, n_samples=len(dev_part))

                    # Store predictions
                    resolved_id_col = id_col if id_col in dev_part.columns else dev_part.columns[0]
                    for i in range(len(dev_part)):
                        rec = {
                            resolved_id_col: dev_part[resolved_id_col].iloc[i],
                            "fold": fold_idx,
                            "input_variant": variant,
                        }
                        if partition_col and partition_col in dev_part.columns:
                            rec[partition_col] = dev_part[partition_col].iloc[i]

                        for k, v in routed_preds.items():
                            pred_column = k if k.startswith("pred_") else f"pred_{k}"
                            if isinstance(v, (np.ndarray, list)):
                                val = v[i]
                                if isinstance(val, (np.ndarray, list)):
                                    rec[pred_column] = ",".join(str(x) for x in val)
                                else:
                                    rec[pred_column] = val
                            else:
                                rec[pred_column] = v
                        oof_records.append(rec)

                    # Compute Metrics using UniversalTaskSpec
                    eval_metrics = self.task_plugin.evaluate_predictions(self.plugin_context, dev_part, routed_preds)

                    # Optional custom domain evaluator hook
                    custom_eval_hook = self.cfg.get("paths", {}).get("metrics_evaluator")
                    if custom_eval_hook:
                        try:
                            import importlib
                            c_mod, c_fn = custom_eval_hook.rsplit(".", 1)
                            custom_res = getattr(importlib.import_module(c_mod), c_fn)(dev_part, routed_preds)
                            eval_metrics.update(custom_res)
                        except Exception:
                            pass

                metric_record = {
                    "experiment_id": self.exp_id,
                    "input_variant": variant,
                    "dev_fold": fold_idx,
                    "train_rows": len(train_part),
                    "dev_rows": len(dev_part),
                    "fit_time_seconds": round(fit_sec, 2),
                    **eval_metrics,
                }
                detailed_results.append(metric_record)

                summary_row = {
                    "input_variant": variant,
                    "dev_fold": fold_idx,
                    "train_rows": len(train_part),
                    "dev_rows": len(dev_part),
                    "composite_score": eval_metrics.get("composite_score", 0.0),
                }
                if partition_col and part_key != "all":
                    summary_row[partition_col] = part_key

                for st_id, st_val in eval_metrics.items():
                    if isinstance(st_val, dict) and "macro_f1" in st_val:
                        summary_row[f"{st_id}_f1"] = st_val.get("macro_f1", 0.0)

                summary_rows.append(summary_row)
                score_str = f"Composite: {eval_metrics.get('composite_score', 0.0):.4f}"
                self.logger.info(f"[{variant:12}] Fold {fold_idx} | {score_str}")

        return self._save_and_finalize(df, detailed_results, summary_rows, oof_records, git_hash)

    def _run_llm_prompting(self, df: pd.DataFrame, git_hash: str) -> Dict[str, Any]:
        """Execute LLM In-Context / API Prompting experiment."""
        from harness.llm_engine import LLMEngine, LLMConfig

        llm_cfg_dict = self.cfg.get("llm", {})
        llm_conf = LLMConfig(
            provider=llm_cfg_dict.get("provider", "gemini"),
            model_name=llm_cfg_dict.get("model_name", "gemini-2.0-flash"),
            temperature=llm_cfg_dict.get("temperature", 0.0),
            system_prompt=llm_cfg_dict.get("system_prompt", "You are an expert AI competition classifier."),
            prompt_template=llm_cfg_dict.get("prompt_template", "{text}"),
            use_cot=llm_cfg_dict.get("use_cot", False),
            output_format=llm_cfg_dict.get("output_format", "json"),
        )

        id_col = self.task_spec.id_column
        if id_col not in df.columns:
            id_col = df.columns[0]

        # Use dev split or sample
        dev_split_name = self.cfg.get("validation", {}).get("dev_split_name", "dev")
        try:
            eval_df = self.task_spec.load_dataset_split(dev_split_name, project_root=self.project_root)
        except Exception:
            eval_df = df.sample(n=min(len(df), 100), random_state=42).reset_index(drop=True)

        sample_limit = self.cfg.get("validation", {}).get("sample_limit", len(eval_df))
        eval_df = eval_df.iloc[:sample_limit].reset_index(drop=True)

        self.logger.info(f"Running LLM inference with {llm_conf.provider}/{llm_conf.model_name} on {len(eval_df)} samples...")

        preds_dict: Dict[str, List[Any]] = {st_id: [] for st_id in self.task_spec.subtasks}
        pred_records: List[Dict[str, Any]] = []

        t0 = time.time()
        for idx, row in eval_df.iterrows():
            row_dict = row.to_dict()
            sys_p, user_p = LLMEngine.format_prompt(
                template=llm_conf.prompt_template,
                row_data=row_dict,
                system_prompt=llm_conf.system_prompt,
                few_shots=llm_conf.few_shot_examples,
            )
            resp = LLMEngine.generate_completion(llm_conf, user_prompt=user_p, system_prompt=sys_p)
            extracted = LLMEngine.extract_prediction(resp.get("text", ""))

            rec = {id_col: row[id_col], "llm_raw_response": resp.get("text", "")}
            for st_id, spec in self.task_spec.subtasks.items():
                val = extracted.get(st_id) or extracted.get(spec.column) or spec.classes[0] if spec.classes else "none"
                preds_dict[st_id].append(val)
                rec[f"pred_{st_id}"] = val

            pred_records.append(rec)
            if (idx + 1) % 20 == 0 or idx == len(eval_df) - 1:
                self.logger.info(f"Processed {idx + 1}/{len(eval_df)} samples...")

        infer_sec = time.time() - t0
        eval_metrics = self.task_plugin.evaluate_predictions(self.plugin_context, eval_df, preds_dict)

        metric_record = {
            "experiment_id": self.exp_id,
            "approach": "llm_prompt",
            "model_name": llm_conf.model_name,
            "provider": llm_conf.provider,
            "samples_evaluated": len(eval_df),
            "inference_seconds": round(infer_sec, 2),
            **eval_metrics,
        }
        summary_rows = [{
            "approach": "llm_prompt",
            "model": f"{llm_conf.provider}/{llm_conf.model_name}",
            "samples": len(eval_df),
            "composite_score": eval_metrics.get("composite_score", 0.0),
        }]

        return self._save_and_finalize(eval_df, [metric_record], summary_rows, pred_records, git_hash)

    def _save_and_finalize(
        self,
        df_context: pd.DataFrame,
        detailed_results: List[Dict[str, Any]],
        summary_rows: List[Dict[str, Any]],
        oof_records: List[Dict[str, Any]],
        git_hash: str,
    ) -> Dict[str, Any]:
        """Save artifacts, metrics, predictions, summary cards, and update TreeJournal."""
        self.logger.info("\nSaving experiment artifacts...")
        save_json(self.exp_dir / "metrics.json", detailed_results)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.exp_dir / "summary.csv", index=False)

        oof_df = pd.DataFrame(oof_records)
        oof_df.to_csv(self.exp_dir / "predictions.csv", index=False)
        oof_df.to_csv(self.exp_dir / "oof_predictions.csv", index=False)

        # Markdown summary & card
        card_md = render_experiment_card(self.cfg, summary_df, git_hash, task_spec=self.task_spec)
        (self.exp_dir / "EXPERIMENT_CARD.md").write_text(card_md, encoding="utf-8")
        (self.exp_dir / "summary.md").write_text(summary_df.to_markdown(index=False), encoding="utf-8")

        # Diagnostics & Error Slices
        self.logger.info("Computing failure diagnostics & error slices...")
        try:
            DiagnosticEngine.save_diagnostics_bundle(
                eval_df=df_context,
                oof_df=oof_df,
                output_dir=self.exp_dir,
                exp_id=self.exp_id,
            )
        except Exception as e:
            self.logger.warning(f"Diagnostic generation skipped or failed: {e}")

        # Update TreeJournal
        journal_path = self.artifacts_dir / "journal.json"
        try:
            journal = TreeJournal(
                journal_path=journal_path,
                artifacts_dir=self.artifacts_dir,
                task_spec=self.task_spec,
            )
            comp_score = 0.0
            if detailed_results:
                comp_score = detailed_results[0].get("composite_score", 0.0)

            parent_id = self.cfg.get("experiment", {}).get("parent_id", "exp000_template")
            branch = self.cfg.get("experiment", {}).get("branch", "feat")
            hypothesis = self.cfg.get("grounding", {}).get("hypothesis", "")
            approach = self.cfg.get("experiment", {}).get("approach", "ml_classical")

            journal.add_node(
                node_id=self.exp_id,
                parent_id=parent_id,
                branch=branch,
                hypothesis=hypothesis,
                score=comp_score,
                metrics={self.task_spec.primary_metric: comp_score, "composite_score": comp_score},
                approach_type=approach,
            )
            journal.save()
        except Exception as e:
            self.logger.warning(f"TreeJournal update skipped: {e}")

        # Update Leaderboard
        try:
            exp_info = self.cfg.get("experiment", {})
            new_row = {
                "exp_id": self.exp_id,
                "branch": exp_info.get("branch", "general"),
                "parent_id": exp_info.get("parent_id", "none"),
                "date": exp_info.get("date"),
                "model": self.cfg.get("model", {}).get("name", "baseline"),
                "score": comp_score,
                "notes": exp_info.get("problem_addressed", ""),
            }
            if not getattr(self.args, "skip_leaderboard", False):
                leaderboard_value = (
                    self.project.manifest.get("leaderboard_path")
                    if self.project.manifest
                    else self.cfg.get("paths", {}).get("leaderboard_path")
                ) or "EXPERIMENTS.md"
                leaderboard = Path(
                    leaderboard_value
                )
                if not leaderboard.is_absolute():
                    leaderboard = self.project_root / leaderboard
                update_experiments_leaderboard(leaderboard, [new_row], task_spec=self.task_spec)
        except Exception as e:
            self.logger.debug(f"Leaderboard update skipped: {e}")

        self.logger.info(f"✔ Experiment {self.exp_id} completed successfully. Score: {comp_score:.4f}")
        return detailed_results[0] if detailed_results else {}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for universal training runner."""
    parser = argparse.ArgumentParser(description="Universal Competition Experiment Runner")
    parser.add_argument("--project-root", type=str, default=None, help="Competition workspace root")
    parser.add_argument("--task-config", type=str, default=None, help="Task specification YAML")
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
        if project.plugin_id not in (None, "default") or project.artifacts_dir != project.root / "artifacts":
            parser.error("Remote backends do not install selected task adapters or route artifacts through the attached manifest yet; use --backend local.")
        from harness.cloud_runner import run_via_modal
        run_via_modal(config_path=str(config_path))
        return 0
    elif args.backend == "kaggle":
        if project.plugin_id not in (None, "default") or project.artifacts_dir != project.root / "artifacts":
            parser.error("Remote backends do not install selected task adapters or route artifacts through the attached manifest yet; use --backend local.")
        from harness.cloud_runner import run_via_kaggle
        run_via_kaggle(config_path=str(config_path))
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
        print(
            f"ERROR: Experiment directory '{exp_dir}' already exists with completed metrics.\n"
            f"Pass --force or define a new experiment ID in configs/.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.project_root = str(project.root)
    args.task_config = str(project.task_config_path)
    runner = UniversalRunner(cfg=cfg, config_path=str(config_path), args=args)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
