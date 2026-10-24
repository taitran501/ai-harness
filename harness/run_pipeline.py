"""Multi-stage pipeline execution mixin for ``UniversalRunner``.

Extracted verbatim from ``harness/runner.py`` (M4 refactoring, PR-F):
``_resolve_pipeline_stages``, ``_run_pipeline``, and
``_verify_pipeline_manifest``. ``UniversalRunner`` mixes this in, so
pipeline behavior is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from harness.io_utils import save_json


class RunPipelineMixin:
    """Pipeline stage resolution, execution, and manifest verification, moved verbatim."""

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
        truth, validation_env = self._prepare_script_validation(df, val_strategy)

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
            child_env.update(validation_env)
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

        # 5.5. Audit gate: verify the manifest is complete and consistent
        self._verify_pipeline_manifest(len(stages))

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

        from harness.script_predictions import align_prediction_ids

        merged_oof = align_prediction_ids(oof_df, truth[id_col], id_col, kind="OOF")

        variant = "pipeline"
        self.selection_variant = variant
        records = []
        for i in range(len(merged_oof)):
            row = merged_oof.iloc[i]
            rec = {
                id_col: row[id_col],
                "fold": self._script_fold_by_id[row[id_col]],
                "input_variant": variant,
            }
            for target in self.task_spec.subtasks.keys():
                col_name = f"pred_{target}" if f"pred_{target}" in row else target
                if col_name in row:
                    rec[f"pred_{target}"] = row[col_name]
                if f"proba_{target}" in row:
                    rec[f"proba_{target}"] = row[f"proba_{target}"]
                if f"classes_{target}" in row:
                    rec[f"classes_{target}"] = row[f"classes_{target}"]
            records.append(rec)

        total_duration = sum(s["duration_seconds"] for s in stage_manifest)
        details = [{
            "input_variant": variant,
            "dev_fold": "all",
            "train_rows": self._script_training_rows,
            "dev_rows": len(truth),
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
            from harness.script_predictions import extract_test_predictions

            cand_test = extract_test_predictions(t_df, self.task_spec, expected_ids=test_df[id_col])
            fold_test_preds.append(cand_test)

        self.estimator_metadata.append({
            "class": "PipelineAdapter",
            "stages": stage_manifest,
        })

        return self._save_and_finalize(
            truth, details, rows, records, git_hash, test_df=test_df, fold_test_preds=fold_test_preds
        )

    def _verify_pipeline_manifest(self, expected_count: int) -> None:
        """Audit gate: verify stage_manifest.json is complete and consistent.

        Raises ValueError if any check fails:
        - manifest missing
        - stage count != expected_count
        - any exit_code != 0 (inconsistent with fail-fast)
        - index not 0..N-1 (order violated)
        - any per-stage log file missing
        """
        manifest_path = self.exp_dir / "stage_manifest.json"
        if not manifest_path.exists():
            raise ValueError("Pipeline audit gate: stage_manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest) != expected_count:
            raise ValueError(
                f"Pipeline audit gate: expected {expected_count} stages, "
                f"found {len(manifest)}"
            )
        for i, stage in enumerate(manifest):
            if stage.get("index") != i:
                raise ValueError(
                    f"Pipeline audit gate: stage {i} has index {stage.get('index')} "
                    f"(order violated)"
                )
            if stage.get("exit_code") != 0:
                raise ValueError(
                    f"Pipeline audit gate: stage '{stage.get('name')}' "
                    f"exit code {stage.get('exit_code')} != 0"
                )
            log_rel = stage.get("log_file", "")
            if not log_rel:
                raise ValueError(
                    f"Pipeline audit gate: stage '{stage.get('name')}' "
                    f"has no log_file"
                )
            log_path = self.exp_dir / log_rel
            if not log_path.exists():
                raise ValueError(
                    f"Pipeline audit gate: log file missing for stage "
                    f"'{stage.get('name')}': {log_rel}"
                )
