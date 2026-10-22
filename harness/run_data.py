"""Data-loading & validation-split mixin for ``UniversalRunner``.

Extracted verbatim from ``harness/runner.py`` (M4 refactoring, PR-F):
``_load_training_data`` and ``_resolve_validation_splits``. ``UniversalRunner`` mixes this in, so data behavior is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from harness.artifacts import digest
from harness.validator import ExperimentValidator


class RunDataMixin:
    """Training-data loading and validation-split resolution, moved verbatim."""

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
        # Script adapters also need the complete legacy fold map, even when
        # the execution below selects just one fold for a probe.
        self._all_validation_splits = result
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

    def _prepare_script_validation(self, df, strategy):
        """Export explicit execution partitions and return only evaluation truth.

        Child scripts own their model/data code. This manifest is the shared
        contract for selected folds and probe bounds; output coverage is checked
        against the same IDs when the child returns.
        """
        import json

        from harness.io_utils import save_json

        splits = self._resolve_validation_splits(df, strategy)
        limit = self._probe_sample_limit()
        selected = []
        for fold, train, dev in splits:
            if limit:
                train = train.sample(n=min(len(train), limit), random_state=self.seed)
                dev = dev.head(limit)
            selected.append((fold, train, dev))
        key = self.task_spec.id_column
        truth = pd.concat([dev for _, _, dev in selected])
        if truth[key].duplicated().any():
            raise ValueError("Script validation partitions must contain each evaluation ID once")
        self.expected_validation_ids = set(truth[key])
        self._script_training_rows = len({value for _, train, _ in selected for value in train[key]})
        self._script_fold_by_id = {value: fold for fold, _, dev in selected for value in dev[key]}
        self.manifest["split_sha256"] = digest([
            (str(fold), train[key].tolist(), dev[key].tolist()) for fold, train, dev in selected
        ])
        manifest_path = self.exp_dir / "validation_partitions.json"
        save_json(manifest_path, {
            "schema_version": 1, "run_kind": self.run_kind, "id_column": key,
            "sample_limit": limit,
            "partitions": [{"fold": fold, "train_ids": train[key].tolist(),
                            "validation_ids": dev[key].tolist()} for fold, train, dev in selected],
        })
        # Keep a complete fold map for existing full-CV scripts. -1 identifies
        # training-only rows (holdout/time-series); the JSON manifest remains
        # authoritative for which partitions are actually executed.
        all_splits = self._all_validation_splits
        fold_map = {value: fold for fold, _, dev in all_splits for value in dev[key]}
        rows = pd.concat([df[[key]], *[dev[[key]] for _, _, dev in all_splits]]).drop_duplicates(key)
        rows["fold"] = rows[key].map(lambda value: fold_map.get(value, -1))
        rows.to_csv(self.exp_dir / "splits.tsv", sep="\t", index=False)
        return truth, {
            "HARNESS_VALIDATION_MANIFEST": str(manifest_path),
            "HARNESS_RUN_KIND": self.run_kind,
            "HARNESS_SELECTED_FOLDS": json.dumps([fold for fold, _, _ in selected]),
            "HARNESS_SAMPLE_LIMIT": str(limit) if limit is not None else "",
        }
