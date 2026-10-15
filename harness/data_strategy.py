"""Universal Data Strategy and Leakage-Proof Splitting Engine for AI-Harness.

Provides standardized dataset ingestion, multiple cross-validation strategies
(GroupKFold, StratifiedGroupKFold, Holdout Test, Zero-Shot domain split),
and physical Data Leakage auditing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


class SplitMode(str, Enum):
    CANONICAL = "canonical"
    GROUP_KFOLD = "group_kfold"
    STRATIFIED_GROUP_KFOLD = "stratified_group_kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    HOLDOUT_TEST = "holdout_test"
    ZERO_SHOT_DOMAIN = "zero_shot_domain"
    TIME_SERIES = "time_series"
    TEMPORAL = "temporal"
    PURGED_GROUP_TIME_SERIES = "purged_group_time_series"
    MULTILABEL_STRATIFIED_KFOLD = "multilabel_stratified_kfold"


class DataLeakageError(RuntimeError):
    """Raised when data leakage between train and evaluation partitions is detected."""


class LeakageAuditor:
    """Verifies that no identical groups or texts cross the train/validation boundary."""

    @staticmethod
    def audit_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        group_col: Optional[str] = None,
        content_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform physical leakage audit and raise DataLeakageError if violated."""
        report = {"leakage_detected": False, "leaked_groups": 0, "leaked_texts": 0}

        if group_col and group_col in train_df.columns and group_col in val_df.columns:
            train_groups: Set[Any] = set(train_df[group_col].dropna().unique())
            val_groups: Set[Any] = set(val_df[group_col].dropna().unique())
            overlap = train_groups.intersection(val_groups)
            if overlap:
                report["leakage_detected"] = True
                report["leaked_groups"] = len(overlap)
                sample_leaks = list(overlap)[:5]
                raise DataLeakageError(
                    f"CRITICAL LEAKAGE DETECTED: {len(overlap)} groups from '{group_col}' "
                    f"appear in both train and validation sets! Examples: {sample_leaks}"
                )

        if content_cols:
            for col in content_cols:
                if col in train_df.columns and col in val_df.columns:
                    train_texts = set(train_df[col].dropna().astype(str).str.strip().unique())
                    val_texts = set(val_df[col].dropna().astype(str).str.strip().unique())
                    overlap_txt = train_texts.intersection(val_texts)
                    # Filter out empty or trivial strings
                    overlap_txt = {t for t in overlap_txt if len(t) > 20}
                    if overlap_txt:
                        report["leakage_detected"] = True
                        report["leaked_texts"] += len(overlap_txt)
                        sample_leaks = list(overlap_txt)[:3]
                        raise DataLeakageError(
                            f"CRITICAL CONTENT LEAKAGE: {len(overlap_txt)} identical text values in '{col}' "
                            f"cross the partition boundary! Examples: {sample_leaks}"
                        )

        return report


class DataStrategyEngine:
    """Universal splitter and dataset partitioner."""

    @staticmethod
    def compute_file_hash(path: Path | str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def generate_splits(
        df: pd.DataFrame,
        mode: SplitMode | str,
        n_splits: int = 5,
        group_col: Optional[str] = None,
        stratify_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        holdout_ratio: float = 0.15,
        seed: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_indices, val_indices) pairs for the given strategy."""
        if isinstance(mode, str):
            clean_mode = mode.lower().replace("-", "_").strip()
            if clean_mode in ("timeseries", "timeseries_split", "time_series", "time_series_split"):
                mode_val = SplitMode.TIME_SERIES
            elif clean_mode == "temporal":
                mode_val = SplitMode.TEMPORAL
            elif clean_mode in ("purged_group_time_series", "purged_time_series", "purged_kfold", "financial_time_series"):
                mode_val = SplitMode.PURGED_GROUP_TIME_SERIES
            elif clean_mode in ("multilabel_stratified", "multilabel_stratified_kfold", "multilabel_kfold"):
                mode_val = SplitMode.MULTILABEL_STRATIFIED_KFOLD
            else:
                mode_val = SplitMode(mode)
        else:
            mode_val = mode
        n_samples = len(df)
        indices = np.arange(n_samples)

        if mode_val == SplitMode.GROUP_KFOLD:
            if not group_col or group_col not in df.columns:
                raise ValueError(f"group_col '{group_col}' required for GroupKFold")
            gkf = GroupKFold(n_splits=n_splits)
            return list(gkf.split(df, groups=df[group_col]))

        elif mode_val == SplitMode.STRATIFIED_GROUP_KFOLD:
            if not group_col or not stratify_col:
                raise ValueError("Both group_col and stratify_col required for StratifiedGroupKFold")
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(sgkf.split(df, y=df[stratify_col], groups=df[group_col]))

        elif mode_val == SplitMode.STRATIFIED_KFOLD:
            if not stratify_col or stratify_col not in df.columns:
                raise ValueError(f"stratify_col '{stratify_col}' required for StratifiedKFold")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(skf.split(df, y=df[stratify_col]))

        elif mode_val == SplitMode.KFOLD:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(kf.split(df))

        elif mode_val in (SplitMode.TIME_SERIES, SplitMode.TEMPORAL):
            tss = TimeSeriesSplit(n_splits=n_splits)
            return list(tss.split(df))

        elif mode_val == SplitMode.PURGED_GROUP_TIME_SERIES:
            if not group_col or group_col not in df.columns:
                raise ValueError(f"group_col (time group identifier) required for PurgedGroupTimeSeriesSplit")
            unique_groups = np.sort(df[group_col].dropna().unique())
            n_groups = len(unique_groups)
            if n_groups < n_splits + 1:
                raise ValueError(f"Not enough unique time groups ({n_groups}) for {n_splits} splits")
            embargo_groups = max(1, int(n_groups * holdout_ratio)) if holdout_ratio > 0 else 0
            splits = []
            group_split_size = max(1, n_groups // (n_splits + 1))
            for i in range(n_splits):
                train_group_end = (i + 1) * group_split_size
                val_group_start = train_group_end + embargo_groups
                val_group_end = min(val_group_start + group_split_size, n_groups)
                if val_group_start >= n_groups or val_group_start >= val_group_end:
                    break
                tr_groups = set(unique_groups[:train_group_end])
                vl_groups = set(unique_groups[val_group_start:val_group_end])
                tr_idx = indices[df[group_col].isin(tr_groups).to_numpy()]
                vl_idx = indices[df[group_col].isin(vl_groups).to_numpy()]
                if len(tr_idx) > 0 and len(vl_idx) > 0:
                    splits.append((tr_idx, vl_idx))
            if not splits:
                raise ValueError("PurgedGroupTimeSeriesSplit produced no valid non-empty partitions")
            return splits

        elif mode_val == SplitMode.MULTILABEL_STRATIFIED_KFOLD:
            target_cols_to_use = target_cols or ([stratify_col] if stratify_col else [])
            valid_targets = [c for c in target_cols_to_use if c in df.columns]
            if not valid_targets:
                raise ValueError("target_cols or stratify_col required for MultilabelStratifiedKFold")
            Y = df[valid_targets].to_numpy()
            n_samples = len(df)
            fold_indices = [[] for _ in range(n_splits)]
            label_counts = np.maximum(Y.sum(axis=0), 1)
            sample_scores = Y @ (1.0 / label_counts)
            rng = np.random.RandomState(seed)
            order = np.lexsort((rng.rand(n_samples), -sample_scores))
            for idx in order:
                fold_sizes = [len(f) for f in fold_indices]
                min_fold = int(np.argmin(fold_sizes))
                fold_indices[min_fold].append(idx)
            splits = []
            for f in range(n_splits):
                val_idx = np.array(fold_indices[f], dtype=int)
                train_idx = np.setdiff1d(indices, val_idx)
                splits.append((train_idx, val_idx))
            return splits

        elif mode_val == SplitMode.HOLDOUT_TEST:
            rng = np.random.RandomState(seed)
            if group_col and group_col in df.columns:
                unique_groups = df[group_col].unique()
                rng.shuffle(unique_groups)
                n_holdout = max(1, int(len(unique_groups) * holdout_ratio))
                holdout_groups = set(unique_groups[:n_holdout])
                val_mask = df[group_col].isin(holdout_groups).to_numpy()
                train_idx = indices[~val_mask]
                val_idx = indices[val_mask]
                return [(train_idx, val_idx)]
            else:
                shuffled = rng.permutation(indices)
                n_holdout = max(1, int(n_samples * holdout_ratio))
                return [(shuffled[n_holdout:], shuffled[:n_holdout])]

        else:
            raise ValueError(f"Unsupported SplitMode: {mode}")

    @staticmethod
    def create_split_manifest(
        df: pd.DataFrame,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        id_col: str,
        group_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a complete audit manifest for the split distribution."""
        manifest: Dict[str, Any] = {
            "total_samples": len(df),
            "n_folds": len(splits),
            "metadata": metadata or {},
            "folds": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_sub = df.iloc[train_idx]
            val_sub = df.iloc[val_idx]

            fold_info: Dict[str, Any] = {
                "fold": fold_idx,
                "train_count": len(train_idx),
                "val_count": len(val_idx),
            }

            if group_col and group_col in df.columns:
                fold_info["train_groups"] = int(train_sub[group_col].nunique())
                fold_info["val_groups"] = int(val_sub[group_col].nunique())

            if target_cols:
                fold_info["val_distributions"] = {}
                for t_col in target_cols:
                    if t_col in val_sub.columns:
                        fold_info["val_distributions"][t_col] = (
                            val_sub[t_col].value_counts().to_dict()
                        )

            manifest["folds"].append(fold_info)

        return manifest
