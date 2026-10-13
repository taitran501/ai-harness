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
)


class SplitMode(str, Enum):
    CANONICAL = "canonical"
    GROUP_KFOLD = "group_kfold"
    STRATIFIED_GROUP_KFOLD = "stratified_group_kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    HOLDOUT_TEST = "holdout_test"
    ZERO_SHOT_DOMAIN = "zero_shot_domain"


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
        holdout_ratio: float = 0.15,
        seed: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_indices, val_indices) pairs for the given strategy."""
        mode_val = SplitMode(mode) if isinstance(mode, str) else mode
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

        elif mode_val == SplitMode.HOLDOUT_TEST:
            # Create a single train / holdout test split respecting group_col if present
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
