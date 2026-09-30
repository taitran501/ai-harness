"""Universal Task Specification and Conditional Logic Engine for ML-Harness.

Enables declarative definition of multi-task, hierarchical, and conditional
relationships (if-else routing, oracle masks) between prediction targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ntpath
from pathlib import Path
import shutil
import stat
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


@dataclass
class InputVariantSpec:
    """Specification for an input representation variant."""

    name: str
    type: str  # 'text', 'text_template', 'tabular', 'multimodal'
    source_columns: List[str] = field(default_factory=list)
    template: Optional[str] = None

    def compose_text(self, df: pd.DataFrame) -> pd.Series:
        """Compose text series according to the specification."""
        if self.type == "text":
            col = self.source_columns[0]
            return df[col].fillna("").astype(str)

        if self.type == "text_template" and self.template:
            def format_row(row):
                return self.template.format(**{col: str(row.get(col, "")) for col in self.source_columns})

            return df.apply(format_row, axis=1)

        if len(self.source_columns) > 0:
            return df[self.source_columns[0]].fillna("").astype(str)

        raise ValueError(f"Cannot compose input for variant '{self.name}' with type '{self.type}'.")


@dataclass
class ConditionalRoutingRule:
    """Specifies an if-else dependency between a child target and parent target."""

    parent_target: str
    operator: str  # 'in', 'not_in', '==', '!=', 'gt', 'lt'
    values: Any
    fallback_value: Any
    eval_condition_only: bool = True  # If True, metric is evaluated only on rows where condition is True (e.g. oracle_hate)

    def evaluate_condition(self, parent_values: pd.Series | np.ndarray) -> np.ndarray:
        """Return boolean array where condition holds true."""
        series = pd.Series(parent_values) if not isinstance(parent_values, pd.Series) else parent_values

        if self.operator == "==":
            return (series == self.values).to_numpy()
        elif self.operator == "!=":
            return (series != self.values).to_numpy()
        elif self.operator == "in":
            val_list = self.values if isinstance(self.values, (list, tuple, set)) else [self.values]
            return series.isin(val_list).to_numpy()
        elif self.operator == "not_in":
            val_list = self.values if isinstance(self.values, (list, tuple, set)) else [self.values]
            return (~series.isin(val_list)).to_numpy()
        elif self.operator == "gt":
            return (series.astype(float) > float(self.values)).to_numpy()
        elif self.operator == "lt":
            return (series.astype(float) < float(self.values)).to_numpy()
        else:
            raise ValueError(f"Unsupported conditional operator: {self.operator}")


@dataclass
class SubtaskSpec:
    """Specification for an individual subtask."""

    id: str
    name: str
    task_type: str  # 'binary_classification', 'multiclass_classification', 'multilabel_classification', 'regression'
    column: Optional[str] = None
    column_prefix: Optional[str] = None  # Column prefix for multi-label targets (e.g., 'label_')
    classes: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)  # For multi-label
    primary_metric: str = "macro_f1"
    direction: str = "maximize"
    weight: float = 1.0
    condition: Optional[ConditionalRoutingRule] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubtaskSpec:
        cond_data = data.get("condition")
        cond_obj = None
        if cond_data:
            cond_obj = ConditionalRoutingRule(
                parent_target=cond_data["parent_target"],
                operator=cond_data.get("operator", "=="),
                values=cond_data["values"],
                fallback_value=cond_data.get("fallback_value"),
                eval_condition_only=cond_data.get("eval_condition_only", True),
            )

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            task_type=data.get("task_type", data.get("type", "binary_classification")),
            column=data.get("column"),
            column_prefix=data.get("column_prefix"),
            classes=data.get("classes", []),
            labels=data.get("labels", []),
            primary_metric=data.get("primary_metric", data.get("metric", "macro_f1")),
            direction=data.get("direction", "maximize"),
            weight=float(data.get("weight", 1.0)),
            condition=cond_obj,
        )


@dataclass
class ValidationSpec:
    """Specification for validation strategy across ML, DL, and LLM approaches."""

    strategy: str = "kfold"  # 'kfold', 'holdout', 'zero_shot_eval', 'sample_probe'
    num_folds: int = 5
    dev_split_name: str = "dev"
    probe_samples: int = 200
    split_column: Optional[str] = None
    split_file: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationSpec:
        if not data:
            return cls()
        return cls(
            strategy=data.get("strategy", "kfold"),
            num_folds=int(data.get("num_folds", 5)),
            dev_split_name=data.get("dev_split_name", "dev"),
            probe_samples=int(data.get("probe_samples", 200)),
            split_column=data.get("split_column"),
            split_file=data.get("split_file"),
        )


def load_dataframe_file(path: Path | str) -> pd.DataFrame:
    """Load arbitrary tabular data format (csv, tsv, parquet, feather, jsonl, sqlite)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    elif suffix == ".csv":
        return pd.read_csv(p)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    elif suffix == ".feather":
        return pd.read_feather(p)
    elif suffix in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    elif suffix == ".json":
        try:
            return pd.read_json(p)
        except Exception:
            return pd.read_json(p, lines=True)
    elif suffix in (".db", ".sqlite", ".sqlite3"):
        import sqlite3
        conn = sqlite3.connect(p)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if len(tables) > 0:
            tbl_name = tables.iloc[0, 0]
            df = pd.read_sql_query(f"SELECT * FROM {tbl_name}", conn)
            conn.close()
            return df
        conn.close()
    return pd.read_csv(p)


def extract_archives_in_directory(dir_path: Path | str) -> List[Path]:
    """Scan and unpack any archive files (.zip, .tar.gz, .tgz, .tar) found in a directory."""
    d = Path(dir_path)
    if not d.is_dir():
        return []
    extracted = []
    import tarfile
    import zipfile

    def destination_for(target: Path, member_name: str) -> Path:
        normalized = member_name.replace("\\", "/")
        drive, _ = ntpath.splitdrive(normalized)
        member_path = Path(normalized)
        if drive or member_path.is_absolute() or any(part == ".." for part in normalized.split("/")):
            raise ValueError(f"Unsafe archive member path: {member_name!r}")
        destination = (target / Path(*[part for part in normalized.split("/") if part not in ("", ".")])).resolve()
        try:
            destination.relative_to(target.resolve())
        except ValueError as exc:
            raise ValueError(f"Archive member escapes extraction directory: {member_name!r}") from exc
        return destination

    def extract_zip(archive: Path, target: Path) -> None:
        with zipfile.ZipFile(archive, "r") as zf:
            for item in zf.infolist():
                destination = destination_for(target, item.filename)
                mode = item.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError(f"Unsupported archive entry type: {item.filename!r}")
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(item, "r") as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    def extract_tar(archive: Path, target: Path) -> None:
        with tarfile.open(archive, "r:*") as tf:
            for item in tf.getmembers():
                destination = destination_for(target, item.name)
                if item.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not item.isfile():
                    raise ValueError(f"Unsupported archive entry type: {item.name!r}")
                source = tf.extractfile(item)
                if source is None:
                    raise ValueError(f"Could not read archive entry: {item.name!r}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    for f in list(d.iterdir()):
        if f.is_file():
            if f.suffix.lower() == ".zip":
                target_out = d / f.stem
                if not target_out.exists():
                    target_out.mkdir(parents=True, exist_ok=True)
                    extract_zip(f, target_out)
                    extracted.append(target_out)
            elif f.name.lower().endswith((".tar.gz", ".tgz", ".tar")):
                stem = f.name.split(".")[0]
                target_out = d / stem
                if not target_out.exists():
                    target_out.mkdir(parents=True, exist_ok=True)
                    extract_tar(f, target_out)
                    extracted.append(target_out)
    return extracted


@dataclass
class UniversalTaskSpec:
    """Complete specification of a competition/experiment task."""

    name: str
    task_type: str
    primary_metric: str
    direction: str = "maximize"
    inputs: List[InputVariantSpec] = field(default_factory=list)
    subtasks: Dict[str, SubtaskSpec] = field(default_factory=dict)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def platform(self) -> str:
        """Competition platform (Kaggle, Codabench, etc.)."""
        return self.metadata.get("platform", "")

    @property
    def id_column(self) -> Optional[str]:
        """ID column from data config."""
        data_cfg = self.metadata.get("data_config") or self.metadata.get("data", {})
        return data_cfg.get("id_column")

    @property
    def group_column(self) -> Optional[str]:
        """Group column for leakage prevention."""
        data_cfg = self.metadata.get("data_config") or self.metadata.get("data", {})
        return data_cfg.get("group_column") or self.metadata.get("group_column")

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> UniversalTaskSpec:
        task_cfg = cfg.get("task", cfg)
        inputs_list = []
        for inp in task_cfg.get("inputs", []):
            inputs_list.append(
                InputVariantSpec(
                    name=inp["name"],
                    type=inp.get("type", "text"),
                    source_columns=inp.get("source_columns", []),
                    template=inp.get("template"),
                )
            )

        subtasks_dict = {}
        st_configs = task_cfg.get("subtasks", task_cfg.get("metrics", []))
        for st in st_configs:
            spec = SubtaskSpec.from_dict(st)
            subtasks_dict[spec.id] = spec

        val_cfg = task_cfg.get("validation", cfg.get("validation", {}))
        val_spec = ValidationSpec.from_dict(val_cfg)

        return cls(
            name=task_cfg.get("name", task_cfg.get("task_name", "universal_task")),
            task_type=task_cfg.get("task_type", task_cfg.get("type", "classification")),
            primary_metric=task_cfg.get("primary_metric", "macro_f1"),
            direction=task_cfg.get("direction", "maximize"),
            inputs=inputs_list,
            subtasks=subtasks_dict,
            validation=val_spec,
            metadata=task_cfg,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> UniversalTaskSpec:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    def apply_routing(
        self, raw_predictions: Dict[str, Any], n_samples: int
    ) -> Dict[str, Any]:
        """Apply conditional fallback rules across subtasks during inference."""
        routed = dict(raw_predictions)

        for subtask_id, spec in self.subtasks.items():
            if spec.condition is None:
                continue

            cond = spec.condition
            parent_pred = routed.get(cond.parent_target)
            if parent_pred is None:
                continue

            active_mask = cond.evaluate_condition(parent_pred)
            inactive_mask = ~active_mask

            current_preds = routed.get(subtask_id)
            if current_preds is None:
                continue

            # Route fallback values for inactive instances
            if spec.task_type in ("binary_classification", "multiclass_classification"):
                preds_arr = np.array(current_preds, dtype=object)
                preds_arr[inactive_mask] = cond.fallback_value
                routed[subtask_id] = preds_arr

            elif spec.task_type == "multilabel_classification":
                preds_arr = np.array(current_preds)
                fallback = cond.fallback_value
                if fallback is None or fallback == 0:
                    preds_arr[inactive_mask] = 0
                elif isinstance(fallback, (list, tuple, np.ndarray)):
                    preds_arr[inactive_mask] = np.array(fallback)
                routed[subtask_id] = preds_arr

            elif spec.task_type == "regression":
                preds_arr = np.array(current_preds, dtype=float)
                preds_arr[inactive_mask] = float(cond.fallback_value if cond.fallback_value is not None else 0.0)
                routed[subtask_id] = preds_arr

        return routed

    def compute_evaluation_masks(self, ground_truth_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute evaluation mask per subtask (handles oracle condition masks)."""
        masks = {}
        n_samples = len(ground_truth_df)

        for subtask_id, spec in self.subtasks.items():
            if spec.condition is not None and spec.condition.eval_condition_only:
                parent_subtask = self.subtasks.get(spec.condition.parent_target)
                parent_col = parent_subtask.column if parent_subtask else None
                if parent_col and parent_col in ground_truth_df.columns:
                    parent_vals = ground_truth_df[parent_col]
                    masks[subtask_id] = spec.condition.evaluate_condition(parent_vals)
                else:
                    masks[subtask_id] = np.ones(n_samples, dtype=bool)
            else:
                masks[subtask_id] = np.ones(n_samples, dtype=bool)

        return masks

    def evaluate_predictions(
        self,
        ground_truth_df: pd.DataFrame,
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute metrics across all declared subtasks dynamically according to spec."""
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        results: Dict[str, Any] = {}
        masks = self.compute_evaluation_masks(ground_truth_df)
        total_weight = 0.0
        weighted_score_sum = 0.0

        for subtask_id, spec in self.subtasks.items():
            mask = masks.get(subtask_id, np.ones(len(ground_truth_df), dtype=bool))
            df_eval = ground_truth_df[mask].reset_index(drop=True)

            # Retrieve predictions for this subtask
            pred_vals = None
            cand_keys = [
                subtask_id,
                spec.column,
                f"pred_{spec.column}" if spec.column else None,
                f"pred_{subtask_id}",
            ]
            for k in cand_keys:
                if k and k in predictions:
                    pred_vals = predictions[k]
                    break

            if pred_vals is None or len(df_eval) == 0:
                results[subtask_id] = {"macro_f1": 0.0, "accuracy": 0.0, "score": 0.0}
                continue

            # Apply mask to predictions if needed
            if len(pred_vals) == len(ground_truth_df):
                eval_preds = np.array(pred_vals)[mask]
            else:
                eval_preds = np.array(pred_vals)

            if spec.task_type in ("binary_classification", "multiclass_classification"):
                col = spec.column
                if not col or col not in df_eval.columns:
                    continue
                y_true = df_eval[col].astype(str)
                y_pred = pd.Series(eval_preds).astype(str)
                labels = [str(c) for c in spec.classes] if spec.classes else sorted(list(y_true.unique()))
                macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
                acc = float(accuracy_score(y_true, y_pred))
                st_res = {
                    "macro_f1": macro_f1,
                    "accuracy": acc,
                    "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
                }
                results[subtask_id] = st_res
                if spec.condition and spec.condition.eval_condition_only:
                    results[f"{subtask_id}_oracle"] = st_res

                primary = macro_f1 if spec.primary_metric == "macro_f1" else acc
                weighted_score_sum += primary * spec.weight
                total_weight += spec.weight

            elif spec.task_type == "multilabel_classification":
                label_cols = []
                if spec.column_prefix:
                    label_cols = [f"{spec.column_prefix}{l}" for l in spec.labels]
                elif spec.column and spec.column in df_eval.columns:
                    label_cols = [spec.column]

                if label_cols and all(c in df_eval.columns for c in label_cols):
                    y_true = df_eval[label_cols].to_numpy()
                    y_pred = np.array(eval_preds)
                    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
                    acc = float(accuracy_score(y_true, y_pred))
                    st_res = {
                        "macro_f1": macro_f1,
                        "micro_f1": micro_f1,
                        "exact_match": acc,
                    }
                    results[subtask_id] = st_res
                    if spec.condition and spec.condition.eval_condition_only:
                        results[f"{subtask_id}_oracle"] = st_res

                    primary = macro_f1 if spec.primary_metric == "macro_f1" else micro_f1
                    weighted_score_sum += primary * spec.weight
                    total_weight += spec.weight

            elif spec.task_type == "regression":
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                col = spec.column
                if col and col in df_eval.columns:
                    y_true = df_eval[col].astype(float)
                    y_pred = pd.Series(eval_preds).astype(float)
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    mae = float(mean_absolute_error(y_true, y_pred))
                    r2 = float(r2_score(y_true, y_pred))
                    results[subtask_id] = {"rmse": rmse, "mae": mae, "r2": r2}
                    primary = r2 if spec.primary_metric == "r2" else -rmse
                    weighted_score_sum += primary * spec.weight
                    total_weight += spec.weight

        results["composite_score"] = float(weighted_score_sum / total_weight) if total_weight > 0 else 0.0
        return results

    @property
    def id_column(self) -> str:
        """Resolve primary ID column declared in task data_config."""
        return self.metadata.get("data_config", {}).get("id_column", "id")

    @property
    def group_column(self) -> Optional[str]:
        """Resolve anti-leakage group column if defined."""
        return self.metadata.get("data_config", {}).get("group_column")

    @property
    def train_path(self) -> str:
        """Resolve raw training dataset directory or file."""
        return self.metadata.get("data_config", {}).get("train_path", "dataset/")

    @property
    def test_path(self) -> str:
        """Resolve raw test dataset directory or file."""
        return self.metadata.get("data_config", {}).get("test_path", "dataset/test/")

    @property
    def primary_target(self) -> Optional[str]:
        """Resolve primary target column or subtask identifier."""
        if not self.subtasks:
            return None
        # Highest weighted subtask
        sorted_st = sorted(self.subtasks.values(), key=lambda s: s.weight, reverse=True)
        primary = sorted_st[0]
        return primary.column or primary.id

    def load_dataset_split(
        self,
        split: str = "train",
        project_root: Path | str = ".",
    ) -> pd.DataFrame:
        """Load tabular/text dataset for a given split in a completely task-agnostic manner."""
        root = Path(project_root)
        data_dir = root / "dataset"
        if data_dir.is_dir():
            extract_archives_in_directory(data_dir)

        # 0. If 'split' is an absolute/valid file path, load directly
        split_path = Path(split)
        if split_path.is_absolute() or split_path.exists():
            if split_path.is_file():
                return load_dataframe_file(split_path)

        # Support both 'data_config' and 'data' keys (universal task.yaml convention)
        data_cfg = self.metadata.get("data_config") or self.metadata.get("data", {})

        # 1. Custom path override in data_config
        custom_path = data_cfg.get(f"{split}_path")
        if custom_path:
            p = Path(custom_path)
            if not p.is_absolute():
                p = root / custom_path
            if p.is_file() and p.exists():
                return load_dataframe_file(p)
            elif p.is_dir() and p.exists():
                extract_archives_in_directory(p)
                exts = ("*.tsv", "*.csv", "*.parquet", "*.pq", "*.feather", "*.jsonl", "*.json")
                files = []
                for ext in exts:
                    files.extend(p.glob(ext))
                files = sorted(files)
                if files:
                    return pd.concat([load_dataframe_file(f) for f in files], ignore_index=True)

        # 2. Standard convention search
        extensions = [".csv", ".tsv", ".parquet", ".pq", ".feather", ".jsonl", ".json"]
        candidates = []
        for ext in extensions:
            candidates.extend([
                f"{split}{ext}",
                f"dataset/{split}{ext}",
                f"dataset/{split}_data{ext}",
            ])

        if split == "train":
            candidates.extend(["dataset/train/", "dataset/"])
        elif split in ("dev", "val", "validation"):
            for ext in extensions:
                candidates.extend([f"val{ext}", f"dev{ext}", f"dataset/val{ext}", f"dataset/dev{ext}"])
            candidates.extend(["dataset/val/", "dataset/dev/"])
        elif split == "test":
            for ext in extensions:
                candidates.extend([f"test{ext}", f"dataset/test{ext}", f"dataset/test_data{ext}"])
            candidates.extend(["dataset/test/"])

        for cand in candidates:
            cand_p = root / cand
            if cand_p.is_file() and cand_p.exists():
                return load_dataframe_file(cand_p)
            elif cand_p.is_dir() and cand_p.exists():
                extract_archives_in_directory(cand_p)
                exts = ("*.tsv", "*.csv", "*.parquet", "*.pq", "*.feather", "*.jsonl", "*.json")
                files = []
                for ext in exts:
                    files.extend(cand_p.glob(ext))
                if split == "train":
                    files = [f for f in files if "test" not in f.name.lower() and "val" not in f.name.lower() and "dev" not in f.name.lower()]
                if files:
                    return pd.concat([load_dataframe_file(f) for f in sorted(files)], ignore_index=True)

        raise FileNotFoundError(f"Could not locate dataset for split '{split}' in '{root}'.")


_ACTIVE_TASK_SPECS: Dict[Path, UniversalTaskSpec] = {}


def load_active_task_spec(
    config_path: Optional[Path | str] = None,
    *,
    force_reload: bool = False,
) -> UniversalTaskSpec:
    """Load the task spec for the selected workspace without cross-project cache reuse."""
    from harness.project import HarnessProject, resolve_project_root

    if config_path is None:
        p = HarnessProject.discover().task_config_path
    else:
        p = Path(config_path).expanduser()
        if not p.is_absolute():
            p = resolve_project_root() / p
        p = p.resolve()
    if p.is_file():
        cached = _ACTIVE_TASK_SPECS.get(p)
        if cached is not None and not force_reload:
            return cached
        spec = UniversalTaskSpec.from_yaml(p)
        _ACTIVE_TASK_SPECS[p] = spec
        return spec
    return UniversalTaskSpec(
        name="universal_task",
        task_type="classification",
        primary_metric="macro_f1",
        direction="maximize",
        inputs=[],
        subtasks={},
        metadata={},
    )
