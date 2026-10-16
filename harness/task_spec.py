"""Universal Task Specification and Conditional Logic Engine for AI-Harness.

Enables declarative definition of multi-task, hierarchical, and conditional
relationships (if-else routing, oracle masks) between prediction targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ntpath
from pathlib import Path
import shutil
import stat
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class InputVariantSpec:
    """Specification for an input representation variant."""

    name: str
    type: str  # 'text', 'text_template', 'tabular', 'multimodal', 'image', 'audio'
    source_columns: List[str] = field(default_factory=list)
    template: Optional[str] = None
    media_dir: Optional[str] = None

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

    def resolve_media_paths(
        self,
        df: pd.DataFrame,
        project_root: Path | str | None = None,
        column: Optional[str] = None,
        extensions: Optional[List[str]] = None,
    ) -> pd.Series:
        """Resolve image/audio file paths in a DataFrame column across candidate data directories.

        Searches candidate roots (media_dir, project_root, dataset/, images/, audio/, /kaggle/input/, etc.)
        and returns a pd.Series of absolute or resolved path strings.
        """
        root = Path(project_root) if project_root else Path.cwd()
        col = column or (self.source_columns[0] if self.source_columns else "filepath")
        if col not in df.columns:
            raise KeyError(f"Media column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")

        candidate_dirs = []
        if self.media_dir:
            md = Path(self.media_dir)
            candidate_dirs.append(md if md.is_absolute() else root / md)
        candidate_dirs.extend([
            root,
            root / "dataset",
            root / "data",
            root / "images",
            root / "audio",
            root / "train",
            root / "test",
            root / "raw",
        ])
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            candidate_dirs.append(kaggle_input)
            for sub in kaggle_input.iterdir():
                if sub.is_dir():
                    candidate_dirs.append(sub)

        discovered_base: Optional[Path] = None

        def resolve_one(val: Any) -> str:
            nonlocal discovered_base
            if pd.isna(val) or val is None or val == "":
                return ""
            raw_str = str(val).strip()
            p = Path(raw_str)
            if p.is_absolute() and p.exists():
                return str(p)

            if discovered_base is not None:
                cand = discovered_base / raw_str
                if cand.exists():
                    return str(cand)
                if extensions:
                    for ext in extensions:
                        cand_ext = discovered_base / f"{raw_str}{ext}"
                        if cand_ext.exists():
                            return str(cand_ext)

            for base in candidate_dirs:
                if not base.exists():
                    continue
                cand = base / raw_str
                if cand.exists():
                    discovered_base = base
                    return str(cand)
                for sub in ("images", "train", "test", "audio", "data"):
                    sub_cand = base / sub / raw_str
                    if sub_cand.exists():
                        discovered_base = base / sub
                        return str(sub_cand)
                if extensions:
                    for ext in extensions:
                        cand_ext = base / f"{raw_str}{ext}"
                        if cand_ext.exists():
                            discovered_base = base
                            return str(cand_ext)

            if self.media_dir:
                md = Path(self.media_dir)
                return str(md / raw_str if md.is_absolute() else root / md / raw_str)
            return str(root / raw_str)

        return df[col].apply(resolve_one)


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


TASK_TYPE_NORMALIZATIONS = {
    # Classification
    "classification": "multiclass_classification",
    "multiclass": "multiclass_classification",
    "binary": "binary_classification",
    "binary_class": "binary_classification",
    "multi_label": "multilabel_classification",
    "multilabel": "multilabel_classification",

    # Regression & Forecasting
    "reg": "regression",
    "regression": "regression",
    "forecasting": "time_series_forecasting",
    "time_series_forecasting": "time_series_forecasting",
    "timeseries": "time_series_forecasting",
    "time_series": "time_series_forecasting",
    "ts_forecast": "time_series_forecasting",
    "multi_output_regression": "multi_output_regression",

    # Ranking & Retrieval
    "rank": "ranking",
    "ranking": "ranking",
    "ltr": "ranking",
    "learning_to_rank": "ranking",
    "retrieval": "ranking",
    "recommendation": "ranking",

    # Computer Vision
    "object_detection": "object_detection",
    "detection": "object_detection",
    "segmentation": "segmentation",
    "semantic_segmentation": "segmentation",
    "instance_segmentation": "segmentation",

    # NLP & Speech
    "ner": "token_classification",
    "token_classification": "token_classification",
    "span_extraction": "token_classification",
    "asr": "speech_recognition",
    "speech_recognition": "speech_recognition",
    "transcription": "speech_recognition",
    "generation": "text_generation",
    "text_generation": "text_generation",
    "seq2seq": "text_generation",
    "summarization": "text_generation",
    "translation": "text_generation",
}


@dataclass(init=False)
class SubtaskSpec:
    """Specification for an individual subtask."""

    id: str
    name: str
    task_type: str
    column: Optional[str] = None
    column_prefix: Optional[str] = None  # Column prefix for multi-label targets (e.g., 'label_')
    classes: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)  # For multi-label
    primary_metric: str = "macro_f1"
    direction: str = "maximize"
    weight: float = 1.0
    condition: Optional[ConditionalRoutingRule] = None

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        task_type: str = "binary_classification",
        column: Optional[str] = None,
        column_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        primary_metric: Optional[str] = None,
        direction: str = "maximize",
        weight: float = 1.0,
        condition: Optional[ConditionalRoutingRule] = None,
        *,
        metric: Optional[str] = None,
    ) -> None:
        # ``metric`` is accepted as an alias for ``primary_metric``: task configs
        # and agents routinely use the singular name, and rejecting it at
        # construction time is a pure friction bug (no information is lost).
        if primary_metric is None:
            primary_metric = metric if metric is not None else "macro_f1"
        elif metric is not None and metric != primary_metric:
            raise ValueError(
                f"Conflicting metric aliases: primary_metric={primary_metric!r} "
                f"and metric={metric!r}"
            )
        self.id = id
        self.name = name if name is not None else id
        self.task_type = task_type
        self.column = column
        self.column_prefix = column_prefix
        self.classes = list(classes) if classes else []
        self.labels = list(labels) if labels else []
        self.primary_metric = primary_metric
        self.direction = direction
        self.weight = weight
        self.condition = condition

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

        raw_type = str(data.get("task_type", data.get("type", "binary_classification"))).lower().strip()
        resolved_type = TASK_TYPE_NORMALIZATIONS.get(raw_type, raw_type)
        metric = data.get("primary_metric", data.get("metric", "macro_f1"))
        from harness.evaluation import default_metric_direction
        direction = data.get("direction", default_metric_direction(metric))

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            task_type=resolved_type,
            column=data.get("column"),
            column_prefix=data.get("column_prefix"),
            classes=data.get("classes", []),
            labels=data.get("labels", []),
            primary_metric=metric,
            direction=direction,
            weight=float(data.get("weight", 1.0)),
            condition=cond_obj,
        )


@dataclass
class ValidationSpec:
    """Specification for validation strategy across ML, DL, and LLM approaches."""

    strategy: str = "kfold"
    num_folds: int = 5
    dev_split_name: str = "dev"
    probe_samples: int = 200
    split_column: Optional[str] = None
    split_file: Optional[str] = None
    embargo_pct: float = 0.01

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
            embargo_pct=float(data.get("embargo_pct", 0.01)),
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
        """ID column from data config or task metadata."""
        data_cfg = self.metadata.get("data_config") or self.metadata.get("data", {})
        return (
            data_cfg.get("id_column")
            or self.metadata.get("id_column")
            or self.metadata.get("id_col")
            or "id"
        )

    @property
    def group_column(self) -> Optional[str]:
        """Group column for leakage prevention."""
        data_cfg = self.metadata.get("data_config") or self.metadata.get("data", {})
        return data_cfg.get("group_column") or self.metadata.get("group_column")

    @property
    def modality(self):
        return self.metadata.get("modality", "tabular")

    @property
    def inputs_dict(self):
        return {item.name: item for item in self.inputs}

    @property
    def data_config(self):
        return self.metadata.get("data_config") or self.metadata.get("data", {})

    @property
    def submission_spec(self) -> Optional[Dict[str, Any]]:
        """Resolved submission specification from metadata."""
        return self.metadata.get("submission_spec")

    @property
    def execution_config(self) -> Dict[str, Any]:
        """Execution policy declared in the task contract.

        Recognized keys:
            backend (str): "local" | "kaggle" | "modal" - default execution
                backend for `run_full_validation`. Overrides the per-call
                "local" default so a Kaggle-locked workspace does not depend
                on every agent remembering to pass `--backend kaggle`.
            locked (bool): when true, execution actions must use `backend`
                and requests for any other backend are rejected.
        """
        exec_cfg = self.metadata.get("execution")
        return exec_cfg if isinstance(exec_cfg, dict) else {}

    @property
    def execution_backend(self) -> Optional[str]:
        """Contract default execution backend, or None when undeclared."""
        backend = self.execution_config.get("backend")
        return str(backend).strip().lower() if backend else None

    @property
    def execution_locked(self) -> bool:
        """True when the contract locks the execution backend."""
        return bool(self.execution_config.get("locked", False))

    @property
    def test_path(self) -> Optional[str]:
        """Resolved test set path from data config or metadata."""
        return self.data_config.get("test_path") or self.metadata.get("test_path")

    @property
    def train_path(self) -> Optional[str]:
        """Resolved train set path from data config or metadata."""
        return self.data_config.get("train_path") or self.metadata.get("train_path")

    @property
    def dev_path(self) -> Optional[str]:
        """Resolved dev/validation set path from data config or metadata."""
        return self.data_config.get("dev_path") or self.metadata.get("dev_path")

    def resolve_seed(self, runtime_seed=42):
        return int(self.metadata.get("seed",self.metadata.get("validation", {}).get("seed", self.data_config.get("seed", runtime_seed))))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> UniversalTaskSpec:
        from harness.evaluation import metric_name
        task_cfg = cfg.get("task", cfg)
        inputs_list = []
        inputs_raw = task_cfg.get("inputs", [])
        if isinstance(inputs_raw, dict):
            inputs_raw = [{"name": key, **value} for key, value in inputs_raw.items()]
        for inp in inputs_raw:
            inputs_list.append(
                InputVariantSpec(
                    name=inp.get("name", inp.get("id")),
                    type=inp.get("type", "text"),
                    source_columns=inp.get("source_columns", []),
                    template=inp.get("template"),
                    media_dir=inp.get("media_dir"),
                )
            )

        subtasks_dict = {}
        st_configs = task_cfg.get("subtasks", task_cfg.get("metrics", []))
        if isinstance(st_configs,dict): st_configs=[{"id":key,**value} for key,value in st_configs.items()]
        for st in st_configs:
            spec = SubtaskSpec.from_dict(st)
            subtasks_dict[spec.id] = spec

        val_cfg = task_cfg.get("validation", cfg.get("validation", {}))
        val_spec = ValidationSpec.from_dict(val_cfg)

        return cls(
            name=task_cfg.get("name", task_cfg.get("task_name", "universal_task")),
            task_type={"Tabular Regression":"regression","Tabular Binary Classification":"classification","Multi-Task Text Classification":"multi_task_classification","multi_label":"multilabel_classification"}.get(task_cfg.get("task_type",task_cfg.get("type")),task_cfg.get("task_type", task_cfg.get("type", "classification"))),
            primary_metric=metric_name(task_cfg.get("primary_metric", "macro_f1")),
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
                    raise ValueError(f"Missing conditional parent target {parent_col}")
            else:
                masks[subtask_id] = np.ones(n_samples, dtype=bool)

        return masks

    def evaluate_predictions(self, ground_truth_df, predictions):
        from harness.evaluation import evaluate
        return evaluate(self, ground_truth_df, predictions)

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

        # 1. Custom path override in data_config or metadata
        custom_path = (
            data_cfg.get(f"{split}_path")
            or data_cfg.get(split)
            or self.metadata.get(f"{split}_path")
            or self.metadata.get(split)
        )
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

            # Fallback for cross-environment / portable execution:
            # If an absolute path configured on another machine/container (e.g. /kaggle/working/dataset/train.csv)
            # is missing on this machine, search for the basename in root and root/dataset/
            for candidate in [root / p.name, root / "dataset" / p.name]:
                if candidate.is_file() and candidate.exists():
                    return load_dataframe_file(candidate)
                elif candidate.is_dir() and candidate.exists():
                    extract_archives_in_directory(candidate)
                    exts = ("*.tsv", "*.csv", "*.parquet", "*.pq", "*.feather", "*.jsonl", "*.json")
                    files = sorted([f for ext in exts for f in candidate.glob(ext)])
                    if files:
                        return pd.concat([load_dataframe_file(f) for f in files], ignore_index=True)

            raise FileNotFoundError(f"Declared {split} dataset path is missing or empty: {p}")

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
        import hashlib
        content_hash=hashlib.sha256(p.read_bytes()).hexdigest()
        cached = _ACTIVE_TASK_SPECS.get(p)
        if cached is not None and cached[0]==content_hash and not force_reload:
            return cached[1]
        spec = UniversalTaskSpec.from_yaml(p)
        _ACTIVE_TASK_SPECS[p] = (content_hash,spec)
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
