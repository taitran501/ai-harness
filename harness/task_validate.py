"""Static validation of ``task.yaml`` — instant feedback, no data rows.

Failure modes that currently surface only mid-run (minutes to hours in):

- declared data files that do not exist,
- subtask/input columns absent from the training set,
- GroupKFold with a group column that has fewer uniques than folds
  (observed live: ``Null validation folds``),
- constant target/fold columns,
- unknown task types or directions,
- hygiene allow-lists that whitelist genuine run-output dirs (observed
  live: ``kernel_output_*``/``kaggle_results`` hidden from the adoption
  scan for months),

are all caught here in well under a second. The validator reads only the
spec YAML plus the cheap metadata of declared datasets (parquet schema /
single-column scans); it never loads row contents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KNOWN_TASK_TYPES = {
    "classification",
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multi_task_classification",
    "regression",
    "ranking",
}
KNOWN_DIRECTIONS = {"maximize", "minimize"}
KNOWN_BACKENDS = {"local", "kaggle", "modal"}
#: Top-level dir names that look like run outputs. Whitelisting one of
#: these in ``task.hygiene.allowed_top_dirs`` is almost always a mistake.
SUSPECT_RUN_OUTPUT_DIRS = (
    "kernel_output",
    "kaggle_results",
    "run_output",
    "submission_output",
)
# Parquet scans (group-unique / target-constant) are bounded: larger files
# get a "skipped" note instead of a slow check.
_MAX_SCAN_BYTES = 100 * 1024 * 1024


def _parquet_columns(path: Path) -> Optional[List[str]]:
    """Column names from the parquet footer; None when unreadable."""
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:  # noqa: BLE001 - validator is advisory on data IO
        return None


def _column_uniques(path: Path, column: str) -> Optional[Any]:
    """Distinct values of one column (numpy set), or None when unreadable.

    Bounded by file size — the checks that use this (GroupKFold group
    breadth, target constancy) are meaningless to run over multi-GB scans
    from a health command.
    """
    try:
        if path.stat().st_size > _MAX_SCAN_BYTES:
            return "too_large"
        import pyarrow.parquet as pq

        import numpy as np

        table = pq.read_table(path, columns=[column])
        arr = np.asarray(table[column].to_pylist())
        return np.unique(arr)
    except Exception:  # noqa: BLE001 - validator is advisory on data IO
        return None


def _load_spec(config_path: Path) -> Tuple[Optional[Any], List[str]]:
    from harness.task_spec import UniversalTaskSpec

    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return None, [f"cannot read config: {exc}"]
    try:
        import yaml

        raw = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001
        return None, [f"invalid YAML: {exc}"]
    if not isinstance(raw, dict):
        return None, ["top-level YAML must be a mapping"]
    try:
        return UniversalTaskSpec.from_dict(raw), []
    except Exception as exc:  # noqa: BLE001
        return None, [f"task spec failed to parse: {exc}"]


def validate_task(config_path: Path, root: Path) -> Dict[str, Any]:
    """Validate one task spec against its declared data. Never raises."""
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    config_path = config_path if config_path.is_absolute() else root / config_path
    if not config_path.is_file():
        return {"ok": False, "config": str(config_path), "errors":
                [f"task config not found: {config_path}"], "warnings": [], "info": {}}

    spec, parse_errors = _load_spec(config_path)
    errors.extend(parse_errors)
    if spec is None:
        return {"ok": False, "config": str(config_path), "errors": errors,
                "warnings": warnings, "info": info}

    # ---- basic fields -------------------------------------------------
    ttype = str(getattr(spec, "task_type", "") or "")
    if ttype not in KNOWN_TASK_TYPES:
        errors.append(f"unknown task_type {ttype!r} (known: {sorted(KNOWN_TASK_TYPES)})")
    if str(getattr(spec, "direction", "") or "") not in KNOWN_DIRECTIONS:
        errors.append(f"unknown direction {getattr(spec, 'direction', None)!r} "
                      f"(known: {sorted(KNOWN_DIRECTIONS)})")
    if not str(getattr(spec, "primary_metric", "") or "").strip():
        errors.append("primary_metric is empty")

    # ---- subtasks / metrics --------------------------------------------
    subtasks = dict(getattr(spec, "subtasks", {}) or {})
    if not subtasks:
        warnings.append("no subtasks (or metrics) declared; the runner will "
                        "have no training target to optimize")
    for sid, st in subtasks.items():
        st_type = str(getattr(st, "task_type", "") or "")
        if st_type and st_type not in KNOWN_TASK_TYPES:
            warnings.append(f"subtask {sid}: unknown task_type {st_type!r}")
        if not str(getattr(st, "column", "") or ""):
            warnings.append(f"subtask {sid}: no target column declared")
        st_dir = str(getattr(st, "direction", "") or "")
        if st_dir and st_dir not in KNOWN_DIRECTIONS:
            warnings.append(f"subtask {sid}: unknown direction {st_dir!r}")

    # ---- execution backend --------------------------------------------
    exec_cfg = dict(getattr(spec, "execution_config", {}) or {})
    backend = str(exec_cfg.get("backend", "local"))
    if backend not in KNOWN_BACKENDS:
        errors.append(f"execution.backend {backend!r} not in {sorted(KNOWN_BACKENDS)}")
    if exec_cfg.get("locked") and backend == "kaggle":
        info["kaggle_locked"] = True

    # ---- hygiene allow-list sanity -------------------------------------
    meta = dict(getattr(spec, "metadata", {}) or {})
    allowed = (meta.get("hygiene", {}) or {}).get("allowed_top_dirs", [])
    for d in allowed or []:
        name = str(d).lower()
        if any(name == s or name.startswith(s + "_") for s in SUSPECT_RUN_OUTPUT_DIRS):
            warnings.append(
                f"hygiene.allowed_top_dirs whitelists {d!r}, which looks like a "
                f"run-output dir — verified runs belong under .ai-harness/artifacts/, "
                f"not in the allow-list (bypasses would be hidden from the adoption scan)"
            )

    # ---- data files + column cross-checks ------------------------------
    data_cfg = dict(getattr(spec, "data_config", {}) or {})
    train_rel = data_cfg.get("train_path")
    train_path: Optional[Path] = None
    if train_rel:
        train_path = (root / train_rel) if not Path(train_rel).is_absolute() else Path(train_rel)
        if not train_path.is_file():
            errors.append(f"train_path missing: {train_rel}")
        else:
            info["train_rows"] = _train_row_count(train_path)
    test_rel = data_cfg.get("test_path")
    if test_rel:
        test_path = (root / test_rel) if not Path(test_rel).is_absolute() else Path(test_rel)
        if not test_path.exists():
            warnings.append(f"test_path not found: {test_rel}")
    for aux in data_cfg.get("auxiliary_train_paths", []) or []:
        p = (root / aux) if not Path(aux).is_absolute() else Path(aux)
        if not p.is_file():
            warnings.append(f"auxiliary_train_paths entry not found: {aux}")

    train_cols: Optional[List[str]] = None
    if train_path is not None and train_path.is_file():
        train_cols = _parquet_columns(train_path)
        if train_cols is not None:
            info["train_columns"] = len(train_cols)
            missing: List[str] = []
            for sid, st in subtasks.items():
                col = str(getattr(st, "column", "") or "")
                if col and col not in train_cols:
                    missing.append(f"subtask {sid} column {col!r}")
            for inp in getattr(spec, "inputs", []) or []:
                for col in getattr(inp, "source_columns", []) or []:
                    if col not in train_cols:
                        missing.append(f"input {getattr(inp, 'name', '?')} column {col!r}")
            id_col = data_cfg.get("id_column")
            if id_col and id_col not in train_cols:
                missing.append(f"id_column {id_col!r}")
            if missing:
                errors.append("columns not present in "
                              f"{Path(train_rel).name if train_rel else 'train file'}: "
                              + ", ".join(sorted(set(missing))))

    # ---- fold strategy sanity (AIMO's live 'Null validation folds' class)
    strategy = str(data_cfg.get("split_strategy", "") or "")
    n_folds = data_cfg.get("n_folds") or getattr(spec.validation, "num_folds", None)
    if n_folds is not None:
        try:
            n_folds = int(n_folds)
            if n_folds < 2:
                warnings.append(f"n_folds={n_folds}: cross-validation needs >= 2 folds")
        except (TypeError, ValueError):
            warnings.append(f"n_folds is not an integer: {n_folds!r}")
    group_col = data_cfg.get("group_column")
    if "group" in strategy.lower() and group_col and train_path is not None and train_path.is_file():
        uniques = _column_uniques(train_path, str(group_col))
        if uniques is None:
            warnings.append(f"group_column {group_col!r} not readable; group breadth unverified")
        elif isinstance(uniques, str):  # "too_large" sentinel
            info["group_breadth"] = "skipped (file > 100 MB)"
        elif len(uniques) < (n_folds or 1):
            errors.append(
                f"split_strategy={strategy} needs {n_folds} folds but group_column "
                f"{group_col!r} has only {len(uniques)} unique values — "
                f"GroupKFold cannot produce {n_folds} non-empty folds"
            )

    # ---- target constancy (a constant target can never beat baseline) ---
    target_col = data_cfg.get("target_column") or (
        subtasks[next(iter(subtasks))].column if subtasks else None
    )
    if target_col and train_path is not None and train_path.is_file():
        uniques = _column_uniques(train_path, str(target_col))
        if isinstance(uniques, str):  # "too_large" sentinel
            info["target_constancy"] = "skipped (file > 100 MB)"
        elif uniques is not None and len(uniques) <= 1:
            warnings.append(
                f"target column {str(target_col)!r} is constant in the train file — "
                f"any metric optimization is meaningless on this split"
            )

    ok = not errors
    return {"ok": ok, "config": str(config_path), "errors": errors,
            "warnings": warnings, "info": info}


def _train_row_count(path: Path) -> Optional[int]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:  # noqa: BLE001
        return None
