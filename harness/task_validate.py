"""Static validation of ``task.yaml`` — instant feedback, no data rows.

Failure modes that currently surface only mid-run (minutes to hours in):

- declared data files that do not exist,
- subtask/input columns absent from the training set,
- GroupKFold with a group column that has fewer uniques than folds
  (observed live: ``Null validation folds``),
- constant target/fold columns,
- unknown task types or directions,
- metric/direction spec inconsistencies that the generic aggregate would
  reject only at scoring time (observed live: a subtask declared
  ``primary_metric: rmse, direction: maximize`` under a spec-level
  ``rhae``/maximize — every official evaluation failed with
  ``Primary metric/direction must match the generic aggregate`` after the
  compute had already run),
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
    "interactive_agent",
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
#: Substring tokens that indicate an uncustomized template placeholder
#: (adoption guard — warn when a config was copied from a template without
#: being customized, so the agent is nudged to fill in real values).
PLACEHOLDER_TOKENS = ("YOUR_", "CHANGE_ME", "REPLACE_ME", "FILL_IN")
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
        import numpy as np
        import pyarrow.parquet as pq

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

    # ---- metric direction consistency + generic-aggregate feasibility ----
    # The generic aggregate scores one (metric, direction) pair that must
    # equal the spec's primary metric/direction, and refuses metrics declared
    # against their canonical direction. Catch these at spec time instead of
    # after an expensive run has scored.
    from harness.evaluation import (
        GENERIC_ALLOWED_METRICS,
        canonical_direction_conflicts,
        metric_name,
    )

    spec_metric = str(getattr(spec, "primary_metric", "") or "")
    spec_dir = str(getattr(spec, "direction", "") or "")
    spec_conflict = canonical_direction_conflicts(spec_metric, spec_dir)
    if spec_conflict:
        warnings.append(
            f"spec primary_metric {spec_metric!r}: {spec_conflict}"
        )

    active_pairs = []
    unsupported = []
    for sid, st in subtasks.items():
        st_metric = str(getattr(st, "primary_metric", "") or "")
        st_dir = str(getattr(st, "direction", "") or "")
        st_type = str(getattr(st, "task_type", "") or "")
        conflict = canonical_direction_conflicts(st_metric, st_dir)
        if conflict:
            warnings.append(f"subtask {sid}: {conflict}")
        try:
            weight = float(getattr(st, "weight", 1.0) or 0.0)
        except (TypeError, ValueError):
            weight = 1.0
        if weight > 0 and st_metric:
            active_pairs.append((metric_name(st_metric), st_dir or "maximize"))
            # The generic aggregate cannot score a metric that is not in its
            # available set for the subtask's task type (e.g. a custom
            # competition metric under a stock task type). Flag it at spec
            # time; it is only fatal when no custom scorer is declared.
            if st_type in GENERIC_ALLOWED_METRICS and metric_name(
                st_metric
            ) not in GENERIC_ALLOWED_METRICS[st_type]:
                unsupported.append(
                    f"subtask {sid}: {metric_name(st_metric)} is not a generic "
                    f"metric for {st_type!r}"
                )

    paths_cfg = dict((getattr(spec, "metadata", {}) or {}).get("paths", {}) or {})
    custom_scorer = str(paths_cfg.get("metrics_evaluator", "") or "")
    spec_pair = (metric_name(spec_metric), spec_dir) if spec_metric else None
    if len(set(active_pairs)) > 1:
        msg = (
            f"subtasks mix metrics/directions {sorted(set(active_pairs))} — "
            "the generic aggregate cannot score them"
        )
        if custom_scorer:
            warnings.append(
                msg + f"; the official score will come from "
                f"paths.metrics_evaluator {custom_scorer!r}"
            )
        else:
            errors.append(
                msg + " — declare one consistent subtask metric matching the "
                "spec, or configure paths.metrics_evaluator in the task spec"
            )
    elif active_pairs and spec_pair and active_pairs[0] != spec_pair:
        msg = (
            f"subtask {active_pairs[0]} does not match the spec's "
            f"primary_metric/direction {spec_pair} — the generic aggregate "
            "refuses to score this spec"
        )
        if custom_scorer:
            warnings.append(
                msg + f"; the official score will come from "
                f"paths.metrics_evaluator {custom_scorer!r} (make sure it "
                f"returns the spec's primary metric {spec_pair[0]!r})"
            )
        else:
            errors.append(
                msg + " — align the subtask primary_metric/direction with "
                "the spec, or declare paths.metrics_evaluator"
            )
    if unsupported:
        _unsupported_msg = "; ".join(unsupported)
        if custom_scorer:
            warnings.append(
                f"{_unsupported_msg} — the generic aggregate cannot score "
                "this subtask; the official score will come from "
                f"paths.metrics_evaluator {custom_scorer!r}"
            )
        else:
            errors.append(
                f"{_unsupported_msg} — use a generic metric for that task "
                "type, or declare paths.metrics_evaluator"
            )

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

    # ---- template not customized (adoption guard) ----------------------
    # Warn (do not error) when train_path / target_column still carry a
    # template placeholder token — the config was copied without being
    # customized. This is the single most common "copied template, ran
    # as-is" failure mode.
    if train_rel and any(tok in str(train_rel) for tok in PLACEHOLDER_TOKENS):
        warnings.append(
            f"train_path {train_rel!r} looks like an uncustomized template "
            "placeholder — did you forget to customize the config?"
        )
    raw_target_col = data_cfg.get("target_column")
    if raw_target_col and any(tok in str(raw_target_col) for tok in PLACEHOLDER_TOKENS):
        warnings.append(
            f"target_column {str(raw_target_col)!r} looks like an uncustomized "
            "template placeholder — did you forget to customize the config?"
        )
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
