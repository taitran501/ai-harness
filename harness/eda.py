"""Privacy-conscious, task-agnostic exploratory data analysis helpers."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from harness.io_utils import atomic_write_json
from harness.project import HarnessProject
from harness.task_spec import load_dataframe_file


SUPPORTED_DATA_SUFFIXES = {".csv", ".tsv", ".tab", ".parquet", ".pq", ".feather", ".json", ".jsonl", ".ndjson"}


def _target_columns(task: Dict[str, Any]) -> List[str]:
    data = task.get("data_config") or task.get("data") or {}
    found: List[str] = []
    for key in ("target_column", "target_columns", "label_column", "label_columns"):
        value = data.get(key)
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            found.extend(str(item) for item in value)
    for subtask in task.get("subtasks", []):
        if isinstance(subtask, dict) and subtask.get("column"):
            found.append(str(subtask["column"]))
    return list(dict.fromkeys(found))


def profile_dataframe(
    data: pd.DataFrame,
    *,
    target_columns: Optional[Iterable[str]] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize schema, missingness, distributions, lengths, and duplicates without raw rows."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("EDA expects a pandas DataFrame.")
    targets = set(target_columns or [])
    columns: List[Dict[str, Any]] = []
    target_profiles: Dict[str, Any] = {}
    for name in data.columns:
        series = data[name]
        non_null = series.dropna()
        profile: Dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_fraction": round(float(series.isna().mean()), 6) if len(series) else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                profile["numeric"] = {
                    "mean": _finite_or_none(numeric.mean()),
                    "std": _finite_or_none(numeric.std()),
                    "min": _finite_or_none(numeric.min()),
                    "median": _finite_or_none(numeric.median()),
                    "max": _finite_or_none(numeric.max()),
                    "zero_count": int((numeric == 0).sum()),
                }
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            lengths = non_null.astype(str).str.len()
            if len(lengths):
                profile["text_length"] = {
                    "mean": _finite_or_none(lengths.mean()),
                    "median": _finite_or_none(lengths.median()),
                    "p95": _finite_or_none(lengths.quantile(0.95)),
                    "max": int(lengths.max()),
                    "empty_count": int((non_null.astype(str).str.strip() == "").sum()),
                }
        columns.append(profile)
        if str(name) in targets:
            counts = series.astype("string").fillna("<missing>").value_counts(dropna=False)
            target_profiles[str(name)] = {
                "class_count": int(series.nunique(dropna=True)),
                "counts": {str(key): int(value) for key, value in counts.head(100).items()},
                "tail_class_count_omitted": max(0, int(len(counts) - 100)),
            }
    return {
        "source": source,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "duplicate_row_count": int(data.duplicated().sum()) if len(data.columns) else 0,
        "columns": columns,
        "target_distributions": target_profiles,
        "privacy": "No raw row values are included in this report.",
    }


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.suffix.lower() in SUPPORTED_DATA_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )
    if not files:
        raise FileNotFoundError(f"No supported dataset files found under {path}")
    return files


def _resolve_training_path(project: HarnessProject, task: Dict[str, Any], override: Optional[str]) -> Path:
    value = override or (task.get("data_config") or task.get("data") or {}).get("train_path") or "dataset"
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project.root / path).resolve()


def run_eda(
    project_root: Path | str = ".",
    *,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load task training data through its adapter or common file formats and save reports."""
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.task_spec import load_active_task_spec

    project = HarnessProject.discover(project_root)
    document = yaml.safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
    task = document.get("task", document)
    task_spec = load_active_task_spec(project.task_config_path)
    plugin_context = TaskPluginContext(
        project_root=project.root,
        task_config_path=project.task_config_path,
        task_spec=task_spec,
        plugin_id=project.plugin_id or "default",
    )
    plugin = load_task_plugin(project.plugin_id, plugin_context)

    frames: List[tuple[str, pd.DataFrame]] = []
    if data_path:
        # A selected external dataset is an explicit request to profile that source,
        # even when the active task plugin provides its own training-data loader.
        resolved = _resolve_training_path(project, task, data_path)
        for file_path in _input_files(resolved):
            frames.append((file_path.relative_to(project.root).as_posix(), load_dataframe_file(file_path)))
    else:
        plugin_frame = plugin.load_training_data(plugin_context)
        if plugin_frame is not None:
            frames.append((f"task-plugin:{plugin.plugin_id}", plugin_frame))
        else:
            resolved = _resolve_training_path(project, task, data_path)
            for file_path in _input_files(resolved):
                frames.append((file_path.relative_to(project.root).as_posix(), load_dataframe_file(file_path)))

    target_columns = _target_columns(task)
    reports = [profile_dataframe(frame, target_columns=target_columns, source=source) for source, frame in frames]
    extension_report = plugin.extend_eda_report(plugin_context, reports) if hasattr(plugin, "extend_eda_report") else None
    if extension_report:
        for report, extension in zip(reports, extension_report if isinstance(extension_report, list) else [extension_report]):
            report["task_plugin_extension"] = extension

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest_root = Path(output_dir).expanduser() if output_dir else project.artifacts_dir / "eda"
    if not dest_root.is_absolute():
        dest_root = project.root / dest_root
    dest = dest_root.resolve()
    try:
        dest.relative_to(project.root)
    except ValueError as exc:
        raise ValueError("EDA output directory must stay inside the attached workspace.") from exc
    dest.mkdir(parents=True, exist_ok=True)
    json_path = dest / f"eda_{timestamp}.json"
    markdown_path = dest / f"eda_{timestamp}.md"
    result = {
        "schema_version": 1,
        "task_id": project.manifest.get("task_id"),
        "task_name": task.get("name") or task.get("task_name"),
        "plugin_id": project.plugin_id or "default",
        "primary_metric": task.get("primary_metric"),
        "reports": reports,
        "paths": {
            "json": json_path.relative_to(project.root).as_posix(),
            "markdown": markdown_path.relative_to(project.root).as_posix(),
        },
    }
    atomic_write_json(json_path, result, indent=2, ensure_ascii=False)
    markdown_path.write_text(render_eda_markdown(result), encoding="utf-8")
    return result


def render_eda_markdown(result: Dict[str, Any]) -> str:
    lines = [
        f"# EDA summary: {result.get('task_name') or 'task'}",
        "",
        f"- Plugin: `{result.get('plugin_id')}`",
        f"- Primary metric: `{result.get('primary_metric') or 'not declared'}`",
        "- Privacy: no raw row values were copied into this report.",
    ]
    for report in result.get("reports", []):
        lines.extend([
            "",
            f"## {report.get('source')}",
            f"Rows: {report.get('row_count')} | Columns: {report.get('column_count')} | Exact duplicate rows: {report.get('duplicate_row_count')}",
            "",
            "| Column | Type | Missing | Unique | Text length (mean / p95 / max) |",
            "|---|---|---:|---:|---:|",
        ])
        for column in report.get("columns", []):
            lengths = column.get("text_length", {})
            length_text = " / ".join(
                "—" if lengths.get(key) is None else str(round(lengths[key], 2))
                for key in ("mean", "p95", "max")
            )
            lines.append(
                f"| {column['name']} | {column['dtype']} | {column['missing_count']} ({column['missing_fraction']:.1%}) | "
                f"{column['unique_count']} | {length_text} |"
            )
        for target, profile in report.get("target_distributions", {}).items():
            lines.extend(["", f"### Target distribution: `{target}`", ""])
            lines.extend(f"- `{label}`: {count}" for label, count in profile.get("counts", {}).items())
        if report.get("task_plugin_extension"):
            lines.extend(["", "### Task-plugin checks", "", "```json", json.dumps(report["task_plugin_extension"], indent=2, ensure_ascii=False), "```"])
    lines.extend(["", "## Agent brief", "", "Use this report to identify schema, missingness, imbalance, length, and duplicate-data questions. Confirm task-specific leakage and split constraints before modeling."])
    return "\n".join(lines) + "\n"
