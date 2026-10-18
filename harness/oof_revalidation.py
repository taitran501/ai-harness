"""Re-derive a sealed run's primary score from on-disk evidence.

Used by `ai-harness repair --verify`: when a run's *evidence* files
(metrics.json / OOF / predictions / truth) drifted, repair must not blindly
re-seal. It recomputes the primary metric from the run folder's own
`oof_predictions.csv` (+ fallback `predictions.csv`) and
`evaluation_truth.csv` using the exact scoring path the runner used:

    variant rows  ->  pred_/proba_/classes_ extraction  (mirrors
    UniversalRunner._save_and_finalize)  ->  task_plugin.evaluate_predictions
    (+ optional run_config paths.metrics_evaluator hook, mirroring
    UniversalRunner._evaluate)  ->  resolve_evaluation

If that path diverges from the runner, the recomputed score will not match
the recorded one and repair refuses — which is exactly the failure mode this
gate exists to catch.

This module is deliberately NOT part of the scorer identity hash scope
(`scorer_sha256` seals `harness.evaluation` + `harness.task_spec` sources);
it only *calls* those, so hardening the gate can never invalidate sealed
runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Relative tolerance for comparing the recorded vs recomputed primary score.
# The CSV round-trip is repr-exact for floats, so an identical code path
# reproduces the score exactly; the tolerance absorbs float-format edge cases
# (decimal truncation in custom scorers) without accepting meaningful drift.
SCORE_REL_TOL = 1e-6

_EVIDENCE_FILES = frozenset(
    {
        "metrics.json",
        "oof_predictions.csv",
        "predictions.csv",
        "test_predictions.csv",
        "evaluation_truth.csv",
    }
)

# Files whose drift is "evidence drift": re-sealing them without re-deriving
# the score is precisely the tamper path the gate blocks.
IS_EVIDENCE_FILE = _EVIDENCE_FILES.__contains__


def is_evidence_file(name: str) -> bool:
    return name in _EVIDENCE_FILES


def load_evidence_frame(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load the run's prediction evidence: OOF first, predictions.csv fallback.

    Returns None when neither file exists (the run is then not verifiable).
    """
    for name in ("oof_predictions.csv", "predictions.csv"):
        path = run_dir / name
        if path.is_file():
            return pd.read_csv(path)
    return None


def load_truth_frame(run_dir: Path) -> Optional[pd.DataFrame]:
    path = run_dir / "evaluation_truth.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def build_variant_predictions(selected: pd.DataFrame, task_spec: Any) -> Dict[str, Any]:
    """Extract the prediction mapping from one variant's OOF rows.

    Mirrors the `pred_` / `proba_` / `classes_` extraction in
    `UniversalRunner._save_and_finalize` byte-for-byte: JSON-decoded cell
    values, single-value `classes_` columns, subtask probability columns.
    Both paths MUST stay in sync; the runner imports this function.
    """
    predictions: Dict[str, Any] = {}
    for col in selected:
        if col.startswith("pred_"):
            values = selected[col].tolist()
            predictions[col] = [
                json.loads(v) if isinstance(v, str) and v.startswith("[") else v
                for v in values
            ]
        elif col.startswith("proba_"):
            if any(f"proba_{target}" == col for target in task_spec.subtasks.keys()):
                predictions[col] = [
                    json.loads(v)
                    if isinstance(v, str) and (v.startswith("[") or v.startswith("{"))
                    else v
                    for v in selected[col].tolist()
                ]
        elif col.startswith("classes_"):
            if selected[col].nunique() != 1:
                raise ValueError("Class order changed across folds")
            predictions[col] = json.loads(selected[col].iloc[0])
    return predictions


def _custom_hook(run_config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    hook = run_config.get("paths", {}).get("metrics_evaluator")
    if not hook:
        return None
    hook = str(hook)
    return hook.split(":", 1) if ":" in hook else tuple(hook.rsplit(".", 1))


def _score_variant(
    task_spec: Any,
    plugin: Any,
    plugin_context: Any,
    run_config: Dict[str, Any],
    truth: pd.DataFrame,
    predictions: Dict[str, Any],
) -> dict:
    """Score one variant through the runner's exact evaluation path."""
    import importlib
    import sys

    from harness.evaluation import resolve_evaluation, validate_prediction_contract

    validate_prediction_contract(task_spec, truth, predictions)
    hook = _custom_hook(run_config)
    if hook:
        project_root = getattr(plugin_context, "project_root", None)
        if project_root is not None and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        module, name = hook
        result = getattr(importlib.import_module(module), name)(truth, predictions)
        if not isinstance(result, dict):
            raise TypeError("Custom metrics_evaluator must return a metric mapping")
        try:
            base_result = plugin.evaluate_predictions(plugin_context, truth, predictions)
            result = {**base_result, **result}
        except Exception as exc:  # noqa: BLE001 — mirror runner: surface, keep custom
            result.setdefault("_evaluation_warnings", []).append(
                f"official evaluation failed: {exc.__class__.__name__}: {exc}"
            )
    else:
        result = plugin.evaluate_predictions(plugin_context, truth, predictions)
    return resolve_evaluation(task_spec, result, len(truth))


def rederive_primary_score(
    run_dir: str | Path,
    *,
    task_spec: Any,
    plugin: Any,
    plugin_context: Any,
) -> Dict[str, Any]:
    """Recompute the primary score from on-disk evidence for one run.

    Returns a report:
      verifiable   — evidence was present and could be scored
      ok           — recorded vs recomputed score agree within SCORE_REL_TOL
      recorded     — primary_score from the run's metrics.json (or None)
      recomputed   — primary_score recomputed from OOF + truth (or None)
      selection_variant, detail
    """
    run_dir = Path(run_dir)
    report: Dict[str, Any] = {
        "verifiable": False,
        "ok": False,
        "recorded": None,
        "recomputed": None,
        "selection_variant": None,
        "detail": "",
    }

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        report["detail"] = "metrics.json missing; recorded score unknown"
        return report
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["detail"] = f"metrics.json unreadable: {exc}"
        return report
    recorded = metrics.get("primary_score")
    if not isinstance(recorded, (int, float)):
        report["detail"] = "metrics.json has no numeric primary_score"
        return report
    report["recorded"] = recorded
    selection = metrics.get("selection_variant")

    oof = load_evidence_frame(run_dir)
    truth = load_truth_frame(run_dir)
    if oof is None:
        report["detail"] = "no oof_predictions.csv or predictions.csv on disk"
        return report
    if truth is None:
        report["detail"] = "evaluation_truth.csv missing on disk"
        return report

    try:
        import yaml

        run_config = yaml.safe_load(
            (run_dir / "run_config.yaml").read_text(encoding="utf-8")
        ) or {}
    except (OSError, yaml.YAMLError, ImportError) as exc:
        report["detail"] = f"run_config.yaml unreadable: {exc}"
        return report

    key = task_spec.id_column
    if key not in oof.columns:
        report["detail"] = f"evidence frame lacks id column '{key}'"
        return report

    variants: Dict[str, dict] = {}
    try:
        if "input_variant" in oof.columns:
            for variant in oof["input_variant"].unique():
                selected = oof[oof["input_variant"] == variant]
                merged = selected[[key]].merge(
                    truth, on=key, how="left", validate="one_to_one"
                )
                variants[str(variant)] = _score_variant(
                    task_spec,
                    plugin,
                    plugin_context,
                    run_config,
                    merged,
                    build_variant_predictions(selected, task_spec),
                )
        else:
            merged = oof[[key]].merge(
                truth, on=key, how="left", validate="one_to_one"
            )
            variants[""] = _score_variant(
                task_spec,
                plugin,
                plugin_context,
                run_config,
                merged,
                build_variant_predictions(oof, task_spec),
            )
    except Exception as exc:  # noqa: BLE001 — any scoring failure = not verifiable
        report["detail"] = f"re-derivation failed: {exc.__class__.__name__}: {exc}"
        return report

    target = str(selection) if selection else (list(variants)[0] if variants else None)
    if target is None or target not in variants:
        report["detail"] = (
            f"selection variant '{selection}' not present in evidence "
            f"(variants: {sorted(variants)})"
        )
        return report

    recomputed = variants[target].get("primary_score")
    if not isinstance(recomputed, (int, float)):
        report["detail"] = "recomputed result has no numeric primary_score"
        return report
    report["selection_variant"] = target
    report["recomputed"] = recomputed
    report["verifiable"] = True

    tol = max(1e-9, SCORE_REL_TOL * max(abs(float(recorded)), 1.0))
    if abs(float(recorded) - float(recomputed)) <= tol:
        report["ok"] = True
        report["detail"] = "recorded and recomputed primary scores agree"
    else:
        report["ok"] = False
        report["detail"] = (
            f"primary score mismatch: recorded={recorded} recomputed={recomputed} "
            f"(tolerance {tol:.2e})"
        )
    return report
