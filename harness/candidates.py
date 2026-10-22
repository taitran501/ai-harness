"""Candidate construction from exact, verified and aligned evidence."""

from __future__ import annotations

from harness.io_utils import safe_relpath

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from harness.artifacts import (
    digest,
    ensure_canonical_truth,
    mirror_csv,
    sha256,
    validate_run,
)
from harness.ensemble_guard import EnsembleGuard
from harness.evaluation import resolve_evaluation
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.oof_revalidation import score_predictions
from harness.task_spec import load_active_task_spec


def predictions(frame):
    result = {}
    for col in frame:
        if col.startswith(("pred_", "proba_")):
            result[col] = [
                json.loads(v) if isinstance(v, str) and v.startswith("[") else v
                for v in frame[col]
            ]
        elif col.startswith("classes_"):
            if frame[col].nunique() != 1:
                raise ValueError("Inconsistent class order")
            result[col] = json.loads(frame[col].iloc[0])
    return result


def _candidate_recipe_fingerprint(mode, ids, source_map):
    """Hash the candidate recipe so a re-synthesis of the same sources+mode+weights
    is detectable regardless of the display name the agent gives it."""
    import hashlib
    import json

    normalized = {
        "mode": mode,
        "source_ids": sorted(ids),
        "weights": sorted([float(w) for w in source_map.get("weights", [])])
        if source_map.get("weights")
        else None,
        "routing": (
            {k: sorted(v) if isinstance(v, list) else v
             for k, v in sorted((source_map.get("routing") or source_map.get("target_routing") or {}).items())}
            if (source_map.get("routing") or source_map.get("target_routing"))
            else None
        ),
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidate(project, name, mode, source_map):
    spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(
        project.root, project.task_config_path, spec, project.plugin_id or "default"
    )
    plugin = load_task_plugin(project.plugin_id, context)
    routing = source_map.get("routing") or source_map.get("target_routing")
    if mode in {"single_model", "checkpoint_selection"}:
        ids = [source_map.get("source_exp") or source_map.get("model_exp")]
    elif mode == "specialist_router":
        if not routing:
            raise ValueError("Specialist routing must be explicit")
        ids = list(dict.fromkeys(routing.values()))
    elif mode in {"ensemble_vote", "weighted_blend", "probability_blend"}:
        ids = source_map.get("components", [])
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("Require distinct ensemble sources")
    else:
        raise ValueError("Unsupported candidate mode")
    if not ids or any(not x or Path(x).name != x for x in ids):
        raise ValueError("Require exact source experiment IDs")
    frames = []
    manifests = []
    scores = []
    key = spec.id_column
    for exp in ids:
        folder = project.artifacts_dir / exp
        manifests.append(
            validate_run(
                folder, task_spec=spec, plugin=plugin, expected={"exp_id": exp}
            )
        )
        metrics = json.loads((folder / "metrics.json").read_text())
        frame = pd.read_csv(folder / "oof_predictions.csv")
        frame = frame[frame.input_variant == metrics["selection_variant"]].copy()
        if frame.empty or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError("Invalid source IDs")
        if frames and set(frame[key]) != set(frames[0][key]):
            raise ValueError("Source ID sets differ")
        frames.append(
            frame.set_index(key)
            .reindex(frames[0][key] if frames else frame[key])
            .reset_index()
        )
        scores.append(metrics["utility"])
    for manifest in manifests[1:]:
        for field in ("task_sha256", "scorer_sha256", "custom_scorer_sha256", "split_sha256", "data_sha256"):
            if manifest.get(field) != manifests[0].get(field):
                raise ValueError("Source contract/split mismatch: " + field)
    has_test_preds = all(
        (project.artifacts_dir / exp / "test_predictions.csv").is_file()
        for exp in ids
    )
    test_frames = []
    if has_test_preds:
        for exp in ids:
            t_frame = pd.read_csv(project.artifacts_dir / exp / "test_predictions.csv")
            if t_frame.empty or key not in t_frame or t_frame[key].isna().any() or t_frame[key].duplicated().any():
                has_test_preds = False
                break
            if test_frames and set(t_frame[key]) != set(test_frames[0][key]):
                has_test_preds = False
                break
            test_frames.append(
                t_frame.set_index(key)
                .reindex(test_frames[0][key] if test_frames else t_frame[key])
                .reset_index()
            )
    merged = frames[0].copy()
    merged_test = test_frames[0].copy() if has_test_preds else None
    if mode == "specialist_router":
        merged = merged[[key, "fold"]].copy()
        if has_test_preds:
            merged_test = test_frames[0][[key]].copy()
        for target, source in routing.items():
            target = target.removeprefix("pred_")
            frame = frames[ids.index(source)]
            for prefix in ("pred_", "proba_", "classes_"):
                col = prefix + target
                if prefix == "pred_" and col not in frame:
                    raise ValueError("Missing specialist target: " + col)
                if col in frame:
                    merged[col] = frame[col].values
            if has_test_preds:
                t_f = test_frames[ids.index(source)]
                for prefix in ("pred_", "proba_", "classes_"):
                    col = prefix + target
                    if col in t_f:
                        merged_test[col] = t_f[col].values
                if target in t_f and target not in merged_test:
                    merged_test[target] = t_f[target].values
                elif "pred_" + target in merged_test and target not in merged_test:
                    merged_test[target] = merged_test["pred_" + target]
                if target in spec.subtasks and spec.subtasks[target].column:
                    st_col = spec.subtasks[target].column
                    if st_col not in merged_test and "pred_" + target in merged_test:
                        merged_test[st_col] = merged_test["pred_" + target]
    elif len(ids) > 1:
        weights = np.asarray(source_map.get("weights", [1.0] * len(ids)), float)
        if (
            len(weights) != len(ids)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Invalid blend weights")
        weights /= weights.sum()
        if has_test_preds:
            merged_test = test_frames[0][[key]].copy()
        for target, st in spec.subtasks.items():
            col = "pred_" + target
            if any(col not in f for f in frames):
                raise ValueError("Missing target: " + target)
            if st.weight > 0:
                checked = [
                    (
                        f.assign(**{col: f[col].astype(str)})
                        if st.task_type != "regression"
                        else f
                    )
                    for f in frames
                ]
                sub_spec = getattr(spec, "submission_spec", {}) or {}
                max_k = 0.9999 if st.task_type in ("regression", "ranking") else float(sub_spec.get("max_kappa", 0.85))
                min_dis = float(sub_spec.get("min_disagreement", 5.0))
                passed, errors, _ = EnsembleGuard.audit_constituent_diversity(
                    checked, col=col, max_kappa=max_k, min_disagreement_pct=min_dis
                )
                if not passed:
                    raise ValueError("Diversity guard: " + "; ".join(errors))
            if st.task_type in ("regression", "ranking"):
                merged[col] = sum(
                    w * f[col].astype(float).values for w, f in zip(weights, frames)
                )
                if has_test_preds:
                    merged_test[col] = sum(
                        w * f[col].astype(float).values for w, f in zip(weights, test_frames)
                    )
            elif (
                st.task_type == "multilabel_classification"
                and mode != "probability_blend"
            ):
                values = np.asarray([predictions(f)[col] for f in frames], float)
                combined = np.sum(values * weights[:, None, None], axis=0) > 0.5
                merged[col] = [json.dumps(x.astype(int).tolist()) for x in combined]
                for prefix in ("proba_", "classes_"):
                    merged.drop(
                        columns=[prefix + target], errors="ignore", inplace=True
                    )
                if has_test_preds:
                    t_values = np.asarray([predictions(f)[col] for f in test_frames], float)
                    t_comb = np.sum(t_values * weights[:, None, None], axis=0) > 0.5
                    merged_test[col] = [json.dumps(x.astype(int).tolist()) for x in t_comb]
                    for prefix in ("proba_", "classes_"):
                        merged_test.drop(
                            columns=[prefix + target], errors="ignore", inplace=True
                        )
            elif mode == "probability_blend":
                orders = [
                    json.loads(f["classes_" + target].iloc[0])
                    if isinstance(f["classes_" + target].iloc[0], str)
                    else f["classes_" + target].iloc[0]
                    for f in frames
                ]
                if any(x != orders[0] for x in orders):
                    raise ValueError("Probability class orders differ")
                proba = sum(
                    w * np.asarray(predictions(f)["proba_" + target], float)
                    for w, f in zip(weights, frames)
                )
                merged["proba_" + target] = [json.dumps(x.tolist()) for x in proba]
                if st.task_type == "multilabel_classification":
                    merged[col] = [
                        json.dumps((x > 0.5).astype(int).tolist()) for x in proba
                    ]
                else:
                    merged[col] = np.asarray(orders[0])[proba.argmax(axis=1)]
                if has_test_preds:
                    t_orders = [
                        json.loads(f["classes_" + target].iloc[0])
                        if isinstance(f["classes_" + target].iloc[0], str)
                        else f["classes_" + target].iloc[0]
                        for f in test_frames
                    ]
                    if any(x != t_orders[0] for x in t_orders):
                        raise ValueError("Probability class orders differ in test predictions")
                    t_proba = sum(
                        w * np.asarray(predictions(f)["proba_" + target], float)
                        for w, f in zip(weights, test_frames)
                    )
                    merged_test["proba_" + target] = [json.dumps(np.asarray(x).tolist()) for x in t_proba]
                    merged_test["classes_" + target] = json.dumps(np.asarray(t_orders[0]).tolist())
                    if st.task_type == "multilabel_classification":
                        merged_test[col] = [
                            json.dumps((x > 0.5).astype(int).tolist()) for x in t_proba
                        ]
                    else:
                        merged_test[col] = np.asarray(t_orders[0])[t_proba.argmax(axis=1)]
            else:
                order_raw = list(st.classes) if st.classes else []
                order = list(map(str, st.classes))
                if not order:
                    raise ValueError("Voting requires declared class order")
                values = np.asarray([f[col].astype(str).values for f in frames])
                votes = np.asarray(
                    [np.sum((values == c) * weights[:, None], axis=0) for c in order]
                )
                merged[col] = np.asarray(order_raw or order)[votes.argmax(axis=0)]
                for prefix in ("proba_", "classes_"):
                    merged.drop(
                        columns=[prefix + target], errors="ignore", inplace=True
                    )
                if has_test_preds:
                    t_values = np.asarray([f[col].astype(str).values for f in test_frames])
                    t_votes = np.asarray(
                        [np.sum((t_values == c) * weights[:, None], axis=0) for c in order]
                    )
                    merged_test[col] = np.asarray(order_raw or order)[t_votes.argmax(axis=0)]
                    for prefix in ("proba_", "classes_"):
                        merged_test.drop(
                            columns=[prefix + target], errors="ignore", inplace=True
                        )
            if has_test_preds:
                target_col = st.column or target
                if target_col not in merged_test and col in merged_test:
                    merged_test[target_col] = merged_test[col]
    elif mode in {"single_model", "checkpoint_selection"}:
        if has_test_preds:
            merged_test = test_frames[0].copy()
    truth = pd.read_csv(project.artifacts_dir / ids[0] / "evaluation_truth.csv")
    if set(truth[key]) != set(merged[key]):
        raise ValueError("Incomplete evaluation coverage")
    truth = merged[[key]].merge(truth, on=key, validate="one_to_one")
    source_config = yaml.safe_load(
        (project.artifacts_dir / ids[0] / "run_config.yaml").read_text(encoding="utf-8")
    ) or {}
    result = score_predictions(spec, plugin, context, source_config, truth, predictions(merged))
    if len(ids) > 1 and result["utility"] <= max(scores):
        raise ValueError("Superiority guard: ensemble must improve best constituent")
    candidate_id = "cand_" + uuid.uuid4().hex
    folder = project.artifacts_dir / candidate_id
    folder.mkdir(exist_ok=False)
    merged["input_variant"] = mode
    merged.to_csv(folder / "oof_predictions.csv", index=False)
    if has_test_preds and merged_test is not None:
        merged_test["input_variant"] = mode
        merged_test.to_csv(folder / "test_predictions.csv", index=False)
    entry = {
        "id": candidate_id,
        "name": name,
        "mode": mode,
        "verified": True,
        "eligible": True,
        "source_ids": ids,
        # Deterministic identity of the candidate *recipe*: which sources, in which
        # mode, with which weights/routing. This is what lets the bridge reject a
        # duplicate that an agent might re-submit under a new name to "preserve"
        # a submission or pad the candidate pool.
        "recipe_fingerprint": _candidate_recipe_fingerprint(mode, ids, source_map),
        "source_hashes": {
            exp: sha256(project.artifacts_dir / exp / "artifact_manifest.json")
            for exp in ids
        },
        "oof_path": safe_relpath(folder / "oof_predictions.csv", project.root),
        "rows": len(merged),
        "details": source_map,
        **result,
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    manifest_data = {
        **entry,
        "schema_version": 2,
        "prediction_sha256": sha256(folder / "oof_predictions.csv"),
        "metrics_sha256": sha256(folder / "metrics.json"),
    }
    if has_test_preds and (folder / "test_predictions.csv").is_file():
        test_sha = sha256(folder / "test_predictions.csv")
        manifest_data["test_prediction_sha256"] = test_sha
        manifest_data["prediction_splits"] = {"test_predictions.csv": "test"}
        entry["test_prediction_sha256"] = test_sha
        entry["prediction_splits"] = {"test_predictions.csv": "test"}
        entry["test_predictions_path"] = safe_relpath(folder / "test_predictions.csv", project.root)
    (folder / "candidate_manifest.json").write_text(
        json.dumps(manifest_data, indent=2)
    )
    return entry


def validate_candidate(project, entry):
    if not entry.get("eligible") or not entry.get("verified"):
        raise ValueError("Candidate is ineligible")
    folder = project.artifacts_dir / entry["id"]
    manifest = json.loads((folder / "candidate_manifest.json").read_text())
    if sha256(folder / "metrics.json") != manifest["metrics_sha256"]:
        raise ValueError("Candidate metric hash mismatch")
    if sha256(folder / "oof_predictions.csv") != manifest["prediction_sha256"]:
        raise ValueError("Candidate prediction hash mismatch")
    if "test_prediction_sha256" in manifest:
        if (
            not (folder / "test_predictions.csv").is_file()
            or sha256(folder / "test_predictions.csv") != manifest["test_prediction_sha256"]
        ):
            raise ValueError("Candidate test prediction hash mismatch")
    spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(
        project.root, project.task_config_path, spec, project.plugin_id or "default"
    )
    plugin = load_task_plugin(project.plugin_id, context)
    for exp, value in manifest["source_hashes"].items():
        validate_run(project.artifacts_dir / exp, task_spec=spec, plugin=plugin)
        if sha256(project.artifacts_dir / exp / "artifact_manifest.json") != value:
            raise ValueError("Candidate source changed")
    return manifest


def find_optimal_blend_weights(
    project,
    component_ids: list[str],
    blend_mode: str | None = None,
    maxiter: int = 150,
) -> dict[str, Any]:
    """Finds optimal blending weights for component models using SLSQP optimization."""
    from scipy.optimize import minimize

    from harness.project import HarnessProject

    if isinstance(project, (str, Path)):
        project = HarnessProject.discover(project)

    if not component_ids or len(component_ids) < 2:
        raise ValueError("Blend optimization requires at least 2 distinct component IDs")

    spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(
        project.root, project.task_config_path, spec, project.plugin_id or "default"
    )
    plugin = load_task_plugin(project.plugin_id, context)
    key = spec.id_column

    frames = []
    scores = []
    manifests = []
    source_configs = []
    for exp in component_ids:
        folder = project.artifacts_dir / exp
        manifests.append(validate_run(folder, task_spec=spec, plugin=plugin, expected={"exp_id": exp}))
        source_configs.append(yaml.safe_load((folder / "run_config.yaml").read_text(encoding="utf-8")) or {})
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(folder / "oof_predictions.csv")
        frame = frame[frame.input_variant == metrics["selection_variant"]].copy()
        frames.append(
            frame.set_index(key)
            .reindex(frames[0][key] if frames else frame[key])
            .reset_index()
        )
        scores.append(float(metrics.get("utility", metrics.get("primary_score", 0.0))))

    for manifest in manifests[1:]:
        for field in ("task_sha256", "scorer_sha256", "custom_scorer_sha256", "data_sha256", "split_sha256"):
            if manifest.get(field) != manifests[0].get(field):
                raise ValueError("Source contract/split mismatch: " + field)

    truth = pd.read_csv(project.artifacts_dir / component_ids[0] / "evaluation_truth.csv")
    truth = frames[0][[key]].merge(truth, on=key, validate="one_to_one")

    # Determine mode
    if blend_mode is None:
        has_proba = any(any(col.startswith("proba_") for col in f.columns) for f in frames)
        is_regression = all(st.task_type in ("regression", "ranking") for st in spec.subtasks.values())
        if is_regression:
            blend_mode = "weighted_blend"
        elif has_proba:
            blend_mode = "probability_blend"
        else:
            blend_mode = "ensemble_vote"

    n_models = len(component_ids)

    # Pre-extract prediction arrays for fast optimization
    subtask_data = {}
    for target, st in spec.subtasks.items():
        col = "pred_" + target
        if blend_mode == "probability_blend":
            orders = [
                json.loads(f["classes_" + target].iloc[0])
                if isinstance(f["classes_" + target].iloc[0], str)
                else f["classes_" + target].iloc[0]
                for f in frames
            ]
            class_order = orders[0]
            proba_arrays = [np.asarray(predictions(f)["proba_" + target], float) for f in frames]
            subtask_data[target] = {
                "type": "probability",
                "class_order": class_order,
                "proba_arrays": proba_arrays,
            }
        elif st.task_type in ("regression", "ranking") or blend_mode == "weighted_blend":
            pred_arrays = [f[col].astype(float).values for f in frames]
            subtask_data[target] = {
                "type": "numeric",
                "pred_arrays": pred_arrays,
            }
        else:
            order_raw = list(st.classes) if st.classes else []
            order = list(map(str, st.classes))
            values = [f[col].astype(str).values for f in frames]
            subtask_data[target] = {
                "type": "vote",
                "order_raw": order_raw,
                "order": order,
                "values": values,
            }

    def _eval_weights(w: np.ndarray) -> tuple[float, dict[str, Any]]:
        w = np.asarray(w, float)
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w).all():
            return -1e9, {}
        w = w / w_sum

        pred_dict = {}
        for target, st in spec.subtasks.items():
            s_info = subtask_data[target]
            col = "pred_" + target
            if s_info["type"] == "probability":
                blended_proba = sum(wi * pa for wi, pa in zip(w, s_info["proba_arrays"]))
                pred_dict["proba_" + target] = blended_proba.tolist()
                pred_dict["classes_" + target] = s_info["class_order"]
                if st.task_type == "multilabel_classification":
                    pred_dict[col] = (blended_proba > 0.5).astype(int).tolist()
                else:
                    pred_dict[col] = np.asarray(s_info["class_order"])[blended_proba.argmax(axis=1)].tolist()
            elif s_info["type"] == "numeric":
                blended_pred = sum(wi * pa for wi, pa in zip(w, s_info["pred_arrays"]))
                pred_dict[col] = blended_pred.tolist()
            else:
                vote_mat = np.asarray(
                    [sum(wi * (val == c) for wi, val in zip(w, s_info["values"])) for c in s_info["order"]]
                )
                pred_dict[col] = np.asarray(s_info["order_raw"] or s_info["order"])[vote_mat.argmax(axis=0)].tolist()

        res = score_predictions(spec, plugin, context, source_configs[0], truth, pred_dict)
        return float(res["utility"]), res

    def _objective(w: np.ndarray) -> float:
        score, _ = _eval_weights(w)
        return -score

    w0 = np.ones(n_models) / n_models
    eq_score, eq_res = _eval_weights(w0)

    best_w = w0
    best_score = eq_score
    best_res = eq_res

    try:
        bounds = [(0.0, 1.0)] * n_models
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        opt = minimize(
            _objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-5},
        )
        if opt.success or opt.fun < -best_score:
            cand_w = np.clip(opt.x, 0.0, 1.0)
            if cand_w.sum() > 0:
                cand_w /= cand_w.sum()
                cand_score, cand_res = _eval_weights(cand_w)
                if cand_score > best_score:
                    best_score = cand_score
                    best_w = cand_w
                    best_res = cand_res
    except Exception:
        pass

    constituent_scores = dict(zip(component_ids, scores))
    best_constituent_id = max(constituent_scores, key=constituent_scores.get)
    max_constituent_score = constituent_scores[best_constituent_id]
    gain = best_score - max_constituent_score

    weights_dict = {exp: round(float(wi), 4) for exp, wi in zip(component_ids, best_w)}

    return {
        "component_ids": component_ids,
        "blend_mode": blend_mode,
        "optimal_weights": [round(float(wi), 4) for wi in best_w],
        "weights_dict": weights_dict,
        "best_score": round(float(best_score), 5),
        "equal_weights_score": round(float(eq_score), 5),
        "constituent_scores": constituent_scores,
        "best_constituent": best_constituent_id,
        "gain_vs_best_constituent": round(float(gain), 5),
        "superiority_satisfied": gain > 0,
        "evaluation_details": best_res,
    }


def register_external_prediction(
    project,
    name: str,
    oof_path,
    test_path=None,
    author: str = "external",
    source: str = "kaggle_kernel",
    notes: str = "",
):
    """Ingest external Grandmaster OOF/test predictions with full audit and provenance verification.

    Generates a full, verified experiment artifact package in .ai-harness/artifacts/<name>
    evaluated against official evaluation truth and plugged into TreeJournal & Candidate manager.
    """
    import shutil
    import uuid

    import yaml

    from harness.artifacts import contract_identity, frame_hash
    from harness.project import HarnessProject
    from harness.tree_journal import Node, TreeJournal

    if isinstance(project, (str, Path)):
        project = HarnessProject.discover(project)

    oof_file = Path(oof_path)
    if not oof_file.is_absolute():
        oof_file = project.root / oof_file
    if not oof_file.exists():
        raise FileNotFoundError(f"External OOF file '{oof_file}' not found")

    test_file = None
    if test_path:
        test_file = Path(test_path)
        if not test_file.is_absolute():
            test_file = project.root / test_file
        if not test_file.exists():
            raise FileNotFoundError(f"External test file '{test_file}' not found")

    spec = load_active_task_spec(project.task_config_path)
    context = TaskPluginContext(
        project.root, project.task_config_path, spec, project.plugin_id or "default"
    )
    plugin = load_task_plugin(project.plugin_id, context)

    # 1. Load ground truth
    truth = plugin.load_training_data(context)
    if isinstance(truth, tuple):
        truth = truth[0]
    if truth is None:
        truth = spec.load_dataset_split("train", project_root=project.root)

    id_col = spec.id_column
    if id_col not in truth:
        raise ValueError(f"Task ground truth missing id_column '{id_col}'")

    # 2. Load OOF
    if oof_file.suffix.lower() == ".parquet":
        raw_oof = pd.read_parquet(oof_file)
    elif oof_file.suffix.lower() == ".npy":
        arr = np.load(oof_file)
        if len(arr) != len(truth):
            raise ValueError(f"NumPy OOF length {len(arr)} does not match ground truth {len(truth)}")
        primary_col = f"pred_{spec.primary_target}"
        raw_oof = pd.DataFrame({id_col: truth[id_col].values, primary_col: arr.ravel()})
    else:
        raw_oof = pd.read_csv(oof_file)

    if id_col not in raw_oof.columns:
        if len(raw_oof) == len(truth):
            raw_oof[id_col] = truth[id_col].values
        else:
            raise ValueError(f"External OOF predictions missing ID column '{id_col}' and row counts do not match")

    # Merge OOF with truth to align rows exactly. Evaluation must consume the
    # aligned frame — external files may be sorted differently from the
    # training split, and positional evaluation would silently mislabel rows.
    if truth[id_col].duplicated().any():
        raise ValueError(f"Ground truth has duplicate IDs in '{id_col}'")
    if raw_oof[id_col].duplicated().any():
        raise ValueError(f"External OOF file has duplicate IDs in '{id_col}'")
    aligned = truth[[id_col]].merge(raw_oof, on=id_col, how="inner")
    if len(aligned) != len(truth):
        raise ValueError(f"OOF predictions match {len(aligned)} rows out of {len(truth)} ground truth rows")

    # Normalize the managed artifact schema, independently of the external
    # exporter's optional metadata. Preserve genuine fold assignments; -1 is
    # explicitly unknown, not a claim that the exporter trained fold zero.
    aligned["input_variant"] = "external"
    if "fold" not in aligned:
        aligned["fold"] = -1
    for target in spec.subtasks:
        if f"pred_{target}" not in aligned and target in aligned:
            aligned[f"pred_{target}"] = aligned[target]
    preds_dict = predictions(aligned)

    # 3. Official evaluation
    evaluation = plugin.evaluate_predictions(context, truth, preds_dict)
    eval_resolved = resolve_evaluation(spec, evaluation, len(truth))

    # 4. Create artifact directory
    # Immutable evidence: refuse to overwrite an existing run directory, the
    # same contract the runner enforces (runner.py `exist_ok=False`). A second
    # registration with the same name is a distinct name, not a re-run.
    exp_dir = project.artifacts_dir / name
    if exp_dir.exists():
        raise ValueError(
            f"Refusing to overwrite existing run directory `{exp_dir}`. "
            "External predictions are immutable; use a different `name` for a new registration."
        )
    exp_dir.mkdir(parents=True, exist_ok=False)

    # 5. Write artifacts
    run_cfg = {
        "experiment": {
            "id": name,
            "approach": "external_import",
            "author": author,
            "source": source,
            "notes": notes,
        },
        "model": {
            "name": f"external_{author}_{name}",
            "family": "external_ensemble",
        },
    }
    (exp_dir / "run_config.yaml").write_text(yaml.safe_dump(run_cfg), encoding="utf-8")

    metrics_data = {
        "schema_version": 2,
        "experiment_id": name,
        "selection_variant": "external",
        "model": f"external_{author}",
        "model_family": "external",
        **eval_resolved,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    # predictions.csv mirrors the OOF frame (hardlink, 0 extra bytes); the
    # evaluation truth mirrors a content-addressed canonical master shared by
    # every run. Plain writes remain the fallback when hardlinks are unavailable.
    oof_path = exp_dir / "oof_predictions.csv"
    aligned.to_csv(oof_path, index=False)
    if not mirror_csv(oof_path, exp_dir / "predictions.csv"):
        aligned.to_csv(exp_dir / "predictions.csv", index=False)
    canonical_truth = ensure_canonical_truth(project.artifacts_dir, truth)
    if not mirror_csv(canonical_truth, exp_dir / "evaluation_truth.csv"):
        truth.to_csv(exp_dir / "evaluation_truth.csv", index=False)

    summary_df = pd.DataFrame([{
        "input_variant": "external",
        "dev_fold": "all",
        "primary_score": eval_resolved["primary_score"],
    }])
    summary_df.to_csv(exp_dir / "summary.csv", index=False)

    diagnostics = {
        "model_name": f"external_{name}",
        "critical_bottlenecks": [],
        "source": source,
        "author": author,
    }
    (exp_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    prediction_splits = {
        "oof_predictions.csv": "dev",
        "evaluation_truth.csv": "dev",
    }

    files = {
        "run_config.yaml": sha256(exp_dir / "run_config.yaml"),
        "metrics.json": sha256(exp_dir / "metrics.json"),
        "oof_predictions.csv": sha256(exp_dir / "oof_predictions.csv"),
        "evaluation_truth.csv": sha256(exp_dir / "evaluation_truth.csv"),
        "predictions.csv": sha256(exp_dir / "predictions.csv"),
        "summary.csv": sha256(exp_dir / "summary.csv"),
        "diagnostics.json": sha256(exp_dir / "diagnostics.json"),
    }

    if test_file:
        shutil.copy2(test_file, exp_dir / "test_predictions.csv")
        files["test_predictions.csv"] = sha256(exp_dir / "test_predictions.csv")
        prediction_splits["test_predictions.csv"] = "test"

    manifest = {
        "schema_version": 2,
        "run_id": f"ext_{uuid.uuid4().hex[:12]}",
        "exp_id": name,
        "status": "COMPLETED",
        "run_kind": "full",
        **contract_identity(spec, plugin),
        "data_sha256": frame_hash(truth),
        "split_sha256": digest("external_full"),
        "resolved_config_sha256": sha256(exp_dir / "run_config.yaml"),
        "files": files,
        "prediction_splits": prediction_splits,
    }
    (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 6. Record into TreeJournal (canonical `journal.json` — every other
    # consumer reads that path; a separate tree_journal.json was orphaned)
    journal = TreeJournal(project.artifacts_dir / "journal.json")
    node = Node(
        id=name,
        branch="external",
        model_family="external",
        metrics=eval_resolved,
        primary_metric_name=spec.primary_metric,
        direction=spec.direction,
        verified=True,
        eligible=True,
        status="COMPLETED",
        artifact_dir=str(exp_dir),
        atomic_hypothesis=f"External prediction from {author} ({source}): {notes}",
        intervention="external_import",
        diagnostics_summary=diagnostics,
    )
    journal.append(node)

    return {
        "name": name,
        "author": author,
        "source": source,
        "primary_score": eval_resolved["primary_score"],
        "metric": spec.primary_metric,
        "rows": len(aligned),
        "has_test_predictions": test_file is not None,
        "artifact_dir": safe_relpath(exp_dir, project.root),
    }
