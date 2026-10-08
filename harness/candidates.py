"""Candidate construction from exact, verified and aligned evidence."""

import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from harness.artifacts import validate_run, sha256, digest
from harness.evaluation import resolve_evaluation
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.task_spec import load_active_task_spec
from harness.ensemble_guard import EnsembleGuard


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
        for field in ("task_sha256", "scorer_sha256", "split_sha256", "data_sha256"):
            if manifest[field] != manifests[0][field]:
                raise ValueError("Source contract/split mismatch: " + field)
    merged = frames[0].copy()
    if mode == "specialist_router":
        merged = merged[[key, "fold"]].copy()
        for target, source in routing.items():
            target = target.removeprefix("pred_")
            frame = frames[ids.index(source)]
            for prefix in ("pred_", "proba_", "classes_"):
                col = prefix + target
                if prefix == "pred_" and col not in frame:
                    raise ValueError("Missing specialist target: " + col)
                if col in frame:
                    merged[col] = frame[col].values
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
                passed, errors, _ = EnsembleGuard.audit_constituent_diversity(
                    checked, col=col
                )
                if not passed:
                    raise ValueError("Diversity guard: " + "; ".join(errors))
            if st.task_type == "regression":
                merged[col] = sum(
                    w * f[col].astype(float).values for w, f in zip(weights, frames)
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
            elif mode == "probability_blend":
                orders = [json.loads(f["classes_" + target].iloc[0]) for f in frames]
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
            else:
                order = list(map(str, st.classes))
                if not order:
                    raise ValueError("Voting requires declared class order")
                values = np.asarray([f[col].astype(str).values for f in frames])
                votes = np.asarray(
                    [np.sum((values == c) * weights[:, None], axis=0) for c in order]
                )
                merged[col] = np.asarray(order)[votes.argmax(axis=0)]
                for prefix in ("proba_", "classes_"):
                    merged.drop(
                        columns=[prefix + target], errors="ignore", inplace=True
                    )
    truth = pd.read_csv(project.artifacts_dir / ids[0] / "evaluation_truth.csv")
    if set(truth[key]) != set(merged[key]):
        raise ValueError("Incomplete evaluation coverage")
    truth = merged[[key]].merge(truth, on=key, validate="one_to_one")
    result = resolve_evaluation(
        spec,
        plugin.evaluate_predictions(context, truth, predictions(merged)),
        len(merged),
    )
    if len(ids) > 1 and result["utility"] <= max(scores):
        raise ValueError("Superiority guard: ensemble must improve best constituent")
    candidate_id = "cand_" + uuid.uuid4().hex
    folder = project.artifacts_dir / candidate_id
    folder.mkdir(exist_ok=False)
    merged["input_variant"] = mode
    merged.to_csv(folder / "oof_predictions.csv", index=False)
    entry = {
        "id": candidate_id,
        "name": name,
        "mode": mode,
        "verified": True,
        "eligible": True,
        "source_ids": ids,
        "source_hashes": {
            exp: sha256(project.artifacts_dir / exp / "artifact_manifest.json")
            for exp in ids
        },
        "oof_path": str((folder / "oof_predictions.csv").relative_to(project.root)),
        "rows": len(merged),
        "details": source_map,
        **result,
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    (folder / "candidate_manifest.json").write_text(
        json.dumps(
            {
                **entry,
                "schema_version": 2,
                "prediction_sha256": sha256(folder / "oof_predictions.csv"),
                "metrics_sha256": sha256(folder / "metrics.json"),
            },
            indent=2,
        )
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
