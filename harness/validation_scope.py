"""Validation completeness derived from sealed partitions and observed row identities."""

from harness.artifacts import digest


def scope_record(partitions, selected_folds, observed_ids, completed_folds, *, sampled=False):
    required_folds = [str(p["fold"]) for p in partitions]
    required_ids = [value for p in partitions for value in p["validation_ids"]]
    if len(required_folds) != len(set(required_folds)) or len(required_ids) != len(set(required_ids)):
        raise ValueError("Duplicate validation folds or IDs in complete protocol")
    selected = sorted(map(str, selected_folds))
    completed = sorted(map(str, completed_folds))
    observed = list(observed_ids)
    reasons = []
    if set(selected) != set(required_folds):
        reasons.append("PARTIAL_FOLDS")
    if set(completed) != set(required_folds):
        reasons.append("INCOMPLETE_FOLDS")
    if len(observed) != len(set(observed)) or set(observed) != set(required_ids):
        reasons.append("INCOMPLETE_IDS")
    if sampled:
        reasons.append("SAMPLED")
    return {"version": 1, "protocol_sha256": digest(partitions),
            "required_folds": sorted(required_folds), "selected_folds": selected,
            "completed_folds": completed, "required_ids_sha256": digest(sorted(required_ids, key=repr)),
            "observed_ids_sha256": digest(sorted(observed, key=repr)),
            "required_rows": len(required_ids), "observed_rows": len(observed),
            "sampled": sampled, "complete": not reasons, "reason_codes": reasons}


def validate_scope(directory, manifest, *, require_full, task_spec=None):
    import json
    from harness.evaluation import read_prediction_frame

    path = directory / "validation_partitions.json"
    if "validation_partitions.json" not in manifest.get("files", {}) or not path.is_file():
        raise ValueError("VALIDATION_SCOPE_UNVERIFIED: missing sealed partitions")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = data.get("required_partitions")
    if not isinstance(required, list) or not required:
        raise ValueError("VALIDATION_SCOPE_UNVERIFIED: missing complete protocol")
    if task_spec is not None:
        declared = task_spec.validation.complete_folds
        if declared is not None and set(map(str, declared)) != {str(p["fold"]) for p in required}:
            raise ValueError("VALIDATION_SCOPE_UNVERIFIED: task protocol mismatch")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    required_ids = [value for part in required for value in part["validation_ids"]]
    frame = read_prediction_frame(directory / "oof_predictions.csv", expected_ids=required_ids, id_column=data["id_column"])
    if "input_variant" in frame:
        frame = frame[frame.input_variant == metrics["selection_variant"]]
    key = data["id_column"]
    if "fold" not in frame:
        raise ValueError("VALIDATION_SCOPE_UNVERIFIED: missing observed folds")
    selected = data.get("partitions")
    if not isinstance(selected, list) or not selected:
        raise ValueError("VALIDATION_SCOPE_UNVERIFIED: missing selected partitions")
    by_fold = {str(part["fold"]): part for part in required}
    if len({str(part["fold"]) for part in selected}) != len(selected):
        raise ValueError("VALIDATION_SCOPE_INCOMPLETE: duplicate selected folds")
    for part in selected:
        fold = str(part["fold"])
        if fold not in by_fold:
            raise ValueError("VALIDATION_SCOPE_INCOMPLETE: unknown selected fold")
        ids = part["validation_ids"]
        declared_ids = by_fold[fold]["validation_ids"]
        if (len(ids) != len(set(ids)) or not set(ids).issubset(declared_ids)
                or (not data.get("sample_limit") and set(ids) != set(declared_ids))):
            raise ValueError("VALIDATION_SCOPE_INCOMPLETE: changed partition coverage")
        observed = frame.loc[frame["fold"].map(str) == fold, key].tolist()
        if len(observed) != len(ids) or set(observed) != set(ids):
            raise ValueError("VALIDATION_SCOPE_INCOMPLETE: observed fold coverage differs")
    record = scope_record(required, [p["fold"] for p in data["partitions"]],
                          frame[key].tolist(), frame["fold"].unique().tolist(), sampled=bool(data.get("sample_limit")))
    if record != manifest.get("validation_scope"):
        raise ValueError("VALIDATION_SCOPE_UNVERIFIED: scope differs from sealed evidence")
    if require_full and not record["complete"]:
        raise ValueError("VALIDATION_SCOPE_INCOMPLETE: " + ", ".join(record["reason_codes"]))
    return record
