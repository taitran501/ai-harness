"""Importable prediction auditing; no dependency on checkout-relative tools."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from harness.artifacts import digest
from harness.policy_schema import validate_document


def _typed(value):
    if isinstance(value, np.generic):
        value = value.item()
    return (type(value).__name__, value)


def align_prediction_frames(candidate, reference, identity_columns):
    """Exact symmetric composite keys, including scope and typed IDs."""
    keys = list(identity_columns)
    for scope in ("fold", "input_variant", "split"):
        if (scope in candidate) != (scope in reference):
            raise ValueError("IDENTITY_MISMATCH: asymmetric scope")
        if scope in candidate and scope not in keys:
            keys.append(scope)
    rows = []
    for frame in (candidate, reference):
        if not keys or not all(key in frame for key in keys) or frame.empty:
            raise ValueError("IDENTITY_MISMATCH: missing keys/rows")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("IDENTITY_MISMATCH: null/duplicate keys")
        rows.append([tuple(_typed(v) for v in row) for row in frame[keys].itertuples(index=False, name=None)])
    if set(rows[0]) != set(rows[1]):
        raise ValueError("IDENTITY_MISMATCH: unequal coverage")
    positions = {key: i for i, key in enumerate(rows[1])}
    return candidate.reset_index(drop=True), reference.iloc[[positions[key] for key in rows[0]]].reset_index(drop=True)


def prediction_contract(spec, frame):
    """Resolve output semantics from declarations, never numeric dtype guesses."""
    if isinstance(spec, dict):
        keys = spec.get("identity_columns") or [spec.get("id_column", "id")]
        columns = spec.get("prediction_columns", {})
        if isinstance(columns, list):
            raise ValueError("SCHEMA_MISMATCH: output types must be declared")
        discrete_floor = spec.get("hard_label_min_disagreement", .02)
    else:
        keys, columns = [spec.id_column], {}
        for key, sub in spec.subtasks.items():
            is_continuous = sub.task_type in {"regression", "forecasting"}
            for name in (f"pred_{key}", f"proba_{key}"):
                if name in frame:
                    columns[name] = "continuous" if name.startswith("proba_") or is_continuous else (
                        "text" if sub.task_type in {"generative", "generative_nlp", "text_generation"} else "discrete")
            if f"pred_{key}" not in columns:
                raise ValueError(f"SCHEMA_MISMATCH: missing pred_{key}")
            if sub.task_type in {"ranking", "retrieval"}:
                columns[f"pred_{key}"] = "unsupported"
        metadata = getattr(spec, "submission_spec", None) or spec.metadata.get("submission_spec", {})
        discrete_floor = float(metadata.get("min_diff_pct", 2)) / 100
    if not columns or any(c not in frame for c in columns):
        raise ValueError("SCHEMA_MISMATCH: missing declared outputs")
    for scope in ("fold", "input_variant", "split"):
        if scope in frame and scope not in keys:
            keys.append(scope)
    if not 0 <= discrete_floor <= 1:
        raise ValueError("Invalid hard label disagreement floor")
    return keys, columns, discrete_floor


def _values(series, *, parse_json=True):
    result = []
    for value in series:
        if parse_json and isinstance(value, str) and value.lstrip().startswith(("[", "{")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        result.append(value)
    return result


def semantic_prediction_hash(frame, spec):
    keys, outputs, _ = prediction_contract(spec, frame)
    align_prediction_frames(frame, frame, keys)
    cols = sorted(set(keys) | set(outputs) | {c for c in frame if c.startswith("classes_")})
    def canonical(value, column):
        if column in keys:
            return _typed(value)
        parsed = _values([value], parse_json=outputs.get(column) != "text")[0]
        if isinstance(parsed, np.generic):
            parsed = parsed.item()
        if type(parsed) in (float, int):
            if not np.isfinite(parsed):
                raise ValueError("NONFINITE_PREDICTION")
            return float(parsed)
        if isinstance(parsed, list):
            return [canonical(v, column) for v in parsed]
        if parsed is None:
            raise ValueError("NONFINITE_PREDICTION")
        return parsed
    records = [[canonical(v, col) for col, v in zip(cols, row)] for row in frame[cols].itertuples(index=False, name=None)]
    return digest({"columns": cols, "rows": sorted(records, key=lambda r: json.dumps(r, default=str, sort_keys=True))})


def check_prediction_diversity(candidate, references, *, spec, threshold=.995):
    """Compare every aligned declared output to every supplied portfolio entry.

    References are frames or mappings with id/frame/source. Missing required
    history is an error; callers must never filter absent snapshots away.
    """
    from datetime import datetime, timezone
    from scipy.stats import spearmanr

    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid max_spearman")
    candidate = pd.read_csv(candidate) if isinstance(candidate, (Path, str)) else candidate
    keys, outputs, floor = prediction_contract(spec, candidate)
    align_prediction_frames(candidate, candidate, keys)
    cand_hash = semantic_prediction_hash(candidate, spec)
    for col, kind in outputs.items():
        if candidate[col].isna().any():
            raise ValueError("NONFINITE_PREDICTION")
        if kind == "continuous":
            values = np.asarray(_values(candidate[col]), dtype=float)
            if not np.isfinite(values).all() or values.ndim not in {1, 2}:
                raise ValueError("NONFINITE_PREDICTION")
    reasons, comparisons, similarities, methods = [], [], [], set()
    refs = list(references)
    for index, reference in enumerate(refs):
        info = reference if isinstance(reference, dict) else {"frame": reference}
        frame = info.get("frame")
        if frame is None:
            path = info.get("path")
            if not path or not Path(path).is_file():
                raise ValueError("HISTORY_INCOMPLETE: missing prediction snapshot")
            frame = pd.read_csv(path)
        c, r = align_prediction_frames(candidate, frame, keys)
        rkeys, routputs, _ = prediction_contract(spec, r)
        if outputs != routputs or keys != rkeys:
            raise ValueError("SCHEMA_MISMATCH")
        for col in {col for f in (c, r) for col in f if col.startswith("classes_")}:
            if col not in c or col not in r or _values(c[col]) != _values(r[col]):
                raise ValueError("SCHEMA_MISMATCH: class order")
        column_results = []
        ref_id = str(info.get("id", index))
        for col, kind in outputs.items():
            left, right = _values(c[col], parse_json=kind != "text"), _values(r[col], parse_json=kind != "text")
            method = "spearman" if kind == "continuous" else "exact_disagreement"
            methods.add(method)
            if any(v is None for v in left + right):
                raise ValueError("NONFINITE_PREDICTION")
            if kind == "unsupported":
                column_results.append(dict(name=col, method=method, value=None, status="not_applicable"))
                reasons.append("UNSUPPORTED_REPRESENTATION")
                continue
            if kind == "continuous":
                a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("NONFINITE_PREDICTION")
                if a.ndim == 1:
                    a, b = a[:, None], b[:, None]
                if a.ndim != 2:
                    raise ValueError("UNSUPPORTED_REPRESENTATION")
                for j in range(a.shape[1]):
                    value, status = None, "inconclusive"
                    if np.array_equal(a[:, j], b[:, j]):
                        value, status = 1.0, "near_duplicate"
                        reasons.append("SEMANTIC_DUPLICATE")
                    elif len(a) < 3:
                        reasons.append("INSUFFICIENT_ROWS")
                    elif np.ptp(a[:, j]) == 0 or np.ptp(b[:, j]) == 0:
                        reasons.append("UNDEFINED_CORRELATION")
                    else:
                        value = float(spearmanr(a[:, j], b[:, j]).statistic)
                        status = "near_duplicate" if value >= threshold else "diverse"
                        if status == "near_duplicate":
                            reasons.append("RANK_SIMILARITY_EXCEEDS_LIMIT")
                    if value is not None:
                        similarities.append((value, ref_id))
                    column_results.append(dict(name=col if a.shape[1] == 1 else f"{col}[{j}]", method=method, value=value, status=status))
            else:
                different = [json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True) for x, y in zip(left, right)]
                value = float(np.mean(different))
                status = "near_duplicate" if value == 0 or value < floor else "diverse"
                if status == "near_duplicate":
                    reasons.append("SEMANTIC_DUPLICATE" if value == 0 else "DISAGREEMENT_BELOW_LIMIT")
                column_results.append(dict(name=col, method=method, value=value, status=status))
        statuses = [col["status"] for col in column_results]
        status = ("near_duplicate" if all(s == "near_duplicate" for s in statuses) else
                  "inconclusive" if any(s in {"inconclusive", "not_applicable"} for s in statuses) else "diverse")
        comparisons.append(dict(reference_id=ref_id, reference_prediction_sha256=semantic_prediction_hash(r, spec),
                                source=info.get("source", "dispatched"), rows=len(r), columns=column_results, status=status))
    statuses = [r["status"] for r in comparisons]
    status = "near_duplicate" if "near_duplicate" in statuses else "inconclusive" if "inconclusive" in statuses else "diverse"
    if all(kind == "unsupported" for kind in outputs.values()):
        status = "not_applicable"
        reasons.append("UNSUPPORTED_REPRESENTATION")
    if not refs:
        reasons.append("INITIAL_PORTFOLIO")
    value, nearest = max(similarities) if similarities else (None, None)
    receipt = dict(schema_version=1, check="prediction_diversity", status=status,
                   metric=next(iter(methods)) if len(methods) == 1 else "mixed" if methods else "none",
                   max_spearman=threshold, hard_label_min_disagreement=floor if set(outputs.values()) & {"discrete", "text"} else None,
                   candidate_prediction_sha256=cand_hash, comparison_identity_sha256=digest({"keys": keys, "outputs": outputs}),
                   history_snapshot_sha256=digest(comparisons), identity_columns=keys, prediction_columns=list(outputs),
                   rows=len(candidate), reference_count=len(refs), comparisons=comparisons,
                   nearest_reference_id=nearest, maximum_similarity=value, reason_codes=list(dict.fromkeys(reasons)),
                   computed_at=datetime.now(timezone.utc).isoformat())
    validate_document("prediction_diversity_receipt", receipt)
    return receipt


def evaluate_submission_candidate(candidate, references, *, spec, threshold=.995):
    """Shared structural and portfolio check used during prepare and dispatch."""
    frame = pd.read_csv(candidate) if isinstance(candidate, (str, Path)) else candidate
    errors = audit_prediction_integrity(frame, id_col=getattr(spec, "id_column", None), spec=spec)
    if errors:
        raise ValueError("PREDICTION_INTEGRITY: " + "; ".join(errors))
    return check_prediction_diversity(frame, references, spec=spec, threshold=threshold)

def compute_sha256(filepath: Path | str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Retrieve current git commit hash if in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def audit_prediction_integrity(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    spec: Optional[Any] = None,
) -> List[str]:
    """Verify structural validity and constraints of the predictions DataFrame."""
    if spec is None:
        from harness.task_spec import load_active_task_spec
        try:
            spec = load_active_task_spec()
        except Exception:
            spec = None
    key = id_col or (spec.id_column if spec else "id")
    errors = []
    if not any(c.startswith(("pred_", "proba_")) for c in df) and (not spec or not spec.subtasks):
        errors.append("Missing predictions")
    if key not in df:
        errors.append("Missing declared ID column")
    else:
        if df[key].isna().any():
            errors.append(f"IDs {key} must be non-null")
        else:
            # OOF / multi-variant tables legitimately repeat the same id
            # across folds and input variants; uniqueness is then defined
            # over the full row identity, not over id alone (a plain
            # df[key].duplicated() rejects every multi-variant OOF run).
            scope = [c for c in ("fold", "input_variant", "split") if c in df.columns]
            if scope and df.duplicated(subset=[key] + scope).any():
                errors.append(f"IDs {key} must be unique within {'/'.join(scope)}")
            elif not scope and df[key].duplicated().any():
                errors.append(f"IDs {key} must be non-null and unique")
    if df.empty:
        errors.append("Prediction coverage is empty")
    if spec and spec.subtasks:
        for target, st in spec.subtasks.items():
            col = next((c for c in ("pred_" + target, "pred_" + (st.column or target), st.column or target) if c in df), None)
            if col is None:
                errors.append("Missing prediction: " + target)
                continue
            if df[col].isna().any():
                errors.append("Null prediction: " + target)
            if st.task_type == "regression":
                values = pd.to_numeric(df[col], errors="coerce")
                if not np.isfinite(values).all():
                    errors.append("Non-finite prediction: " + target)
            elif st.task_type == "multilabel_classification":
                try:
                    values = np.asarray([json.loads(x) if isinstance(x, str) else x for x in df[col]])
                    if values.shape != (len(df), len(st.labels or st.classes)) or not np.isin(values, [0, 1]).all():
                        errors.append("Invalid multilabel shape/domain")
                except (ValueError, TypeError):
                    errors.append("Invalid multilabel encoding")
            elif st.task_type in ("ranking", "recommendation"):
                try:
                    values = pd.to_numeric(df[col], errors="raise")
                    if not np.isfinite(values).all():
                        errors.append("Non-finite prediction: " + target)
                except (ValueError, TypeError):
                    pass
            elif st.task_type in ("generative_nlp", "generation", "text_generation", "seq2seq", "nlp", "generative"):
                pass
            elif getattr(st, "classes", None) and len(st.classes) > 0:
                if not df[col].astype(str).isin(list(map(str, st.classes))).all():
                    errors.append("invalid values outside declared classes: " + target)
    return errors


def compute_diff_metrics(
    cand_df: pd.DataFrame,
    base_df: pd.DataFrame,
    id_col: Optional[str] = None,
    rel_epsilon: float = 0.01,
) -> Tuple[int, Dict[str, Any]]:
    """Compute differential row counts and metrics between candidate and baseline predictions."""
    key = id_col
    if not key:
        for candidate_key in ["id", "ID", "sample_id", "row_id", "uid"]:
            if candidate_key in cand_df.columns and candidate_key in base_df.columns:
                key = candidate_key
                break
    if not key:
        key = "id"

    for frame in (cand_df, base_df):
        if key not in frame or frame[key].isna().any():
            raise ValueError("Invalid comparison IDs")
    if set(cand_df[key]) != set(base_df[key]):
        raise ValueError("Prediction ID sets differ")

    # OOF / multi-variant tables repeat ids across folds and input variants;
    # then row identity is the composite key, and both frames must carry the
    # same scope columns to be comparable.
    scope = [
        c
        for c in ("fold", "input_variant", "split")
        if c in cand_df.columns and c in base_df.columns
    ]
    if scope:
        identity = [key] + scope
        for frame in (cand_df, base_df):
            if frame[identity].isna().any().any() or frame.duplicated(subset=identity).any():
                raise ValueError("Invalid comparison IDs")
        cand = cand_df.set_index(identity)
        base = base_df.set_index(identity).reindex(cand.index)
    else:
        if cand_df[key].duplicated().any() or base_df[key].duplicated().any():
            raise ValueError("Invalid comparison IDs")
        cand = cand_df.set_index(key)
        base = base_df.set_index(key).reindex(cand.index)
    cols = [c for c in cand if c.startswith(("pred_", "proba_"))]
    if not cols:
        raise ValueError("No prediction columns to compare")
    if set(cols) != {c for c in base if c.startswith(("pred_", "proba_"))}:
        raise ValueError("Prediction schemas differ")

    details = {}
    changed = pd.Series(False, index=cand.index)
    for col in cols:
        cand_s = cand[col]
        base_s = base[col]
        is_continuous = False
        try:
            cand_num = pd.to_numeric(cand_s, errors="raise")
            base_num = pd.to_numeric(base_s, errors="raise")
            if pd.api.types.is_float_dtype(cand_num) or pd.api.types.is_float_dtype(base_num):
                is_continuous = True
        except (ValueError, TypeError):
            is_continuous = False

        if is_continuous:
            cand_vals = cand_num.to_numpy(dtype=float)
            base_vals = base_num.to_numpy(dtype=float)
            std = float(np.std(base_vals))
            mean_abs = float(np.mean(np.abs(base_vals))) if len(base_vals) > 0 else 1.0
            S = std if (not np.isnan(std) and std > 1e-8) else (mean_abs if (not np.isnan(mean_abs) and mean_abs > 1e-8) else 1.0)
            if np.isnan(S) or S <= 1e-8:
                S = 1.0
            diff = pd.Series((np.abs(cand_vals - base_vals) / S) >= rel_epsilon, index=cand.index)
        else:
            diff = cand_s != base_s

        changed |= diff
        details[col] = {
            "different_count": int(diff.sum()),
            "pct_changed": float(diff.mean() * 100),
            "rel_epsilon": float(rel_epsilon) if is_continuous else None,
            "is_continuous": is_continuous,
        }
    return int(changed.sum()), details


def compute_continuous_significance(
    y_true: pd.Series | np.ndarray,
    cand_pred: pd.Series | np.ndarray,
    base_pred: pd.Series | np.ndarray,
) -> Dict[str, Any]:
    """Compute paired error test (paired t-test on squared errors) between candidate and baseline."""
    from scipy.stats import ttest_rel
    y = np.asarray(y_true, dtype=float)
    cand = np.asarray(cand_pred, dtype=float)
    base = np.asarray(base_pred, dtype=float)

    cand_errors = (cand - y) ** 2
    base_errors = (base - y) ** 2

    b = int((cand_errors < base_errors).sum())
    c = int((base_errors < cand_errors).sum())

    if len(cand_errors) < 2 or (b == 0 and c == 0) or np.allclose(cand_errors, base_errors, atol=0.0, rtol=1e-7):
        return {
            "cand_better_count": b,
            "base_better_count": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "test_name": "paired_ttest",
        }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = ttest_rel(cand_errors, base_errors)
    stat = float(res.statistic) if np.isfinite(res.statistic) else 0.0
    p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0

    return {
        "cand_better_count": b,
        "base_better_count": c,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "is_significant": bool(p_val < 0.05),
        "test_name": "paired_ttest",
    }


def compute_mcnemar_significance(
    y_true: pd.Series | np.ndarray,
    cand_pred: pd.Series | np.ndarray,
    base_pred: pd.Series | np.ndarray,
) -> Dict[str, Any]:
    """Compute McNemar's test for paired classification accuracy between candidate and baseline."""
    from scipy.stats import chi2
    y = pd.Series(y_true)
    cand = pd.Series(cand_pred)
    base = pd.Series(base_pred)

    cand_correct = cand == y
    base_correct = base == y

    # b: candidate correct, baseline wrong (candidate won)
    b = int((cand_correct & (~base_correct)).sum())
    # c: candidate wrong, baseline correct (baseline won)
    c = int(((~cand_correct) & base_correct).sum())

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "cand_better_count": b,
            "base_better_count": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "test_name": "mcnemar",
        }

    # Edwards continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "cand_better_count": b,
        "base_better_count": c,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "is_significant": bool(p_val < 0.05),
        "test_name": "mcnemar",
    }


