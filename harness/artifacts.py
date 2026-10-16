"""Immutable run evidence shared by execution and downstream consumers."""

from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path

REQUIRED = (
    "run_config.yaml",
    "metrics.json",
    "oof_predictions.csv",
    "evaluation_truth.csv",
)


_HASH_CACHE: dict = {}


def sha256(path, chunk_size: int = 1048576) -> str:
    """Compute SHA256 using chunked reading and mtime/size caching to avoid I/O bottlenecks."""
    p = Path(path).resolve()
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key in _HASH_CACHE:
            return _HASH_CACHE[key]
    except OSError:
        key = None

    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    digest_val = h.hexdigest()
    if key is not None:
        _HASH_CACHE[key] = digest_val
    return digest_val


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def contract_identity(task_spec, plugin=None):
    from harness.plugins import TaskPlugin

    scorer = plugin.evaluate_predictions if plugin else TaskPlugin.evaluate_predictions
    if plugin and type(plugin).evaluate_predictions is TaskPlugin.evaluate_predictions:
        scorer = TaskPlugin.evaluate_predictions
    try:
        implementation = inspect.getsource(scorer)
        if scorer is TaskPlugin.evaluate_predictions:
            from harness import evaluation
            from harness import task_spec as task_module

            implementation += inspect.getsource(evaluation) + inspect.getsource(
                task_module
            )
    except (OSError, TypeError):
        raise ValueError("Scorer implementation cannot be identified")
    return {
        "task_sha256": digest(task_spec.metadata),
        "scorer_sha256": digest(implementation),
    }


def validate_run(directory, *, task_spec=None, plugin=None, expected=None):
    directory = Path(directory)
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("LEGACY_UNVERIFIED: missing manifest v2")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "COMPLETED"
        or manifest.get("run_kind") != "full"
    ):
        raise ValueError("Run is not completed, verified full validation")
    for field in (
        "run_id",
        "exp_id",
        "task_sha256",
        "scorer_sha256",
        "data_sha256",
        "split_sha256",
        "resolved_config_sha256",
    ):
        if not manifest.get(field):
            raise ValueError(f"Missing run identity: {field}")
    checks = dict(expected or {})
    if task_spec is not None:
        checks.update(contract_identity(task_spec, plugin))
    for field, value in checks.items():
        if manifest.get(field) != value:
            raise ValueError(f"Run identity mismatch: {field}")
    files = manifest.get("files", {})
    for name in REQUIRED:
        if name not in files:
            raise ValueError(f"Missing required artifact: {name}")
    for name, value in files.items():
        if (
            Path(name).name != name
            or not (directory / name).is_file()
            or sha256(directory / name) != value
        ):
            raise ValueError(f"Artifact hash mismatch: {name}")
    import yaml, importlib

    cfg = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
    hook = cfg.get("paths", {}).get("metrics_evaluator")
    if hook:
        module, name = hook.rsplit(".", 1)
        if manifest.get("custom_scorer_sha256") != digest(
            inspect.getsource(getattr(importlib.import_module(module), name))
        ):
            raise ValueError("Custom scorer implementation changed")
    if sha256(directory / "run_config.yaml") != manifest["resolved_config_sha256"]:
        raise ValueError("Resolved configuration identity mismatch")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    from harness.evaluation import utility

    utility(metrics["primary_score"], metrics["direction"])
    if metrics.get("experiment_id") != manifest["exp_id"]:
        raise ValueError("Metrics experiment identity mismatch")
    return manifest


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# CSV deduplication (hardlink mirrors + content-addressed canonical truth)
#
# Full runs write ``oof_predictions.csv`` and ``predictions.csv`` as byte-identical
# copies of the same frame, and every run re-writes the same evaluation truth.
# On disk those copies waste space (tens of MB x 3 per experiment on large
# tabular tasks). The helpers below store them once:
#
# - ``predictions.csv`` becomes a hardlink mirror of ``oof_predictions.csv``.
# - ``evaluation_truth.csv`` becomes a hardlink mirror of a content-addressed
#   canonical master (``canonical_evaluation_truth_<sha16>.csv``) shared by all
#   runs. Content addressing matters: the runner writes the *split* truth while
#   ``register_external_prediction`` writes the *full* training truth, so a fixed
#   filename would silently link stale content after a split change.
#
# Every consumer (validate_run, repair, research_memory, cloud import, notebook
# export) is content-based, so hardlinks are transparent. All fallbacks avoid
# writing in place over an existing (potentially shared-inode) file.
# ---------------------------------------------------------------------------


def mirror_csv(source: Path | str, target: Path | str) -> bool:
    """Make ``target`` a hardlink to ``source`` (zero extra bytes).

    An existing ``target`` is removed first so re-runs refresh the mirror.
    Returns ``True`` when the mirror was (re)created; ``False`` when hardlinks
    are unavailable (cross-device, filesystem without support) and the caller
    should fall back to a regular write.
    """
    import os

    source, target = Path(source), Path(target)
    try:
        if target.exists() or target.is_symlink():
            target.unlink()
        os.link(source, target)
        return True
    except OSError:
        return False


def write_csv_dedup(frame, target: Path | str, mirror_from: Path | str | None = None):
    """Write ``frame`` to ``target``, preferring a hardlink mirror when possible.

    The fallback never writes in place over an existing file: a stale file may
    be a hardlink to another artifact, and an in-place ``to_csv`` would
    truncate the shared inode. The stale file is unlinked first.
    """
    if mirror_from is not None and mirror_csv(mirror_from, target):
        return
    target = Path(target)
    try:
        if target.exists() or target.is_symlink():
            target.unlink()
    except OSError:
        pass
    frame.to_csv(target, index=False)


def canonical_truth_path(artifacts_dir: Path | str, frame) -> Path:
    """Content-addressed canonical location for an evaluation truth frame."""
    return Path(artifacts_dir) / f"canonical_evaluation_truth_{frame_hash(frame)[:16]}.csv"


def ensure_canonical_truth(artifacts_dir: Path | str, frame) -> Path:
    """Create the canonical truth master if missing; return its path.

    Creation is atomic (temp file + ``os.replace``) so concurrent runs never
    expose a half-written master. The content-addressed name makes concurrent
    writers idempotent: identical content maps to the same path.
    """
    import os

    path = canonical_truth_path(artifacts_dir, frame)
    if path.exists():
        return path
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def validate_test_prediction(path, *, task_spec, plugin=None):
    path = Path(path).resolve()
    cand_manifest_path = path.parent / "candidate_manifest.json"
    if cand_manifest_path.is_file() and not (path.parent / "artifact_manifest.json").is_file():
        cand_manifest = json.loads(cand_manifest_path.read_text(encoding="utf-8"))
        if (
            path.name != "test_predictions.csv"
            or cand_manifest.get("prediction_splits", {}).get(path.name) != "test"
            or sha256(path) != cand_manifest.get("test_prediction_sha256")
        ):
            raise ValueError(
                "Submission requires proven test predictions; OOF is validation evidence"
            )
        return cand_manifest
    manifest = validate_run(path.parent, task_spec=task_spec, plugin=plugin)
    if (
        path.name != "test_predictions.csv"
        or path.name not in manifest["files"]
        or manifest.get("prediction_splits", {}).get(path.name) != "test"
    ):
        raise ValueError(
            "Submission requires proven test predictions; OOF is validation evidence"
        )
    return manifest
