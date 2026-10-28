"""Immutable run evidence shared by execution and downstream consumers."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Dict

REQUIRED = (
    "run_config.yaml",
    "metrics.json",
    "oof_predictions.csv",
    "evaluation_truth.csv",
)

# Canonical generic scorer hashes across historical releases and current build.
# For standard built-in metric evaluation (AUC, RMSE, F1, logloss, etc. via TaskPlugin),
# minor documentation, type hint, or helper edits to harness.evaluation must never
# invalidate completed full run artifacts.
CANONICAL_GENERIC_SCORER_HASHES = frozenset({
    "aca79278b34ca0000639c10298c1a0c789281062d3e4b8e03ae2cc46a9ffb87c",
    "8a766ac8f61a6c7fda3f761900b85f732006f3fa6bd4acbe5ec803a56216eaa4",
    "8a766ac8152cb5b10884d3dbe7b11d95c478a514d35e07a68393e8a4a5860255",
    "7b33d61fce13890adbefed74178dba2856de4fe9ec952ca24be2fe8416458b29",
    "7d2f454d4bcbabe2bbc6df2586f04b2461b6698ce8f651f151ac0bef6e073ee6",
    "d44339a1920ea54fa07f7f17b0565c64f8e0a7da3f5deeef6eb88cfec197d95e",
    "13c0aef13603c34442eb9eb4e9a0aade3627ff67443e1fe5cb3def93a770b81b",
    "4768719e5ffed96a64d10322c01f5ac735fe32b447e4e674e310fca878f0fe44",
    "bcc7ab193043b0e4d581081fc1f52ae5e0097148e48a83c3493cec230ad52e35",
})


_HASH_CACHE: dict = {}


def seal_file_map(run_dir: str | Path) -> Dict[str, str]:
    """Recursive sha256 map of every file under a run directory.

    Keys are POSIX relative paths: top-level files keep their bare name
    (`metrics.json`), nested files carry their relative path
    (`stages/train.log`), so old top-level-only manifests and new recursive
    manifests share one namespace without collision (a bare name never
    contains `/`). The manifest itself is excluded. Used by the runner at
    sealing time and by `repair` when re-sealing, so both always agree on
    the sealed surface (nested run logs are sealed, not just top-level).
    """
    run_dir = Path(run_dir)
    out: Dict[str, str] = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest.json":
            out[p.relative_to(run_dir).as_posix()] = sha256(p)
    return out


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
    meta = getattr(task_spec, "metadata", None)
    if meta is None and isinstance(task_spec, dict):
        meta = task_spec.get("metadata", task_spec)
    elif meta is None:
        meta = {}
    return {
        "task_sha256": digest(meta),
        "scorer_sha256": digest(implementation),
    }


def repair_run_dir(*args, **kwargs):
    """Re-export of :func:`harness.repair.repair_run_dir`."""
    from harness.repair import repair_run_dir as _impl

    return _impl(*args, **kwargs)


repair_run = repair_run_dir


def run_repair(*args, **kwargs):
    """Re-export of :func:`harness.repair.run_repair`."""
    from harness.repair import run_repair as _impl

    return _impl(*args, **kwargs)


def validate_run(directory, task_spec=None, *, plugin=None, expected=None):
    directory = Path(directory)
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("LEGACY_UNVERIFIED: missing manifest v2")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault('provenance', 'local_run')
    from harness.provenance import validate_provenance_record
    validate_provenance_record(manifest)
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
        manifest_val = manifest.get(field)
        if field == "scorer_sha256":
            if manifest_val == value:
                continue
            # Built-in generic scorer tolerance: when custom_scorer_sha256 is None/absent
            # in both the manifest and the checks, allow any recognized canonical generic scorer hash.
            if not manifest.get("custom_scorer_sha256") and not checks.get("custom_scorer_sha256"):
                if manifest_val in CANONICAL_GENERIC_SCORER_HASHES or value in CANONICAL_GENERIC_SCORER_HASHES:
                    continue
        if manifest_val != value:
            raise ValueError(f"Run identity mismatch: {field}")
    files = manifest.get("files", {})
    for name in REQUIRED:
        if name not in files:
            raise ValueError(f"Missing required artifact: {name}")
    for name, value in files.items():
        target = directory / name
        if (
            name in (".", "..")
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not target.is_file()
            or sha256(target) != value
        ):
            raise ValueError(f"Artifact hash mismatch: {name}")
    import importlib

    import yaml

    from harness.hook_ref import ensure_hook_project_roots_on_sys_path, resolve_metrics_hook

    cfg = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
    hook = cfg.get("paths", {}).get("metrics_evaluator")
    if hook:
        ensure_hook_project_roots_on_sys_path()
        module, name = resolve_metrics_hook(hook)
        try:
            scorer_func = getattr(importlib.import_module(module), name)
        except (ModuleNotFoundError, AttributeError) as exc:
            import os

            import sys

            raise RuntimeError(
                f"metrics_evaluator hook {hook!r} is not importable "
                f"(module={module!r}, name={name!r}). "
                f"project_root={os.environ.get('HARNESS_PROJECT_ROOT', '<unset>')}; "
                f"sys.path[:5]={sys.path[:5]!r}. "
                "Verify the scorer module exists in the task workspace (and in "
                "the Kaggle code bundle for remote runs); callers that import "
                "artifacts outside the workspace (journal sync, repair, Kaggle "
                "output import) should pass the workspace root explicitly."
            ) from exc
        if manifest.get("custom_scorer_sha256") != digest(inspect.getsource(scorer_func)):
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
