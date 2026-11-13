"""Immutable run evidence shared by execution and downstream consumers."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import os
import shutil
import stat
import sys
import tempfile

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
    "a6811e42c5a410bcb03df6607eaf373ff9522abcd8c14751727e29d9fd13c16b",
    "871e6c3c03de49f111cdb2f95434b79906efc5201e5ab2a65af9f381c396f657",
    "ac2101968711361422972bb6ebc7553133e65a9715f9a4b8e689aa54c81d1abb",
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


class ArtifactValidationError(ValueError):
    pass


class ArtifactConflictError(ValueError):
    pass


class AtomicPublicationUnsupportedError(OSError):
    pass


@dataclass(frozen=True)
class PublicationResult:
    destination: Path
    disposition: str
    manifest_sha256: str
    files: tuple[str, ...]


def _read_artifact_json(path):
    from harness.state_store import read_json_state, StateIntegrityError
    try:
        return read_json_state(path)
    except StateIntegrityError as exc:
        raise ArtifactValidationError(str(exc)) from exc


def _fresh_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rename_directory_noreplace(source, destination):
    """Atomic no-replace operation; never fall back to an overwriting rename."""
    if os.name == "nt":
        os.rename(source, destination)
    elif sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise AtomicPublicationUnsupportedError("renameat2(RENAME_NOREPLACE) unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            error = ctypes.get_errno()
            import errno
            if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.EXDEV}:
                raise AtomicPublicationUnsupportedError(error, "Atomic no-replace rename unavailable", str(destination))
            raise OSError(error, os.strerror(error), str(destination))
    else:
        raise AtomicPublicationUnsupportedError(f"Atomic no-replace publication unsupported on {sys.platform}")


def _publication_inventory(directory):
    directory = Path(directory)
    if (not directory.is_dir() or directory.is_symlink()
            or getattr(directory.lstat(), "st_file_attributes", 0) & 0x400):
        raise ArtifactValidationError("Publication source must be a real directory")
    inventory, aliases = {}, set()
    pending = [directory]
    while pending:
        parent = pending.pop()
        # Inspect links/reparse points before scheduling descent. In particular,
        # Windows junctions must never be traversed by a recursive glob first.
        for path in sorted(parent.iterdir()):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ArtifactValidationError(f"Linked/reparse artifact is forbidden: {path}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            _add_publication_file(directory, path, info, inventory, aliases)
    return inventory


def _add_publication_file(directory, path, info, inventory, aliases):
    """Validate one entry whose parent has already passed the no-link check."""
    name = path.relative_to(directory).as_posix()
    if any(part.endswith((".", " ")) or ":" in part or "\\" in part for part in path.relative_to(directory).parts):
        raise ArtifactValidationError(f"Unsafe artifact path: {name}")
    alias = name.casefold()
    if alias in aliases:
        raise ArtifactValidationError(f"Artifact path alias: {name}")
    aliases.add(alias)
    if stat.S_ISREG(info.st_mode):
        inventory[name] = _fresh_sha256(path)
    elif not stat.S_ISDIR(info.st_mode):
        raise ArtifactValidationError(f"Special artifact file: {name}")


def _validate_publication(directory, *, task_spec, plugin, expected, project_root, run_kind):
    inventory = _publication_inventory(directory)
    manifest = _read_artifact_json(Path(directory) / "artifact_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactValidationError("Manifest files must be an object")
    for name in files:
        if (not isinstance(name, str) or not name or name.startswith(("/", "\\"))
                or "\\" in name or ":" in name or any(p in {"", ".", ".."} for p in name.split("/"))):
            raise ArtifactValidationError(f"Unsafe manifest path: {name}")
    if set(inventory) != {*files, "artifact_manifest.json"}:
        raise ArtifactValidationError("Undeclared or missing publication files")
    if any(inventory[name] != value for name, value in files.items()):
        raise ArtifactValidationError("Artifact SHA-256 mismatch (fresh read)")
    validate_run(directory, task_spec, plugin=plugin, expected=expected, project_root=project_root, _run_kind=run_kind)
    if task_spec is not None:
        import pandas as pd
        from harness.evaluation import align_prediction_frame, validate_prediction_contract, read_prediction_frame
        from harness.oof_revalidation import build_variant_predictions
        from harness.script_predictions import extract_test_predictions

        metrics = _read_artifact_json(Path(directory) / "metrics.json")
        truth = pd.read_csv(Path(directory) / "evaluation_truth.csv")
        frame = read_prediction_frame(Path(directory) / "oof_predictions.csv", expected_ids=truth[task_spec.id_column], id_column=task_spec.id_column)
        variants = frame.groupby("input_variant", sort=False) if "input_variant" in frame else [(None, frame)]
        for _, variant in variants:
            aligned = align_prediction_frame(variant, expected_ids=truth[task_spec.id_column], id_column=task_spec.id_column, purpose="oof")
            validate_prediction_contract(task_spec, truth, build_variant_predictions(aligned, task_spec))
        if "test_predictions.csv" in files:
            test = task_spec.load_dataset_split("test", project_root=project_root)
            extract_test_predictions(read_prediction_frame(Path(directory) / "test_predictions.csv", expected_ids=test[task_spec.id_column], id_column=task_spec.id_column), task_spec, expected_ids=test[task_spec.id_column])
        if metrics.get("primary_metric") != task_spec.primary_metric:
            raise ArtifactValidationError("Primary metric mismatch")
    return inventory


def publish_run_artifacts(staging_dir, destination, *, task_spec, plugin=None, expected,
                          project_root, run_kind):
    """Validate, privately copy, and atomically publish immutable run evidence."""
    from harness.state_store import workspace_lock

    if run_kind not in {"full", "probe"}:
        raise ArtifactValidationError("Unknown publication run kind")
    source, destination = Path(staging_dir), Path(destination).absolute()
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ArtifactValidationError("Publication source and destination must be separate trees")
    if plugin is None and task_spec is not None:
        from harness.project import HarnessProject
        from harness.plugins import TaskPluginContext, load_task_plugin
        project = HarnessProject.discover(project_root)
        if project.plugin_id not in {None, "default"}:
            context = TaskPluginContext(project.root, project.task_config_path, task_spec, project.plugin_id)
            plugin = load_task_plugin(project.plugin_id, context)
    kwargs = dict(task_spec=task_spec, plugin=plugin, expected=expected, project_root=project_root, run_kind=run_kind)
    original = _validate_publication(source, **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination.parent) as private:
        staged = Path(private) / "run"
        shutil.copytree(source, staged)
        copied = _validate_publication(staged, **kwargs)
        if copied != original:
            raise ArtifactValidationError("Artifacts changed during publication copy")
        with workspace_lock(destination.parent / ".publication-locks"):
            if destination.exists() or destination.is_symlink():
                try:
                    existing = _validate_publication(destination, **kwargs)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    raise ArtifactConflictError(f"ARTIFACT_PUBLICATION_CONFLICT: {destination}: {exc}") from exc
                if existing != copied:
                    raise ArtifactConflictError(f"ARTIFACT_PUBLICATION_CONFLICT: {destination}")
                disposition = "already_present"
            else:
                try:
                    _rename_directory_noreplace(staged, destination)
                except OSError as exc:
                    if destination.exists() or destination.is_symlink():
                        raise ArtifactConflictError(f"ARTIFACT_PUBLICATION_CONFLICT: {destination}") from exc
                    raise
                disposition = "published"
    return PublicationResult(destination, disposition, copied["artifact_manifest.json"], tuple(sorted(copied)))


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


def _stable_task_spec_source(source: str) -> str:
    """Keep non-scoring model policy accessors out of legacy scorer identity."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    removals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "UniversalTaskSpec":
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name not in {"competition_rules", "model_governance_config"}:
                continue
            if not any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in member.decorator_list
            ):
                continue
            start = min([member.lineno, *(decorator.lineno for decorator in member.decorator_list)])
            end = member.end_lineno or member.lineno
            if end < len(lines) and not lines[end].strip():
                end += 1
            removals.append((start - 1, end))

    for start, end in sorted(removals, reverse=True):
        del lines[start:end]
    return "".join(lines)


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

            implementation += inspect.getsource(evaluation) + _stable_task_spec_source(
                inspect.getsource(task_module)
            )
    except (OSError, TypeError):
        raise ValueError("Scorer implementation cannot be identified")
    meta = getattr(task_spec, "metadata", None)
    if meta is None and isinstance(task_spec, dict):
        meta = task_spec.get("metadata", task_spec)
    elif meta is None:
        meta = {}
    return {
        "task_sha256": digest(scientific_task_metadata(meta)),
        "scorer_sha256": digest(implementation),
    }


def scientific_task_metadata(meta):
    """Backend selection and locking do not change the scientific contract."""
    import copy

    meta = copy.deepcopy(meta)
    if isinstance(meta, dict):
        execution = meta.get("execution")
        if isinstance(execution, dict):
            execution.pop("backend", None)
            execution.pop("locked", None)
            if not execution:
                meta.pop("execution")
        elif execution is None:
            meta.pop("execution", None)
    return meta


def _legacy_task_hashes(task_spec):
    """Recognize old hashes only for finite, known operational variations."""
    import copy

    meta = getattr(task_spec, "metadata", {})
    core = scientific_task_metadata(meta)
    hashes = {digest(core)}
    if "execution" not in core:
        hashes.add(digest({**core, "execution": None}))
    for backend in (None, "local", "modal", "kaggle"):
        for locked in (None, False, True):
            variant = copy.deepcopy(core)
            execution = variant.setdefault("execution", {})
            if backend is not None:
                execution["backend"] = backend
            if locked is not None:
                execution["locked"] = locked
            hashes.add(digest(variant))
    return hashes


def repair_run_dir(*args, **kwargs):
    """Re-export of :func:`harness.repair.repair_run_dir`."""
    from harness.repair import repair_run_dir as _impl

    return _impl(*args, **kwargs)


repair_run = repair_run_dir


def run_repair(*args, **kwargs):
    """Re-export of :func:`harness.repair.run_repair`."""
    from harness.repair import run_repair as _impl

    return _impl(*args, **kwargs)


def validate_run(directory, task_spec=None, *, plugin=None, expected=None, project_root=None, _run_kind="full"):
    directory = Path(directory)
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("LEGACY_UNVERIFIED: missing manifest v2")
    manifest = _read_artifact_json(path)
    manifest.setdefault('provenance', 'local_run')
    from harness.provenance import validate_provenance_record
    validate_provenance_record(manifest)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "COMPLETED"
        or manifest.get("run_kind") != _run_kind
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
        if field == "task_sha256" and task_spec is not None and (manifest_val == value or manifest_val in _legacy_task_hashes(task_spec)):
            continue
        if field == "scorer_sha256":
            if manifest_val == value:
                continue
            # Built-in generic scorer tolerance: when custom_scorer_sha256 is None/absent
            # in both the manifest and the checks, allow any recognized canonical generic scorer hash.
            if not manifest.get("custom_scorer_sha256") and not checks.get("custom_scorer_sha256"):
                if manifest_val in CANONICAL_GENERIC_SCORER_HASHES and value in CANONICAL_GENERIC_SCORER_HASHES:
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
    import yaml

    from harness.hook_ref import isolated_metrics_hook, resolve_metrics_hook

    cfg = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
    hook = cfg.get("paths", {}).get("metrics_evaluator")
    if hook:
        module, name = resolve_metrics_hook(hook)
        try:
            with isolated_metrics_hook(hook, project_root) as scorer_func:
                scorer_hash = digest(inspect.getsource(scorer_func))
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
        if manifest.get("custom_scorer_sha256") != scorer_hash:
            raise ValueError("Custom scorer implementation changed")
    if sha256(directory / "run_config.yaml") != manifest["resolved_config_sha256"]:
        raise ValueError("Resolved configuration identity mismatch")
    metrics = _read_artifact_json(directory / "metrics.json")
    from harness.evaluation import utility

    if isinstance(metrics.get("primary_score"), bool) or not isinstance(metrics.get("primary_score"), (int, float)):
        raise ValueError("Primary score must be finite numeric evidence")
    utility(metrics["primary_score"], metrics["direction"])
    if metrics.get("experiment_id") != manifest["exp_id"]:
        raise ValueError("Metrics experiment identity mismatch")
    if task_spec is not None and (metrics.get("primary_metric") != task_spec.primary_metric or metrics.get("direction") != task_spec.direction):
        raise ValueError("Metrics contract mismatch")
    if metrics.get("utility") != utility(metrics["primary_score"], metrics["direction"]):
        raise ValueError("Metrics utility mismatch")
    from harness.validation_scope import validate_scope
    if _run_kind != "external":
        validate_scope(directory, manifest, require_full=_run_kind == "full", task_spec=task_spec)
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


def validate_test_prediction(path, *, task_spec, plugin=None, project_root=None):
    path = Path(path).resolve()
    from harness.script_predictions import extract_test_predictions
    from harness.evaluation import read_prediction_frame

    if project_root is None:
        project_root = next((p for p in path.parents if (p / ".ai-harness/project.yaml").is_file()
                             or (p / ".ml-harness/project.yaml").is_file()), None)
    if project_root is None:
        raise ValueError("Test prediction validation requires the owning project_root")
    test = task_spec.load_dataset_split("test", project_root=project_root)
    extract_test_predictions(read_prediction_frame(path, expected_ids=test[task_spec.id_column], id_column=task_spec.id_column), task_spec, expected_ids=test[task_spec.id_column])
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
