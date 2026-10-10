"""Runtime helpers loaded by path before the harness dependencies are installed."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile


BASE_REQUIREMENTS = [
    "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.3.0",
    "pyyaml>=6.0", "scipy>=1.11.0",
]


def ensure_dependencies(runtime, *, enable_internet=False):
    """Check distribution versions without importing native extensions; fail early."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement

    def satisfied(value):
        req = Requirement(value)
        if req.marker and not req.marker.evaluate():
            return True
        try:
            return req.specifier.contains(metadata.version(req.name), prereleases=True)
        except metadata.PackageNotFoundError:
            return False

    requirements = list(dict.fromkeys(BASE_REQUIREMENTS + runtime.get("requirements", [])))
    missing = [value for value in requirements if not satisfied(value)]
    plugin = runtime.get("plugin") or {}
    install = plugin.get("install") or {}
    plugin_requirement = (
        f"{plugin['distribution']}=={plugin['version']}"
        if plugin.get("distribution") and plugin.get("version") else None
    )
    plugin_ready = bool(plugin_requirement and satisfied(plugin_requirement))
    if not enable_internet and (missing or install.get("type") == "index" and not plugin_ready):
        needed = missing + ([install["requirement"]] if install.get("type") == "index" and not plugin_ready else [])
        raise RuntimeError(
            "Kaggle internet is disabled and the image does not satisfy: " + ", ".join(needed)
            + ". Supply these versions in the image or explicitly enable task.runtime.enable_internet."
        )
    if missing:
        # Include all constraints so a newly installed dependency cannot invalidate a pin.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *requirements], check=True, timeout=600)
    if install and not plugin_ready:
        if install.get("type") == "wheel":
            args = ["--no-deps", install["path"]]
        elif install.get("type") == "index":
            args = ["--no-deps", install["requirement"]]
        else:
            raise ValueError("Unsupported remote plugin installation type")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True, timeout=600)
    remaining = [value for value in requirements if not satisfied(value)]
    if plugin_requirement and not satisfied(plugin_requirement):
        remaining.append(plugin_requirement)
    if remaining:
        raise RuntimeError("Kaggle dependency versions remain unsatisfied: " + ", ".join(remaining))


def _relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Invalid uploaded data path: {value}")
    return Path(*path.parts)


def restore_dataset(input_root, workspace):
    """Restore declared paths, including directories uploaded with --dir-mode zip."""
    input_root, workspace = Path(input_root), Path(workspace)
    if not input_root.is_dir():
        raise RuntimeError(f"The selected Kaggle dataset mount is missing: {input_root}")
    manifest_path = input_root / "mlh-data-bundle.json"
    if not manifest_path.is_file():
        # Existing user-selected datasets retain the legacy dataset/ and splits/ layout.
        for item in input_root.iterdir():
            if item.name in {"dataset-metadata.json", "README.md"}:
                continue
            relative = item.name if item.name in {"dataset", "splits"} else f"dataset/{item.name}"
            target = workspace / relative
            if target.exists():
                raise RuntimeError(f"Uploaded task data conflicts with bundled task code: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or []
    if not files:
        raise RuntimeError("Uploaded task data manifest has no files")
    # Kaggle may mount top-level uploaded directories as <directory>.zip.
    # Extract into a temporary source tree; only manifest-listed files enter the run.
    with tempfile.TemporaryDirectory(prefix="mlh-kaggle-data-") as temporary:
        cache = Path(temporary)
        opened = set()
        for value in files:
            relative = _relative(value)
            source = input_root / relative
            if not source.is_file():
                top = relative.parts[0]
                archive_path = input_root / f"{top}.zip"
                if archive_path.is_file() and top not in opened:
                    extracted = cache / top
                    extracted.mkdir()
                    with zipfile.ZipFile(archive_path) as archive:
                        for member in archive.infolist():
                            _relative(member.filename.rstrip("/"))
                        archive.extractall(extracted)
                    opened.add(top)
                # DirectoryArchive stores files relative to the uploaded directory.
                source = cache / top / Path(*relative.parts[1:])
                if not source.is_file():
                    # Also accept archives containing the top-level directory itself.
                    source = cache / top / relative
            if not source.is_file():
                raise RuntimeError(f"Uploaded task data path is missing: {value}")
            target = workspace / relative
            if target.exists():
                raise RuntimeError(f"Uploaded task data conflicts with bundled task code: {value}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def export_run(exp_dir, output_root, identity):
    """Export every sealed file once, without exposing the training workspace."""
    from harness.artifacts import validate_run

    exp_dir, output_root = Path(exp_dir), Path(output_root)
    manifest = validate_run(exp_dir, expected={"exp_id": identity["exp_id"]})
    for name in [*manifest["files"], "artifact_manifest.json"]:
        if (output_root / name).exists():
            raise RuntimeError(f"Refusing to overwrite Kaggle output: {name}")
    for name in manifest["files"]:
        shutil.copy2(exp_dir / name, output_root / name)
    manifest.update(identity)
    (output_root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_run(output_root)
