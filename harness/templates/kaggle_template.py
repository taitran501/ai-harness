"""Kaggle Kernel Runner, Dataset Sync, Status Monitor, and Artifact Puller.

Enables complete headless execution via Kaggle API:
1. `push_dataset_to_kaggle`: Packages `dataset/` + `splits/` into a private Kaggle Dataset once (`kaggle datasets create` / `version`).
2. `setup_kernel_package`: Generates `kernel-metadata.json` with attached `dataset_sources` and packs lightweight code (`train.py`, `src/`, `configs/`).
3. `get_kernel_status`: Polls `kaggle kernels status <slug>` to monitor running/completed state.
4. `pull_kernel_output`: Runs `kaggle kernels output <slug>` and migrates lightweight prediction/metric files (`oof_predictions.csv`, `test_predictions.csv`, `metrics.json`, `run.log`, `EXPERIMENT_CARD.md`) directly into `artifacts/<exp_id>/`.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class KaggleKernelManager:
    """Manages Kaggle Dataset sync, Kernel dispatch, status monitoring, and output pulling."""

    @staticmethod
    def get_authenticated_username() -> str:
        """Detect current Kaggle username from kaggle.json, env, or CLI."""
        env_user = os.environ.get("KAGGLE_USERNAME")
        if env_user:
            return env_user
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                data = json.loads(kaggle_json.read_text(encoding="utf-8"))
                if data.get("username"):
                    return data["username"]
            except Exception:
                pass
        try:
            res = subprocess.run(
                ["kaggle", "config", "view"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in (res.stdout or "").splitlines():
                if "username:" in line.lower():
                    val = line.split(":", 1)[1].strip()
                    if val and val != "None":
                        return val
        except Exception:
            pass
        return ""

    @classmethod
    def _get_task_slug(cls) -> str:
        """Derive a URL/API-safe slug from active TaskSpec."""
        try:
            import re
            from harness.task_spec import load_active_task_spec

            spec = load_active_task_spec()
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", spec.name.lower()).strip("-")
            # Kaggle kernel titles cannot exceed 50 characters.
            # Kernel title is derived as `f"{slug}-{exp_id}".replace('-', ' ').title()`.
            # To ensure title <= 50 chars even with experiment names, limit task slug to 25 chars.
            if len(slug) > 25:
                slug = slug[:25].rstrip("-")
            return slug or "competition"
        except Exception:
            return "competition"

    @classmethod
    def push_dataset_to_kaggle(
        cls,
        dataset_slug: Optional[str] = None,
        *,
        license_name: Optional[str] = None,
        project_root: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Package task data and push it without inventing a dataset license."""
        from harness.project import HarnessProject
        from harness.data_bundle import stage_declared_data
        import yaml

        root = Path(project_root or Path.cwd()).expanduser().resolve()
        project = HarnessProject.discover(root)
        task_doc = (
            yaml.safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
        )
        task = task_doc.get("task", task_doc)
        data_config = task.get("data_config") or task.get("data") or {}
        license_name = (
            license_name
            or data_config.get("license")
            or data_config.get("dataset_license")
        )
        if not license_name:
            return {
                "success": False,
                "error": "Dataset upload needs data_config.license in the task specification; ML-Harness will not invent a license.",
            }
        from harness.cloud_policy import (
            UploadConsentStore,
            dataset_fingerprint,
            private_upload_resource_names,
        )

        fingerprint_data = dataset_fingerprint(project)
        if not UploadConsentStore(project).is_granted(
            "kaggle", fingerprint_data["fingerprint"]
        ):
            return {
                "success": False,
                "error": "Private dataset upload requires explicit consent for this task, provider, and data fingerprint.",
                "fingerprint": fingerprint_data["fingerprint"],
            }
        username = cls.get_authenticated_username()
        try:
            resources = private_upload_resource_names(
                project, fingerprint_data=fingerprint_data, kaggle_username=username
            )
        except ValueError as exc:
            return {
                "success": False,
                "error": str(exc),
                "fingerprint": fingerprint_data["fingerprint"],
            }
        full_slug = resources["kaggle_dataset_slug"]
        if not full_slug:
            return {
                "success": False,
                "error": "Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry.",
                "fingerprint": fingerprint_data["fingerprint"],
            }
        slug_name = resources["kaggle_dataset_name"]

        spec_name = "Competition"
        try:
            from harness.task_spec import load_active_task_spec

            spec_name = load_active_task_spec(project.task_config_path).name
        except Exception:
            pass

        raw_title = f"{spec_name[:30].strip()} Data ({slug_name[-8:]})"
        dataset_title = raw_title[:50].strip() if len(raw_title) >= 6 else f"Data {slug_name}"[:50]
        metadata = {
            "title": dataset_title,
            "id": full_slug,
            "licenses": [{"name": license_name}],
        }
        logs = []
        try:
            with tempfile.TemporaryDirectory(prefix="mlh-kaggle-dataset-") as temp_dir:
                stage_dir = Path(temp_dir)
                staged = stage_declared_data(project, stage_dir)
                (stage_dir / "dataset-metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                logs.append(
                    f"Staged {staged['file_count']} task data file(s) for private upload."
                )

                # Kaggle datasets are private by default; public visibility is never requested here.
                create_cmd = [
                    "kaggle",
                    "datasets",
                    "create",
                    "-p",
                    str(stage_dir),
                    "--dir-mode",
                    "zip",
                ]
                logs.append(f"$ {' '.join(create_cmd)}")
                proc = subprocess.run(
                    create_cmd, capture_output=True, text=True, timeout=300
                )
                out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                logs.append(out.strip())

                if (
                    proc.returncode != 0
                    or "already exists" in out.lower()
                    or "409" in out.lower()
                ):
                    ver_cmd = [
                        "kaggle",
                        "datasets",
                        "version",
                        "-p",
                        str(stage_dir),
                        "-m",
                        "Sync task data and validation splits",
                        "--dir-mode",
                        "zip",
                    ]
                    logs.append(f"$ {' '.join(ver_cmd)}")
                    proc_v = subprocess.run(
                        ver_cmd, capture_output=True, text=True, timeout=300
                    )
                    out_v = (proc_v.stdout or "") + "\n" + (proc_v.stderr or "")
                    logs.append(out_v.strip())
                    success = proc_v.returncode == 0
                else:
                    success = True
        except Exception as exc:
            return {
                "success": False,
                "dataset_slug": full_slug,
                "error": str(exc),
                "logs": "\n".join(logs),
            }

        return {
            "success": success,
            "dataset_slug": full_slug,
            "visibility": "private",
            "logs": "\n".join(logs),
        }

    @classmethod
    def generate_kernel_metadata(
        cls,
        kernel_slug: str,
        code_file: str = "run_kaggle.py",
        enable_gpu: bool = True,
        enable_internet: bool = True,
        dataset_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate kernel-metadata.json configuration dictionary."""
        username = cls.get_authenticated_username()
        owner = username or "ml-harness-user"
        full_kernel_id = kernel_slug if "/" in kernel_slug else f"{owner}/{kernel_slug}"
        return {
            "id": full_kernel_id,
            "title": full_kernel_id.split("/")[-1].replace("-", " ").title(),
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "true" if enable_internet else "false",
            "dataset_sources": dataset_sources or [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }

    @classmethod
    def setup_kernel_package(
        cls,
        target_dir: Path | str,
        config_path: str,
        exp_id: str,
        dataset_slug: Optional[str] = None,
        *,
        project_root: Optional[Path | str] = None,
        task_config_path: Optional[Path | str] = None,
        task_id: Optional[str] = None,
        job_id: Optional[str] = None,
        runtime_manifest: Optional[Dict[str, Any]] = None,
        enable_internet: bool = False,
    ) -> Path:
        """Create self-contained staging directory for Kaggle kernel push with code + mounted dataset."""
        from harness.project import HarnessProject

        root = Path(project_root or Path.cwd()).expanduser().resolve()
        project = HarnessProject.discover(root, task_config_path)
        config_file = Path(config_path).expanduser()
        if not config_file.is_absolute():
            config_file = root / config_file
        config_file = config_file.resolve()
        config_rel = config_file.relative_to(root).as_posix()
        config_sha256 = (
            __import__("hashlib").sha256(config_file.read_bytes()).hexdigest()
        )
        stage = Path(target_dir)
        stage.mkdir(parents=True, exist_ok=True)

        username = cls.get_authenticated_username()
        t_slug = cls._get_task_slug()
        slug = f"{t_slug}-{exp_id.replace('_', '-')}"
        resolved_dataset = dataset_slug
        if not resolved_dataset or resolved_dataset == "antigravity/dataset":
            from harness.cloud_policy import private_upload_resource_names

            resources = private_upload_resource_names(project, kaggle_username=username)
            resolved_dataset = resources["kaggle_dataset_slug"]
            if not resolved_dataset:
                raise ValueError(
                    "Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry."
                )

        meta = cls.generate_kernel_metadata(
            kernel_slug=slug,
            code_file="run_kaggle.py",
            dataset_sources=[resolved_dataset] if resolved_dataset else [],
            enable_internet=enable_internet,
        )
        with open(stage / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Bundle only code/config; dataset contents use the separately consented provider dataset.
        import tempfile

        runtime = runtime_manifest or {
            "schema_version": 2,
            "core": {
                "distribution": "ml-harness",
                "version": "0.2.0",
                "source": "bundled",
            },
            "plugin": {
                "id": project.plugin_id or "default",
                "version": "1",
                "install": None,
            },
            "requirements": [],
            "bundle_files": [],
        }
        with tempfile.TemporaryDirectory(prefix="mlh-kaggle-code-") as temp_stage:
            bundle_base = Path(temp_stage) / "code_bundle"
            bundle_base.mkdir(parents=True, exist_ok=True)
            for rel in ("src", "configs"):
                source = root / rel
                if source.is_dir():
                    shutil.copytree(source, bundle_base / rel, dirs_exist_ok=True)
            # Bundle the exact installed harness source tree, not a workspace-local fork.
            harness_source = Path(__file__).resolve().parents[1]
            shutil.copytree(
                harness_source,
                bundle_base / "harness",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
            )
            manifest_source = root / ".ml-harness" / "project.yaml"
            if manifest_source.is_file():
                manifest_dest = bundle_base / ".ml-harness" / "project.yaml"
                manifest_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(manifest_source, manifest_dest)
            if not config_file.is_relative_to(root):
                raise ValueError("Experiment config must be inside the task workspace.")
            config_dest = bundle_base / config_rel
            if not config_dest.exists():
                config_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, config_dest)
            task_config = project.task_config_path
            if not task_config.is_relative_to(root):
                raise ValueError("Task config must be inside the task workspace.")
            task_config_rel = task_config.relative_to(root).as_posix()
            if not (bundle_base / task_config_rel).exists():
                (bundle_base / task_config_rel).parent.mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copy2(task_config, bundle_base / task_config_rel)

            runtime_files_dir = bundle_base / "runtime_files"
            install = (runtime.get("plugin") or {}).get("install") or {}
            if install.get("type") == "wheel":
                wheel_path = Path(install["path"])
                runtime_files_dir.mkdir(parents=True, exist_ok=True)
                wheel_name = wheel_path.name
                shutil.copy2(wheel_path, runtime_files_dir / wheel_name)
                runtime["plugin"]["install"] = {
                    "type": "wheel",
                    "path": f"runtime_files/{wheel_name}",
                }
            bundled_rel_paths = []
            for file_name in runtime.get("bundle_files", []):
                source = Path(file_name).resolve()
                try:
                    rel_path = source.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"Remote bundle file must be inside the task workspace: {source}"
                    ) from exc
                destination = bundle_base / rel_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                bundled_rel_paths.append(rel_path.as_posix())
            runtime["bundle_files"] = bundled_rel_paths
            (bundle_base / "ml-harness-runtime.json").write_text(
                json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            import io, base64, zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file_path in bundle_base.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(bundle_base))
            bundle_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        dataset_folder_name = resolved_dataset.split("/")[-1]

        task_id = (
            task_id or project.manifest.get("task_id") or project.plugin_id or "task"
        )
        artifact_dir = project.artifacts_dir.relative_to(root).as_posix()
        runner_content = f"""import hashlib
import json
import subprocess
import sys
import os
import shutil
import zipfile
import base64
import io
from pathlib import Path

print("[Kaggle Kernel] Starting execution for: {exp_id}")

# 1. Extract embedded code bundle
bundle_bytes = base64.b64decode('{bundle_b64}')
with zipfile.ZipFile(io.BytesIO(bundle_bytes), 'r') as zf:
    zf.extractall('.')

# 2. Restore only the task data paths included in the separately consented dataset.
input_root = Path("/kaggle/input/{dataset_folder_name}")
if not input_root.exists():
    # Fallback scan in /kaggle/input
    for candidate in Path("/kaggle/input").glob("*"):
        if (candidate / "dataset").exists() or any(candidate.glob("*.csv")) or any(candidate.glob("*.tsv")):
            input_root = candidate
            break

data_manifest_path = input_root / "mlh-data-bundle.json"
if data_manifest_path.is_file():
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    for relative in data_manifest.get("paths", []):
        source = input_root / relative
        target = Path(relative)
        if not source.exists():
            raise RuntimeError(f"Uploaded task data path is missing: {{relative}}")
        if target.exists():
            raise RuntimeError(f"Uploaded task data conflicts with bundled task code: {{relative}}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
elif input_root.exists():
    # Compatibility with a user-selected Kaggle dataset that was not created by this harness.
    if (input_root / "dataset").exists():
        shutil.copytree(input_root / "dataset", "dataset", dirs_exist_ok=True)
    if (input_root / "splits").exists():
        shutil.copytree(input_root / "splits", "splits", dirs_exist_ok=True)
    if not (input_root / "dataset").exists():
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for item in input_root.iterdir():
            if item.name in {{"dataset-metadata.json", "README.md", "mlh-data-bundle.json"}}:
                continue
            target = dataset_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, target)

# 3. Check and install missing dependencies
runtime = json.loads(Path("ml-harness-runtime.json").read_text(encoding="utf-8"))
requirements = ["numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.3.0", "pyyaml>=6.0", "scipy>=1.11.0"] + runtime.get("requirements", [])
missing = []
for req in requirements:
    pkg = req.split(">=")[0].split("==")[0].split("<")[0].strip()
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        missing.append(req)

if missing:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True, timeout=120)
    except Exception as exc:
        print(f"[Warning] Failed to pip install missing packages ({{missing}}): {{exc}}", file=sys.stderr)

install = (runtime.get("plugin") or {{}}).get("install") or {{}}
if install.get("type") == "wheel":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", install["path"]], check=True)
elif install.get("type") == "index":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", install["requirement"]], check=True)

# 4. Run training
cmd = [sys.executable, "-m", "harness.runner", "--project-root", ".", "--task-config", "{task_config_rel}", "--config", "{config_rel}"]
print(f"Executing: {{' '.join(cmd)}}")
ret = subprocess.run(cmd)

if ret.returncode != 0:
    print(f"Execution failed with returncode {{ret.returncode}}", file=sys.stderr)
    sys.exit(ret.returncode)

# 5. Copy lightweight output artifacts to /kaggle/working root so `kaggle kernels output` downloads them cleanly
exp_dir = Path("{artifact_dir}") / "{exp_id}"
essential_files = [
    "metrics.json",
    "summary.csv",
    "summary.md",
    "predictions.csv",
    "oof_predictions.csv",
        "evaluation_truth.csv",
    "test_predictions.csv",
    "submission.csv",
    "DIAGNOSTICS.md",
    "diagnostics.json",
    "run.log",
    "EXPERIMENT_CARD.md",
    "run_config.yaml",
]
if exp_dir.exists():
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8")) if (exp_dir / "metrics.json").is_file() else None
    records = metrics if isinstance(metrics, list) else [metrics]
    if not any(isinstance(row, dict) and row.get("experiment_id", row.get("exp_id")) == "{exp_id}" for row in records):
        raise RuntimeError("Run metrics are missing the expected experiment identity.")
    manifest = json.loads((exp_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest.update({{
        "schema_version": 2,
        "job_id": {json.dumps(job_id)},
        "task_id": {json.dumps(str(task_id))},
        "exp_id": "{exp_id}",
        "config_sha256": "{config_sha256}",
        "core_version": (runtime.get("core") or {{}}).get("version"),
        "plugin_id": (runtime.get("plugin") or {{}}).get("id"),
        "files": {{}},
    }})
    for fname in essential_files:
        src_file = exp_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, Path(".") / fname)
            manifest["files"][fname] = hashlib.sha256(src_file.read_bytes()).hexdigest()
    Path("artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
else:
    raise RuntimeError(f"Expected experiment artifact directory was not created: {{exp_dir}}")

print("Training completed. Lightweight output artifacts copied to working root.")
"""
        (stage / "run_kaggle.py").write_text(runner_content, encoding="utf-8")
        return stage

    @classmethod
    def get_kernel_status(
        cls, exp_id: str, kernel_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check status of a Kaggle kernel via `kaggle kernels status <slug>`."""
        username = cls.get_authenticated_username()
        t_slug = cls._get_task_slug()
        slug = kernel_slug or f"{username}/{t_slug}-{exp_id.replace('_', '-')}"
        if "/" not in slug:
            slug = f"{username}/{slug}"

        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "status", slug],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            status_state = "UNKNOWN"
            lower_out = output.lower()
            tokens = set(re.findall(r"[a-z]+", lower_out))
            if tokens.intersection({"failed", "failure", "error", "invalid"}):
                status_state = "FAILED"
            elif tokens.intersection({"cancelled", "canceled", "cancel"}):
                status_state = "CANCELLED"
            elif "queued" in tokens:
                status_state = "QUEUED"
            elif "running" in tokens:
                status_state = "RUNNING"
            elif tokens.intersection({"complete", "completed", "success", "succeeded"}):
                status_state = "COMPLETED"

            return {
                "success": proc.returncode == 0,
                "kernel_slug": slug,
                "status": status_state,
                "raw_output": output,
            }
        except Exception as exc:
            return {
                "success": False,
                "kernel_slug": slug,
                "status": "ERROR",
                "raw_output": str(exc),
            }

    @classmethod
    def wait_for_kernel(
        cls,
        exp_id: str,
        kernel_slug: Optional[str] = None,
        *,
        poll_interval: float = 10.0,
        timeout_seconds: float = 24 * 60 * 60,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll until Kaggle reports a terminal state; queued/running are never treated as complete."""
        deadline = time.monotonic() + timeout_seconds
        last: Dict[str, Any] = {
            "success": False,
            "status": "UNKNOWN",
            "kernel_slug": kernel_slug,
        }
        while time.monotonic() < deadline:
            last = cls.get_kernel_status(exp_id, kernel_slug)
            if status_callback:
                status_callback(last)
            state = last.get("status")
            if state in {"COMPLETED", "FAILED", "CANCELLED"}:
                return last
            if state == "ERROR":
                return last
            time.sleep(max(0.0, poll_interval))
        return {**last, "success": False, "status": "TIMEOUT", "timed_out": True}

    @classmethod
    def pull_kernel_output(
        cls,
        exp_id: str,
        kernel_slug: Optional[str] = None,
        *,
        artifact_dir: Optional[Path | str] = None,
        expected_identity: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Download outputs only after artifact identity matches the submitted task and run."""
        username = cls.get_authenticated_username()
        t_slug = cls._get_task_slug()
        slug = kernel_slug or f"{username}/{t_slug}-{exp_id.replace('_', '-')}"
        if "/" not in slug:
            slug = f"{username}/{slug}"

        from harness.project import HarnessProject

        project = HarnessProject.discover()
        target_artifacts_dir = (
            Path(artifact_dir).expanduser()
            if artifact_dir
            else project.artifacts_dir / exp_id
        )
        if not target_artifacts_dir.is_absolute():
            target_artifacts_dir = project.root / target_artifacts_dir
        target_artifacts_dir = target_artifacts_dir.resolve()
        try:
            target_artifacts_dir.relative_to(project.root)
        except ValueError:
            return {
                "success": False,
                "kernel_slug": slug,
                "exp_id": exp_id,
                "raw_output": "Artifact destination must be inside the attached workspace.",
            }
        if target_artifacts_dir.exists() and any(target_artifacts_dir.iterdir()):
            return {
                "success": False,
                "kernel_slug": slug,
                "exp_id": exp_id,
                "artifacts_dir": str(target_artifacts_dir),
                "raw_output": "Refusing to overwrite an existing experiment artifact directory.",
            }

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"mlh-kaggle-{exp_id}-"
            ) as temp_dir:
                download_dir = Path(temp_dir)
                proc = subprocess.run(
                    ["kaggle", "kernels", "output", slug, "-p", str(download_dir)],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                manifest_path = next(download_dir.rglob("artifact_manifest.json"), None)
                manifest = None
                if manifest_path:
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        manifest = None
                if expected_identity:
                    if not isinstance(manifest, dict):
                        return {
                            "success": False,
                            "kernel_slug": slug,
                            "exp_id": exp_id,
                            "raw_output": "Downloaded output is missing a valid artifact_manifest.json.",
                        }
                    mismatches = {
                        key: {"expected": value, "actual": manifest.get(key)}
                        for key, value in expected_identity.items()
                        if value is not None and manifest.get(key) != value
                    }
                    if mismatches:
                        return {
                            "success": False,
                            "kernel_slug": slug,
                            "exp_id": exp_id,
                            "identity_mismatches": mismatches,
                            "raw_output": "Downloaded Kaggle artifacts do not match this task/config/job.",
                        }
                    for file_name, expected_hash in (
                        manifest.get("files") or {}
                    ).items():
                        file_path = next(download_dir.rglob(file_name), None)
                        if (
                            not file_path
                            or hashlib.sha256(file_path.read_bytes()).hexdigest()
                            != expected_hash
                        ):
                            return {
                                "success": False,
                                "kernel_slug": slug,
                                "exp_id": exp_id,
                                "invalid_artifact": file_name,
                                "raw_output": "Downloaded Kaggle artifact failed its SHA-256 identity check.",
                            }
                target_artifacts_dir.mkdir(parents=True, exist_ok=True)
                migrated = cls.migrate_downloaded_artifacts(
                    download_dir, target_artifacts_dir
                )
                metrics_path = target_artifacts_dir / "metrics.json"
                metrics_data = None
                if metrics_path.exists():
                    try:
                        metrics_data = json.loads(
                            metrics_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        pass
                success = (
                    len(migrated) > 0 or proc.returncode == 0
                ) and metrics_data is not None
                if not success:
                    shutil.rmtree(target_artifacts_dir, ignore_errors=True)
                return {
                    "success": success,
                    "kernel_slug": slug,
                    "exp_id": exp_id,
                    "artifacts_dir": str(target_artifacts_dir),
                    "migrated_files": migrated,
                    "metrics": metrics_data,
                    "identity_verified": bool(
                        manifest is not None and expected_identity
                    ),
                    "raw_output": output,
                }
        except Exception as exc:
            return {
                "success": False,
                "kernel_slug": slug,
                "exp_id": exp_id,
                "migrated_files": [],
                "raw_output": str(exc),
            }

    @staticmethod
    def migrate_downloaded_artifacts(
        download_dir: Path | str,
        target_artifacts_dir: Path | str,
    ) -> List[str]:
        """Import one verified run, preserving every sealed artifact byte."""
        from harness.artifacts import validate_run

        source = Path(download_dir)
        manifests = list(source.rglob("artifact_manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Remote import requires exactly one manifest v2")
        source_run = manifests[0].parent
        manifest = validate_run(source_run)
        destination = Path(target_artifacts_dir)
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Refusing to overwrite imported artifacts")
        destination.mkdir(parents=True, exist_ok=True)
        names = [*manifest["files"], "artifact_manifest.json"]
        for name in names:
            shutil.copy2(source_run / name, destination / name)
        validate_run(destination)
        return names
