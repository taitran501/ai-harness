"""Kaggle Kernel Runner, Dataset Sync, Status Monitor, and Artifact Puller.

Enables complete headless execution via Kaggle API:
1. `push_dataset_to_kaggle`: Packages `dataset/` + `splits/` into a private Kaggle Dataset once (`kaggle datasets create` / `version`).
2. `setup_kernel_package`: Generates a short Python script and a separately mounted code/config bundle.
3. `get_kernel_status`: Polls `kaggle kernels status <slug>` to monitor running/completed state.
4. `pull_kernel_output`: Downloads and verifies one sealed run in the attached workspace.
"""

from __future__ import annotations

import json
import base64
import copy
import io
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Optional test-only bundle cache. Production code never sets this; tests inject
# a callable so repeated identical bundle builds (copy+zip of the source tree,
# ~1s each) are computed once per process. See tests/test_kaggle_runtime.py.
_BUNDLE_BYTES_CACHE: Optional[Dict[tuple, bytes]] = None


def _is_critical_identity_mismatch(key: str, mismatch: Dict[str, Any]) -> bool:
    """Decide whether an artifact identity mismatch should block the import.

    ``core_version`` drift is always tolerated (forward-compatible harness
    versions). A ``job_id`` mismatch is only critical when the remote kernel
    actually reports a *different* concrete job id; remote kernels do not know
    about the locally generated job id, so an empty/``None`` actual value is not
    a real conflict and must not block the import.
    """
    if key == "core_version":
        return False
    if key == "job_id" and mismatch.get("actual") in (None, ""):
        return False
    return True


class KaggleKernelManager:
    """Manages Kaggle Dataset sync, Kernel dispatch, status monitoring, and output pulling."""

    @staticmethod
    def get_authenticated_username() -> str:
        """Detect current Kaggle username from kaggle.json, env, or CLI."""
        env_user = os.environ.get("KAGGLE_USERNAME")
        if env_user:
            return env_user
        config_dirs = [Path.home() / ".kaggle", Path.home() / ".config" / "kaggle"]
        if os.environ.get("KAGGLE_CONFIG_DIR"):
            config_dirs.insert(0, Path(os.environ["KAGGLE_CONFIG_DIR"]).expanduser())
        for directory in config_dirs:
            for filename in ("kaggle.json", "credentials.json"):
                try:
                    data = json.loads((directory / filename).read_text(encoding="utf-8"))
                    if data.get("username"):
                        return data["username"]
                except (OSError, ValueError):
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
    def _get_task_slug(cls, task_config_path=None) -> str:
        """Derive a URL/API-safe slug from active TaskSpec."""
        try:
            from harness.task_spec import load_active_task_spec

            spec = load_active_task_spec(task_config_path)
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", spec.name.lower()).strip("-")
            # Leave space for the experiment; _bounded_slug limits the final ID/title.
            if len(slug) > 25:
                slug = slug[:25].rstrip("-")
            return slug or "competition"
        except Exception:
            return "competition"

    @staticmethod
    def _bounded_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        if len(slug) < 5:
            slug = "run-" + (slug or "harness")
        if len(slug) > 50:
            slug = slug[:39].rstrip("-") + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]
        return slug

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
            license_name = "CC0-1.0"
            print("[ai-harness:kaggle] Notice: 'data_config.license' not specified in task spec; defaulting to 'CC0-1.0' for private dataset upload.", flush=True)
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
            with tempfile.TemporaryDirectory(prefix="aih-kaggle-dataset-") as temp_dir:
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
                    "--keep-tabular",
                ]
                logs.append(f"$ {' '.join(create_cmd)}")
                proc = subprocess.run(
                    create_cmd, capture_output=True, text=True, timeout=300
                )
                out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                logs.append(out.strip())

                if (
                    "already exists" in out.lower()
                    or "already in use" in out.lower()
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
                        "--keep-tabular",
                    ]
                    logs.append(f"$ {' '.join(ver_cmd)}")
                    proc_v = subprocess.run(
                        ver_cmd, capture_output=True, text=True, timeout=300
                    )
                    out_v = (proc_v.stdout or "") + "\n" + (proc_v.stderr or "")
                    logs.append(out_v.strip())
                    success = proc_v.returncode == 0 and "creation error:" not in out_v.lower()
                else:
                    # Some CLI releases print provider errors but still exit zero.
                    success = proc.returncode == 0 and "creation error:" not in out.lower()
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

    @staticmethod
    def wait_for_dataset(
        dataset_slug,
        *,
        poll_interval=10.0,
        timeout_seconds=600.0,
        max_consecutive_errors=5,
    ):
        """Dataset upload is asynchronous; dispatch only after processing is ready."""
        deadline = time.monotonic() + timeout_seconds
        last = "UNKNOWN"
        consecutive_errors = 0
        last_error = ""
        while time.monotonic() < deadline:
            proc = subprocess.run(
                ["kaggle", "datasets", "status", dataset_slug],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if proc.returncode != 0:
                consecutive_errors += 1
                last_error = output
                # Tolerate transient HTTP 403/404/500 or connection errors during initial asynchronous propagation
                if consecutive_errors < max_consecutive_errors and any(
                    code in output.lower()
                    for code in [
                        "403",
                        "404",
                        "500",
                        "502",
                        "503",
                        "forbidden",
                        "not found",
                        "connection",
                        "timeout",
                    ]
                ):
                    time.sleep(max(2.0, poll_interval))
                    continue
                raise RuntimeError(f"Kaggle dataset status failed: {output}")

            consecutive_errors = 0
            last = (proc.stdout or "").strip().strip('"').rsplit(".", 1)[-1].upper()
            if last == "READY":
                return
            if last in {"ERROR", "FAILED", "FAILURE"}:
                raise RuntimeError(f"Kaggle dataset processing failed: {output}")
            time.sleep(max(0.0, poll_interval))
        raise TimeoutError(
            f"Kaggle dataset did not become ready before timeout (last status: {last}, last error: {last_error})"
        )

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
        owner = username or "ai-harness-user"
        if "/" in kernel_slug:
            owner, kernel_slug = kernel_slug.split("/", 1)
        kernel_slug = cls._bounded_slug(kernel_slug)
        full_kernel_id = f"{owner}/{kernel_slug}"
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
        bundle_mode: str = "dataset",
    ) -> Path:
        """Stage a script and code dataset; embedded mode retains the legacy single-file upload."""
        from harness.project import HarnessProject

        if bundle_mode not in {"dataset", "embedded"}:
            raise ValueError("Kaggle bundle_mode must be 'dataset' or 'embedded'.")
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
        t_slug = cls._get_task_slug(project.task_config_path)
        slug = f"{t_slug}-{exp_id.replace('_', '-')}"
        if job_id:
            # Polling 'latest' must refer to this dispatch, not an older run of the same experiment.
            slug += "-" + hashlib.sha256(job_id.encode()).hexdigest()[:10]
        resolved_dataset = dataset_slug
        if not resolved_dataset or resolved_dataset == "antigravity/dataset":
            from harness.cloud_policy import private_upload_resource_names

            resources = private_upload_resource_names(project, kaggle_username=username)
            resolved_dataset = resources["kaggle_dataset_slug"]
            if not resolved_dataset:
                raise ValueError(
                    "Could not determine the authenticated Kaggle username. Run `kaggle auth login` and retry."
                )

        # Bundle only code/config; dataset contents use the separately consented provider dataset.
        import tempfile

        # Cache key: content of everything that goes into the bundle. Identical
        # bytes always produce identical bundle bytes (zip entries are timestamp-free
        # and deterministically ordered), so caching preserves the sha256 identity
        # that code_dataset reuse and the identity tests rely on.
        # Key on (config, task_config, src/, configs/) only — those are the inputs
        # that vary between test workspaces. .ai-harness/project.yaml, LICENSE and
        # runtime_manifest are workspace-stable across the tests that hit this code.
        bundle_cache_key = None
        if _BUNDLE_BYTES_CACHE is not None:
            try:
                tree_digest = hashlib.sha256()
                for tree in ("src", "configs"):
                    tree_dir = root / tree
                    if tree_dir.is_dir():
                        for p in sorted(tree_dir.rglob("*")):
                            if p.is_file():
                                tree_digest.update(p.name.encode("utf-8"))
                                tree_digest.update(p.read_bytes())
                bundle_cache_key = (
                    config_file.read_bytes(),
                    project.task_config_path.read_bytes(),
                    tree_digest.hexdigest(),
                    bundle_mode,
                )
            except Exception:
                bundle_cache_key = None
        cached_bytes = _BUNDLE_BYTES_CACHE.get(bundle_cache_key) if bundle_cache_key else None

        runtime = copy.deepcopy(runtime_manifest) if runtime_manifest else {
            "schema_version": 2,
            "core": {
                "distribution": "ai-harness",
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
        with tempfile.TemporaryDirectory(prefix="aih-kaggle-code-") as temp_stage:
            if cached_bytes is None:
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
                for source, destination in (
                    (root / "LICENSE", bundle_base / "LICENSE"),
                    (harness_source.parent / "LICENSE", bundle_base / "harness" / "LICENSE"),
                ):
                    if source.is_file():
                        shutil.copy2(source, destination)
                for manifest_name in (".ai-harness",):
                    manifest_source = root / manifest_name / "project.yaml"
                    if manifest_source.is_file():
                        manifest_dest = bundle_base / manifest_name / "project.yaml"
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
                (bundle_base / "ai-harness-runtime.json").write_text(
                    json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for file_path in sorted(bundle_base.rglob("*")):
                        if file_path.is_file():
                            # Stable bytes allow identical bundles to reuse a code dataset.
                            entry = zipfile.ZipInfo(file_path.relative_to(bundle_base).as_posix())
                            entry.compress_type = zipfile.ZIP_DEFLATED
                            entry.external_attr = 0o644 << 16
                            zf.writestr(entry, file_path.read_bytes())
                bundle_bytes = buf.getvalue()
            else:
                bundle_bytes = cached_bytes

        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        sources = [resolved_dataset] if resolved_dataset else []
        bundle_info = {"mode": bundle_mode, "sha256": bundle_sha256}
        if bundle_mode == "dataset":
            if not username:
                raise ValueError("Code bundle upload requires an authenticated Kaggle username.")
            code_slug = f"aih-code-{bundle_sha256[:32]}"
            code_dataset = f"{username}/{code_slug}"
            filename = f"aih-code-{bundle_sha256[:16]}.zip"
            code_stage = stage / "code" / bundle_sha256[:12]
            code_stage.mkdir(parents=True, exist_ok=True)
            (code_stage / filename).write_bytes(bundle_bytes)
            (code_stage / "dataset-metadata.json").write_text(json.dumps({
                "id": code_dataset, "title": f"AI Harness Code {bundle_sha256[:32]}",
                "licenses": [{"name": "other"}],
                "description": "Private execution bundle. Original source licenses and author rights are retained.",
            }, indent=2), encoding="utf-8")
            sources.append(code_dataset)
            bundle_info.update({"dataset_slug": code_dataset, "filename": filename,
                                "directory": f"code/{bundle_sha256[:12]}"})
            bundle_loader = f'''# Code/config are mounted as a private input, not embedded in this source.
bundle_name = {filename!r}
input_base = Path("/kaggle/input")
known_paths = [input_base / {code_slug!r} / bundle_name,
               input_base / "datasets" / {username!r} / {code_slug!r} / bundle_name]
matches = [path for path in known_paths if path.is_file()]
if not matches:
    matches = [path for path in input_base.rglob(bundle_name) if path.is_file()]
if len(matches) != 1:
    raise RuntimeError(f"Expected one mounted code bundle {{bundle_name}}, found {{len(matches)}}. Upload and attach the code dataset before pushing the kernel.")
bundle_bytes = matches[0].read_bytes()
if hashlib.sha256(bundle_bytes).hexdigest() != {bundle_sha256!r}:
    raise RuntimeError("Code bundle SHA-256 mismatch; refusing to execute a different revision.")
'''
        else:
            bundle_b64 = base64.b64encode(bundle_bytes).decode("ascii")
            bundle_loader = f"bundle_bytes = base64.b64decode({bundle_b64!r})\n"
        (stage / "code-bundle.json").write_text(json.dumps(bundle_info, indent=2), encoding="utf-8")
        meta = cls.generate_kernel_metadata(
            kernel_slug=slug, code_file="run_kaggle.py", dataset_sources=sources,
            enable_internet=enable_internet,
        )
        (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        task_id = (
            task_id or project.manifest.get("task_id") or project.plugin_id or "task"
        )
        artifact_dir = project.artifacts_dir.relative_to(root).as_posix()
        runner_content = f"""import base64
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import zipfile

print({('[Kaggle Kernel] Starting execution for: ' + exp_id)!r}, flush=True)
{bundle_loader}
output_root = Path.cwd().resolve()
# Kaggle saves /kaggle/working recursively. Keep code, data and the original
# run outside that directory, then publish exactly one sealed artifact set.
with tempfile.TemporaryDirectory(prefix="aih-kaggle-work-") as temporary:
    workspace = Path(temporary)
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        archive.extractall(workspace)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    try:
        # Loading by path avoids importing harness.__init__ before dependencies exist.
        helpers = runpy.run_path(str(workspace / "harness" / "kaggle_runtime.py"))
        runtime_path = Path("ai-harness-runtime.json")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        helpers["ensure_dependencies"](runtime, enable_internet={enable_internet!r})
        input_root = helpers["find_dataset_mount"](Path("/kaggle/input"), {resolved_dataset!r})
        print(f"[Kaggle Kernel] Using input_root: {{input_root}}", flush=True)
        helpers["restore_dataset"](input_root, workspace)
        cmd = [sys.executable, "-u", "-m", "harness.runner", "--project-root", ".",
               "--task-config", {task_config_rel!r}, "--config", {config_rel!r}]
        _remote_job_id = {job_id!r}
        if _remote_job_id:
            cmd += ["--job-id", _remote_job_id]
        print(f"Executing: {{' '.join(cmd)}}", flush=True)
        subprocess.run(cmd, check=True)
        helpers["export_run"](Path({artifact_dir!r}) / {exp_id!r}, output_root, {{
            "job_id": {job_id!r}, "task_id": {str(task_id)!r}, "exp_id": {exp_id!r},
            "config_sha256": {config_sha256!r},
            "core_version": (runtime.get("core") or {{}}).get("version"),
            "plugin_id": (runtime.get("plugin") or {{}}).get("id"),
        }})
    finally:
        os.chdir(output_root)
        sys.path.remove(str(workspace))
print("Training completed. Sealed output artifacts copied to working root.", flush=True)
"""
        (stage / "run_kaggle.py").write_text(runner_content, encoding="utf-8")
        return stage

    @classmethod
    def push_code_bundle(cls, package_dir: Path | str) -> Dict[str, Any]:
        """Upload a private, content-addressed code dataset before the kernel push."""
        stage = Path(package_dir)
        bundle = json.loads((stage / "code-bundle.json").read_text(encoding="utf-8"))
        if bundle["mode"] == "embedded":
            return {"success": True, "mode": "embedded"}
        code_stage = (stage / bundle["directory"]).resolve()
        if not code_stage.is_relative_to(stage.resolve()):
            raise RuntimeError("Code bundle directory must be inside the staged package.")
        archive = code_stage / bundle["filename"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != bundle["sha256"]:
            raise RuntimeError("Staged code bundle SHA-256 mismatch; refusing upload.")
        metadata = json.loads((code_stage / "dataset-metadata.json").read_text(encoding="utf-8"))
        kernel = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
        if metadata["id"] != bundle["dataset_slug"] or metadata["id"] not in kernel["dataset_sources"]:
            raise RuntimeError("Code dataset identity does not match the staged kernel.")
        proc = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(code_stage), "--dir-mode", "skip", "--keep-tabular"],
            capture_output=True, text=True, timeout=300,
        )
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        lowered = output.lower()
        reused = any(message in lowered for message in ("already exists", "already in use")) or bool(
            re.search(r"\b409\b", lowered)
        )
        if re.search(r"\b(?:401|403)\b", lowered) or (
            (proc.returncode != 0 or "creation error:" in lowered) and not reused
        ):
            raise RuntimeError(f"Kaggle private code bundle upload failed: {output}")
        # Never version/overwrite a content-addressed code dataset, even on reuse.
        cls.wait_for_dataset(bundle["dataset_slug"])
        return {"success": True, "dataset_slug": bundle["dataset_slug"], "reused": reused}

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
            # Kernel names and failure messages can contain 'error', 'running', etc.
            # Parse only the provider's status field, not every word in stdout/stderr.
            match = re.search(r'has status\s+["\x27]([^"\x27]+)["\x27]', proc.stdout or "", re.I)
            state = match.group(1).rsplit(".", 1)[-1].upper() if match else (proc.stdout or "").strip().upper()
            if proc.returncode != 0:
                status_state = "ERROR"
            elif state in {"FAILED", "FAILURE", "ERROR", "INVALID"}:
                status_state = "FAILED"
            elif state in {"CANCELLED", "CANCELED", "CANCEL"}:
                status_state = "CANCELLED"
            elif state == "QUEUED":
                status_state = "QUEUED"
            elif state == "RUNNING":
                status_state = "RUNNING"
            elif state in {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}:
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
    def get_kernel_error_log(cls, kernel_slug: str, max_lines: int = 60) -> Optional[str]:
        """Download remote kernel log and extract failure traceback/stderr chunks."""
        try:
            with tempfile.TemporaryDirectory(prefix="aih-klog-") as temp_dir:
                proc = subprocess.run(
                    ["kaggle", "kernels", "output", kernel_slug, "-p", temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                log_files = list(Path(temp_dir).glob("*.log"))
                if not log_files:
                    return None

                stderr_chunks = []
                all_lines = []
                for log_file in log_files:
                    try:
                        content = log_file.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    for line in content.splitlines():
                        line_str = line.strip()
                        if line_str.startswith(",") or line_str.startswith("["):
                            line_str = line_str.lstrip("[,").strip()
                        if line_str.endswith("]"):
                            line_str = line_str.rstrip("]").strip()
                        if not line_str:
                            continue
                        try:
                            entry = json.loads(line_str)
                            stream = entry.get("stream_name")
                            data = entry.get("data", "")
                            if stream == "stderr":
                                stderr_chunks.append(data)
                            all_lines.append(data)
                        except Exception:
                            all_lines.append(line)
                            if "traceback" in line.lower() or "error" in line.lower():
                                stderr_chunks.append(line)

                if stderr_chunks:
                    full_stderr = "".join(stderr_chunks).strip()
                    lines = full_stderr.splitlines()
                    return "\n".join(lines[-max_lines:])
                elif all_lines:
                    full_out = "".join(all_lines).strip()
                    lines = full_out.splitlines()
                    return "\n".join(lines[-max_lines:])
                return None
        except Exception as exc:
            return f"(Could not retrieve remote kernel log: {exc})"

    @classmethod
    def _download_output_api(cls, slug, download_dir):
        """Use the public API's pagination and HTTP checks for root-level run output."""
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        # Root exports have no experiment ID in their filenames. Filter by layout,
        # not by ID, and avoid old nested source/data copies and duplicate manifests.
        api.kernels_output(slug, path=str(download_dir), file_pattern=r"^[^/\\]+$", quiet=True)

    @classmethod
    def pull_kernel_output(
        cls,
        exp_id: str,
        kernel_slug: Optional[str] = None,
        *,
        artifact_dir: Optional[Path | str] = None,
        expected_identity: Optional[Dict[str, Any]] = None,
        project_root: Optional[Path | str] = None,
        force_import: bool = False,
    ) -> Dict[str, Any]:
        """Download outputs only after artifact identity matches the submitted task and run."""
        username = cls.get_authenticated_username()
        t_slug = cls._get_task_slug()
        slug = kernel_slug or f"{username}/{t_slug}-{exp_id.replace('_', '-')}"
        if "/" not in slug:
            slug = f"{username}/{slug}"

        from harness.project import HarnessProject

        project = HarnessProject.discover(project_root)
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
                prefix=f"aih-kaggle-{exp_id}-"
            ) as temp_dir:
                download_dir = Path(temp_dir)
                try:
                    cls._download_output_api(slug, download_dir)
                    output = "Downloaded via KaggleApi"
                except (Exception, SystemExit) as api_error:
                    # The CLI may be installed in pipx/uv while its SDK is absent here.
                    # Discard any partial API files before the independent CLI retry.
                    shutil.rmtree(download_dir)
                    download_dir.mkdir()
                    proc = subprocess.run(
                        ["kaggle", "kernels", "output", slug, "-p", str(download_dir)],
                        capture_output=True, text=True, timeout=300,
                    )
                    output = f"Direct API download failed ({api_error}), falling back to CLI\n"
                    output += ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                    if proc.returncode != 0:
                        return {"success": False, "kernel_slug": slug, "exp_id": exp_id, "raw_output": output}
                manifests = list(download_dir.rglob("artifact_manifest.json"))
                if len(manifests) != 1:
                    return {"success": False, "kernel_slug": slug, "exp_id": exp_id,
                            "raw_output": "Remote import requires exactly one manifest v2"}
                manifest_path = manifests[0]
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
                    critical_mismatches = {
                        k: v
                        for k, v in mismatches.items()
                        if not _is_critical_identity_mismatch(k, v)
                    }
                    if mismatches:
                        if force_import:
                            import logging
                            logging.getLogger("ai-harness").warning(
                                f"Overriding identity mismatches due to --force-import: {mismatches}"
                            )
                        elif not critical_mismatches:
                            import logging
                            non_critical = ", ".join(sorted(mismatches))
                            logging.getLogger("ai-harness").warning(
                                f"Kaggle artifact identity mismatch on non-critical field(s) "
                                f"[{non_critical}]; the remote kernel did not report a concrete "
                                f"conflicting value. Proceeding with import."
                            )
                        else:
                            return {
                                "success": False,
                                "kernel_slug": slug,
                                "exp_id": exp_id,
                                "identity_mismatches": mismatches,
                                "raw_output": f"Downloaded Kaggle artifacts do not match this task/config/job: {critical_mismatches}. Use --force-import to override.",
                            }
                    for file_name, expected_hash in (
                        manifest.get("files") or {}
                    ).items():
                        file_path = manifest_path.parent / file_name
                        if (
                            Path(file_name).name != file_name
                            or not file_path.is_file()
                            or hashlib.sha256(file_path.read_bytes()).hexdigest()
                            != expected_hash
                        ):
                            if force_import:
                                import logging
                                logging.getLogger("ai-harness").warning(
                                    f"File {file_name} failed hash check; proceeding due to --force-import."
                                )
                            else:
                                return {
                                    "success": False,
                                    "kernel_slug": slug,
                                    "exp_id": exp_id,
                                    "invalid_artifact": file_name,
                                    "raw_output": "Downloaded Kaggle artifact failed its SHA-256 identity check.",
                                }
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
                success = bool(migrated) and metrics_data is not None
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


def main(argv=None):
    """Push an already generated package, including its required code input."""
    import argparse

    parser = argparse.ArgumentParser(description="Push a staged Kaggle script and code bundle.")
    parser.add_argument("command", choices=["push-package"])
    parser.add_argument("--path", required=True, type=Path)
    args = parser.parse_args(argv)
    KaggleKernelManager.push_code_bundle(args.path)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(args.path)], check=True)


if __name__ == "__main__":
    main()
