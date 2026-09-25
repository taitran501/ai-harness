"""Kaggle Kernel Runner, Dataset Sync, Status Monitor, and Artifact Puller.

Enables complete headless execution via Kaggle API:
1. `push_dataset_to_kaggle`: Packages `dataset/` + `splits/` into a private Kaggle Dataset once (`kaggle datasets create` / `version`).
2. `setup_kernel_package`: Generates `kernel-metadata.json` with attached `dataset_sources` and packs lightweight code (`train.py`, `src/`, `configs/`).
3. `get_kernel_status`: Polls `kaggle kernels status <slug>` to monitor running/completed state.
4. `pull_kernel_output`: Runs `kaggle kernels output <slug>` and migrates lightweight prediction/metric files (`oof_predictions.csv`, `test_predictions.csv`, `metrics.json`, `run.log`, `EXPERIMENT_CARD.md`) directly into `artifacts/<exp_id>/`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        return "antigravity"

    @classmethod
    def push_dataset_to_kaggle(cls, dataset_slug: Optional[str] = None) -> Dict[str, Any]:
        """Package `dataset/` and `splits/` and push to Kaggle Datasets (create or new version)."""
        username = cls.get_authenticated_username()
        if not dataset_slug or "/" not in dataset_slug:
            slug_name = (dataset_slug or "sqe-competition-dataset").split("/")[-1]
            slug_name = slug_name.replace("_", "-").lower()
            full_slug = f"{username}/{slug_name}"
        else:
            full_slug = dataset_slug
            slug_name = full_slug.split("/", 1)[1]

        stage_dir = Path("tmp") / "kaggle_dataset_sync"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Copy dataset/ and splits/ into staging folder
        if Path("dataset").exists():
            shutil.copytree("dataset", stage_dir / "dataset", dirs_exist_ok=True)
        if Path("splits").exists():
            shutil.copytree("splits", stage_dir / "splits", dirs_exist_ok=True)

        metadata = {
            "title": f"SQE Competition Dataset ({slug_name})",
            "id": full_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (stage_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        logs = []
        # Try creating dataset first; if it already exists, push a new version
        create_cmd = ["kaggle", "datasets", "create", "-p", str(stage_dir), "--dir-mode", "zip"]
        logs.append(f"$ {' '.join(create_cmd)}")
        proc = subprocess.run(create_cmd, capture_output=True, text=True, timeout=300)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        logs.append(out.strip())

        if proc.returncode != 0 or "already exists" in out.lower() or "409" in out.lower():
            ver_cmd = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(stage_dir),
                "-m",
                "Sync dataset and CV splits",
                "--dir-mode",
                "zip",
            ]
            logs.append(f"$ {' '.join(ver_cmd)}")
            proc_v = subprocess.run(ver_cmd, capture_output=True, text=True, timeout=300)
            out_v = (proc_v.stdout or "") + "\n" + (proc_v.stderr or "")
            logs.append(out_v.strip())
            success = proc_v.returncode == 0
        else:
            success = True

        return {
            "success": success,
            "dataset_slug": full_slug,
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
        full_kernel_id = kernel_slug if "/" in kernel_slug else f"{username}/{kernel_slug}"
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
    ) -> Path:
        """Create self-contained staging directory for Kaggle kernel push with code + mounted dataset."""
        stage = Path(target_dir)
        stage.mkdir(parents=True, exist_ok=True)

        username = cls.get_authenticated_username()
        slug = f"sqe-{exp_id.replace('_', '-')}"
        resolved_dataset = dataset_slug
        if not resolved_dataset or resolved_dataset == "antigravity/dataset":
            resolved_dataset = f"{username}/sqe-competition-dataset"

        meta = cls.generate_kernel_metadata(
            kernel_slug=slug,
            code_file="run_kaggle.py",
            dataset_sources=[resolved_dataset] if resolved_dataset else [],
        )
        with open(stage / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Bundle lightweight project code (train.py, src/, configs/) into code_bundle.zip inside the kernel directory
        bundle_base = stage / "code_bundle_staging"
        if bundle_base.exists():
            shutil.rmtree(bundle_base)
        bundle_base.mkdir(parents=True, exist_ok=True)

        if Path("train.py").exists():
            shutil.copy2("train.py", bundle_base / "train.py")
        if Path("src").exists():
            shutil.copytree("src", bundle_base / "src", dirs_exist_ok=True)
        if Path("configs").exists():
            shutil.copytree("configs", bundle_base / "configs", dirs_exist_ok=True)
        if Path("splits").exists():
            shutil.copytree("splits", bundle_base / "splits", dirs_exist_ok=True)

        shutil.make_archive(str(stage / "code_bundle"), "zip", root_dir=bundle_base)
        shutil.rmtree(bundle_base, ignore_errors=True)

        dataset_folder_name = resolved_dataset.split("/")[-1]

        runner_content = f'''import subprocess
import sys
import os
import shutil
import zipfile
from pathlib import Path

print("[Kaggle Kernel] Starting execution for: {exp_id}")

# 1. Extract code bundle if present
if Path("code_bundle.zip").exists():
    with zipfile.ZipFile("code_bundle.zip", "r") as zf:
        zf.extractall(".")

# 2. Mount dataset/ and splits/ from /kaggle/input/{dataset_folder_name}
input_root = Path("/kaggle/input/{dataset_folder_name}")
if not input_root.exists():
    # Fallback scan in /kaggle/input
    for candidate in Path("/kaggle/input").glob("*"):
        if (candidate / "dataset").exists() or any(candidate.glob("*.csv")) or any(candidate.glob("*.tsv")):
            input_root = candidate
            break

if (input_root / "dataset").exists() and not Path("dataset").exists():
    shutil.copytree(input_root / "dataset", "dataset", dirs_exist_ok=True)
elif input_root.exists() and not Path("dataset").exists():
    shutil.copytree(input_root, "dataset", dirs_exist_ok=True)

if (input_root / "splits").exists() and not Path("splits").exists():
    shutil.copytree(input_root / "splits", "splits", dirs_exist_ok=True)

# 3. Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn>=1.3.0"])

# 4. Run training
cmd = [sys.executable, "train.py", "--config", "{config_path}"]
print(f"Executing: {{' '.join(cmd)}}")
ret = subprocess.run(cmd)

if ret.returncode != 0:
    print(f"Execution failed with returncode {{ret.returncode}}", file=sys.stderr)
    sys.exit(ret.returncode)

# 5. Copy lightweight output artifacts to /kaggle/working root so `kaggle kernels output` downloads them cleanly
exp_dir = Path("artifacts") / "{exp_id}"
essential_files = [
    "metrics.json",
    "summary.csv",
    "summary.md",
    "oof_predictions.csv",
    "test_predictions.csv",
    "submission.csv",
    "DIAGNOSTICS.md",
    "diagnostics.json",
    "run.log",
    "EXPERIMENT_CARD.md",
    "run_config.yaml",
]
if exp_dir.exists():
    for fname in essential_files:
        src_file = exp_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, Path(".") / fname)

print("Training completed. Lightweight output artifacts copied to working root.")
'''
        (stage / "run_kaggle.py").write_text(runner_content, encoding="utf-8")
        return stage

    @classmethod
    def get_kernel_status(cls, exp_id: str, kernel_slug: Optional[str] = None) -> Dict[str, Any]:
        """Check status of a Kaggle kernel via `kaggle kernels status <slug>`."""
        username = cls.get_authenticated_username()
        slug = kernel_slug or f"{username}/sqe-{exp_id.replace('_', '-')}"
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
            if "complete" in lower_out:
                status_state = "COMPLETED"
            elif "running" in lower_out or "queued" in lower_out:
                status_state = "RUNNING"
            elif "error" in lower_out or "cancel" in lower_out:
                status_state = "FAILED"

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
    def pull_kernel_output(cls, exp_id: str, kernel_slug: Optional[str] = None) -> Dict[str, Any]:
        """Download lightweight artifacts from Kaggle Kernel into `artifacts/<exp_id>/`."""
        username = cls.get_authenticated_username()
        slug = kernel_slug or f"{username}/sqe-{exp_id.replace('_', '-')}"
        if "/" not in slug:
            slug = f"{username}/{slug}"

        download_dir = Path("tmp") / "kaggle_outputs" / exp_id
        if download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        target_artifacts_dir = Path("artifacts") / exp_id
        target_artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "output", slug, "-p", str(download_dir)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            migrated = cls.migrate_downloaded_artifacts(download_dir, target_artifacts_dir)

            # Also read metrics.json if downloaded so UI can display CV scores immediately
            metrics_data = None
            metrics_path = target_artifacts_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            return {
                "success": len(migrated) > 0 or proc.returncode == 0,
                "kernel_slug": slug,
                "exp_id": exp_id,
                "artifacts_dir": str(target_artifacts_dir),
                "migrated_files": migrated,
                "metrics": metrics_data,
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
        """Filter downloaded outputs and migrate only lightweight essential files into artifacts/<exp_id>/."""
        src_d = Path(download_dir)
        dest_d = Path(target_artifacts_dir)
        dest_d.mkdir(parents=True, exist_ok=True)

        essential_filenames = {
            "metrics.json",
            "summary.csv",
            "summary.md",
            "oof_predictions.csv",
            "test_predictions.csv",
            "submission.csv",
            "DIAGNOSTICS.md",
            "diagnostics.json",
            "run.log",
            "EXPERIMENT_CARD.md",
            "run_config.yaml",
        }

        migrated = []
        for root, _, files in os.walk(src_d):
            for fname in files:
                if fname in essential_filenames or fname.endswith(".log"):
                    source_p = Path(root) / fname
                    target_name = "run.log" if fname.endswith(".log") and fname != "run.log" else fname
                    target_p = dest_d / target_name
                    shutil.copy2(source_p, target_p)
                    if target_name not in migrated:
                        migrated.append(target_name)

        return migrated
