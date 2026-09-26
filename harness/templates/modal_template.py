"""Modal.com Serverless GPU Runner Template.

Enables remote training on A10G / A100 / H100 with:
1. Dataset Volume syncing (`modal volume put`) so dataset/ and splits/ are stored once on Modal.
2. Live log streaming to the local terminal.
3. Automatic execution of `run_training_remote.remote()` in `@app.local_entrypoint()`.
4. Selective lightweight artifact download (`oof_predictions.csv`, `test_predictions.csv`, `metrics.json`, `summary.md`, `run.log`, `EXPERIMENT_CARD.md`) directly into `artifacts/<exp_id>/` without downloading multi-GB model checkpoints.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Any


def push_dataset_to_modal(volume_name: Optional[str] = None) -> Dict[str, Any]:
    """Create Modal Volume if needed and push dataset/ and splits/ to the volume."""
    if volume_name is None:
        from harness.task_spec import load_active_task_spec
        spec = load_active_task_spec()
        t_slug = "".join(c if c.isalnum() else "_" for c in spec.name.lower()).strip("_") or "harness"
        volume_name = f"{t_slug}-dataset-vol"

    logs = []
    try:
        # 1. Ensure volume exists
        create_proc = subprocess.run(
            ["modal", "volume", "create", volume_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        logs.append(f"$ modal volume create {volume_name}")
        if create_proc.stdout:
            logs.append(create_proc.stdout.strip())

        # 2. Upload dataset/ directory if present
        if Path("dataset").exists():
            put_data = subprocess.run(
                ["modal", "volume", "put", volume_name, "dataset", "/dataset", "--force"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            logs.append(f"$ modal volume put {volume_name} dataset /dataset --force")
            if put_data.returncode != 0 and put_data.stderr:
                logs.append(put_data.stderr.strip())
            else:
                logs.append("Uploaded dataset/ to Modal volume /dataset.")

        # 3. Upload splits/ directory if present
        if Path("splits").exists():
            put_splits = subprocess.run(
                ["modal", "volume", "put", volume_name, "splits", "/splits", "--force"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            logs.append(f"$ modal volume put {volume_name} splits /splits --force")
            if put_splits.returncode != 0 and put_splits.stderr:
                logs.append(put_splits.stderr.strip())
            else:
                logs.append("Uploaded splits/ to Modal volume /splits.")

        return {
            "success": True,
            "volume_name": volume_name,
            "logs": "\n".join(logs),
        }
    except Exception as exc:
        return {
            "success": False,
            "volume_name": volume_name,
            "error": str(exc),
            "logs": "\n".join(logs),
        }


def generate_modal_script(
    config_path: str,
    gpu: str = "A10G",
    exp_id: str = "remote_exp",
    volume_name: Optional[str] = None,
) -> str:
    """Generate self-contained Modal app script that runs training and pulls lightweight artifacts back to local disk."""
    from harness.task_spec import load_active_task_spec
    spec = load_active_task_spec()
    t_slug = "".join(c if c.isalnum() else "_" for c in spec.name.lower()).strip("_") or "harness"
    if volume_name is None:
        volume_name = f"{t_slug}-dataset-vol"
    app_name = f"{t_slug}-{exp_id}"
    cache_name = f"{t_slug}-hf-cache"

    return f'''import modal
import os
import shutil
import subprocess
from pathlib import Path

app = modal.App("{app_name}")

# 1. Container Image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "accelerate>=0.28.0",
        "scipy>=1.11.0",
    )
)

# 2. Persistent Volumes for HuggingFace cache and Dataset
hf_cache = modal.Volume.from_name("{cache_name}", create_if_missing=True)
data_volume = modal.Volume.from_name("{volume_name}", create_if_missing=True)

# 3. Mount lightweight local code & config
project_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/root/workspace",
    condition=lambda p: not any(
        x in p for x in [".git", "tmp", "__pycache__", "venv", ".venv", "artifacts", "node_modules", "dashboard"]
    ),
)

@app.function(
    image=image,
    gpu="{gpu}",
    timeout=14400,
    mounts=[project_mount],
    volumes={{
        "/root/.cache/huggingface": hf_cache,
        "/mnt/cloud_data": data_volume,
    }},
)
def run_training_remote(config_path: str, exp_id: str):
    import os
    os.chdir("/root/workspace")

    # Symlink dataset/ and splits/ from Modal Volume if not mounted directly
    if Path("/mnt/cloud_data/dataset").exists() and not Path("dataset").exists():
        os.symlink("/mnt/cloud_data/dataset", "dataset")
    if Path("/mnt/cloud_data/splits").exists() and not Path("splits").exists():
        os.symlink("/mnt/cloud_data/splits", "splits")

    print(f"[Modal Remote] Starting GPU training for experiment: {{exp_id}} on {gpu}")
    print(f"[Modal Remote] Config: {{config_path}}")

    cmd = ["python", "train.py", "--config", config_path]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")

    process.stdout.close()
    ret = process.wait()

    if ret != 0:
        raise RuntimeError(f"Remote training failed with exit code: {{ret}}")

    # Package lightweight output artifacts (no heavy .safetensors / .bin weights)
    exp_dir = Path("artifacts") / exp_id
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

    bundle = {{}}
    for fname in essential_files:
        p = exp_dir / fname
        if p.exists():
            bundle[fname] = p.read_bytes()

    print(f"[Modal Remote] Returning {{len(bundle)}} lightweight artifact files to local host.")
    return bundle

@app.local_entrypoint()
def main():
    exp_id = "{exp_id}"
    config_path = "{config_path}"
    print(f"Dispatching remote job '{{exp_id}}' to Modal GPU ({gpu})...")
    bundle = run_training_remote.remote(config_path, exp_id)

    local_exp_dir = Path("artifacts") / exp_id
    local_exp_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for fname, content in (bundle or {{}}).items():
        out_path = local_exp_dir / fname
        out_path.write_bytes(content)
        saved_files.append(fname)

    print(f"Pulled {{len(saved_files)}} artifacts into {{local_exp_dir}}: {{', '.join(saved_files)}}")
'''
