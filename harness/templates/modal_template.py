"""Modal.com Serverless GPU Runner Template.

Enables remote training on A10G / A100 / H100 with:
1. Live log streaming to the local terminal.
2. Direct mounting of local codebase and datasets.
3. HuggingFace model cache volume to prevent redownloading weights.
4. Selective artifact download (only oof_predictions, metrics, summary, diagnostics).
"""

from __future__ import annotations

import os
from pathlib import Path


def generate_modal_script(config_path: str, gpu: str = "A10G", exp_id: str = "remote_exp") -> str:
    """Generate self-contained Modal app script."""
    return f'''import modal
import os
import shutil
import subprocess
from pathlib import Path

app = modal.App("stereoqueer-{exp_id}")

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

# 2. Cache volume for HuggingFace pre-trained weights
hf_cache = modal.Volume.from_name("stereoqueer-hf-cache", create_if_missing=True)

# 3. Mount local project directory
project_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/root/workspace",
    condition=lambda p: not any(
        x in p for x in [".git", "tmp", "__pycache__", "venv", ".venv", "artifacts"]
    ),
)

@app.function(
    image=image,
    gpu="{gpu}",
    timeout=14400,
    mounts=[project_mount],
    volumes={{"/root/.cache/huggingface": hf_cache}},
)
def run_training_remote(config_path: str, exp_id: str):
    import os
    os.chdir("/root/workspace")
    print(f"🚀 [Modal Remote] Starting training for experiment: {{exp_id}}")
    print(f"🚀 [Modal Remote] Config: {{config_path}}")

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

    # Read essential artifacts into memory dictionary to return to local machine
    exp_dir = Path("artifacts") / exp_id
    essential_files = [
        "metrics.json",
        "summary.csv",
        "summary.md",
        "oof_predictions.csv",
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

    print(f"✔ [Modal Remote] Packaged {{len(bundle)}} essential artifact files.")
    return bundle

@app.local_entrypoint()
def main():
    print("Remote execution complete.")
'''
