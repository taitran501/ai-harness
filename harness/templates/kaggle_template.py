"""Kaggle Kernel Runner and Artifact Migrator Template.

Enables headless execution via Kaggle Kernels API:
1. Generates kernel-metadata.json with GPU acceleration (T4 x 2 or P100).
2. Pushes self-contained execution bundle via `kaggle kernels push`.
3. Monitors execution progress until completion.
4. Downloads outputs and migrates essential artifacts to `artifacts/<exp_id>/`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class KaggleKernelManager:
    """Manages Kaggle Kernel dispatch, polling, and artifact migration."""

    @staticmethod
    def generate_kernel_metadata(
        kernel_slug: str,
        code_file: str = "run_kaggle.py",
        enable_gpu: bool = True,
        enable_internet: bool = True,
    ) -> Dict[str, Any]:
        """Generate kernel-metadata.json configuration dictionary."""
        return {
            "id": f"antigravity/{kernel_slug}",
            "title": kernel_slug.replace("-", " ").title(),
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "true" if enable_internet else "false",
            "dataset_sources": [],
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
    ) -> Path:
        """Create self-contained staging directory for Kaggle kernel push."""
        stage = Path(target_dir)
        stage.mkdir(parents=True, exist_ok=True)

        # 1. Metadata
        slug = f"sqe-{exp_id.replace('_', '-')}"
        meta = cls.generate_kernel_metadata(kernel_slug=slug, code_file="run_kaggle.py")
        with open(stage / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 2. Runner Script
        runner_content = f'''import subprocess
import sys
import os
from pathlib import Path

print("🚀 [Kaggle Kernel] Starting execution for: {exp_id}")

# Install any extra dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn>=1.3.0"])

# Run training
cmd = [sys.executable, "train.py", "--config", "{config_path}"]
print(f"Executing: {{' '.join(cmd)}}")
ret = subprocess.run(cmd)

if ret.returncode != 0:
    print(f"❌ Execution failed with returncode {{ret.returncode}}", file=sys.stderr)
    sys.exit(ret.returncode)

print("✔ Training completed successfully. Output files saved in working directory.")
'''
        (stage / "run_kaggle.py").write_text(runner_content, encoding="utf-8")
        return stage

    @staticmethod
    def migrate_downloaded_artifacts(
        download_dir: Path | str,
        target_artifacts_dir: Path | str,
    ) -> List[str]:
        """Filter downloaded outputs and migrate only essential files into artifacts/<exp_id>/."""
        src_d = Path(download_dir)
        dest_d = Path(target_artifacts_dir)
        dest_d.mkdir(parents=True, exist_ok=True)

        essential_filenames = {
            "metrics.json",
            "summary.csv",
            "summary.md",
            "oof_predictions.csv",
            "DIAGNOSTICS.md",
            "diagnostics.json",
            "run.log",
            "EXPERIMENT_CARD.md",
            "run_config.yaml",
        }

        migrated = []
        for root, _, files in os.walk(src_d):
            for fname in files:
                if fname in essential_filenames:
                    source_p = Path(root) / fname
                    target_p = dest_d / fname
                    shutil.copy2(source_p, target_p)
                    migrated.append(fname)

        return migrated
