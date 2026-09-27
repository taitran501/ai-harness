"""Cloud Runner Bridge for Heavy ML Experiments.

Provides headless execution interfaces for:
1. Modal.com (Serverless Cloud GPU: A10G, A100, H100 with live log streaming).
2. Kaggle Kernels API (Free 30h GPU quota).

Features:
- Mandatory local Pre-flight Inspection (schema, datasets, syntax) to prevent costly cloud failures.
- Automatic artifact pulling and migration to artifacts/<exp_id>/.
- Optional --auto-submit to directly register verified submission package upon completion.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cloud_inspector import CloudPreflightInspector
from harness.submission_manager import SubmissionManager
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script


def run_via_modal(
    config_path: str,
    gpu: str = "A10G",
    auto_submit: bool = False,
    baseline_csv: Optional[str] = None,
) -> None:
    """Run training remotely on Modal.com serverless GPU infrastructure."""
    print(f"===========================================================")
    print(f"🚀 MODAL.COM CLOUD DISPATCH: {config_path}")
    print(f"===========================================================")

    # 1. Pre-flight Check
    passed, errors, cfg = CloudPreflightInspector.run_full_preflight_check(config_path)
    if not passed:
        print("❌ [DISPATCH ABORTED] Pre-flight inspection failed:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        sys.exit(1)

    exp_id = cfg.get("experiment", {}).get("id", Path(config_path).stem)

    try:
        import modal
    except ImportError:
        print(
            "❌ Modal is not installed. Run `pip install modal` and `modal setup` to authenticate.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Generate and write staging modal app
    stage_dir = Path("tmp") / "modal"
    stage_dir.mkdir(parents=True, exist_ok=True)
    modal_script_path = stage_dir / f"app_{exp_id}.py"

    code_content = generate_modal_script(config_path, gpu=gpu, exp_id=exp_id)
    modal_script_path.write_text(code_content, encoding="utf-8")

    # 3. Execute Modal Run
    cmd = ["modal", "run", str(modal_script_path)]
    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd)

    if ret.returncode != 0:
        print(f"❌ Modal execution failed with code {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

    print(f"\n✔ Modal remote execution finished successfully.")

    # 4. Optional Auto-Submit
    if auto_submit:
        handle_auto_submit(exp_id, baseline_csv)


def run_via_kaggle(
    config_path: str,
    auto_submit: bool = False,
    baseline_csv: Optional[str] = None,
) -> None:
    """Push and run execution via Kaggle Kernels API."""
    print(f"===========================================================")
    print(f"🚀 KAGGLE KERNELS API DISPATCH: {config_path}")
    print(f"===========================================================")

    # 1. Pre-flight Check
    passed, errors, cfg = CloudPreflightInspector.run_full_preflight_check(config_path)
    if not passed:
        print("❌ [DISPATCH ABORTED] Pre-flight inspection failed:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        sys.exit(1)

    exp_id = cfg.get("experiment", {}).get("id", Path(config_path).stem)

    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle API not installed. Run `pip install kaggle`.", file=sys.stderr)
        sys.exit(1)

    # 2. Package Kernel Staging
    stage_dir = Path("tmp") / "kaggle" / exp_id
    KaggleKernelManager.setup_kernel_package(stage_dir, config_path, exp_id)

    # 3. Push to Kaggle
    cmd = ["kaggle", "kernels", "push", "-p", str(stage_dir)]
    print(f"Pushing kernel: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"❌ Kaggle push failed with code {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

    print(f"✔ Kernel pushed. Staging directory: {stage_dir}")
    print(f"Monitor status with: `kaggle kernels status antigravity/sqe-{exp_id.replace('_', '-')}`")

    # 4. Optional Auto-Submit
    if auto_submit:
        handle_auto_submit(exp_id, baseline_csv)


def handle_auto_submit(exp_id: str, baseline_csv: Optional[str] = None) -> None:
    """Register verified submission if predictions.csv or oof_predictions.csv exists."""
    pred_path = Path("artifacts") / exp_id / "predictions.csv"
    if not pred_path.exists():
        pred_path = Path("artifacts") / exp_id / "oof_predictions.csv"
    if not pred_path.exists():
        print(f"⚠ Auto-submit skipped: neither predictions.csv nor oof_predictions.csv found on local disk.", file=sys.stderr)
        return

    print(f"\n--- Initiating Auto-Submit for '{exp_id}' ---")
    mgr = SubmissionManager()
    try:
        entry = mgr.register_submission(
            exp_id=exp_id,
            candidate_csv=str(pred_path),
            baseline_csv=baseline_csv,
            notes="Auto-submitted after cloud execution",
        )
        print(f"🎉 Auto-Submit Succeeded! Submission ID: {entry.id}")
        print(f"Registered at: submissions/SUBMISSIONS.md")
    except Exception as e:
        print(f"❌ Auto-submit failed: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-Harness Cloud GPU Dispatcher")
    parser.add_argument("--backend", choices=["modal", "kaggle"], default="modal")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--gpu", default="A10G", help="Modal GPU type (A10G, A100, T4)")
    parser.add_argument(
        "--auto-submit",
        action="store_true",
        help="Automatically package and register submission upon successful run",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline predictions for auto-submit diff check",
    )

    args = parser.parse_args()
    if args.backend == "modal":
        run_via_modal(args.config, gpu=args.gpu, auto_submit=args.auto_submit, baseline_csv=args.baseline)
    elif args.backend == "kaggle":
        run_via_kaggle(args.config, auto_submit=args.auto_submit, baseline_csv=args.baseline)


if __name__ == "__main__":
    main()
