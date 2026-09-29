#!/usr/bin/env python3
"""Unified training and evaluation CLI runner for competition experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

HARNESS_DIR = Path(__file__).resolve().parent / "ml-harness"
if HARNESS_DIR.exists() and str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from harness.runner import UniversalRunner, main as runner_main, merge_configs
from harness.io_utils import load_yaml_config


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute training run via UniversalRunner."""
    base_cfg = load_yaml_config(args.base_config) if Path(args.base_config).exists() else {}
    exp_cfg = load_yaml_config(args.config)
    cfg = merge_configs(base_cfg, exp_cfg)

    # Immutability Guard
    exp_info = cfg.get("experiment", {})
    exp_id = exp_info.get("id", Path(args.config).stem)
    paths = cfg.get("paths", {})
    artifacts_root = Path(paths.get("artifacts_dir", "artifacts"))
    exp_dir = artifacts_root / exp_id

    if exp_dir.exists() and (exp_dir / "metrics.json").exists() and not getattr(args, "force", False):
        print(
            f"ERROR: Experiment directory '{exp_dir}' already exists with completed metrics.\n"
            f"Experiments are immutable by contract. To test or re-run, pass --force or define a new experiment ID in configs/.",
            file=sys.stderr,
        )
        sys.exit(1)

    runner = UniversalRunner(cfg=cfg, config_path=args.config, args=args)
    return runner.run()


def main() -> None:
    runner_main()


if __name__ == "__main__":
    main()
