#!/usr/bin/env python3
"""Unified training and evaluation CLI runner for competition experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from harness.runner import UniversalRunner, main as runner_main, merge_configs
from harness.io_utils import load_yaml_config


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute training run via UniversalRunner."""
    base_cfg = load_yaml_config(args.base_config) if Path(args.base_config).exists() else {}
    exp_cfg = load_yaml_config(args.config)
    cfg = merge_configs(base_cfg, exp_cfg)

    runner = UniversalRunner(cfg=cfg, config_path=args.config, args=args)
    return runner.run()


def main() -> None:
    runner_main()


if __name__ == "__main__":
    main()
