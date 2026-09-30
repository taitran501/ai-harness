#!/usr/bin/env python3
"""CLI utility to inject/update competition-tailored AI agent skill."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.skill_injector import inject_skill_file, load_task_config
from tools.dashboard_server import TASK_PRESETS


def main():
    parser = argparse.ArgumentParser(description="Inject competition-specific agent skill into .agents/skills/")
    parser.add_argument("--config", type=str, default="configs/task.yaml", help="Path to task configuration YAML")
    parser.add_argument("--preset", type=str, default=None, help="Preset ID (e.g., kaggle_tabular_regression, kaggle_binary_classification, codabench_nlp_multitask, vision_multiclass)")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for SKILL.md")
    args = parser.parse_args()

    task_cfg = None
    if args.preset:
        matched = [p for p in TASK_PRESETS if p.get("id") == args.preset]
        if not matched:
            print(f"❌ Error: Preset '{args.preset}' not found. Available presets:")
            for p in TASK_PRESETS:
                print(f"  - {p['id']} ({p.get('preset_name')})")
            sys.exit(1)
        task_cfg = matched[0]
        print(f"📋 Using preset: {task_cfg.get('preset_name')}")

    out_path = Path(args.output) if args.output else None
    injected_path = inject_skill_file(task_config=task_cfg, config_path=Path(args.config), output_path=out_path)
    print(f"✔ AI Agent skill dynamically injected into: {injected_path}")


if __name__ == "__main__":
    main()
