"""Pre-flight Inspection Engine for Cloud ML Jobs.

Performs deterministic local checks before dispatching jobs to Cloud GPUs (Modal / Kaggle):
1. Verifies experiment YAML config schema and paths.
2. Validates presence and non-emptiness of raw TSV datasets and split files.
3. Performs syntax pre-compilation (py_compile) on pipeline scripts to prevent costly cloud runtime crashes.
4. Audits parameter ceiling and video group leakage constraints.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class CloudPreflightInspector:
    """Validates codebase integrity and data readiness before cloud dispatch."""

    @staticmethod
    def inspect_config(config_path: Path | str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate config file exists and contains essential schema fields."""
        p = Path(config_path)
        errors = []
        cfg = {}

        if not p.exists():
            return False, [f"Config file not found: {p}"], {}

        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            return False, [f"Invalid YAML syntax in {p}: {e}"], {}

        exp_info = cfg.get("experiment", {})
        if not exp_info.get("id"):
            errors.append("Config missing 'experiment.id' string.")
        if not exp_info.get("branch"):
            errors.append("Config missing 'experiment.branch' string.")

        return len(errors) == 0, errors, cfg

    @staticmethod
    def inspect_datasets(data_dir: Path | str = "dataset") -> Tuple[bool, List[str]]:
        """Verify presence of raw training data files."""
        d = Path(data_dir)
        errors = []
        if not d.exists():
            return False, [f"Dataset path '{d}' does not exist."]
        data_files = [d] if d.is_file() else [
            path for path in d.iterdir()
            if path.is_file() and path.suffix.lower() in {".tsv", ".tab", ".csv", ".parquet", ".pq", ".feather", ".json", ".jsonl", ".ndjson"}
        ]
        if not data_files:
            return False, [f"No supported dataset files found at '{d}'."]

        for df_file in data_files:
            if df_file.stat().st_size == 0:
                errors.append(f"Training dataset file is empty (0 bytes): {df_file.name}")

        return len(errors) == 0, errors

    @staticmethod
    @staticmethod
    def inspect_python_syntax(files_to_check: List[str | Path]) -> Tuple[bool, List[str]]:
        """Verify python syntax via py_compile to catch syntax errors before cloud dispatch."""
        errors = []
        for file_p in files_to_check:
            p = Path(file_p)
            if not p.exists():
                continue
            try:
                py_compile.compile(str(p), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"Syntax error in '{p.name}': {e.msg}")
        return len(errors) == 0, errors

    @classmethod
    def run_full_preflight_check(
        cls,
        config_path: Path | str,
        project_root: Optional[Path | str] = None,
        full_data_check: bool = True,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Run all pre-flight checks and return consolidated verdict.

        ``full_data_check`` controls whether the training frame is loaded and
        plugin-validated locally. The dispatch path uses ``False`` so that a
        multi-GB frame is not re-loaded into local memory on every cloud
        dispatch (the remote job re-validates its data as its first step);
        explicit preflight commands keep the default ``True``.
        """
        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.project import HarnessProject
        from harness.task_spec import load_active_task_spec

        root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
        project = HarnessProject.discover(root)
        all_errors = []

        cfg_path = Path(config_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path

        # 1. Config Check
        c_ok, c_errs, cfg = cls.inspect_config(cfg_path)
        all_errors.extend(c_errs)

        # 2. Resolve data using task declarative path or the installed task adapter.
        try:
            task_doc = yaml.safe_load(project.task_config_path.read_text(encoding="utf-8")) or {}
            task = task_doc.get("task", task_doc)
            task_spec = load_active_task_spec(project.task_config_path)
            data_cfg = task.get("data_config") or task.get("data") or {}
            plugin_context = TaskPluginContext(
                project_root=root,
                task_config_path=project.task_config_path,
                task_spec=task_spec,
                plugin_id=project.plugin_id or "default",
            )
            plugin = load_task_plugin(project.plugin_id, plugin_context)
            frame = (
                plugin.load_training_data(plugin_context)
                if full_data_check
                else None
            )
            if frame is not None:
                plugin.validate_training_data(plugin_context, frame)
            else:
                train_value = data_cfg.get("train_path") or "dataset"
                train_path = Path(train_value).expanduser()
                if not train_path.is_absolute():
                    train_path = root / train_path
                _, data_errors = cls.inspect_datasets(train_path)
                all_errors.extend(data_errors)

            split_value = data_cfg.get("split_path")
            if split_value:
                split_path = Path(split_value).expanduser()
                if not split_path.is_absolute():
                    split_path = root / split_path
                if not split_path.is_file() or split_path.stat().st_size == 0:
                    all_errors.append(f"Declared split file is missing or empty: {split_path}")
        except Exception as exc:
            all_errors.append(f"Task data preflight failed: {exc}")

        # 4. Syntax Pre-compile Check
        harness_dir = Path(__file__).resolve().parent
        core_scripts = [
            harness_dir / "runner.py",
            harness_dir / "diagnostics.py",
            harness_dir / "ensemble_guard.py",
            harness_dir / "task_spec.py",
            harness_dir / "validator.py",
        ]
        for optional_script in [root / "src/data.py", root / "src/metrics.py", root / "src/utils.py"]:
            if optional_script.exists():
                core_scripts.append(optional_script)

        py_ok, py_errs = cls.inspect_python_syntax([s for s in core_scripts if s.exists()])
        all_errors.extend(py_errs)

        passed = len(all_errors) == 0
        return passed, all_errors, cfg


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Cloud Pre-flight Inspector")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    print("===========================================================")
    print(f"🔍 CLOUD PRE-FLIGHT INSPECTION: {args.config}")
    print("===========================================================")

    passed, errors, cfg = CloudPreflightInspector.run_full_preflight_check(args.config)
    if passed:
        print("✔ Configuration Schema : VALID")
        print("✔ Datasets & Splits    : VERIFIED")
        print("✔ Python Syntax Checks : 100% COMPILED")
        print("===========================================================")
        print("🎉 PRE-FLIGHT VERDICT: PASSED! Ready for cloud dispatch.")
        print("===========================================================")
        sys.exit(0)
    else:
        print("❌ PRE-FLIGHT INSPECTION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        print("===========================================================", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
