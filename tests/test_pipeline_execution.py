"""Unit tests for multi-stage pipeline execution in UniversalRunner.

Covers: sequential stage execution, per-stage audit trail (stages/<name>.log +
stage_manifest.json), stage-specific env vars, fail-fast on a failing stage,
back-compat (mode:script unchanged), and the _resolve_pipeline_stages helper
(named stages + simple entrypoints + name-collision handling).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from harness.runner import UniversalRunner
from tests.test_correctness_contracts import task

STAGE1_CODE = """
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

project_root = os.environ.get("HARNESS_PROJECT_ROOT", ".")
train_df = pd.read_csv(os.path.join(project_root, "dataset", "train.csv"))

print(f"STAGE_NAME={os.environ.get('HARNESS_STAGE_NAME')}")
print(f"STAGE_INDEX={os.environ.get('HARNESS_STAGE_INDEX')}")
print(f"STAGE_TOTAL={os.environ.get('HARNESS_STAGE_TOTAL')}")
print(f"STAGE_DIR={os.environ.get('HARNESS_STAGE_DIR')}")
print(f"EXP_ID={os.environ.get('HARNESS_EXP_ID')}")

oof = train_df[["id"]].copy()
oof["pred_target"] = train_df["target"].values * 0.5
oof["fold"] = 0
oof.to_csv(os.path.join(args.exp_dir, "oof_stage1.csv"), index=False)
print("stage1 done")
"""

STAGE2_CODE = """
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

stage1 = pd.read_csv(os.path.join(args.exp_dir, "oof_stage1.csv"))
oof = stage1[["id", "pred_target", "fold"]].copy()
oof["pred_target"] = oof["pred_target"] * 2.0
oof.to_csv(os.path.join(args.exp_dir, "oof_predictions.csv"), index=False)
print("stage2 done")
"""


def _make_root(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "dataset").mkdir()
    (root / "src").mkdir()
    spec = task()
    doc = {"task": spec.metadata}
    doc["task"]["data"]["train_path"] = "dataset/train.csv"
    doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
    (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    frame = pd.DataFrame({
        "id": range(12),
        "group": range(12),
        "x": range(12),
        "category": ["a", "b"] * 6,
        "target": [float(x * 2.0) for x in range(12)],
    })
    frame.to_csv(root / "dataset/train.csv", index=False)


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=str(root),
        task_config=None,
        fold=None,
        debug=False,
        probe_samples=None,
    )


def _exp_cfg(exp_id: str, execution: dict, approach: str = "custom_script") -> dict:
    return {
        "experiment": {
            "id": exp_id,
            "branch": "baseline",
            "parent_id": "none",
            "problem_addressed": "Pipeline execution test",
            "hypothesis": "Pipeline runs stages in sequence with an audit trail",
            "intervention": "Run the pipeline",
            "approach": approach,
        },
        "execution": execution,
    }


class TestPipelineExecution(unittest.TestCase):
    def test_pipeline_success_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_root(root)
            (root / "src" / "stage_train.py").write_text(STAGE1_CODE, encoding="utf-8")
            (root / "src" / "stage_blend.py").write_text(STAGE2_CODE, encoding="utf-8")

            cfg = _exp_cfg(
                "exp000_pipeline",
                {
                    "mode": "pipeline",
                    "pipeline": [
                        {"name": "train", "entrypoint": "src/stage_train.py"},
                        {"name": "blend", "entrypoint": "src/stage_blend.py"},
                    ],
                },
            )
            runner = UniversalRunner(cfg, "config.yaml", _args(root))
            result = runner.run()

            self.assertEqual(runner.manifest["status"], "COMPLETED")
            # Per-stage audit trail
            self.assertTrue((runner.exp_dir / "stages" / "train.log").exists())
            self.assertTrue((runner.exp_dir / "stages" / "blend.log").exists())
            manifest = json.loads((runner.exp_dir / "stage_manifest.json").read_text())
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest[0]["name"], "train")
            self.assertEqual(manifest[0]["exit_code"], 0)
            self.assertEqual(manifest[1]["name"], "blend")
            self.assertEqual(manifest[1]["exit_code"], 0)
            self.assertEqual(manifest[1]["total"], 2)
            # Stage-specific env vars reach the subprocess
            train_log = (runner.exp_dir / "stages" / "train.log").read_text()
            self.assertIn("STAGE_NAME=train", train_log)
            self.assertIn("STAGE_INDEX=0", train_log)
            self.assertIn("STAGE_TOTAL=2", train_log)
            self.assertIn("STAGE_DIR=", train_log)
            # Final OOF is correct: blend recovers target exactly -> RMSE ~ 0
            oof = pd.read_csv(runner.exp_dir / "oof_predictions.csv")
            self.assertEqual(len(oof), 12)
            self.assertAlmostEqual(result["primary_score"], 0.0, places=5)

    def test_pipeline_simple_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_root(root)
            (root / "src" / "stage_train.py").write_text(STAGE1_CODE, encoding="utf-8")
            (root / "src" / "stage_blend.py").write_text(STAGE2_CODE, encoding="utf-8")

            cfg = _exp_cfg(
                "exp000_pipeline_simple",
                {
                    "mode": "pipeline",
                    "entrypoints": ["src/stage_train.py", "src/stage_blend.py"],
                },
            )
            runner = UniversalRunner(cfg, "config.yaml", _args(root))
            result = runner.run()

            self.assertEqual(runner.manifest["status"], "COMPLETED")
            manifest = json.loads((runner.exp_dir / "stage_manifest.json").read_text())
            self.assertEqual(len(manifest), 2)
            # Names derived from entrypoint stems
            self.assertEqual(manifest[0]["name"], "stage_train")
            self.assertEqual(manifest[1]["name"], "stage_blend")
            self.assertAlmostEqual(result["primary_score"], 0.0, places=5)

    def test_pipeline_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_root(root)
            (root / "src" / "stage_fail.py").write_text(
                "import sys\nsys.stderr.write('BROKEN STAGE\\n')\nsys.exit(7)\n",
                encoding="utf-8",
            )
            (root / "src" / "stage_blend.py").write_text(STAGE2_CODE, encoding="utf-8")

            cfg = _exp_cfg(
                "exp000_pipeline_fail",
                {
                    "mode": "pipeline",
                    "pipeline": [
                        {"name": "train", "entrypoint": "src/stage_fail.py"},
                        {"name": "blend", "entrypoint": "src/stage_blend.py"},
                    ],
                },
            )
            runner = UniversalRunner(cfg, "config.yaml", _args(root))
            with self.assertRaises(RuntimeError) as ctx:
                runner.run()
            msg = str(ctx.exception)
            self.assertIn("train", msg)
            self.assertIn("BROKEN STAGE", msg)
            # Stage manifest is persisted even on failure (fail-fast)
            manifest = json.loads((runner.exp_dir / "stage_manifest.json").read_text())
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["exit_code"], 7)
            # The second stage must NOT have run
            self.assertFalse((runner.exp_dir / "stages" / "blend.log").exists())

    def test_pipeline_approach_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_root(root)
            (root / "src" / "stage_train.py").write_text(STAGE1_CODE, encoding="utf-8")

            cfg = _exp_cfg(
                "exp000_pipeline_bad_approach",
                {
                    "mode": "pipeline",
                    "pipeline": [{"name": "train", "entrypoint": "src/stage_train.py"}],
                },
                approach="ml_classical",
            )
            runner = UniversalRunner(cfg, "config.yaml", _args(root))
            with self.assertRaises(ValueError) as ctx:
                runner.run()
            self.assertIn("Pipeline execution requires", str(ctx.exception))

    def test_resolve_pipeline_stages_named(self):
        runner = UniversalRunner.__new__(UniversalRunner)
        stages = UniversalRunner._resolve_pipeline_stages(
            runner,
            {
                "pipeline": [
                    {"name": "train", "entrypoint": "src/a.py", "args": ["--x", "1"]},
                    "src/b.py",
                    {"entrypoint": "src/c.py"},
                ],
            },
        )
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0]["name"], "train")
        self.assertEqual(stages[0]["entrypoint"], "src/a.py")
        self.assertEqual(stages[0]["args"], ["--x", "1"])
        self.assertEqual(stages[1]["name"], "b")
        self.assertEqual(stages[1]["entrypoint"], "src/b.py")
        self.assertEqual(stages[1]["args"], [])
        self.assertEqual(stages[2]["name"], "c")
        self.assertEqual(stages[2]["entrypoint"], "src/c.py")

    def test_resolve_pipeline_stages_simple(self):
        runner = UniversalRunner.__new__(UniversalRunner)
        stages = UniversalRunner._resolve_pipeline_stages(
            runner, {"entrypoints": ["src/a.py", "src/b.py"]}
        )
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["name"], "a")
        self.assertEqual(stages[0]["entrypoint"], "src/a.py")
        self.assertEqual(stages[1]["name"], "b")
        self.assertEqual(stages[1]["entrypoint"], "src/b.py")

    def test_resolve_pipeline_stages_empty(self):
        runner = UniversalRunner.__new__(UniversalRunner)
        self.assertEqual(UniversalRunner._resolve_pipeline_stages(runner, {}), [])

    def test_resolve_pipeline_stages_collision(self):
        runner = UniversalRunner.__new__(UniversalRunner)
        stages = UniversalRunner._resolve_pipeline_stages(
            runner, {"entrypoints": ["src/a.py", "other/a.py"]}
        )
        self.assertEqual(stages[0]["name"], "a")
        self.assertEqual(stages[1]["name"], "a_1")

    def test_resolve_pipeline_stages_missing_entrypoint(self):
        runner = UniversalRunner.__new__(UniversalRunner)
        with self.assertRaises(ValueError):
            UniversalRunner._resolve_pipeline_stages(
                runner, {"pipeline": [{"name": "train"}]}
            )

    # ---- _verify_pipeline_manifest (audit gate) ----

    def _manifest_runner(self, exp_dir: Path) -> UniversalRunner:
        runner = UniversalRunner.__new__(UniversalRunner)
        runner.exp_dir = exp_dir
        return runner

    def test_manifest_verify_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "stages").mkdir()
            (exp_dir / "stages" / "train.log").write_text("train done\n")
            (exp_dir / "stages" / "blend.log").write_text("blend done\n")
            stages = [
                {"name": "train", "index": 0, "total": 2, "exit_code": 0,
                 "duration_seconds": 0.1, "log_file": "stages/train.log"},
                {"name": "blend", "index": 1, "total": 2, "exit_code": 0,
                 "duration_seconds": 0.2, "log_file": "stages/blend.log"},
            ]
            (exp_dir / "stage_manifest.json").write_text(json.dumps(stages), encoding="utf-8")
            # Should not raise
            self._manifest_runner(exp_dir)._verify_pipeline_manifest(2)

    def test_manifest_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self._manifest_runner(Path(tmp))._verify_pipeline_manifest(2)
            self.assertIn("stage_manifest.json missing", str(ctx.exception))

    def test_manifest_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "stages").mkdir()
            (exp_dir / "stages" / "train.log").write_text("train done\n")
            stages = [
                {"name": "train", "index": 0, "total": 1, "exit_code": 0,
                 "duration_seconds": 0.1, "log_file": "stages/train.log"},
            ]
            (exp_dir / "stage_manifest.json").write_text(json.dumps(stages), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._manifest_runner(exp_dir)._verify_pipeline_manifest(2)
            self.assertIn("expected 2 stages, found 1", str(ctx.exception))

    def test_manifest_bad_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "stages").mkdir()
            (exp_dir / "stages" / "train.log").write_text("train done\n")
            stages = [
                {"name": "train", "index": 0, "total": 1, "exit_code": 7,
                 "duration_seconds": 0.1, "log_file": "stages/train.log"},
            ]
            (exp_dir / "stage_manifest.json").write_text(json.dumps(stages), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._manifest_runner(exp_dir)._verify_pipeline_manifest(1)
            self.assertIn("exit code 7 != 0", str(ctx.exception))

    def test_manifest_order_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "stages").mkdir()
            (exp_dir / "stages" / "train.log").write_text("train done\n")
            (exp_dir / "stages" / "blend.log").write_text("blend done\n")
            stages = [
                {"name": "train", "index": 1, "total": 2, "exit_code": 0,
                 "duration_seconds": 0.1, "log_file": "stages/train.log"},
                {"name": "blend", "index": 0, "total": 2, "exit_code": 0,
                 "duration_seconds": 0.2, "log_file": "stages/blend.log"},
            ]
            (exp_dir / "stage_manifest.json").write_text(json.dumps(stages), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._manifest_runner(exp_dir)._verify_pipeline_manifest(2)
            self.assertIn("order violated", str(ctx.exception))

    def test_manifest_log_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "stages").mkdir()
            # NOTE: train.log is intentionally NOT created (missing)
            stages = [
                {"name": "train", "index": 0, "total": 1, "exit_code": 0,
                 "duration_seconds": 0.1, "log_file": "stages/train.log"},
            ]
            (exp_dir / "stage_manifest.json").write_text(json.dumps(stages), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._manifest_runner(exp_dir)._verify_pipeline_manifest(1)
            self.assertIn("log file missing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
