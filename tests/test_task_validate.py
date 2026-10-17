"""Tests for static task.yaml validation (``harness.task_validate``).

Every failure mode asserted here was observed live in a workspace before
surfacing mid-run: a GroupKFold group column with fewer uniques than folds
(``Null validation folds``), subtask columns absent from the train file,
missing data files, and hygiene allow-lists that whitelisted genuine
run-output dirs (``kernel_output_*``) so the adoption scan never saw them.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from harness.task_validate import validate_task


def _write_task(ws: Path, raw: dict) -> Path:
    p = ws / "configs"
    p.mkdir(parents=True, exist_ok=True)
    path = p / "task.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _valid_raw(train: str) -> dict:
    return {
        "task": {
            "id": "unit_task",
            "name": "Unit Task",
            "task_type": "binary_classification",
            "primary_metric": "accuracy",
            "direction": "maximize",
            "inputs": [{"name": "text", "type": "text", "source_columns": ["body"]}],
            "subtasks": [{"id": "main", "task_type": "binary_classification",
                          "column": "label", "direction": "maximize"}],
            "data_config": {
                "train_path": train,
                "id_column": "id",
                "target_column": "label",
                "group_column": "group_id",
                "split_strategy": "GroupKFold",
                "n_folds": 3,
                "seed": 42,
            },
            "execution": {"backend": "local"},
        }
    }


class TestTaskValidate(unittest.TestCase):
    def _make_ws(self, prefix: str) -> Path:
        tmp = tempfile.TemporaryDirectory(prefix=prefix)
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def _make_parquet(self, ws: Path, rows: list) -> Path:
        """Write a real parquet so schema/column checks exercise the real path."""
        import pandas as pd

        d = ws / "data"
        d.mkdir(exist_ok=True)
        path = d / "train.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def test_valid_task_passes_clean(self) -> None:
        ws = self._make_ws("tv_ok_")
        rows = [{"id": str(i), "body": f"t{i}", "label": i % 2,
                 "group_id": f"g{i % 5}"} for i in range(20)]
        train = self._make_parquet(ws, rows)
        cfg = _write_task(ws, _valid_raw(f"data/{train.name}"))
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["info"]["train_rows"], 20)

    def test_group_column_with_too_few_uniques_is_an_error(self) -> None:
        """The live ``Null validation folds`` class: 3 folds but 2 groups."""
        ws = self._make_ws("tv_fewgroups_")
        rows = [{"id": str(i), "body": f"t{i}", "label": i % 2,
                 "group_id": f"g{i % 2}"} for i in range(20)]
        train = self._make_parquet(ws, rows)
        cfg = _write_task(ws, _valid_raw(f"data/{train.name}"))
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("only 2 unique values" in e for e in result["errors"]),
                        result["errors"])

    def test_missing_columns_are_errors(self) -> None:
        ws = self._make_ws("tv_badcols_")
        rows = [{"id": str(i), "body": f"t{i}", "label": i % 2,
                 "group_id": f"g{i % 5}"} for i in range(20)]
        train = self._make_parquet(ws, rows)
        raw = _valid_raw(f"data/{train.name}")
        raw["task"]["inputs"][0]["source_columns"] = ["nope_missing"]
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("nope_missing" in e for e in result["errors"]),
                        result["errors"])

    def test_missing_train_file_is_an_error(self) -> None:
        ws = self._make_ws("tv_notrain_")
        cfg = _write_task(ws, _valid_raw("data/absent.parquet"))
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("train_path missing" in e for e in result["errors"]))

    def test_whitelisted_run_output_dir_is_flagged(self) -> None:
        """The rsna-knee lesson: allow-lists hiding genuine bypasses."""
        ws = self._make_ws("tv_hyg_")
        rows = [{"id": str(i), "body": f"t{i}", "label": i % 2,
                 "group_id": f"g{i % 5}"} for i in range(20)]
        train = self._make_parquet(ws, rows)
        raw = _valid_raw(f"data/{train.name}")
        raw["task"]["hygiene"] = {"allowed_top_dirs": ["baselines", "kernel_output_exp003"]}
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(any("kernel_output_exp003" in w for w in result["warnings"]),
                        result["warnings"])

    def test_constant_target_is_warned(self) -> None:
        ws = self._make_ws("tv_const_")
        rows = [{"id": str(i), "body": f"t{i}", "label": 1,
                 "group_id": f"g{i % 5}"} for i in range(20)]
        train = self._make_parquet(ws, rows)
        cfg = _write_task(ws, _valid_raw(f"data/{train.name}"))
        result = validate_task(cfg, ws)
        self.assertTrue(any("constant in the train file" in w for w in result["warnings"]),
                        result["warnings"])

    def test_unknown_task_type_is_an_error(self) -> None:
        ws = self._make_ws("tv_type_")
        raw = _valid_raw("data/train.parquet")
        raw["task"]["task_type"] = "astrology"
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("unknown task_type" in e for e in result["errors"]))

    def test_missing_config_reports_cleanly(self) -> None:
        ws = self._make_ws("tv_absent_")
        result = validate_task(Path("configs") / "task.yaml", ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("task config not found" in e for e in result["errors"]))

    def test_no_subtasks_is_a_warning_not_error(self) -> None:
        ws = self._make_ws("tv_nosub_")
        rows = [{"id": str(i), "body": f"t{i}", "label": i % 2,
                 "group_id": f"g{i % 5}"} for i in range(20)]
        train = self._make_parquet(ws, rows)
        raw = _valid_raw(f"data/{train.name}")
        raw["task"].pop("subtasks")
        raw["task"]["data_config"].pop("target_column")
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertTrue(any("no subtasks" in w for w in result["warnings"]))

    def test_non_dict_yaml_is_an_error(self) -> None:
        ws = self._make_ws("tv_list_")
        p = ws / "configs"
        p.mkdir(parents=True, exist_ok=True)
        cfg = p / "task.yaml"
        cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("top-level YAML must be a mapping" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
