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

    def test_placeholder_values_are_warned(self) -> None:
        """Adoption guard: a copied template with placeholder values is warned.

        The single most common 'copied template, ran as-is' failure mode —
        train_path / target_column still carry a placeholder token.
        """
        ws = self._make_ws("tv_placeholder_")
        raw = _valid_raw("dataset/YOUR_DATASET.parquet")
        raw["task"]["data_config"]["target_column"] = "YOUR_TARGET_COL"
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        # train_path is missing (file doesn't exist) -> error
        self.assertFalse(result["ok"])
        self.assertTrue(any("train_path missing" in e for e in result["errors"]))
        # AND the placeholder guard warns on both train_path and target_column
        warnings_text = "\n".join(result["warnings"])
        self.assertIn("uncustomized template placeholder", warnings_text)
        self.assertIn("train_path", warnings_text)
        self.assertIn("target_column", warnings_text)


    def test_canonical_direction_conflict_warns(self) -> None:
        """A metric declared against its canonical direction is warned.

        Observed live (ARC-AGI-3 workspace): a subtask declared
        ``primary_metric: rmse, direction: maximize`` — RMSE is canonically
        minimized, so the spec is almost certainly wrong.
        """
        ws = self._make_ws("tv_dirconflict_")
        raw = {
            "task": {
                "id": "unit_task",
                "name": "Unit Task",
                "task_type": "regression",
                "primary_metric": "rmse",
                "direction": "maximize",
                "subtasks": [
                    {"id": "main", "task_type": "regression", "column": "y",
                     "primary_metric": "rmse", "direction": "maximize"}
                ],
                "execution": {"backend": "local"},
            }
        }
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        # Warnings only — the spec is still structurally valid.
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["errors"], [])
        conflicts = [w for w in result["warnings"] if "canonically minimized" in w]
        self.assertGreaterEqual(len(conflicts), 2, result["warnings"])

    def test_aggregate_mismatch_without_custom_scorer_errors(self) -> None:
        """Subtask metric != spec metric with no custom evaluator is an error.

        This is the exact class that failed mid-run on the ARC-AGI-3
        workspace: spec ``rhae``/maximize vs subtask ``rmse``/maximize —
        the generic aggregate rejects it at scoring time, after compute.
        """
        ws = self._make_ws("tv_agg_mismatch_")
        raw = {
            "task": {
                "id": "unit_task",
                "name": "Unit Task",
                "task_type": "interactive_agent",
                "primary_metric": "rhae",
                "direction": "maximize",
                "subtasks": [
                    {"id": "rhae", "task_type": "regression", "column": "rhae",
                     "primary_metric": "rmse", "direction": "maximize"}
                ],
                "execution": {"backend": "local"},
            }
        }
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("does not match the spec" in e and "paths.metrics_evaluator" in e
                for e in result["errors"]),
            result["errors"],
        )

    def test_aggregate_mismatch_with_custom_scorer_warns(self) -> None:
        """Same mismatch is a warning when paths.metrics_evaluator is declared:
        the custom hook is the authoritative scorer."""
        ws = self._make_ws("tv_agg_custom_")
        raw = {
            "task": {
                "id": "unit_task",
                "name": "Unit Task",
                "task_type": "interactive_agent",
                "primary_metric": "rhae",
                "direction": "maximize",
                "paths": {"metrics_evaluator": "src.evaluator.evaluate_rhae"},
                "subtasks": [
                    {"id": "rhae", "task_type": "regression", "column": "rhae",
                     "primary_metric": "rmse", "direction": "maximize"}
                ],
                "execution": {"backend": "local"},
            }
        }
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["errors"], [])
        self.assertTrue(
            any("metrics_evaluator" in w and "rhae" in w for w in result["warnings"]),
            result["warnings"],
        )

    def test_mixed_subtask_metrics_require_custom_scorer(self) -> None:
        ws = self._make_ws("tv_mixed_")
        raw = {
            "task": {
                "id": "unit_task",
                "name": "Unit Task",
                "task_type": "binary_classification",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": [
                    {"id": "a", "task_type": "binary_classification",
                     "column": "label", "primary_metric": "accuracy",
                     "direction": "maximize"},
                    {"id": "b", "task_type": "binary_classification",
                     "column": "label", "primary_metric": "roc_auc",
                     "direction": "maximize"},
                ],
                "execution": {"backend": "local"},
            }
        }
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("mix metrics/directions" in e for e in result["errors"]),
            result["errors"],
        )
        # Declaring a custom evaluator downgrades the condition to a warning.
        raw["task"]["paths"] = {"metrics_evaluator": "src.scorer.official"}
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertTrue(
            any("mix metrics/directions" in w for w in result["warnings"]),
            result["warnings"],
        )


    def test_custom_metric_under_stock_task_type(self) -> None:
        """A non-generic metric under a stock task type cannot be scored by
        the generic aggregate — error without a custom scorer, warning with
        one (the exact ARC-AGI-3 case: subtask rhae under regression)."""
        ws = self._make_ws("tv_custom_metric_")
        raw = {
            "task": {
                "id": "unit_task",
                "name": "Unit Task",
                "task_type": "interactive_agent",
                "primary_metric": "rhae",
                "direction": "maximize",
                "subtasks": [
                    {"id": "rhae", "task_type": "regression", "column": "rhae",
                     "primary_metric": "rhae", "direction": "maximize"}
                ],
                "execution": {"backend": "local"},
            }
        }
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("is not a generic metric" in e for e in result["errors"]),
            result["errors"],
        )
        # With a custom evaluator the condition is advisory.
        raw["task"]["paths"] = {"metrics_evaluator": "src.evaluator.evaluate_rhae"}
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertTrue(
            any("is not a generic metric" in w for w in result["warnings"]),
            result["warnings"],
        )

    def test_scorable_metrics_not_flagged_as_unsupported(self) -> None:
        """Metrics the generic aggregate *can* score must not be flagged.

        Guards against the allow-list being narrower than ``evaluate()``:
        ``rmsle`` (regression) and ``weighted_f1`` (binary) are scorable, so a
        spec declaring them is valid and must not raise an "unsupported
        metric" error (this was a live false-positive before the table was
        kept in sync with ``evaluate()``).
        """
        ws = self._make_ws("tv_scorable_")
        raw = {
            "task": {
                "id": "unit_task",
                "name": "Unit Task",
                "task_type": "regression",
                "primary_metric": "rmsle",
                "direction": "minimize",
                "subtasks": [
                    {"id": "main", "task_type": "regression", "column": "target",
                     "primary_metric": "rmsle", "direction": "minimize"}
                ],
                "execution": {"backend": "local"},
            }
        }
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertFalse(
            any("is not a generic metric" in e for e in result["errors"]),
            result["errors"],
        )
        self.assertFalse(
            any("is not a generic metric" in w for w in result["warnings"]),
            result["warnings"],
        )

        # Same guard for a scorable classification metric.
        raw["task"]["task_type"] = "binary_classification"
        raw["task"]["primary_metric"] = "weighted_f1"
        raw["task"]["direction"] = "maximize"
        raw["task"]["subtasks"] = [
            {"id": "main", "task_type": "binary_classification", "column": "label",
             "primary_metric": "weighted_f1", "direction": "maximize"}
        ]
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(
            any("is not a generic metric" in m for m in result["errors"] + result["warnings"]),
            result,
        )

    def test_allowed_table_matches_evaluate_for_regression(self) -> None:
        """GENERIC_ALLOWED_METRICS['regression'] must equal the metrics the
        generic aggregate can actually score: no false rejections (narrow
        table) and no false passes (wide table)."""
        import pandas as pd

        from harness.evaluation import GENERIC_ALLOWED_METRICS, evaluate
        from harness.task_spec import UniversalTaskSpec

        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [10.0, 20.0, 30.0, 40.0]})
        preds = {"target": [11.0, 19.0, 31.0, 39.0]}

        def scores(metric: str) -> bool:
            spec = UniversalTaskSpec.from_dict({
                "name": "t", "task_type": "regression",
                "primary_metric": metric, "direction": "minimize",
                "id_column": "id",
                "subtasks": [{"id": "target", "task_type": "regression",
                              "primary_metric": metric, "direction": "minimize",
                              "weight": 1.0}],
            })
            try:
                evaluate(spec, truth, preds)
                return True
            except ValueError as exc:
                return "Unsupported metric" not in str(exc)

        allowed = GENERIC_ALLOWED_METRICS["regression"]
        for metric in sorted(allowed):
            self.assertTrue(scores(metric), f"{metric} is in the table but evaluate() rejects it")
        # A metric that evaluate() genuinely cannot score for regression must
        # not be in the table (guards against a too-wide table).
        self.assertFalse(scores("accuracy"))
        self.assertNotIn("accuracy", allowed)

    def _binary_classification_fixture(self):
        """Binary-classification fixture with hard labels plus per-class
        probabilities, so probability metrics (roc_auc, logloss, pr_auc,
        brier_score) are genuinely scored rather than skipped."""
        import pandas as pd

        truth = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]}
        )
        preds = {
            "label": [0, 1, 1, 1, 0, 1],
            "proba_label": [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.3, 0.7],
            ],
            "classes_label": [0, 1],
        }
        return truth, preds

    def test_allowed_table_matches_evaluate_for_binary_classification(self) -> None:
        """GENERIC_ALLOWED_METRICS['binary_classification'] must equal the
        metrics the generic aggregate can actually score: no false
        rejections (narrow table) and no false passes (wide table).

        Mirrors the regression drift guard but for the classification
        family. The fixture carries per-class probabilities so the
        probability-dependent metrics are exercised end-to-end.
        """
        from harness.evaluation import GENERIC_ALLOWED_METRICS, evaluate
        from harness.task_spec import UniversalTaskSpec

        truth, preds = self._binary_classification_fixture()

        def scores(metric: str) -> bool:
            spec = UniversalTaskSpec.from_dict({
                "name": "t", "task_type": "binary_classification",
                "primary_metric": metric, "direction": "maximize",
                "id_column": "id",
                "subtasks": [{"id": "label", "task_type": "binary_classification",
                              "column": "label", "primary_metric": metric,
                              "direction": "maximize", "weight": 1.0,
                              "classes": ["0", "1"]}],
            })
            try:
                evaluate(spec, truth, preds)
                return True
            except ValueError as exc:
                return "Unsupported metric" not in str(exc)

        allowed = GENERIC_ALLOWED_METRICS["binary_classification"]
        for metric in sorted(allowed):
            self.assertTrue(
                scores(metric), f"{metric} is in the table but evaluate() rejects it"
            )
        # A regression-only metric must not be offered for classification
        # (guards against a too-wide table).
        self.assertFalse(scores("rmse"))
        self.assertNotIn("rmse", allowed)

    def test_multilabel_table_differs_from_binary_only_by_ndcg(self) -> None:
        """The multilabel allow-list is the classification set with
        qwk/brier_score swapped for ndcg — the exact contract evaluated by
        ``evaluate()``. Locks the table against drift from the binary family.
        """
        from harness.evaluation import GENERIC_ALLOWED_METRICS

        binary = GENERIC_ALLOWED_METRICS["binary_classification"]
        multilabel = GENERIC_ALLOWED_METRICS["multilabel_classification"]
        self.assertEqual(multilabel, binary - {"qwk", "brier_score"} | {"ndcg"})
        self.assertIn("ndcg", multilabel)
        self.assertNotIn("qwk", multilabel)
        self.assertNotIn("brier_score", multilabel)


if __name__ == "__main__":
    unittest.main()
