"""Consolidated unit test suite for TaskSpec configuration, validation, API references, failure classification, and validators.

Consolidates:
- tests/unit/test_task_validate.py (TestTaskValidate)
- ValidatorTests from tests/test_harness.py
- tests/test_failure_classifier.py (TestFailureClassifier)
- tests/test_api_reference.py (ValidateReferenceTests, RenderReferenceTests, SkillInjectionTests, HallucinatedNamesAbsentTests, SubmissionFieldsTests)
- Config & CLI usability tests from tests/agent_bridge/test_usability_phase3.py, test_usability_phase6.py, test_usability_phase7.py
- tests/test_harness_universal.py (TestUniversalTaskSpec routing)
"""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

import harness.api_reference as api_ref
from harness.evaluation import GENERIC_ALLOWED_METRICS, evaluate
from harness.failure_classifier import FailureCode, classify_failure
from harness.task_spec import (
    ConditionalRoutingRule,
    InputVariantSpec,
    SubtaskSpec,
    UniversalTaskSpec,
)
from harness.task_validate import validate_task
from harness.validator import ExperimentValidator
from tools.agent_bridge_cli import _parse_dynamic_kv_args, _parse_json_arg


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
        ws = self._make_ws("tv_placeholder_")
        raw = _valid_raw("dataset/YOUR_DATASET.parquet")
        raw["task"]["data_config"]["target_column"] = "YOUR_TARGET_COL"
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertFalse(result["ok"])
        self.assertTrue(any("train_path missing" in e for e in result["errors"]))
        warnings_text = "\n".join(result["warnings"])
        self.assertIn("uncustomized template placeholder", warnings_text)
        self.assertIn("train_path", warnings_text)
        self.assertIn("target_column", warnings_text)

    def test_canonical_direction_conflict_warns(self) -> None:
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
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["errors"], [])
        conflicts = [w for w in result["warnings"] if "canonically minimized" in w]
        self.assertGreaterEqual(len(conflicts), 2, result["warnings"])

    def test_aggregate_mismatch_without_custom_scorer_errors(self) -> None:
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
        raw["task"]["paths"] = {"metrics_evaluator": "src.scorer.official"}
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertTrue(
            any("mix metrics/directions" in w for w in result["warnings"]),
            result["warnings"],
        )

    def test_custom_metric_under_stock_task_type(self) -> None:
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
        raw["task"]["paths"] = {"metrics_evaluator": "src.evaluator.evaluate_rhae"}
        cfg = _write_task(ws, raw)
        result = validate_task(cfg, ws)
        self.assertTrue(result["ok"], result)
        self.assertTrue(
            any("is not a generic metric" in w for w in result["warnings"]),
            result["warnings"],
        )

    def test_scorable_metrics_not_flagged_as_unsupported(self) -> None:
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
        self.assertFalse(scores("accuracy"))
        self.assertNotIn("accuracy", allowed)

    def _binary_classification_fixture(self):
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
        self.assertFalse(scores("rmse"))
        self.assertNotIn("rmse", allowed)

    def test_multilabel_table_differs_from_binary_only_by_ndcg(self) -> None:
        binary = GENERIC_ALLOWED_METRICS["binary_classification"]
        multilabel = GENERIC_ALLOWED_METRICS["multilabel_classification"]
        self.assertEqual(multilabel, binary - {"qwk", "brier_score"} | {"ndcg"})
        self.assertIn("ndcg", multilabel)
        self.assertNotIn("qwk", multilabel)
        self.assertNotIn("brier_score", multilabel)


class ValidatorTests(unittest.TestCase):
    def test_assert_parameter_limit_exceeded(self):
        class DummyModel:
            def num_parameters(self):
                return 200_000_000

        model = DummyModel()
        with self.assertRaises(ValueError):
            ExperimentValidator.assert_parameter_limit(model, max_params=150_000_000)

    def test_assert_parameter_limit_passed(self):
        class DummyModel:
            def num_parameters(self):
                return 50_000_000

        model = DummyModel()
        count = ExperimentValidator.assert_parameter_limit(model, max_params=150_000_000)
        self.assertEqual(count, 50_000_000)

    def test_assert_no_group_leakage_detected(self):
        train_df = pd.DataFrame({"group_id": ["A", "B", "C"]})
        val_df = pd.DataFrame({"group_id": ["C", "D"]})

        with self.assertRaises(AssertionError):
            ExperimentValidator.assert_no_group_data_leakage(train_df, val_df, group_column="group_id")

    def test_assert_no_group_leakage_clean(self):
        train_df = pd.DataFrame({"group_id": ["A", "B"]})
        val_df = pd.DataFrame({"group_id": ["C", "D"]})
        ExperimentValidator.assert_no_group_data_leakage(train_df, val_df, group_column="group_id")

    def test_assert_task_conditional_integrity(self):
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "name": "conditional_example",
                "task_type": "multi_task_classification",
                "primary_metric": "macro_f1",
                "subtasks": [
                    {"id": "primary", "column": "primary", "task_type": "binary_classification"},
                    {
                        "id": "secondary",
                        "column": "secondary",
                        "task_type": "multiclass_classification",
                        "condition": {
                            "parent_target": "primary",
                            "operator": "==",
                            "values": "active",
                            "fallback_value": "none",
                        },
                    },
                ],
            }
        })
        clean_df = pd.DataFrame(
            {
                "primary": ["inactive", "active"],
                "secondary": ["none", "choice_a"],
            }
        )
        ExperimentValidator.assert_task_conditional_integrity(clean_df, spec=spec)

        violating_df = pd.DataFrame(
            {
                "primary": ["inactive", "active"],
                "secondary": ["choice_b", "choice_a"],
            }
        )
        with self.assertRaises(AssertionError):
            ExperimentValidator.assert_task_conditional_integrity(violating_df, spec=spec)


class TestFailureClassifier(unittest.TestCase):
    def test_missing_dependency_module_not_found(self):
        log = "Traceback:\nModuleNotFoundError: No module named 'lightgbm'"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.DEPENDENCY_MISSING)
        self.assertEqual(res["category"], "environment")
        self.assertFalse(res["retryable"])
        self.assertIn("pip install lightgbm", res["recovery"]["suggested_command"])

    def test_missing_symbol_cannot_import(self):
        log = "ImportError: cannot import name 'FastModel' from 'models'"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.IMPORT_SYMBOL_ERROR)
        self.assertIsNone(res["recovery"]["suggested_command"])
        self.assertIn("FastModel", res["recovery"]["suggested_action"])

    def test_cuda_oom(self):
        log = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.RESOURCE_OOM)
        self.assertEqual(res["category"], "resource")
        self.assertIn("batch size", res["recovery"]["suggested_action"])

    def test_auth_required(self):
        log = "Kaggle authentication unavailable. Please set KAGGLE_USERNAME and KAGGLE_KEY."
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.AUTH_REQUIRED)
        self.assertEqual(res["category"], "provider")

    def test_quota_exceeded(self):
        log = "API error: 429 Too Many Requests. Quota exceeded for today."
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.QUOTA_EXCEEDED)
        self.assertTrue(res["retryable"])
        self.assertTrue(res["recovery"]["retryable"])

    def test_config_invalid(self):
        log = "yaml.parser.ParserError: while parsing a block mapping in configs/exp.yaml"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.CONFIG_INVALID)
        self.assertEqual(res["recovery"]["suggested_command"], "ai-harness validate-task")

    def test_artifact_unverifiable(self):
        log = "Run exp001 has artifact hash drift; seal is broken."
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.ARTIFACT_UNVERIFIABLE)
        self.assertEqual(res["recovery"]["suggested_command"], "ai-harness repair --check")

    def test_data_contract_invalid(self):
        log = "ValueError: Task ground truth missing id_column 'sample_id'"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.DATA_CONTRACT_INVALID)
        self.assertEqual(res["category"], "data")

    def test_unknown_fallback(self):
        log = "Fatal syntax or unhandled exception in line 42: division by zero"
        res = classify_failure(log, returncode=139)
        self.assertEqual(res["code"], FailureCode.UNKNOWN)
        self.assertFalse(res["retryable"])
        self.assertIn("139", res["detail"])


class ValidateReferenceTests(unittest.TestCase):
    def test_reference_is_fully_valid(self):
        problems = api_ref.validate_reference()
        self.assertEqual(
            problems, [],
            msg="stale reference entries:\n" + "\n".join(problems),
        )

    def test_all_cli_commands_registered(self):
        commands = api_ref._ai_harness_commands()
        for name, _ in api_ref.CLI_COMMANDS:
            self.assertIn(name, commands, "%r not a real ai-harness command" % name)

    def test_facade_methods_exist_on_client(self):
        client = api_ref._resolve("harness.kaggle_compat.KaggleClient")
        for method, _ in api_ref.FACADE_METHODS:
            self.assertTrue(hasattr(client, method), "KaggleClient.%s missing" % method)

    def test_key_modules_import(self):
        for dotted, _ in api_ref.KEY_MODULES:
            importlib.import_module(dotted)


class RenderReferenceTests(unittest.TestCase):
    def test_render_includes_verified_names(self):
        md = api_ref.render_reference()
        for needle in (
            "## 7. Verified API surface",
            "harness.kaggle_compat.KaggleClient",
            "harness.task_spec",
            "list_submissions",
            "submission",
            "dispatch-kernel",
            "ai-harness-agent",
        ):
            self.assertIn(needle, md)

    def test_render_warns_against_guessing(self):
        md = api_ref.render_reference()
        self.assertIn("do NOT invent", md)
        self.assertIn("never guess", md)


class SkillInjectionTests(unittest.TestCase):
    def test_skill_md_contains_reference(self):
        from harness.skill_injector import generate_skill_content

        md = generate_skill_content({"name": "T", "platform": "kaggle"})
        self.assertIn("## 7. Verified API surface", md)
        self.assertIn("harness.task_spec", md)


class HallucinatedNamesAbsentTests(unittest.TestCase):
    def test_invented_module_names_do_not_exist(self):
        for bad in ("harness.agents", "harness.task_config"):
            with self.assertRaises(
                ModuleNotFoundError, msg="%s should not exist" % bad
            ):
                importlib.import_module(bad)


class SubmissionFieldsTests(unittest.TestCase):
    def test_submission_fields_resolve_on_live_shape(self):
        from harness.kaggle_compat import _NormalisedObject

        sub = _NormalisedObject(api_ref._submission_fixture(), "kagglesdk")
        for name, _ in api_ref.SUBMISSION_FIELDS:
            with self.subTest(field=name):
                getattr(sub, name)

    def test_submission_fields_resolve_on_legacy_shape(self):
        from harness.kaggle_compat import _NormalisedObject

        sub = _NormalisedObject(api_ref._submission_fixture(), "legacy")
        for name, _ in api_ref.SUBMISSION_FIELDS:
            with self.subTest(field=name):
                getattr(sub, name)


class TestUniversalTaskSpecRoutingAndUsability(unittest.TestCase):
    def test_conditional_routing(self):
        spec = UniversalTaskSpec(
            name="GenericMultiTask",
            task_type="classification",
            primary_metric="macro_f1",
            subtasks={
                "subtask_b": SubtaskSpec(id="subtask_b", name="subtask_b", task_type="multiclass_classification", primary_metric="macro_f1"),
                "subtask_c_type": SubtaskSpec(
                    id="subtask_c_type",
                    name="subtask_c_type",
                    task_type="multiclass_classification",
                    primary_metric="macro_f1",
                    condition=ConditionalRoutingRule(
                        parent_target="subtask_b",
                        operator="not_in",
                        values=["no"],
                        fallback_value="none",
                    ),
                ),
                "subtask_c_identity": SubtaskSpec(
                    id="subtask_c_identity",
                    name="subtask_c_identity",
                    task_type="multilabel_classification",
                    primary_metric="macro_f1",
                    condition=ConditionalRoutingRule(
                        parent_target="subtask_b",
                        operator="not_in",
                        values=["no"],
                        fallback_value=0,
                    ),
                ),
            },
        )

        cond = spec.subtasks["subtask_c_type"].condition
        self.assertIsNotNone(cond)
        self.assertEqual(cond.parent_target, "subtask_b")
        self.assertEqual(cond.operator, "not_in")
        self.assertEqual(cond.fallback_value, "none")

        raw_preds = {
            "subtask_b": np.array(["no", "yes_explicit", "no", "yes_implicit"]),
            "subtask_c_type": np.array(["group", "individual", "individual", "group"]),
            "subtask_c_identity": np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]),
        }

        routed = spec.apply_routing(raw_preds, n_samples=4)
        self.assertEqual(routed["subtask_c_type"][0], "none")
        np.testing.assert_array_equal(routed["subtask_c_identity"][0], np.zeros(9))
        self.assertEqual(routed["subtask_c_type"][1], "individual")
        self.assertEqual(routed["subtask_c_identity"][1, 1], 1)
        self.assertEqual(routed["subtask_c_type"][2], "none")
        np.testing.assert_array_equal(routed["subtask_c_identity"][2], np.zeros(9))
        self.assertEqual(routed["subtask_c_type"][3], "group")
        self.assertEqual(routed["subtask_c_identity"][3, 3], 1)

    def test_json_parsing_resilience_powershell_compat(self):
        self.assertEqual(_parse_json_arg('{"branch": "feat", "num": 1}'), {"branch": "feat", "num": 1})
        self.assertEqual(_parse_json_arg("{'branch': 'feat', 'num': 2}"), {"branch": "feat", "num": 2})
        self.assertEqual(_parse_json_arg("{branch: feat, num: 3}"), {"branch": "feat", "num": 3})

    def test_dynamic_cli_kv_arguments(self):
        args = ["--branch", "feat", "--descriptor", "v1", "--weight", "0.75", "--flag", "--config-path", "configs/test.yaml"]
        res = _parse_dynamic_kv_args(args)
        self.assertEqual(res["branch"], "feat")
        self.assertEqual(res["descriptor"], "v1")
        self.assertEqual(res["weight"], 0.75)
        self.assertEqual(res["flag"], True)
        self.assertEqual(res["config_path"], "configs/test.yaml")

    def test_cross_environment_dataset_path_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds_dir = root / "dataset"
            ds_dir.mkdir()
            local_train = ds_dir / "train.csv"
            local_train.write_text("id,val\n1,10\n2,20\n", encoding="utf-8")

            foreign_abs_path = "/kaggle/working/dataset/train.csv"
            spec = UniversalTaskSpec.from_dict({
                "task": {
                    "id_column": "id",
                    "primary_metric": "accuracy",
                    "data": {
                        "train": foreign_abs_path,
                    },
                }
            })

            df = spec.load_dataset_split("train", project_root=root)
            self.assertEqual(len(df), 2)
            self.assertEqual(df["id"].tolist(), [1, 2])

    def test_resolve_media_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img_dir = tmp_path / "images"
            img_dir.mkdir(parents=True)
            (img_dir / "sample_01.jpg").write_text("dummy image 1")
            (img_dir / "sample_02.png").write_text("dummy image 2")

            df = pd.DataFrame({
                "img_filename": ["sample_01.jpg", "sample_02"],
            })
            variant = InputVariantSpec(
                name="media_input",
                type="image",
                source_columns=["img_filename"],
            )

            resolved = variant.resolve_media_paths(
                df,
                project_root=tmp_path,
                column="img_filename",
                extensions=[".png", ".jpg"],
            )
            self.assertEqual(len(resolved), 2)
            self.assertTrue(Path(resolved.iloc[0]).exists())
            self.assertTrue(resolved.iloc[0].endswith("sample_01.jpg"))
            self.assertTrue(Path(resolved.iloc[1]).exists())
            self.assertTrue(resolved.iloc[1].endswith("sample_02.png"))


if __name__ == "__main__":
    unittest.main()
