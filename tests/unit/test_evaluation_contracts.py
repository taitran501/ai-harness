"""Consolidated unit test suite for evaluation contracts, metrics, and probability bounds.

Consolidates:
- tests/unit/test_correctness_contracts.py (EvaluationContracts, MetricReferenceTests)
- tests/test_interactive_prediction_contract.py (TestInteractivePredictionContract)
- tests/agent_bridge/test_usability_phase4.py (TestEvaluationUsabilityAudit)
- Evaluation and metric tests from tests/agent_bridge/test_usability_phase8.py & tests/test_harness_universal.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from sklearn import metrics

from harness.evaluation import (
    compute_bleu,
    compute_box_iou,
    compute_cer,
    compute_dice_score,
    compute_exact_match,
    compute_iou_score,
    compute_mrr,
    compute_pearson,
    compute_pinball_loss,
    compute_rmsle,
    compute_smape,
    compute_wape,
    compute_wer,
    default_metric_direction,
    evaluate,
    mask_to_rle,
    rle_to_mask,
    validate_prediction_contract,
)
from harness.models import NativeCategoricalTransformer
from harness.oof_revalidation import score_predictions
from harness.task_spec import SubtaskSpec, UniversalTaskSpec
from harness.templates.kaggle_template import KaggleKernelManager
from harness.tuning_engine import ProbabilityCalibrator, ThresholdOptimizer


def task(kind="regression", metric="rmse", direction="minimize"):
    return UniversalTaskSpec.from_dict(
        {
            "task": {
                "name": "fixture",
                "modality": "tabular",
                "primary_metric": metric,
                "direction": direction,
                "data": {"id_column": "id", "group_column": "group", "seed": 13},
                "inputs": [
                    {
                        "name": "features",
                        "type": "tabular",
                        "source_columns": ["x", "category"],
                    }
                ],
                "subtasks": [
                    {
                        "id": "target",
                        "column": "target",
                        "task_type": kind,
                        "primary_metric": metric,
                        "direction": direction,
                        "classes": [] if kind == "regression" else ["0", "1"],
                    }
                ],
            }
        }
    )


class EvaluationContracts(unittest.TestCase):
    def test_raw_regression_and_utility(self):
        result = task().evaluate_predictions(
            pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}), {"target": [0.0, 3.0]}
        )
        self.assertAlmostEqual(result["primary_score"], np.sqrt(2))
        self.assertAlmostEqual(result["utility"], -np.sqrt(2))

    def test_probability_metric(self):
        spec = task("binary_classification", "ROC-AUC", "maximize")
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            spec.evaluate_predictions(truth, {"target": [0, 0, 1, 1]})
        result = spec.evaluate_predictions(
            truth,
            {
                "target": [0, 0, 1, 1],
                "proba_target": [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
                "classes_target": [0, 1],
            },
        )
        self.assertEqual(result["primary_score"], 1)

    def test_multilabel_probability_metric_does_not_require_sum_to_one(self):
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "name": "multilabel_fixture",
                "modality": "tabular",
                "primary_metric": "roc_auc",
                "direction": "maximize",
                "data": {"id_column": "id"},
                "inputs": [{"name": "features", "type": "tabular", "source_columns": ["x"]}],
                "subtasks": [{
                    "id": "target",
                    "column_prefix": "lbl_",
                    "task_type": "multilabel_classification",
                    "primary_metric": "roc_auc",
                    "direction": "maximize",
                    "labels": ["a", "b"],
                }],
            }
        })
        truth = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "lbl_a": [0, 0, 1, 1],
            "lbl_b": [1, 0, 1, 0],
        })
        # Probabilities do NOT sum to 1 across columns (e.g. [0.8, 0.9] sums to 1.7)
        result = spec.evaluate_predictions(
            truth,
            {
                "target": [[0, 1], [0, 0], [1, 1], [1, 0]],
                "proba_target": [[0.1, 0.8], [0.2, 0.3], [0.9, 0.9], [0.8, 0.2]],
                "classes_target": ["a", "b"],
            },
        )
        self.assertGreater(result["primary_score"], 0.5)

    def test_multiclass_strictly_enforces_sum_to_one(self):
        spec = task("multiclass_classification", "logloss", "minimize")
        truth = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
        # Probabilities sum to 1.5 instead of 1.0 -> must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            spec.evaluate_predictions(
                truth,
                {
                    "target": [0, 1],
                    "proba_target": [[0.5, 1.0], [0.8, 0.7]],
                    "classes_target": [0, 1],
                },
            )
        self.assertIn("sum to one", str(ctx.exception))

    def test_binary_proba_does_not_require_sum_to_one(self):
        spec = task("binary_classification", "roc_auc", "maximize")
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
        # Positive class calibrated probabilities [0.1, 0.2, 0.8, 0.9] with arbitrary negative class column
        result = spec.evaluate_predictions(
            truth,
            {
                "target": [0, 0, 1, 1],
                "proba_target": [[0.0, 0.1], [0.0, 0.2], [0.0, 0.8], [0.0, 0.9]],
                "classes_target": [0, 1],
            },
        )
        self.assertEqual(result["primary_score"], 1.0)

    def test_native_categorical_transformer_edge_cases(self):
        df = pd.DataFrame({
            "num": [1.0, 2.5, 3.8],
            "int_cat": pd.Series([10, 20, 30], dtype="category"),
            "str_cat": pd.Series(["a", "b", "c"], dtype="category"),
            "mixed_obj": [1, "1", 2],
            "unhashable_obj": [[1, 2], [3], [1, 2]],
        })
        t = NativeCategoricalTransformer()
        out = t.fit_transform(df)
        # 1. Numeric untouched
        self.assertEqual(out["num"].tolist(), [1.0, 2.5, 3.8])
        # 2. Integer categories preserved (not converted to all-NaN!)
        self.assertEqual(out["int_cat"].tolist(), [10, 20, 30])
        # 3. String categories preserved
        self.assertEqual(out["str_cat"].tolist(), ["a", "b", "c"])
        # 4. Mixed object cleanly handled without duplicate categories error
        self.assertEqual(out["mixed_obj"].tolist(), ["1", "1", "2"])
        # 5. Unhashable objects handled without TypeError
        self.assertEqual(out["unhashable_obj"].tolist(), ["[1, 2]", "[3]", "[1, 2]"])

        # 6. Novel categories in transform become NaN gracefully
        test_df = pd.DataFrame({
            "num": [4.0],
            "int_cat": pd.Series([99], dtype="category"),
            "str_cat": pd.Series(["z"], dtype="category"),
            "mixed_obj": ["novel"],
            "unhashable_obj": ["[novel]"],
        })
        out_test = t.transform(test_df)
        self.assertTrue(pd.isna(out_test["int_cat"].iloc[0]))
        self.assertTrue(pd.isna(out_test["str_cat"].iloc[0]))
        self.assertTrue(pd.isna(out_test["mixed_obj"].iloc[0]))

    def test_contract_properties_and_incomplete(self):
        spec = task()
        self.assertEqual(spec.group_column, "group")
        self.assertEqual(spec.resolve_seed(), 13)
        self.assertIn("features", spec.inputs_dict)
        with self.assertRaises(ValueError):
            spec.evaluate_predictions(
                pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}), {"target": [0.0]}
            )

    def test_unsupported_metric(self):
        with self.assertRaises(ValueError):
            task(metric="nonsense").evaluate_predictions(
                pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}),
                {"target": [0.0, 1.0]},
            )


class MetricReferenceTests(unittest.TestCase):
    def test_regression_metrics_match_sklearn(self):
        truth = pd.DataFrame({"id": [1, 2, 3], "target": [1.0, 2.0, 4.0]})
        predicted = [1.2, 1.8, 3.0]
        for name, expected, direction in [
            ("MAE", metrics.mean_absolute_error(truth.target, predicted), "minimize"),
            ("R_squared", metrics.r2_score(truth.target, predicted), "maximize"),
            (
                "RMSE",
                np.sqrt(metrics.mean_squared_error(truth.target, predicted)),
                "minimize",
            ),
        ]:
            result = task(metric=name, direction=direction).evaluate_predictions(
                truth, {"target": predicted}
            )
            self.assertAlmostEqual(result["primary_score"], expected)

    def test_declared_non_sorted_probability_order(self):
        truth = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
        result = task(
            "binary_classification", "logloss", "minimize"
        ).evaluate_predictions(
            truth,
            {
                "target": [0, 1],
                "proba_target": [[0.1, 0.9], [0.8, 0.2]],
                "classes_target": [1, 0],
            },
        )
        self.assertAlmostEqual(result["primary_score"], -np.log(0.9 * 0.8) / 2)

    def test_multi_target_missing_is_not_dropped(self):
        spec = task("binary_classification", "accuracy", "maximize")
        spec.subtasks["second"] = SubtaskSpec(
            "second",
            "Second",
            "binary_classification",
            column="second",
            classes=[0, 1],
            primary_metric="accuracy",
        )
        truth = pd.DataFrame({"id": [1, 2], "target": [0, 1], "second": [1, 0]})
        with self.assertRaises(ValueError):
            spec.evaluate_predictions(truth, {"target": [0, 1]})
        result = spec.evaluate_predictions(truth, {"target": [0, 1], "second": [0, 0]})
        self.assertEqual(result["primary_score"], 0.75)


class TestInteractivePredictionContract(unittest.TestCase):
    """Wrapped interactive agent prediction contract tests."""

    @staticmethod
    def _interactive_spec():
        return UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "LifeMate Score",
                "direction": "maximize",
                "subtasks": {
                    "tool_calls": {
                        "column": "label",
                        "task_type": "interactive_agent",
                        "primary_metric": "LifeMate Score",
                        "weight": 1,
                    }
                },
            }
        })

    def test_ragged_episode_calls_preserve_payload(self):
        calls = [[], [{"name": "x", "arguments": {}}], [{"name": "x"}, {"name": "y"}]]
        truth = pd.DataFrame({"id": ["a", "b", "c"], "label": [{}, {}, {}]})
        batch = validate_prediction_contract(self._interactive_spec(), truth, {"tool_calls": calls})
        self.assertEqual(batch.labels["tool_calls"], calls)

    def test_interactive_missing_coverage_still_rejected(self):
        cases = [[[]], [[], None]]
        for calls in cases:
            with self.subTest(calls=calls):
                truth = pd.DataFrame({"id": ["a", "b"], "label": [{}, {}]})
                with self.assertRaises(ValueError) as ctx:
                    validate_prediction_contract(self._interactive_spec(), truth, {"tool_calls": calls})
                self.assertIn("Missing", str(ctx.exception))

    def test_interactive_custom_hook_is_authoritative(self):
        with patch.dict(
            sys.modules,
            {
                "_interactive_test_hook": SimpleNamespace(
                    score=lambda truth, predictions: {"LifeMate Score": 0.75}
                )
            },
        ):
            plugin = SimpleNamespace(evaluate_predictions=Mock(side_effect=AssertionError("generic scorer invoked")))
            truth = pd.DataFrame({"id": ["a", "b"], "label": [{}, {}]})
            result = score_predictions(
                self._interactive_spec(),
                plugin,
                SimpleNamespace(project_root=None),
                {"paths": {"metrics_evaluator": "_interactive_test_hook:score"}},
                truth,
                {"tool_calls": [[], [{"name": "x"}]]},
            )
            self.assertEqual(result["primary_score"], 0.75)
            plugin.evaluate_predictions.assert_not_called()


class TestEvaluationUsabilityAudit(unittest.TestCase):
    """Consolidated usability audit tests for evaluation, runner flags, and remote error extraction."""

    def test_import_aliases(self):
        from ai_harness.task_spec import UniversalTaskSpec as SpecAlias
        from ml_harness.runner import build_parser

        import ai_harness
        import ml_harness
        from harness import KaggleKernelManager as KKM
        from harness import kaggle_manager
        from harness.agent import AgentBridge
        from harness.endgame_submission import (
            SubmissionPreflightValidator,
            SubmissionValidator,
        )

        self.assertIsNotNone(ai_harness)
        self.assertIsNotNone(ml_harness)
        self.assertIs(SpecAlias, UniversalTaskSpec)
        self.assertTrue(callable(build_parser))
        self.assertIs(SubmissionValidator, SubmissionPreflightValidator)
        self.assertIs(kaggle_manager, KKM)
        self.assertIsNotNone(AgentBridge)

    def test_runner_build_parser_and_force_import(self):
        from harness.runner import build_parser

        parser = build_parser()
        args = parser.parse_args(["--config", "configs/test.yaml", "--force-import"])
        self.assertTrue(args.force_import)
        self.assertEqual(args.config, "configs/test.yaml")

    def test_evaluation_float_class_normalization(self):
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "target": {
                        "task_type": "binary_classification",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "weight": 1.0,
                        "classes": [0, 1],
                    }
                },
            }
        })
        truth_df = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 1.0, 0.0, 1.0]})
        preds = {"target": np.array([0.0, 1.0, 1.0, 1.0])}

        batch = validate_prediction_contract(spec, truth_df, preds)
        self.assertIsNotNone(batch)

        result = evaluate(spec, truth_df, preds)
        self.assertAlmostEqual(result["primary_score"], 0.75)

    def test_evaluation_continuous_probability_diagnostic(self):
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "target": {
                        "task_type": "binary_classification",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "weight": 1.0,
                        "classes": [0, 1],
                    }
                },
            }
        })
        truth_df = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
        preds = {"target": np.array([0.25, 0.85, 0.40])}

        with self.assertRaises(ValueError) as ctx:
            validate_prediction_contract(spec, truth_df, preds)

        msg = str(ctx.exception)
        self.assertIn("contain continuous probabilities", msg)
        self.assertIn("proba_target", msg)

    def test_wait_for_dataset_transient_error_retry(self):
        with patch("subprocess.run") as mock_run:
            call1 = MagicMock(returncode=1, stdout="", stderr="403 Client Error: Forbidden for url")
            call2 = MagicMock(returncode=1, stdout="", stderr="403 Client Error: Forbidden for url")
            call3 = MagicMock(returncode=0, stdout='"READY"', stderr="")
            mock_run.side_effect = [call1, call2, call3]

            KaggleKernelManager.wait_for_dataset(
                "owner/test-dataset", poll_interval=0.01, timeout_seconds=5.0
            )
            self.assertEqual(mock_run.call_count, 3)

    def test_get_kernel_error_log(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_kernel.log"
            log_entries = [
                {"stream_name": "stdout", "data": "Starting kernel execution\n"},
                {"stream_name": "stderr", "data": "Traceback (most recent call last):\n"},
                {"stream_name": "stderr", "data": "  File 'runner.py', line 10, in <module>\n"},
                {"stream_name": "stderr", "data": "ValueError: Test intentional failure\n"},
            ]
            lines = [json.dumps(e) for e in log_entries]
            log_file.write_text("\n".join(lines), encoding="utf-8")

            with patch("subprocess.run") as mock_run:
                def fake_run(cmd, capture_output=True, text=True, timeout=60):
                    dest_dir = cmd[cmd.index("-p") + 1]
                    import shutil
                    shutil.copy(log_file, Path(dest_dir) / "test_kernel.log")
                    return MagicMock(returncode=0, stdout="", stderr="")

                mock_run.side_effect = fake_run
                err_log = KaggleKernelManager.get_kernel_error_log("owner/test-kernel")
                self.assertIsNotNone(err_log)
                self.assertIn("ValueError: Test intentional failure", err_log)
                self.assertIn("Traceback", err_log)


class TestUniversalMetricsAndTuning(unittest.TestCase):
    """Consolidated metrics across forecasting, IR, CV, speech, and probability tuning."""

    def test_forecasting_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 32.0, 38.0])

        rmsle = compute_rmsle(y_true, y_pred)
        self.assertGreater(rmsle, 0.0)
        self.assertLess(rmsle, 0.2)

        smape = compute_smape(y_true, y_pred)
        self.assertGreater(smape, 0.0)
        self.assertLess(smape, 0.2)

        wape = compute_wape(y_true, y_pred)
        self.assertGreater(wape, 0.0)
        self.assertLess(wape, 0.1)

        pinball = compute_pinball_loss(y_true, y_pred, alpha=0.5)
        self.assertGreater(pinball, 0.0)

        pearson = compute_pearson(y_true, y_pred)
        self.assertGreater(pearson, 0.95)

    def test_ranking_and_ir_metrics(self):
        y_true = np.array([0, 0, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.9, 0.4])
        self.assertAlmostEqual(compute_mrr(y_true, y_pred), 1.0)

        y_pred_2 = np.array([0.1, 0.9, 0.5, 0.2])
        self.assertAlmostEqual(compute_mrr(y_true, y_pred_2), 0.5)

    def test_computer_vision_segmentation_rle_and_dice(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:6] = 1

        rle = mask_to_rle(mask)
        self.assertIsInstance(rle, str)
        self.assertGreater(len(rle), 0)

        recovered = rle_to_mask(rle, (10, 10))
        np.testing.assert_array_equal(mask, recovered)

        dice = compute_dice_score(mask, recovered)
        self.assertAlmostEqual(dice, 1.0, places=4)
        iou = compute_iou_score(mask, recovered)
        self.assertAlmostEqual(iou, 1.0, places=4)

        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_box_iou(box1, box2), 1.0)
        box3 = [60, 60, 100, 100]
        self.assertAlmostEqual(compute_box_iou(box1, box3), 0.0)

    def test_speech_and_text_generation_metrics(self):
        refs = ["the quick brown fox", "hello world"]
        hyps = ["the fast brown fox", "hello world"]

        wer = compute_wer(refs, hyps)
        self.assertGreater(wer, 0.0)
        self.assertLess(wer, 0.3)

        cer = compute_cer(refs, hyps)
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 0.3)

        bleu = compute_bleu(refs, hyps)
        self.assertGreater(bleu, 0.4)

        em = compute_exact_match(refs, hyps)
        self.assertEqual(em, 0.5)

    def test_default_metric_direction(self):
        self.assertEqual(default_metric_direction("rmsle"), "minimize")
        self.assertEqual(default_metric_direction("wer"), "minimize")
        self.assertEqual(default_metric_direction("cer"), "minimize")
        self.assertEqual(default_metric_direction("pinball_loss"), "minimize")
        self.assertEqual(default_metric_direction("logloss"), "minimize")
        self.assertEqual(default_metric_direction("dice"), "maximize")
        self.assertEqual(default_metric_direction("iou"), "maximize")
        self.assertEqual(default_metric_direction("bleu"), "maximize")
        self.assertEqual(default_metric_direction("pr_auc"), "maximize")
        self.assertEqual(default_metric_direction("balanced_accuracy"), "maximize")

    def test_evaluate_segmentation_and_speech_tasks(self):
        seg_spec_dict = {
            "task": {
                "name": "lung_seg",
                "task_type": "segmentation",
                "primary_metric": "dice",
                "id_column": "id",
                "subtasks": {
                    "mask": {
                        "task_type": "segmentation",
                        "primary_metric": "dice",
                    }
                },
            }
        }
        spec = UniversalTaskSpec.from_dict(seg_spec_dict)
        truth = pd.DataFrame({"id": [1, 2], "mask": [[1, 1, 0, 0], [0, 1, 1, 0]]})
        preds = {"mask": [[1, 1, 0, 0], [0, 1, 0, 0]]}
        res = evaluate(spec, truth, preds)
        self.assertIn("primary_score", res)
        self.assertGreater(res["primary_score"], 0.7)

        asr_spec_dict = {
            "task": {
                "name": "voice_transcription",
                "task_type": "speech_recognition",
                "primary_metric": "wer",
                "direction": "minimize",
                "id_column": "id",
                "subtasks": {
                    "text": {
                        "task_type": "speech_recognition",
                        "primary_metric": "wer",
                        "direction": "minimize",
                    }
                },
            }
        }
        asr_spec = UniversalTaskSpec.from_dict(asr_spec_dict)
        truth_asr = pd.DataFrame({"id": [1, 2], "text": ["hello world", "good morning"]})
        preds_asr = {"text": ["hello world", "good evening"]}
        res_asr = evaluate(asr_spec, truth_asr, preds_asr)
        self.assertIn("primary_score", res_asr)
        self.assertLess(res_asr["primary_score"], 0.5)

    def test_binary_threshold_optimizer(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.4, 0.45, 0.6, 0.8, 0.9, 0.3, 0.25, 0.7])

        thresh, f1 = ThresholdOptimizer.optimize_binary_threshold(y_true, y_prob)
        self.assertGreater(thresh, 0.0)
        self.assertLess(thresh, 1.0)
        self.assertGreaterEqual(f1, 0.8)

    def test_probability_calibrator(self):
        y_prob = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.9])
        y_true = np.array([0, 0, 0, 1, 1, 1])

        calib = ProbabilityCalibrator(method="sigmoid")
        calib.fit(y_prob, y_true)
        calibrated = calib.transform(y_prob)

        self.assertEqual(calibrated.shape, (6, 1))
        self.assertLess(calibrated[0, 0], calibrated[-1, 0])


if __name__ == "__main__":
    unittest.main()
