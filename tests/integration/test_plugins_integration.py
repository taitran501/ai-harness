"""Integration test suite consolidating TaskPlugin contracts, docs, reference adapters, and universal pluggability.

Consolidates:
- tests/test_plugin_contract_validation.py
- tests/test_plugin_docs_and_example.py
- tests/test_universal_plugin.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
examples_task_plugin = str(ROOT / "examples" / "task_plugin")
if examples_task_plugin not in sys.path:
    sys.path.insert(0, examples_task_plugin)

from harness.agent_bridge import AgentBridge
from harness.diagnostics import DiagnosticsEngine
from harness.grounding_gate import GroundingGate
import harness.plugins as plugins
from harness.plugins import (
    DeclarativeTaskPlugin,
    PluginCapabilityError,
    TaskPlugin,
    TaskPluginContext,
    TaskPluginError,
    load_task_plugin,
    validate_plugin_contract,
)
from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import TreeJournal
from tests.fixtures import make_workspace, task_document  # noqa: F401
from tests.evidence_fixtures import seal_fixture  # noqa: F401


# ---------------------------------------------------------------------------
# Dummy plugin implementations for contract validation
# ---------------------------------------------------------------------------


class DummyValidPlugin(TaskPlugin):
    plugin_id = "valid_dummy"

    def evaluate_predictions(self, context, ground_truth, predictions):
        return {"primary_score": 1.0}

    def load_training_data(self, context):
        return None


class BadSignaturePlugin(TaskPlugin):
    plugin_id = "bad_sig"

    # evaluate_predictions in TaskPlugin expects (self, context, ground_truth, predictions)
    # Here only 1 argument is accepted!
    def evaluate_predictions(self, context):
        return {}


class NonCallableHookPlugin(TaskPlugin):
    plugin_id = "non_callable"

    # evaluate_predictions is an int, not a method
    evaluate_predictions = 123  # type: ignore[assignment]


class BadCapabilitiesPlugin(TaskPlugin):
    plugin_id = "bad_caps"

    def capabilities(self):  # type: ignore[override]
        return "not a dict"


# ---------------------------------------------------------------------------
# Contract validation tests
# ---------------------------------------------------------------------------


class TestPluginContractValidation(unittest.TestCase):
    def setUp(self):
        self.ctx = TaskPluginContext(
            Path("/mock"), Path("/mock/task.yaml"), SimpleNamespace(), "test_plugin"
        )

    def test_declarative_plugin_passes_contract(self):
        plugin = DeclarativeTaskPlugin(self.ctx)
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertEqual(errors, [])

    def test_valid_custom_plugin_passes(self):
        plugin = DummyValidPlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=True)
        self.assertEqual(errors, [])

    def test_bad_signature_detected_and_raised(self):
        plugin = BadSignaturePlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertTrue(any("Hook 'evaluate_predictions' signature mismatch" in e for e in errors))

        with self.assertRaises(PluginCapabilityError) as cm:
            validate_plugin_contract(plugin, raise_on_error=True)
        self.assertIn("evaluate_predictions", str(cm.exception))

    def test_non_callable_hook_detected(self):
        plugin = NonCallableHookPlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertTrue(any("is not callable" in e for e in errors))

    def test_invalid_capabilities_detected(self):
        plugin = BadCapabilitiesPlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertTrue(any("capabilities() must return a dict" in e for e in errors))

    def test_load_task_plugin_enforces_contract(self):
        ep = mock.Mock()
        ep.name = "bad_sig"
        ep.load.return_value = BadSignaturePlugin
        with mock.patch.object(plugins, "_plugin_entry_points", return_value=[ep]):
            with self.assertRaises(PluginCapabilityError) as cm:
                load_task_plugin("bad_sig", self.ctx)
            self.assertIn("failed contract validation", str(cm.exception))


# ---------------------------------------------------------------------------
# Documentation consistency tests
# ---------------------------------------------------------------------------

DOC = ROOT / "docs" / "plugin_architecture.md"
STALE_HOOKS = ("load_train_data", "load_test_data", "run_eda")


def _documented_hooks() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    return re.findall(r"^\| `(\w+)\(\)` \|", text, flags=re.M)


class TestPluginDocs(unittest.TestCase):
    def test_every_documented_hook_exists_on_taskplugin(self):
        hooks = _documented_hooks()
        self.assertTrue(hooks, "hook table not found in plugin_architecture.md")
        for hook in hooks:
            self.assertTrue(
                callable(getattr(TaskPlugin, hook, None)),
                f"docs mention `{hook}()` but TaskPlugin has no such hook",
            )

    def test_docs_do_not_mention_stale_hooks_as_hooks(self):
        for hook in _documented_hooks():
            self.assertNotIn(hook, STALE_HOOKS)

    def test_docs_reference_example_adapter(self):
        self.assertIn("examples/task_plugin/example_adapter.py", DOC.read_text(encoding="utf-8"))
        self.assertTrue((ROOT / "examples" / "task_plugin" / "example_adapter.py").is_file())


# ---------------------------------------------------------------------------
# Reference adapter tests
# ---------------------------------------------------------------------------


class TestExampleAdapter(unittest.TestCase):
    def setUp(self):
        import example_adapter

        self.mod = example_adapter
        self.root = Path(tempfile.mkdtemp(prefix="aih-example-plugin-"))
        (self.root / "dataset").mkdir()
        pd.DataFrame(
            {"id": range(8), "label": [0, 1, 0, 1, 0, 1, 0, 0]}
        ).to_csv(self.root / "dataset" / "train.csv", index=False)
        self.ctx = TaskPluginContext(
            self.root, self.root / "task.yaml", SimpleNamespace(), "example_adapter"
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _load(self):
        ep = mock.Mock()
        ep.name = "example_adapter"
        ep.load.return_value = self.mod.ExampleAdapterPlugin
        with mock.patch.object(plugins, "_plugin_entry_points", return_value=[ep]):
            return load_task_plugin("example_adapter", self.ctx)

    def test_loads_and_declares_expected_capabilities(self):
        caps = self._load().capabilities()
        for key in (
            "custom_data_loader",
            "custom_metrics",
            "custom_validation_split",
            "submission_builder",
            "submission_validator",
        ):
            self.assertTrue(caps[key], key)
        self.assertFalse(caps["custom_model_factory"])

    def test_data_splits_and_custom_evaluation(self):
        plugin = self._load()
        data = plugin.load_training_data(self.ctx)
        splits = plugin.build_validation_splits(self.ctx, data, "kfold", {"num_folds": 2})
        self.assertEqual(len(splits), 2)
        for _, train, val in splits:
            self.assertFalse(set(train["id"]) & set(val["id"]))
        # Defer unrecognized strategies to core harness defaults
        self.assertIsNone(plugin.build_validation_splits(self.ctx, data, "group_kfold", {}))

        # Perfect predictions -> 1.0; all-zero predictions -> recall (1.0, 0.0) -> 0.5
        perfect = plugin.evaluate_predictions(
            self.ctx, data, {"pred_label": data["label"].tolist()}
        )
        self.assertAlmostEqual(perfect["primary_score"], 1.0)
        zeros = plugin.evaluate_predictions(self.ctx, data, {"pred_label": [0] * len(data)})
        self.assertAlmostEqual(zeros["primary_score"], 0.5)

    def test_submission_roundtrip_and_validation_failure(self):
        plugin = self._load()
        preds = pd.DataFrame({"id": ["a", "b"], "pred_label": [0, 1]})
        out = plugin.build_submission(self.ctx, preds, self.root / "sub.csv")
        plugin.validate_submission(self.ctx, out)

        bad = self.root / "bad.csv"
        pd.DataFrame({"id": ["a"], "label": [7]}).to_csv(bad, index=False)
        with self.assertRaises(TaskPluginError):
            plugin.validate_submission(self.ctx, bad)


# ---------------------------------------------------------------------------
# Universal pluggability tests
# ---------------------------------------------------------------------------


class TestUniversalPlugin(unittest.TestCase):
    """Verifies that AI-Harness operates seamlessly on arbitrary competition domains."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_harness_plugin_"))
        self.dataset_dir = self.test_dir / "mock_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create mock dataset: SentimentPulse (e-commerce review sentiment)
        self.train_csv = self.dataset_dir / "train.csv"
        self.dev_csv = self.dataset_dir / "dev.csv"

        rows_train = [
            {
                "review_id": f"rev_{i}",
                "customer_text": f"Review text sample {i}",
                "sentiment": "positive" if i % 2 == 0 else "negative",
            }
            for i in range(40)
        ]
        rows_dev = [
            {
                "review_id": f"dev_{i}",
                "customer_text": f"Evaluation text {i}",
                "sentiment": "positive" if i % 3 == 0 else "negative",
            }
            for i in range(20)
        ]

        for csv_path, rows in [(self.train_csv, rows_train), (self.dev_csv, rows_dev)]:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["review_id", "customer_text", "sentiment"])
                writer.writeheader()
                writer.writerows(rows)

        # 2. Create mock task.yaml
        self.task_yaml = self.test_dir / "task.yaml"
        task_data = {
            "name": "SentimentPulse-2027",
            "platform": "Kaggle",
            "task_type": "binary_classification",
            "modality": "text",
            "primary_metric": "macro_f1",
            "subtasks": [
                {
                    "id": "sentiment",
                    "name": "Sentiment",
                    "metric": "macro_f1",
                    "weight": 1.0,
                    "target_column": "sentiment",
                    "classes": ["positive", "negative"],
                }
            ],
            "data": {
                "train_path": str(self.train_csv).replace("\\", "/"),
                "id_column": "review_id",
                "text_column": "customer_text",
                "target_column": "sentiment",
            },
            "validation": {
                "strategy": "holdout",
                "dev_split_name": str(self.dev_csv).replace("\\", "/"),
                "probe_samples": 5,
            },
        }
        with open(self.task_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(task_data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_universal_task_spec_loading(self):
        """UniversalTaskSpec must dynamically resolve custom schemas and validation strategies."""
        spec = UniversalTaskSpec.from_yaml(self.task_yaml)
        self.assertEqual(spec.name, "SentimentPulse-2027")
        self.assertEqual(spec.platform, "Kaggle")
        self.assertEqual(spec.primary_target, "sentiment")
        self.assertEqual(spec.validation.strategy, "holdout")
        self.assertEqual(spec.validation.probe_samples, 5)

        # Verify dataset loader with absolute path
        dev_df = spec.load_dataset_split(str(self.dev_csv))
        self.assertEqual(len(dev_df), 20)
        self.assertIn("review_id", dev_df.columns)
        self.assertIn("sentiment", dev_df.columns)

    def test_agent_bridge_accepts_task_spec_path(self):
        """AgentBridge must accept a custom task_spec_path and load spec for new competition."""
        bridge = AgentBridge(project_root=self.test_dir, task_spec_path=self.task_yaml)
        spec = bridge.comp_state.spec
        self.assertEqual(spec.get("name"), "SentimentPulse-2027")
        spec_str = json.dumps(spec)
        self.assertNotIn("StereoQueerEval", spec_str)
        self.assertNotIn("yt_title", spec_str)

    def test_grounding_gate_verification(self):
        """GroundingGate must accept properly justified configurations for arbitrary domains."""
        test_config = {
            "experiment": {
                "id": "exp001_tfidf_logistic",
                "branch": "feat",
                "parent_id": "exp000_baseline",
                "description": "TF-IDF n-grams with LogisticRegression",
                "model_family": "logistic_regression",
                "problem_addressed": "Baseline accuracy is below the 0.75 threshold for sentiment detection.",
                "hypothesis": "Character-level n-grams (1-3) with L2-C=0.1 will reduce confusions by >=3 pp.",
                "intervention": "Control: word unigram TF-IDF. Treatment: char(1-3)+word(1-2) TF-IDF stack.",
            },
            "grounding": {
                "source_type": "empirical_journal",
                "source_id": "Manning_IR_2008_textbook",
                "retrieval_summary": "Character n-gram TF-IDF consistently outperforms word-only on noisy review text.",
                "verified_via": "local_registry",
            },
            "assumption_register": {
                "known_facts": ["Training data contains e-commerce reviews, predominantly English"],
                "assumptions": ["Character n-grams capture morphological negation markers effectively"],
                "uncertainty_level": "low",
                "verification_probe": "Check confusion matrix on dev set for positive/negative confusion rates",
            },
        }

        passed, errors, receipt = GroundingGate.verify_config(test_config, raise_on_error=False)
        self.assertTrue(passed, f"GroundingGate rejected valid config with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_dynamic_diagnostics_generation(self):
        """DiagnosticsEngine must compute confusion matrix and slices without domain hardcoding."""
        spec = UniversalTaskSpec.from_yaml(self.task_yaml)
        dev_df = spec.load_dataset_split(str(self.dev_csv))

        preds_df = dev_df.copy()
        # 80% accuracy simulation: flip every 5th prediction
        preds_df["pred_sentiment"] = [
            row["sentiment"]
            if i % 5 != 0
            else ("negative" if row["sentiment"] == "positive" else "positive")
            for i, (_, row) in enumerate(preds_df.iterrows())
        ]

        diag = DiagnosticsEngine.generate_diagnostics(eval_df=preds_df, oof_df=preds_df)
        md_report = DiagnosticsEngine.render_markdown_report(diag, exp_id="exp001")

        self.assertIsInstance(diag, dict)
        self.assertIn("target_confusions", diag)
        self.assertIn("sentiment", diag["target_confusions"])
        self.assertIn("Confusion Matrix", md_report)
        self.assertNotIn("StereoQueerEval", md_report)
        self.assertNotIn("LGBTQIA", md_report)
        self.assertNotIn("yt_title", md_report)

    def test_solution_tree_composite_and_ucb(self):
        """TreeJournal must calculate composite score and UCB recommendations dynamically."""
        journal_path = self.test_dir / "journal.json"

        journal = TreeJournal(
            journal_path=journal_path,
            task_spec=UniversalTaskSpec.from_yaml(self.task_yaml),
        )

        # Add root baseline node
        root_node = journal.add_node(
            node_id="exp000_baseline",
            parent_id=None,
            branch="baseline",
            hypothesis="Baseline tfidf sentiment model",
            score=0.7500,
            metrics={"macro_f1": 0.7500, "overall_composite": 0.7500},
        )
        self.assertEqual(root_node.get_score(), 0.7500)

        # Add child node
        child_node = journal.add_node(
            node_id="exp001_ngram_feat",
            parent_id="exp000_baseline",
            branch="feat",
            hypothesis="Added subword n-grams: char(1-3) + word(1-2)",
            score=0.7950,
            metrics={"macro_f1": 0.7950, "overall_composite": 0.7950},
        )
        self.assertEqual(child_node.get_score(), 0.7950)

        root_node.eligible = root_node.verified = True
        child_node.eligible = child_node.verified = True
        journal.append(root_node)
        journal.append(child_node)
        best = journal.get_best_node()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "exp001_ngram_feat")

        # Test UCB recommendation
        rec = journal.recommend_next_branch()
        self.assertTrue(len(rec) > 0)
        self.assertIn("parent_id", rec)

    def test_submission_manager_generic_scores(self):
        """SubmissionManager must support arbitrary competition metrics without hardcoded subtasks."""
        from harness.submission_manager import SubmissionEntry, SubmissionManager

        sub_dir = self.test_dir / "submissions"
        mgr = SubmissionManager(submissions_dir=sub_dir)

        cand_csv = self.test_dir / "cand_preds.csv"
        base_csv = self.test_dir / "base_preds.csv"
        cand_csv.write_text("review_id,pred_sentiment\nrev_0,positive\nrev_1,negative\n", encoding="utf-8")
        base_csv.write_text("review_id,pred_sentiment\nrev_0,negative\nrev_1,positive\n", encoding="utf-8")

        entry = SubmissionEntry(
            id="fixture-sub",
            exp_id="exp001_ngram",
            created_at="fixture",
            status="LOCAL_VERIFIED",
            local_cv_scores={"macro_f1": 0.8},
        )
        mgr.submissions[entry.id] = entry

        # Record arbitrary leaderboard scores (e.g. Kaggle competition metrics)
        scored_entry = mgr.record_official_score(
            sub_id=entry.id,
            primary_score=0.8842,
            scores={"accuracy": 0.9120, "auc_roc": 0.9450},
            split_type="public",
            notes="Kaggle public LB score",
        )
        self.assertEqual(scored_entry.status, "SCORED")
        self.assertIn(0.8842, scored_entry.official_scores.values())
        self.assertEqual(scored_entry.official_scores["accuracy"], 0.9120)
        self.assertEqual(scored_entry.official_scores["auc_roc"], 0.9450)
        self.assertNotIn("subtask_a_macro", scored_entry.official_scores)

    def test_no_stereo_queer_eval_imports_in_harness(self):
        """Harness core modules must not import or reference StereoQueerEval-specific hardcoded identifiers."""
        harness_dir = ROOT / "harness"
        banned_tokens = [
            'IDENTITIES = ["l", "g"',
            "StereoQueerEval_id",
            "subtask_a_macro_f1",
            "yt_title_group_split",
            "subtask_a_macro",
            "subtask_b_macro",
            "c_target_type_macro_f1_oracle_hate",
            "pred_stereotype",
            "pred_hate_speech",
            "pred_target_type",
            "pred_identities",
            "yt_title",
            "exp001_baseline_tfidf",
            "from src.data import IDENTITIES",
            "from src.metrics import evaluate_predictions",
            "stereoqueereval.yaml",
            "sqe-",
        ]
        for py_file in harness_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for token in banned_tokens:
                self.assertNotIn(
                    token,
                    content,
                    f"Banned hardcoded token '{token}' found in harness/{py_file.name}",
                )

        # tools/ python files must also be free of domain leaks
        tools_dir = ROOT / "tools"
        for py_file in tools_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for token in [
                "StereoQueerEval_id",
                "pred_stereotype",
                "pred_hate_speech",
                "yt_title",
                "stereoqueer",
                "sqe-",
            ]:
                self.assertNotIn(
                    token,
                    content,
                    f"Banned token '{token}' found in tools/{py_file.name}",
                )

        # dashboard/src must also be free of task-specific identifiers
        dashboard_src = ROOT / "dashboard" / "src"
        if dashboard_src.exists():
            for src_file in dashboard_src.rglob("*.tsx"):
                content = src_file.read_text(encoding="utf-8", errors="ignore")
                for token in [
                    "StereoQueerEval",
                    "yt_title",
                    "yt_comment",
                    "pred_stereotype",
                    "stereoqueer",
                    "sqe-",
                    "sqe_",
                ]:
                    self.assertNotIn(
                        token,
                        content,
                        f"Banned token '{token}' found in dashboard/src/{src_file.name}",
                    )


if __name__ == "__main__":
    unittest.main()
