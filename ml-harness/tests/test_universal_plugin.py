"""Integration test proving Universal Pluggability of ML-Harness.

Demonstrates that the harness can be dropped into an entirely new competition
(e.g., SentimentPulse binary sentiment classification) with ZERO code modifications
to core harness, and execute:

1. Declarative TaskSpec parsing (holdout validation, custom schema).
2. AgentBridge context generation (zero legacy task leakage).
3. Epistemic GroundingGate config verification (proper schema).
4. Dynamic DiagnosticsEngine report generation (confusion matrix, label distribution).
5. TreeJournal node synchronization and UCB scoring.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.agent_bridge import AgentBridge
from harness.diagnostics import DiagnosticsEngine
from harness.grounding_gate import GroundingGate
from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import TreeJournal


class TestUniversalPlugin(unittest.TestCase):
    """Verifies that ML-Harness operates seamlessly on arbitrary competition domains."""

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
        # Should not raise
        bridge = AgentBridge(task_spec_path=self.task_yaml)
        # Competition spec should reference the new task
        spec = bridge.comp_state.spec
        self.assertEqual(spec.get("name"), "SentimentPulse-2027")
        # No StereoQueerEval tokens in the loaded spec
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

        import pandas as pd
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
        # The sentiment column should be auto-detected
        self.assertIn("sentiment", diag["target_confusions"])
        self.assertIn("Confusion Matrix", md_report)
        # Must not contain any StereoQueerEval-specific terminology
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

        best = journal.get_best_node()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "exp001_ngram_feat")

        # Test UCB recommendation
        rec = journal.recommend_next_branch()
        self.assertTrue(len(rec) > 0)
        self.assertIn("parent_id", rec)

    def test_submission_manager_generic_scores(self):
        """SubmissionManager must support arbitrary competition metrics without hardcoded subtasks."""
        from harness.submission_manager import SubmissionManager
        sub_dir = self.test_dir / "submissions"
        mgr = SubmissionManager(submissions_dir=sub_dir)

        # Create mock candidate and baseline predictions
        cand_csv = self.test_dir / "cand_preds.csv"
        base_csv = self.test_dir / "base_preds.csv"
        cand_csv.write_text("review_id,pred_sentiment\nrev_0,positive\nrev_1,negative\n", encoding="utf-8")
        base_csv.write_text("review_id,pred_sentiment\nrev_0,negative\nrev_1,positive\n", encoding="utf-8")

        entry = mgr.register_submission(
            exp_id="exp001_ngram",
            candidate_csv=cand_csv,
            baseline_csv=base_csv,
            notes="Testing generic metric registration",
            min_diff_pct=0.0,
        )
        self.assertEqual(entry.status, "LOCAL_VERIFIED")

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
        harness_dir = PROJECT_ROOT / "harness"
        banned_tokens = [
            'IDENTITIES = ["l", "g"',
            'StereoQueerEval_id',
            'subtask_a_macro_f1',
            'yt_title_group_split',
            'subtask_a_macro',
            'subtask_b_macro',
            'c_target_type_macro_f1_oracle_hate',
            'pred_stereotype',
            'pred_hate_speech',
            'pred_target_type',
            'pred_identities',
            'yt_title',
            'exp001_baseline_tfidf',
            'from src.data import IDENTITIES',
            'from src.metrics import evaluate_predictions',
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
        tools_dir = PROJECT_ROOT / "tools"
        for py_file in tools_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for token in ["StereoQueerEval_id", "pred_stereotype", "pred_hate_speech", "yt_title"]:
                self.assertNotIn(
                    token,
                    content,
                    f"Banned token '{token}' found in tools/{py_file.name}",
                )

        # dashboard/src must also be free of task-specific identifiers
        dashboard_src = PROJECT_ROOT / "dashboard" / "src"
        if dashboard_src.exists():
            for src_file in dashboard_src.rglob("*.tsx"):
                content = src_file.read_text(encoding="utf-8", errors="ignore")
                for token in ["StereoQueerEval", "yt_title", "yt_comment", "pred_stereotype"]:
                    self.assertNotIn(
                        token,
                        content,
                        f"Banned token '{token}' found in dashboard/src/{src_file.name}",
                    )


if __name__ == "__main__":
    unittest.main()

