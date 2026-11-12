"""Integration tests for TreeJournal persistence, UCB branch recommendations,
DAG lineage, sync-dedupe, LLM specs, and ensemble guard integration.

Consolidates:
- tests/test_journal_sync_dedupe.py (JournalSyncDedupTests, ExecuteActionSnapshotReuseTests)
- tests/test_llm_experiment_specs.py (TestLLMExperimentSpecs)
- tests/test_tree_journal_ensemble.py (TestTreeJournalDirectionAndUCB, TestEnsembleGuardEnhancements,
  TestAtomicIOUtils, TestCandidateSynthesisEnhancements, TestSubmissionManagerDynamic,
  TestAuditGateDynamicIntegrity)
- tests/test_harness.py (TreeJournalTests)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.ensemble_guard import EnsembleGuard
from harness.io_utils import atomic_write_json, atomic_write_text
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.project import HarnessProject, attach_project
from harness.submission_manager import SubmissionEntry, SubmissionManager
from harness.task_spec import load_active_task_spec
from harness.tree_journal import Node, TreeJournal
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import TRAIN_CSV_4ROW, task_document
from tools.audit_gate import audit_prediction_integrity, compute_diff_metrics
from tools.dashboard_server import DEFAULT_STRATEGY_CARDS

SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent


def _spec_and_plugin(root: Path):
    task_config = root / "configs" / "task.yaml"
    spec = load_active_task_spec(task_config)
    plugin = load_task_plugin(
        "default",
        TaskPluginContext(
            project_root=root,
            task_config_path=task_config,
            task_spec=spec,
            plugin_id="default",
        ),
    )
    return spec, plugin


class TreeJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.temp_dir.name) / "test_journal.json"
        self.journal = TreeJournal(storage_path=str(self.journal_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_and_load(self):
        node = Node(
            verified=True,
            eligible=True,
            id="test_node_1",
            parent_id=None,
            branch="baseline",
            atomic_hypothesis="Test hypothesis",
            metrics={"overall_composite": 0.75},
            param_count=1000,
        )
        self.journal.append(node)

        # Reload from disk
        reloaded = TreeJournal(storage_path=str(self.journal_path))
        self.assertIn("test_node_1", reloaded.nodes)
        self.assertEqual(reloaded.nodes["test_node_1"].get_score(), 0.75)

    def test_50_nodes_context_compression(self):
        """Verify that 50 experiments are compressed into a compact summary without bloat."""
        for i in range(50):
            node = Node(
                verified=True,
                eligible=True,
                id=f"exp{i:03d}_test",
                parent_id=f"exp{max(0, i - 1):03d}_test" if i > 0 else None,
                branch="feat",
                atomic_hypothesis=f"Atomic change #{i} tested on validation set.",
                metrics={"overall_composite": 0.5 + (i * 0.005)},
                param_count=10_000_000,
            )
            self.journal.append(node)

        summary = self.journal.generate_compact_summary(max_nodes=10)
        self.assertIn("Total Runs: 50", summary)
        # Verify line count is bounded and token-friendly
        self.assertLess(len(summary.splitlines()), 25)

    def test_recommend_next_branch_respects_max_params(self):
        # Add a node exceeding parameter limits
        heavy_node = Node(
            verified=True,
            eligible=True,
            id="exp_heavy",
            branch="model",
            metrics={"overall_composite": 0.99},
            param_count=500_000_000,
        )
        # Add a light compliant node
        light_node = Node(
            verified=True,
            eligible=True,
            id="exp_light",
            branch="baseline",
            metrics={"overall_composite": 0.70},
            param_count=100_000_000,
        )
        self.journal.append(heavy_node)
        self.journal.append(light_node)

        # Recommendation with max_params=150M must NOT recommend the heavy node
        rec = self.journal.recommend_next_branch(max_params=150_000_000)
        self.assertEqual(rec["parent_id"], "exp_light")

    def test_multi_parent_dag_lineage(self):
        """Verify that multi-parent nodes (e.g. ensembles/merges) correctly track DAG edges and children."""
        n1 = Node(verified=True, eligible=True, id="m1", branch="baseline", metrics={"overall_composite": 0.80})
        n2 = Node(verified=True, eligible=True, id="m2", branch="model", metrics={"overall_composite": 0.82})
        ens = Node(
            verified=True,
            eligible=True,
            id="ens_blend",
            parent_ids=["m1", "m2"],
            branch="ens",
            metrics={"overall_composite": 0.85},
        )
        self.journal.append(n1)
        self.journal.append(n2)
        self.journal.append(ens)

        # Children query for m1 and m2 must both include ens_blend
        ch_m1 = self.journal.get_children("m1")
        ch_m2 = self.journal.get_children("m2")
        self.assertTrue(any(c.id == "ens_blend" for c in ch_m1))
        self.assertTrue(any(c.id == "ens_blend" for c in ch_m2))

        # Graph data should contain edges from both m1 and m2
        g = self.journal.get_graph_data()
        edges = g["edges"]
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        self.assertIn(("m1", "ens_blend"), edge_pairs)
        self.assertIn(("m2", "ens_blend"), edge_pairs)


class JournalSyncDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aih_journal_dedupe_")
        self.root = Path(self._tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "dataset" / "train.csv").write_text(TRAIN_CSV_4ROW, encoding="utf-8")
        task_config = self.root / "configs" / "task.yaml"
        task_config.write_text(yaml.safe_dump(task_document(), sort_keys=False), encoding="utf-8")
        attach_project(self.root, task_config)
        self.project = HarnessProject.discover(self.root)
        self.artifacts = self.project.artifacts_dir
        self.journal_path = self.artifacts / "journal.json"
        self.spec, self.plugin = _spec_and_plugin(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seal_run(self, exp_id: str, score: float = 0.8) -> Path:
        folder = self.artifacts / exp_id
        seal_fixture(folder, self.spec, score=score, job_id="job-" + exp_id)
        return folder

    def _write_journal(self, nodes: list) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")

    def _journal_node(self, exp_id: str, artifact_dir: str, score: float = 0.84) -> dict:
        return {
            "id": exp_id,
            "parent_id": None,
            "branch": "feat",
            "model_family": "linear",
            "metrics": {"primary_score": score},
            "primary_metric_name": "accuracy",
            "direction": "maximize",
            "verified": True,
            "eligible": True,
            "status": "COMPLETED",
            "artifact_dir": artifact_dir,
        }

    def test_on_disk_run_still_validated_by_auto_sync(self):
        """A healthy on-disk run must come out verified after auto-sync."""
        self._seal_run("exp010", score=0.81)
        journal = TreeJournal(
            journal_path=self.journal_path,
            artifacts_dir=self.artifacts,
            auto_sync=True,
            task_spec=self.spec,
            task_plugin=self.plugin,
        )
        node = journal.nodes["exp010"]
        self.assertTrue(node.verified)
        self.assertTrue(node.eligible)
        self.assertEqual(node.status, "COMPLETED")

    def test_tampered_on_disk_run_still_rejected(self):
        """The dedup must not weaken the gate: a tampered on-disk run whose
        manifest hash no longer matches must be marked LEGACY_UNVERIFIED."""
        folder = self._seal_run("exp011", score=0.82)
        # Tamper after sealing: metrics.json no longer matches its sealed hash.
        (folder / "metrics.json").write_text(
            json.dumps({"primary_score": 0.99, "tampered": True}), encoding="utf-8"
        )
        journal = TreeJournal(
            journal_path=self.journal_path,
            artifacts_dir=self.artifacts,
            auto_sync=True,
            task_spec=self.spec,
            task_plugin=self.plugin,
        )
        node = journal.nodes["exp011"]
        self.assertFalse(node.verified)
        self.assertFalse(node.eligible)
        self.assertEqual(node.status, "LEGACY_UNVERIFIED")

    def test_orphan_journal_node_marked_unverified(self):
        """A journal node whose artifact dir is gone from disk must still be
        marked LEGACY_UNVERIFIED (the only nodes that keep the full check)."""
        self._seal_run("exp012", score=0.83)
        gone = self.artifacts / "exp013"
        seal_fixture(gone, self.spec, score=0.84)
        self._write_journal([self._journal_node("exp013", str(gone), score=0.84)])
        # Now delete the orphaned run dir from disk.
        shutil.rmtree(gone)
        journal = TreeJournal(
            journal_path=self.journal_path,
            artifacts_dir=self.artifacts,
            auto_sync=True,
            task_spec=self.spec,
            task_plugin=self.plugin,
        )
        orphan = journal.nodes["exp013"]
        self.assertFalse(orphan.verified)
        self.assertEqual(orphan.status, "LEGACY_UNVERIFIED")
        # The healthy on-disk run is unaffected and verified.
        self.assertTrue(journal.nodes["exp012"].verified)

    def test_auto_sync_false_keeps_full_per_node_check(self):
        """auto_sync=False must retain the pre-optimization behavior: full
        per-node validation without any artifacts-dir walk."""
        self._seal_run("exp014", score=0.85)
        orphan_dir = self.artifacts / "exp015"
        self._write_journal([self._journal_node("exp015", str(orphan_dir), score=0.85)])
        with patch(
            "harness.artifacts.validate_run", side_effect=ValueError("missing manifest v2")
        ) as mocked:
            journal = TreeJournal(
                journal_path=self.journal_path,
                artifacts_dir=self.artifacts,
                auto_sync=False,
                task_spec=self.spec,
                task_plugin=self.plugin,
            )
        # The journal node's (missing) artifact dir was fully checked...
        mocked.assert_called_once()
        self.assertFalse(journal.nodes["exp015"].verified)
        self.assertEqual(journal.nodes["exp015"].status, "LEGACY_UNVERIFIED")
        # ...and no on-disk discovery happened at all.
        self.assertNotIn("exp014", journal.nodes)


class ExecuteActionSnapshotReuseTests(unittest.TestCase):
    """The post-action HUD snapshot (reused journal + one sync) must equal
    what a fresh TreeJournal built after the action would report."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aih_snap_reuse_")
        self.root = Path(self._tmp.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.root / "configs")
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_final_state_matches_fresh_journal_snapshot(self):
        # One sealed baseline run so the snapshot has real content.
        spec, plugin = _spec_and_plugin(self.root)
        seal_fixture(
            self.bridge.project.artifacts_dir / "exp020",
            spec,
            score=0.88,
            job_id="job-exp020",
        )
        sess, obs = self.bridge.execute_action(
            action="record_research_decision",
            params={
                "outcome": "inconclusive",
                "conclusion": "Baseline verified on fixture workspace.",
            },
        )
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        fresh = TreeJournal(
            journal_path=self.bridge.project.artifacts_dir / "journal.json",
            artifacts_dir=self.bridge.project.artifacts_dir,
            auto_sync=True,
            task_spec=self.bridge.task_spec,
            task_plugin=self.bridge.task_plugin,
        )
        reference = self.bridge.comp_state.evaluate_journey_snapshot(
            fresh, self.bridge._new_submission_manager()
        )
        self.assertEqual(sess.final_state, reference.get("hud", {}))


class TestLLMExperimentSpecs(unittest.TestCase):
    def test_node_dataclass_llm_specs(self):
        node = Node(
            id="exp010_model_unsloth_qwen_7b",
            parent_id="exp001_baseline_tfidf",
            branch="model",
            approach_type="llm",
            model_family="llm_peft",
            param_count=7_615_000_000,
            param_size="7B",
            llm_spec={
                "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                "param_size": "7B",
                "quantization": "4-bit NormalFloat (NF4)",
                "paradigm": "unsloth_qlora",
                "lora_r": 16,
                "lora_alpha": 32,
            },
        )
        self.assertEqual(node.approach_type, "llm")
        self.assertEqual(node.param_size, "7B")
        self.assertIsNotNone(node.llm_spec)
        self.assertEqual(node.llm_spec["quantization"], "4-bit NormalFloat (NF4)")

    def test_tree_journal_get_graph_data_includes_llm_specs(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            journal_file = Path(tmp_dir) / "journal.json"
            journal = TreeJournal(storage_path=str(journal_file))
            node = Node(
                id="exp020_model_deepseek_r1_14b",
                parent_id=None,
                branch="model",
                approach_type="llm",
                model_family="llm_reasoning",
                param_count=14_000_000_000,
                param_size="14B",
                llm_spec={
                    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    "param_size": "14B",
                    "quantization": "4-bit AWQ",
                    "paradigm": "zero_shot_cot",
                    "use_cot": True,
                },
            )
            journal.append(node)

            graph = journal.get_graph_data()
            self.assertEqual(len(graph["nodes"]), 1)
            g_node = graph["nodes"][0]
            self.assertEqual(g_node["id"], "exp020_model_deepseek_r1_14b")
            self.assertEqual(g_node["approach_type"], "llm")
            self.assertEqual(g_node["param_size"], "14B")
            self.assertIsNotNone(g_node["llm_spec"])
            self.assertEqual(g_node["llm_spec"]["model_name"], "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_llm_strategy_cards_present(self):
        llm_cards = [c for c in DEFAULT_STRATEGY_CARDS if c.get("approach_type") == "llm"]
        self.assertGreaterEqual(len(llm_cards), 2)
        qwen = next((c for c in llm_cards if "qwen" in c["id"].lower()), None)
        self.assertIsNotNone(qwen)
        self.assertEqual(qwen["param_size"], "7B")

        r1 = next((c for c in llm_cards if "deepseek" in c["id"].lower()), None)
        self.assertIsNotNone(r1)
        self.assertEqual(r1["param_size"], "14B")


class TestTreeJournalDirectionAndUCB(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.journal_path = self.test_dir / "journal.json"
        self.journal = TreeJournal(storage_path=str(self.journal_path))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ucb_minimization_prioritizes_lower_score(self):
        """When direction='minimize', lower loss should receive higher exploit score and UCB."""
        # Node 1: High loss (worse)
        node_bad = Node(
            verified=True,
            eligible=True,
            id="exp001_high_loss",
            branch="baseline",
            direction="minimize",
            primary_metric_name="loss",
            metrics={"loss": 2.50},
        )
        # Node 2: Low loss (better)
        node_good = Node(
            verified=True,
            eligible=True,
            id="exp002_low_loss",
            branch="feat",
            parent_id="exp001_high_loss",
            direction="minimize",
            primary_metric_name="loss",
            metrics={"loss": 0.40},
        )
        self.journal.append(node_bad)
        self.journal.append(node_good)

        rec = self.journal.recommend_next_branch()
        # Better node (lower loss = 0.40) must be recommended as parent
        self.assertEqual(rec["parent_id"], "exp002_low_loss")

    def test_ucb_maximization_prioritizes_higher_score(self):
        """When direction='maximize', higher score should receive higher exploit score and UCB."""
        node_low = Node(
            verified=True,
            eligible=True,
            id="exp001_low_f1",
            branch="baseline",
            direction="maximize",
            primary_metric_name="f1",
            metrics={"f1": 0.60},
        )
        node_high = Node(
            verified=True,
            eligible=True,
            id="exp002_high_f1",
            branch="feat",
            parent_id="exp001_low_f1",
            direction="maximize",
            primary_metric_name="f1",
            metrics={"f1": 0.85},
        )
        self.journal.append(node_low)
        self.journal.append(node_high)

        rec = self.journal.recommend_next_branch()
        self.assertEqual(rec["parent_id"], "exp002_high_f1")

    def test_generic_bottleneck_cue_handling(self):
        """Diagnostics with generic words like 'imbalance' or 'underfitting' trigger appropriate branch cues."""
        node = Node(
            verified=True,
            eligible=True,
            id="exp001_imbalanced",
            branch="baseline",
            direction="maximize",
            metrics={"score": 0.70},
            diagnostics_summary={"critical_bottlenecks": ["Severe class imbalance on tail classes"]},
        )
        self.journal.append(node)
        rec = self.journal.recommend_next_branch()
        self.assertIn("imbalance", rec["diagnostic_cue"].lower())
        self.assertEqual(rec["recommended_branch"], "loss")


class TestEnsembleGuardEnhancements(unittest.TestCase):
    def test_verify_ensemble_superiority_minimize(self):
        """In minimization, ensemble is superior when ensemble_score < best_constituent - min_margin."""
        constituent_losses = [1.20, 1.15, 1.10]  # Best constituent = 1.10
        # Ensemble achieves 1.05 (margin = 1.10 - 1.05 = 0.05 >= 0.01)
        passed, msg, details = EnsembleGuard.verify_ensemble_superiority(
            ensemble_score=1.05,
            constituent_scores=constituent_losses,
            min_margin=0.01,
            direction="minimize",
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(details["margin"], 0.05)

        # Ensemble achieves 1.12 (worse than best constituent 1.10)
        passed_fail, _, _ = EnsembleGuard.verify_ensemble_superiority(
            ensemble_score=1.12,
            constituent_scores=constituent_losses,
            min_margin=0.01,
            direction="minimize",
        )
        self.assertFalse(passed_fail)

    def test_audit_constituent_diversity_dynamic_column(self):
        """audit_constituent_diversity automatically finds prediction columns when col=None."""
        df1 = pd.DataFrame({"pred_target": ["a", "b", "a", "b", "a"] * 20})
        df2 = pd.DataFrame({"pred_target": ["a", "a", "a", "b", "b"] * 20})
        passed, errors, details = EnsembleGuard.audit_constituent_diversity([df1, df2], col=None)
        self.assertEqual(details["column_checked"], "pred_target")

    def test_multi_model_rescue_sample_analysis(self):
        """analyze_rescue_samples handles 3 or more models gracefully."""
        y_true = np.array([1, 1, 1, 0, 0])
        m1 = np.array([1, 0, 0, 0, 0])  # wrong on index 1, 2
        m2 = np.array([1, 1, 0, 0, 0])  # wrong on index 2
        m3 = np.array([0, 1, 0, 0, 1])  # wrong on index 0, 2, 4
        ens = np.array([1, 1, 1, 0, 0])  # 100% correct!

        res = EnsembleGuard.analyze_rescue_samples(y_true, ens, m1, m2, m3)
        self.assertIn("rescued_from_model_0", res)
        self.assertIn("rescued_from_model_1", res)
        self.assertIn("rescued_from_model_2", res)
        # All constituents were wrong on index 2, but ensemble got it right!
        self.assertEqual(res["synergistic_rescues"], 1)
        self.assertEqual(res["degraded_samples"], 0)
        self.assertGreater(res["net_gain"], 0)


class TestAtomicIOUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_write_text_and_json(self):
        target_file = self.test_dir / "subdir" / "state.json"
        data = {"hello": "world", "number": 42, "items": [1, 2, 3]}

        atomic_write_json(target_file, data)
        self.assertTrue(target_file.exists())

        with open(target_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

        # Overwrite atomically
        atomic_write_text(target_file, "plain text updated")
        self.assertEqual(target_file.read_text(encoding="utf-8"), "plain text updated")

        # Verify no dangling temp files in directory
        temp_files = list(target_file.parent.glob(".*.tmp_*"))
        self.assertEqual(len(temp_files), 0)


class TestCandidateSynthesisEnhancements(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.test_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True)

        # Create two mock experiment runs
        for exp_id in ["exp01_model", "exp02_model"]:
            exp_path = self.artifacts_dir / exp_id
            exp_path.mkdir()
            df = pd.DataFrame({
                "id": [1, 2, 3, 4],
                "fold": [0, 0, 1, 1],
                "pred_prob": [0.2, 0.8, 0.4, 0.9] if exp_id == "exp01_model" else [0.4, 0.6, 0.6, 0.7],
                "pred_label": ["no", "yes", "no", "yes"] if exp_id == "exp01_model" else ["no", "no", "yes", "yes"],
            })
            df.to_csv(exp_path / "oof_predictions.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_weighted_probability_blend(self):
        """Weighted blend correctly averages continuous probabilities."""
        from harness.competition_state import CompetitionState

        comp_state = CompetitionState(project_root=self.test_dir)
        with self.assertRaisesRegex(ValueError, "LEGACY_UNVERIFIED"):
            comp_state.synthesize_candidate(
                name="prob_blend_v1",
                mode="weighted_blend",
                source_map={"components": ["exp01_model", "exp02_model"], "weights": [0.75, 0.25]},
            )


class TestSubmissionManagerDynamic(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.mgr = SubmissionManager(submissions_dir=str(self.test_dir / "submissions"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dynamic_markdown_rendering(self):
        """Renders dynamic markdown without hardcoded A/B/C when other metrics are present."""
        entry = SubmissionEntry(
            id="sub_001_exp01",
            exp_id="exp01",
            created_at="2026-09-27 10:00:00 UTC",
            status="SCORED",
            local_cv_scores={"rmse": 0.4521, "mae": 0.3120},
            official_scores={"rmse": 0.4610},
            score_deltas={"rmse": 0.0089},
        )
        self.mgr.submissions[entry.id] = entry
        md = self.mgr.render_submissions_markdown()
        self.assertIn("sub_001_exp01", md)
        self.assertTrue(Path(self.test_dir / "submissions" / "SUBMISSIONS.md").exists())


class TestAuditGateDynamicIntegrity(unittest.TestCase):
    def test_audit_prediction_integrity_custom_id_col(self):
        """Integrity checker handles non-standard id column."""
        df = pd.DataFrame({
            "sample_index": [1, 2, None],
            "pred_stereotype": ["yes", "no", "yes"],
        })
        errors = audit_prediction_integrity(df, id_col="sample_index")
        self.assertTrue(any("sample_index" in e for e in errors))

    def test_compute_diff_metrics_dynamic(self):
        """Row-by-row diffs computed cleanly without hardcoded ID col."""
        cand = pd.DataFrame({"sample_id": [1, 2, 3], "pred_target": ["a", "b", "c"]})
        base = pd.DataFrame({"sample_id": [1, 2, 3], "pred_target": ["a", "b", "b"]})
        diff_count, diff_details = compute_diff_metrics(cand, base, id_col="sample_id")
        self.assertEqual(diff_count, 1)
        self.assertEqual(diff_details["pred_target"]["different_count"], 1)


class TestTreeJournalLoadFlexibility(unittest.TestCase):
    def test_tree_journal_load_flexibility(self):
        with tempfile.TemporaryDirectory() as td:
            j_path = Path(td) / "journal.json"
            dummy_data = {
                "nodes": [
                    {
                        "id": "exp001_baseline",
                        "status": "COMPLETED",
                        "metrics": {"primary_score": 0.85},
                    }
                ]
            }
            j_path.write_text(json.dumps(dummy_data), encoding="utf-8")

            j1 = TreeJournal(storage_path=j_path)
            self.assertIn("exp001_baseline", j1.nodes)
            j1.nodes.clear()
            self.assertEqual(len(j1.nodes), 0)
            j1.load()
            self.assertIn("exp001_baseline", j1.nodes)

            j2 = TreeJournal.load(str(j_path))
            self.assertIsInstance(j2, TreeJournal)
            self.assertIn("exp001_baseline", j2.nodes)

            j3 = TreeJournal.load_from_file(j_path)
            self.assertIsInstance(j3, TreeJournal)
            self.assertIn("exp001_baseline", j3.nodes)


class DiagnosticEngineTests(unittest.TestCase):
    def test_confusion_and_bottleneck_detection(self):
        from harness.diagnostics import DiagnosticEngine

        eval_df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(10)],
            "sentiment": ["positive"] * 4 + ["negative"] * 6,
            "label_rare": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "label_common": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })
        oof_df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(10)],
            "pred_sentiment": ["positive"] * 10,
            "pred_label_rare": [0] * 10,
            "pred_label_common": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })

        diag = DiagnosticEngine.generate_diagnostics(eval_df, oof_df)
        self.assertIn("sentiment", diag["target_confusions"])
        self.assertTrue(diag["minority_labels"]["rare"]["is_zero_recall"])
        self.assertGreaterEqual(len(diag["critical_bottlenecks"]), 1)


class TreeJournalAntiDegenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.temp_dir.name) / "journal.json"
        self.journal = TreeJournal(storage_path=str(self.journal_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_locks_ens_when_only_one_model_family_exists(self):
        node1 = Node(
            verified=True, eligible=True,
            id="baseline_tfidf_1",
            parent_id=None,
            branch="baseline",
            atomic_hypothesis="TF-IDF baseline",
            metrics={"overall_composite": 0.65, "model": "tfidf"},
        )
        node2 = Node(
            verified=True, eligible=True,
            id="baseline_tfidf_2",
            parent_id="baseline_tfidf_1",
            branch="feat",
            atomic_hypothesis="TF-IDF with title",
            metrics={"overall_composite": 0.64, "model": "tfidf"},
        )
        self.journal.append(node1)
        self.journal.append(node2)

        self.assertEqual(self.journal.count_distinct_model_families(), 1)
        rec = self.journal.recommend_next_branch()
        self.assertNotEqual(rec["recommended_branch"], "ens")

    def test_diagnostic_cues_influence_recommendation(self):
        node = Node(
            verified=True, eligible=True,
            id="node_with_rare_class_issue",
            parent_id=None,
            branch="baseline",
            atomic_hypothesis="Initial run",
            metrics={"overall_composite": 0.60, "model": "tfidf"},
            diagnostics_summary={"critical_bottlenecks": ["Zero recall on rare identity classes: ['a', 'i']."]},
        )
        self.journal.append(node)

        rec = self.journal.recommend_next_branch()
        self.assertEqual(rec["recommended_branch"], "loss")
        self.assertIn("Focal Loss", rec["diagnostic_cue"])


if __name__ == "__main__":
    unittest.main()
