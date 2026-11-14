"""Consolidated integration test suite for Submission Manager and dispatch governance.

Consolidates:
- tests/test_audit_gate_submission.py
- tests/test_competitive_endgame.py
- tests/test_dispatch_governance.py
- tests/test_endgame_tools.py
- tests/test_leaderboard_direction.py
- tests/test_notebook_prediction_integrity.py
- tests/test_preserve_submission.py
- tests/test_provenance_dispatch_guard.py
- tests/test_submission_and_cloud.py
- tests/test_submission_dispatch.py
- tests/test_submission_guards.py
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

from jsonschema import Draft202012Validator
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import yaml

from harness import dispatch_audit, kernel_dispatch, submission_dispatch
from harness.agent_bridge import AgentBridge, TypedActionName, get_action_schema
from harness.artifacts import sha256
from harness.audit_gate import enforce_dispatch_script_guard, scan_dispatch_scripts, scan_markdown_citations
from harness.candidate_preflight import run_candidate_preflight
from harness.candidates import find_optimal_blend_weights, register_external_prediction
from harness.citation_sources import SourceRegistry
from harness.cloud_inspector import CloudPreflightInspector
from harness.competition_state import CompetitionStateMachine
from harness.decision_rules import register_decision_rule, reserve_comparison_slot
from harness.endgame_submission import (
    KaggleLeaderboardTracker,
    MicroDoseProbeGenerator,
    QuantileMatcher,
    SubmissionPreflightValidator,
    ThresholdOptimizer,
)
from harness.environment_detector import EnvironmentDetector
from harness.foundation_models import (
    HierarchicalDiscretizer,
    MultiScaleBinner,
    TeacherLogitPrior,
)
from harness.kaggle_compat import KaggleClient
from harness.kaggle_notebook_builder import build_kaggle_submission_notebook
from harness.provenance import (
    PredictionProvenance,
    derive_submission_eligibility,
    require_allowed_provenance,
    validate_provenance_record,
)
from harness.runner import UniversalRunner
from harness.stacking import (
    L2LogitStacker,
    RankAverager,
    align_prediction_frames,
    to_clipped_logit,
)
from harness.submission_gate import (
    SubmissionGate,
    SubmissionGateError,
    consume_dispatch_permit,
)
from harness.submission_manager import SubmissionEntry, SubmissionManager
from harness.task_spec import SubtaskSpec, UniversalTaskSpec, load_active_task_spec
from harness.templates.kaggle_template import KaggleKernelManager
from harness.tree_journal import Node, TreeJournal
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import (
    SOURCE_ROOT,
    make_submission_evidence,
    make_workspace,
    rule_document,
    task,
    task_document,
    workspace_with_verified_pair,
)
from tools.agent_bridge_cli import (
    _coerce_value,
    _format_action_schema,
    _parse_dynamic_kv_args,
    _safe_json_loads,
)
from tools.audit_gate import audit_prediction_integrity, compute_diff_metrics


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Section 1: Audit Gate Refinements & Dynamic Class Validation
# (from tests/test_audit_gate_submission.py)
# ---------------------------------------------------------------------------


class TestAuditGateRefinements(unittest.TestCase):
    def test_compute_diff_metrics_excludes_resolved_id_when_none(self):
        """When id_col is None, compute_diff_metrics must still exclude resolved_id from prediction comparison."""
        cand_df = pd.DataFrame({
            "sample_id": ["id_1", "id_2", "id_3"],
            "pred_col": ["a", "b", "c"],
        })
        base_df = pd.DataFrame({
            "sample_id": ["id_1", "id_2", "id_3"],
            "pred_col": ["a", "x", "c"],
        })

        diff_count, diff_details = compute_diff_metrics(cand_df, base_df, id_col="sample_id")
        self.assertEqual(diff_count, 1)
        self.assertIn("pred_col", diff_details)
        self.assertNotIn("sample_id", diff_details)

    def test_audit_prediction_integrity_valid_and_invalid_classes(self):
        """Dynamic class validation catches values outside allowed classes."""
        valid_df = pd.DataFrame({
            "sample_id": ["id_1", "id_2"],
            "pred_stereotype": ["yes", "no"],
            "pred_hate_speech": ["no", "yes_explicit"],
            "pred_target_type": ["none", "individual"],
        })
        spec = UniversalTaskSpec(
            name="Stereo fixture", metadata={"data": {"id_column": "sample_id"}},
            task_type="multi_task_classification",
            primary_metric="macro_f1",
            subtasks={
                "stereotype": SubtaskSpec("stereotype", "Stereotype", "binary_classification", column="stereotype", classes=["yes", "no"]),
                "hate_speech": SubtaskSpec("hate_speech", "Hate Speech", "multiclass_classification", column="hate_speech", classes=["no", "yes_explicit"]),
                "target_type": SubtaskSpec("target_type", "Target Type", "multiclass_classification", column="target_type", classes=["none", "individual"]),
            },
        )
        errors = audit_prediction_integrity(valid_df, spec=spec)
        self.assertEqual(errors, [])

        bad_df = pd.DataFrame({
            "sample_id": ["id_1"],
            "pred_stereotype": ["maybe"],
        })
        bad_errors = audit_prediction_integrity(bad_df, spec=spec)
        self.assertTrue(any("invalid values" in err for err in bad_errors))


class TestSubmissionManagerDirectionAwareDivergence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.mgr = SubmissionManager(submissions_dir=str(self.test_dir))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_official_score_custom_scores_and_divergence_maximize(self):
        """For maximize direction, LB drop > 0.05 triggers divergence warning."""
        entry = SubmissionEntry(
            id="sub_test_01",
            exp_id="exp001",
            created_at="2026-09-27T00:00:00Z",
            status="LOCAL_VERIFIED",
            local_cv_scores={"macro_f1": 0.85},
        )
        self.mgr.submissions[entry.id] = entry

        updated = self.mgr.record_official_score(
            sub_id="sub_test_01",
            custom_scores={"macro_f1": 0.75},
        )
        self.assertTrue(updated.divergence_warning)
        self.assertAlmostEqual(updated.score_deltas.get("macro_f1"), -0.10)

    def test_record_official_score_divergence_minimize(self):
        """For minimize direction, LB increase > 0.05 triggers divergence warning."""
        entry = SubmissionEntry(
            id="sub_test_min",
            exp_id="exp002",
            created_at="2026-09-27T00:00:00Z",
            status="LOCAL_VERIFIED",
            local_cv_scores={"loss": 0.20},
        )
        self.mgr.submissions[entry.id] = entry

        custom_spec = UniversalTaskSpec(
            name="test_reg",
            task_type="regression",
            primary_metric="loss",
            direction="minimize",
        )
        with mock.patch("harness.task_spec.load_active_task_spec", return_value=custom_spec):
            updated = self.mgr.record_official_score(
                sub_id="sub_test_min",
                custom_scores={"loss": 0.35},
            )
            self.assertTrue(updated.divergence_warning)
            self.assertAlmostEqual(updated.score_deltas.get("loss"), 0.15)


class TestAgentBridgeDynamicCandidateSlots(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.bridge = AgentBridge(project_root=str(self.test_dir))
        self.bridge.comp_state.state["mode"] = "PARALLEL_EXPLORATION"
        self.session = self.bridge.start_session(goal="Test dynamic freeze")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_freeze_candidate_dynamically_picks_from_candidate_pool(self):
        """When slot IDs are omitted, freeze_candidate picks available candidates from pool."""
        self.bridge.comp_state.candidates = [
            {"id": "cand_alpha", "name": "Model Alpha", "score": 0.88},
            {"id": "cand_beta", "name": "Model Beta", "score": 0.86},
        ]
        _, obs = self.bridge.execute_action(
            action="freeze_candidate",
            params={},
            session_id=self.session.session_id,
            user_approved=True,
        )
        self.assertEqual(obs.status, "FAILED")
        self.assertFalse(self.bridge.comp_state.state["gates"]["freeze_locked"])

    def test_record_external_score_forwards_custom_scores(self):
        """RECORD_EXTERNAL_SCORE action correctly passes arbitrary scores dictionary."""
        entry = SubmissionEntry(
            id="sub_forward_test",
            exp_id="exp003",
            created_at="2026-09-27T00:00:00Z",
            status="LOCAL_VERIFIED",
            local_cv_scores={"composite_score": 0.80},
        )
        sub_mgr = SubmissionManager(submissions_dir=str(self.test_dir / "submissions"))
        sub_mgr.submissions[entry.id] = entry
        sub_mgr.save()

        with mock.patch("harness.agent_bridge.SubmissionManager", return_value=sub_mgr):
            _, obs = self.bridge.execute_action(
                action="record_external_score",
                params={"sub_id": "sub_forward_test", "scores": {"composite_score": 0.82}},
                session_id=self.session.session_id,
                user_approved=True,
            )
            self.assertEqual(obs.status, "SUCCESS")
            data = obs.data
            self.assertEqual(data["official_scores"].get("composite_score"), 0.82)


class TestEnvironmentDetector(unittest.TestCase):
    def test_detect_gpu_structure(self):
        """detect_gpu returns dictionary with required hardware keys without crashing."""
        res = EnvironmentDetector.detect_gpu()
        self.assertIsInstance(res, dict)
        self.assertIn("detected", res)
        self.assertIn("gpu_name", res)
        self.assertIn("cuda_available", res)


# ---------------------------------------------------------------------------
# Section 2: Competitive ML Endgame Stacking, Clipping, Blending
# (from tests/test_competitive_endgame.py)
# ---------------------------------------------------------------------------


class TestClippedLogit(unittest.TestCase):
    def test_boundary_handling(self):
        probs = [0.0, 1.0, 0.5, 1e-18, 1.0 - 1e-18]
        logits = to_clipped_logit(probs, logit_bounds=(-15.0, 15.0))
        self.assertEqual(len(logits), 5)
        self.assertAlmostEqual(logits[2], 0.0, places=5)
        self.assertEqual(logits[0], -15.0)
        self.assertEqual(logits[1], 15.0)
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_pass_through_existing_logits(self):
        raw_logits = np.array([-20.0, 0.0, 20.0])
        clipped = to_clipped_logit(raw_logits, logit_bounds=(-12.0, 12.0))
        self.assertEqual(clipped[0], -12.0)
        self.assertEqual(clipped[1], 0.0)
        self.assertEqual(clipped[2], 12.0)


class TestPredictionAligner(unittest.TestCase):
    def test_scrambled_id_alignment(self):
        ref_df = pd.DataFrame({"id": [10, 20, 30, 40], "feat": [1, 2, 3, 4]})
        df_a = pd.DataFrame({"id": [40, 30, 20, 10], "pred": [0.4, 0.3, 0.2, 0.1]})
        df_b = pd.DataFrame({"id": [20, 40, 10, 30], "proba": ["[0.8, 0.2]", "[0.6, 0.4]", "[0.9, 0.1]", "[0.7, 0.3]"]})

        aligned, _names = align_prediction_frames([df_a, df_b], ref_df, id_column="id")
        self.assertEqual(aligned.shape, (4, 2))
        np.testing.assert_allclose(aligned[:, 0], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(aligned[:, 1], [0.1, 0.2, 0.3, 0.4])

    def test_missing_id_raises(self):
        ref_df = pd.DataFrame({"id": [1, 2, 3]})
        bad_df = pd.DataFrame({"id": [1, 2], "pred": [0.1, 0.2]})
        with self.assertRaises(ValueError):
            align_prediction_frames([bad_df], ref_df, id_column="id")


class TestL2LogitStacker(unittest.TestCase):
    def test_nested_stacking_improves_or_matches_auc(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.integers(0, 2, size=n)

        m1 = np.clip(y * 0.7 + rng.normal(0, 0.3, size=n), 0.01, 0.99)
        m2 = np.clip(y * 0.6 + rng.normal(0, 0.4, size=n), 0.01, 0.99)
        m3 = np.clip(y * 0.5 + rng.normal(0, 0.5, size=n), 0.01, 0.99)

        X_oof = np.column_stack([m1, m2, m3])
        X_test = np.column_stack([m1[:50], m2[:50], m3[:50]])

        stacker = L2LogitStacker(C=0.01, n_splits=3, random_state=42)
        oof_pred, test_pred = stacker.fit_predict(X_oof, y, X_test=X_test)

        self.assertEqual(len(oof_pred), n)
        self.assertEqual(len(test_pred), 50)
        self.assertTrue(np.all((oof_pred >= 0.0) & (oof_pred <= 1.0)))

        auc_m1 = roc_auc_score(y, m1)
        auc_stack = roc_auc_score(y, oof_pred)
        self.assertGreaterEqual(auc_stack, auc_m1 - 0.02)

        fi = stacker.get_feature_importance()
        self.assertEqual(len(fi), 3)
        self.assertIn("weight", fi.columns)


class TestRankAverager(unittest.TestCase):
    def test_rank_percentile_blend(self):
        m1 = np.array([0.7, 0.8, 0.9, 0.95])
        m2 = np.array([0.1, 0.2, 0.3, 0.4])

        blended = RankAverager.blend([m1, m2], weights=[0.5, 0.5])
        self.assertEqual(len(blended), 4)
        self.assertTrue(np.all(np.diff(blended) > 0))
        self.assertAlmostEqual(blended[0], 0.25)
        self.assertAlmostEqual(blended[3], 1.0)


class TestTeacherLogitPrior(unittest.TestCase):
    def test_teacher_prior_extraction(self):
        aux_df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat2": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 0, 0, 1, 1, 1],
        })
        target_df = pd.DataFrame({
            "feat1": [2.5, 5.5],
            "feat2": ["a", "b"],
        })

        teacher = TeacherLogitPrior(target_column="target", model_type="logistic_regression")
        teacher.fit(aux_df)

        meta_feats = teacher.transform(target_df)
        self.assertEqual(len(meta_feats), 2)
        self.assertIn("teacher_prob", meta_feats.columns)
        self.assertIn("teacher_logit", meta_feats.columns)
        self.assertTrue(np.all(np.isfinite(meta_feats["teacher_logit"])))


class TestMultiScaleBinner(unittest.TestCase):
    def test_multi_scale_binning(self):
        s = pd.Series([15, 235, 1250], name="metric")
        res = MultiScaleBinner.bin_scales(s, scales=(10, 100, 1000))
        self.assertEqual(res.shape, (3, 4))
        self.assertIn("metric_exact", res.columns)
        self.assertIn("metric_scale_10", res.columns)
        self.assertIn("metric_scale_100", res.columns)
        self.assertIn("metric_scale_1000", res.columns)
        self.assertEqual(res["metric_scale_10"].iloc[0], 1)
        self.assertEqual(res["metric_scale_100"].iloc[1], 2)
        self.assertEqual(res["metric_scale_1000"].iloc[2], 1)

    def test_backwards_compatible_alias(self):
        s = pd.Series([12, 45, 102], name="dist")
        res = HierarchicalDiscretizer.decompose_distance(s, col_name="val")
        self.assertIn("val_exact", res.columns)
        self.assertEqual(res["val_exact"].iloc[0], 12)


class TestSubmissionPreflightValidator(unittest.TestCase):
    def test_valid_and_invalid_submissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref_path = tmp / "test.csv"
            pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]}).to_csv(ref_path, index=False)

            valid_sub = tmp / "valid.csv"
            pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.5, 0.9]}).to_csv(valid_sub, index=False)
            rep = SubmissionPreflightValidator.audit_submission(valid_sub, ref_path)
            self.assertTrue(rep.is_valid)
            self.assertEqual(len(rep.errors), 0)

            nan_sub = tmp / "nan.csv"
            pd.DataFrame({"id": [1, 2, 3], "target": [0.1, np.nan, 0.9]}).to_csv(nan_sub, index=False)
            rep_nan = SubmissionPreflightValidator.audit_submission(nan_sub, ref_path)
            self.assertFalse(rep_nan.is_valid)
            self.assertIn("NaN/null", rep_nan.errors[0])

            oob_sub = tmp / "oob.csv"
            pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 1.5, 0.9]}).to_csv(oob_sub, index=False)
            rep_oob = SubmissionPreflightValidator.audit_submission(oob_sub, ref_path)
            self.assertFalse(rep_oob.is_valid)


# ---------------------------------------------------------------------------
# Section 3: Dispatch Governance & Kernel Audits
# (from tests/test_dispatch_governance.py)
# ---------------------------------------------------------------------------


class _FakeSubmitClient:
    """In-memory Kaggle client for the tracker (records every submit)."""

    def __init__(self):
        self.calls = []

    def get_submission_limits(self, competition):
        return dict(daily_quota=5, num_today=0, has_remaining_quota=True)

    def submit_submission(self, file_name, message, competition, kernel=None, kernel_version=None, *, _permit=None):
        consume_dispatch_permit(_permit, file_name=file_name, competition=competition, kernel=kernel, kernel_version=kernel_version)
        self.calls.append(
            dict(file_name=file_name, message=message, competition=competition,
                 kernel=kernel, kernel_version=kernel_version)
        )
        return SimpleNamespace(ref=1000 + len(self.calls), message="accepted")

    def list_submissions(self, competition):
        return []


class _FakeKernelClient:
    """In-memory Kaggle client for kernel push (records every push)."""

    def __init__(self):
        self.calls = []

    def kernels_push(self, folder, timeout=None, acc=None):
        self.calls.append(dict(folder=folder, timeout=timeout, acc=acc))
        return SimpleNamespace(
            ref=700 + len(self.calls),
            url=f"https://www.kaggle.com/kernels/own/k/{len(self.calls)}",
            kernel_id=555,
            version_number=len(self.calls),
            error=None,
            invalid_dataset_sources=None,
            invalid_competition_sources=None,
            invalid_kernel_sources=None,
            invalid_model_sources=None,
        )


def _attached_workspace(root: Path) -> Path:
    """Minimal attached workspace: manifest present so the audit resolves it."""
    (root / ".ai-harness").mkdir(parents=True, exist_ok=True)
    (root / ".ai-harness" / "project.yaml").write_text(
        "task_config: configs/task.yaml\nplugin_id: default\n", encoding="utf-8"
    )
    return root


def _log_entries(root: Path) -> list:
    path = root / ".ai-harness" / "dispatch_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SubmissionAuditTests(unittest.TestCase):
    """Real shared gate with a fake transport; kernel pushes retain v0 logging."""

    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.gate = SubmissionGate(self.root)
        self.client = _FakeSubmitClient()
        self.tracker = KaggleLeaderboardTracker(client=self.client, sleep_fn=lambda _: None)
        self.sub = self.gate.plugin.build_submission(self.gate.context,
            pd.read_csv(self.gate.project.artifacts_dir / 'control/test_predictions.csv'), self.root / 'submission.csv')

    def send(self, **options):
        return self.tracker.dispatch('comp-a', str(self.sub), 'Exp 1', project_root=self.root,
            exp_id='control', user_approved=True, **options)

    def test_dispatch_records_durable_intent_before_provider_call(self):
        result = self.send()
        events = dispatch_audit.read_submission_events(self.root)
        self.assertEqual([e['event'] for e in events], ['prepared', 'sending', 'submitted'])
        self.assertEqual(result['ref'], 1001)
        self.assertEqual(events[-1]['payload_sha256'], _sha256(self.sub.read_bytes()))

    def test_identical_resubmission_is_blocked_before_provider_call(self):
        self.send()
        with self.assertRaisesRegex(SubmissionGateError, 'DUPLICATE_PAYLOAD'):
            self.send()
        self.assertEqual(len(self.client.calls), 1)

    def test_changed_file_requires_new_sealed_evidence(self):
        self.send()
        self.sub.write_text('id,label\n40,1\n')
        with self.assertRaisesRegex(SubmissionGateError, 'PAYLOAD_EVIDENCE_MISMATCH'):
            self.send()
        self.assertEqual(len(self.client.calls), 1)

    def test_missing_file_is_blocked(self):
        self.sub.unlink()
        with self.assertRaisesRegex(ValueError, 'PAYLOAD_MISSING'):
            self.send()
        self.assertEqual(self.client.calls, [])

    def test_durable_audit_failure_blocks_provider(self):
        with mock.patch('harness.submission_gate.append_dispatch_event', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.send()
        self.assertEqual(self.client.calls, [])

    def test_missing_workspace_cannot_dispatch_through_fake_client(self):
        with self.assertRaisesRegex(ValueError, 'WORKSPACE_REQUIRED'):
            self.tracker.dispatch('comp-a', str(self.sub), project_root=self.root / 'bare')
        self.assertEqual(self.client.calls, [])


class KernelAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aih-kernel-gov-")
        self.root = _attached_workspace(Path(self.tmp.name))
        self.client = _FakeKernelClient()
        self.nb = self.root / "exp.ipynb"
        self.nb.write_text("{'cells': []}", encoding="utf-8")
        self.env = {"HARNESS_PROJECT_ROOT": str(self.root)}

    def tearDown(self):
        self.tmp.cleanup()

    def _push(self, kernel_id="own/exp048"):
        with mock.patch.dict(os.environ, self.env):
            return kernel_dispatch.dispatch_kernel(
                str(self.nb), kernel_id, "Exp048", client=self.client,
            )

    def test_kernel_push_appends_audit_with_notebook_sha(self):
        result = self._push()
        self.assertTrue(result["ok"])
        entries = _log_entries(self.root)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["kind"], "kernel")
        self.assertEqual(e["kernel_id"], "own/exp048")
        self.assertEqual(e["notebook_sha256"], _sha256(self.nb.read_bytes()))
        self.assertEqual(e["ref"], 701)

    def test_same_kernel_same_content_warns(self):
        self._push()
        r2 = self._push()
        self.assertIn("duplicate_warning", r2)
        self.assertEqual(r2["duplicate_of"]["kernel_id"], "own/exp048")
        self.assertEqual(len(self.client.calls), 2, "v1 must not hard-block")

    def test_version_bump_different_content_no_warning(self):
        self._push()
        self.nb.write_text("{'cells': [{'v': 2}]}", encoding="utf-8")
        r2 = self._push()
        self.assertNotIn("duplicate_warning", r2)


class CliWarningTests(unittest.TestCase):
    def test_cli_reports_blocked_unaudited_dispatch_as_json(self):
        client = _FakeSubmitClient()
        tracker = KaggleLeaderboardTracker(client=client, sleep_fn=lambda _: None)
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(submission_dispatch, 'KaggleLeaderboardTracker', return_value=tracker), contextlib.redirect_stdout(out):
                code = submission_dispatch.main(['dispatch', 'comp-a', 'missing.csv', '--project-root', tmp])
        payload = json.loads(out.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(payload['ok'])
        self.assertEqual(payload['error_code'], 'WORKSPACE_REQUIRED')
        self.assertEqual(client.calls, [])


# ---------------------------------------------------------------------------
# Section 4: Endgame Tools & Optimization Actions
# (from tests/test_endgame_tools.py)
# ---------------------------------------------------------------------------


class TestThresholdOptimizer(unittest.TestCase):
    def test_imbalanced_f1_optimization(self):
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], size=500, p=[0.9, 0.1])
        y_prob = np.clip(y_true * 0.4 + rng.uniform(0.0, 0.25, size=500), 0.01, 0.99)

        score_05 = f1_score(y_true, (y_prob >= 0.5).astype(int))
        best_th, best_score = ThresholdOptimizer.optimize_threshold(
            y_true, y_prob, metric="f1", num_thresholds=181
        )

        self.assertGreaterEqual(best_score, score_05)
        self.assertLess(best_th, 0.5)


class TestQuantileMatcher(unittest.TestCase):
    def test_quantile_matching_distribution(self):
        rng = np.random.default_rng(42)
        source = rng.normal(loc=0.0, scale=1.0, size=1000)
        target = rng.exponential(scale=2.0, size=1000)

        matched = QuantileMatcher.match_quantiles(source, target)
        self.assertEqual(len(matched), len(source))
        self.assertGreaterEqual(matched.min(), 0.0)
        self.assertAlmostEqual(np.median(matched), np.median(target), places=1)


class TestMicroDoseProbeGenerator(unittest.TestCase):
    def test_generate_probe_logit_and_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_csv = tmp / "base.csv"
            probe_csv = tmp / "probe.csv"
            out_csv = tmp / "probe_out.csv"

            n = 200
            df_base = pd.DataFrame({"id": range(n), "target": np.linspace(0.1, 0.9, n)})
            df_probe = pd.DataFrame({"id": range(n), "target": np.linspace(0.2, 0.8, n)})
            df_base.to_csv(base_csv, index=False)
            df_probe.to_csv(probe_csv, index=False)

            rep_logit = MicroDoseProbeGenerator.generate_probe(
                base_csv, probe_csv, out_csv, alpha=0.03, space="logit"
            )
            self.assertTrue(out_csv.exists())
            self.assertGreater(rep_logit["spearman_correlation_vs_base"], 0.99)
            self.assertLess(rep_logit["max_absolute_diff"], 0.05)

            rep_rank = MicroDoseProbeGenerator.generate_probe(
                base_csv, probe_csv, out_csv, alpha=0.05, space="rank"
            )
            self.assertGreater(rep_rank["spearman_correlation_vs_base"], 0.99)


class TestEndgameCandidatesAndBridge(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "artifacts").mkdir()

        spec = task()
        doc = {"task": spec.metadata}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

        n = 20
        self.n = n
        self.train_df = pd.DataFrame({
            "id": range(n),
            "group": range(n),
            "x": range(n),
            "category": ["a", "b"] * (n // 2),
            "target": [float(x * 2.0) for x in range(n)],
        })
        self.train_df.to_csv(self.root / "dataset/train.csv", index=False)

        self.journal_path = self.root / "artifacts" / "tree_journal.json"
        self.journal = TreeJournal(storage_path=self.journal_path)

        for exp_id, bias in [("exp001", 0.5), ("exp002", -0.5)]:
            exp_dir = self.root / "artifacts" / exp_id
            exp_dir.mkdir()

            oof = pd.DataFrame({
                "id": range(n),
                "fold": [i % 2 for i in range(n)],
                "input_variant": "features",
                "pred_target": [float(x * 2.0 + bias) for x in range(n)],
            })
            oof.to_csv(exp_dir / "oof_predictions.csv", index=False)
            oof.to_csv(exp_dir / "predictions.csv", index=False)
            self.train_df.to_csv(exp_dir / "evaluation_truth.csv", index=False)

            test_p = pd.DataFrame({
                "id": range(5),
                "target": [float(x * 2.0 + bias) for x in range(5)],
                "pred_target": [float(x * 2.0 + bias) for x in range(5)],
            })
            test_p.to_csv(exp_dir / "test_predictions.csv", index=False)

            metrics = {
                "schema_version": 2,
                "experiment_id": exp_id,
                "selection_variant": "features",
                "primary_score": abs(bias),
                "utility": -abs(bias),
            }
            (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            cfg = {
                "experiment": {"id": exp_id, "branch": "baseline"},
                "model": {"name": "ridge"},
            }
            (exp_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

            manifest = {
                "schema_version": 2,
                "provenance_version": 1,
                "provenance": "local_run",
                "exp_id": exp_id,
                "task_sha256": sha256(self.root / "configs/task.yaml"),
                "scorer_sha256": "default_scorer",
                "split_sha256": "split_hash_mock",
                "data_sha256": "data_hash_mock",
                "resolved_config_sha256": sha256(exp_dir / "run_config.yaml"),
                "prediction_splits": {"test_predictions.csv": "test"},
                "files": {
                    "run_config.yaml": sha256(exp_dir / "run_config.yaml"),
                    "metrics.json": sha256(exp_dir / "metrics.json"),
                    "oof_predictions.csv": sha256(exp_dir / "oof_predictions.csv"),
                    "evaluation_truth.csv": sha256(exp_dir / "evaluation_truth.csv"),
                    "test_predictions.csv": sha256(exp_dir / "test_predictions.csv"),
                },
            }
            (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            self.journal.append(Node(
                id=exp_id,
                branch="baseline",
                metrics={"primary_score": abs(bias), "score": abs(bias)},
                primary_metric_name="mse",
                direction="minimize",
                verified=True,
                eligible=True,
                status="COMPLETED",
                artifact_dir=str(exp_dir),
            ))

        self.bridge = AgentBridge(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_find_optimal_blend_weights(self):
        spec = load_active_task_spec(self.root / "configs/task.yaml")
        for exp_id in ("exp001", "exp002"):
            seal_fixture(self.root / "artifacts" / exp_id, spec, score=.5)
        opt = find_optimal_blend_weights(
            project=self.root,
            component_ids=["exp001", "exp002"],
            blend_mode="weighted_blend",
        )
        self.assertEqual(opt["component_ids"], ["exp001", "exp002"])
        self.assertEqual(len(opt["optimal_weights"]), 2)
        self.assertAlmostEqual(sum(opt["optimal_weights"]), 1.0, places=3)
        self.assertIn("best_score", opt)

    def test_register_external_prediction(self):
        ext_oof = pd.DataFrame({
            "id": range(self.n),
            "fold": 0,
            "pred_target": [float(x * 2.0 + 0.05) for x in range(self.n)],
        })
        ext_oof_path = self.root / "ext_oof.csv"
        ext_oof.to_csv(ext_oof_path, index=False)

        res = register_external_prediction(
            project=self.root,
            name="ext_gm_01",
            oof_path=ext_oof_path,
            author="Grandmaster_T",
            source="kaggle_kernel",
            notes="PyTorch CatBoost Stacking",
        )
        self.assertEqual(res["name"], "ext_gm_01")
        self.assertEqual(res["author"], "Grandmaster_T")
        self.assertTrue((self.root / "artifacts" / "ext_gm_01" / "artifact_manifest.json").exists())
        self.assertTrue((self.root / "artifacts" / "ext_gm_01" / "metrics.json").exists())

    def test_agent_bridge_endgame_actions(self):
        sess = self.bridge.start_session(goal="Test endgame actions")

        sess, obs_th = self.bridge.execute_action(
            action=TypedActionName.OPTIMIZE_THRESHOLD.value,
            params={"exp_id": "exp001", "metric": "f1"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs_th.status, "FAILED")
        self.assertIn("binary", obs_th.summary.lower())

        base_p = self.root / "artifacts" / "exp001" / "test_predictions.csv"
        probe_p = self.root / "artifacts" / "exp002" / "test_predictions.csv"
        sess, obs_probe = self.bridge.execute_action(
            action=TypedActionName.GENERATE_MICRODOSE_PROBE.value,
            params={
                "base": str(base_p),
                "probe": str(probe_p),
                "alpha": 0.03,
                "space": "probability",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs_probe.status, "SUCCESS")
        self.assertGreater(obs_probe.data["spearman_correlation_vs_base"], 0.99)

        ext_oof = pd.DataFrame({
            "id": range(self.n),
            "pred_target": [float(x * 2.0) for x in range(self.n)],
        })
        ext_path = self.root / "ext_pred.csv"
        ext_oof.to_csv(ext_path, index=False)

        sess, obs_ext = self.bridge.execute_action(
            action=TypedActionName.REGISTER_EXTERNAL_PREDICTION.value,
            params={"name": "ext_bridge_01", "oof_file": str(ext_path), "author": "GM"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs_ext.status, "SUCCESS")
        self.assertEqual(obs_ext.data["name"], "ext_bridge_01")


# ---------------------------------------------------------------------------
# Section 5: Leaderboard Direction Resolution & Isolation
# (from tests/test_leaderboard_direction.py)
# ---------------------------------------------------------------------------


def _direction_tracker(scores=(.1, .5)):
    submissions = [SimpleNamespace(ref=i, status="COMPLETE", public_score=str(value))
                   for i, value in enumerate(scores, 1)]
    return KaggleLeaderboardTracker(client=SimpleNamespace(list_submissions=lambda competition: submissions))


class LeaderboardDirectionTests(unittest.TestCase):
    def minimizing_workspace(self):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        frame = pd.DataFrame({"id": range(4), "group": range(4), "x": range(4),
                              "category": ["a", "b"] * 2, "target": range(4)})
        root, _ = make_workspace(task=doc, train_csv=frame.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)

    def test_api_defaults_to_the_attached_task_direction(self):
        self.minimizing_workspace()
        result = _direction_tracker().latest("rmse-competition")
        self.assertEqual(result["best"]["ref"], 1)
        self.assertEqual(result["direction"], "minimize")
        self.assertEqual(result["direction_source"], "task")

    def test_explicit_direction_overrides_the_workspace(self):
        self.minimizing_workspace()
        result = _direction_tracker().latest("other-competition", direction="maximize")
        self.assertEqual(result["best"]["ref"], 2)
        self.assertEqual(result["direction_source"], "explicit")

    def test_cli_uses_the_task_and_exposes_an_explicit_override(self):
        self.minimizing_workspace()
        for extra, expected in (([], 1), (["--direction", "maximize"], 2)):
            with self.subTest(extra=extra):
                output = io.StringIO()
                with (patch.object(submission_dispatch, "KaggleLeaderboardTracker", return_value=_direction_tracker()),
                      contextlib.redirect_stdout(output)):
                    code = submission_dispatch.main(["latest", "rmse-competition", *extra])
                self.assertEqual(code, 0, output.getvalue())
                self.assertEqual(json.loads(output.getvalue())["best"]["ref"], expected)

    def test_without_a_task_the_legacy_maximize_default_is_explicit(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": root}):
            os.environ.pop("HARNESS_TASK_CONFIG", None)
            result = _direction_tracker().latest("comp")
        self.assertEqual(result["best"]["ref"], 2)
        self.assertEqual(result["direction"], "maximize")
        self.assertEqual(result["direction_source"], "default")

    def test_zero_is_a_valid_minimum_and_nonfinite_scores_are_not_ranked(self):
        result = _direction_tracker([float("nan"), float("inf"), 0, .5]).latest("comp", direction="minimize")
        self.assertEqual(result["scored"], 2)
        self.assertEqual(result["best"]["ref"], 3)
        self.assertEqual(result["best"]["public_score_f"], 0)
        with self.assertRaisesRegex(ValueError, "direction"):
            _direction_tracker().latest("comp", direction="sideways")


# ---------------------------------------------------------------------------
# Section 6: Notebook Prediction Integrity Execution
# (from tests/test_notebook_prediction_integrity.py)
# ---------------------------------------------------------------------------


class NotebookPredictionIntegrityTests(unittest.TestCase):
    def make_notebook(self, predictions=None, test=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        exp = root / "exp001"
        exp.mkdir()
        test = pd.DataFrame({"id": [10, 11]}) if test is None else test
        test_path = root / "chosen_test.csv"
        test.to_csv(test_path, index=False)
        if predictions is not None:
            predictions.to_csv(exp / "test_predictions.csv", index=False)
        output = root / "submission.ipynb"
        build_kaggle_submission_notebook(exp, output, test_path_override=str(test_path))
        return root, json.loads(output.read_text())

    def execute(self, notebook, glob_results=None):
        namespace = {}
        with (patch("glob.glob", side_effect=glob_results or (lambda *a, **kw: [])),
              patch.object(pd.DataFrame, "to_csv", autospec=True) as writer,
              contextlib.redirect_stdout(io.StringIO())):
            try:
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        exec("".join(cell["source"]), namespace)
            except Exception:
                writer.assert_not_called()
                raise
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(writer.call_args.args[1], "submission.csv")
            return writer.call_args.args[0].copy()

    def test_missing_prediction_artifact_cannot_create_placeholder_submission(self):
        _, notebook = self.make_notebook()
        with self.assertRaisesRegex(FileNotFoundError, "prediction|inference"):
            self.execute(notebook)

    def test_shuffled_predictions_use_explicit_test_path_and_match_ids(self):
        _, notebook = self.make_notebook(pd.DataFrame({"id": [11, 10], "pred_target": [.8, .2],
                                                       "fold": [0, 0], "classes_target": "[0, 1]"}))
        result = self.execute(notebook)
        self.assertEqual(result.columns.tolist(), ["id", "target"])
        self.assertEqual(result.to_dict("list"), {"id": [10, 11], "target": [.2, .8]})

    def test_malformed_prediction_artifacts_fail_before_writing(self):
        for predictions in (
            pd.DataFrame({"id": [10, 10], "target": [.2, .8]}),
            pd.DataFrame({"id": [10, None], "target": [.2, .8]}),
            pd.DataFrame({"id": [10], "target": [.2]}),
            pd.DataFrame({"id": [10, 12], "target": [.2, .8]}),
            pd.DataFrame({"id": [10, 11, 12], "target": [.2, .8, .5]}),
            pd.DataFrame({"target": [.2, .8]}),
            pd.DataFrame({"id": [10, 11], "wrong_target": [.2, .8]}),
            pd.DataFrame({"id": [10, 11], "target": [.2, float("inf")]}),
        ):
            with self.subTest(predictions=predictions.to_dict()):
                _, notebook = self.make_notebook(predictions)
                with self.assertRaises((ValueError, KeyError)):
                    self.execute(notebook)

    def test_invalid_test_ids_cannot_be_fabricated_or_duplicated(self):
        for test in (pd.DataFrame({"feature": [1, 2]}), pd.DataFrame({"id": [10, 10]})):
            with self.subTest(test=test.to_dict()):
                _, notebook = self.make_notebook(pd.DataFrame({"id": [10, 11], "target": [.2, .8]}), test)
                with self.assertRaises(ValueError):
                    self.execute(notebook)

    def test_ambiguous_attached_predictions_require_explicit_selection(self):
        root, notebook = self.make_notebook()
        files = []
        for name in ("one", "two"):
            folder = root / name
            folder.mkdir()
            path = folder / "test_predictions.csv"
            pd.DataFrame({"id": [10, 11], "target": [.2, .8]}).to_csv(path, index=False)
            files.append(str(path))
        with self.assertRaisesRegex(ValueError, "Multiple|multiple|Ambiguous"):
            self.execute(notebook, lambda pattern, **kw: files if "/exp001/" in pattern else [])

    def test_failed_offline_install_is_not_reported_as_success(self):
        _, notebook = self.make_notebook()
        with (patch("glob.glob", return_value=["/offline/dependency.whl"]),
              patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")) as run,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(subprocess.CalledProcessError):
                exec("".join(notebook["cells"][1]["source"]), {})
            self.assertTrue(run.call_args.kwargs["check"])


# ---------------------------------------------------------------------------
# Section 7: Preserve Submission & CLI Dynamic Args Parsing
# (from tests/test_preserve_submission.py)
# ---------------------------------------------------------------------------


class PreserveSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)
        self.sub_dir = self.p / "submissions"
        self.mgr = SubmissionManager(submissions_dir=str(self.sub_dir), project_root=str(self.p))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preserve_file_submission_success(self):
        sample_file = self.p / "custom_submission.zip"
        sample_file.write_bytes(b"PK\x03\x04mock_zip_binary_payload")
        expected_hash = sha256(sample_file)

        entry = self.mgr.preserve_file_submission(
            file_path=sample_file,
            notes="Trained on external cluster from debate consensus",
            reported_score=0.895,
            reported_partition="public",
        )

        self.assertTrue(entry.id.endswith("_preserved"))
        self.assertEqual(entry.status, "PRESERVED_UNVERIFIED")
        self.assertEqual(entry.exp_id, "preserved")
        self.assertEqual(entry.payload_sha256, expected_hash)
        self.assertEqual(entry.official_scores, {})
        self.assertEqual(entry.reported_scores, {"reported_public": 0.895})
        self.assertEqual(entry.official_score_source, "user_reported_unverified")
        self.assertEqual(sha256(Path(entry.archive_path)), expected_hash)

        receipt_path = Path(entry.receipt_path)
        self.assertTrue(receipt_path.is_file())
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)
        self.assertEqual(receipt["record_type"], "preservation_only")
        self.assertEqual(receipt["verification_status"], "unverified")
        self.assertFalse(receipt["dispatch_authorized"])
        self.assertEqual(receipt["payload_sha256"], expected_hash)
        self.assertEqual(receipt["byte_count"], sample_file.stat().st_size)
        self.assertEqual(receipt["reported_score"], 0.895)
        self.assertEqual(receipt["reported_partition"], "public")

        md_text = self.mgr.render_submissions_markdown()
        self.assertIn("📦 `PRESERVED_UNVERIFIED`", md_text)
        self.assertIn(f"| `{entry.id}` | `preserved` | 📦 `PRESERVED` |", md_text)
        self.assertNotIn(f"| `{entry.id}` | `preserved` | 🔒 `LOCAL_VERIFIED` |", md_text)
        self.assertIn("| - | - | [`zip`]", md_text)
        self.assertIn("Reported: reported_public:0.895", md_text)

    def test_preserve_file_submission_validation_errors(self):
        with self.assertRaises(FileNotFoundError):
            self.mgr.preserve_file_submission(file_path=self.p / "does_not_exist.csv")

        with tempfile.TemporaryDirectory() as external_tmp:
            ext_file = Path(external_tmp) / "external.csv"
            ext_file.write_text("x,y\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.mgr.preserve_file_submission(file_path=ext_file)
            self.assertIn("project root", str(ctx.exception))

        for reserved in ("PRESERVATION_RECEIPT.json", "metadata.json", "registry.json"):
            res_file = self.p / reserved
            res_file.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.mgr.preserve_file_submission(file_path=res_file)
            self.assertIn("reserved metadata filename", str(ctx.exception))

        valid_file = self.p / "valid_name.csv"
        valid_file.write_text("a,b\n1,2\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=float("nan"))
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=float("inf"))
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=True)

    def test_bridge_preserve_submission_action(self):
        root, task_config_path = make_workspace(prefix="test_preserve_ws_")
        bridge = AgentBridge(project_root=root, task_spec_path=task_config_path)

        sub_file = root / "final_ensemble.zip"
        sub_file.write_bytes(b"PK\x03\x04test_data")
        expected_hash = sha256(sub_file)

        _sess, obs = bridge.act(
            action="preserve_submission",
            params={
                "file": str(sub_file),
                "notes": "Adjudicated debate winner",
                "score": 0.912,
                "partition": "private",
            },
        )

        self.assertEqual(obs.status, "SUCCESS")
        self.assertIn("Preserved submission artifact", obs.summary)
        self.assertEqual(obs.data["sha256"], expected_hash)
        self.assertFalse(obs.data["dispatch_authorized"])
        self.assertEqual(obs.data["status"], "PRESERVED_UNVERIFIED")

    def test_agent_bridge_cli_formatting_and_parsing(self):
        schema = get_action_schema("preserve_submission")
        self.assertIsNotNone(schema)
        formatted = _format_action_schema("preserve_submission", schema)
        self.assertIn("--file <value>", formatted)
        self.assertIn("--notes <value> (optional)", formatted)
        self.assertIn('ai-harness-agent act --action preserve_submission --file "VALUE"', formatted)
        self.assertIn("PowerShell:", formatted)

        param_types = schema.get("parameter_types", {})
        parsed = _parse_dynamic_kv_args(
            ["--file", "123", "--notes", "false", "--score", "0.85"],
            parameter_types=param_types,
        )
        self.assertEqual(parsed["file"], "123")
        self.assertIsInstance(parsed["file"], str)
        self.assertEqual(parsed["notes"], "false")
        self.assertIsInstance(parsed["notes"], str)
        self.assertEqual(parsed["score"], 0.85)

        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["--score", "true"], parameter_types=param_types)

        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["--file"], parameter_types=param_types)

        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["unexpected_positional_arg"])

        with self.assertRaises(ValueError):
            _safe_json_loads('{"score": NaN}')
        with self.assertRaises(ValueError):
            _safe_json_loads('{"score": Infinity}')

        self.assertEqual(_coerce_value("nan"), "nan")
        self.assertEqual(_coerce_value("inf"), "inf")
        self.assertEqual(_coerce_value("-inf"), "-inf")
        self.assertEqual(_coerce_value("123"), 123)
        self.assertEqual(_coerce_value("12.34"), 12.34)
        self.assertIsNone(_coerce_value("null"))
        self.assertTrue(_coerce_value("true"))
        self.assertFalse(_coerce_value("false"))

    def test_runner_notice_and_spec_conflict(self):
        doc = task_document(primary_metric="rmse", direction="maximize")
        root, task_config_path = make_workspace(
            prefix="test_runner_notice_",
            task=doc,
        )

        scorer_py = root / "custom_eval.py"
        scorer_py.write_text("def score(y, p): return 1.0\n", encoding="utf-8")

        cfg = {
            "experiment": {"id": "exp001", "name": "notice_test"},
            "runtime": {"seed": 42},
            "paths": {"metrics_evaluator": "custom_eval:score"},
        }
        exp_yaml = root / "configs" / "exp001.yaml"
        exp_yaml.parent.mkdir(parents=True, exist_ok=True)
        exp_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        args = argparse.Namespace(
            project_root=str(root),
            task_config=str(task_config_path),
            fold=0,
            debug=True,
            probe_samples=None,
            job_id=None,
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            UniversalRunner(cfg, str(exp_yaml), args)
        out = f.getvalue()
        self.assertIn("[ai-harness] INFO: Custom metric scorer active:", out)
        self.assertIn("custom_eval:score", out)
        self.assertIn("[ai-harness] WARNING: spec:", out)


# ---------------------------------------------------------------------------
# Section 8: Provenance Contract & Backdoor Dispatch Guard
# (from tests/test_provenance_dispatch_guard.py)
# ---------------------------------------------------------------------------


class ProvenanceDispatchGuardTests(unittest.TestCase):
    def test_provenance_contract_and_eligibility_derivation(self):
        self.assertEqual(PredictionProvenance.LOCAL_RUN.value, "local_run")
        with self.assertRaisesRegex(ValueError, "PROVENANCE_MISSING_OR_INVALID"):
            validate_provenance_record({})
        with self.assertRaisesRegex(ValueError, "EXTERNAL_PREDICTION_NOT_SUBMITTABLE"):
            require_allowed_provenance([{"provenance": "external_prediction"}], purpose="candidate")
        self.assertFalse(derive_submission_eligibility(
            provenance_valid=True, validation_valid=True,
            test_predictions_valid=True, promotion_valid=False,
        )["submission_eligible"])

    def test_scanner_finds_root_and_scratch_dispatch_patterns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dispatch.py").write_text("kaggle competitions submit -f x.csv\n")
            (root / "scratch").mkdir()
            (root / "scratch" / "kernel.py").write_text("client.kernels_push_directory(path)\n")
            report = scan_dispatch_scripts(root)
            self.assertEqual(report["status"], "findings")
            self.assertEqual({entry["pattern"] for entry in report["findings"]},
                             {"kaggle competitions submit", "kernels_push_directory"})
            self.assertEqual(enforce_dispatch_script_guard(root)["blocker"], "DISPATCH_BACKDOOR_DETECTED")

    def test_acknowledged_script_is_reported_but_not_blocking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "raw.py").write_text("competitions_submissions_create()\n")
            ack = root / ".ai-harness"
            ack.mkdir()
            (ack / "acknowledged_dispatch_scripts.json").write_text('{"acknowledged": ["raw.py"]}')
            self.assertEqual(len(scan_dispatch_scripts(root)["findings"]), 1)
            self.assertEqual(enforce_dispatch_script_guard(root)["status"], "clean")

    def test_scan_markdown_citations_finds_missing_and_ungrounded_topics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            disc_dir = root / "discussions"
            disc_dir.mkdir()
            SourceRegistry(root).register_topic_source(
                "750885", {"id": 750885, "title": "Negative results",
                           "content": "14 streams vs 10 streams on RTX Pro 6000 and 110k token cutoff."},
                adapter="normalized_topic_v1",
            )
            md_path = root / "summary.md"
            md_path.write_text(
                "# Analysis\n"
                "- Proven scheduler D-prime by shiiin9 prevents high-level starvation (Topic #750885).\n"
                "- Untracked breakthrough reported in (Topic #999999).\n",
                encoding="utf-8"
            )
            report = scan_markdown_citations(root, file_path=md_path)
            self.assertEqual(report["status"], "findings")
            self.assertEqual(report["total_citations"], 2)
            issues = {f["issue"] for f in report["findings"]}
            self.assertIn("MISSING_TOPIC_FILE", issues)
            self.assertIn("UNGROUNDED_TOPIC_CITATION", issues)
            ungrounded = [f for f in report["findings"] if f["issue"] == "UNGROUNDED_TOPIC_CITATION"][0]
            self.assertEqual(ungrounded["topic_id"], "750885")
            self.assertIn("shiiin9", ungrounded["missing_terms"])


# ---------------------------------------------------------------------------
# Section 9: Submission Lifecycle & Cloud Pre-flight
# (from tests/test_submission_and_cloud.py)
# ---------------------------------------------------------------------------


class SubmissionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)
        self.sub_dir = self.p / "submissions"
        self.mgr = SubmissionManager(submissions_dir=str(self.sub_dir))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_id_generation(self):
        first = self.mgr.get_next_submission_id()
        self.assertRegex(first, r"^sub_([0-9a-f]{32}|\d{3})$")
        entry = SubmissionEntry(
            id="sub_001_exp001",
            exp_id="exp001",
            created_at="2026-09-24",
            status="LOCAL_VERIFIED",
        )
        self.mgr.submissions[entry.id] = entry
        self.assertNotEqual(first, self.mgr.get_next_submission_id())

    def test_record_official_score_and_divergence(self):
        entry = SubmissionEntry(
            id="sub_001_exp001",
            exp_id="exp001",
            created_at="2026-09-24",
            status="LOCAL_VERIFIED",
            local_cv_scores={"subtask_a_macro": 0.7000, "subtask_b_macro": 0.5500},
        )
        self.mgr.submissions[entry.id] = entry

        updated = self.mgr.record_official_score(
            sub_id="sub_001_exp001",
            scores={"subtask_a_macro": 0.7100, "subtask_b_macro": 0.5400},
            split_type="public",
        )
        self.assertEqual(updated.status, "SCORED")
        self.assertAlmostEqual(updated.score_deltas["subtask_a_macro"], 0.0100)
        self.assertAlmostEqual(updated.score_deltas["subtask_b_macro"], -0.0100)
        self.assertFalse(updated.divergence_warning)

        updated_div = self.mgr.record_official_score(
            sub_id="sub_001_exp001",
            scores={"subtask_a_macro": 0.6200, "subtask_b_macro": 0.5400},
        )
        self.assertTrue(updated_div.divergence_warning)
        self.assertAlmostEqual(updated_div.score_deltas["subtask_a_macro"], -0.0800)

    def test_render_submissions_markdown(self):
        entry = SubmissionEntry(
            id="sub_001_exp001",
            exp_id="exp001",
            created_at="2026-09-24",
            status="LOCAL_VERIFIED",
            local_cv_scores={"subtask_a_macro": 0.6962, "subtask_b_macro": 0.5238},
        )
        self.mgr.submissions[entry.id] = entry
        md = self.mgr.render_submissions_markdown()

        self.assertIn("Submissions & Leaderboard Registry", md)
        self.assertIn("sub_001_exp001", md)
        self.assertIn("LOCAL_VERIFIED", md)
        self.assertTrue((self.sub_dir / "SUBMISSIONS.md").exists())


class CloudPreflightInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inspect_config(self):
        cfg_path = self.p / "test_cfg.yaml"
        cfg_path.write_text("experiment:\n  id: exp999\n", encoding="utf-8")
        ok, errs, _ = CloudPreflightInspector.inspect_config(cfg_path)
        self.assertFalse(ok)
        self.assertTrue(any("branch" in e for e in errs))

        cfg_path.write_text(
            "experiment:\n  id: exp999\n  branch: feat\n", encoding="utf-8"
        )
        ok, errs, _ = CloudPreflightInspector.inspect_config(cfg_path)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

    def test_inspect_python_syntax(self):
        bad_py = self.p / "bad.py"
        bad_py.write_text("def broken_func(\n", encoding="utf-8")

        good_py = self.p / "good.py"
        good_py.write_text("def fine_func():\n    return 42\n", encoding="utf-8")

        ok, errs = CloudPreflightInspector.inspect_python_syntax([bad_py, good_py])
        self.assertFalse(ok)
        self.assertEqual(len(errs), 1)
        self.assertIn("Syntax error", errs[0])


class KaggleKernelManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_generation(self):
        with patch.object(
            KaggleKernelManager,
            "get_authenticated_username",
            return_value="fixture-user",
        ):
            meta = KaggleKernelManager.generate_kernel_metadata(
                "test-kernel", enable_gpu=True
            )
        self.assertEqual(meta["id"], "fixture-user/test-kernel")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(meta["enable_internet"], "true")

    def test_migrate_downloaded_artifacts(self):
        from tests.fixtures import make_workspace
        root, _ = make_workspace(root_path=self.p / "workspace", task={"task": task().metadata})
        download_d = self.p / "download"
        download_d.mkdir()
        seal_fixture(download_d, task())
        target_d = root / "artifacts/exp_test"
        extra = download_d / "heavy_model.pt"
        extra.write_text("BINARY DUMP", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            KaggleKernelManager.migrate_downloaded_artifacts(download_d, target_d, project_root=root)
        self.assertFalse(target_d.exists())
        extra.unlink()
        migrated = KaggleKernelManager.migrate_downloaded_artifacts(download_d, target_d, project_root=root)
        self.assertIn("oof_predictions.csv", migrated)
        self.assertIn("metrics.json", migrated)
        self.assertEqual(KaggleKernelManager.migrate_downloaded_artifacts(download_d, target_d, project_root=root), migrated)
        (download_d / "metrics.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            KaggleKernelManager.migrate_downloaded_artifacts(download_d, root / "invalid", project_root=root)
        self.assertFalse((root / "invalid").exists())


# ---------------------------------------------------------------------------
# Section 10: Submission Dispatch CLI & Gate Verification
# (from tests/test_submission_dispatch.py)
# ---------------------------------------------------------------------------


class _FakeTracker:
    """In-memory stand-in for ``KaggleLeaderboardTracker`` (records calls)."""

    def __init__(self):
        self.dispatch_calls = []
        self.status_calls = []
        self.watch_calls = []
        self.latest_calls = []

    def dispatch(self, competition, file, message, kernel=None, kernel_version=None, **options):
        self.dispatch_calls.append(dict(
            competition=competition, file=file, message=message,
            kernel=kernel, kernel_version=kernel_version,
            **options,
        ))
        return {"ref": 123456, "message": "submitted"}

    def get_submission(self, competition, ref):
        self.status_calls.append((competition, ref))
        return {"ref": ref, "status": "complete", "public_score": 0.9}

    def wait_for_score(self, competition, ref, timeout_s=0, poll_interval_s=0, max_poll_interval_s=0):
        self.watch_calls.append(dict(
            competition=competition, ref=ref, timeout_s=timeout_s,
            poll_interval_s=poll_interval_s, max_poll_interval_s=max_poll_interval_s,
        ))
        return {"ref": ref, "status": "complete", "public_score": 0.87, "attempts": 3}

    def latest(self, competition):
        self.latest_calls.append(competition)
        return {"ref": 999, "public_score": 0.93, "message": "best"}


def _run(argv, tracker):
    """Run ``submission_dispatch.main`` with the tracker mocked; return (code, payload)."""
    buf = io.StringIO()
    with mock.patch.object(
        submission_dispatch, "KaggleLeaderboardTracker", return_value=tracker
    ):
        with contextlib.redirect_stdout(buf):
            code = submission_dispatch.main(argv)
    return code, json.loads(buf.getvalue())


class SubmissionDispatchTests(unittest.TestCase):
    def test_client_initialization_error_is_parseable_json(self):
        out = io.StringIO()
        with mock.patch.object(submission_dispatch, "KaggleLeaderboardTracker", side_effect=RuntimeError("authentication unavailable")), contextlib.redirect_stdout(out):
            code = submission_dispatch.main(["dispatch", "comp", "sub.csv"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())["error"], "authentication unavailable")

    def test_dispatch_file_submits_and_reports_ref(self):
        t = _FakeTracker()
        code, payload = _run(["dispatch", "comp", "sub.csv", "-m", "Exp 22"], t)
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["action"], "dispatch")
        self.assertEqual(payload["ref"], 123456)
        self.assertEqual(t.dispatch_calls[0]["message"], "Exp 22")
        self.assertIsNone(t.dispatch_calls[0]["kernel"])

    def test_dispatch_code_competition_passes_kernel(self):
        t = _FakeTracker()
        code, payload = _run(
            ["dispatch", "comp", "out.csv", "-k", "owner/k", "--kernel-version", "3"], t
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "dispatch")
        self.assertEqual(t.dispatch_calls[0]["kernel"], "owner/k")
        self.assertEqual(t.dispatch_calls[0]["kernel_version"], "3")

    def test_status_returns_tracker_payload(self):
        t = _FakeTracker()
        code, payload = _run(["status", "comp", "57314333"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "status")
        self.assertEqual(payload["public_score"], 0.9)
        self.assertEqual(t.status_calls[0], ("comp", "57314333"))

    def test_watch_passes_poll_args(self):
        t = _FakeTracker()
        code, payload = _run(["watch", "comp", "57314333", "--timeout", "30", "--poll", "5"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "watch")
        self.assertEqual(payload["attempts"], 3)
        self.assertEqual(t.watch_calls[0]["timeout_s"], 30)
        self.assertEqual(t.watch_calls[0]["poll_interval_s"], 5)

    def test_latest_returns_best(self):
        t = _FakeTracker()
        code, payload = _run(["latest", "comp"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "latest")
        self.assertEqual(payload["ref"], 999)
        self.assertEqual(t.latest_calls[0], "comp")

    def test_error_path_returns_nonzero_and_json_error(self):
        t = _FakeTracker()

        def _boom(*args, **kwargs):
            raise RuntimeError("no such ref")

        t.get_submission = _boom
        code, payload = _run(["status", "comp", "0"], t)
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["action"], "status")
        self.assertIn("no such ref", payload["error"])


class SubmissionGateTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        project_manifest = yaml.safe_load((self.root / ".ai-harness/project.yaml").read_text(encoding="utf-8"))
        state_path = self.root / project_manifest["artifacts_dir"] / "competition_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"gates": {"baseline_locked": True, "baseline_exp_id": "control"}}), encoding="utf-8")
        state_machine = CompetitionStateMachine(self.root)
        state_machine.state["gates"].update(baseline_locked=True, baseline_exp_id="control")
        state_machine.save()
        self.gate = SubmissionGate(self.root)
        predictions = pd.read_csv(self.gate.project.artifacts_dir / "control/test_predictions.csv")
        self.payload = self.gate.plugin.build_submission(self.gate.context, predictions, self.root / "submission.csv")
        self.quota = dict(daily_quota=5, num_today=0, has_remaining_quota=True)

    def authorize(self, **kwargs):
        options = dict(exp_id="control", user_approved=True, quota=self.quota)
        options.update(kwargs)
        return self.gate.authorize_dispatch("fixture-comp", self.payload, **options)

    def test_dry_run_does_not_reserve_or_send(self):
        result = self.authorize(dry_run=True)
        self.assertEqual(result["receipt"]["status"], "pass")
        self.assertFalse((self.root / ".ai-harness/submission_events.jsonl").exists())
        schema = json.loads((SOURCE_ROOT / "harness/schemas/submission_gate_receipt.v2.schema.json").read_text())
        Draft202012Validator(schema).validate(result["receipt"])

    def test_missing_approval_blocks_and_does_not_reserve(self):
        with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_APPROVAL_REQUIRED"):
            self.authorize(user_approved=False)
        self.assertFalse((self.root / ".ai-harness/submission_events.jsonl").exists())

    def test_missing_quota_and_exhausted_quota_block(self):
        for quota, reason in ((None, "QUOTA_UNKNOWN"), (dict(daily_quota=5, num_today=5), "QUOTA_EXHAUSTED")):
            with self.assertRaisesRegex(SubmissionGateError, reason):
                self.authorize(quota=quota)

    def test_permit_is_single_use_and_bound_to_competition(self):
        auth = self.authorize()
        with self.assertRaisesRegex(SubmissionGateError, "PERMIT_BINDING_MISMATCH"):
            consume_dispatch_permit(auth["permit"], file_name=auth["intent"]["file_name"], competition="wrong")
        with self.assertRaisesRegex(SubmissionGateError, "EXPIRED_OR_CONSUMED"):
            consume_dispatch_permit(auth["permit"], file_name=auth["intent"]["file_name"], competition="fixture-comp")

    def test_prepared_sending_intent_prevents_concurrent_send(self):
        self.authorize()
        with self.assertRaisesRegex(SubmissionGateError, "UNRESOLVED_DISPATCH"):
            self.authorize()

    def test_payload_change_and_forged_receipt_cannot_authorize(self):
        self.payload.write_text("id,label\n40,999\n")
        with self.assertRaisesRegex(SubmissionGateError, "PAYLOAD_EVIDENCE_MISMATCH"):
            self.authorize()
        with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_PERMIT_REQUIRED"):
            consume_dispatch_permit({"status": "pass"}, file_name=str(self.payload), competition="fixture-comp")

    def test_duplicate_is_blocked_after_confirmed_dispatch(self):
        auth = self.authorize()
        self.gate.finish_submission_intent(auth, provider_ref=123)
        with self.assertRaisesRegex(SubmissionGateError, "DUPLICATE_PAYLOAD"):
            self.authorize()

    def test_tracker_failure_records_unknown_and_never_retries(self):
        client = mock.Mock()
        client.get_submission_limits.return_value = self.quota
        client.list_submissions.return_value = []
        client.submit_submission.side_effect = TimeoutError("ambiguous send")
        tracker = KaggleLeaderboardTracker(client=client)
        with self.assertRaises(TimeoutError):
            tracker.dispatch("fixture-comp", str(self.payload), project_root=self.root, exp_id="control", user_approved=True)
        self.assertEqual(client.submit_submission.call_count, 1)
        with self.assertRaisesRegex(SubmissionGateError, "UNRESOLVED_DISPATCH"):
            self.authorize()

    def test_facade_without_permit_cannot_reach_sdk(self):
        client = object.__new__(KaggleClient)
        with mock.patch.object(KaggleClient, "_legacy_api_instance") as raw:
            with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_PERMIT_REQUIRED"):
                client.submit_submission(str(self.payload), "test", "fixture-comp")
        raw.assert_not_called()

    def test_corrupt_authoritative_history_blocks(self):
        path = self.root / ".ai-harness/submission_events.jsonl"
        path.write_text("not-json\n")
        with self.assertRaises(ValueError):
            self.authorize()

    def test_kernel_output_requires_verified_version_binding(self):
        with self.assertRaisesRegex(SubmissionGateError, "UNVERIFIED_KERNEL_OUTPUT"):
            self.authorize(kernel="owner/kernel", kernel_version="3")

    def test_remote_history_cannot_be_hidden_by_empty_local_registry(self):
        with self.assertRaisesRegex(SubmissionGateError, "HISTORY_INCOMPLETE"):
            self.authorize(provider_refs=["untracked-remote-submission"])

    def test_candidate_file_dispatch_without_manifest_is_rejected(self):
        cand_file = self.root / "candidate_blend.csv"
        shutil.copyfile(self.payload, cand_file)
        with self.assertRaisesRegex(SubmissionGateError, "EXPERIMENT_EVIDENCE_REQUIRED"):
            self.gate.authorize_dispatch("fixture-comp", cand_file, exp_id=None, user_approved=True, dry_run=True, quota=self.quota)

    def test_preflight_or_approval_does_not_replace_promotion_receipt(self):
        incumbent = self.root / "incumbent.csv"
        incumbent.write_bytes(self.payload.read_bytes())
        preflight = run_candidate_preflight(project_root=self.root, candidate_path=self.payload,
                                            incumbent_path=incumbent, id_col="id", pred_col="label")
        unpromoted_dir = self.gate.project.artifacts_dir / "unpromoted"
        shutil.copytree(self.gate.project.artifacts_dir / "control", unpromoted_dir)
        unpromoted_manifest = json.loads((unpromoted_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        unpromoted_manifest["exp_id"] = "unpromoted"
        (unpromoted_dir / "artifact_manifest.json").write_text(json.dumps(unpromoted_manifest), encoding="utf-8")
        unpromoted_metrics = json.loads((unpromoted_dir / "metrics.json").read_text(encoding="utf-8"))
        unpromoted_metrics["experiment_id"] = "unpromoted"
        (unpromoted_dir / "metrics.json").write_text(json.dumps(unpromoted_metrics), encoding="utf-8")
        unpromoted_cfg = yaml.safe_load((unpromoted_dir / "run_config.yaml").read_text(encoding="utf-8"))
        unpromoted_cfg.setdefault("experiment", {})["id"] = "unpromoted"
        (unpromoted_dir / "run_config.yaml").write_text(yaml.safe_dump(unpromoted_cfg), encoding="utf-8")
        unpromoted_manifest["files"] = {
            p.name: sha256(p)
            for p in unpromoted_dir.iterdir()
            if p.is_file() and p.name != "artifact_manifest.json"
        }
        unpromoted_manifest["resolved_config_sha256"] = sha256(unpromoted_dir / "run_config.yaml")
        (unpromoted_dir / "artifact_manifest.json").write_text(json.dumps(unpromoted_manifest), encoding="utf-8")
        with self.assertRaisesRegex(SubmissionGateError, "PROMOTION_EVIDENCE_REQUIRED"):
            self.gate.evaluate("fixture-comp", self.payload, exp_id="unpromoted", user_approved=True,
                               preflight_id=preflight["preflight_id"], quota=self.quota)

    def test_backdoor_script_locks_dispatch(self):
        script = self.root / "scratch"
        script.mkdir(exist_ok=True)
        (script / "push.py").write_text("kaggle.api.competition_submit(file_name='x')\n")
        with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_BACKDOOR_DETECTED"):
            self.gate.evaluate("fixture-comp", self.payload, exp_id="control", user_approved=True, quota=self.quota)

    def test_consumption_rechecks_expiry_and_payload_bytes(self):
        auth = self.authorize()
        with mock.patch("harness.submission_gate.time.monotonic", return_value=float("inf")):
            with self.assertRaisesRegex(SubmissionGateError, "EXPIRED_OR_CONSUMED"):
                consume_dispatch_permit(auth["permit"], file_name=auth["intent"]["file_name"], competition="fixture-comp")

    def test_verified_prospective_promotion_can_dispatch(self):
        auth = self.authorize()
        self.gate.finish_submission_intent(auth, provider_ref=123)
        registration = register_decision_rule(rule_document(), project=self.gate.project, task_spec=self.gate.spec)
        cfg = dict(experiment=dict(id="promoted"), validation=dict(partition="oof"),
                   research=dict(decision_rule_id=registration["rule_id"]))
        reserve_comparison_slot(self.gate.project, self.gate.spec, cfg)
        directory = make_submission_evidence(self.root, "promoted", cfg=cfg)
        payload = self.gate.plugin.build_submission(self.gate.context, pd.read_csv(directory / "test_predictions.csv"), self.root / "promoted.csv")
        auth = self.gate.authorize_dispatch("fixture-comp", payload, exp_id="promoted", rule_id=registration["rule_id"],
            quota=self.quota, user_approved=True)
        self.assertEqual(auth["receipt"]["purpose"], "promotion")
        self.assertEqual(auth["receipt"]["status"], "pass")
        self.gate.finish_submission_intent(auth, provider_ref=124)


# ---------------------------------------------------------------------------
# Section 11: Submission Guards & Runtime Validation
# (from tests/test_submission_guards.py)
# ---------------------------------------------------------------------------


class TestSubmissionGuards(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.bridge = AgentBridge(self.root)
        self.art = self.bridge.artifacts_dir

    def test_preparation_validates_declared_predictions(self):
        evidence = self.bridge._guard_submission(self.art / "control/test_predictions.csv", "control", self.bridge._new_submission_manager())
        self.assertEqual(evidence["scope"], "validation")
        self.assertEqual(evidence["checks"]["integrity"], "PASS")
        self.assertNotIn("mmes", evidence["checks"])

    def test_semantic_duplicates_are_blocked_across_prepared_models(self):
        path = make_submission_evidence(self.root, "candidate", labels=[0] * 40)
        manager = self.bridge._new_submission_manager()
        manager.submissions["prior"] = SubmissionEntry(id="prior", exp_id="control", created_at="t", status="LOCAL_VERIFIED",
            candidate_path=str(self.art / "control/test_predictions.csv"))
        with self.assertRaisesRegex(ValueError, "SEMANTIC_DUPLICATE"):
            self.bridge._guard_submission(path / "test_predictions.csv", "candidate", manager)

    def test_empty_prediction_files_fail_integrity(self):
        path = self.root / "empty.csv"
        path.write_text("id,pred_label\n")
        with self.assertRaisesRegex(ValueError, "PREDICTION_INTEGRITY"):
            self.bridge._guard_submission(path, "candidate", self.bridge._new_submission_manager())

    def test_missing_prior_snapshots_are_not_silently_ignored(self):
        manager = self.bridge._new_submission_manager()
        manager.submissions["prior"] = SubmissionEntry(id="prior", exp_id="prior", created_at="t", status="LOCAL_VERIFIED",
            candidate_path=str(self.root / "missing.csv"))
        with self.assertRaisesRegex(ValueError, "HISTORY_INCOMPLETE"):
            self.bridge._guard_submission(self.art / "control/test_predictions.csv", "control", manager)

    def test_preparation_never_claims_metric_promotion(self):
        path = make_submission_evidence(self.root, "candidate")
        evidence = self.bridge._guard_submission(path / "test_predictions.csv", "candidate", self.bridge._new_submission_manager())
        self.assertEqual(evidence["scope"], "validation")
        self.assertNotIn("outcome", evidence)


class TestGatewayKeepalive(unittest.TestCase):
    def test_init_defaults(self):
        from harness.gateway_keepalive import GatewayKeepaliveDaemon

        daemon = GatewayKeepaliveDaemon()
        self.assertEqual(daemon.gateway_url, "http://gateway:8001")
        self.assertEqual(daemon.action, "RESET")
        self.assertGreaterEqual(daemon.ping_interval_s, 10.0)

    def test_init_env_overrides(self):
        from harness.gateway_keepalive import GatewayKeepaliveDaemon

        with patch.dict("os.environ", {"ARC_BASE_URL": "http://custom-hearth:9000", "ARC_API_KEY": "secret-key"}):
            daemon = GatewayKeepaliveDaemon()
            self.assertEqual(daemon.gateway_url, "http://custom-hearth:9000")
            self.assertEqual(daemon.api_key, "secret-key")
            headers = daemon._get_headers()
            self.assertEqual(headers["X-API-Key"], "secret-key")

    @patch("urllib.request.urlopen")
    def test_discovery_and_heartbeat(self, mock_urlopen):
        from unittest.mock import MagicMock
        from harness.gateway_keepalive import GatewayKeepaliveDaemon

        def side_effect(req, timeout=10):
            url = req.get_full_url() if hasattr(req, "get_full_url") else str(req)
            resp = MagicMock()
            resp.__enter__.return_value = resp
            if "/api/games" in url:
                resp.status = 200
                resp.read.return_value = json.dumps([{"game_id": "game_alpha_01"}]).encode()
            elif "/api/game/game_alpha_01/step" in url:
                resp.status = 200
                resp.read.return_value = b'{"success": true}'
            else:
                resp.status = 404
                resp.read.return_value = b"{}"
            return resp

        mock_urlopen.side_effect = side_effect

        daemon = GatewayKeepaliveDaemon(
            gateway_url="http://test-gateway:8001",
            ping_interval_s=10.0,
            action="RESET",
        )

        with daemon:
            discovered = daemon._discover_game_id()
            self.assertEqual(discovered, "game_alpha_01")
            success = daemon._send_heartbeat(discovered)
            self.assertTrue(success)
            self.assertEqual(daemon.last_status, 200)
            self.assertGreaterEqual(daemon.pulses_sent, 1)


class TestSubmissionUsabilityAudit(unittest.TestCase):
    def test_submission_preflight_validator_string_predictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ref_path = tmp_path / "test.csv"
            pd.DataFrame({"id": [1, 2, 3]}).to_csv(ref_path, index=False)

            sub_path = tmp_path / "submission.csv"
            pd.DataFrame({"id": [1, 2, 3], "prediction": ["cat", "dog", "bird"]}).to_csv(sub_path, index=False)

            report = SubmissionPreflightValidator.audit_submission(
                sub_path,
                ref_path,
                id_column="id",
                target_column="prediction",
                probability_mode=False,
            )
            self.assertTrue(report.is_valid)
            self.assertEqual(len(report.errors), 0)

    def test_kaggle_submission_notebook_builder(self):
        from harness.kaggle_notebook_builder import build_kaggle_submission_notebook

        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td) / "exp001_baseline"
            exp_dir.mkdir(parents=True)
            nb_out = Path(td) / "submission.ipynb"

            res = build_kaggle_submission_notebook(
                exp_dir=exp_dir,
                output_path=nb_out,
                id_column="id",
                target_column="satisfaction",
            )
            self.assertEqual(res["status"], "SUCCESS")
            self.assertTrue(nb_out.exists())

            nb_data = json.loads(nb_out.read_text(encoding="utf-8"))
            self.assertEqual(nb_data["nbformat"], 4)
            self.assertGreaterEqual(len(nb_data["cells"]), 5)
            cell_sources = ["".join(c["source"]) for c in nb_data["cells"]]
            self.assertTrue(any("offline_wheels" in s or "pip install" in s for s in cell_sources))
            self.assertTrue(any("submission.csv" in s for s in cell_sources))


if __name__ == "__main__":
    unittest.main()


class DeepSubmissionTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, _ = make_workspace(root_path=Path(self.tmp.name))
        self.manager = SubmissionManager(project_root=self.root)
        self.entry = SubmissionEntry("sub_001", "exp1", "now", "SUBMITTED_PENDING",
            local_cv_scores={"accuracy": .2}, dispatch_intent_id="intent1",
            payload_sha256="payload", prediction_sha256="prediction")
        (self.manager.submissions_dir / self.entry.id).mkdir()
        (self.root / "submissions" / self.entry.id).mkdir(parents=True, exist_ok=True)
        self.manager.submissions[self.entry.id] = self.entry
        self.manager.save()
        self.observation = dict(intent_id="intent1", provider_ref="ref", partition="public",
            observation_id="observation1", metric="accuracy", score=.123456789,
            source="provider_api", payload_sha256="payload", prediction_sha256="prediction")

    def observe(self, **kwargs):
        return self.manager.record_official_observation(self.entry.id, client=object(), provider_ref="ref", **kwargs)

    def test_observation_commits_once_precision_and_active_identity(self):
        before = json.loads(self.manager.registry_file.read_text())["revision"]
        with patch("harness.submission_gate.SubmissionGate.resolve_official_score", return_value=self.observation):
            result = self.observe()
        self.assertIs(result, self.entry)
        self.assertIs(result, self.manager.submissions[result.id])
        self.assertEqual(json.loads(self.manager.registry_file.read_text())["revision"], before + 1)
        loaded = SubmissionManager(project_root=self.root).submissions[result.id]
        self.assertEqual(loaded.official_scores["accuracy"], .123456789)
        self.assertEqual(loaded.official_observation_id, "observation1")
        self.assertEqual(loaded.official_score_source, "provider_api")
        for directory in {self.manager.submissions_dir, self.root / "submissions"}:
            self.assertEqual(json.loads((directory / "registry.json").read_text())["submissions"][0]["official_scores"]["accuracy"], .123456789)
            self.assertEqual(json.loads((directory / result.id / "metadata.json").read_text())["official_scores"]["accuracy"], .123456789)
        result = self.manager.record_official_score(result.id, scores={"accuracy": .14999999})
        self.assertIsNone(result.official_observation_id)
        self.assertEqual(result.official_score_source, "declared")
        self.assertTrue(result.divergence_warning)

    def test_invalid_observation_and_commit_failure_leave_state_unchanged(self):
        original = self.manager.registry_file.read_bytes()
        for overrides in ({"intent_id": "other"}, {"partition": "private"}, {"payload_sha256": "other"},
                          {"metric": ""}, {"metric": " "}, {"metric": 1}, {"observation_id": ""},
                          {"score": True}, {"score": float("nan")}):
            with self.subTest(overrides=overrides), patch(
                    "harness.submission_gate.SubmissionGate.resolve_official_score", return_value={**self.observation, **overrides}):
                with self.assertRaises(ValueError):
                    self.observe()
                self.assertEqual(self.manager.registry_file.read_bytes(), original)
        with patch("harness.submission_gate.SubmissionGate.resolve_official_score", return_value=self.observation), patch(
                "harness.state_store.atomic_write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.observe()
        self.assertEqual(self.manager.registry_file.read_bytes(), original)
        self.assertEqual(self.entry.official_scores, {})

    def test_pending_provider_call_preserves_unrelated_changes_and_rejects_score_races(self):
        def unrelated(*args, **kwargs):
            other = SubmissionManager(project_root=self.root)
            other.submissions[self.entry.id].notes = "concurrent note"
            other.submissions["sub_002"] = SubmissionEntry("sub_002", "other", "now", "LOCAL_VERIFIED")
            other.save()
            return self.observation
        with patch("harness.submission_gate.SubmissionGate.resolve_official_score", side_effect=unrelated):
            self.observe()
        self.assertEqual(self.entry.notes, "concurrent note")
        self.assertIn("sub_002", self.manager.submissions)
        original_id = self.entry.official_observation_id
        def competing(*args, **kwargs):
            other = SubmissionManager(project_root=self.root)
            other.record_official_score(self.entry.id, scores={"accuracy": .3})
            return {**self.observation, "observation_id": "observation2"}
        with patch("harness.submission_gate.SubmissionGate.resolve_official_score", side_effect=competing):
            with self.assertRaisesRegex(ValueError, "Concurrent state conflict"):
                self.observe()
        self.assertEqual(self.entry.official_observation_id, original_id)
        self.assertEqual(SubmissionManager(project_root=self.root).submissions[self.entry.id].official_scores["accuracy"], .3)

    def test_deprecated_arguments_warn_only_for_nondefaults(self):
        import warnings
        with patch.object(self.manager, "register_plugin_submission", return_value=self.entry), patch(
                "harness.plugins.load_task_plugin", return_value=SimpleNamespace(capabilities=lambda: {
                    "submission_builder": True, "submission_validator": True})):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.manager.register_submission("exp1", candidate_csv="pred.csv")
            self.assertFalse([w for w in caught if issubclass(w.category, DeprecationWarning)])
            for kwargs in ({"baseline_csv": "base.csv"}, {"ensemble_components": ["a"]}, {"min_diff_pct": 3}):
                with self.assertWarnsRegex(DeprecationWarning, "does not enforce submission gates"):
                    self.manager.register_submission("exp1", candidate_csv="pred.csv", **kwargs)

    def test_sparse_registry_entry_can_receive_a_manual_score(self):
        atomic = {"submissions": [{"id": "legacy", "exp_id": "exp1", "created_at": "now", "status": "LOCAL_VERIFIED"}]}
        self.manager.registry_file.write_text(json.dumps(atomic))
        self.manager.load()
        result = self.manager.record_official_score("legacy", scores={"accuracy": .123456789})
        self.assertEqual(result.official_scores["accuracy"], .123456789)

    def test_reserved_names_in_both_packaging_paths(self):
        from harness.submission_manager import RESERVED_SUBMISSION_FILENAMES
        for name in RESERVED_SUBMISSION_FILENAMES:
            for variant in (name, name.swapcase()):
                with self.subTest(name=variant):
                    source = self.root / variant
                    source.write_text("payload")
                    with self.assertRaisesRegex(ValueError, "reserved metadata filename"):
                        self.manager.preserve_file_submission(file_path=source)
                    with self.assertRaisesRegex(ValueError, "reserved metadata filename"):
                        self.manager.register_plugin_submission(exp_id="exp1", candidate_csv=source,
                            plugin=mock.Mock(), plugin_context=mock.Mock(), file_name=variant)
        self.assertEqual(len(self.manager.submissions), 1)

    def test_plugin_cannot_return_or_create_reserved_metadata(self):
        source = self.root / "predictions.csv"
        source.write_text("id,pred\na,1\n")
        for side_file in (False, True):
            def build(context, predictions, target):
                target.write_text("payload")
                reserved = target.parent / "Metadata.JSON"
                reserved.write_text("malicious")
                return target if side_file else reserved
            plugin = mock.Mock()
            plugin.build_submission.side_effect = build
            with patch("harness.artifacts.validate_test_prediction"), self.assertRaises(ValueError):
                self.manager.register_plugin_submission(exp_id="exp1", candidate_csv=source,
                    plugin=plugin, plugin_context=mock.Mock())
            self.assertEqual(len(self.manager.submissions), 1)
            self.assertFalse((self.manager.submissions_dir / "sub_002_exp1").exists())
            self.assertFalse((self.root / "submissions/sub_002_exp1").exists())

    def test_filename_aliases_are_rejected_before_builder_or_copy(self):
        source = self.root / "payload.csv"
        source.write_text("payload")
        for name in ("payload.csv.", "payload.csv ", "../payload.csv", "a/b.csv", "a\\b.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.manager.register_plugin_submission(exp_id="exp1", candidate_csv=source,
                    plugin=mock.Mock(), plugin_context=mock.Mock(), file_name=name)
        for suffix in (".", " "):
            with self.assertRaises(ValueError):
                self.manager.preserve_file_submission(file_path=str(source) + suffix)
