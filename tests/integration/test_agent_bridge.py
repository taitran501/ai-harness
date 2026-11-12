"""Consolidated integration tests for AgentBridge and IDE agent interactions.

Consolidates:
- tests/agent_bridge/test_core_bridge.py (TestAgentBridge)
- tests/agent_bridge/test_cloud_hygiene_bypass.py (CloudAvailabilityMappingTests, HygieneBypassDetectionTests, AgentsMdTests, SkillConsolidationTests)
- tests/agent_bridge/test_friction_fixes.py (all 9 classes)
- tests/agent_bridge/test_grounding_guards.py (TestGroundingVerifiedViaEnforced, TestDuplicateCandidateGuard, TestSuggestNextInterventions)
- tests/agent_bridge/test_state_readiness_copilot.py (all 9 classes)
- tests/test_f821_undefined_names.py (TestInspectExperimentHandler)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from harness.agent_bridge import (
    MAX_SESSION_ACTIONS,
    MAX_SESSION_OBSERVATIONS,
    AgentBridge,
    TypedActionName,
)
from harness.endgame_submission import SubmissionPreflightValidator, ValidationReport
from harness.grounding_gate import GroundingGate
from harness.project import (
    AGENTS_MD_MARKER_END,
    AGENTS_MD_MARKER_START,
    attach_project,
    update_agents_md,
)
from harness.research_memory import ResearchMemory
from harness.runner import UniversalRunner, build_parser
from harness.skill_injector import generate_skill_content
from harness.task_spec import SubtaskSpec, UniversalTaskSpec
from harness.templates.kaggle_template import _is_critical_identity_mismatch
from tests.fixtures import (
    copy_repo_configs,
    make_workspace as _make_workspace,
    task,
    text_task_document as task_document,
)

_TASK = task_document()


def _make_bridge_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="test_friction_"))
    copy_repo_configs(workspace)
    return workspace


def _make_bridge(root: Path) -> AgentBridge:
    copy_repo_configs(root)
    return AgentBridge(root)


# ===========================================================================
# 1. Core AgentBridge
# ===========================================================================

class TestAgentBridge(unittest.TestCase):
    def setUp(self) -> None:
        self._workspace = tempfile.TemporaryDirectory(prefix="test_agent_bridge_")
        self.project_root = Path(self._workspace.name)
        copy_repo_configs(self.project_root)
        self.bridge = AgentBridge(self.project_root)

    def tearDown(self) -> None:
        self._workspace.cleanup()

    def test_context_packet_and_markdown(self) -> None:
        pkt = self.bridge.get_context_packet()
        self.assertEqual(pkt["runtime_owner"], "Antigravity (IDE)")
        self.assertIn("competition_hud", pkt)
        self.assertIn("ucb_recommendation", pkt)
        self.assertIn("available_typed_actions", pkt)

        md = self.bridge.render_context_markdown()
        self.assertIn("ANTIGRAVITY <-> HARNESS CONTEXT PACKET", md)
        self.assertIn("Typed Bridge Actions", md)

    def test_create_experiment_passes_grounding_gate(self) -> None:
        sess = self.bridge.start_session(goal="Test GroundingGate schema compliance")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp999_feat_bridge_test",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "bridge_test",
                "source_type": "verified_baseline",
                "source_id": "sklearn_tfidf_logreg_canonical",
                "retrieval_summary": "Canonical TFIDF/logreg baseline; parent diagnostics show rare-slice underfitting.",
                "verified_via": "local_registry",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["grounding_status"], "PASSED")
        cfg_file = self.project_root / "configs" / "exp999_feat_bridge_test.yaml"
        self.assertTrue(cfg_file.exists())

        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            passed, errors, _ = GroundingGate.verify_config(
                cfg_data, raise_on_error=False
            )
            self.assertTrue(passed, f"GroundingGate failed with errors: {errors}")
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_create_experiment_unverified_grounding_flagged(self) -> None:
        sess = self.bridge.start_session(goal="Test unverified grounding is flagged")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp997_unverified_grounding",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "no_source",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["grounding_status"], "UNVERIFIED")
        self.assertEqual(obs.data["grounding_verification"], "unverified_placeholder")
        self.assertTrue(obs.data["grounding_errors"])
        cfg_file = self.project_root / "configs" / "exp997_unverified_grounding.yaml"
        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_data["grounding"]["verified_via"], "not_verified")
            passed, errors, _ = GroundingGate.verify_config(cfg_data, raise_on_error=False)
            self.assertFalse(passed)
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_create_experiment_flexible_model_config(self) -> None:
        sess = self.bridge.start_session(goal="Test flexible model config in create_experiment")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp998_feat_custom_model",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "custom_model_test",
                "model": {
                    "name": "xgboost",
                    "params": {"n_estimators": 50, "max_depth": 4},
                },
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.project_root / "configs" / "exp998_feat_custom_model.yaml"
        self.assertTrue(cfg_file.exists())
        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_data["model"]["name"], "xgboost")
            self.assertEqual(cfg_data["model"]["params"]["n_estimators"], 50)
            self.assertEqual(cfg_data["model"]["params"]["max_depth"], 4)
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_create_experiment_preserves_custom_factory(self) -> None:
        sess = self.bridge.start_session(goal="Test custom factory model config in create_experiment")
        sess, obs = self.bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp997_feat_custom_factory",
                "parent_id": "baseline_v1_comment",
                "branch": "feat",
                "descriptor": "custom_factory_test",
                "model": {
                    "factory": "custom_module.models.build_custom_model",
                    "params": {"learning_rate": 1e-4, "hidden_dim": 256},
                },
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.project_root / "configs" / "exp997_feat_custom_factory.yaml"
        self.assertTrue(cfg_file.exists())
        try:
            cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_data["model"]["factory"], "custom_module.models.build_custom_model")
            self.assertEqual(cfg_data["model"]["params"]["learning_rate"], 1e-4)
            self.assertEqual(cfg_data["model"]["params"]["hidden_dim"], 256)
        finally:
            if cfg_file.exists():
                cfg_file.unlink()

    def test_session_budget_and_approval_guards(self) -> None:
        sess = self.bridge.start_session(
            goal="Test session budget & USER_APPROVAL enforcement",
            budget={"max_experiments": 1, "max_consecutive_rejections": 2},
        )

        sess, obs_pause = self.bridge.execute_action(
            action=TypedActionName.FREEZE_CANDIDATE.value,
            params={"slot_1_id": "baseline_v1_comment"},
            session_id=sess.session_id,
            user_approved=False,
        )
        self.assertEqual(obs_pause.status, "PAUSED_FOR_APPROVAL")
        self.assertEqual(sess.status, "PAUSED_FOR_APPROVAL")
        self.assertIsNotNone(sess.pending_approval_action)

        sess, obs_approved = self.bridge.approve_pending_action(sess.session_id)
        self.assertIsNotNone(obs_approved)
        self.assertEqual(obs_approved.status, "FAILED")
        sess, obs = self.bridge.execute_action(
            "synthesize_candidate",
            params={"name": "missing", "mode": "specialist_router"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "FAILED")
        self.assertEqual(sess.compute_usage["experiments_used"], 0)
        from harness.resource_ledger import reserve

        reserve(
            self.bridge.artifacts_dir / "resource_ledger.json",
            sess.session_id,
            "run_full_validation",
            "local",
            {"max_experiments": 1},
        )
        _, obs = self.bridge.execute_action(
            "run_full_validation",
            params={"config_path": "missing.yaml"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "BLOCKED_BY_BUDGET")

    def test_soft_stop_reassess_recommended_and_spec_min_improvement(self) -> None:
        spec_min = self.bridge._get_spec_min_improvement()
        self.assertEqual(spec_min, 0.0)

        sess = self.bridge.start_session(
            goal="Test soft stop REASSESS_RECOMMENDED vs hard stop",
            budget={
                "max_consecutive_rejections": 1,
                "min_meaningful_improvement": 99.0,
                "hard_stop_on_consecutive_rejections": False,
            },
        )
        sess, obs1 = self.bridge.execute_action(
            action=TypedActionName.SYNTHESIZE_CANDIDATE.value,
            params={"name": "soft_stop_probe_1", "mode": "specialist_router"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs1.status, "FAILED")
        sess.consecutive_rejections = 1
        self.bridge.save_session(sess)
        self.assertIsNone(
            self.bridge._check_budget_and_stop_guards(sess, "run_full_validation", {})
        )
        sess.budget["hard_stop_on_consecutive_rejections"] = True
        self.assertEqual(
            self.bridge._check_budget_and_stop_guards(sess, "run_full_validation", {})[0],
            "STOP_POLICY_TRIGGERED",
        )

    def test_bootstrap_task_typed_action(self) -> None:
        t_dir = Path(tempfile.mkdtemp(prefix="test_bridge_bootstrap_"))
        try:
            d_dir = t_dir / "dataset"
            d_dir.mkdir(parents=True)
            df = pd.DataFrame(
                {"id": [1, 2, 3], "feat": ["a", "b", "c"], "target": [0, 1, 0]}
            )
            df.to_csv(d_dir / "train.csv", index=False)
            doc_file = t_dir / "TASK.md"
            doc_file.write_text(
                "- **Competition Name**: Bridge Boot Test\n- **Target Columns**: target\n",
                encoding="utf-8",
            )

            sess = self.bridge.start_session(goal="Test bootstrap_task typed action")
            sess, obs = self.bridge.execute_action(
                action=TypedActionName.BOOTSTRAP_TASK.value,
                params={
                    "dataset_dir": str(d_dir),
                    "context_file": str(doc_file),
                    "output_config": str(t_dir / "task.yaml"),
                    "project_root": str(t_dir),
                },
                session_id=sess.session_id,
            )
            self.assertEqual(obs.status, "SUCCESS", f"Failed with: {obs.summary}")
            self.assertIn("Bootstrapped task", obs.summary)
        finally:
            shutil.rmtree(t_dir, ignore_errors=True)


# ===========================================================================
# 2. Cloud Hygiene Bypass & Adoption
# ===========================================================================

class CloudAvailabilityMappingTests(unittest.TestCase):
    def test_kaggle_authenticated_reports_available(self):
        from harness import status_report

        env = {
            "kaggle": {"installed": True, "authenticated": True, "username": "u"},
            "modal": {"installed": True, "has_credentials": True},
            "docker": {"installed": True, "running": False},
            "gpu": {"detected": True},
            "llm": {"total_configured": 0},
        }
        with patch("harness.environment_detector.EnvironmentDetector.get_full_status",
                   return_value=env):
            section = status_report._cloud_section(None)
        accounts = section["accounts"]
        self.assertTrue(accounts["kaggle"]["available"])
        self.assertTrue(accounts["modal"]["available"])
        self.assertFalse(accounts["docker"]["available"])
        self.assertTrue(accounts["gpu"]["available"])
        self.assertFalse(accounts["llm"]["available"])

    def test_kaggle_not_authenticated_reports_unavailable(self):
        from harness import status_report

        env = {
            "kaggle": {"installed": True, "authenticated": False, "username": None},
            "modal": {"installed": False, "has_credentials": False},
            "docker": {"installed": False},
            "gpu": {"detected": False},
            "llm": {"total_configured": 0},
        }
        with patch("harness.environment_detector.EnvironmentDetector.get_full_status",
                   return_value=env):
            section = status_report._cloud_section(None)
        self.assertFalse(section["accounts"]["kaggle"]["available"])
        self.assertFalse(section["accounts"]["modal"]["available"])


class HygieneBypassDetectionTests(unittest.TestCase):
    def _workspace(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-bypass-hygiene-")
        root = Path(temp.name)
        (root / "configs").mkdir()
        (root / "configs/task.yaml").write_text(yaml.safe_dump(_TASK))
        manual = root / "kaggle_sub_exp010_manual"
        manual.mkdir()
        (manual / "submission.csv").write_text("id,label\n")
        sealed = root / "kaggle_sub_exp011_harness"
        sealed.mkdir()
        (sealed / "submission.csv").write_text("id,label\n")
        (sealed / "artifact_manifest.json").write_text("{}")
        return temp, root

    def test_manual_and_sealed_dirs_are_reported_separately(self):
        from harness.hygiene import scan_workspace_hygiene

        temp, root = self._workspace()
        try:
            result = scan_workspace_hygiene(root)
        finally:
            temp.cleanup()
        details = {f["path"]: f for f in result["findings"]}
        self.assertFalse(details["kaggle_sub_exp010_manual"].get("sealed"))
        self.assertTrue(details["kaggle_sub_exp011_harness"].get("sealed"))

    def test_aggregated_bypass_finding_when_many_manual_dirs(self):
        from harness.hygiene import scan_workspace_hygiene

        temp, root = self._workspace()
        try:
            for i in range(3, 10):
                d = root / f"kaggle_sub_exp{i:03d}_manual"
                d.mkdir()
                (d / "submission.csv").write_text("id,label\n")
            result = scan_workspace_hygiene(root)
        finally:
            temp.cleanup()
        bypass = [f for f in result["findings"] if f["kind"] == "bypassed_run_output"]
        stale = [f for f in result["findings"] if f["kind"] == "stale_run_output"]
        self.assertEqual(len(bypass), 1)
        self.assertIn("artifact manifest", bypass[0]["detail"])
        self.assertIn("manual kaggle pushes", bypass[0]["detail"])
        self.assertTrue(any("exp011_harness" in f["detail"] for f in stale),
                        "sealed dir should stay in the stale group")


class AgentsMdTests(unittest.TestCase):
    def _attach(self, root):
        (root / "configs").mkdir(exist_ok=True)
        (root / "configs/task.yaml").write_text(yaml.safe_dump(_TASK))
        return attach_project(root, task_config=root / "configs/task.yaml", force=True)

    def test_attach_creates_managed_agents_md(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-agents-md-")
        root = Path(temp.name)
        try:
            result = self._attach(root)
            self.assertEqual(result["agents_md"], "created")
            content = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn(AGENTS_MD_MARKER_START, content)
            self.assertIn(AGENTS_MD_MARKER_END, content)
            self.assertIn("Never push kernels manually", content)
            self.assertIn(".agents/skills/harness/SKILL.md", content)
            self.assertIn("NO UNGROUNDED BRAINSTORMING", content)
            self.assertIn("STANDARDIZED SUBMISSIONS", content)
            self.assertTrue((root / ".agents" / "rules" / "harness.md").is_file())
        finally:
            temp.cleanup()

    def test_user_owned_agents_md_is_not_touched(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-agents-md-user-")
        root = Path(temp.name)
        (root / "AGENTS.md").write_text("# My own rules\n- do the thing\n", encoding="utf-8")
        try:
            result = self._attach(root)
            self.assertEqual(result["agents_md"], "skipped_user_owned")
            self.assertEqual(
                (root / "AGENTS.md").read_text(encoding="utf-8"),
                "# My own rules\n- do the thing\n",
            )
        finally:
            temp.cleanup()

    def test_managed_block_refresh_preserves_user_content(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-agents-md-refresh-")
        root = Path(temp.name)
        try:
            self._attach(root)
            agents = root / "AGENTS.md"
            original = agents.read_text(encoding="utf-8")
            with_marker = original.replace(
                AGENTS_MD_MARKER_END,
                AGENTS_MD_MARKER_END + "\n\n## User section\n- custom rule kept\n",
            )
            agents.write_text(with_marker, encoding="utf-8")
            updated = update_agents_md(root)
            self.assertEqual(updated["status"], "refreshed")
            content = agents.read_text(encoding="utf-8")
            self.assertIn("custom rule kept", content)
            self.assertEqual(content.count(AGENTS_MD_MARKER_START), 1)
        finally:
            temp.cleanup()


class SkillConsolidationTests(unittest.TestCase):
    def test_attach_skill_matches_sync_skill_content(self):
        temp = tempfile.TemporaryDirectory(prefix="aih-skill-consolidation-")
        root = Path(temp.name)
        try:
            (root / "configs").mkdir()
            (root / "configs/task.yaml").write_text(yaml.safe_dump(_TASK))
            attach_project(root, task_config=root / "configs/task.yaml", force=True)
            written = (root / ".agents/skills/harness/SKILL.md").read_text(encoding="utf-8")
            self.assertFalse((root / ".agents/skills/ai-harness").exists())

            generated = generate_skill_content(_TASK, artifact_path=".ai-harness/artifacts/<task-id>")
            for probe in (
                "Research continuity action contracts",
                "trace_research_insight",
                "confirm_task_spec",
                "phase render",
                "script adapter",
            ):
                self.assertIn(probe, written, f"attached skill missing: {probe}")
                self.assertIn(probe, generated)
        finally:
            temp.cleanup()


# ===========================================================================
# 3. Friction Fixes
# ===========================================================================

class TestValidationReportStdVal(unittest.TestCase):
    def _audit(self, values):
        tmp = Path(tempfile.mkdtemp(prefix="test_stdval_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        ids = list(range(len(values)))
        sub = pd.DataFrame({"id": ids, "target": values})
        ref = pd.DataFrame({"id": ids})
        sub_path = tmp / "submission.csv"
        ref_path = tmp / "test.csv"
        sub.to_csv(sub_path, index=False)
        ref.to_csv(ref_path, index=False)
        return SubmissionPreflightValidator.audit_submission(
            sub_path, ref_path, target_column="target"
        )

    def test_varied_predictions_have_positive_std(self) -> None:
        report = self._audit([0.1, 0.5, 0.9, 0.3])
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(hasattr(report, "std_val"))
        self.assertGreater(report.std_val, 0.0)

    def test_constant_predictions_have_zero_std(self) -> None:
        report = self._audit([0.5, 0.5, 0.5])
        self.assertEqual(report.std_val, 0.0)

    def test_positional_constructor_still_works(self) -> None:
        report = ValidationReport(
            True, 3, "id", "target", 0.0, 1.0, 0.5, 0, []
        )
        self.assertEqual(report.std_val, 0.0)


class TestRunnerExpectedValidationIdsEagerInit(unittest.TestCase):
    def test_attribute_present_after_construction(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_evi_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "configs").mkdir()
        (tmp / "dataset").mkdir()
        spec = task()
        doc = {"task": spec.metadata}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        (tmp / "configs/task.yaml").write_text(yaml.safe_dump(doc))
        pd.DataFrame(
            {
                "id": range(12),
                "group": range(12),
                "x": range(12),
                "category": ["a", "b"] * 6,
                "target": [x * 2.0 for x in range(12)],
            }
        ).to_csv(tmp / "dataset/train.csv", index=False)

        args = SimpleNamespace(
            project_root=str(tmp),
            task_config=None,
            fold=None,
            debug=False,
            probe_samples=None,
        )
        cfg = {
            "experiment": {
                "id": "exp001",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "baseline",
                "hypothesis": "linear control",
                "intervention": "train ridge",
            },
            "model": {"name": "ridge", "params": {"alpha": 0.1}},
        }
        runner = UniversalRunner(cfg, "config.yaml", args)
        self.assertTrue(hasattr(runner, "expected_validation_ids"))
        self.assertIsInstance(runner.expected_validation_ids, set)
        self.assertEqual(runner.expected_validation_ids, set())


class TestAgentBridgeTypedActionAliases(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _make_bridge_workspace()
        self.addCleanup(shutil.rmtree, self.ws, True)
        self.bridge = AgentBridge(self.ws)

    def test_instance_alias_exists_for_every_typed_action(self) -> None:
        for action in TypedActionName:
            with self.subTest(action=action.value):
                self.assertTrue(callable(getattr(self.bridge, action.value)))

    def test_class_level_access_returns_callable_proxy(self) -> None:
        proxy = AgentBridge.review_hypothesis  # type: ignore[attr-defined]
        self.assertTrue(callable(proxy))
        self.assertEqual(proxy.__name__, "review_hypothesis")
        self.assertIn("review_hypothesis", proxy.__doc__ or "")
        with self.assertRaises(AttributeError) as ctx:
            _ = AgentBridge.not_a_real_action  # type: ignore[attr-defined]
        self.assertIn("not_a_real_action", str(ctx.exception))

    def test_alias_forwards_to_execute_action(self) -> None:
        observation = self.bridge.get_next_experiment_id(
            params={"branch": "feat", "descriptor": "iteration"}
        )
        self.assertEqual(observation.status, "SUCCESS")
        self.assertEqual(observation.action, "get_next_experiment_id")
        self.assertIn("next_num", observation.data)

    def test_alias_routes_reserved_keywords(self) -> None:
        observation = self.bridge.get_next_experiment_id(
            {}, rationale="friction regression test"
        )
        self.assertEqual(observation.status, "SUCCESS")


class TestResearchMemoryEntriesDualAccess(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_entries_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "configs").mkdir()
        (tmp / "configs/task.yaml").write_text(
            yaml.safe_dump(task().metadata)
        )
        self.spec = UniversalTaskSpec.from_dict(task().metadata)
        artifacts = tmp / "artifacts"
        artifacts.mkdir()
        self.memory = ResearchMemory(tmp, artifacts, self.spec)

    def test_property_and_callable_both_work(self) -> None:
        entry = self.memory.record_insight(
            title="Minority recall",
            observation="Minority errors persist",
            mechanism="Loss weighting could help",
            next_action="Compare a fixed control",
        )
        entries = self.memory.entries
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], entry["id"])
        self.assertEqual(len(self.memory.entries()), 1)
        self.assertEqual(self.memory.entries[-1]["id"], entry["id"])
        self.assertEqual(entries[0].id, entry["id"])
        self.assertEqual(entries[0].kind, "insight")
        self.assertEqual(entries[0].title, "Minority recall")
        json.dumps(entries)
        self.assertEqual(entries[0].get("scope"), self.memory.scope)


class TestKaggleJobIdMismatchPolicy(unittest.TestCase):
    def test_core_version_drift_is_never_critical(self) -> None:
        self.assertFalse(
            _is_critical_identity_mismatch(
                "core_version", {"expected": "1.0", "actual": "1.1"}
            )
        )

    def test_unset_remote_job_id_is_not_critical(self) -> None:
        self.assertFalse(
            _is_critical_identity_mismatch("job_id", {"expected": "job_abc", "actual": None})
        )
        self.assertFalse(
            _is_critical_identity_mismatch("job_id", {"expected": "job_abc", "actual": ""})
        )

    def test_conflicting_job_id_is_critical(self) -> None:
        self.assertTrue(
            _is_critical_identity_mismatch("job_id", {"expected": "job_abc", "actual": "job_xyz"})
        )

    def test_task_id_conflict_is_critical(self) -> None:
        self.assertTrue(
            _is_critical_identity_mismatch("task_id", {"expected": "t1", "actual": "t2"})
        )


class TestUploadConsentGrantKwargs(unittest.TestCase):
    def test_grant_accepts_confirmed_upload_and_flags(self) -> None:
        ws = _make_bridge_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        bridge = AgentBridge(ws)
        from harness.cloud_policy import UploadConsentStore

        store = UploadConsentStore(bridge.project)
        record = store.grant(
            "kaggle",
            {"task_id": "t", "fingerprint": "fp123", "file_count": 2},
            confirmed_upload=True,
            visibility="private",
        )
        self.assertTrue(record["granted"])
        self.assertTrue(record["confirmed_upload"])
        self.assertEqual(record["flags"], {"visibility": "private"})
        persisted = json.loads(store.path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["consents"][-1]["fingerprint"], "fp123")
        self.assertTrue(store.is_granted("kaggle", "fp123"))


class TestSubtaskSpecMetricAlias(unittest.TestCase):
    def test_metric_alias_sets_primary_metric(self) -> None:
        spec = SubtaskSpec(id="target_a", metric="rmsle")
        self.assertEqual(spec.primary_metric, "rmsle")
        self.assertEqual(spec.name, "target_a")

    def test_matching_aliases_accepted(self) -> None:
        spec = SubtaskSpec(id="t", primary_metric="rmsle", metric="rmsle")
        self.assertEqual(spec.primary_metric, "rmsle")

    def test_conflicting_aliases_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SubtaskSpec(id="t", primary_metric="rmsle", metric="mae")
        self.assertIn("Conflicting metric aliases", str(ctx.exception))

    def test_positional_construction_still_works(self) -> None:
        spec = SubtaskSpec("rank_target", "Rank", "ranking", column="rank_target", classes=[])
        self.assertEqual(spec.task_type, "ranking")
        self.assertEqual(spec.primary_metric, "macro_f1")

    def test_from_dict_still_prefers_primary_metric(self) -> None:
        spec = SubtaskSpec.from_dict({"id": "t", "primary_metric": "accuracy"})
        self.assertEqual(spec.primary_metric, "accuracy")
        spec2 = SubtaskSpec.from_dict({"id": "t", "metric": "f1"})
        self.assertEqual(spec2.primary_metric, "f1")


class TestCreateExperimentAutoIdOnCollision(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _make_bridge_workspace()
        self.addCleanup(shutil.rmtree, self.ws, True)
        self.bridge = AgentBridge(self.ws)

    def test_collision_auto_assigns_and_reports(self) -> None:
        collision_dir = (
            self.ws / ".ai-harness" / "artifacts" / "task_test" / "exp001_feat_iteration"
        )
        collision_dir.mkdir(parents=True)
        (collision_dir / "artifact_manifest.json").write_text("{}")

        _, obs = self.bridge.execute_action(
            "create_experiment",
            params={
                "exp_id": "exp001_feat_iteration",
                "branch": "feat",
                "descriptor": "iteration",
                "hypothesis": "feature engineering helps",
                "intervention": "add engineered features",
                "problem_addressed": "baseline underperforms",
            },
            goal="auto id test",
        )
        self.assertEqual(obs.status, "SUCCESS")
        auto_id = obs.data.get("auto_assigned_exp_id")
        self.assertIsNotNone(auto_id)
        self.assertNotEqual(auto_id, "exp001_feat_iteration")
        self.assertTrue(auto_id.startswith("exp"))
        self.assertIn("auto-assigned", obs.summary)
        self.assertTrue((collision_dir / "artifact_manifest.json").exists())
        self.assertTrue((self.ws / "configs" / f"{auto_id}.yaml").exists())

    def test_free_exp_id_is_used_as_is(self) -> None:
        _, obs = self.bridge.execute_action(
            "create_experiment",
            params={
                "exp_id": "exp099_feat_custom",
                "branch": "feat",
                "descriptor": "custom",
                "hypothesis": "h",
                "intervention": "i",
                "problem_addressed": "p",
            },
            goal="free id test",
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data.get("exp_id"), "exp099_feat_custom")
        self.assertIsNone(obs.data.get("auto_assigned_exp_id"))


class TestWaitCloudJobKernelSlugFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _make_bridge_workspace()
        self.bridge = AgentBridge(self.ws)

    def tearDown(self) -> None:
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_wait_cloud_job_with_kernel_slug_fallback(self) -> None:
        fake_wait = {"kernel_slug": "taitran501/test-taaf-m2", "status": "COMPLETED", "success": True}
        with patch("harness.templates.kaggle_template.KaggleKernelManager.wait_for_kernel", return_value=fake_wait) as m:
            _, obs = self.bridge.execute_action(
                "wait_cloud_job",
                params={"job_id": "taitran501/test-taaf-m2", "timeout_seconds": 60, "poll_interval": 2},
                goal="wait for remote kernel",
            )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertIn("taitran501/test-taaf-m2", obs.summary)
        self.assertEqual(obs.data["job"]["status"], "COMPLETED")
        m.assert_called_once_with(exp_id="", kernel_slug="taitran501/test-taaf-m2", poll_interval=2.0, timeout_seconds=60.0)

    def test_inspect_cloud_job_with_kernel_slug_fallback(self) -> None:
        fake_status = {"kernel_slug": "taitran501/test-taaf-m2", "status": "RUNNING"}
        with patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status", return_value=fake_status) as m:
            _, obs = self.bridge.execute_action(
                "inspect_cloud_job",
                params={"job_id": "taitran501/test-taaf-m2"},
                goal="inspect remote kernel",
            )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["job"]["status"], "RUNNING")
        m.assert_called_once_with(exp_id="", kernel_slug="taitran501/test-taaf-m2")


# ===========================================================================
# 4. Grounding Guards
# ===========================================================================

class TestGroundingVerifiedViaEnforced(unittest.TestCase):
    def _cfg(self, verified_via):
        cfg = {
            "experiment": {
                "id": "exp020_feat_check",
                "branch": "feat",
                "parent_id": "exp001_baseline",
                "problem_addressed": "Address the rare-slice underfitting seen in parent diagnostics.",
                "hypothesis": "Targeted feature cleaning lifts rare-slice F1 by at least 0.02.",
                "intervention": "Apply a single targeted cleaning step to the input.",
            },
            "grounding": {
                "source_type": "arxiv",
                "source_id": "arxiv:2305.12345",
                "retrieval_summary": "Cites an empirical method that targets the diagnosed slice.",
            },
            "assumption_register": {
                "known_facts": ["Parent exp001 achieved F1=0.62."],
                "assumptions": ["Cleaning helps rare slices without raising variance."],
                "uncertainty_level": "low",
            },
        }
        if verified_via is not None:
            cfg["grounding"]["verified_via"] = verified_via
        return cfg

    def test_unverified_rejected(self) -> None:
        for bad in ("not_verified", "", "unverified", "none"):
            cfg = self._cfg(bad)
            ok, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
            self.assertFalse(ok, f"Should reject verified_via={bad!r}")
            self.assertTrue(any("GROUNDING VERIFICATION ERROR" in e for e in errors))

    def test_verified_passes(self) -> None:
        for good in ("search_web", "read_url_content", "local_registry"):
            ok, errors, _ = GroundingGate.verify_config(self._cfg(good), raise_on_error=False)
            self.assertTrue(ok, f"Should accept verified_via={good!r}: {errors}")


class TestDuplicateCandidateGuard(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_dupcand_")
        self.root = Path(self._ws.name)
        self.bridge = _make_bridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def _prime_pool(self, fingerprint):
        self.bridge._refresh_task_context()
        self.bridge.comp_state.state["candidate_pool"]["synthesized_candidates"] = [
            {"id": "cand_0001", "name": "first", "mode": "single_model",
             "recipe_fingerprint": fingerprint}
        ]
        return patch.object(self.bridge, "_refresh_task_context")

    def test_duplicate_recipe_blocked(self) -> None:
        fp = "abc"
        with self._prime_pool(fp), \
             patch("harness.candidates._candidate_recipe_fingerprint", return_value=fp), \
             patch.object(self.bridge.comp_state, "synthesize_candidate") as synth:
            synth.return_value = {"id": "cand_0002", "recipe_fingerprint": fp}
            _sess, obs = self.bridge.execute_action(
                action=TypedActionName.PROMOTE_CANDIDATE.value,
                params={"exp_id": "exp005"},
            )
        self.assertEqual(obs.status, "FAILED")
        self.assertIn("already", obs.summary.lower())
        synth.assert_not_called()

    def test_revisit_reason_allows_duplicate(self) -> None:
        fp = "abc"
        with self._prime_pool(fp), \
             patch("harness.candidates._candidate_recipe_fingerprint", return_value=fp), \
             patch.object(self.bridge.comp_state, "synthesize_candidate") as synth:
            synth.return_value = {"id": "cand_0002", "recipe_fingerprint": fp}
            _sess, _obs = self.bridge.execute_action(
                action=TypedActionName.PROMOTE_CANDIDATE.value,
                params={"exp_id": "exp005", "revisit_reason": "re-evaluate after data fix"},
            )
        synth.assert_called_once()


class TestSuggestNextInterventions(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_suggest_")
        self.root = Path(self._ws.name)
        self.bridge = _make_bridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_returns_well_formed_suggestions(self) -> None:
        with patch.object(self.bridge, "_new_journal") as journal_mock, \
             patch.object(self.bridge, "_research_memory") as mem_mock:
            journal = journal_mock.return_value
            journal.nodes = {}
            journal.recommend_next_branch.return_value = {
                "parent_id": "exp000_baseline",
                "recommended_branch": "feat",
                "ucb_score": 0.5,
                "bottlenecks": ["rare-slice underfitting"],
                "diagnostic_cue": "rare-slice underfitting",
            }
            mem = mem_mock.return_value
            mem.context.return_value = {
                "research_frontier": [
                    {"insight_id": "i1", "title": "Try target attention",
                     "next_action": "Add target attention to the head",
                     "mechanism": "Cross-attention aligns rare slices.",
                     "status": "needs_probe_plan"},
                ],
                "reassess_recommended": False,
                "consecutive_non_improvements": 0,
            }
            res = self.bridge._suggest_next_interventions(limit=5)

        self.assertIn("suggestions", res)
        self.assertIn("stagnation", res)
        self.assertEqual(res["parent_id"], "exp000_baseline")
        sources = [s["source"] for s in res["suggestions"]]
        self.assertIn("research_frontier", sources)

    def test_graveyard_filter(self) -> None:
        with patch.object(self.bridge, "_new_journal") as journal_mock, \
             patch.object(self.bridge, "_research_memory") as mem_mock:
            journal = journal_mock.return_value
            journal.nodes = {}
            journal.recommend_next_branch.return_value = {
                "parent_id": "exp000_baseline",
                "recommended_branch": "feat",
                "ucb_score": 0.5,
                "bottlenecks": [],
                "diagnostic_cue": None,
            }
            mem = mem_mock.return_value
            mem.context.return_value = {
                "research_frontier": [
                    {"insight_id": "i1", "title": "Try stem cleaning",
                     "next_action": "Try stem cleaning on the input",
                     "mechanism": "", "status": "needs_probe_plan"},
                ],
                "reassess_recommended": False,
                "consecutive_non_improvements": 0,
            }
        with patch.object(self.bridge, "_extract_idea_graveyard",
                          return_value=["Try stem cleaning on the input"]):
            res = self.bridge._suggest_next_interventions(limit=5)
        for s in res["suggestions"]:
            self.assertNotIn("stem cleaning", s["hypothesis"].lower(),
                             "Graveyard idea should not be re-suggested")


# ===========================================================================
# 5. State Readiness & Co-pilot
# ===========================================================================

class TestDataReadinessMissingDataset(unittest.TestCase):
    def test_missing_dataset_is_blocked_not_ready(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace(with_train_file=False, train_path="dataset/missing.csv")
        try:
            comp = CompetitionState(project_root=root)
            report = comp.run_data_readiness_audit()
            self.assertEqual(report["dataset_overview"]["total_rows"], 0)
            self.assertEqual(report["status"], "BLOCKED")
            ids = [w.get("id") for w in report["warnings"]]
            self.assertIn("DATA_MISSING", ids)
            data_missing = next(w for w in report["warnings"] if w["id"] == "DATA_MISSING")
            self.assertEqual(data_missing["severity"], "critical")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_present_dataset_is_not_blocked_by_guard(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace(with_train_file=True)
        try:
            comp = CompetitionState(project_root=root)
            report = comp.run_data_readiness_audit()
            self.assertGreater(report["dataset_overview"]["total_rows"], 0)
            self.assertNotEqual(report["status"], "BLOCKED")
            self.assertNotIn(
                "DATA_MISSING", [w.get("id") for w in report["warnings"]]
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


_MAPPINGS = {
    "CONFIRM_TASK_SPEC": "confirm_task_spec",
    "CONFIRM_DATA_RISKS": "inspect_data_readiness",
    "LOCK_BASELINE_ANCHOR": "establish_baseline_anchor",
    "BUILD_SPECIALIST_ROUTER": "inspect_diagnostics",
    "FREEZE_FOR_PRIVATE_LB": "freeze_candidate",
    "GENERATE_FINAL_PAPER": "finalize_competition",
}


class TestCopilotActionMapping(unittest.TestCase):
    def _snapshot_for(self, action_id: str) -> dict:
        return {
            "hud": {"what_should_i_do_next": {"action_id": action_id}},
            "candidate_pool": {"baseline_id": "exp001", "baseline_score": 0.5},
        }

    def test_all_hud_recommendation_ids_map_to_real_actions(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace()
        try:
            comp = CompetitionState(project_root=root)
            calls = []

            class FakeBridge:
                def __init__(self, *a, **k):
                    pass

                def execute_action(self, action, params=None):
                    calls.append((action, dict(params or {})))
                    return (
                        SimpleNamespace(),
                        SimpleNamespace(
                            __dict__={"status": "SUCCESS", "action": action}
                        ),
                    )

            for action_id, expected_action in _MAPPINGS.items():
                calls.clear()
                with (
                    patch(
                        "harness.agent_bridge.AgentBridge", FakeBridge
                    ),
                    patch.object(
                        CompetitionState,
                        "evaluate_journey_snapshot",
                        return_value=self._snapshot_for(action_id),
                    ),
                ):
                    result = comp.execute_agent_copilot_action(None, None)
                self.assertNotEqual(result.get("status"), "FAILED",
                                   f"unexpected FAILED for {action_id}: {result}")
                self.assertIn("observation", result)
                expected_params = (
                    {"exp_id": "baseline"}
                    if expected_action == "establish_baseline_anchor"
                    else {}
                )
                self.assertEqual(calls, [(expected_action, expected_params)])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_composite_baseline_establishment_is_named_not_substituted(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace()
        try:
            comp = CompetitionState(project_root=root)

            with (
                patch(
                    "harness.agent_bridge.AgentBridge",
                    side_effect=AssertionError("bridge must not be used for composite workflow"),
                ),
                patch.object(
                    CompetitionState,
                    "evaluate_journey_snapshot",
                    return_value=self._snapshot_for("ESTABLISH_VERIFIED_BASELINE"),
                ),
            ):
                result = comp.execute_agent_copilot_action(None, None)

            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["action_id"], "ESTABLISH_VERIFIED_BASELINE")
            self.assertIn("composite workflow", result["reason"])
            self.assertEqual(result["observation"]["status"], "FAILED")
            self.assertIn(
                "create_experiment -> run_full_validation -> establish_baseline_anchor",
                result["observation"]["summary"],
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestSmokeProbeFriction(unittest.TestCase):
    def _bridge(self, root: Path):
        return AgentBridge(root)

    def _call(self, bridge, act="run_smoke_test"):
        return SimpleNamespace(step_index=1, action=act, rationale="test")

    def test_smoke_test_rejects_traversal_config_path(self):
        root, _ = _make_workspace()
        try:
            bridge = self._bridge(root)
            session = bridge.get_active_or_create_session()
            with self.assertRaises(ValueError):
                bridge._handle_run_smoke_test(
                    self._call(bridge),
                    session,
                    None,
                    None,
                    "run_smoke_test",
                    {"config_path": "../../etc/passwd.yaml"},
                    0.0,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_probe_rejects_traversal_config_path(self):
        root, _ = _make_workspace()
        try:
            bridge = self._bridge(root)
            session = bridge.get_active_or_create_session()
            with self.assertRaises(ValueError):
                bridge._handle_run_probe(
                    self._call(bridge, "run_probe"),
                    session,
                    None,
                    None,
                    "run_probe",
                    {"config_path": "configs/../../../evil.yaml"},
                    0.0,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_smoke_runs_only_counts_passed_runs(self):
        root, _ = _make_workspace()
        try:
            bridge = self._bridge(root)
            session = bridge.get_active_or_create_session()
            session.compute_usage = {"smoke_runs": 0}

            with patch.object(
                bridge, "_run_subprocess_safe", return_value=(True, 0, "ok", 0.1)
            ):
                obs = bridge._handle_run_smoke_test(
                    self._call(bridge), session, None, None, "run_smoke_test", {}, 0.0
                )
            self.assertEqual(obs.status, "SUCCESS")
            self.assertEqual(session.compute_usage["smoke_runs"], 1)

            with patch.object(
                bridge, "_run_subprocess_safe", return_value=(False, 1, "boom", 0.1)
            ):
                obs = bridge._handle_run_smoke_test(
                    self._call(bridge), session, None, None, "run_smoke_test", {}, 0.0
                )
            self.assertEqual(obs.status, "FAILED")
            self.assertEqual(session.compute_usage["smoke_runs"], 1, "FAILED smoke must not consume budget")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestPhaseHeadingMatching(unittest.TestCase):
    def test_numbered_and_bold_headings_accepted(self):
        from harness.phase_workflow import _phase_heading_matches

        self.assertTrue(_phase_heading_matches("## Summary", "Summary"))
        self.assertTrue(_phase_heading_matches("## 1. Summary", "Summary"))
        self.assertTrue(_phase_heading_matches("### 2) Analysis - details", "Analysis"))
        self.assertTrue(_phase_heading_matches("**Summary:**", "Summary"))
        self.assertTrue(_phase_heading_matches("#### iii. Evidence", "Evidence"))
        self.assertFalse(_phase_heading_matches("## Data Leakage", "Summary"))
        self.assertFalse(_phase_heading_matches("", "Summary"))

    def test_record_accepts_numbered_sections(self):
        from harness.phase_workflow import PhaseWorkflow

        root, _ = _make_workspace()
        try:
            flow = PhaseWorkflow(root)
            response = (
                "# setup\n\n"
                "## 1. Summary\n\nSummary body.\n\n"
                "## 2) Evidence\n\nEvidence body.\n\n"
                "## 3. Decisions\n\nDecision body.\n\n"
                "**Next steps:**\n\nNext step body.\n"
            )
            flow.record("setup", response)
            self.assertTrue(flow.phase_status()["phases"][0]["completed"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestKaggleInspectKernelOutput(unittest.TestCase):
    def test_nonzero_returncode_reports_error(self):
        from harness.kaggle_inspect import kernel_output

        with (
            patch(
                "harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status",
                return_value={"kernel_slug": "user/kern", "status": "complete"},
            ),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=1, stderr="quota exceeded"),
            ),
        ):
            result = kernel_output(kernel_slug="user/kern")
        self.assertIn("error", result)
        self.assertIn("rc=1", result["error"])
        self.assertIn("quota exceeded", result["error"])

    def test_zero_returncode_has_no_error(self):
        from harness.kaggle_inspect import kernel_output

        with (
            patch(
                "harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status",
                return_value={"kernel_slug": "user/kern", "status": "complete"},
            ),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0, stderr=""),
            ),
        ):
            result = kernel_output(kernel_slug="user/kern")
        self.assertNotIn("error", result)
        self.assertEqual(result["kernel_slug"], "user/kern")


class TestStateSaveStamping(unittest.TestCase):
    def test_save_stamps_updated_at(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace()
        try:
            comp = CompetitionState(project_root=root)
            comp.state["mode"] = "competition"
            comp.save()
            self.assertIsNotNone(comp.state.get("updated_at"))
            persisted = json.loads(comp.state_file.read_text(encoding="utf-8"))
            self.assertEqual(persisted.get("updated_at"), comp.state["updated_at"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_save_raw_dead_code_removed(self):
        from harness.competition_state import CompetitionState

        self.assertFalse(hasattr(CompetitionState, "_save_raw"))


class TestSessionHistoryCaps(unittest.TestCase):
    def test_save_session_caps_actions_and_observations(self):
        root, _ = _make_workspace()
        try:
            bridge = AgentBridge(root)
            session = bridge.get_active_or_create_session()
            for i in range(MAX_SESSION_ACTIONS + 50):
                session.actions.append({"step_index": i, "action": "create_experiment"})
                session.observations.append(
                    {
                        "step_index": i,
                        "action": "create_experiment",
                        "status": "SUCCESS",
                        "summary": "ok",
                    }
                )

            bridge.save_session(session)

            payload = json.loads(
                (bridge.sessions_dir / f"{session.session_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertLessEqual(len(payload["actions"]), MAX_SESSION_ACTIONS)
            self.assertLessEqual(
                len(payload["observations"]), MAX_SESSION_OBSERVATIONS
            )
            self.assertEqual(payload["actions"][-1]["step_index"], MAX_SESSION_ACTIONS + 49)
            self.assertEqual(
                payload["observations"][-1]["step_index"], MAX_SESSION_ACTIONS + 49
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestAgentContextConfigFiltering(unittest.TestCase):
    def _comp(self, root):
        from harness.competition_state import CompetitionState

        return CompetitionState(project_root=root)

    def test_render_tolerates_unknown_persisted_keys(self):
        root, _ = _make_workspace()
        try:
            comp = self._comp(root)
            comp.state["agent_context_config"]["bogus_unknown_key"] = "x"
            with patch.object(
                type(comp),
                "evaluate_journey_snapshot",
                return_value={
                    "hud": {
                        "where_am_i": {
                            "phase_title": "t",
                            "mode": "competition",
                            "active_tracks": {},
                            "phase_question": "q?",
                        },
                        "what_is_current_best": {
                            "baseline": {"id": "exp000", "score": 0.5},
                            "cv_best": {"id": "exp001", "score": 0.6, "delta": 0.1},
                            "public_lb_best": {"label": "Kaggle", "sub_id": None, "score": None},
                            "private_frozen": {"max_candidates": 3, "is_frozen": False},
                        },
                        "what_is_blocking_me": [],
                        "what_should_i_do_next": {
                            "title": "t",
                            "detail": "d",
                            "agent_plan": ["step"],
                        },
                    }
                },
            ):
                block = comp.render_agent_context_block(None, None)
            self.assertIsInstance(block, str)
            self.assertGreater(len(block), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_update_workspace_settings_filters_unknown_keys(self):
        root, _ = _make_workspace()
        try:
            bridge = AgentBridge(root)
            session = bridge.get_active_or_create_session()
            call = SimpleNamespace(step_index=1, action="update_workspace_settings", rationale="t")

            obs = bridge._handle_update_workspace_settings(
                call,
                session,
                None,
                None,
                "update_workspace_settings",
                {"agent_context_config": {"inject_phase_hud": False, "evil_key": 1}},
                0.0,
            )
            self.assertEqual(obs.status, "SUCCESS")
            stored = bridge.comp_state.state["agent_context_config"]
            self.assertFalse(stored["inject_phase_hud"])
            self.assertNotIn("evil_key", stored)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestForceHelpText(unittest.TestCase):
    def test_force_help_does_not_claim_overwrite(self):
        help_text = build_parser().format_help()
        self.assertIn("NEVER overwritten", help_text)
        self.assertNotIn("Force overwrite existing experiment artifacts", help_text)


# ===========================================================================
# 6. F821 Inspect Experiment Handler
# ===========================================================================

class TestInspectExperimentHandler(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="f821_br_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def _dispatch(self, action: str, params: dict) -> object:
        journal = self.bridge._new_journal(auto_sync=False)
        call = types.SimpleNamespace(
            step_index=1, action=action, params=params
        )
        return self.bridge._dispatch_typed_action(
            call, types.SimpleNamespace(), journal, None
        )

    def test_inspect_experiment_succeeds_with_explicit_id(self) -> None:
        obs = self._dispatch("inspect_experiment", {"exp_id": "baseline"})
        self.assertEqual(obs.status, "SUCCESS", msg=obs.summary)
        self.assertIn("baseline", obs.summary)

    def test_inspect_experiment_succeeds_with_default_id(self) -> None:
        obs = self._dispatch("inspect_experiment", {})
        self.assertEqual(obs.status, "SUCCESS", msg=obs.summary)
        self.assertIn("Inspected", obs.summary)


# ===========================================================================
# 7. CLI & Bridge Usability Ergonomics
# ===========================================================================

class TestCliErgonomics(unittest.TestCase):
    def test_coerce_value(self):
        from tools.agent_bridge_cli import _coerce_value

        self.assertEqual(_coerce_value("true"), True)
        self.assertEqual(_coerce_value("False"), False)
        self.assertIsNone(_coerce_value("None"))
        self.assertIsNone(_coerce_value("null"))
        self.assertEqual(_coerce_value("123"), 123)
        self.assertEqual(_coerce_value("-45"), -45)
        self.assertEqual(_coerce_value("3.1415"), 3.1415)
        self.assertEqual(_coerce_value("exp001"), "exp001")
        self.assertEqual(_coerce_value('{"a": 1}'), {"a": 1})
        self.assertEqual(_coerce_value('[1, 2, 3]'), [1, 2, 3])

    def test_parse_dynamic_kv_args(self):
        from tools.agent_bridge_cli import _parse_dynamic_kv_args

        args = [
            "--exp-id", "exp001_baseline",
            "--config-path", "configs/exp001.yaml",
            "--num-folds", "5",
            "--subsample", "0.85",
            "--enable-gpu",
            "--custom-opt=adamw",
        ]
        params = _parse_dynamic_kv_args(args)
        self.assertEqual(params["exp_id"], "exp001_baseline")
        self.assertEqual(params["config_path"], "configs/exp001.yaml")
        self.assertEqual(params["num_folds"], 5)
        self.assertEqual(params["subsample"], 0.85)
        self.assertEqual(params["enable_gpu"], True)
        self.assertEqual(params["custom_opt"], "adamw")


class TestAgentBridgeErgonomicsUsability(unittest.TestCase):
    def test_get_next_experiment_num_and_collision_prevention(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            configs_dir = root / "configs"
            configs_dir.mkdir(parents=True)
            task_yaml = root / "configs" / "task.yaml"
            task_yaml.write_text(
                "task:\n  name: TestTask\n  id: test_task\n  primary_metric: roc_auc\n  data:\n    train: train.csv\n",
                encoding="utf-8"
            )
            (root / "train.csv").write_text("id,text,target\n1,a,0\n2,b,1\n", encoding="utf-8")

            bridge = AgentBridge(root, task_spec_path=task_yaml)
            self.assertEqual(bridge.get_next_experiment_num(), 1)

            (configs_dir / "exp001_baseline.yaml").write_text("dummy", encoding="utf-8")
            self.assertEqual(bridge.get_next_experiment_num(), 2)

            art_dir = root / ".ai-harness" / "artifacts" / "test_task" / "exp002_feat_v1"
            art_dir.mkdir(parents=True)
            self.assertEqual(bridge.get_next_experiment_num(), 3)

            sess = bridge.start_session(goal="Test next id")
            sess, obs = bridge.execute_action(
                action=TypedActionName.GET_NEXT_EXPERIMENT_ID.value,
                params={"branch": "model", "descriptor": "lgbm"},
                session_id=sess.session_id,
            )
            self.assertEqual(obs.status, "SUCCESS")
            self.assertEqual(obs.data["next_num"], 3)
            self.assertEqual(obs.data["suggested_exp_id"], "exp003_model_lgbm")

            sess, obs_collide = bridge.execute_action(
                action=TypedActionName.CREATE_EXPERIMENT.value,
                params={"exp_id": "exp002_feat_v1", "branch": "feat", "descriptor": "v1"},
                session_id=sess.session_id,
            )
            self.assertEqual(obs_collide.status, "SUCCESS")
            self.assertEqual(obs_collide.data["exp_id"], "exp003_feat_v1")
            self.assertEqual(obs_collide.data["auto_assigned_exp_id"], "exp003_feat_v1")
            self.assertIn("auto-assigned", obs_collide.summary)
            self.assertIn("immutable", obs_collide.summary)

    def test_agent_bridge_act_alias_and_main(self):
        from harness import agent_bridge

        self.assertTrue(callable(agent_bridge.main))
        self.assertIs(AgentBridge.act, AgentBridge.execute_action)


if __name__ == "__main__":
    unittest.main()
