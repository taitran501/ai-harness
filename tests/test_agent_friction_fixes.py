"""Regression tests for agent-friction fixes mined from Antigravity chat audits.

Each test targets a concrete error observed in real agent sessions:

- AttributeError: 'ValidationReport' object has no attribute 'std_val'
- AttributeError: 'UniversalRunner' object has no attribute 'expected_validation_ids'
- AttributeError: type object 'AgentBridge' has no attribute 'review_hypothesis'
- TypeError: 'method' object is not subscriptable  (ResearchMemory.entries[-1])
- RuntimeError: Kaggle output import failed: ... do not match this task/config/job: {'job_id': ...}
- TypeError: UploadConsentStore.grant() got an unexpected keyword argument 'confirmed_upload'
- TypeError: SubtaskSpec.__init__() got an unexpected keyword argument 'metric'
- create_experiment failing hard on immutable-artifact ID collisions
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge, TypedActionName
from harness.endgame_submission import SubmissionPreflightValidator, ValidationReport
from harness.research_memory import ResearchMemory
from harness.runner import UniversalRunner
from harness.task_spec import SubtaskSpec, UniversalTaskSpec
from harness.templates.kaggle_template import _is_critical_identity_mismatch
from tests.test_correctness_contracts import task

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _make_bridge_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="test_friction_"))
    shutil.copytree(SOURCE_ROOT / "configs", workspace / "configs")
    return workspace


class TestValidationReportStdVal(unittest.TestCase):
    """Agents use ValidationReport.std_val to detect prediction collapse."""

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
    """expected_validation_ids must exist right after construction, before
    split resolution (script adapters and standalone checks reference it)."""

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
        from types import SimpleNamespace

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
    """Agents call bridge.review_hypothesis(...) as if it were a method."""

    def setUp(self) -> None:
        self.ws = _make_bridge_workspace()
        self.addCleanup(shutil.rmtree, self.ws, True)
        self.bridge = AgentBridge(self.ws)

    def test_instance_alias_exists_for_every_typed_action(self) -> None:
        for action in TypedActionName:
            with self.subTest(action=action.value):
                self.assertTrue(callable(getattr(self.bridge, action.value)))

    def test_class_level_call_gets_actionable_error(self) -> None:
        with self.assertRaises(AttributeError) as ctx:
            AgentBridge.review_hypothesis  # type: ignore[attr-defined]
        msg = str(ctx.exception)
        self.assertIn("review_hypothesis", msg)
        self.assertIn("call it on an instance", msg)

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
    """memory.entries and memory.entries() must both work; entries support
    attribute access (entry.kind) as well as dict access (entry['kind'])."""

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
        entries = self.memory.entries  # property access (no call)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], entry["id"])
        # legacy callable form
        self.assertEqual(len(self.memory.entries()), 1)
        # subscripting (the original 'method not subscriptable' crash)
        self.assertEqual(self.memory.entries[-1]["id"], entry["id"])
        # attribute access on the record
        self.assertEqual(entries[0].id, entry["id"])
        self.assertEqual(entries[0].kind, "insight")
        self.assertEqual(entries[0].title, "Minority recall")
        # records are real dicts (json-serializable, .get works)
        json.dumps(entries)
        self.assertEqual(entries[0].get("scope"), self.memory.scope)


class TestKaggleJobIdMismatchPolicy(unittest.TestCase):
    """A remote kernel that does not stamp a job_id must not block import;
    a genuinely conflicting job_id must."""

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
    """grant() must accept confirmed_upload (and extra flags) without TypeError."""

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
        # persisted record carries the same fields
        persisted = json.loads(store.path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["consents"][-1]["fingerprint"], "fp123")
        self.assertTrue(store.is_granted("kaggle", "fp123"))


class TestSubtaskSpecMetricAlias(unittest.TestCase):
    """SubtaskSpec(metric=...) is a valid alias for primary_metric."""

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
    """A colliding exp_id must auto-assign the next free ID instead of
    failing — completed runs stay immutable, the agent moves on."""

    def setUp(self) -> None:
        self.ws = _make_bridge_workspace()
        self.addCleanup(shutil.rmtree, self.ws, True)
        self.bridge = AgentBridge(self.ws)

    def test_collision_auto_assigns_and_reports(self) -> None:
        # Simulate existing immutable artifacts for the requested ID.
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
        # The immutable artifacts were not touched.
        self.assertTrue((collision_dir / "artifact_manifest.json").exists())
        # The new config was written under the auto-assigned ID.
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


if __name__ == "__main__":
    unittest.main()
