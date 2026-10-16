"""Regression tests for PR #A — correctness / silent-wrong-result fixes.

Covered findings:
- P0: shadowed ``train_path``/``test_path`` properties in TaskSpec
      (truthiness checks were always true when no split was declared).
- P0: cloud templates omitted ``--job-id`` so the remote runner produced a
      fresh run_id and the manager identity gate failed 100% of managed
      dispatches (retry then blocked on the non-empty artifact dir).
- P1: external OOF ingestion evaluated predictions before row alignment
      (files sorted differently from the training split mislabeled rows).
- P1: custom metrics_evaluator silently swallowed official plugin failures.
- P1: probe/sample runs ignored ``validation.sample_limit`` on the ML path
      and ``--probe-samples`` on the LLM path.
- P1: official LB reads used a ``primary`` key that ``record_official_score``
      never writes (public LB best was always None/N-A).
- P1: baseline anchor gate never re-verified contract identity of the
      anchored run against the current task/scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from harness.artifacts import contract_identity, sha256
from harness.competition_state import official_primary_value
from harness.candidates import register_external_prediction
from harness.project import HarnessProject
from harness.runner import UniversalRunner
from harness.task_spec import UniversalTaskSpec, load_active_task_spec
from tests.test_correctness_contracts import task


def _write_task_yaml(root: Path, spec: UniversalTaskSpec, train_path: str = None) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    doc = {"task": spec.metadata}
    if train_path:
        doc["task"]["data"]["train_path"] = train_path
    (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


class TestTaskSpecPathProperties(unittest.TestCase):
    """P0: undeclared splits must be falsy, not default-directory truthy."""

    def test_undeclared_paths_are_none(self):
        spec = task()
        self.assertIsNone(spec.train_path)
        self.assertIsNone(spec.test_path)
        # The original bug: these checks were always true before the fix.
        self.assertFalse(spec.test_path)
        self.assertFalse(spec.train_path)

    def test_declared_paths_return_declared_values(self):
        spec = UniversalTaskSpec.from_dict(
            {
                "task": {
                    "name": "fixture",
                    "modality": "tabular",
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "data": {
                        "id_column": "id",
                        "seed": 13,
                        "train_path": "my/train.csv",
                        "test_path": "my/test/",
                    },
                    "inputs": [
                        {"name": "features", "type": "tabular", "source_columns": ["x"]}
                    ],
                    "subtasks": [
                        {
                            "id": "target",
                            "column": "target",
                            "task_type": "regression",
                            "primary_metric": "rmse",
                            "direction": "minimize",
                            "classes": [],
                        }
                    ],
                }
            }
        )
        self.assertEqual(spec.train_path, "my/train.csv")
        self.assertEqual(spec.test_path, "my/test/")


class TestCloudTemplatesBindJobId(unittest.TestCase):
    """P0: generated remote runners must bind the dispatched job id."""

    def test_kaggle_template_cmd_includes_job_id(self):
        from harness.templates.kaggle_template import KaggleKernelManager
        from tests.test_research_continuity import config, task_document

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            document = task_document()
            document["task"]["name"] = "Job Bind Task"
            (root / "configs/task.yaml").write_text(
                yaml.safe_dump(document), encoding="utf-8"
            )
            (root / "dataset/train.csv").write_text(
                "id,text,target,fold\n"
                + "".join(
                    f"{i},{'nice happy dog' if i % 2 == 0 else 'awful angry cat'},{i % 2},{(i // 2) % 2}\n"
                    for i in range(12)
                ),
                encoding="utf-8",
            )
            (root / "configs/exp.yaml").write_text(
                yaml.safe_dump(config("exp001_jobbind")), encoding="utf-8"
            )
            from harness.project import attach_project

            attach_project(root, root / "configs/task.yaml")

            with patch.object(
                KaggleKernelManager,
                "get_authenticated_username",
                return_value="fixture-user",
            ):
                package = KaggleKernelManager.setup_kernel_package(
                    root / "stage",
                    "configs/exp.yaml",
                    "exp001_jobbind",
                    "fixture-user/fixture-data",
                    project_root=root,
                    job_id="job_fixture_dispatch",
                    runtime_manifest={
                        "core": {"version": "0.2.0"},
                        "plugin": {"id": "default", "install": None},
                        "requirements": [],
                        "bundle_files": [],
                    },
                )
            source = (package / "run_kaggle.py").read_text(encoding="utf-8")
            self.assertIn('cmd += ["--job-id", _remote_job_id]', source)
            self.assertIn("_remote_job_id = 'job_fixture_dispatch'", source)

    def test_modal_template_cmd_includes_job_id(self):
        from harness.templates.modal_template import generate_modal_script
        from tests.test_research_continuity import config, task_document

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            document = task_document()
            document["task"]["name"] = "Modal Job Bind Task"
            (root / "configs/task.yaml").write_text(
                yaml.safe_dump(document), encoding="utf-8"
            )
            (root / "dataset/train.csv").write_text(
                "id,text,target,fold\n"
                + "".join(
                    f"{i},{'nice happy dog' if i % 2 == 0 else 'awful angry cat'},{i % 2},{(i // 2) % 2}\n"
                    for i in range(12)
                ),
                encoding="utf-8",
            )
            (root / "configs/exp.yaml").write_text(
                yaml.safe_dump(config("exp002_jobbind")), encoding="utf-8"
            )
            from harness.project import attach_project

            attach_project(root, root / "configs/task.yaml")

            source = generate_modal_script(
                "configs/exp.yaml",
                exp_id="exp002_jobbind",
                volume_name="fixture-volume",
                project_root=root,
                task_id="modal-jobbind",
                job_id="modal_fixture_dispatch",
            )
        self.assertIn('cmd += ["--job-id", _remote_job_id]', source)
        self.assertIn("_remote_job_id = 'modal_fixture_dispatch'", source)


class _ExternalIngestFixture:
    """Shared fixture for external OOF alignment tests."""

    def build(self) -> tuple[Path, int]:
        tmp = tempfile.TemporaryDirectory()
        self._tmp = tmp
        root = Path(tmp.name)
        (root / "configs").mkdir()
        (root / "dataset").mkdir()
        (root / "artifacts").mkdir()

        spec = task()
        doc = {"task": spec.metadata}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

        n = 20
        self.n = n
        train_df = pd.DataFrame(
            {
                "id": range(n),
                "group": range(n),
                "x": range(n),
                "target": [float(x * 2.0) for x in range(n)],
            }
        )
        train_df.to_csv(root / "dataset/train.csv", index=False)
        return root, n


class TestExternalIngestRowAlignment(_ExternalIngestFixture, unittest.TestCase):
    """P1: evaluation must consume the aligned frame, not the raw file order."""

    def tearDown(self):
        self._tmp.cleanup()

    def _register(self, root: Path, oof: pd.DataFrame, name: str) -> dict:
        path = root / f"{name}.csv"
        oof.to_csv(path, index=False)
        return register_external_prediction(
            project=root,
            name=name,
            oof_path=path,
            author="TestGM",
            source="kaggle_kernel",
            notes="alignment regression",
        )

    def test_shuffled_oof_scores_identically_to_ordered_oof(self):
        root, n = self.build()
        bias = 0.75
        ordered = pd.DataFrame(
            {
                "id": range(n),
                "pred_target": [float(x * 2.0 + bias) for x in range(n)],
            }
        )
        shuffled = ordered.sample(frac=1.0, random_state=7).reset_index(drop=True)
        # Sanity: the two files really differ in row order.
        self.assertFalse(list(shuffled["id"]) == list(ordered["id"]))

        res_ordered = self._register(root, ordered, "ext_ordered")
        res_shuffled = self._register(root, shuffled, "ext_shuffled")

        # Constant offset bias -> RMSE == |bias| regardless of row order.
        self.assertAlmostEqual(res_ordered["primary_score"], abs(bias), places=6)
        self.assertAlmostEqual(res_shuffled["primary_score"], abs(bias), places=6)
        self.assertAlmostEqual(
            res_shuffled["primary_score"], res_ordered["primary_score"], places=10
        )

    def test_duplicate_oof_ids_are_rejected(self):
        root, n = self.build()
        dup = pd.DataFrame(
            {
                "id": [0, 0, *range(2, n)],
                "pred_target": [float(x) for x in range(n)],
            }
        )
        with self.assertRaises(ValueError):
            self._register(root, dup, "ext_dup")


class TestRunnerEvaluationFailuresSurface(unittest.TestCase):
    """P1: a failing official scorer behind a custom hook must be recorded."""

    def _bare_runner(self, root: Path, cfg: dict, hook: str):
        runner = UniversalRunner.__new__(UniversalRunner)
        runner.cfg = cfg
        runner.config_path = str(root / "configs/exp.yaml")
        runner.args = argparse.Namespace(
            debug=False, probe_samples=None, fold=None, job_id=None
        )
        runner.project_root = str(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        runner.task_spec = task()
        runner.plugin_context = None

        class _FailingPlugin:
            def evaluate_predictions(self, context, truth, predictions):
                raise RuntimeError("official scorer exploded")

        runner.task_plugin = _FailingPlugin()
        return runner

    def test_failure_is_recorded_and_custom_result_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "custom_scorer_pr_a.py").write_text(
                textwrap.dedent(
                    """
                    def custom_eval(truth, predictions):
                        return {"rmse": 0.123}
                    """
                ).strip(),
                encoding="utf-8",
            )
            runner = self._bare_runner(
                root, {"paths": {"metrics_evaluator": "custom_scorer_pr_a:custom_eval"}},
                "custom_scorer_pr_a:custom_eval",
            )
            truth = pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 1.0, 2.0]})
            result = runner._evaluate(truth, {"target": [0.1, 0.9, 1.9]})
            self.assertAlmostEqual(result["primary_score"], 0.123)
            self.assertIn("_evaluation_warnings", result)
            self.assertIn(
                "official scorer exploded", result["_evaluation_warnings"][0]
            )

    def test_clean_run_has_no_warning_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "clean_scorer_pr_a.py").write_text(
                textwrap.dedent(
                    """
                    def clean_eval(truth, predictions):
                        return {"rmse": 0.456}
                    """
                ).strip(),
                encoding="utf-8",
            )
            runner = self._bare_runner(
                root, {"paths": {"metrics_evaluator": "clean_scorer_pr_a:clean_eval"}},
                "clean_scorer_pr_a:clean_eval",
            )

            class _WorkingPlugin:
                def evaluate_predictions(self, context, truth, predictions):
                    return {"rmse": 0.999}

            runner.task_plugin = _WorkingPlugin()
            truth = pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 1.0, 2.0]})
            result = runner._evaluate(truth, {"target": [0.1, 0.9, 1.9]})
            self.assertNotIn("_evaluation_warnings", result)
            # Custom result overrides the official value.
            self.assertAlmostEqual(result["primary_score"], 0.456)


class TestProbeSampleLimit(unittest.TestCase):
    """P1: probe subsampling must respect config and CLI consistently."""

    @staticmethod
    def _bare_runner(**kwargs) -> UniversalRunner:
        runner = UniversalRunner.__new__(UniversalRunner)
        runner.cfg = kwargs.pop("cfg", {})
        runner.args = argparse.Namespace(
            debug=kwargs.pop("debug", False),
            probe_samples=kwargs.pop("probe_samples", None),
            fold=kwargs.pop("fold", None),
            job_id=None,
        )
        return runner

    def test_config_sample_limit_applies_without_cli_flags(self):
        runner = self._bare_runner(cfg={"validation": {"sample_limit": 128}})
        self.assertEqual(runner._probe_sample_limit(), 128)

    def test_cli_probe_samples_override_config(self):
        runner = self._bare_runner(
            cfg={"validation": {"sample_limit": 128}}, probe_samples=32
        )
        self.assertEqual(runner._probe_sample_limit(), 32)

    def test_debug_defaults_to_50(self):
        runner = self._bare_runner(debug=True)
        self.assertEqual(runner._probe_sample_limit(), 50)

    def test_debug_wins_over_config_sample_limit(self):
        runner = self._bare_runner(debug=True, cfg={"validation": {"sample_limit": 128}})
        self.assertEqual(runner._probe_sample_limit(), 50)

    def test_cli_wins_over_debug(self):
        runner = self._bare_runner(debug=True, probe_samples=25)
        self.assertEqual(runner._probe_sample_limit(), 25)

    def test_full_run_has_no_limit(self):
        runner = self._bare_runner()
        self.assertIsNone(runner._probe_sample_limit())

    def test_single_fold_without_flags_is_not_subsampled(self):
        runner = self._bare_runner(fold=0)
        self.assertIsNone(runner._probe_sample_limit())


class TestOfficialScoreKeyContract(unittest.TestCase):
    """P1: public-LB readers must read what record_official_score writes."""

    def test_helper_probes_all_known_keys(self):
        class _Entry:
            official_scores = {"rmse": 0.5, "split_type": "public"}

        self.assertEqual(official_primary_value(_Entry(), "rmse"), 0.5)

        class _LegacyEntry:
            official_scores = {"primary": 0.75}

        self.assertEqual(official_primary_value(_LegacyEntry(), "rmse"), 0.75)

        class _ScoreEntry:
            official_scores = {"score": 1.25}

        self.assertEqual(official_primary_value(_ScoreEntry(), "rmse"), 1.25)

        class _MissingEntry:
            official_scores = {"split_type": "public"}

        self.assertIsNone(official_primary_value(_MissingEntry(), "rmse"))

    def test_record_then_read_roundtrip(self):
        from harness.submission_manager import SubmissionEntry, SubmissionManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task_yaml(root, task(), train_path="dataset/train.csv")
            (root / "dataset").mkdir(exist_ok=True)
            manager = SubmissionManager(
                artifacts_dir=root / "artifacts",
                project_root=root,
                task_config_path=root / "configs/task.yaml",
            )
            entry = SubmissionEntry(
                id="sub_001",
                exp_id="exp000",
                created_at="2026-01-01T00:00:00Z",
                status="SUBMITTED_PENDING",
                local_cv_scores={"rmse": 0.62},
            )
            manager.submissions[entry.id] = entry
            manager.record_official_score(entry.id, primary_score=0.5)

            # record_official_score writes under the spec's metric name...
            self.assertEqual(entry.official_scores.get("rmse"), 0.5)
            # ...and the journey-reader helper must be able to read it back.
            self.assertEqual(official_primary_value(entry, "rmse"), 0.5)


class TestBaselineAnchorIdentityGate(unittest.TestCase):
    """P1: the locked baseline must still match the current task/scorer."""

    def _sealed_run_artifacts(self, root: Path, exp_id: str) -> Path:
        from harness.plugins import TaskPluginContext, load_task_plugin

        project = HarnessProject.discover(root)
        spec = load_active_task_spec(project.task_config_path)
        plugin = load_task_plugin(
            None, TaskPluginContext(project.root, project.task_config_path, spec, "default")
        )
        exp_dir = project.artifacts_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"experiment": {"id": exp_id}, "model": {"name": "baseline"}}
        (exp_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        metrics = {
            "schema_version": 2,
            "experiment_id": exp_id,
            "selection_variant": "all",
            "primary_metric": "rmse",
            "primary_score": 0.5,
            "direction": "minimize",
        }
        (exp_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        pd.DataFrame({"id": [1, 2], "pred_target": [0.1, 0.2]}).to_csv(
            exp_dir / "oof_predictions.csv", index=False
        )
        pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]}).to_csv(
            exp_dir / "evaluation_truth.csv", index=False
        )
        manifest = {
            "schema_version": 2,
            "run_id": f"run_{exp_id}",
            "exp_id": exp_id,
            "status": "COMPLETED",
            "run_kind": "full",
            **contract_identity(spec, plugin),
            "data_sha256": "data_hash",
            "split_sha256": "split_hash",
            "resolved_config_sha256": sha256(exp_dir / "run_config.yaml"),
            "files": {
                name: sha256(exp_dir / name)
                for name in (
                    "run_config.yaml",
                    "metrics.json",
                    "oof_predictions.csv",
                    "evaluation_truth.csv",
                )
            },
            "prediction_splits": {"oof_predictions.csv": "dev"},
        }
        (exp_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return exp_dir

    def test_stale_contract_blocks_baseline_gate(self):
        from harness.phase_workflow import PhaseWorkflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            spec = task()
            _write_task_yaml(root, spec, train_path="dataset/train.csv")
            (root / "dataset/train.csv").write_text("id,x,target\n1,1.0,0.0\n", encoding="utf-8")
            from harness.project import attach_project

            attach_project(root, root / "configs/task.yaml")
            artifacts_dir = HarnessProject.discover(root).artifacts_dir
            exp_dir = self._sealed_run_artifacts(root, "exp000_anchor")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "competition_state.json").write_text(
                json.dumps(
                    {"gates": {"baseline_locked": True, "baseline_exp_id": "exp000_anchor"}}
                ),
                encoding="utf-8",
            )

            workflow = PhaseWorkflow(root)
            # A sealed run matching the current contract passes the gate...
            state = workflow._phase_gate_state()
            self.assertIsNone(state["baseline"])

            # ...but a stale task identity (task.yaml changed after the run)
            # must now block the anchor.
            manifest = json.loads((exp_dir / "artifact_manifest.json").read_text())
            manifest["task_sha256"] = "deadbeef" * 8
            (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest))
            state = workflow._phase_gate_state()
            self.assertIsNotNone(state["baseline"])


if __name__ == "__main__":
    unittest.main()
