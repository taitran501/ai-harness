"""Integration tests for evidence sealing, deduplication, and tamper detection.

Consolidates:
- Artifact deduplication (mirror_csv, write_csv_dedup, canonical_truth, external candidate registration, runner deduplication)
- Data bundle metadata skipping (.git vs protected files)
- Recursive manifest sealing and tamper detection
- Threshold recommendation immutable evidence verification
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from harness import import_local
from harness.agent_bridge import AgentBridge
from harness.artifacts import (
    canonical_truth_path,
    ensure_canonical_truth,
    mirror_csv,
    seal_file_map,
    sha256,
    validate_run,
    write_csv_dedup,
)
from harness.candidates import register_external_prediction
from harness.data_bundle import DataBundleError, stage_declared_data
from harness.endgame_submission import ThresholdOptimizer
from harness.kaggle_runtime import export_run
from harness.project import HarnessProject, attach_project
from tests.evidence_fixtures import seal_fixture, seal_scope_fixture
from tests.fixtures import make_workspace, task


# ---------------------------------------------------------------------------
# CSV Deduplication Helpers & Tests (from test_artifact_dedup.py)
# ---------------------------------------------------------------------------


def _df(seed: int = 0) -> pd.DataFrame:
    return pd.DataFrame({"id": range(10), "v": [float(i + seed) for i in range(10)]})


def _norm(path: Path) -> str:
    """Read a CSV with newlines normalized (Windows CRLF vs Unix LF)."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False).replace("\r\n", "\n")


class TestMirrorCsv(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test_dedup_"))

    def test_creates_hardlink_mirror(self) -> None:
        src = self.tmp / "src.csv"
        tgt = self.tmp / "tgt.csv"
        _df().to_csv(src, index=False)
        ok = mirror_csv(src, tgt)
        self.assertTrue(ok)
        self.assertTrue(tgt.is_file())
        self.assertEqual(src.read_bytes(), tgt.read_bytes())
        self.assertEqual(src.stat().st_ino, tgt.stat().st_ino)

    def test_recreates_stale_target(self) -> None:
        src = self.tmp / "src.csv"
        tgt = self.tmp / "tgt.csv"
        _df(1).to_csv(src, index=False)
        _df(2).to_csv(tgt, index=False)  # stale content
        ok = mirror_csv(src, tgt)
        self.assertTrue(ok)
        self.assertEqual(src.read_bytes(), tgt.read_bytes())

    def test_returns_false_when_link_unsupported(self) -> None:
        src = self.tmp / "src.csv"
        tgt = self.tmp / "tgt.csv"
        _df().to_csv(src, index=False)
        with mock.patch("os.link", side_effect=OSError("cross-device")):
            self.assertFalse(mirror_csv(src, tgt))
        self.assertFalse(tgt.exists())

    def test_removes_existing_symlink_target(self) -> None:
        src = self.tmp / "src.csv"
        tgt = self.tmp / "tgt.csv"
        _df().to_csv(src, index=False)
        try:
            tgt.symlink_to(src)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation not permitted on this system")
        ok = mirror_csv(src, tgt)
        self.assertTrue(ok)
        self.assertFalse(tgt.is_symlink())
        self.assertEqual(src.read_bytes(), tgt.read_bytes())


class TestWriteCsvDedup(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test_dedup_"))

    def test_prefers_mirror(self) -> None:
        src = self.tmp / "src.csv"
        tgt = self.tmp / "tgt.csv"
        frame = _df()
        frame.to_csv(src, index=False)
        write_csv_dedup(frame, tgt, mirror_from=src)
        self.assertEqual(src.stat().st_ino, tgt.stat().st_ino)
        self.assertEqual(_csv(frame), _norm(tgt))

    def test_fallback_unlinks_stale_hardlink_before_write(self) -> None:
        """The fallback must never write in place over a shared inode."""
        src = self.tmp / "src.csv"
        tgt = self.tmp / "tgt.csv"
        original = _df(1)
        original.to_csv(src, index=False)
        mirror_csv(src, tgt)  # tgt shares inode with src
        new_frame = _df(2)
        with mock.patch("harness.artifacts.mirror_csv", return_value=False):
            write_csv_dedup(new_frame, tgt, mirror_from=src)
        # tgt got the new content; src (the linked original) must be untouched
        self.assertEqual(_csv(new_frame), _norm(tgt))
        self.assertEqual(_csv(original), _norm(src))
        self.assertNotEqual(src.stat().st_ino, tgt.stat().st_ino)

    def test_fallback_plain_write_without_mirror(self) -> None:
        tgt = self.tmp / "plain.csv"
        frame = _df(3)
        write_csv_dedup(frame, tgt)
        self.assertEqual(_csv(frame), _norm(tgt))


class TestCanonicalTruth(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test_dedup_"))

    def test_path_is_content_addressed_and_stable(self) -> None:
        a = _df()
        p1 = canonical_truth_path(self.tmp, a)
        p2 = canonical_truth_path(self.tmp, _df())
        p3 = canonical_truth_path(self.tmp, _df(5))
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertIn("canonical_evaluation_truth_", p1.name)

    def test_ensure_creates_and_is_idempotent(self) -> None:
        frame = _df()
        p1 = ensure_canonical_truth(self.tmp, frame)
        self.assertTrue(p1.is_file())
        first_hash = sha256(p1)
        p2 = ensure_canonical_truth(self.tmp, frame)
        self.assertEqual(p1, p2)
        self.assertEqual(sha256(p2), first_hash)
        # no temp litter
        leftovers = [p for p in self.tmp.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_ensure_different_frames_produce_distinct_masters(self) -> None:
        p1 = ensure_canonical_truth(self.tmp, _df())
        p2 = ensure_canonical_truth(self.tmp, _df(7))
        self.assertNotEqual(p1, p2)
        self.assertTrue(p1.is_file() and p2.is_file())


class TestRunnerDedupEndToEnd(unittest.TestCase):
    """Full runner run: predictions/truth are hardlink mirrors and validate."""

    def setUp(self) -> None:
        from tests.e2e.test_pipeline_e2e import TestEndToEndPipelineIntegration

        # Reuse the e2e fixture (synthetic clinical-risk dataset + bootstrap)
        self._fixture = TestEndToEndPipelineIntegration()
        self._fixture.setUp()
        self.workspace_dir = self._fixture.workspace_dir
        self.dataset_dir = self._fixture.dataset_dir
        self.docs_dir = self._fixture.docs_dir
        self.configs_dir = self._fixture.configs_dir

        from tools.bootstrap_task import bootstrap_competition_task

        report = bootstrap_competition_task(
            dataset_dir=self.dataset_dir,
            context_file=self.docs_dir / "TASK.md",
            output_config=self.configs_dir / "task.yaml",
            project_root=self.workspace_dir,
        )
        self.assertEqual(report["status"], "SUCCESS")

    def tearDown(self) -> None:
        self._fixture.tearDown()

    def test_full_run_dedups_and_still_validates(self) -> None:
        from harness.runner import UniversalRunner

        baseline_cfg = self.configs_dir / "exp000_baseline.yaml"
        task_cfg = self.configs_dir / "task.yaml"
        args = Namespace(
            project_root=str(self.workspace_dir),
            task_config=str(task_cfg),
            job_id="dedup_probe_001",
            config=str(baseline_cfg),
            base_config=None,
            fold=0,
            debug=True,
            force=False,
            probe_samples=20,
            backend="local",
            skip_leaderboard=False,
        )
        with open(baseline_cfg, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)
        runner = UniversalRunner(cfg=cfg_data, config_path=baseline_cfg, args=args)
        result = runner.run()
        self.assertIsNotNone(result)

        exp_dir = runner.exp_dir
        oof = exp_dir / "oof_predictions.csv"
        pred = exp_dir / "predictions.csv"
        truth = exp_dir / "evaluation_truth.csv"
        for f in (oof, pred, truth):
            self.assertTrue(f.is_file(), f"{f.name} missing")
        self.assertEqual(oof.read_bytes(), pred.read_bytes())

        canonicals = [
            p for p in runner.artifacts_dir.iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(canonicals), 1)
        self.assertEqual(truth.read_bytes(), canonicals[0].read_bytes())

        # immutable-evidence contract: manifest hashes match the (linked) files
        manifest = json.loads((exp_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "COMPLETED")
        self.assertEqual(manifest["files"]["oof_predictions.csv"], sha256(oof))
        self.assertEqual(manifest["files"]["predictions.csv"], sha256(pred))
        self.assertEqual(manifest["files"]["oof_predictions.csv"], manifest["files"]["predictions.csv"])
        self.assertEqual(manifest["files"]["evaluation_truth.csv"], sha256(canonicals[0]))

    def test_second_run_shares_same_canonical_truth(self) -> None:
        from harness.runner import UniversalRunner

        def run_exp(cfg_name: str, job_id: str):
            cfg_path = self.configs_dir / cfg_name
            args = Namespace(
                project_root=str(self.workspace_dir),
                task_config=str(self.configs_dir / "task.yaml"),
                job_id=job_id,
                config=str(cfg_path),
                base_config=None,
                fold=0,
                debug=True,
                force=False,
                probe_samples=20,
                backend="local",
                skip_leaderboard=False,
            )
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f)
            runner = UniversalRunner(cfg=cfg_data, config_path=cfg_path, args=args)
            runner.run()
            return runner.exp_dir, runner.artifacts_dir

        exp_a, artifacts = run_exp("exp000_baseline.yaml", "dedup_a")
        # second run needs a fresh experiment id
        import shutil as _sh

        cfg2 = self.configs_dir / "exp002_dedup_copy.yaml"
        _sh.copy2(self.configs_dir / "exp000_baseline.yaml", cfg2)
        cfg2_text = cfg2.read_text(encoding="utf-8").replace(
            "exp000_baseline", "exp002_dedup_copy"
        )
        cfg2.write_text(cfg2_text, encoding="utf-8")
        exp_b, _ = run_exp("exp002_dedup_copy.yaml", "dedup_b")

        truth_a = exp_a / "evaluation_truth.csv"
        truth_b = exp_b / "evaluation_truth.csv"
        self.assertEqual(truth_a.read_bytes(), truth_b.read_bytes())
        masters = [
            p for p in artifacts.iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(masters), 1)


class TestRegisterExternalDedup(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "artifacts").mkdir()

        spec = task()
        doc = {"task": spec.metadata}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        (self.root / "configs" / "task.yaml").write_text(
            yaml.safe_dump(doc), encoding="utf-8"
        )
        self.n = 20
        self.train_df = pd.DataFrame(
            {
                "id": range(self.n),
                "group": range(self.n),
                "x": range(self.n),
                "category": ["a", "b"] * (self.n // 2),
                "target": [float(x * 2.0) for x in range(self.n)],
            }
        )
        self.train_df.to_csv(self.root / "dataset" / "train.csv", index=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _ext_oof(self, bias: float) -> Path:
        oof = pd.DataFrame(
            {
                "id": range(self.n),
                "fold": [i % 2 for i in range(self.n)],
                "pred_target": [float(x * 2.0 + bias) for x in range(self.n)],
            }
        )
        path = self.root / f"ext_oof_{abs(bias)}.csv"
        oof.to_csv(path, index=False)
        return path

    def test_register_creates_linked_artifacts(self) -> None:
        res = register_external_prediction(
            project=self.root,
            name="ext_gm_01",
            oof_path=self._ext_oof(0.5),
            author="GM",
            source="kaggle_kernel",
        )
        self.assertEqual(res["rows"], self.n)
        exp_dir = self.root / "artifacts" / "ext_gm_01"
        self.assertTrue((exp_dir / "artifact_manifest.json").is_file())
        oof = exp_dir / "oof_predictions.csv"
        pred = exp_dir / "predictions.csv"
        self.assertEqual(oof.read_bytes(), pred.read_bytes())
        truth = exp_dir / "evaluation_truth.csv"
        masters = [
            p for p in (self.root / "artifacts").iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(masters), 1)
        self.assertEqual(truth.stat().st_ino, masters[0].stat().st_ino)

    def test_reregister_same_name_rejected(self) -> None:
        register_external_prediction(
            project=self.root, name="ext_gm_02", oof_path=self._ext_oof(0.1),
            author="GM", source="kaggle_kernel",
        )
        with self.assertRaises(ValueError):
            register_external_prediction(
                project=self.root, name="ext_gm_02", oof_path=self._ext_oof(0.9),
                author="GM", source="kaggle_kernel",
            )
        # the original run is untouched
        exp_dir = self.root / "artifacts" / "ext_gm_02"
        oof = exp_dir / "oof_predictions.csv"
        pred = exp_dir / "predictions.csv"
        self.assertEqual(oof.read_bytes(), pred.read_bytes())
        masters = [
            p for p in (self.root / "artifacts").iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(masters), 1)
        self.assertEqual((exp_dir / "evaluation_truth.csv").read_bytes(), masters[0].read_bytes())


# ---------------------------------------------------------------------------
# Data Bundle Metadata Skip Tests (from test_data_bundle_protected_paths.py)
# ---------------------------------------------------------------------------


class DataBundleMetadataSkipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-databundle-git-")
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "dataset" / "train.csv").write_text(
            "id,text\n0,nice dog\n1,awful cat\n", encoding="utf-8"
        )
        (self.root / "configs" / "task.yaml").write_text(
            yaml.safe_dump(
                {
                    "task": {
                        "name": "bundle-git",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "data_config": {"train_path": "dataset"},
                    }
                }
            ),
            encoding="utf-8",
        )
        attach_project(self.root, self.root / "configs/task.yaml")
        self.project = HarnessProject.discover(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _add_git_metadata(self):
        git_dir = self.root / "dataset" / ".git"
        (git_dir / "objects").mkdir(parents=True)
        (git_dir / "config").write_text(
            "[core]\n\trepositoryformatversion = 0\n", encoding="utf-8"
        )
        (git_dir / "objects" / "pack").mkdir()
        (git_dir / "objects" / "pack" / "pack-abc123.idx").write_bytes(b"\x00")

    def test_git_metadata_inside_declared_dataset_is_skipped_not_fatal(self):
        self._add_git_metadata()
        with tempfile.TemporaryDirectory(prefix="aih-bundle-dest-") as dest:
            result = stage_declared_data(self.project, dest)
            manifest = result["manifest"]
        self.assertEqual(result["file_count"], 1)
        self.assertEqual(manifest["files"], ["dataset/train.csv"])
        self.assertFalse(any(".git" in f for f in manifest["files"]))

    def test_explicit_git_declaration_still_refused(self):
        self._add_git_metadata()
        document = yaml.safe_load((self.root / "configs" / "task.yaml").read_text(encoding="utf-8"))
        document["task"]["data_config"]["train_path"] = "dataset/.git"
        (self.root / "configs" / "task.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        with self.assertRaisesRegex(DataBundleError, "protected path"):
            stage_declared_data(self.project, self.root / "bundle")

    def test_protected_file_inside_dataset_still_refused(self):
        (self.root / "dataset" / ".env").write_text("token=secret\n", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="aih-bundle-dest-") as dest:
            with self.assertRaisesRegex(DataBundleError, "protected task data file"):
                stage_declared_data(self.project, dest)

    def test_metadata_only_directory_fails_with_clear_error(self):
        (self.root / "dataset" / "train.csv").unlink()
        self._add_git_metadata()
        with tempfile.TemporaryDirectory(prefix="aih-bundle-dest-") as dest:
            with self.assertRaisesRegex(DataBundleError, "no uploadable files"):
                stage_declared_data(self.project, dest)


# ---------------------------------------------------------------------------
# Recursive Sealing Tests (from test_recursive_sealing.py)
# ---------------------------------------------------------------------------


def _make_run(root: Path, name: str, nested: bool = True) -> Path:
    run = root / name
    (run / "stages").mkdir(parents=True)
    (run / "run_config.yaml").write_text("model: {name: fixture}\n", encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({"primary_score": 0.5, "utility": -0.5, "direction": "minimize", "experiment_id": name}),
        encoding="utf-8",
    )
    (run / "oof_predictions.csv").write_text("id,pred_y\n0,0.5\n", encoding="utf-8")
    (run / "evaluation_truth.csv").write_text("id,y\n0,0.0\n", encoding="utf-8")
    if nested:
        (run / "stages" / "fold0").mkdir(parents=True)
        (run / "stages" / "train.log").write_text("stage train ok\n", encoding="utf-8")
        (run / "stages" / "fold0" / "stdout.log").write_text(
            "fold output\n", encoding="utf-8"
        )
    return run


def _full_manifest(run: Path, exp_id: str) -> dict:
    return {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": "r1",
        "exp_id": exp_id,
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "validation_scope": seal_scope_fixture(run, SimpleNamespace(id_column="id")),
        "files": seal_file_map(run),
    }


class RecursiveSealingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_seal_file_map_is_recursive_with_relpath_keys(self):
        run = _make_run(self.tmp_path, "exp_rec")
        files = seal_file_map(run)
        self.assertIn("metrics.json", files)  # top-level keeps bare name
        self.assertIn("stages/train.log", files)  # nested sealed with POSIX relpath
        self.assertIn("stages/fold0/stdout.log", files)  # deeper nesting works
        self.assertNotIn("artifact_manifest.json", files)  # manifest never seals itself
        for name, value in files.items():
            self.assertEqual(value, sha256(run / name))

    def test_validate_run_accepts_recursive_manifest(self):
        run = _make_run(self.tmp_path, "exp_val")
        (run / "artifact_manifest.json").write_text(
            json.dumps(_full_manifest(run, "exp_val")), encoding="utf-8"
        )
        validate_run(run)  # nested sealed files verify

    def test_validate_run_detects_nested_tamper(self):
        """The exact reviewer blind spot: stages/*.log could previously be
        rewritten freely. Now the sealed hash catches it."""
        run = _make_run(self.tmp_path, "exp_tamper")
        (run / "artifact_manifest.json").write_text(
            json.dumps(_full_manifest(run, "exp_tamper")), encoding="utf-8"
        )
        validate_run(run)

        (run / "stages" / "train.log").write_text("rewritten by an actor\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            validate_run(run)
        self.assertIn("stages/train.log", str(ctx.exception))

    def test_validate_run_rejects_traversal_and_absolute_keys(self):
        run = _make_run(self.tmp_path, "exp_trav", nested=False)
        manifest = _full_manifest(run, "exp_trav")
        for evil in ("../outside.log", "..\\outside.log", "stages/../../escape.log"):
            manifest["files"][evil] = sha256(run / "metrics.json")
            (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_run(run)
            del manifest["files"][evil]

    def test_old_top_level_manifest_still_validates(self):
        """Backward compatibility: pre-recursive manifests (bare names only)."""
        run = _make_run(self.tmp_path, "exp_old", nested=False)
        manifest = {
            "schema_version": 2,
            "status": "COMPLETED",
            "run_kind": "full",
            "run_id": "r1",
            "exp_id": "exp_old",
            "task_sha256": "a" * 64,
            "scorer_sha256": "b" * 64,
            "data_sha256": "c" * 64,
            "split_sha256": "d" * 64,
            "resolved_config_sha256": sha256(run / "run_config.yaml"),
        "validation_scope": seal_scope_fixture(run, SimpleNamespace(id_column="id")),
            "files": {
                p.name: sha256(p) for p in run.iterdir() if p.is_file()
            },
        }
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        validate_run(run)


class ImportContentVerifiedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_import_content_verified_accepts_nested(self):
        run = _make_run(self.tmp_path, "exp_imp")
        files = seal_file_map(run)
        manifest = {"files": files}
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIsNone(import_local._content_verified(run, manifest))

    def test_import_content_verified_rejects_traversal(self):
        run = _make_run(self.tmp_path, "exp_imp2", nested=False)
        manifest = {"files": {"../evil.log": "deadbeef"}}
        self.assertEqual(
            import_local._content_verified(run, manifest),
            "Manifest file name unsafe: ../evil.log",
        )


class ExportNestedFilesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_run_materializes_nested_files(self):
        run = _make_run(self.tmp_path, "exp_exp")
        (run / "artifact_manifest.json").write_text(
            json.dumps(_full_manifest(run, "exp_exp")), encoding="utf-8"
        )

        out = self.tmp_path / "kaggle_output"
        export_run(run, out, {"exp_id": "exp_exp"})
        self.assertTrue((out / "stages" / "train.log").is_file())
        self.assertTrue((out / "stages" / "fold0" / "stdout.log").is_file())
        validate_run(out)


# ---------------------------------------------------------------------------
# Threshold Evidence Tests (from test_threshold_evidence.py)
# ---------------------------------------------------------------------------


class ThresholdEvidenceTests(unittest.TestCase):
    def make_bridge(self, *, alias=False, strings=False, reverse=False, variants=False, regression=False, probabilities=True):
        spec = task() if regression else task("binary_classification", "accuracy", "maximize")
        doc = {"task": copy.deepcopy(spec.metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        column = "label" if alias else "target"
        doc["task"]["subtasks"][0]["column"] = column
        labels = ["no", "yes"] if strings else [0, 1]
        if strings:
            doc["task"]["subtasks"][0]["classes"] = labels
        truth = pd.DataFrame({"id": range(4), "group": range(4), "x": range(4),
                              "category": ["a", "b"] * 2, column: [labels[0]] * 2 + [labels[1]] * 2})
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)
        bridge = AgentBridge(root)
        values = [.1, .2, .8, .9]
        oof = pd.DataFrame({"id": truth.id, "pred_target": truth[column],
                            "proba_target": [json.dumps([v, 1 - v] if reverse else [1 - v, v]) for v in values],
                            "classes_target": json.dumps(labels[::-1] if reverse else labels)})
        if not probabilities or regression:
            oof = oof.drop(columns=["proba_target", "classes_target"])
        if variants:
            folder = bridge.artifacts_dir / "A"
            folder.mkdir(parents=True)
            selected = oof.assign(fold=0, input_variant="features")
            decoy = selected.copy()
            decoy["input_variant"] = "bad_variant"
            decoy["proba_target"] = decoy.proba_target.iloc[::-1].to_numpy()
            pd.concat([decoy, selected]).to_csv(folder / "oof_predictions.csv", index=False)
            truth.to_csv(folder / "evaluation_truth.csv", index=False)
            seal_fixture(folder, bridge.task_spec, score=1.0)
        else:
            source = root / "external.csv"
            oof.iloc[::-1].to_csv(source, index=False)
            register_external_prediction(root, "A", source)
        return bridge

    def test_lexical_ids_align_without_mutating_sealed_evidence(self):
        from harness.threshold_data import load_binary_threshold_data
        for ids in (["01", "1", "9007199254740993", "9007199254740994"],):
            bridge = self.make_bridge(variants=True)
            folder = bridge.artifacts_dir / "lexical"
            folder.mkdir()
            pd.DataFrame({"id": ids, "target": [0, 0, 1, 1]}).to_csv(folder / "evaluation_truth.csv", index=False)
            frame = pd.DataFrame({"id": ids, "fold": [0] * 4, "input_variant": ["features"] * 4,
                "pred_target": [0, 0, 1, 1],
                "proba_target": [json.dumps([v, 1-v]) for v in [.1, .2, .8, .9]],
                "classes_target": [json.dumps([1, 0])] * 4})
            frame.iloc[::-1].to_csv(folder / "oof_predictions.csv", index=False)
            # Reversed estimator order is explicitly declared in this synthetic contract.
            bridge.task_spec.subtasks["target"].classes = [1, 0]
            seal_fixture(folder, bridge.task_spec, string_ids=True)
            original = {p.name: p.read_bytes() for p in folder.iterdir()}
            labels, probabilities, _ = load_binary_threshold_data(bridge.project, bridge.task_spec, bridge.task_plugin, "lexical")
            np.testing.assert_array_equal(labels, [1, 1, 0, 0])
            np.testing.assert_allclose(probabilities, [.9, .8, .2, .1])
            self.assertEqual(original, {p.name: p.read_bytes() for p in folder.iterdir()})
            for broken_ids in (["01", "01", ids[2], ids[3]], ["missing", *ids[1:]]):
                broken = folder.parent / ("broken" + str(len(list(folder.parent.iterdir()))))
                broken.mkdir()
                pd.DataFrame({"id": ids, "target": [0, 0, 1, 1]}).to_csv(broken / "evaluation_truth.csv", index=False)
                frame.assign(id=broken_ids).to_csv(broken / "oof_predictions.csv", index=False)
                with self.assertRaises(ValueError):
                    seal_fixture(broken, bridge.task_spec, string_ids=True)
                    load_binary_threshold_data(bridge.project, bridge.task_spec, bridge.task_plugin, broken.name)

    def optimize(self, bridge, **params):
        return bridge.execute_action("optimize_threshold", {"exp_id": "A", "metric": "accuracy", **params})[1]

    def test_undeclared_reversed_probability_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "class order"):
            self.make_bridge(reverse=True)
        bridge = self.make_bridge(reverse=False)
        obs = self.optimize(bridge)
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertEqual(obs.composite_score, 1.0)

    def test_target_column_alias_and_string_labels_are_supported(self):
        for strings in (False, True):
            with self.subTest(strings=strings):
                bridge = self.make_bridge(alias=True, strings=strings)
                obs = self.optimize(bridge)
                self.assertEqual(obs.status, "SUCCESS", obs.summary)
                self.assertEqual(obs.composite_score, 1.0)
                self.assertEqual(obs.data["target"], "target")

    def test_only_the_selected_input_variant_is_used(self):
        obs = self.optimize(self.make_bridge(variants=True))
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertEqual(obs.composite_score, 1.0)
        self.assertEqual(obs.data["n_samples"], 4)
        self.assertEqual(obs.data["selection_variant"], "features")

    def test_regression_is_not_median_binarized(self):
        obs = self.optimize(self.make_bridge(regression=True))
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("binary", obs.summary.lower())

    def test_modified_source_evidence_cannot_drive_a_threshold_recommendation(self):
        bridge = self.make_bridge()
        path = bridge.artifacts_dir / "A/oof_predictions.csv"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        obs = self.optimize(bridge)
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("hash mismatch", obs.summary)

    def test_hard_labels_are_not_used_as_if_they_were_probabilities(self):
        obs = self.optimize(self.make_bridge(probabilities=False))
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("probabilities", obs.summary)

    def test_unknown_metrics_and_invalid_probabilities_fail_instead_of_using_f1(self):
        bridge = self.make_bridge()
        obs = self.optimize(bridge, metric="roc_auc")
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("metric", obs.summary.lower())
        for prob in ([.1, float("nan")], [.1, 1.1], [.1], []):
            with self.subTest(prob=prob):
                with self.assertRaises(ValueError):
                    ThresholdOptimizer.optimize_threshold(np.array([0, 1]), np.array(prob))

    def test_bounded_threshold_grid_keeps_the_baseline_as_a_candidate(self):
        threshold, score = ThresholdOptimizer.optimize_threshold(np.array([0, 1]), np.array([.2, .8]), num_thresholds=2)
        self.assertEqual(threshold, .5)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
