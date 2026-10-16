"""Tests for artifact CSV deduplication (hardlink mirrors + canonical truth).

Covers the helpers in harness.artifacts (mirror_csv, write_csv_dedup,
canonical_truth_path, ensure_canonical_truth) plus end-to-end verification
that UniversalRunner and register_external_prediction produce linked
artifacts that still pass validate_run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from harness.artifacts import (
    canonical_truth_path,
    ensure_canonical_truth,
    mirror_csv,
    sha256,
    write_csv_dedup,
)


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
        other = self.tmp / "other.csv"
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
        from tests.test_e2e_pipeline import TestEndToEndPipelineIntegration

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
        self.assertEqual(oof.stat().st_ino, pred.stat().st_ino)

        canonicals = [
            p for p in runner.artifacts_dir.iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(canonicals), 1)
        self.assertEqual(truth.stat().st_ino, canonicals[0].stat().st_ino)
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
        self.assertEqual(truth_a.stat().st_ino, truth_b.stat().st_ino)
        masters = [
            p for p in artifacts.iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(masters), 1)


class TestRegisterExternalDedup(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_correctness_contracts import task

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
        from harness.candidates import register_external_prediction

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
        truth = exp_dir / "evaluation_truth.csv"
        self.assertEqual(oof.stat().st_ino, pred.stat().st_ino)
        masters = [
            p for p in (self.root / "artifacts").iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(masters), 1)
        self.assertEqual(truth.stat().st_ino, masters[0].stat().st_ino)

    def test_reregister_same_name_updates_links_consistently(self) -> None:
        from harness.candidates import register_external_prediction

        register_external_prediction(
            project=self.root, name="ext_gm_02", oof_path=self._ext_oof(0.1),
            author="GM", source="kaggle_kernel",
        )
        exp_dir = self.root / "artifacts" / "ext_gm_02"
        oof_before = (exp_dir / "oof_predictions.csv").read_bytes()
        # re-register the same name with different OOF content
        register_external_prediction(
            project=self.root, name="ext_gm_02", oof_path=self._ext_oof(0.9),
            author="GM", source="kaggle_kernel",
        )
        oof_after = (exp_dir / "oof_predictions.csv").read_bytes()
        pred_after = (exp_dir / "predictions.csv").read_bytes()
        truth_after = (exp_dir / "evaluation_truth.csv").read_bytes()
        self.assertNotEqual(oof_before, oof_after)
        self.assertEqual(oof_after, pred_after)
        # truth is identical content -> same canonical master, still linked
        masters = [
            p for p in (self.root / "artifacts").iterdir()
            if p.name.startswith("canonical_evaluation_truth_")
        ]
        self.assertEqual(len(masters), 1)
        self.assertEqual(truth_after, masters[0].read_bytes())


if __name__ == "__main__":
    unittest.main()
