"""Unit tests for evidence manifest schema, SHA-256 calculation, and relpath traversal checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.artifacts import seal_file_map, sha256, validate_run
from harness.endgame_submission import ThresholdOptimizer


def _make_run(root: Path, name: str, nested: bool = True) -> Path:
    run = root / name
    (run / "stages").mkdir(parents=True)
    (run / "run_config.yaml").write_text("model: {name: fixture}\n", encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({"primary_score": 0.5, "direction": "minimize", "experiment_id": name}),
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
        "files": seal_file_map(run),
    }


class EvidenceManifestPureLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_seal_file_map_is_recursive_with_relpath_keys(self) -> None:
        run = _make_run(self.tmp_path, "exp_rec")
        files = seal_file_map(run)
        self.assertIn("metrics.json", files)  # top-level keeps bare name
        self.assertIn("stages/train.log", files)  # nested sealed with POSIX relpath
        self.assertIn("stages/fold0/stdout.log", files)  # deeper nesting works
        self.assertNotIn("artifact_manifest.json", files)  # manifest never seals itself
        for name, value in files.items():
            self.assertEqual(value, sha256(run / name))

    def test_validate_run_rejects_traversal_and_absolute_keys(self) -> None:
        run = _make_run(self.tmp_path, "exp_trav", nested=False)
        manifest = _full_manifest(run, "exp_trav")
        for evil in ("../outside.log", "..\\outside.log", "stages/../../escape.log"):
            manifest["files"][evil] = sha256(run / "metrics.json")
            (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_run(run)
            del manifest["files"][evil]

    def test_old_top_level_manifest_still_validates(self) -> None:
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
            "files": {
                p.name: sha256(p) for p in run.iterdir() if p.is_file()
            },
        }
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        validate_run(run)

    def test_validate_run_detects_missing_required_artifact(self) -> None:
        run = _make_run(self.tmp_path, "exp_missing_req", nested=False)
        manifest = _full_manifest(run, "exp_missing_req")
        del manifest["files"]["oof_predictions.csv"]
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            validate_run(run)
        self.assertIn("Missing required artifact: oof_predictions.csv", str(ctx.exception))

    def test_validate_run_detects_missing_run_identity(self) -> None:
        run = _make_run(self.tmp_path, "exp_missing_id", nested=False)
        manifest = _full_manifest(run, "exp_missing_id")
        del manifest["exp_id"]
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            validate_run(run)
        self.assertIn("Missing run identity: exp_id", str(ctx.exception))

    def test_sha256_calculation_and_caching(self) -> None:
        test_file = self.tmp_path / "sample.txt"
        test_file.write_text("immutable evidence payload\n", encoding="utf-8")
        h1 = sha256(test_file)
        self.assertEqual(len(h1), 64)
        h2 = sha256(test_file)
        self.assertEqual(h1, h2)


class ThresholdEvidenceManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_modified_source_evidence_rejected_by_manifest(self) -> None:
        """Threshold evidence cannot be accepted if source predictions have been tampered."""
        run = _make_run(self.tmp_path, "exp_thresh", nested=False)
        manifest = _full_manifest(run, "exp_thresh")
        (run / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        validate_run(run)

        # Alter oof predictions
        oof_path = run / "oof_predictions.csv"
        oof_path.write_text(oof_path.read_text(encoding="utf-8") + "1,0.9\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            validate_run(run)
        self.assertIn("Artifact hash mismatch: oof_predictions.csv", str(ctx.exception))

    def test_unknown_metrics_and_invalid_probabilities_fail_closed(self) -> None:
        for prob in ([0.1, float("nan")], [0.1, 1.1], [0.1], []):
            with self.subTest(prob=prob):
                with self.assertRaises(ValueError):
                    ThresholdOptimizer.optimize_threshold(np.array([0, 1]), np.array(prob))

    def test_bounded_threshold_grid_keeps_the_baseline_as_a_candidate(self) -> None:
        threshold, score = ThresholdOptimizer.optimize_threshold(
            np.array([0, 1]), np.array([0.2, 0.8]), num_thresholds=2
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
