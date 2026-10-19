"""Regression: task data bundles must skip (not abort on) cloned-dataset
version-control metadata while still refusing protected files.

Live origin: arc-prize-2026 staging refused to upload the competition data
directory because the dataset was cloned from GitHub and therefore
contained ``.git/`` metadata. ``.git`` inside a declared data directory is
not competition data; it is skipped. Protected paths (credentials, harness
evidence, provider state) still abort staging.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from harness.data_bundle import DataBundleError, stage_declared_data
from harness.project import HarnessProject, attach_project


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
        # A declared directory whose only files are .git metadata cannot
        # produce an uploadable bundle.
        (self.root / "dataset" / "train.csv").unlink()
        self._add_git_metadata()
        with tempfile.TemporaryDirectory(prefix="aih-bundle-dest-") as dest:
            with self.assertRaisesRegex(DataBundleError, "no uploadable files"):
                stage_declared_data(self.project, dest)


if __name__ == "__main__":
    unittest.main()
