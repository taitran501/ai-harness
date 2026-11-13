import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.artifacts import (ArtifactConflictError, publish_run_artifacts,
                               _rename_directory_noreplace, sha256)
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import task


class ArtifactPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.dest = self.root / "source", self.root / "final"
        self.spec = task()
        seal_fixture(self.source, self.spec)

    def publish(self):
        return publish_run_artifacts(self.source, self.dest, task_spec=self.spec,
                                     expected={"exp_id": "source"}, project_root=self.root, run_kind="full")

    def inventory(self, path):
        return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}

    def test_publish_and_exact_retry(self):
        self.assertEqual(self.publish().disposition, "published")
        before = self.inventory(self.dest)
        self.assertEqual(self.publish().disposition, "already_present")
        self.assertEqual(before, self.inventory(self.dest))

    def test_conflicting_and_empty_destination_preserved(self):
        self.dest.mkdir()
        for populated in (False, True):
            if populated:
                (self.dest / "sentinel").write_bytes(b"keep")
            before = self.inventory(self.dest)
            with self.assertRaises(ArtifactConflictError):
                self.publish()
            self.assertEqual(self.inventory(self.dest), before)

    def test_fresh_hash_detects_same_size_timestamp_corruption(self):
        path = self.source / "metrics.json"
        sha256(path)
        info = path.stat()
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"0.75", b"0.85"))
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(self.dest.exists())

    def test_missing_undeclared_and_unsafe_files_fail_without_destination(self):
        (self.source / "extra").touch()
        with self.assertRaises(ValueError):
            self.publish()
        (self.source / "extra").unlink()
        path = self.source / "artifact_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["../escape"] = "bad"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(self.dest.exists())

    def test_copy_failure_and_destination_creation_race(self):
        with patch("harness.artifacts.shutil.copytree", side_effect=OSError("copy failed")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertFalse(self.dest.exists())
        real = _rename_directory_noreplace
        def race(source, destination):
            destination.mkdir()
            (destination / "sentinel").write_bytes(b"keep")
            return real(source, destination)
        with patch("harness.artifacts._rename_directory_noreplace", side_effect=race):
            with self.assertRaises((ArtifactConflictError, OSError)):
                self.publish()
        self.assertEqual((self.dest / "sentinel").read_bytes(), b"keep")

    def test_platform_rename_refuses_existing_empty_directory(self):
        source = self.root / "rename-source"
        source.mkdir()
        self.dest.mkdir()
        with self.assertRaises(OSError):
            _rename_directory_noreplace(source, self.dest)
        self.assertTrue(source.is_dir())
        self.assertTrue(self.dest.is_dir())

    def test_reparse_directory_is_rejected_before_descending(self):
        from types import SimpleNamespace
        import stat
        linked = self.source / "linked"
        linked.mkdir()
        real_stat, real_iter = Path.lstat, Path.iterdir
        def metadata(path):
            if path == linked:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return real_stat(path)
        def children(path):
            if path == linked:
                self.fail("Publication traversed a reparse point")
            return real_iter(path)
        with patch.object(Path, "lstat", metadata), patch.object(Path, "iterdir", children):
            with self.assertRaisesRegex(ValueError, "reparse"):
                self.publish()
        self.assertFalse(self.dest.exists())
