"""Job state: worker job-ID validation surfaces the expected pattern.

Agents repeatedly dispatched with test job ids (``test_job``, ``job_test``...)
and got only "Job ID contains unsupported characters." — with no pattern or
example — so they retried blindly. The error must name the offending id, the
expected regex, and a valid example (arc-prize-2026 audit, gap G5).
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from harness.job_state import JOB_ID_PATTERN, worker_state_path
from harness.project import HarnessProject
from tests.fixtures import make_workspace


class WorkerStatePathTests(unittest.TestCase):
    def setUp(self):
        self.root: Path = make_workspace()[0]
        self.project = HarnessProject.discover(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_job_id_resolves_inside_artifacts_dir(self):
        path = worker_state_path(self.project, "job_exp001_full")
        self.assertEqual(path.name, "job_exp001_full.worker.json")
        self.assertIn(str(self.project.artifacts_dir), str(path))

    def test_invalid_job_id_error_names_id_pattern_and_example(self):
        with self.assertRaises(ValueError) as ctx:
            worker_state_path(self.project, "test_job")
        message = str(ctx.exception)
        self.assertIn("test_job", message)
        self.assertIn(JOB_ID_PATTERN.pattern, message)
        self.assertIn("job_exp001_full", message)

    def test_invalid_job_id_rejects_space_and_wrong_prefix(self):
        for bad in ("job ", "JOB_x", "x_job_1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    worker_state_path(self.project, bad)


if __name__ == "__main__":
    unittest.main()
