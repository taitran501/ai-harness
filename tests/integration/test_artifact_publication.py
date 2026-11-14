import multiprocessing
import tempfile
import unittest
from pathlib import Path

from harness.artifacts import ArtifactConflictError, publish_run_artifacts
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import task


def _publish_worker(source, dest, ready, output):
    ready.wait(20)
    try:
        result = publish_run_artifacts(source, dest, task_spec=None, expected={}, project_root=Path(source).parent, run_kind="full")
        output.put(result.disposition)
    except ArtifactConflictError:
        output.put("conflict")
    except Exception as error:
        output.put(repr(error))


class ConcurrentPublicationTests(unittest.TestCase):
    def test_real_process_publishers(self):
        for identical in (True, False):
            with self.subTest(identical=identical), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first, second = root / "first", root / "second"
                seal_fixture(first, task())
                if identical:
                    second = first
                else:
                    seal_fixture(second, task(), score=.8)
                ctx = multiprocessing.get_context("spawn")
                ready, output = ctx.Event(), ctx.Queue()
                processes = [ctx.Process(target=_publish_worker, args=(str(source), str(root / "final"), ready, output)) for source in (first, second)]
                for process in processes:
                    process.start()
                ready.set()
                for process in processes:
                    process.join(30)
                    self.assertEqual(process.exitcode, 0)
                outcomes = sorted(output.get(timeout=5) for _ in processes)
                self.assertEqual(outcomes, sorted(["published", "already_present" if identical else "conflict"]))
                output.close()
