import concurrent.futures
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
import yaml
from harness.state_store import transaction, save_snapshot
from harness.task_spec import load_active_task_spec
from tests.test_correctness_contracts import task


def increment(path, count):
    for _ in range(count):
        with transaction(path, {"count": 0}) as value:
            value["count"] += 1


class StateContracts(unittest.TestCase):
    def test_threads_and_processes_preserve_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "state.json")
            with concurrent.futures.ThreadPoolExecutor(4) as executor:
                list(executor.map(lambda _: increment(path, 10), range(4)))
            workers = []
            for _ in range(3):
                worker = multiprocessing.Process(target=increment, args=(path, 10))
                # Windows spawn can transiently fail under load (leaked handles
                # from earlier subprocess-heavy tests); one retry is enough.
                try:
                    worker.start()
                except Exception:
                    # Windows spawn can transiently fail under load (leaked
                    # handles from earlier subprocess-heavy tests); one retry.
                    worker = multiprocessing.Process(target=increment, args=(path, 10))
                    worker.start()
                workers.append(worker)
            for worker in workers:
                worker.join(30)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(json.loads(Path(path).read_text())["count"], 70)

    def test_snapshot_preserves_unrelated_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            base = {"events": [], "gates": {"a": False, "b": False}}
            save_snapshot(path, {}, base)
            save_snapshot(
                path, base, {"events": ["one"], "gates": {"a": True, "b": False}}
            )
            result = save_snapshot(
                path, base, {"events": ["two"], "gates": {"a": False, "b": True}}
            )
            self.assertEqual(set(result["events"]), {"one", "two"})
            self.assertEqual(result["gates"], {"a": True, "b": True})

    def test_task_cache_tracks_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            doc = {"task": task().metadata}
            path.write_text(yaml.safe_dump(doc))
            first = load_active_task_spec(path)
            doc["task"]["name"] = "changed"
            path.write_text(yaml.safe_dump(doc))
            self.assertNotEqual(first.name, load_active_task_spec(path).name)
