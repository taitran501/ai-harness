import copy
import tempfile
import unittest
from pathlib import Path

from harness.state_store import StateIntegrityError, read_json_state, save_snapshot, transaction
from harness.state_store import merge_changed
from harness.bridge_common import migrate_session, validate_session
from harness.competition_state import validate_competition_state
from harness.research_memory import validate_research_state


class StateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "state.json"

    def test_numeric_assignments_conflict_atomically(self):
        for fields in ({"score": .8}, {"official": {"score": .8}},
                       {"metrics": {"local_runs": 10}}, {"timestamp": 100}):
            with self.subTest(fields=fields):
                self.path.unlink(missing_ok=True)
                baseline = {"first": "old", **fields}
                def changed(value, delta):
                    return {k: changed(v, delta) if isinstance(v, dict) else v + delta
                            for k, v in value.items()}
                save_snapshot(self.path, {}, baseline)
                save_snapshot(self.path, baseline, {"first": "old", **changed(fields, 1)})
                before = self.path.read_bytes()
                with self.assertRaisesRegex(ValueError, "Concurrent state conflict"):
                    save_snapshot(self.path, baseline, {"first": "new", **changed(fields, 2)})
                self.assertEqual(self.path.read_bytes(), before)

    def test_assignments_coalesce_and_independent_changes_survive(self):
        self.assertEqual(merge_changed({"x": 2, "y": 3}, {"x": 1, "y": 1}, {"x": 2, "y": 1}),
                         {"x": 2, "y": 3})
        with self.assertRaisesRegex(ValueError, "x"):
            merge_changed({"x": 1}, {}, {"x": 2})

    def test_only_declared_counter_paths_add(self):
        path = ("compute_usage", "local_runs")
        for baseline, current, incoming, expected in ((10, 12, 13, 15), (0, 1, 1, 2)):
            result = merge_changed({"compute_usage": {"local_runs": current}},
                                   {"compute_usage": {"local_runs": baseline}},
                                   {"compute_usage": {"local_runs": incoming}}, additive_paths={path})
            self.assertEqual(result["compute_usage"]["local_runs"], expected)
        with self.assertRaisesRegex(ValueError, "metrics.local_runs"):
            merge_changed({"metrics": {"local_runs": 12}}, {"metrics": {"local_runs": 10}},
                          {"metrics": {"local_runs": 13}}, additive_paths={path})
        for current, incoming in ((12, 0), (0, 12)):
            with self.assertRaises(ValueError):
                merge_changed({"n": current}, {"n": 10}, {"n": incoming}, additive_paths={("n",)})

    def test_counter_types_and_fractional_time(self):
        path = ("compute_usage", "wall_seconds_used")
        result = merge_changed({"compute_usage": {"wall_seconds_used": 1.25}},
                               {"compute_usage": {"wall_seconds_used": 1.0}},
                               {"compute_usage": {"wall_seconds_used": 1.125}}, additive_paths={path})
        self.assertEqual(result["compute_usage"]["wall_seconds_used"], 1.375)
        for operands in ((0, 1, True), (False, 1, 2), (0, True, 2), (0, 1, float("inf")),
                         (0, 1, float("nan")), (0, 1, 1.5)):
            with self.subTest(operands=operands), self.assertRaises(ValueError):
                merge_changed({"n": operands[1]}, {"n": operands[0]}, {"n": operands[2]},
                              additive_paths={("n",)})

    def test_ledger_usage_is_an_absolute_total_on_stale_saves(self):
        from harness.agent_bridge import AgentBridge
        from tests.fixtures import make_workspace
        from harness.io_utils import atomic_write_json
        root, _ = make_workspace(root_path=Path(self.tmp.name) / "workspace")
        first = AgentBridge(root)
        session = first.start_session(goal="Counter reconciliation")
        other = AgentBridge(root)
        stale = other.load_session(session.session_id)
        atomic_write_json(first.artifacts_dir / "resource_ledger.json", {"reservations": {
            "one": {"session_id": session.session_id, "status": "COMPLETED", "category": "training",
                    "backend": "local", "wall_seconds": 1.125}}})
        first.save_session(session)
        other.save_session(stale)
        self.assertEqual(other.load_session(session.session_id).compute_usage["experiments_used"], 1)
        self.assertEqual(stale.compute_usage["wall_seconds_used"], 1.125)

    def test_absent_initializes_and_named_absent_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_json_state(self.path)
        with transaction(self.path, {"entries": []}) as state:
            state["entries"].append(1)
        self.assertEqual(read_json_state(self.path)["entries"], [1])

    def test_corruption_never_yields_or_overwrites(self):
        for raw in (b"", b"{", b"null", b"[]", b'"x"', b'{"x":1,"x":2}',
                    b'{"x":NaN}', b'\xff', b'{"revision":true}'):
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                with self.assertRaises(StateIntegrityError):
                    with transaction(self.path):
                        self.fail("corruption reached mutation body")
                self.assertEqual(self.path.read_bytes(), raw)

    def test_schema_and_output_validation(self):
        self.path.write_text('{"entries": []}', encoding="utf-8")
        original = self.path.read_bytes()
        with self.assertRaises(StateIntegrityError):
            with transaction(self.path, validate=validate_research_state) as value:
                value["entries"] = "bad"
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaises(ValueError):
            validate_competition_state({"gates": {"freeze_locked": "false"}})

    def test_snapshot_detects_new_corruption(self):
        self.path.write_bytes(b"{")
        with self.assertRaises(StateIntegrityError):
            save_snapshot(self.path, {"x": 1}, {"x": 2})
        self.assertEqual(self.path.read_bytes(), b"{")

    def test_legacy_session_migration(self):
        value = {"id": "run_1", "goal": "test", "status": "ACTIVE", "created_at": "now", "updated_at": "now"}
        old = copy.deepcopy(value)
        migrated = migrate_session(value)
        validate_session(migrated)
        self.assertEqual(migrated["session_id"], "run_1")
        self.assertEqual(value, old)

    def test_directory_is_unreadable_not_missing(self):
        self.path.mkdir()
        with self.assertRaises(StateIntegrityError) as error:
            read_json_state(self.path, default_factory=dict)
        self.assertEqual(error.exception.code, "STATE_UNREADABLE")

    def test_corrupt_named_session_and_enumeration_never_fall_back(self):
        from harness.agent_bridge import AgentBridge
        from tests.fixtures import make_workspace
        root, _ = make_workspace(root_path=Path(self.tmp.name) / "workspace")
        bridge = AgentBridge(root)
        session = bridge.start_session(goal="Preserve state")
        path = bridge.sessions_dir / f"{session.session_id}.json"
        path.write_bytes(b"{")
        for action in (lambda: bridge.load_session(session.session_id), bridge.list_sessions,
                       lambda: bridge.get_context_packet(session.session_id), lambda: bridge.save_session(session)):
            with self.assertRaises(StateIntegrityError):
                action()
            self.assertEqual(path.read_bytes(), b"{")

    def test_corrupt_memory_cannot_append_or_write_progress(self):
        from harness.research_memory import ResearchMemory
        from tests.fixtures import make_workspace
        root, _ = make_workspace(root_path=Path(self.tmp.name) / "workspace")
        memory = ResearchMemory(root)
        memory.path.parent.mkdir(parents=True, exist_ok=True)
        memory.path.write_bytes(b'{"entries":"bad"}')
        before = memory.path.read_bytes()
        with self.assertRaises(StateIntegrityError):
            memory._append("observation", title="must fail")
        self.assertEqual(memory.path.read_bytes(), before)
        self.assertFalse(memory.progress_path.exists())
