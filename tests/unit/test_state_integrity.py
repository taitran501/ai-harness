import copy
import tempfile
import unittest
from pathlib import Path

from harness.state_store import StateIntegrityError, read_json_state, save_snapshot, transaction
from harness.bridge_common import migrate_session, validate_session
from harness.competition_state import validate_competition_state
from harness.research_memory import validate_research_state


class StateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "state.json"

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
