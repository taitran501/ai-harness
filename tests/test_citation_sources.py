import json
import tempfile
import unittest
from pathlib import Path

from harness.citation_sources import CitationSourceError, SourceRegistry


class CitationSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = SourceRegistry(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_bom_file_is_read_with_utf8_sig_and_hashes_raw_bytes(self):
        path = self.root / "topic_750885.json"
        raw = b"\xef\xbb\xbf" + json.dumps({"id": 750885, "content": "Exact topic evidence."}).encode("utf-8")
        path.write_bytes(raw)
        record = self.registry.register_topic_source(750885, {}, adapter="normalized_topic_v1", source_path=path)
        self.assertEqual(record["raw_sha256"], __import__("hashlib").sha256(raw).hexdigest())
        self.assertEqual(self.registry.resolve_source(750885)["segments"][0]["text"], "Exact topic evidence.")

    def test_malformed_json_and_invalid_encoding_are_distinct(self):
        malformed = self.root / "topic_1.json"
        malformed.write_bytes(b"{")
        with self.assertRaisesRegex(CitationSourceError, "MALFORMED_JSON"):
            self.registry.register_topic_source(1, {}, adapter="normalized_topic_v1", source_path=malformed)
        invalid = self.root / "topic_2.json"
        invalid.write_bytes(b"\xff\xfe\xfa")
        with self.assertRaisesRegex(CitationSourceError, "INVALID_ENCODING"):
            self.registry.register_topic_source(2, {}, adapter="normalized_topic_v1", source_path=invalid)

    def test_adapters_are_explicit_and_keep_authors_separate(self):
        payload = {"id": 12, "title": "Thread", "content": "Body text",
                   "comments": [{"id": "c-9", "author": "Alice", "content": "Comment text"},
                                {"author_name": "Bob", "content": "Second comment"}]}
        record = self.registry.register_topic_source(12, payload, adapter="normalized_topic_v1")
        comment, missing_id_comment = record["segments"][2:]
        self.assertEqual(comment["comment_id"], "c-9")
        self.assertEqual(comment["author"], {"author": "Alice"})
        self.assertEqual(comment["text"], "Comment text")
        self.assertIsNone(missing_id_comment["comment_id"])
        self.assertEqual(missing_id_comment["locator"], "/comments/1/content")
        self.assertEqual(missing_id_comment["author"], {"author_name": "Bob"})
        with self.assertRaisesRegex(CitationSourceError, "UNSUPPORTED_ADAPTER"):
            self.registry.register_topic_source(13, {"content": "text"}, adapter="auto")

    def test_supported_data_wrapper_and_string_list_schemas(self):
        wrapped = self.registry.register_topic_source(
            20, {"data": {"id": 20, "title": "wrapped", "body": "Wrapper content"}},
            adapter="data_wrapper_v1")
        listed = self.registry.register_topic_source(21, ["first string", "second string"], adapter="string_list_v1")
        self.assertEqual(wrapped["segments"][-1]["text"], "Wrapper content")
        self.assertEqual([segment["locator"] for segment in listed["segments"]], ["/0", "/1"])

    def test_id_agreement_empty_payload_and_incomplete_retrieval(self):
        with self.assertRaisesRegex(CitationSourceError, "TOPIC_ID_MISMATCH"):
            self.registry.register_topic_source(30, {"id": 31, "body": "text"}, adapter="normalized_topic_v1")
        with self.assertRaisesRegex(CitationSourceError, "TOPIC_ID_MISMATCH"):
            wrong_file = self.root / "topic_31.json"
            wrong_file.write_text('{"id": 31, "body": "text"}', encoding="utf-8")
            self.registry.register_topic_source(30, {}, adapter="normalized_topic_v1", source_path=wrong_file)
        with self.assertRaisesRegex(CitationSourceError, "EMPTY_EVIDENCE"):
            self.registry.register_topic_source(32, {"id": 32}, adapter="normalized_topic_v1")
        record = self.registry.register_topic_source(33, {"id": 33, "body": "topic body"},
                                                    adapter="normalized_topic_v1", complete=False)
        self.assertFalse(record["completeness"]["comments_retrieved"])

    def test_identical_snapshots_deduplicate_and_conflicts_become_ambiguous(self):
        first = {"id": 40, "body": "same"}
        a = self.registry.register_topic_source(40, first, adapter="normalized_topic_v1")
        b = self.registry.register_topic_source(40, first, adapter="normalized_topic_v1")
        self.assertEqual(a["source_id"], b["source_id"])
        with self.assertRaisesRegex(CitationSourceError, "AMBIGUOUS_SOURCE_VERSION"):
            self.registry.register_topic_source(40, {"id": 40, "body": "different"}, adapter="normalized_topic_v1")
        with self.assertRaisesRegex(CitationSourceError, "AMBIGUOUS_SOURCE_VERSION"):
            self.registry.resolve_source(40)
        self.assertEqual(self.registry.resolve_source(40, source_hash=a["raw_sha256"])["source_id"], a["source_id"])

    def test_all_payload_id_aliases_agree_and_explicit_identical_version_resolves(self):
        with self.assertRaisesRegex(CitationSourceError, "TOPIC_ID_MISMATCH"):
            self.registry.register_topic_source(45,
                {"topic_id": 45, "id": 46, "body": "conflicting IDs"}, adapter="normalized_topic_v1")
        first = self.registry.register_topic_source(47, {"id": 47, "body": "same bytes"},
                                                    adapter="normalized_topic_v1")
        second = self.registry.register_topic_source(47, {"id": 47, "body": "same bytes"},
                                                     adapter="normalized_topic_v1", snapshot_version="release-2")
        self.assertEqual(first["raw_sha256"], second["raw_sha256"])
        self.assertEqual(self.registry.resolve_source(47, snapshot_version="release-2")["source_id"],
                         second["source_id"])

    def test_conflicting_ids_in_source_path_and_unreadable_path(self):
        path = self.root / "topic_50.json"
        path.write_text('{"id": 51, "body": "text"}', encoding="utf-8")
        with self.assertRaisesRegex(CitationSourceError, "TOPIC_ID_MISMATCH"):
            self.registry.register_topic_source(50, {}, adapter="normalized_topic_v1", source_path=path)
        with self.assertRaisesRegex(CitationSourceError, "SOURCE_FILE_NOT_FOUND"):
            self.registry.register_topic_source(52, {}, adapter="normalized_topic_v1", source_path=self.root / "missing.json")

    def test_reversing_conflicting_duplicate_registration_never_selects_a_source(self):
        payloads = [{"id": 60, "body": "version A"}, {"id": 60, "body": "version B"}]
        ambiguous_markers = []
        for order in (payloads, list(reversed(payloads))):
            with tempfile.TemporaryDirectory() as temp:
                registry = SourceRegistry(temp)
                registry.register_topic_source(60, order[0], adapter="normalized_topic_v1")
                with self.assertRaisesRegex(CitationSourceError, "AMBIGUOUS_SOURCE_VERSION"):
                    registry.register_topic_source(60, order[1], adapter="normalized_topic_v1")
                state = registry._load()
                ambiguous_markers.append(state["ambiguous_sources"][0]["raw_sha256s"])
                with self.assertRaisesRegex(CitationSourceError, "AMBIGUOUS_SOURCE_VERSION"):
                    registry.resolve_source(60)
        self.assertEqual(ambiguous_markers[0], ambiguous_markers[1])

    def test_configured_evidence_path_cannot_escape_workspace(self):
        with self.assertRaisesRegex(CitationSourceError, "EVIDENCE_PATH_ESCAPE"):
            SourceRegistry(self.root, evidence_dir=self.root.parent / "outside")


if __name__ == "__main__":
    unittest.main()
