import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.citation_audit import CitationAuditInputError, audit_claims, discover_documents, parse_citations
from harness.citation_sources import SourceRegistry


class CitationAuditTests(unittest.TestCase):
    def test_markdown_citation_forms_and_exclusions(self):
        markdown = """TOPIC 750885; Topic #750886; Topic # 750887; topic_750888.json
Source #750889, color #123456, https://www.kaggle.com/competitions/demo/discussion/750890#comment-42,
and incidental https://example.com/path#750891
`Topic #750892` and `topic_750893.json`
```text
Topic #750894
https://www.kaggle.com/c/demo/discussion/750895#comment_3
```
"""
        found = parse_citations(markdown)
        self.assertEqual([item["topic_id"] for item in found],
                         ["750885", "750886", "750887", "750888", "750889", "750890"])
        self.assertEqual(found[-1]["form"], "kaggle_url")
        self.assertEqual(found[-1]["comment_anchor"], "comment-42")

    def test_discovery_is_task_aware_and_does_not_select_wrong_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs").mkdir()
            (root / "configs" / "task.yaml").write_text("task:\n  id: active-task\n  name: Active\n", encoding="utf-8")
            (root / ".ai-harness").mkdir()
            (root / ".ai-harness" / "project.yaml").write_text(
                "task_config: configs/task.yaml\nartifacts_dir: .ai-harness/artifacts\n"
                "citation_audit:\n  documents:\n    - reports/custom.md\n", encoding="utf-8")
            (root / "root.md").write_text("root", encoding="utf-8")
            (root / ".ai-harness" / "immediate.md").write_text("immediate", encoding="utf-8")
            custom = root / "reports" / "custom.md"
            custom.parent.mkdir()
            custom.write_text("custom", encoding="utf-8")
            active = root / ".ai-harness" / "artifacts" / "active-task"
            active.mkdir(parents=True)
            (active / "RESEARCH_PROGRESS.md").write_text("active progress", encoding="utf-8")
            wrong = root / ".ai-harness" / "artifacts" / "wrong-task"
            wrong.mkdir()
            (wrong / "RESEARCH_PROGRESS.md").write_text("wrong progress", encoding="utf-8")
            nested = root / ".ai-harness" / "artifacts" / "other" / "note.md"
            nested.parent.mkdir()
            nested.write_text("not selected", encoding="utf-8")
            report = discover_documents(root)
            paths = {item["relative_path"] for item in report["documents"]}
            self.assertEqual(paths, {"root.md", ".ai-harness/immediate.md", "reports/custom.md",
                                     ".ai-harness/artifacts/active-task/RESEARCH_PROGRESS.md"})
            self.assertEqual(report["coverage"], "complete")

    def test_unresolved_task_scope_is_reported_as_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "note.md").write_text("No task configured.", encoding="utf-8")
            old_task = root / "artifacts" / "old-task"
            old_task.mkdir(parents=True)
            (old_task / "research_memory.json").write_text("{}", encoding="utf-8")
            (old_task / "RESEARCH_PROGRESS.md").write_text("Do not select this task.", encoding="utf-8")
            report = discover_documents(root)
            self.assertEqual(report["coverage"], "partial")
            self.assertIn("ACTIVE_TASK_SCOPE_UNRESOLVED", report["scope_warnings"])
            self.assertEqual({item["relative_path"] for item in report["documents"]}, {"note.md"})

    def test_formula_citation_is_integrity_only_and_never_supported_by_lint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            SourceRegistry(root).register_topic_source(
                738766, {"id": 738766, "title": "A scoring question", "body": "Could the score be 0.12 times 0.75?"},
                adapter="normalized_topic_v1")
            doc = root / "formula.md"
            doc.write_text("The formula is 0.12 x 0.75 = 0.09 (Topic #738766).\n", encoding="utf-8")
            report = audit_claims(root, files=[doc], strict=True)
            self.assertEqual(report["results"][0]["evidence_integrity"], "verified")
            self.assertEqual(report["results"][0]["claim_support"], "not_reviewed")
            self.assertFalse(report["strict_pass"])

    def test_next_line_citation_does_not_bind_a_fabricated_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            SourceRegistry(root).register_topic_source(750885, {"id": 750885, "body": "Parallel streams reached a token cutoff."},
                                                       adapter="normalized_topic_v1")
            doc = root / "claims.md"
            doc.write_text("The shiiin9 D-prime scheduler proved starvation prevention.\nSource: Topic #750885\n",
                           encoding="utf-8")
            claims = {"schema_version": 1, "claims": [{"claim_id": "c1", "claim_text": "The shiiin9 D-prime scheduler proved starvation prevention.",
                       "empirical_status": "external_report", "topic_id": "750885",
                       "document": {"path": "claims.md", "start_line": 1, "end_line": 1},
                       "evidence": [{"topic_id": "750885", "locator": "/body", "excerpt": "Parallel streams reached a token cutoff."}]}]}
            claim_file = root / "claims.json"
            claim_file.write_text(json.dumps(claims), encoding="utf-8")
            report = audit_claims(root, files=[doc], claims_file=claim_file, strict=True)
            self.assertEqual(report["citations_bound"], 0)
            self.assertFalse(report["strict_pass"])

    def test_invalid_document_scope_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(CitationAuditInputError):
                discover_documents(root, files=["missing.md"])

    def test_repeated_citations_read_and_parse_one_snapshot_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = SourceRegistry(root).register_topic_source(
                750901, {"id": 750901, "body": "A registered source segment."}, adapter="normalized_topic_v1")
            snapshot = (root / ".ai-harness" / "evidence" / record["snapshot_path"]).resolve()
            doc = root / "repeated.md"
            doc.write_text("Source A (Topic #750901); source B (Topic #750901).\n", encoding="utf-8")
            original = Path.read_bytes
            calls = []
            def counted(path):
                calls.append(Path(path).resolve())
                return original(path)
            with patch.object(Path, "read_bytes", counted):
                report = audit_claims(root, files=[doc])
            self.assertEqual(report["citations_discovered"], 2)
            self.assertEqual(calls.count(snapshot), 1)


if __name__ == "__main__":
    unittest.main()
