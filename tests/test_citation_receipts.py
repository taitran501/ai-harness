import json
import os
import tempfile
import unittest
from pathlib import Path

from harness.citation_audit import (
    CitationAuditInputError, audit_claims, build_portable_citation_bundle, enforce_citation_admission,
    validate_audit_receipt, validate_portable_citation_bundle, verify_claim_evidence,
)
from harness.citation_review import run_claim_review
from harness.citation_sources import SourceRegistry


class CitationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs" / "task.yaml").write_text(
            "name: Citation Test\ntask_type: classification\nprimary_metric: macro_f1\nmetadata:\n  id: citation-test\n",
            encoding="utf-8")
        (self.root / ".ai-harness").mkdir()
        (self.root / ".ai-harness" / "project.yaml").write_text(
            "task_config: configs/task.yaml\nartifacts_dir: .ai-harness/artifacts\n", encoding="utf-8")
        self.registry = SourceRegistry(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def make_claim(self, source, claim_id, claim_text, excerpt, line=1):
        return {"claim_id": claim_id, "claim_text": claim_text, "empirical_status": "hypothesis",
                "topic_id": source["topic_id"],
                "document": {"path": "summary.md", "start_line": line, "end_line": line},
                "evidence": [{"topic_id": source["topic_id"], "source_sha256": source["raw_sha256"],
                              "locator": "/body", "excerpt": excerpt}]}

    def reviewed_claim(self, claim, verdict="supported"):
        result = run_claim_review(self.root, claim, verifier_adapter=lambda _: {
            "verdict": verdict, "support_explanation": "Reviewed the claim against the quoted segment.",
            "limitations": []}, proposer_id="proposer", verifier_id="verifier")
        return {**claim, "verdict_ref": result["verdict_ref"]}

    def audit(self, doc, claims):
        claims_path = self.root / "claims.json"
        claims_path.write_text(json.dumps({"schema_version": 1, "claims": claims}), encoding="utf-8")
        return audit_claims(self.root, files=[doc], claims_file=claims_path, strict=True)

    def test_valid_receipt_revalidates_source_claim_and_verdict(self):
        source = self.registry.register_topic_source(750885,
            {"id": 750885, "body": "The scheduler batches token windows."}, adapter="normalized_topic_v1")
        claim = self.make_claim(source, "c1", "The scheduler batches token windows", "The scheduler batches token windows.")
        doc = self.root / "summary.md"
        doc.write_text("The scheduler batches token windows (Topic #750885).\nUnrelated note.\n", encoding="utf-8")
        claim = self.reviewed_claim(claim)
        report = self.audit(doc, [claim])
        self.assertTrue(report["strict_pass"])
        receipt = validate_audit_receipt(self.root, report["receipt"])
        self.assertEqual(receipt["claim_results"][0]["claim_support"], "supported")
        allowed = enforce_citation_admission(self.root, {
            "citation_dependencies": [report["receipt"]], "citation_claims": [claim]}, action="test")
        self.assertEqual(allowed["admission"], "allowed")
        blocked = enforce_citation_admission(self.root, {
            "citation_dependencies": [report["receipt"]],
            "citation_claims": [{"claim_id": "other", "claim_text": "Unrelated claim"}]}, action="test")
        self.assertEqual(blocked["reason"], "CITATION_CLAIM_NOT_BOUND_TO_RECEIPT")
        bundle = build_portable_citation_bundle(self.root, [report["receipt"]["path"]])
        self.assertNotIn(str(self.root), json.dumps(bundle))
        portable = validate_portable_citation_bundle(
            bundle, expected_task_id="citation-test", expected_scorer_sha256=receipt["scope"]["scorer_sha256"])
        self.assertEqual(portable["admission"], "allowed")
        doc.write_text("The scheduler batches token windows (Topic #750885).\nEdited unrelated note.\n", encoding="utf-8")
        with self.assertRaisesRegex(CitationAuditInputError, "DOCUMENT_HASH_MISMATCH"):
            validate_audit_receipt(self.root, report["receipt"])

    def test_two_claims_on_one_line_keep_independent_evidence_bindings(self):
        source = self.registry.register_topic_source(750886,
            {"id": 750886, "body": "Scheduler batches token windows. Scheduler yields fairly."},
            adapter="normalized_topic_v1")
        first = self.make_claim(source, "batch", "Scheduler batches token windows", "Scheduler batches token windows.")
        second = self.make_claim(source, "yield", "Scheduler yields fairly", "Scheduler yields fairly.")
        doc = self.root / "summary.md"
        doc.write_text("Scheduler batches token windows; Scheduler yields fairly (Topic #750886).\n", encoding="utf-8")
        report = self.audit(doc, [self.reviewed_claim(first), self.reviewed_claim(second)])
        self.assertTrue(report["strict_pass"])
        self.assertEqual({item["claim_id"] for item in report["results"][0]["claims"]}, {"batch", "yield"})

    def test_negation_and_question_quotes_require_semantic_review(self):
        source = self.registry.register_topic_source(750887,
            {"id": 750887, "body": "The scheduler does not prevent starvation. Could the scheduler scale to 10 streams?"},
            adapter="normalized_topic_v1")
        negation = self.make_claim(source, "neg", "The scheduler prevents starvation",
                                   "The scheduler does not prevent starvation.")
        question = self.make_claim(source, "question", "The scheduler scales to 10 streams",
                                   "Could the scheduler scale to 10 streams?")
        integrity = verify_claim_evidence(self.root, negation, negation["evidence"][0])
        self.assertEqual(integrity["integrity"], "verified")
        self.assertEqual(integrity["claim_support"], "not_reviewed")
        doc = self.root / "summary.md"
        doc.write_text("The scheduler prevents starvation; The scheduler scales to 10 streams (Topic #750887).\n",
                       encoding="utf-8")
        reviewed = [self.reviewed_claim(negation, "contradicted"), self.reviewed_claim(question, "insufficient")]
        report = self.audit(doc, reviewed)
        self.assertFalse(report["strict_pass"])
        self.assertEqual({item["claim_support"] for item in report["results"][0]["claims"]},
                         {"contradicted", "insufficient"})

    def test_missing_or_wrong_locator_excerpt_and_source_edit_fail_integrity(self):
        source = self.registry.register_topic_source(750888,
            {"id": 750888, "body": "A quote in the body. Another segment elsewhere."}, adapter="normalized_topic_v1")
        wrong_locator = self.make_claim(source, "bad", "claim", "A quote in the body.")
        wrong_locator["evidence"][0]["locator"] = "/comments/0/content"
        self.assertEqual(verify_claim_evidence(self.root, wrong_locator, wrong_locator["evidence"][0])["integrity"], "invalid")
        elsewhere = self.make_claim(source, "elsewhere", "claim", "Another segment elsewhere.")
        elsewhere["evidence"][0]["locator"] = "/title"
        self.assertEqual(verify_claim_evidence(self.root, elsewhere, elsewhere["evidence"][0])["error"],
                         "SEGMENT_LOCATOR_NOT_FOUND")
        snapshot = self.root / ".ai-harness" / "evidence" / source["snapshot_path"]
        snapshot.write_text('{"id":750888,"body":"edited"}', encoding="utf-8")
        self.assertEqual(verify_claim_evidence(self.root, elsewhere, elsewhere["evidence"][0])["integrity"], "invalid")

    def test_claim_document_edit_scope_mismatch_and_receipt_tampering_are_rejected(self):
        source = self.registry.register_topic_source(750889,
            {"id": 750889, "body": "A measured observation."}, adapter="normalized_topic_v1")
        claim = self.make_claim(source, "c1", "A measured observation", "A measured observation.")
        doc = self.root / "summary.md"
        doc.write_text("A measured observation (Topic #750889).\n", encoding="utf-8")
        report = self.audit(doc, [self.reviewed_claim(claim)])
        self.assertTrue(report["strict_pass"])
        with self.assertRaisesRegex(CitationAuditInputError, "SCOPE_MISMATCH"):
            validate_audit_receipt(self.root, report["receipt"], expected_scope={"task_id": "wrong"})
        task_path = self.root / "configs" / "task.yaml"
        original_task = task_path.read_text(encoding="utf-8")
        task_path.write_text(original_task.replace("Citation Test", "Citation Test Changed"), encoding="utf-8")
        with self.assertRaisesRegex(CitationAuditInputError, "SCOPE_MISMATCH"):
            validate_audit_receipt(self.root, report["receipt"])
        task_path.write_text(original_task, encoding="utf-8")
        doc.write_text("An edited claim (Topic #750889).\n", encoding="utf-8")
        with self.assertRaisesRegex(CitationAuditInputError, "DOCUMENT_HASH_MISMATCH|CLAIM_DOCUMENT_CHANGED"):
            validate_audit_receipt(self.root, report["receipt"])
        receipt_path = self.root / report["receipt"]["path"]
        receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_data["coverage"] = "nothing_scanned"
        receipt_path.write_text(json.dumps(receipt_data), encoding="utf-8")
        with self.assertRaisesRegex(CitationAuditInputError, "DIGEST_MISMATCH"):
            validate_audit_receipt(self.root, report["receipt"])

    def test_receipt_reference_digest_is_pinned(self):
        source = self.registry.register_topic_source(750892, {"id": 750892, "body": "Bound text."},
                                                     adapter="normalized_topic_v1")
        claim = self.make_claim(source, "c1", "Bound text", "Bound text.")
        doc = self.root / "summary.md"
        doc.write_text("Bound text (Topic #750892).\n", encoding="utf-8")
        report = self.audit(doc, [self.reviewed_claim(claim)])
        ref = dict(report["receipt"], digest="0" * 64)
        with self.assertRaisesRegex(CitationAuditInputError, "REFERENCE_DIGEST_MISMATCH"):
            validate_audit_receipt(self.root, ref)

    def test_receipt_path_escape_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside_temp:
            root = Path(temp)
            outside = Path(outside_temp) / "receipt.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(CitationAuditInputError, "NOT_FOUND_OR_OUTSIDE_ROOT"):
                validate_audit_receipt(root, outside)
            audit_dir = root / ".ai-harness" / "citation_audits"
            audit_dir.mkdir(parents=True)
            link = audit_dir / "escape.json"
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation is unavailable in this Windows environment")
            with self.assertRaisesRegex(CitationAuditInputError, "NOT_FOUND_OR_OUTSIDE_ROOT"):
                validate_audit_receipt(root, link)


if __name__ == "__main__":
    unittest.main()
