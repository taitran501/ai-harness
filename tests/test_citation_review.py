import tempfile
import time
import unittest
from pathlib import Path

from harness.citation_review import CitationReviewError, run_claim_review, validate_verifier_execution
from harness.citation_sources import SourceRegistry


class CitationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = SourceRegistry(self.root).register_topic_source(
            750885, {"id": 750885, "body": "The scheduler does not prevent starvation."},
            adapter="normalized_topic_v1")
        self.claim = {"claim_id": "c1", "claim_text": "The scheduler prevents starvation",
                      "empirical_status": "external_report", "topic_id": "750885",
                      "document": {"path": "summary.md", "start_line": 1, "end_line": 1},
                      "evidence": [{"topic_id": "750885", "source_sha256": self.source["raw_sha256"],
                                    "locator": "/body", "excerpt": "The scheduler does not prevent starvation."}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_verifier_is_not_reviewed(self):
        result = run_claim_review(self.root, self.claim, proposer_id="proposer", verifier_id="verifier")
        self.assertEqual(result["verdict"], "not_reviewed")
        self.assertEqual(result["error"], "VERIFIER_NOT_CONFIGURED")

    def test_self_review_is_rejected(self):
        with self.assertRaisesRegex(CitationReviewError, "SELF_REVIEW_REJECTED"):
            run_claim_review(self.root, self.claim, verifier_adapter=lambda _: {},
                             proposer_id="same", verifier_id="same")

    def test_structured_verdict_is_bound_to_separate_execution_and_current_inputs(self):
        seen = {}
        def adapter(package):
            seen.update(package)
            return {"verdict": "contradicted", "support_explanation": "The source explicitly negates this claim.",
                    "limitations": ["The source is external."], "decision_evaluation_ids": []}
        result = run_claim_review(self.root, self.claim, verifier_adapter=adapter,
                                  proposer_id="proposer-A", verifier_id="verifier-B")
        self.assertEqual(seen["evidence_package"]["segments"][0]["text"], "The scheduler does not prevent starvation.")
        verdict = validate_verifier_execution(self.root, result["verdict_ref"], self.claim)
        self.assertEqual(verdict["verdict"], "contradicted")
        edited = dict(self.claim, claim_text="The scheduler prevents starvation always")
        with self.assertRaisesRegex(CitationReviewError, "STALE_VERIFIER_INPUT_HASH"):
            validate_verifier_execution(self.root, result["verdict_ref"], edited)

    def test_malformed_verdict_and_verifier_crash_are_recorded_as_failures(self):
        with self.assertRaisesRegex(CitationReviewError, "MALFORMED_VERDICT"):
            run_claim_review(self.root, self.claim, verifier_adapter=lambda _: {"verdict": "yes"},
                             proposer_id="p", verifier_id="v")
        with self.assertRaisesRegex(CitationReviewError, "VERIFIER_EXECUTION_FAILED"):
            run_claim_review(self.root, self.claim,
                             verifier_adapter=lambda _: (_ for _ in ()).throw(TimeoutError("timed out")),
                             proposer_id="p", verifier_id="v")
        executions = list((self.root / ".ai-harness" / "citation_audits" / "executions").glob("*.json"))
        self.assertTrue(any('"status": "failed"' in path.read_text(encoding="utf-8") for path in executions))

    def test_verifier_timeout_is_recorded_as_failed_execution(self):
        def slow_adapter(_):
            time.sleep(0.05)
            return {"verdict": "supported", "support_explanation": "late", "limitations": []}
        with self.assertRaisesRegex(CitationReviewError, "VERIFIER_TIMEOUT"):
            run_claim_review(self.root, self.claim, verifier_adapter=slow_adapter,
                             proposer_id="p", verifier_id="v", timeout_seconds=0.001)


if __name__ == "__main__":
    unittest.main()
