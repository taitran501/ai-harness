import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.citation_audit import (
    CitationAuditInputError,
    audit_claims,
    build_portable_citation_bundle,
    discover_documents,
    enforce_citation_admission,
    parse_citations,
    validate_audit_receipt,
    validate_portable_citation_bundle,
    verify_claim_evidence,
)
from harness.citation_review import (
    CitationReviewError,
    run_claim_review,
    validate_verifier_execution,
)
from harness.citation_sources import CitationSourceError, SourceRegistry
from harness.grounding_gate import GroundingGate
from harness.research_memory import ResearchMemory

REPO_ROOT = Path(__file__).resolve().parents[2]


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


class CitationAdmissionTests(unittest.TestCase):
    def test_baseline_is_explicitly_not_applicable_and_omitted_discussion_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = enforce_citation_admission(root, {"experiment": {"branch": "baseline"}})
            self.assertEqual(baseline["admission"], "not_applicable")
            discussion = enforce_citation_admission(root, {"grounding": {"source_type": "kaggle_discussion"}})
            self.assertEqual(discussion["admission"], "blocked")
            self.assertEqual(discussion["reason"], "DISCUSSION_CITATION_RECEIPT_REQUIRED")

    def test_unreviewed_receipt_reference_is_revalidated_and_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = enforce_citation_admission(
                root, {"research": {"citation_dependencies": [".ai-harness/citation_audits/missing.json"]}})
            self.assertEqual(result["admission"], "blocked")

    def test_discussion_grounding_cannot_pass_on_verified_via_alone(self):
        config = {
            "experiment": {"id": "exp1", "branch": "feature", "problem_addressed": "A documented bottleneck in validation",
                           "hypothesis": "The change will improve the score because of a causal mechanism.",
                           "intervention": "Control is fixed; treatment changes one mechanism."},
            "grounding": {"source_type": "kaggle_discussion", "source_id": "topic-750885",
                          "retrieval_summary": "The discussion motivates a bounded mechanism probe.", "verified_via": "kaggle_discussion"},
            "assumption_register": {"known_facts": ["The task contract is loaded."],
                                    "assumptions": ["The mechanism transfers to this split."],
                                    "uncertainty_level": "medium"},
        }
        passed, errors, receipt = GroundingGate.verify_config(config, raise_on_error=False, project_root=".")
        self.assertFalse(passed)
        self.assertTrue(any("CITATION ADMISSION BLOCKED" in error for error in errors))
        self.assertEqual(receipt["citation_admission"]["admission"], "blocked")

    def test_cloud_dispatch_blocks_before_provider_preflight_when_discussion_receipt_is_missing(self):
        from harness.cloud_runner import _prepare_remote_run
        from tests.fixtures import make_workspace

        root, task_path = make_workspace(prefix="aih-citation-cloud-")
        try:
            config = {"experiment": {"id": "exp-citation"},
                      "grounding": {"source_type": "kaggle_discussion"}}
            with patch.dict(os.environ, {}), \
                 patch("harness.cloud_runner.CloudPreflightInspector.run_full_preflight_check",
                       return_value=(True, [], config)), \
                 patch("harness.cloud_policy.provider_preflight") as provider:
                with self.assertRaisesRegex(ValueError, "CITATION_ADMISSION_BLOCKED"):
                    _prepare_remote_run(str(task_path), "kaggle", root)
                provider.assert_not_called()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_unreviewed_external_note_is_kept_as_hypothesis_and_can_have_a_bounded_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = root / ".ai-harness" / "artifacts" / "task"
            memory = ResearchMemory(root, artifacts_dir=artifacts, task_spec={})
            insight = memory.record_insight(
                title="Possible scheduling effect", observation="A discussion asks about fairness.",
                mechanism="A scheduler policy may change which examples are processed first.",
                next_action="Compare one policy against a fixed control.", source_refs=["Topic #750885"],
                empirical_status="external_report")
            self.assertEqual(insight["empirical_status"], "hypothesis")
            self.assertEqual(insight["citation_admission"]["admission"], "blocked")
            plan = memory.plan_probe(
                insight_id=insight["id"], control="fixed scheduler baseline",
                intervention="change only queue scheduling policy",
                expected_observation="processing order changes under the treatment",
                alternative_explanation="the observed difference comes from sample ordering",
                decision_rule="compare paired outcomes under a preregistered rule",
                budget="one smoke fold", stop_rule="stop after the smoke fold shows no mechanism effect")
            self.assertEqual(plan["empirical_status"], "hypothesis")


class CitationCliTests(unittest.TestCase):
    def run_cli(self, root, *args):
        command = [sys.executable, "-m", "harness.cli", "audit-claims", "--project-root", str(root), *map(str, args)]
        return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)

    def test_only_fabricated_shiiin9_citation_is_advisory_in_lint_and_fails_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            SourceRegistry(root).register_topic_source(
                "750885", {"id": 750885, "body": "Parallel streams reached a 110k token cutoff."},
                adapter="normalized_topic_v1")
            doc = root / "fabricated.md"
            doc.write_text("The shiiin9 D-prime scheduler prevents starvation (Topic #750885).\n", encoding="utf-8")
            lint = self.run_cli(root, "--file", doc)
            strict = self.run_cli(root, "--file", doc, "--strict")
            self.assertEqual(lint.returncode, 0, lint.stdout + lint.stderr)
            self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)
            report = json.loads(self.run_cli(root, "--file", doc, "--json").stdout)
            self.assertEqual(report["findings"][0]["issue"], "UNGROUNDED_TOPIC_CITATION")
            self.assertEqual(report["results"][0]["claim_support"], "not_reviewed")
            self.assertFalse(report["strict_pass"])

    def test_json_and_text_report_same_counts_and_missing_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            doc = root / "missing.md"
            doc.write_text("The result is documented at Topic #750999.\n", encoding="utf-8")
            json_run = self.run_cli(root, "--file", doc, "--json")
            text_run = self.run_cli(root, "--file", doc)
            report = json.loads(json_run.stdout)
            self.assertEqual(json_run.returncode, 1)
            self.assertEqual(text_run.returncode, 1)
            self.assertIn("1 discovered, 0 bound", text_run.stdout)
            self.assertEqual(report["citations_discovered"], 1)
            self.assertEqual(report["documents_read"], 1)

    def test_empty_citations_have_lint_zero_strict_one_and_bad_file_is_two(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            doc = root / "empty.md"
            doc.write_text("No external sources here.\n", encoding="utf-8")
            self.assertEqual(self.run_cli(root, "--file", doc).returncode, 0)
            self.assertEqual(self.run_cli(root, "--file", doc, "--strict").returncode, 1)
            bad = self.run_cli(root, "--file", root / "absent.md", "--json")
            self.assertEqual(bad.returncode, 2)
            self.assertEqual(json.loads(bad.stdout)["status"], "error")


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
