import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness.citation_sources import SourceRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
