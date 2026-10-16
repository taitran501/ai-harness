"""Regression tests for the P2 audit items (agent friction follow-up).

1. HUD submissions block — recent submission registry entries surface in the
   Context Packet and the rendered Markdown so agents can audit submissions
   without digging through the workspace.
2. Workspace hygiene guard — deterministic, advisory root-clutter scan
   (scratch files, stray data files, unknown top-level dirs) surfaced in the HUD.
3. Semantic anti-duplicate — mechanism-level (model family) match advisory in
   ResearchMemory.review, on top of the exact-fingerprint guard.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.agent_bridge import AgentBridge
from harness.research_memory import ResearchMemory
from harness.submission_manager import SubmissionEntry
from harness.task_spec import UniversalTaskSpec
from tests.fixtures import copy_repo_configs

PLAIN_TASK = "task:\n  name: P2AuditTest\n  primary_metric: macro_f1\n"


def _make_bridge() -> tuple[AgentBridge, Path]:
    ws = Path(tempfile.mkdtemp(prefix="test_p2_"))
    copy_repo_configs(ws)
    (ws / "configs" / "task.yaml").write_text(PLAIN_TASK, encoding="utf-8")
    return AgentBridge(ws), ws


def _task_spec_dict() -> dict:
    return {
        "task": {
            "name": "p2-mech",
            "modality": "text",
            "primary_metric": "macro_f1",
            "direction": "maximize",
            "data": {
                "id_column": "id",
                "train_path": "dataset/train.csv",
                "fold_column": "fold",
            },
            "validation": {"strategy": "kfold", "num_folds": 2},
            "inputs": [{"name": "text", "type": "text", "source_columns": ["text"]}],
            "subtasks": [
                {
                    "id": "sentiment",
                    "column": "target",
                    "task_type": "binary_classification",
                    "classes": [0, 1],
                    "primary_metric": "macro_f1",
                }
            ],
        }
    }


def _cfg(exp_id: str, model_name: str, c: float = 1.0) -> dict:
    return {
        "experiment": {
            "id": exp_id,
            "branch": "baseline",
            "parent_id": None,
            "problem_addressed": "Recall on minority class.",
            "hypothesis": f"Regularization strength {c} on {model_name}.",
            "intervention": f"Train {model_name} with C={c} under fixed inputs.",
        },
        "model": {"name": model_name, "params": {"C": c}},
    }


class TestHudSubmissionsBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge, self.ws = _make_bridge()
        self.addCleanup(shutil.rmtree, self.ws, True)

    def _register_entry(self, **overrides) -> SubmissionEntry:
        sub_mgr = self.bridge._new_submission_manager()
        defaults = dict(
            id="001_exp001",
            exp_id="exp001",
            created_at="2026-10-16 12:00:00 UTC",
            status="SCORED",
            local_cv_scores={"macro_f1": 0.7123},
            official_scores={"public": 0.7041},
            archive_path=str(self.ws / "submissions" / "001_exp001" / "sub.csv"),
        )
        defaults.update(overrides)
        entry = SubmissionEntry(**defaults)
        sub_mgr.submissions[entry.id] = entry
        sub_mgr.save()
        return entry

    def test_packet_exposes_recent_submissions(self) -> None:
        self._register_entry()
        pkt = self.bridge.get_context_packet()
        subs = pkt.get("recent_submissions") or []
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["id"], "001_exp001")
        self.assertEqual(subs[0]["exp_id"], "exp001")
        self.assertEqual(subs[0]["status"], "SCORED")
        self.assertAlmostEqual(subs[0]["local_cv"]["macro_f1"], 0.7123)
        self.assertAlmostEqual(subs[0]["official"]["public"], 0.7041)
        self.assertEqual(subs[0]["path"], "submissions/001_exp001/sub.csv")

    def test_hud_renders_submissions_block(self) -> None:
        self._register_entry()
        md = self.bridge.render_context_markdown()
        self.assertIn("## 📦 Submissions (last 1)", md)
        self.assertIn("001_exp001", md)
        self.assertIn("SCORED", md)
        self.assertIn("submissions/001_exp001/sub.csv", md)

    def test_hud_empty_state(self) -> None:
        md = self.bridge.render_context_markdown()
        self.assertIn("## 📦 Submissions (last 0)", md)
        self.assertIn("(none yet", md)

    def test_packet_without_submissions_is_empty_list(self) -> None:
        pkt = self.bridge.get_context_packet()
        self.assertEqual(pkt.get("recent_submissions"), [])


class TestWorkspaceHygiene(unittest.TestCase):
    def test_clean_workspace(self) -> None:
        from harness.hygiene import scan_workspace_hygiene

        ws = Path(tempfile.mkdtemp(prefix="test_p2_hyg_clean_"))
        self.addCleanup(shutil.rmtree, ws, True)
        (ws / "configs").mkdir()
        (ws / "src").mkdir()
        (ws / ".ai-harness").mkdir()
        (ws / "main.py").write_text("print('hi')\n", encoding="utf-8")
        result = scan_workspace_hygiene(ws)
        self.assertTrue(result["clean"])
        self.assertEqual(result["findings"], [])

    def test_scratch_and_stray_data_detected(self) -> None:
        from harness.hygiene import scan_workspace_hygiene

        ws = Path(tempfile.mkdtemp(prefix="test_p2_hyg_"))
        self.addCleanup(shutil.rmtree, ws, True)
        (ws / "_fastread.py").write_text("x=1\n", encoding="utf-8")
        (ws / "_kneenet_raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (ws / "dump.parquet").write_bytes(b"\x00\x01")
        (ws / "scratchy").mkdir()
        (ws / "scratchy" / "a.txt").write_text("a", encoding="utf-8")
        (ws / "notes.md").write_text("# notes\n", encoding="utf-8")
        result = scan_workspace_hygiene(ws)
        self.assertFalse(result["clean"])
        kinds = {f["kind"] for f in result["findings"]}
        self.assertIn("scratch_file", kinds)
        self.assertIn("stray_data_file", kinds)
        self.assertIn("unknown_dir", kinds)
        paths = {f["path"] for f in result["findings"]}
        self.assertIn("_fastread.py", paths)
        self.assertIn("_kneenet_raw.csv", paths)
        self.assertIn("dump.parquet", paths)
        self.assertIn("scratchy", paths)
        # legitimate files are not flagged
        self.assertNotIn("notes.md", paths)

    def test_hud_lines(self) -> None:
        from harness.hygiene import hygiene_hud_lines, scan_workspace_hygiene

        ws = Path(tempfile.mkdtemp(prefix="test_p2_hyg_lines_"))
        self.addCleanup(shutil.rmtree, ws, True)
        (ws / "_temp.py").write_text("x=1\n", encoding="utf-8")
        (ws / "junk.csv").write_text("a\n1\n", encoding="utf-8")
        (ws / "mystery").mkdir()
        result = scan_workspace_hygiene(ws)
        lines = hygiene_hud_lines(result)
        self.assertTrue(lines)
        self.assertIn("hygiene issue", lines[0])
        clean_lines = hygiene_hud_lines({"clean": True, "findings": [], "summary": "ok"})
        self.assertEqual(clean_lines, [])

    def test_hud_surfaces_findings_in_markdown(self) -> None:
        from harness.hygiene import scan_workspace_hygiene

        self.bridge, self.ws = _make_bridge()
        self.addCleanup(shutil.rmtree, self.ws, True)
        (self.ws / "_scratch_agent.py").write_text("x=1\n", encoding="utf-8")
        md = self.bridge.render_context_markdown()
        self.assertIn("## 🧹 Workspace Hygiene", md)
        self.assertIn("_scratch_agent.py", md)
        result = scan_workspace_hygiene(self.ws)
        self.assertFalse(result["clean"])


class TestMechanismAntiDuplicate(unittest.TestCase):
    def _memory(self, tmp: Path) -> ResearchMemory:
        artifacts = tmp / ".ai-harness" / "artifacts" / "p2-mech"
        artifacts.mkdir(parents=True, exist_ok=True)
        return ResearchMemory(tmp, artifacts, UniversalTaskSpec.from_dict(_task_spec_dict()))

    def test_same_family_different_params_matches(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_p2_mech_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        memory = self._memory(tmp)
        memory.propose(_cfg("exp001_a", "tfidf_logistic_regression", c=1.0))
        review = memory.review(_cfg("exp002_b", "tfidf_logistic_regression", c=5.0))
        self.assertTrue(review["allowed"])
        self.assertEqual(len(review["mechanism_matches"]), 1)
        match = review["mechanism_matches"][0]
        self.assertEqual(match["exp_id"], "exp001_a")
        self.assertEqual(match["mechanism"], "tfidf_logistic_regression")

    def test_different_family_no_match(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_p2_mech_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        memory = self._memory(tmp)
        memory.propose(_cfg("exp001_a", "tfidf_logistic_regression", c=1.0))
        review = memory.review(_cfg("exp002_b", "gradient_boost_lgbm", c=1.0))
        self.assertEqual(review["mechanism_matches"], [])

    def test_empty_model_no_match_no_crash(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_p2_mech_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        memory = self._memory(tmp)
        memory.propose(_cfg("exp001_a", "tfidf_logistic_regression"))
        cfg = _cfg("exp002_b", "anything")
        cfg["model"] = {}
        review = memory.review(cfg)
        self.assertEqual(review["mechanism_matches"], [])

    def test_same_exp_proposal_not_self_matched(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_p2_mech_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        memory = self._memory(tmp)
        memory.propose(_cfg("exp001_a", "tfidf_logistic_regression"))
        review = memory.review(_cfg("exp001_a", "tfidf_logistic_regression", c=2.0))
        self.assertEqual(
            [m["exp_id"] for m in review["mechanism_matches"]], []
        )

    def test_review_result_shape(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="test_p2_mech_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        memory = self._memory(tmp)
        review = memory.review(_cfg("exp001_a", "tfidf_logistic_regression"))
        self.assertIn("mechanism_matches", review)
        self.assertIn("mechanism_matches", review["interpretation"])


if __name__ == "__main__":
    unittest.main()
