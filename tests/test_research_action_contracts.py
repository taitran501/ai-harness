"""Regression tests for research-action friction observed in a real session.

Source: aimo session_e279045ca2484cc58ce27ff73bcf22fb (2026-10-14..16), where
the agent hit five validation failures while recording research continuity:

1. ``record_research_decision`` with ``revisit_when: null`` failed validation
   (None skipped the auto-default; the agent repeated the same mistake in a
   later step).
2. ``record_research_decision`` with prose in ``outcome`` — the error was
   correct but gave no self-correction hint (``outcome`` is an enum; the
   narrative belongs in ``conclusion``).
3. ``record_research_insight`` with a single-ID string ``evidence`` instead
   of a list.
4. ``record_research_checkpoint`` passed ``conclusion`` instead of
   ``summary`` -> raw ``KeyError: 'summary'``.
5. None of these contracts were documented in the generated SKILL.md, so the
   agent had no choice but to guess.

Each test reconstructs the failing call shape and asserts the fix.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from harness.agent_bridge import AgentBridge
from harness.project import attach_project
from harness.research_memory import ResearchMemory
from harness.skill_injector import generate_skill_content
from harness.task_spec import UniversalTaskSpec
from tests.evidence_fixtures import seal_fixture
from tests.test_research_continuity import config, task_document


class ResearchActionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-research-contract-")
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()))
        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.memory = ResearchMemory(self.root, self.artifacts, self.spec)
        attach_project(self.root, self.root / "configs/task.yaml")
        self.bridge = AgentBridge(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _insight(self):
        # Record through the bridge so the insight lives in the bridge's
        # task-scoped memory instance (scope matches the decision handler).
        _, obs = self.bridge.execute_action("record_research_insight", params={
            "title": "Minority recall",
            "observation": "Minority errors persist",
            "mechanism": "Loss weighting could help",
            "next_action": "Compare a fixed control",
        })
        assert obs.status == "SUCCESS", obs.summary
        return obs.data

    def _verified_run(self, exp_id="exp001"):
        folder = self.artifacts / exp_id
        folder.mkdir()
        (folder / "run_config.yaml").write_text(yaml.safe_dump(config(exp_id)))
        seal_fixture(folder, self.spec, 0.7)
        return folder

    # --- 1. revisit_when: null must behave like an omitted field -----------
    def test_decision_revisit_when_null_uses_auto_default(self):
        insight = self._insight()
        _, obs = self.bridge.execute_action("record_research_decision", params={
            "insight_id": insight["id"],
            "outcome": "rejected",
            "conclusion": "Hard 50/50 partition degraded accuracy.",
            "lesson": "Do not hard-partition small batches.",
            "next_action": "Try soft weighting.",
            "revisit_when": None,  # JSON null, exactly what the agent sent
        })
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        entry = self.bridge._research_memory().entries()[-1]
        self.assertTrue(entry["revisit_when"].strip())
        self.assertIn("different mechanism", entry["revisit_when"])

    def test_decision_null_optional_fields_all_tolerated(self):
        # Without insight_id, next_action/lesson/revisit_when are optional;
        # JSON null for each must behave like omission (no auto-defaults
        # required for non-rejected/deferred-insight decisions).
        _, obs = self.bridge.execute_action("record_research_decision", params={
            "outcome": "inconclusive",
            "conclusion": "V6 verified at 0.71 on Codabench.",
            "next_action": None,
            "lesson": None,
            "revisit_when": None,
            "evidence": None,
        })
        self.assertEqual(obs.status, "SUCCESS", obs.summary)

    # --- 2. prose in outcome gets a self-correcting hint --------------------
    def test_decision_prose_outcome_raises_with_hint(self):
        with self.assertRaises(ValueError) as ctx:
            self.memory.record_decision(
                outcome="Peak Codabench Benchmark Confirmed at 0.71 (10/14)",
                conclusion="V6 verified at 0.71.",
            )
        message = str(ctx.exception)
        self.assertIn("outcome must be supported, rejected, inconclusive, or deferred", message)
        self.assertIn("looks like prose", message)
        self.assertIn("conclusion", message)

    def test_decision_bad_enum_short_value_no_prose_hint(self):
        with self.assertRaises(ValueError) as ctx:
            self.memory.record_decision(outcome="meh", conclusion="c")
        self.assertNotIn("looks like prose", str(ctx.exception))

    # --- 3. evidence accepts a single ID string -----------------------------
    def test_insight_evidence_single_id_string(self):
        self._verified_run("exp001")
        insight = self.memory.record_insight(
            title="Slice", observation="Recall gap", mechanism="Weighting may help",
            next_action="Plan a probe", evidence="exp001",
        )
        self.assertEqual(insight["evidence"], ["exp001"])

    def test_insight_evidence_comma_separated_string(self):
        self._verified_run("exp001")
        self._verified_run("exp002")
        insight = self.memory.record_insight(
            title="Slice", observation="Recall gap", mechanism="Weighting may help",
            next_action="Plan a probe", evidence="exp001, exp002",
        )
        self.assertEqual(insight["evidence"], ["exp001", "exp002"])

    def test_insight_evidence_unknown_id_still_rejected(self):
        self._verified_run("exp001")
        with self.assertRaises(ValueError):
            self.memory.record_insight(
                title="Slice", observation="Recall gap", mechanism="Weighting may help",
                next_action="Plan a probe", evidence="exp999",
            )

    # --- 4. checkpoint gives a friendly error, not a raw KeyError -----------
    def test_checkpoint_missing_summary_friendly_error(self):
        _, obs = self.bridge.execute_action("record_research_checkpoint", params={
            "conclusion": "V7 regression diagnosed.",  # wrong field name
            "next_action": "Implement V8.",
        })
        self.assertEqual(obs.status, "FAILED")
        self.assertIn("'summary'", obs.summary)
        self.assertNotIn("KeyError", obs.summary)
        # Friendly wording, not a raw exception repr.
        self.assertIn("requires", obs.summary)
        self.assertIn("not 'conclusion'", obs.summary)

    def test_checkpoint_with_summary_succeeds(self):
        _, obs = self.bridge.execute_action("record_research_checkpoint", params={
            "summary": "V7 regression diagnosed.",
            "next_action": "Implement V8.",
        })
        self.assertEqual(obs.status, "SUCCESS", obs.summary)

    # --- 5. generated skill documents the contracts --------------------------
    def test_skill_documents_research_action_contracts(self):
        content = generate_skill_content(task_document(), artifact_path="artifacts")
        self.assertIn("record_research_decision", content)
        self.assertIn("supported | rejected | inconclusive | deferred", content)
        self.assertIn("the field is `summary`, not `conclusion`", content)


if __name__ == "__main__":
    unittest.main()
