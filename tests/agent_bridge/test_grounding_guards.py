"""Regression tests for anti-bypass agent-integration hardening.

Covers the three failure modes an IDE agent tends to exploit when it is trying to
get past the harness rules:

- Grounding bypass: a non-baseline proposal whose source was never verified
  (`verified_via='not_verified'` / empty) must be rejected, not silently passed.
- Duplicate-submission padding: re-synthesizing an identical candidate recipe
  under a new display name must be rejected so the agent cannot 'preserve' a
  submission or pad the candidate pool.
- Stagnation: the `suggest_next_interventions` action must surface untried,
  non-graveyard hypotheses so the agent is not left to stop brainstorming.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.agent_bridge import AgentBridge, TypedActionName
from harness.grounding_gate import GroundingGate
from tests.fixtures import SOURCE_ROOT


def _make_bridge(root: Path) -> AgentBridge:
    shutil.copytree(SOURCE_ROOT / "configs", root / "configs")
    return AgentBridge(root)


class TestGroundingVerifiedViaEnforced(unittest.TestCase):
    """A non-baseline proposal whose source was never verified is rejected."""

    def _cfg(self, verified_via):
        cfg = {
            "experiment": {
                "id": "exp020_feat_check",
                "branch": "feat",
                "parent_id": "exp001_baseline",
                "problem_addressed": "Address the rare-slice underfitting seen in parent diagnostics.",
                "hypothesis": "Targeted feature cleaning lifts rare-slice F1 by at least 0.02.",
                "intervention": "Apply a single targeted cleaning step to the input.",
            },
            "grounding": {
                "source_type": "arxiv",
                "source_id": "arxiv:2305.12345",
                "retrieval_summary": "Cites an empirical method that targets the diagnosed slice.",
            },
            "assumption_register": {
                "known_facts": ["Parent exp001 achieved F1=0.62."],
                "assumptions": ["Cleaning helps rare slices without raising variance."],
                "uncertainty_level": "low",
            },
        }
        if verified_via is not None:
            cfg["grounding"]["verified_via"] = verified_via
        return cfg

    def test_unverified_rejected(self) -> None:
        for bad in ("not_verified", "", "unverified", "none"):
            cfg = self._cfg(bad)
            ok, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
            self.assertFalse(ok, f"Should reject verified_via={bad!r}")
            self.assertTrue(any("GROUNDING VERIFICATION ERROR" in e for e in errors))

    def test_verified_passes(self) -> None:
        for good in ("search_web", "read_url_content", "local_registry"):
            ok, errors, _ = GroundingGate.verify_config(self._cfg(good), raise_on_error=False)
            self.assertTrue(ok, f"Should accept verified_via={good!r}: {errors}")


class TestDuplicateCandidateGuard(unittest.TestCase):
    """Re-synthesizing an identical recipe under a new name is rejected."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_dupcand_")
        self.root = Path(self._ws.name)
        self.bridge = _make_bridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def _prime_pool(self, fingerprint):
        # Refresh once so comp_state is loaded, then patch refresh to a no-op so the
        # pool we prime below survives the execute_action() call.
        self.bridge._refresh_task_context()
        self.bridge.comp_state.state["candidate_pool"]["synthesized_candidates"] = [
            {"id": "cand_0001", "name": "first", "mode": "single_model",
             "recipe_fingerprint": fingerprint}
        ]
        return patch.object(self.bridge, "_refresh_task_context")

    def test_duplicate_recipe_blocked(self) -> None:
        fp = "abc"
        with self._prime_pool(fp), \
             patch("harness.candidates._candidate_recipe_fingerprint", return_value=fp), \
             patch.object(self.bridge.comp_state, "synthesize_candidate") as synth:
            synth.return_value = {"id": "cand_0002", "recipe_fingerprint": fp}
            _sess, obs = self.bridge.execute_action(
                action=TypedActionName.PROMOTE_CANDIDATE.value,
                params={"exp_id": "exp005"},
            )
        # execute_action catches the guard's ValueError and returns a FAILED obs.
        self.assertEqual(obs.status, "FAILED")
        self.assertIn("already", obs.summary.lower())
        # The real synthesize_candidate must NOT have been called (blocked early).
        synth.assert_not_called()

    def test_revisit_reason_allows_duplicate(self) -> None:
        fp = "abc"
        with self._prime_pool(fp), \
             patch("harness.candidates._candidate_recipe_fingerprint", return_value=fp), \
             patch.object(self.bridge.comp_state, "synthesize_candidate") as synth:
            synth.return_value = {"id": "cand_0002", "recipe_fingerprint": fp}
            # With an explicit revisit_reason the guard lets the re-synthesis through.
            _sess, _obs = self.bridge.execute_action(
                action=TypedActionName.PROMOTE_CANDIDATE.value,
                params={"exp_id": "exp005", "revisit_reason": "re-evaluate after data fix"},
            )
        synth.assert_called_once()


class TestSuggestNextInterventions(unittest.TestCase):
    """suggest_next_interventions returns untried hypotheses and flags stagnation."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_suggest_")
        self.root = Path(self._ws.name)
        self.bridge = _make_bridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_returns_well_formed_suggestions(self) -> None:
        # Stub the journal/research-memory so the action is deterministic.
        with patch.object(self.bridge, "_new_journal") as journal_mock, \
             patch.object(self.bridge, "_research_memory") as mem_mock:
            journal = journal_mock.return_value
            journal.nodes = {}
            journal.recommend_next_branch.return_value = {
                "parent_id": "exp000_baseline",
                "recommended_branch": "feat",
                "ucb_score": 0.5,
                "bottlenecks": ["rare-slice underfitting"],
                "diagnostic_cue": "rare-slice underfitting",
            }
            mem = mem_mock.return_value
            mem.context.return_value = {
                "research_frontier": [
                    {"insight_id": "i1", "title": "Try target attention",
                     "next_action": "Add target attention to the head",
                     "mechanism": "Cross-attention aligns rare slices.",
                     "status": "needs_probe_plan"},
                ],
                "reassess_recommended": False,
                "consecutive_non_improvements": 0,
            }
            res = self.bridge._suggest_next_interventions(limit=5)

        self.assertIn("suggestions", res)
        self.assertIn("stagnation", res)
        self.assertEqual(res["parent_id"], "exp000_baseline")
        # The frontier insight should be surfaced.
        sources = [s["source"] for s in res["suggestions"]]
        self.assertIn("research_frontier", sources)

    def test_graveyard_filter(self) -> None:
        with patch.object(self.bridge, "_new_journal") as journal_mock, \
             patch.object(self.bridge, "_research_memory") as mem_mock:
            journal = journal_mock.return_value
            journal.nodes = {}
            journal.recommend_next_branch.return_value = {
                "parent_id": "exp000_baseline",
                "recommended_branch": "feat",
                "ucb_score": 0.5,
                "bottlenecks": [],
                "diagnostic_cue": None,
            }
            mem = mem_mock.return_value
            mem.context.return_value = {
                "research_frontier": [
                    {"insight_id": "i1", "title": "Try stem cleaning",
                     "next_action": "Try stem cleaning on the input",
                     "mechanism": "", "status": "needs_probe_plan"},
                ],
                "reassess_recommended": False,
                "consecutive_non_improvements": 0,
            }
        # The bridge's graveyard extractor should be able to exclude a matching idea.
        with patch.object(self.bridge, "_extract_idea_graveyard",
                          return_value=["Try stem cleaning on the input"]):
            res = self.bridge._suggest_next_interventions(limit=5)
        # The graveyard-matching frontier suggestion must be filtered out.
        for s in res["suggestions"]:
            self.assertNotIn("stem cleaning", s["hypothesis"].lower(),
                             "Graveyard idea should not be re-suggested")


if __name__ == "__main__":
    unittest.main()
