import copy
import unittest

from harness.agent_bridge import _render_compact_context


class ContextRenderingBudgetTests(unittest.TestCase):
    def test_adversarial_diagnostics_keep_sections_and_policy(self):
        huge = "🔥\ntraceback " * 100000
        packet = {"active_session": {"goal": huge, "status": "PAUSED_FOR_APPROVAL",
                                    "session_id": huge, "pending_approval_action": {"action": "prepare_submission", "rationale": huge}},
                  "gates": {"freeze_locked": True}, "execution_config": {"backend": "kaggle", "locked": True},
                  "competition_hud": {"what_is_blocking_me": [{"code": "FATAL_STATE", "severity": "fatal", "message": huge}] * 3000},
                  "parent_diagnostics_bottlenecks": [huge] * 1000,
                  "ucb_recommendation": {"diagnostic_cue": huge},
                  "recent_submissions": [{"id": huge}] * 3000,
                  "validated_incumbent": {"evidence_ref": {"id": "control"}, "score": .95, "evaluation_id": "eval1"},
                  "latest_probe": huge}
        before = copy.deepcopy(packet)
        output = _render_compact_context(packet, max_chars=4000)
        self.assertLessEqual(len(output), 4000)
        headings = ["## Objective and state", "## Active Blockers", "## Verified incumbent and metrics",
                    "## Latest probe / unresolved result", "## Next permitted actions"]
        self.assertEqual(sorted(headings, key=output.index), headings)
        for text in ("Goal:", "FATAL_STATE", "LOCKED", "PENDING USER APPROVAL", "control", "Latest probe", "truncated"):
            self.assertIn(text, output)
        self.assertNotIn("Gate-checked next", output)
        self.assertEqual(packet, before)
        self.assertEqual(output, _render_compact_context(packet, max_chars=4000))

    def test_optional_sections_unknown_evidence_and_exact_smaller_budget(self):
        output = _render_compact_context({})
        self.assertIn("Verified incumbent: unknown", output)
        self.assertIn("No unresolved result", output)
        for budget in (0, 20, 500, 4000, 8000):
            self.assertLessEqual(len(_render_compact_context({}, max_chars=budget)), min(budget, 4000))
