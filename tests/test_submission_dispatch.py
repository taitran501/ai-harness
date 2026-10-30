"""Tests for ``harness.submission_dispatch`` (the ``ai-harness submission`` CLI).

Covers all four subcommands (dispatch / status / watch / latest) plus the JSON
error path, with the leaderboard tracker fully mocked so the suite runs offline
(no network, no real Kaggle ref). This module was previously at 0% coverage —
the exact dispatch surface agents hit when the tracker is reached.
"""
from __future__ import annotations

import contextlib
import io
import json
import unittest
import shutil
from unittest import mock

import pandas as pd
import yaml

from harness import submission_dispatch
from harness.endgame_submission import KaggleLeaderboardTracker
from harness.submission_gate import SubmissionGate, SubmissionGateError, consume_dispatch_permit
from tests.fixtures import workspace_with_verified_pair


class _FakeTracker:
    """In-memory stand-in for ``KaggleLeaderboardTracker`` (records calls)."""

    def __init__(self):
        self.dispatch_calls = []
        self.status_calls = []
        self.watch_calls = []
        self.latest_calls = []

    def dispatch(self, competition, file, message, kernel=None, kernel_version=None, **options):
        self.dispatch_calls.append(dict(
            competition=competition, file=file, message=message,
            kernel=kernel, kernel_version=kernel_version,
            **options,
        ))
        return {"ref": 123456, "message": "submitted"}

    def get_submission(self, competition, ref):
        self.status_calls.append((competition, ref))
        return {"ref": ref, "status": "complete", "public_score": 0.9}

    def wait_for_score(self, competition, ref, timeout_s=0, poll_interval_s=0, max_poll_interval_s=0):
        self.watch_calls.append(dict(
            competition=competition, ref=ref, timeout_s=timeout_s,
            poll_interval_s=poll_interval_s, max_poll_interval_s=max_poll_interval_s,
        ))
        return {"ref": ref, "status": "complete", "public_score": 0.87, "attempts": 3}

    def latest(self, competition):
        self.latest_calls.append(competition)
        return {"ref": 999, "public_score": 0.93, "message": "best"}


def _run(argv, tracker):
    """Run ``submission_dispatch.main`` with the tracker mocked; return (code, payload)."""
    buf = io.StringIO()
    with mock.patch.object(
        submission_dispatch, "KaggleLeaderboardTracker", return_value=tracker
    ):
        with contextlib.redirect_stdout(buf):
            code = submission_dispatch.main(argv)
    return code, json.loads(buf.getvalue())


class SubmissionDispatchTests(unittest.TestCase):
    def test_client_initialization_error_is_parseable_json(self):
        out = io.StringIO()
        with mock.patch.object(submission_dispatch, "KaggleLeaderboardTracker", side_effect=RuntimeError("authentication unavailable")), contextlib.redirect_stdout(out):
            code = submission_dispatch.main(["dispatch", "comp", "sub.csv"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())["error"], "authentication unavailable")

    def test_dispatch_file_submits_and_reports_ref(self):
        t = _FakeTracker()
        code, payload = _run(["dispatch", "comp", "sub.csv", "-m", "Exp 22"], t)
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["action"], "dispatch")
        self.assertEqual(payload["ref"], 123456)
        self.assertEqual(t.dispatch_calls[0]["message"], "Exp 22")
        self.assertIsNone(t.dispatch_calls[0]["kernel"])

    def test_dispatch_code_competition_passes_kernel(self):
        t = _FakeTracker()
        code, payload = _run(
            ["dispatch", "comp", "out.csv", "-k", "owner/k", "--kernel-version", "3"], t
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "dispatch")
        self.assertEqual(t.dispatch_calls[0]["kernel"], "owner/k")
        self.assertEqual(t.dispatch_calls[0]["kernel_version"], "3")

    def test_status_returns_tracker_payload(self):
        t = _FakeTracker()
        code, payload = _run(["status", "comp", "57314333"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "status")
        self.assertEqual(payload["public_score"], 0.9)
        self.assertEqual(t.status_calls[0], ("comp", "57314333"))

    def test_watch_passes_poll_args(self):
        t = _FakeTracker()
        code, payload = _run(["watch", "comp", "57314333", "--timeout", "30", "--poll", "5"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "watch")
        self.assertEqual(payload["attempts"], 3)
        self.assertEqual(t.watch_calls[0]["timeout_s"], 30)
        self.assertEqual(t.watch_calls[0]["poll_interval_s"], 5)

    def test_latest_returns_best(self):
        t = _FakeTracker()
        code, payload = _run(["latest", "comp"], t)
        self.assertEqual(code, 0)
        self.assertEqual(payload["action"], "latest")
        self.assertEqual(payload["ref"], 999)
        self.assertEqual(t.latest_calls[0], "comp")

    def test_error_path_returns_nonzero_and_json_error(self):
        t = _FakeTracker()

        def _boom(*args, **kwargs):
            raise RuntimeError("no such ref")

        t.get_submission = _boom
        code, payload = _run(["status", "comp", "0"], t)
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["action"], "status")
        self.assertIn("no such ref", payload["error"])


class SubmissionGateTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        project_manifest = yaml.safe_load((self.root / ".ai-harness/project.yaml").read_text(encoding="utf-8"))
        state_path = self.root / project_manifest["artifacts_dir"] / "competition_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"gates": {"baseline_locked": True, "baseline_exp_id": "control"}}), encoding="utf-8")
        from harness.competition_state import CompetitionStateMachine
        state_machine = CompetitionStateMachine(self.root)
        state_machine.state["gates"].update(baseline_locked=True, baseline_exp_id="control")
        state_machine.save()
        self.gate = SubmissionGate(self.root)
        predictions = pd.read_csv(self.gate.project.artifacts_dir / "control/test_predictions.csv")
        self.payload = self.gate.plugin.build_submission(self.gate.context, predictions, self.root / "submission.csv")
        self.quota = dict(daily_quota=5, num_today=0, has_remaining_quota=True)

    def authorize(self, **kwargs):
        options = dict(exp_id="control", user_approved=True, quota=self.quota)
        options.update(kwargs)
        return self.gate.authorize_dispatch("fixture-comp", self.payload, **options)

    def test_dry_run_does_not_reserve_or_send(self):
        result = self.authorize(dry_run=True)
        self.assertEqual(result["receipt"]["status"], "pass")
        self.assertFalse((self.root / ".ai-harness/submission_events.jsonl").exists())
        from jsonschema import Draft202012Validator
        from pathlib import Path
        schema = json.loads((Path(__file__).resolve().parents[1] / "harness/schemas/submission_gate_receipt.v2.schema.json").read_text())
        Draft202012Validator(schema).validate(result["receipt"])

    def test_missing_approval_blocks_and_does_not_reserve(self):
        with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_APPROVAL_REQUIRED"):
            self.authorize(user_approved=False)
        self.assertFalse((self.root / ".ai-harness/submission_events.jsonl").exists())

    def test_missing_quota_and_exhausted_quota_block(self):
        for quota, reason in ((None, "QUOTA_UNKNOWN"), (dict(daily_quota=5, num_today=5), "QUOTA_EXHAUSTED")):
            with self.assertRaisesRegex(SubmissionGateError, reason):
                self.authorize(quota=quota)

    def test_permit_is_single_use_and_bound_to_competition(self):
        auth = self.authorize()
        with self.assertRaisesRegex(SubmissionGateError, "PERMIT_BINDING_MISMATCH"):
            consume_dispatch_permit(auth["permit"], file_name=auth["intent"]["file_name"], competition="wrong")
        with self.assertRaisesRegex(SubmissionGateError, "EXPIRED_OR_CONSUMED"):
            consume_dispatch_permit(auth["permit"], file_name=auth["intent"]["file_name"], competition="fixture-comp")

    def test_prepared_sending_intent_prevents_concurrent_send(self):
        self.authorize()
        with self.assertRaisesRegex(SubmissionGateError, "UNRESOLVED_DISPATCH"):
            self.authorize()

    def test_payload_change_and_forged_receipt_cannot_authorize(self):
        self.payload.write_text("id,label\n40,999\n")
        with self.assertRaisesRegex(SubmissionGateError, "PAYLOAD_EVIDENCE_MISMATCH"):
            self.authorize()
        with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_PERMIT_REQUIRED"):
            consume_dispatch_permit({"status": "pass"}, file_name=str(self.payload), competition="fixture-comp")

    def test_duplicate_is_blocked_after_confirmed_dispatch(self):
        auth = self.authorize()
        self.gate.finish_submission_intent(auth, provider_ref=123)
        with self.assertRaisesRegex(SubmissionGateError, "DUPLICATE_PAYLOAD"):
            self.authorize()

    def test_tracker_failure_records_unknown_and_never_retries(self):
        client = mock.Mock()
        client.get_submission_limits.return_value = self.quota
        client.list_submissions.return_value = []
        client.submit_submission.side_effect = TimeoutError("ambiguous send")
        tracker = KaggleLeaderboardTracker(client=client)
        with self.assertRaises(TimeoutError):
            tracker.dispatch("fixture-comp", str(self.payload), project_root=self.root, exp_id="control", user_approved=True)
        self.assertEqual(client.submit_submission.call_count, 1)
        with self.assertRaisesRegex(SubmissionGateError, "UNRESOLVED_DISPATCH"):
            self.authorize()

    def test_facade_without_permit_cannot_reach_sdk(self):
        from harness.kaggle_compat import KaggleClient
        client = object.__new__(KaggleClient)
        with mock.patch.object(KaggleClient, "_legacy_api_instance") as raw:
            with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_PERMIT_REQUIRED"):
                client.submit_submission(str(self.payload), "test", "fixture-comp")
        raw.assert_not_called()

    def test_corrupt_authoritative_history_blocks(self):
        path = self.root / ".ai-harness/submission_events.jsonl"
        path.write_text("not-json\n")
        with self.assertRaises(ValueError):
            self.authorize()

    def test_kernel_output_requires_verified_version_binding(self):
        with self.assertRaisesRegex(SubmissionGateError, "UNVERIFIED_KERNEL_OUTPUT"):
            self.authorize(kernel="owner/kernel", kernel_version="3")

    def test_remote_history_cannot_be_hidden_by_empty_local_registry(self):
        with self.assertRaisesRegex(SubmissionGateError, "HISTORY_INCOMPLETE"):
            self.authorize(provider_refs=["untracked-remote-submission"])


    def test_candidate_file_dispatch_without_manifest_is_rejected(self):
        cand_file = self.root / "candidate_blend.csv"
        import shutil
        shutil.copyfile(self.payload, cand_file)
        with self.assertRaisesRegex(SubmissionGateError, "EXPERIMENT_EVIDENCE_REQUIRED"):
            self.gate.authorize_dispatch("fixture-comp", cand_file, exp_id=None, user_approved=True, dry_run=True, quota=self.quota)

    def test_preflight_or_approval_does_not_replace_promotion_receipt(self):
        from harness.candidate_preflight import run_candidate_preflight
        incumbent = self.root / "incumbent.csv"
        incumbent.write_bytes(self.payload.read_bytes())
        preflight = run_candidate_preflight(project_root=self.root, candidate_path=self.payload,
                                            incumbent_path=incumbent, id_col="id", pred_col="label")
        import shutil
        unpromoted_dir = self.gate.project.artifacts_dir / "unpromoted"
        shutil.copytree(self.gate.project.artifacts_dir / "control", unpromoted_dir)
        unpromoted_manifest = json.loads((unpromoted_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        unpromoted_manifest["exp_id"] = "unpromoted"
        (unpromoted_dir / "artifact_manifest.json").write_text(json.dumps(unpromoted_manifest), encoding="utf-8")
        unpromoted_metrics = json.loads((unpromoted_dir / "metrics.json").read_text(encoding="utf-8"))
        unpromoted_metrics["experiment_id"] = "unpromoted"
        (unpromoted_dir / "metrics.json").write_text(json.dumps(unpromoted_metrics), encoding="utf-8")
        unpromoted_cfg = yaml.safe_load((unpromoted_dir / "run_config.yaml").read_text(encoding="utf-8"))
        unpromoted_cfg.setdefault("experiment", {})["id"] = "unpromoted"
        (unpromoted_dir / "run_config.yaml").write_text(yaml.safe_dump(unpromoted_cfg), encoding="utf-8")
        from harness.artifacts import seal_file_map
        unpromoted_manifest["files"] = seal_file_map(unpromoted_dir)
        unpromoted_manifest["resolved_config_sha256"] = __import__("harness.artifacts", fromlist=["sha256"]).sha256(unpromoted_dir / "run_config.yaml")
        (unpromoted_dir / "artifact_manifest.json").write_text(json.dumps(unpromoted_manifest), encoding="utf-8")
        with self.assertRaisesRegex(SubmissionGateError, "PROMOTION_EVIDENCE_REQUIRED"):
            self.gate.evaluate("fixture-comp", self.payload, exp_id="unpromoted", user_approved=True,
                               preflight_id=preflight["preflight_id"], quota=self.quota)

    def test_backdoor_script_locks_dispatch(self):
        script = self.root / "scratch"
        script.mkdir(exist_ok=True)
        (script / "push.py").write_text("kaggle.api.competition_submit(file_name='x')\n")
        with self.assertRaisesRegex(SubmissionGateError, "DISPATCH_BACKDOOR_DETECTED"):
            self.gate.evaluate("fixture-comp", self.payload, exp_id="control", user_approved=True, quota=self.quota)

    def test_consumption_rechecks_expiry_and_payload_bytes(self):
        auth = self.authorize()
        with mock.patch("harness.submission_gate.time.monotonic", return_value=float("inf")):
            with self.assertRaisesRegex(SubmissionGateError, "EXPIRED_OR_CONSUMED"):
                consume_dispatch_permit(auth["permit"], file_name=auth["intent"]["file_name"], competition="fixture-comp")

    def test_verified_prospective_promotion_can_dispatch(self):
        from harness.decision_rules import register_decision_rule, reserve_comparison_slot
        from tests.fixtures import make_submission_evidence
        from tests.test_research_memory import rule_document
        auth = self.authorize()
        self.gate.finish_submission_intent(auth, provider_ref=123)
        registration = register_decision_rule(rule_document(), project=self.gate.project, task_spec=self.gate.spec)
        cfg = dict(experiment=dict(id="promoted"), validation=dict(partition="oof"),
                   research=dict(decision_rule_id=registration["rule_id"]))
        reserve_comparison_slot(self.gate.project, self.gate.spec, cfg)
        directory = make_submission_evidence(self.root, "promoted", cfg=cfg)
        payload = self.gate.plugin.build_submission(self.gate.context, pd.read_csv(directory / "test_predictions.csv"), self.root / "promoted.csv")
        auth = self.gate.authorize_dispatch("fixture-comp", payload, exp_id="promoted", rule_id=registration["rule_id"],
            quota=self.quota, user_approved=True)
        self.assertEqual(auth["receipt"]["purpose"], "promotion")
        self.assertEqual(auth["receipt"]["status"], "pass")
        self.gate.finish_submission_intent(auth, provider_ref=124)


if __name__ == "__main__":
    unittest.main()
