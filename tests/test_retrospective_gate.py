"""Cadence, restart persistence, costly boundaries and bounded recovery."""
import contextlib
import io
import json
import os
import shutil
import unittest
from unittest import mock

import yaml

from harness.agent_bridge import AgentBridge, _research_continuity_hud_lines
from harness.decision_rules import (evaluate_decision_rule, register_decision_rule, reserve_comparison_slot)
from harness.macro_research import (GateViolationError, enforce_retrospective, is_retrospective_due,
    load_memory, record_macro_review, retrospective_status)
from harness.project import HarnessProject
from harness.runner import UniversalRunner, build_parser
from harness.stagnation import StagnationController
from harness.submission_gate import SubmissionGate
from tests.fixtures import make_submission_evidence, workspace_with_verified_pair
from tests.test_macro_research import review_payload
from tests.test_research_memory import rule_document


class RetrospectiveGateTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = workspace_with_verified_pair()
        self.addCleanup(shutil.rmtree, self.root)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.project = HarnessProject.discover(self.root)
        self.memory = load_memory(self.project)

    def add_runs(self, count=4):
        for i in range(count):
            make_submission_evidence(self.root, f"extra{i}")

    def cfg(self, name="next", **research):
        cfg = dict(experiment=dict(id=name), validation=dict(partition="oof"), research=research)
        path = self.root / "configs" / (name + ".yaml")
        path.write_text(yaml.safe_dump(cfg))
        return cfg, path

    def pause(self):
        for i in range(3):
            registration = register_decision_rule(rule_document(f"fail-family-{i}"), project=self.project, task_spec=self.memory.spec)
            cfg, _ = self.cfg(f"failed{i}", decision_rule_id=registration["rule_id"])
            reserve_comparison_slot(self.project, self.memory.spec, cfg)
            make_submission_evidence(self.root, f"failed{i}", labels=[0]*40, cfg=cfg)
            evaluate_decision_rule(registration["rule"], dict(kind="full_run", id=f"failed{i}", partition="oof"),
                project=self.project, task_spec=self.memory.spec, registration=registration)
        states = StagnationController(self.project).reconcile_results()
        return states, next(iter(states.values()))

    def test_exact_five_results_count_not_actions(self):
        self.add_runs(3)
        for i in range(6):
            self.memory.record_execution("run_eda", "SUCCESS", "EDA findings")
        self.assertFalse(is_retrospective_due(self.root, self.memory.sync(), {})[0])
        make_submission_evidence(self.root, "fifth")
        self.assertTrue(is_retrospective_due(self.root, self.memory.sync(), {})[0])
        with self.assertRaisesRegex(GateViolationError, "RETROSPECTIVE_REQUIRED"):
            enforce_retrospective(self.root)

    def test_probe_failed_and_duplicate_records_do_not_advance(self):
        self.add_runs(3)
        runs = self.memory.sync()
        runs.extend([dict(exp_id="smoke", evidence_status="verified_full", run_kind="probe"),
                     dict(exp_id="failed", evidence_status="verified_full", status="FAILED"),
                     dict(exp_id="eda", evidence_status="declared"), runs[0]])
        status = retrospective_status(self.memory, runs, {})
        self.assertFalse(status["due"])
        self.assertEqual(status["runs_since_review"], 4)
        # Real invalid/probe manifests cannot enter the live evidence projection.
        for rid, key, value in (("smoke", "run_kind", "probe"), ("failed", "status", "FAILED")):
            path = make_submission_evidence(self.root, rid) / "artifact_manifest.json"
            manifest = json.loads(path.read_text())
            manifest[key] = value
            path.write_text(json.dumps(manifest))
        self.assertEqual(len(self.memory.sync()), 4)

    def test_review_persists_across_restart_and_resets_only_covered_ids(self):
        self.add_runs()
        record = record_macro_review(self.memory, review_payload(self.memory))
        fresh = load_memory(self.root)
        self.assertFalse(is_retrospective_due(self.root, fresh.sync(), {})[0])
        self.assertEqual(retrospective_status(fresh, fresh.sync(), {})["last_review_id"], record["review_id"])
        # Removal of old evidence must not discount new results.
        (self.project.artifacts_dir / "control/oof_predictions.csv").write_text("tamper")
        for i in range(5):
            make_submission_evidence(self.root, f"new{i}")
        self.assertTrue(is_retrospective_due(self.root, fresh.sync(), {})[0])
        self.assertEqual(retrospective_status(fresh, fresh.sync(), {})["runs_since_review"], 5)

    def test_stagnation_requires_review_before_cadence_and_review_does_not_resume(self):
        states, _ = self.pause()
        self.assertEqual(len(self.memory.sync()), 4)
        due, reason = is_retrospective_due(self.root, self.memory.sync(), states)
        self.assertTrue(due)
        self.assertIn("stagnation", reason)
        record_macro_review(self.memory, review_payload(self.memory))
        self.assertFalse(is_retrospective_due(self.root, self.memory.sync(), states)[0])
        self.assertIsNotNone(StagnationController(self.project).check_action("dispatch"))
        changed = {key: {**state, "generation":state["generation"] + 1} for key, state in states.items()}
        self.assertTrue(is_retrospective_due(self.root, self.memory.sync(), changed)[0])

    def test_unregistered_recovery_flag_cannot_bypass(self):
        self.add_runs()
        for research in ({}, dict(recovery_probe_id="invented", recovery_probe=True)):
            cfg, _ = self.cfg(**research)
            with self.assertRaises(GateViolationError):
                enforce_retrospective(self.root, cfg=cfg, recovery_probe=True)

    def test_registered_recovery_is_bounded_and_never_authorizes_submission(self):
        _, state = self.pause()
        registration = register_decision_rule(rule_document("recovery"), project=self.project, task_spec=self.memory.spec)
        controller = StagnationController(self.project)
        probe = controller.register_recovery_probe(comparison_identity_sha256=state["comparison_identity_sha256"],
            rule_id=registration["rule_id"], diagnosis="Systematic error slice", mechanism="Orthogonal signal",
            stop_rule="One result", max_runs=1)
        cfg, _ = self.cfg("recovery", recovery_probe_id=probe["id"], decision_rule_id=registration["rule_id"])
        enforce_retrospective(self.root, cfg=cfg, recovery_probe=True)
        # Idempotent across Bridge -> CloudJobRunner -> runner boundaries.
        enforce_retrospective(self.root, cfg=cfg, recovery_probe=True)
        with self.assertRaises(GateViolationError):
            enforce_retrospective(self.root, cfg={**cfg, "experiment":{"id":"extra"}}, recovery_probe=True)
        with self.assertRaises(GateViolationError):
            enforce_retrospective(self.root, cfg=cfg, recovery_probe=False)
        with self.assertRaises(GateViolationError):
            enforce_retrospective(self.root, cfg={**cfg, "research":{**cfg["research"], "decision_rule_id":"wrong"}}, recovery_probe=True)

    def test_direct_runner_blocks_before_data_load_but_smoke_remains_available(self):
        self.add_runs()
        cfg, path = self.cfg()
        args = build_parser().parse_args(["--config", str(path), "--project-root", str(self.root)])
        runner = UniversalRunner(cfg, str(path), args)
        with mock.patch("harness.runner.GroundingGate.verify_config", return_value=(None, None, {})), \
             mock.patch("harness.runner.GroundingGate.write_receipt"), \
             mock.patch.object(runner, "_load_training_data", side_effect=RuntimeError("reached training")) as load:
            with self.assertRaises(GateViolationError):
                runner.run()
            load.assert_not_called()
        cfg, path = self.cfg("smoke")
        args = build_parser().parse_args(["--config", str(path), "--project-root", str(self.root), "--debug", "--fold", "0"])
        runner = UniversalRunner(cfg, str(path), args)
        with mock.patch("harness.runner.GroundingGate.verify_config", return_value=(None, None, {})), \
             mock.patch("harness.runner.GroundingGate.write_receipt"), \
             mock.patch.object(runner, "_load_training_data", side_effect=RuntimeError("reached training")):
            with self.assertRaisesRegex(RuntimeError, "reached training"):
                runner.run()

    def test_cloud_jobs_block_before_provider_or_subprocess(self):
        self.add_runs()
        from harness.cloud_job_runner import CloudJobManager
        _, path = self.cfg()
        manager = CloudJobManager(self.root)
        with mock.patch("harness.cloud_policy.provider_preflight") as provider, \
             mock.patch("harness.cloud_job_runner.subprocess.Popen") as popen:
            for backend in ("local", "kaggle", "modal"):
                with self.assertRaises(GateViolationError):
                    manager.start_job(backend, str(path))
            provider.assert_not_called()
            popen.assert_not_called()
        self.assertFalse(manager._jobs)
        script = self.root / "script.py"
        script.write_text("print('ordinary script')")
        with self.assertRaises(GateViolationError):
            manager.start_job("local", str(script), extra_args=["--debug"])
        with self.assertRaisesRegex(ValueError, "locally"):
            manager.start_job("kaggle", str(path), extra_args=["--debug"])

    def test_remote_capsule_dispatch_cannot_bypass(self):
        self.add_runs()
        from harness.decision_rules import prepare_remote_research
        _, path = self.cfg()
        with self.assertRaises(GateViolationError):
            prepare_remote_research(self.project, path)

    def test_submission_gate_blocks_before_payload_or_outbound_work(self):
        self.add_runs()
        gate = SubmissionGate(self.root)
        with mock.patch.object(gate, "_resolve_payload") as payload:
            with self.assertRaises(GateViolationError):
                gate.evaluate("comp", "unused.csv", user_approved=True)
            payload.assert_not_called()

    def test_bridge_map_status_review_and_context_after_restart(self):
        self.add_runs()
        bridge = AgentBridge(self.root)
        _, obs = bridge.execute_action("get_macro_hypothesis_map", {})
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        _, status = bridge.execute_action("check_retrospective_status", {})
        self.assertTrue(status.data["due"])
        context = bridge._research_memory().context()
        self.assertIn("RETROSPECTIVE REVIEW DUE", "\n".join(_research_continuity_hud_lines(context)))
        _, obs = bridge.execute_action("record_macro_review", review_payload(self.memory))
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertFalse(AgentBridge(self.root).execute_action("check_retrospective_status", {})[1].data["due"])

    def test_cli_flat_flags_and_json_payload_record_reviews(self):
        from tools.agent_bridge_cli import main
        for payload_as_json in (False, True):
            payload = review_payload(self.memory)
            payload.pop("session_id")  # The active Bridge session owns this field.
            argv = ["--project-root", str(self.root), "act", "--action", "record_macro_review"]
            if payload_as_json:
                argv += ["--params", json.dumps(payload)]
            else:
                for key, value in payload.items():
                    argv += ["--" + key.replace("_", "-"), json.dumps(value) if isinstance(value, (list, dict)) else str(value)]
            out = io.StringIO()
            with mock.patch("harness.toolchain.ensure_pinned_toolchain", return_value=None), contextlib.redirect_stdout(out):
                code = main(argv)
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(json.loads(out.getvalue())["observation"]["status"], "SUCCESS", out.getvalue())

    def test_session_start_cli_elevates_due_macro_map(self):
        from tools.agent_bridge_cli import main
        self.add_runs()
        out = io.StringIO()
        with mock.patch("harness.toolchain.ensure_pinned_toolchain", return_value=None), contextlib.redirect_stdout(out):
            main(["--project-root", str(self.root), "start-session", "--goal", "Review campaign"])
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["retrospective_status"]["due"])
        self.assertIn("target", payload["macro_hypothesis_map"]["families"])

    def test_bridge_blocks_costly_actions_and_allows_research_actions(self):
        self.add_runs()
        bridge = AgentBridge(self.root)
        _, path = self.cfg()
        session = bridge.start_session(goal="Review campaign")
        for action in ("run_full_validation", "prepare_submission"):
            with self.assertRaises(GateViolationError):
                bridge._check_budget_and_stop_guards(session, action, {"config_path":str(path)})
        for action in ("record_macro_review", "get_macro_hypothesis_map", "check_retrospective_status", "run_smoke_test", "run_eda"):
            self.assertIsNone(bridge._check_budget_and_stop_guards(session, action, {}))


if __name__ == "__main__":
    unittest.main()
