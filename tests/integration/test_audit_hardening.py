"""Adversarial regressions for contract, evidence, import and dispatch boundaries."""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.artifacts import (
    CANONICAL_GENERIC_SCORER_HASHES, contract_identity, digest, seal_file_map, validate_run,
)
from harness.bridge_actions import normalize_action_params
from harness.cli import _set_backend_command
from harness.evaluation import PredictionBatch
from harness.hook_ref import isolated_metrics_hook
from harness.oof_revalidation import score_predictions
from harness.research_memory import ResearchMemory
from harness.runner import resolve_cli_backend
from harness.task_spec import UniversalTaskSpec
from harness.templates.modal_template import ModalAppManager
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace, task_document


class AuditHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc = task_document()
        self.spec = UniversalTaskSpec.from_dict(self.doc)

    def workspace(self, doc=None):
        return make_workspace(root_path=self.root / "workspace", task=doc or self.doc)

    def sealed(self):
        run = self.root / "artifacts" / "exp1"
        seal_fixture(run, self.spec)
        return run

    def manifest(self, run, **updates):
        path = run / "artifact_manifest.json"
        value = json.loads(path.read_text())
        value.update(updates)
        path.write_text(json.dumps(value))
        return value

    def test_backend_updates_preserve_wrapped_and_flat_documents(self):
        for wrapped in (True, False):
            for bridge_action in (True, False):
                with self.subTest(wrapped=wrapped, bridge=bridge_action):
                    doc = copy.deepcopy(self.doc)
                    doc["validation"] = {"strategy": "holdout", "custom": [1, 2]}
                    doc["notes"] = {"owner": "audit"}
                    if not wrapped:
                        doc = {**doc["task"], "notes": doc["notes"]}
                    root, path = self.workspace(doc)
                    if bridge_action:
                        _, obs = AgentBridge(root).execute_action("update_task_contract",
                            params={"backend": "modal", "locked": "false"})
                        self.assertEqual(obs.status, "SUCCESS", obs.summary)
                    else:
                        self.assertEqual(_set_backend_command(argparse.Namespace(
                            project_root=str(root), backend="modal", lock=False, unlock=True)), 0)
                    expected = copy.deepcopy(doc)
                    expected.get("task", expected)["execution"] = {"backend": "modal", "locked": False}
                    self.assertEqual(yaml.safe_load(path.read_text()), expected)

    def test_flat_and_structured_prediction_ids_must_match(self):
        for ids in ([2, 1], [1], [1, 2, 3]):
            for payload in ({"target": [0, 1]}, {"labels": {"target": [0, 1]}}):
                with self.subTest(ids=ids, payload=payload), self.assertRaises(ValueError):
                    PredictionBatch.from_mapping({"ids": ids, **payload}, [1, 2])
        batch = PredictionBatch.from_mapping({"ids": [1, 2], "target": [0, 1]}, [1, 2])
        self.assertEqual(batch.labels, {"target": [0, 1]})

    def test_scorer_compatibility_requires_both_hashes_recognized(self):
        run = self.sealed()
        known = sorted(CANONICAL_GENERIC_SCORER_HASHES)
        unknown = "unknown-custom-plugin-scorer"
        for stored, expected in ((known[0], unknown), (unknown, known[0])):
            self.manifest(run, scorer_sha256=stored)
            with self.assertRaisesRegex(ValueError, "scorer_sha256"):
                validate_run(run, expected={"scorer_sha256": expected})
        self.manifest(run, scorer_sha256=known[0])
        validate_run(run, expected={"scorer_sha256": known[1]})
        self.assertIn(contract_identity(self.spec)["scorer_sha256"], known)

    def test_backend_switch_preserves_verified_runs_including_legacy_hashes(self):
        run = self.sealed()
        before = ResearchMemory(self.root, run.parent, self.spec)
        original = before.sync()
        self.assertEqual(len(original), 1)
        null_doc = copy.deepcopy(self.doc)
        null_doc["task"]["execution"] = None
        self.assertEqual(contract_identity(self.spec), contract_identity(UniversalTaskSpec.from_dict(null_doc)))
        self.manifest(run, task_sha256=digest(null_doc["task"]))
        validate_run(run, task_spec=self.spec)
        old_meta = copy.deepcopy(self.spec.metadata)
        old_meta["execution"] = {"backend": "kaggle", "locked": True}
        self.manifest(run, task_sha256=digest(old_meta))
        edited = copy.deepcopy(self.doc)
        edited["task"]["execution"] = {"backend": "modal", "locked": False}
        spec = UniversalTaskSpec.from_dict(edited)
        self.assertEqual(contract_identity(self.spec), contract_identity(spec))
        after = ResearchMemory(self.root, run.parent, spec)
        switched = after.sync()
        self.assertEqual(len(switched), 1)
        self.assertEqual(original[0]["comparison_group"], switched[0]["comparison_group"])
        for section, value in (("primary_metric", "f1"), ("data_config", {"target_column": "other"}),
                               ("validation", {"strategy": "holdout"})):
            changed = copy.deepcopy(edited)
            changed["task"][section] = value
            changed_spec = UniversalTaskSpec.from_dict(changed)
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "task_sha256"):
                validate_run(run, task_spec=changed_spec)

    def test_nested_sealed_file_tamper_invalidates_research_cache(self):
        run = self.sealed()
        (run / "stages").mkdir()
        log = run / "stages/train.log"
        log.write_text("original")
        self.manifest(run, files=seal_file_map(run))
        memory = ResearchMemory(self.root, run.parent, self.spec)
        self.assertEqual(len(memory.sync()), 1)
        log.write_text("tampered nested file")
        self.assertEqual(memory.sync(), [])

    def test_unknown_backend_rejected_even_for_probes(self):
        for smoke in (True, False):
            for requested, contract in (("kaglle", None), (None, "kaglle"), ("local", "kaglle")):
                with self.subTest(smoke=smoke, requested=requested, contract=contract), self.assertRaises(ValueError):
                    resolve_cli_backend(requested, contract, True, smoke)

    def test_remote_flag_does_not_bypass_full_run_lock(self):
        from harness.runner import main

        doc = copy.deepcopy(self.doc)
        doc["task"]["execution"] = {"backend": "kaggle", "locked": True}
        root, _ = self.workspace(doc)
        (root / "configs/exp.yaml").write_text("experiment: {id: exp1}\nmodel: {name: logistic_regression}\n")
        cwd, path = Path.cwd(), sys.path[:]
        try:
            with patch.dict(os.environ, {"AI_HARNESS_REMOTE_KERNEL": "1"}), redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised:
                    main(["--project-root", str(root), "--config", "configs/exp.yaml", "--backend", "local"])
                self.assertNotEqual(raised.exception.code, 0)
                self.assertIn("Backend lock violation", err.getvalue())
        finally:
            os.chdir(cwd)
            sys.path[:] = path

    def test_custom_scorers_and_dependencies_isolated_across_workspaces(self):
        original_path = sys.path[:]
        previous = {k: v for k, v in sys.modules.items() if k == "audit_scorer" or k.startswith("audit_src")}
        for score in (0.25, 0.75):
            root = self.root / str(score)
            (root / "audit_src").mkdir(parents=True)
            (root / "audit_src/__init__.py").write_text("")
            (root / "audit_src/helper.py").write_text(f"SCORE = {score}\n")
            (root / "audit_scorer.py").write_text(
                "from audit_src.helper import SCORE\ndef evaluate(truth, predictions):\n    return {'accuracy': SCORE}\n")
            result = score_predictions(self.spec, None, SimpleNamespace(project_root=root),
                {"paths": {"metrics_evaluator": "audit_scorer:evaluate"}},
                pd.DataFrame({"id": [1, 2], "label": [0, 1]}), {"label": [0, 1]},
                skip_generic_base_eval=True)
            self.assertEqual(result["primary_score"], score)
            self.assertEqual(sys.path, original_path)
            self.assertEqual({k: v for k, v in sys.modules.items() if k == "audit_scorer" or k.startswith("audit_src")}, previous)
            with self.assertRaisesRegex(RuntimeError, "hook failed"):
                with isolated_metrics_hook("audit_scorer:evaluate", root):
                    raise RuntimeError("hook failed")
            self.assertEqual(sys.path, original_path)

    def test_input_normalization(self):
        for value in ("false", "0", "no", "False"):
            self.assertIs(normalize_action_params({"force": value})["force"], False)
        self.assertEqual(normalize_action_params({"open_questions": "check leakage"}),
                         {"open_questions": ["check leakage"]})
        for value in (float("inf"), float("nan"), "inf", "nan", "1e999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_action_params({"timeout_seconds": value})
        root, _ = self.workspace()
        bridge = AgentBridge(root)
        _, obs = bridge.execute_action("record_research_checkpoint", params={
            "summary": "Audit", "next_action": "Verify", "open_questions": "check leakage"})
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertEqual(obs.data["open_questions"], ["check leakage"])

    def test_cli_failed_action_has_nonzero_exit_code(self):
        from tools.agent_bridge_cli import main

        root, _ = self.workspace()
        original_path = sys.path[:]
        try:
            with patch.dict(os.environ), redirect_stdout(io.StringIO()) as output:
                code = main(["--project-root", str(root), "act", "--action", "record_research_checkpoint",
                             "--params", json.dumps({"next_action": "verify"})])
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(output.getvalue())["observation"]["status"], "FAILED")
        finally:
            sys.path[:] = original_path

    def test_allowlisted_run_outputs_are_still_tracked_past_finding_limit(self):
        from harness.hygiene import scan_workspace_hygiene

        doc = copy.deepcopy(self.doc)
        doc["task"]["hygiene"] = {"allow_dirs": ["kernel_output_a", "run_b"]}
        root, _ = self.workspace(doc)
        for name in ("aaa_unknown", "kernel_output_a", "run_b"):
            (root / name).mkdir()
        findings = scan_workspace_hygiene(root, max_findings=1)["findings"]
        runs = {f["path"]: f for f in findings if f["path"] in {"kernel_output_a", "run_b"}}
        self.assertEqual(set(runs), {"kernel_output_a", "run_b"})
        self.assertTrue(all(not f["sealed"] for f in runs.values()))

    def test_modal_pull_is_idempotent_and_never_overwrites_existing_content(self):
        run = self.sealed()
        dest = self.root / "recovered/exp1"

        def download(cmd, **kwargs):
            shutil.copytree(run, Path(cmd[5]) / "exp1")
            return Mock(returncode=0)

        def pull():
            with patch("harness.templates.modal_template.subprocess.run", side_effect=download):
                return ModalAppManager.pull_app_output("exp1", "volume", dest, {"exp_id": "exp1"})

        self.assertTrue(pull()["success"])
        before = {p.relative_to(dest): (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.rglob("*") if p.is_file()}
        self.assertTrue(pull()["success"])
        self.assertEqual(before, {p.relative_to(dest): (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.rglob("*") if p.is_file()})
        (run / "extra.txt").write_text("different downloaded contents")
        self.manifest(run, files=seal_file_map(run))
        result = pull()
        self.assertFalse(result["success"])
        self.assertIn("immutability", result["error"])
        self.assertEqual(before, {p.relative_to(dest): (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.rglob("*") if p.is_file()})

    def test_remote_research_uses_effective_config(self):
        from harness.decision_rules import prepare_remote_research
        from harness.project import HarnessProject

        root, _ = self.workspace()
        path = root / "configs/exp.yaml"
        path.write_text("experiment: {id: exp1}\n")
        cfg = {"experiment": {"id": "exp1"}, "research": {"insight_id": "inherited-insight"}}
        with patch("harness.macro_research.enforce_retrospective") as review, \
             patch("harness.stagnation.StagnationController.check_action", return_value=None) as guard:
            prepare_remote_research(HarnessProject.discover(root), path, effective_config=cfg)
        self.assertEqual(review.call_args.kwargs["cfg"], cfg)
        self.assertEqual(guard.call_args.kwargs["insight_id"], "inherited-insight")

    def test_test_tiers_and_scope_discovery_have_no_stale_entries(self):
        from tools.run_tests import SLOW_MODULES, build_suite, discover_modules

        self.assertTrue(set(SLOW_MODULES).issubset(discover_modules()))

        def ids(suite):
            for test in suite:
                if isinstance(test, unittest.TestSuite):
                    yield from ids(test)
                else:
                    yield test.id()

        discovered = list(ids(build_suite(["tests.unit.test_research_memory", "tests.unit.test_research_scope_stability"])))
        self.assertEqual(len(discovered), len(set(discovered)))
