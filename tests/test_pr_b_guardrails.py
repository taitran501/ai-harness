"""PR #B regression tests — guardrails & state-contract fixes.

Each test locks in one finding from the 2026-10-16 module review wave that
was verified against the source before implementation:

- external prediction registration: canonical `journal.json` name
  (candidates.py wrote an orphaned `tree_journal.json`)
- repair: default mode never silently shrinks the declared evidence bundle;
  drop-missing mode never adopts undeclared files
- runner: probe runs (CLI `--probe-samples` OR config `validation.sample_limit`)
  must stay local even when the task contract locks a cloud backend
- kaggle_template: transient status-CLI ERRORs no longer abort the poll loop
- cloud_job_runner: one managed job == one GPU (model-name regex removed) and
  audit outputs land outside the sealed run directory
- competition_state: finalize_competition requires the freeze gate; the final
  report attributes per-slot CV/public scores instead of duplicating the
  overall best; single-fold `fold_std` is no longer NaN
- agent_bridge: FINALIZED state enforces the same stop guards as FROZEN;
  `_reconcile_session_usage` never yields gpu_hours_used=None
- llm_engine: Gemini SDK/REST failures surface instead of returning ""
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

from tests.fixtures import make_workspace as _make_workspace, task_document

_PRB_TASK = task_document(task_id="prb-fixture-task", name="PRB Fixture")

SOURCE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared fixture: minimal attached workspace (from tests/fixtures.py)
# ---------------------------------------------------------------------------


class _WorkspaceBase(unittest.TestCase):
    def setUp(self) -> None:
        self.root, self.task_config = _make_workspace(task=_PRB_TASK)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. candidates.py — canonical journal filename
# ---------------------------------------------------------------------------

class TestExternalPredictionJournalName(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_correctness_contracts import task

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "artifacts").mkdir()

        spec = task()
        doc = {"task": spec.metadata}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        (self.root / "configs" / "task.yaml").write_text(
            yaml.safe_dump(doc), encoding="utf-8"
        )
        self.n = 20
        self.train_df = pd.DataFrame(
            {
                "id": range(self.n),
                "group": range(self.n),
                "x": range(self.n),
                "category": ["a", "b"] * (self.n // 2),
                "target": [float(x * 2.0) for x in range(self.n)],
            }
        )
        self.train_df.to_csv(self.root / "dataset" / "train.csv", index=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_register_writes_canonical_journal_json(self) -> None:
        from harness.candidates import register_external_prediction

        n = self.n
        oof = pd.DataFrame(
            {
                "id": range(n),
                "fold": [i % 2 for i in range(n)],
                "pred_target": [float(x * 2.0) for x in range(n)],
            }
        )
        oof_path = self.root / "ext_oof.csv"
        oof.to_csv(oof_path, index=False)

        register_external_prediction(
            project=self.root,
            name="ext_prb_01",
            oof_path=oof_path,
            author="GM",
            source="kaggle_kernel",
        )

        artifacts_dir = self.root / "artifacts"
        journal = artifacts_dir / "journal.json"
        orphan = artifacts_dir / "tree_journal.json"
        self.assertTrue(
            journal.is_file(),
            f"expected canonical journal at {journal}",
        )
        self.assertFalse(
            orphan.exists(),
            "register_external_prediction must not write orphaned tree_journal.json",
        )
        data = json.loads(journal.read_text(encoding="utf-8"))
        nodes = data if isinstance(data, list) else data.get("nodes", [])
        self.assertTrue(
            any(isinstance(n, dict) and n.get("id") == "ext_prb_01" for n in nodes),
            "registered external run missing from canonical journal",
        )


# ---------------------------------------------------------------------------
# 2. repair.py — bundle contract
# ---------------------------------------------------------------------------

class TestRepairBundleContract(_WorkspaceBase):
    def setUp(self):
        super().setUp()
        from tests.test_repair import _seal_into, task_document
        from harness.task_spec import UniversalTaskSpec

        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.folder = _seal_into(self.root / "artifacts" / "exp_rep", self.spec, 0.8)

    def test_default_mode_keeps_missing_declared_entry(self) -> None:
        """Default repair must not silently shrink the evidence bundle.

        A deleted declared file stays declared (stale hash) so the run
        remains visibly unverifiable instead of quietly re-passing.
        """
        from harness.artifacts import validate_run
        from harness.repair import repair_run_dir

        (self.folder / "oof_predictions.csv").unlink()

        res = repair_run_dir(self.folder)

        manifest = json.loads(
            (self.folder / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("oof_predictions.csv", manifest["files"])
        self.assertEqual(res.get("dropped_missing", []), [])
        with self.assertRaises(Exception):
            validate_run(self.folder)

    def test_drop_missing_never_adopts_undeclared_files(self) -> None:
        from harness.repair import repair_run_dir

        (self.folder / "oof_predictions.csv").unlink()
        (self.folder / "stray_undeclared.txt").write_text("noise")

        res = repair_run_dir(self.folder, drop_missing=True)

        manifest = json.loads(
            (self.folder / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("oof_predictions.csv", manifest["files"])
        self.assertNotIn("stray_undeclared.txt", manifest["files"])
        self.assertIn("oof_predictions.csv", res["dropped_missing"])


# ---------------------------------------------------------------------------
# 3. runner.py — probe runs stay local
# ---------------------------------------------------------------------------

class TestProbeRunsStayLocal(unittest.TestCase):
    def _args(self, **kw) -> SimpleNamespace:
        base = {"debug": False, "fold": None, "probe_samples": None, "backend": "auto"}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_debug_is_probe(self) -> None:
        from harness.runner import is_local_only_run

        self.assertTrue(is_local_only_run(self._args(debug=True)))

    def test_fold_is_probe(self) -> None:
        from harness.runner import is_local_only_run

        self.assertTrue(is_local_only_run(self._args(fold=0)))

    def test_probe_samples_is_probe(self) -> None:
        from harness.runner import is_local_only_run

        self.assertTrue(is_local_only_run(self._args(probe_samples=200)))

    def test_config_sample_limit_is_probe(self) -> None:
        from harness.runner import is_local_only_run

        cfg = {"validation": {"sample_limit": 250}}
        self.assertTrue(is_local_only_run(self._args(), cfg))

    def test_full_run_is_not_probe(self) -> None:
        from harness.runner import is_local_only_run

        cfg = {"validation": {"num_folds": 5}}
        self.assertFalse(is_local_only_run(self._args(), cfg))
        self.assertFalse(is_local_only_run(self._args()))


# ---------------------------------------------------------------------------
# 4. kaggle_template — transient CLI errors tolerated while polling
# ---------------------------------------------------------------------------

class TestKagglePollTransientTolerance(unittest.TestCase):
    def test_transient_errors_then_completed(self) -> None:
        from harness.templates.kaggle_template import KaggleKernelManager

        statuses = [
            {"status": "ERROR"},
            {"status": "ERROR"},
            {"status": "COMPLETED", "success": True},
        ]
        with patch.object(
            KaggleKernelManager,
            "get_kernel_status",
            side_effect=statuses,
        ) as mock_status:
            result = KaggleKernelManager.wait_for_kernel(
                "exp001", poll_interval=0, timeout_seconds=30
            )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(mock_status.call_count, 3)

    def test_persistent_errors_terminate(self) -> None:
        from harness.templates.kaggle_template import KaggleKernelManager

        with patch.object(
            KaggleKernelManager,
            "get_kernel_status",
            return_value={"status": "ERROR"},
        ) as mock_status:
            result = KaggleKernelManager.wait_for_kernel(
                "exp001", poll_interval=0, timeout_seconds=60
            )
        self.assertEqual(result["status"], "ERROR")
        # bounded: max_consecutive_errors (10) polls, not 60s of polling
        self.assertEqual(mock_status.call_count, 10)

    def test_terminal_failed_returns_immediately(self) -> None:
        from harness.templates.kaggle_template import KaggleKernelManager

        with patch.object(
            KaggleKernelManager,
            "get_kernel_status",
            return_value={"status": "FAILED"},
        ) as mock_status:
            result = KaggleKernelManager.wait_for_kernel(
                "exp001", poll_interval=0, timeout_seconds=30
            )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(mock_status.call_count, 1)


# ---------------------------------------------------------------------------
# 5. cloud_job_runner — GPU accounting + audit output location
# ---------------------------------------------------------------------------

class TestManagedJobGpuAccounting(_WorkspaceBase):
    def test_one_gpu_per_managed_job(self) -> None:
        """gpu_hours must reflect 1 allocated GPU, never a model-name parse."""
        from harness.cloud_job_runner import CloudJob, CloudJobManager
        from harness.resource_ledger import reserve
        from harness.state_store import transaction

        mgr = CloudJobManager(self.root)
        ledger_path = mgr.project.artifacts_dir / "resource_ledger.json"
        token, _ = reserve(ledger_path, "sess_gpu", "run_full_validation", "kaggle", {})
        with transaction(ledger_path) as ledger:
            ledger["reservations"][token]["job_id"] = "job_gpu_acc_1"

        t0 = time.time()
        job = CloudJob(
            job_id="job_gpu_acc_1",
            backend="kaggle",
            config_path="configs/exp001.yaml",
            exp_id="exp001",
            status="COMPLETED",
            started_at=t0,
            ended_at=t0 + 3600,  # one hour
            artifact_dir=".ai-harness/artifacts/prb-fixture-task/exp001",
            gpu="A100",
        )
        mgr._persist_job(job)

        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        row = ledger["reservations"][token]
        # exactly 1 GPU for 1h wall == 1.0 gpu-hour (old regex gave 4-10x)
        self.assertAlmostEqual(row["gpu_hours"], 1.0, places=6)
        self.assertEqual(row["usage_source"], "provider_allocation")


class TestAuditOutputsOutsideRunDir(_WorkspaceBase):
    def test_audit_zip_and_receipt_outside_sealed_run_dir(self) -> None:
        from harness.cloud_job_runner import CloudJob, CloudJobManager

        mgr = CloudJobManager(self.root)
        exp_dir = self.root / ".ai-harness" / "artifacts" / "prb-fixture-task" / "exp001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "oof_predictions.csv").write_text("id,pred\n1,0\n")
        log_file = exp_dir / "run.log"
        log_file.write_text("seed log\n")

        job = CloudJob(
            job_id="job_audit_loc_1",
            backend="kaggle",
            config_path="configs/exp001.yaml",
            exp_id="exp001",
            status="COMPLETED",
            auto_audit=True,
            log_file=str(log_file),
            artifact_dir=".ai-harness/artifacts/prb-fixture-task/exp001",
        )

        import subprocess

        def fake_run(cmd, **kwargs):
            # simulate audit_gate.py writing its outputs to the requested paths
            out = {}
            for i in range(2, len(cmd) - 1, 2):
                if cmd[i].startswith("--"):
                    out[cmd[i]] = cmd[i + 1]
            Path(out["--output-zip"]).parent.mkdir(parents=True, exist_ok=True)
            Path(out["--output-zip"]).write_text("zip")
            Path(out["--receipt"]).parent.mkdir(parents=True, exist_ok=True)
            Path(out["--receipt"]).write_text("{}")
            return subprocess.CompletedProcess(cmd, 0, stdout="audit ok\n", stderr="")

        with patch("harness.cloud_job_runner.subprocess.run", side_effect=fake_run):
            mgr._handle_post_run(job)

        self.assertTrue(job.audit_passed, "audit should pass")
        self.assertNotIn(
            "candidate-audit.zip",
            [p.name for p in exp_dir.iterdir()],
            "audit zip must not be written inside the sealed run directory",
        )
        self.assertNotIn(
            "AUDIT_RECEIPT.json",
            [p.name for p in exp_dir.iterdir()],
        )
        audit_dir = (
            self.root / ".ai-harness" / "artifacts" / "audit" / "job_audit_loc_1"
        )
        self.assertTrue((audit_dir / "candidate-audit.zip").is_file())
        self.assertTrue((audit_dir / "AUDIT_RECEIPT.json").is_file())


# ---------------------------------------------------------------------------
# 6. competition_state — finalize contract + per-slot attribution
# ---------------------------------------------------------------------------

class TestFinalizeContract(_WorkspaceBase):
    def _journal(self) -> "object":
        from harness.tree_journal import Node, TreeJournal

        journal = TreeJournal(storage_path=str(self.root / "artifacts" / "journal.json"))
        journal.append(
            Node(
                verified=True,
                eligible=True,
                id="exp_safe",
                parent_id=None,
                branch="baseline",
                model_family="linear",
                atomic_hypothesis="safe logistic baseline",
                metrics={"overall_composite": 0.70, "model": "logistic"},
            )
        )
        journal.append(
            Node(
                verified=True,
                eligible=True,
                id="exp_aggr",
                parent_id="exp_safe",
                branch="ens",
                model_family="ensemble",
                atomic_hypothesis="aggressive ensemble",
                metrics={"overall_composite": 0.75, "model": "blend"},
            )
        )
        return journal

    def _sub_mgr(self) -> "object":
        from harness.submission_manager import SubmissionManager, SubmissionEntry

        mgr = SubmissionManager(submissions_dir=str(self.root / "artifacts" / "submissions"))
        mgr.submissions["sub_safe_1"] = SubmissionEntry(
            id="sub_safe_1",
            exp_id="exp_safe",
            created_at="2026-10-16 10:00:00 UTC",
            status="SCORED",
            local_cv_scores={"accuracy": 0.70},
            official_scores={"accuracy": 0.61},
            score_deltas={"accuracy": -0.09},
        )
        mgr.submissions["sub_aggr_1"] = SubmissionEntry(
            id="sub_aggr_1",
            exp_id="exp_aggr",
            created_at="2026-10-16 11:00:00 UTC",
            status="SCORED",
            local_cv_scores={"accuracy": 0.75},
            official_scores={"accuracy": 0.64},
            score_deltas={"accuracy": -0.11},
        )
        return mgr

    def test_finalize_requires_freeze(self) -> None:
        from harness.competition_state import CompetitionState

        comp = CompetitionState(project_root=self.root)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            comp.finalize_competition({}, self._journal(), self._sub_mgr())

    def test_finalize_already_finalized_idempotent(self) -> None:
        from harness.competition_state import CompetitionState

        comp = CompetitionState(project_root=self.root)
        comp.state["gates"]["freeze_locked"] = True
        comp.state["freeze_selection"].update(
            {"slot_1_safe_cv": "exp_safe", "slot_2_aggressive_lb": "exp_aggr"}
        )
        comp.save()
        comp2 = CompetitionState(project_root=self.root)
        res = comp2.finalize_competition({}, self._journal(), self._sub_mgr())
        self.assertIn("finalization", res)
        res2 = comp2.finalize_competition({}, self._journal(), self._sub_mgr())
        self.assertEqual(res2.get("status"), "already_finalized")

    def test_final_report_per_slot_attribution(self) -> None:
        from harness.competition_state import CompetitionState

        comp = CompetitionState(project_root=self.root)
        comp.state["gates"]["freeze_locked"] = True
        comp.state["freeze_selection"].update(
            {"slot_1_safe_cv": "exp_safe", "slot_2_aggressive_lb": "exp_aggr"}
        )
        res = comp.finalize_competition({}, self._journal(), self._sub_mgr())
        report = res["report_markdown"]

        slot1 = next(l for l in report.splitlines() if "Slot 1 (Primary Safe)" in l)
        slot2 = next(l for l in report.splitlines() if "Slot 2 (Secondary)" in l)
        # slot 1 shows ITS OWN cv (0.70) and public (0.61) scores
        self.assertIn("`0.7000`", slot1)
        self.assertIn("`0.6100`", slot1)
        self.assertNotIn("`0.7500`", slot1)
        # slot 2 shows ITS OWN cv (0.75) and public (0.64) scores
        self.assertIn("`0.7500`", slot2)
        self.assertIn("`0.6400`", slot2)
        self.assertNotIn("`0.7000`", slot2)
        self.assertTrue(comp.state["gates"]["finalized"])


# ---------------------------------------------------------------------------
# 7. competition_state — fold_std NaN guard
# ---------------------------------------------------------------------------

class TestFoldStdSingleRow(_WorkspaceBase):
    def test_single_fold_summary_fold_std_is_zero_not_nan(self) -> None:
        from harness.competition_state import CompetitionState
        from harness.tree_journal import Node, TreeJournal

        exp_dir = self.root / ".ai-harness" / "artifacts" / "prb-fixture-task" / "exp_f"
        exp_dir.mkdir(parents=True)
        (exp_dir / "summary.csv").write_text("score\n0.7\n", encoding="utf-8")

        journal = TreeJournal(storage_path=str(self.root / "artifacts" / "journal.json"))
        journal.append(
            Node(
                verified=True,
                eligible=True,
                id="exp_f",
                parent_id=None,
                branch="baseline",
                model_family="linear",
                metrics={"overall_composite": 0.70, "model": "logistic"},
            )
        )

        comp = CompetitionState(project_root=self.root)
        pool = comp.compute_candidate_pool(journal)
        entry = next(c for c in pool["all_candidates"] if c["id"] == "exp_f")
        self.assertEqual(entry["fold_std"], 0.0)
        # must serialize cleanly (NaN would emit invalid JSON for strict parsers)
        json.dumps(entry)


# ---------------------------------------------------------------------------
# 8. agent_bridge — FINALIZED stop gate + gpu reconcile
# ---------------------------------------------------------------------------

def _make_bridge(root: Path) -> "object":
    from harness.agent_bridge import AgentBridge

    shutil.copytree(SOURCE_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    return AgentBridge(root)


class TestFinalizedStopGate(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="prb_br_")
        self.root = Path(self._ws.name)
        self.bridge = _make_bridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def _session(self) -> SimpleNamespace:
        return SimpleNamespace(
            compute_usage={}, consecutive_rejections=0, budget={}
        )

    def test_finalized_blocks_mutation_actions(self) -> None:
        self.bridge.comp_state.state["gates"]["freeze_locked"] = True
        self.bridge.comp_state.state["gates"]["finalized"] = True
        violation = self.bridge._check_budget_and_stop_guards(
            self._session(), "run_full_validation", {}
        )
        self.assertIsNotNone(violation)
        status, message = violation
        self.assertEqual(status, "STOP_POLICY_TRIGGERED")
        self.assertIn("FINALIZED", message)

    def test_finalized_allows_inspection_actions(self) -> None:
        self.bridge.comp_state.state["gates"]["freeze_locked"] = True
        self.bridge.comp_state.state["gates"]["finalized"] = True
        violation = self.bridge._check_budget_and_stop_guards(
            self._session(), "inspect_experiment", {}
        )
        if violation is not None:
            self.assertNotEqual(violation[0], "STOP_POLICY_TRIGGERED")
        violation2 = self.bridge._check_budget_and_stop_guards(
            self._session(), "record_research_decision", {}
        )
        if violation2 is not None:
            self.assertNotEqual(violation2[0], "STOP_POLICY_TRIGGERED")

    def test_frozen_message_stays_frozen_when_only_frozen(self) -> None:
        self.bridge.comp_state.state["gates"]["freeze_locked"] = True
        self.bridge.comp_state.state["gates"]["finalized"] = False
        violation = self.bridge._check_budget_and_stop_guards(
            self._session(), "run_full_validation", {}
        )
        self.assertIsNotNone(violation)
        self.assertIn("FROZEN", violation[1])


class TestSessionGpuReconcile(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="prb_rec_")
        self.root = Path(self._ws.name)
        self.bridge = _make_bridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_terminal_rows_without_gpu_hours_yield_zero(self) -> None:

        ledger = {
            "reservations": {
                "tok1": {
                    "session_id": "s1",
                    "category": "training",
                    "backend": "kaggle",
                    "status": "COMPLETED",
                    "wall_seconds": 100.0,
                    "gpu_hours": None,
                }
            }
        }
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

        session = SimpleNamespace(session_id="s1", compute_usage={})
        self.bridge._reconcile_session_usage(session)

        self.assertEqual(session.compute_usage["gpu_hours_used"], 0.0)
        self.assertEqual(session.compute_usage["experiments_used"], 1)


# ---------------------------------------------------------------------------
# 9. llm_engine — Gemini error surfacing
# ---------------------------------------------------------------------------

class TestGeminiErrorSurfaces(unittest.TestCase):
    def test_rest_error_body_raises_not_empty_text(self) -> None:
        """When the SDK is absent and REST returns an error body, _call_gemini
        must raise — not return text='' which downstream treats as valid."""
        import harness.llm_engine as llm_mod

        class _Body:
            def read(self):
                return json.dumps(
                    {"error": {"code": 429, "message": "quota exceeded"}}
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=12.0):
            return _Body()

        with patch.dict(sys.modules, {"google": None, "google.genai": None}), patch.object(
            llm_mod, "_safe_urlopen", side_effect=fake_urlopen
        ):
            with self.assertRaises(RuntimeError) as ctx:
                llm_mod.LLMEngine._call_gemini(
                    "gemini-1.5-flash", "sys", "user", 0.0, 100, "fake-key", time.time()
                )
        self.assertIn("quota exceeded", str(ctx.exception))

    def test_sdk_empty_candidates_raises(self) -> None:
        """SDK present but returning zero candidates is a provider error; the
        REST retry must also fail (or return an error body) for the failure to
        surface — it must never come back as text=''."""
        import harness.llm_engine as llm_mod

        class _Resp:
            text = None
            candidates = []

        class _Models:
            def generate_content(self, **kwargs):
                return _Resp()

        class _Client:
            def __init__(self, api_key=None):
                self.models = _Models()

        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = _Client
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai

        class _Body:
            def read(self):
                return json.dumps(
                    {"error": {"code": 503, "message": "rest also down"}}
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch.dict(
            sys.modules, {"google": fake_google, "google.genai": fake_genai}
        ), patch.object(
            llm_mod, "_safe_urlopen", side_effect=lambda req, timeout=30.0: _Body()
        ):
            with self.assertRaises(RuntimeError) as ctx:
                llm_mod.LLMEngine._call_gemini(
                    "gemini-1.5-flash", "sys", "user", 0.0, 100, "fake-key", time.time()
                )
        self.assertIn("no candidates", str(ctx.exception).lower())
        self.assertIn("rest also down", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
