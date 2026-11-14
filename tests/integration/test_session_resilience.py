"""Consolidated integration tests for session resilience, governance, state, and remote dispatch.

Consolidates:
- tests/agent_bridge/test_session_approval_fixes.py (all 7 classes)
- tests/test_correctness_governance.py (GovernanceContracts)
- tests/test_correctness_state.py (StateContracts)
- tests/test_job_state.py (WorkerStatePathTests)
- tests/test_llm_api_endpoints.py (TestLLMAPIEndpoints)
- tests/test_reconciliation.py (TestReconciliation)
- tests/test_remote_dispatch_hardening.py (all 7 classes)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import multiprocessing
import os
import shutil
import socketserver
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.cloud_job_runner import CloudJob, CloudJobManager
from harness.cloud_runner import _prepare_remote_run
from harness.dispatch_audit import append_dispatch_event
from harness.environment_detector import EnvironmentDetector
from harness.hygiene import scan_workspace_hygiene, task_allowed_dirs
from harness.job_state import JOB_ID_PATTERN, worker_state_path
from harness.kaggle_inspect import _resolve_exp_kernel_slug
from harness.llm_engine import LLMEngine, _safe_urlopen
from harness.project import HarnessProject, attach_project
from harness.reconciliation import (
    acknowledge_external_submissions,
    reconcile_provider_submissions,
)
from harness.resource_ledger import reserve, terminal
from harness.runner import Runner
from harness.state_store import save_snapshot, transaction
from harness.task_spec import load_active_task_spec
from harness.templates import kaggle_template, modal_template
from tests.fixtures import (
    SOURCE_ROOT,
    copy_repo_configs,
    make_workspace,
    task,
    task_document,
)
from tools.dashboard_server import DashboardHandler

if str(SOURCE_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "tools"))
import audit_gate  # noqa: E402


# Top-level helper function for multiprocessing state test
def increment(path: str, count: int) -> None:
    for _ in range(count):
        with transaction(path, {"count": 0}) as value:
            value["count"] += 1


# ===========================================================================
# 1. Session Approval Fixes (from test_session_approval_fixes.py)
# ===========================================================================

class TestBudgetBlockDoesNotTerminateSession(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_a3_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_blocked_action_keeps_session_active(self) -> None:
        sess = self.bridge.start_session(
            goal="budget test",
            budget={"max_experiments": 0},
        )
        _, obs = self.bridge.execute_action(
            action="run_full_validation",
            params={"config_path": "configs/does_not_exist.yaml"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "BLOCKED_BY_BUDGET")
        reloaded = self.bridge.load_session(sess.session_id)
        self.assertEqual(reloaded.status, "ACTIVE")
        self.assertIn(
            "BLOCKED_BY_BUDGET",
            [o["status"] for o in reloaded.observations],
        )
        self.assertNotIn(
            "run_full_validation",
            [a["action"] for a in reloaded.actions],
        )

    def test_blocked_action_does_not_poison_active_session_pick(self) -> None:
        sess = self.bridge.start_session(
            goal="budget test",
            budget={"max_experiments": 0},
        )
        self.bridge.execute_action(
            action="run_full_validation",
            params={"config_path": "configs/does_not_exist.yaml"},
            session_id=sess.session_id,
        )
        sess2, _ = self.bridge.execute_action(
            action="get_next_experiment_id",
            params={},
            goal="budget test",
        )
        self.assertEqual(sess2.session_id, sess.session_id)


class TestPendingApprovalNotDroppedByOtherAction(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_c1_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_inspect_while_pending_keeps_approval(self) -> None:
        sess = self.bridge.start_session(goal="approval test")
        _, obs = self.bridge.execute_action(
            action="prepare_submission",
            params={"exp_id": "baseline"},
            session_id=sess.session_id,
            user_approved=False,
        )
        self.assertEqual(obs.status, "PAUSED_FOR_APPROVAL")
        pending = self.bridge.load_session(sess.session_id).pending_approval_action
        self.assertIsNotNone(pending)

        _, obs2 = self.bridge.execute_action(
            action="get_next_experiment_id",
            params={},
            session_id=sess.session_id,
        )
        self.assertEqual(obs2.status, "SUCCESS")

        reloaded = self.bridge.load_session(sess.session_id)
        self.assertIsNotNone(reloaded.pending_approval_action)
        self.assertEqual(
            reloaded.pending_approval_action["action"], "prepare_submission"
        )
        self.assertEqual(reloaded.status, "PAUSED_FOR_APPROVAL")

    def test_approving_after_other_action_executes_pending(self) -> None:
        sess = self.bridge.start_session(goal="approval test")
        self.bridge.execute_action(
            action="prepare_submission",
            params={"exp_id": "baseline"},
            session_id=sess.session_id,
            user_approved=False,
        )
        self.bridge.execute_action(
            action="get_next_experiment_id", params={}, session_id=sess.session_id
        )
        sess3, obs = self.bridge.approve_pending_action(sess.session_id)
        self.assertEqual(obs.action, "prepare_submission")


class TestFailedFullValidationDoesNotConsumeBudget(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_a4_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_failed_run_terminal_row_counts_but_session_recovers(self) -> None:
        sess = self.bridge.start_session(
            goal="fail test",
            budget={"max_experiments": 1},
        )
        with patch.object(
            self.bridge, "_run_subprocess_safe", return_value=(False, 1, "boom", 0.01)
        ):
            _, obs = self.bridge.execute_action(
                action="run_full_validation",
                params={"config_path": "configs/exp001_feat_feature_pipeline.yaml"},
                session_id=sess.session_id,
            )
        self.assertEqual(obs.status, "FAILED")
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger = json.loads(ledger_path.read_text())
        terminal_rows = [
            r for r in ledger["reservations"].values()
            if r["status"] in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        self.assertGreaterEqual(len(terminal_rows), 1)
        reloaded = self.bridge.load_session(sess.session_id)
        self.assertGreaterEqual(reloaded.compute_usage.get("experiments_used", 0), 1)
        self.assertEqual(reloaded.status, "ACTIVE")


class TestCloudJobGpuHoursAccounting(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_b4_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_terminal_row_carries_gpu_hours_for_modal(self) -> None:
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        token = "tok_gpu_test"
        with transaction(ledger_path, {"reservations": {}}) as ledger:
            ledger["reservations"][token] = {
                "session_id": "session_0001",
                "category": "training",
                "backend": "modal",
                "status": "RESERVED",
                "reserved_seconds": None,
                "wall_seconds": None,
                "gpu_hours": None,
                "job_id": "job_modal_1",
            }
        result = terminal(
            ledger_path, token, "COMPLETED", 3600.0,
            gpu_count=1, allocation_seconds=3600.0,
        )
        self.assertTrue(result)
        ledger = json.loads(ledger_path.read_text())
        row = ledger["reservations"][token]
        self.assertEqual(row["status"], "COMPLETED")
        self.assertIsNotNone(row["gpu_hours"])
        self.assertAlmostEqual(row["gpu_hours"], 1.0, places=3)

    def test_reconcile_gpu_hours_not_none_when_rows_have_values(self) -> None:
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with transaction(ledger_path, {"reservations": {}}) as ledger:
            ledger["reservations"]["tok1"] = {
                "session_id": "session_0001",
                "category": "training",
                "backend": "modal",
                "status": "COMPLETED",
                "wall_seconds": 3600.0,
                "gpu_hours": 0.5,
            }
        sess = self.bridge.start_session(goal="gpu test")
        sess.session_id = "session_0001"
        self.bridge._reconcile_session_usage(sess)
        self.assertIsNotNone(sess.compute_usage.get("gpu_hours_used"))
        self.assertAlmostEqual(sess.compute_usage["gpu_hours_used"], 0.5, places=3)


class TestActiveSessionSelectionStable(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_a1_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_start_session_marks_active_and_demotes_others(self) -> None:
        s1 = self.bridge.start_session(goal="first")
        s2 = self.bridge.start_session(goal="second")
        reloaded2 = self.bridge.load_session(s2.session_id)
        self.assertTrue(reloaded2.active)
        reloaded1 = self.bridge.load_session(s1.session_id)
        self.assertFalse(reloaded1.active)

    def test_get_active_or_create_prefers_active_flag(self) -> None:
        s1 = self.bridge.start_session(goal="first")
        s2 = self.bridge.start_session(goal="second")
        s1.active = True
        s2.active = False
        self.bridge.save_session(s2)
        self.bridge.save_session(s1)

        picked, _ = self.bridge.execute_action(
            action="get_next_experiment_id", params={},
        )
        self.assertEqual(picked.session_id, s1.session_id)


class TestApproveDoesNotDuplicateAction(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_c2_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_approve_removes_pending_from_actions(self) -> None:
        sess = self.bridge.start_session(goal="c2 test")
        self.bridge.execute_action(
            action="prepare_submission",
            params={"exp_id": "baseline"},
            session_id=sess.session_id,
            user_approved=False,
        )
        self.bridge.approve_pending_action(sess.session_id)
        reloaded = self.bridge.load_session(sess.session_id)
        count = sum(1 for a in reloaded.actions if a["action"] == "prepare_submission")
        self.assertLessEqual(count, 1)


class TestFreezeCheckNotStale(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_b1_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_reserve_blocks_when_state_file_says_frozen(self) -> None:
        state_file = self.bridge.artifacts_dir / "competition_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "gates": {"freeze_locked": True},
            "candidate_pool": {"synthesized_candidates": [], "pinned_candidates": []},
        }
        state_file.write_text(json.dumps(state), encoding="utf-8")
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        with self.assertRaises(ValueError) as ctx:
            reserve(
                ledger_path, "session_0001", "run_full_validation",
                "local", {"max_experiments": 5},
            )
        self.assertIn("frozen", str(ctx.exception).lower())


# ===========================================================================
# 2. Correctness Governance (from test_correctness_governance.py)
# ===========================================================================

class GovernanceContracts(unittest.TestCase):
    def test_tls_failure_never_retries_unverified(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("certificate verify failed"),
        ) as call:
            with self.assertRaises(urllib.error.URLError):
                _safe_urlopen("https://example.invalid")
            self.assertEqual(call.call_count, 1)
            self.assertEqual(
                call.call_args.kwargs["context"].verify_mode, ssl.CERT_REQUIRED
            )

    def test_reservation_and_terminal_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            token, _ = reserve(
                path, "session", "run_full_validation", "local", {"max_experiments": 1}
            )
            with self.assertRaises(ValueError):
                reserve(
                    path,
                    "session",
                    "run_full_validation",
                    "local",
                    {"max_experiments": 1},
                )
            self.assertTrue(terminal(path, token, "FAILED", 2))
            self.assertFalse(terminal(path, token, "COMPLETED", 5))
            reserve(
                path, "session", "promote_candidate", "local", {"max_experiments": 1}
            )

    def test_bridge_and_dashboard_share_hard_budget_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/task.yaml").write_text(
                yaml.safe_dump({"task": task().metadata})
            )
            params = {
                "config_path": "missing.yaml",
                "backend": "kaggle",
                "budget": {"max_wall_seconds": 10},
            }
            _, direct = AgentBridge(root).execute_action(
                "run_full_validation", params=dict(params), budget=params["budget"]
            )
            self.assertEqual(direct.status, "BLOCKED_BY_BUDGET")
            handler = object.__new__(DashboardHandler)
            handler.send_json = Mock()
            with patch("tools.dashboard_server.PROJECT_ROOT", root):
                handler.handle_api_cloud_run(params)
            response = handler.send_json.call_args.args[0]
            self.assertEqual(response["observation"]["status"], direct.status)

    def test_concurrent_reservations_do_not_exceed_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"

            def claim(_):
                try:
                    reserve(
                        path,
                        "session",
                        "run_full_validation",
                        "local",
                        {"max_experiments": 1},
                    )
                    return True
                except ValueError:
                    return False

            with concurrent.futures.ThreadPoolExecutor(5) as workers:
                self.assertEqual(sum(workers.map(claim, range(5))), 1)


# ===========================================================================
# 3. Correctness State (from test_correctness_state.py)
# ===========================================================================

class StateContracts(unittest.TestCase):
    def test_threads_and_processes_preserve_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "state.json")
            with concurrent.futures.ThreadPoolExecutor(4) as executor:
                list(executor.map(lambda _: increment(path, 10), range(4)))
            workers = []
            for _ in range(3):
                worker = multiprocessing.Process(target=increment, args=(path, 10))
                try:
                    worker.start()
                except Exception:
                    worker = multiprocessing.Process(target=increment, args=(path, 10))
                    worker.start()
                workers.append(worker)
            for worker in workers:
                worker.join(30)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(json.loads(Path(path).read_text())["count"], 70)

    def test_snapshot_preserves_unrelated_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            base = {"events": [], "gates": {"a": False, "b": False}}
            save_snapshot(path, {}, base)
            save_snapshot(
                path, base, {"events": ["one"], "gates": {"a": True, "b": False}}
            )
            result = save_snapshot(
                path, base, {"events": ["two"], "gates": {"a": False, "b": True}}
            )
            self.assertEqual(set(result["events"]), {"one", "two"})
            self.assertEqual(result["gates"], {"a": True, "b": True})

    def test_task_cache_tracks_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            doc = {"task": task().metadata}
            path.write_text(yaml.safe_dump(doc))
            first = load_active_task_spec(path)
            doc["task"]["name"] = "changed"
            path.write_text(yaml.safe_dump(doc))
            self.assertNotEqual(first.name, load_active_task_spec(path).name)


# ===========================================================================
# 4. Job State (from test_job_state.py)
# ===========================================================================

class WorkerStatePathTests(unittest.TestCase):
    def setUp(self):
        self.root: Path = make_workspace()[0]
        self.project = HarnessProject.discover(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_job_id_resolves_inside_artifacts_dir(self):
        path = worker_state_path(self.project, "job_exp001_full")
        self.assertEqual(path.name, "job_exp001_full.worker.json")
        self.assertIn(str(self.project.artifacts_dir), str(path))

    def test_invalid_job_id_error_names_id_pattern_and_example(self):
        with self.assertRaises(ValueError) as ctx:
            worker_state_path(self.project, "test_job")
        message = str(ctx.exception)
        self.assertIn("test_job", message)
        self.assertIn(JOB_ID_PATTERN.pattern, message)
        self.assertIn("job_exp001_full", message)

    def test_invalid_job_id_rejects_space_and_wrong_prefix(self):
        for bad in ("job ", "JOB_x", "x_job_1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    worker_state_path(self.project, bad)


# ===========================================================================
# 5. LLM API Endpoints (from test_llm_api_endpoints.py)
# ===========================================================================

class TestLLMAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 0
        cls._class_patches = [
            patch("harness.llm_engine.LLMEngine.check_local_endpoint", return_value=False),
            patch("harness.llm_engine.LLMEngine.fetch_hf_models", return_value=LLMEngine.CURATED_HF_FALLBACK_MODELS[:5]),
            patch("harness.environment_detector.EnvironmentDetector.verify_llm_api_key", return_value={
                "success": True, "valid": False, "provider": "openai", "error": "Invalid API key"
            }),
        ]
        for p in cls._class_patches:
            p.start()

        socketserver.TCPServer.allow_reuse_address = True
        cls.httpd = socketserver.ThreadingTCPServer(("", cls.port), DashboardHandler)
        cls.port = cls.httpd.server_address[1]
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        for p in reversed(cls._class_patches):
            p.stop()

    def test_get_llm_providers(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/providers"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("providers", data)
            self.assertIsInstance(data["providers"], list)
            provider_ids = [p["id"] for p in data["providers"]]
            self.assertIn("gemini", provider_ids)
            self.assertIn("openai", provider_ids)
            self.assertIn("deepseek", provider_ids)
            self.assertIn("groq", provider_ids)
            self.assertIn("openrouter", provider_ids)

    def test_get_llm_sample_data(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/sample_data"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("columns", data)
            self.assertIn("rows", data)
            self.assertIsInstance(data["columns"], list)
            self.assertIsInstance(data["rows"], list)

    def test_post_llm_generate_peft_script(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/generate_peft_script"
        payload = {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "lora_r": 16,
            "lora_alpha": 32,
            "save_to_disk": False,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["success"])
            self.assertIn("LoraConfig", res["script_code"])
            self.assertIn("SFTTrainer", res["script_code"])

    def test_post_save_llm_key(self):
        url = f"http://127.0.0.1:{self.port}/api/environment/llm/key"
        payload = {
            "key_name": "TEST_LLM_API_KEY",
            "key_value": "dummy-key-for-test-12345",
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["success"])

    def test_get_llm_hf_models(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/hf_models?search=qwen&limit=5"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("models", data)
            self.assertIsInstance(data["models"], list)

    def test_get_llm_unsloth_models(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/unsloth_models?limit=10"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("models", data)
            self.assertIsInstance(data["models"], list)

    def test_get_llm_provider_models(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/provider_models?provider=gemini"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["provider"], "gemini")
            self.assertIn("models", data)
            self.assertIsInstance(data["models"], list)
            self.assertGreater(len(data["models"]), 0)

    def test_post_llm_generate_unsloth_script(self):
        url = f"http://127.0.0.1:{self.port}/api/llm/generate_unsloth_script"
        payload = {
            "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "lora_r": 16,
            "lora_alpha": 32,
            "save_to_disk": False,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["success"])
            self.assertIn("FastLanguageModel", res["script_code"])
            self.assertIn("unsloth", res["script_code"])

    def test_post_verify_llm_key_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/api/environment/llm/verify"
        payload = {
            "provider": "openai",
            "key_value": "sk-dummy-test-key-12345",
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["success"])
            self.assertFalse(res["valid"])
            self.assertEqual(res["provider"], "openai")
            self.assertIn("error", res)

    def test_environment_detector_providers_status(self):
        status = EnvironmentDetector.detect_llm_status()
        self.assertIn("providers", status)
        self.assertIn("total_configured", status)
        self.assertIn("gemini", status["providers"])
        self.assertIn("openai", status["providers"])
        self.assertIn("anthropic", status["providers"])
        self.assertIn("deepseek", status["providers"])
        self.assertIn("groq", status["providers"])
        self.assertIn("openrouter", status["providers"])
        self.assertIn("ollama", status["providers"])
        self.assertIn("vllm", status["providers"])
        self.assertTrue(status["providers"]["ollama"]["is_local"])
        self.assertFalse(status["providers"]["openai"]["is_local"])


# ===========================================================================
# 6. Reconciliation (from test_reconciliation.py)
# ===========================================================================

class MockKaggleClient:
    def __init__(self, submissions):
        self._subs = submissions

    def competition_submissions(self, competition):
        return self._subs


class FailingKaggleClient:
    def competition_submissions(self, competition):
        raise RuntimeError("provider request failed")


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.root, self.spec_path = make_workspace(prefix="aih-test-rec-")
        self.addCleanup(shutil.rmtree, self.root)
        self.project = attach_project(self.root, task_config=self.spec_path)

    def test_reconciliation_detects_unrecorded_external_submission(self):
        append_dispatch_event(
            self.root,
            {
                "schema_version": 1,
                "event_id": "event-1",
                "intent_id": "intent-1",
                "event": "submitted",
                "competition": "test-comp",
                "payload_sha256": "a" * 64,
                "prediction_sha256": "b" * 64,
                "history_snapshot_sha256": "c" * 64,
                "session_id": "session-1",
                "receipt_id": "receipt-1",
                "error_code": None,
                "provider_ref": "1001",
                "timestamp": "2026-10-26T12:00:00Z",
            },
        )

        mock_subs = [
            SimpleNamespace(ref="1001", date="2026-10-26", description="legit", fileName="sub1.csv"),
            SimpleNamespace(ref="1002", date="2026-10-26", description="backdoor", fileName="sub2.csv"),
        ]
        client = MockKaggleClient(mock_subs)

        report = reconcile_provider_submissions(
            self.root, competition="test-comp", provider="kaggle", provider_client=client
        )
        self.assertEqual(report["status"], "DISCREPANCY_DETECTED")
        self.assertEqual(len(report["unexplained_submissions"]), 1)
        self.assertEqual(str(report["unexplained_submissions"][0]["ref"]), "1002")

        ack_report = acknowledge_external_submissions(
            self.root, refs=["1002"], reason="Operator emergency submission", operator="admin"
        )
        self.assertIsNotNone(ack_report)
        rec_after = reconcile_provider_submissions(
            self.root, competition="test-comp", provider="kaggle", provider_client=client
        )
        self.assertEqual(rec_after["status"], "CLEAN")
        self.assertEqual(len(rec_after["unexplained_submissions"]), 0)

    def test_provider_query_failure_is_unknown_with_null_count(self):
        report = reconcile_provider_submissions(
            self.root,
            competition="test-comp",
            provider_client=FailingKaggleClient(),
        )

        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["reason_code"], "PROVIDER_UNAVAILABLE")
        self.assertIsNone(report["provider_submissions_count"])

    def test_unavailable_provider_does_not_report_clean(self):
        report = reconcile_provider_submissions(
            self.root,
            competition="test-comp",
            provider="unsupported-provider",
        )

        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["reason_code"], "PROVIDER_UNAVAILABLE")
        self.assertIsNone(report["provider_submissions_count"])

    def test_unavailable_provider_client_does_not_report_clean(self):
        with patch("harness.kaggle_compat.KaggleClient", side_effect=RuntimeError("client unavailable")):
            report = reconcile_provider_submissions(
                self.root,
                competition="test-comp",
                provider_client=None,
            )

        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["reason_code"], "PROVIDER_UNAVAILABLE")
        self.assertIsNone(report["provider_submissions_count"])

    def test_successful_empty_provider_query_is_clean(self):
        report = reconcile_provider_submissions(
            self.root,
            competition="test-comp",
            provider_client=MockKaggleClient([]),
        )

        self.assertEqual(report["status"], "CLEAN")
        self.assertEqual(report["provider_submissions_count"], 0)
        self.assertIsNone(report.get("reason_code"))


# ===========================================================================
# 7. Remote Dispatch Hardening (from test_remote_dispatch_hardening.py)
# ===========================================================================

class TestPidRunningNonlethal(unittest.TestCase):
    def test_liveness_check_does_not_kill_the_process(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            self.assertTrue(CloudJobManager._pid_running(proc.pid))
            self.assertTrue(CloudJobManager._pid_running(proc.pid))
            self.assertIsNone(proc.poll(), "liveness probe killed the worker")
        finally:
            proc.kill()
            proc.wait()

    def test_dead_pid_reports_false(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.assertFalse(CloudJobManager._pid_running(proc.pid))
        self.assertFalse(CloudJobManager._pid_running(None))
        self.assertFalse(CloudJobManager._pid_running(0))


class TestRemoteKernelReentrancyGuard(unittest.TestCase):
    def test_prepare_remote_run_refuses_reentry(self) -> None:
        os.environ["AI_HARNESS_REMOTE_KERNEL"] = "1"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                _prepare_remote_run("configs/does-not-matter.yaml", "kaggle")
            self.assertIn("already inside a dispatched remote kernel", str(ctx.exception))
        finally:
            del os.environ["AI_HARNESS_REMOTE_KERNEL"]

    def test_runner_forces_local_under_remote_kernel_flag(self) -> None:
        doc = task_document(task_id="reentry-task")
        doc["task"]["execution"] = {"backend": "kaggle", "locked": True}
        root, task_config = make_workspace(prefix="aih-reentry-", task=doc)
        try:
            (root / "configs" / "exp-guard.yaml").write_text(
                "task:\n  name: Fixture Task\nmodel:\n  name: logistic_regression\n",
                encoding="utf-8",
            )
            env = dict(os.environ, AI_HARNESS_REMOTE_KERNEL="1")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "harness.runner",
                    "--project-root",
                    str(root),
                    "--config",
                    "configs/exp-guard.yaml",
                    "--debug",
                    "--fold",
                    "0",
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(SOURCE_ROOT),
                timeout=180,
            )
            combined = result.stdout + result.stderr
            self.assertIn(
                "inside a dispatched remote kernel",
                combined,
                f"guard notice missing; output:\n{combined}",
            )
            self.assertNotIn("Backend lock violation", combined)
            self.assertNotIn("kaggle push", combined.lower())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_kaggle_template_sets_remote_kernel_flag(self) -> None:
        source = inspect.getsource(kaggle_template.KaggleKernelManager.setup_kernel_package)
        guard = 'os.environ["AI_HARNESS_REMOTE_KERNEL"] = "1"'
        self.assertIn(guard, source, "kaggle runner script must set the re-entrancy flag")
        self.assertLess(source.index(guard), source.index("harness.runner"))

    def test_modal_template_sets_remote_kernel_flag(self) -> None:
        source = inspect.getsource(modal_template.generate_modal_script)
        guard = 'os.environ["AI_HARNESS_REMOTE_KERNEL"] = "1"'
        self.assertIn(guard, source, "modal remote script must set the re-entrancy flag")
        self.assertLess(source.index(guard), source.index("harness.runner"))


class TestDeadWorkerReconciliation(unittest.TestCase):
    def test_dead_worker_without_terminal_state_is_failed(self) -> None:
        root, _ = make_workspace(prefix="aih-deadworker-")
        try:
            manager = CloudJobManager(root)
            job = CloudJob(
                job_id="job_dead_worker",
                backend="kaggle",
                config_path="configs/exp.yaml",
                exp_id="exp_dead",
                task_id="fixture-task",
                config_sha256="0" * 64,
                artifact_dir=".ai-harness/artifacts/exp_dead",
            )
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            proc.wait()
            job.pid = proc.pid
            job.status = "RUNNING"
            manager._persist_job(job)

            refreshed = CloudJobManager(root).get_job(job.job_id)
            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed["status"], "FAILED")
            self.assertIn("no longer running", refreshed["error_message"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestExpKernelSlugResolution(unittest.TestCase):
    def test_slug_comes_from_job_state_not_derivation(self) -> None:
        root, _ = make_workspace(prefix="aih-slug-")
        old_cwd = os.getcwd()
        try:
            jobs_dir = HarnessProject.discover(root).artifacts_dir / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            newer = {
                "job_id": "job_newer",
                "status": "FAILED",
                "exp_id": "exp_1",
                "remote_id": "tester/fixture-task-exp-1-abc12345",
                "started_at": 200.0,
            }
            older = {
                "job_id": "job_older",
                "status": "COMPLETED",
                "exp_id": "exp_1",
                "remote_id": "tester/fixture-task-exp-1-old99999",
                "started_at": 100.0,
            }
            (jobs_dir / "job_newer.json").write_text(json.dumps(newer))
            (jobs_dir / "job_older.json").write_text(json.dumps(older))
            os.chdir(root)
            slug = _resolve_exp_kernel_slug("exp_1")
            self.assertEqual(slug, "tester/fixture-task-exp-1-abc12345")
            self.assertIsNone(_resolve_exp_kernel_slug("exp_unknown"))
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(root, ignore_errors=True)


class TestConstantFoldColumnError(unittest.TestCase):
    def test_constant_fold_raises_actionable_error(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="aih-foldconst-"))
        try:
            task_yaml = root / "task.yaml"
            task_dict = task_document(
                task_id="fold-task",
                train_path="dataset/train.csv",
                num_folds=2,
            )
            task_dict["task"]["subtasks"] = {
                "label": {"task_type": "classification", "target_column": "label", "classes": [0, 1]}
            }
            task_yaml.write_text(yaml.safe_dump(task_dict), encoding="utf-8")
            exp_yaml = root / "exp.yaml"
            exp_dict = {
                "task": task_dict["task"],
                "model": {"name": "logistic_regression"},
                "validation": {"strategy": "kfold", "num_folds": 2},
            }
            exp_yaml.write_text(yaml.safe_dump(exp_dict), encoding="utf-8")

            args = argparse.Namespace(
                project_root=str(root),
                task_config=str(task_yaml),
                debug=True,
                probe_samples=None,
                fold=None,
                job_id="test_foldconst",
                force_import=False,
            )
            runner = Runner(exp_dict, exp_yaml, args)
            runner.task_plugin.build_validation_splits = lambda *a, **k: None

            df = pd.DataFrame(
                {
                    "id": range(10),
                    "text": [f"row {i}" for i in range(10)],
                    "label": [0, 1] * 5,
                    "fold": [0] * 10,
                }
            )
            with self.assertRaises(ValueError) as ctx:
                runner._resolve_validation_splits(df, "kfold")
            message = str(ctx.exception)
            self.assertIn("constant 'fold' column", message)
            self.assertIn("Remove the column", message)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestHygieneTaskAllowlist(unittest.TestCase):
    def test_task_spec_directories_are_not_flagged(self) -> None:
        doc = task_document()
        doc["task"]["hygiene"] = {"allowed_top_dirs": ["hf_local", "baselines"]}
        root, _ = make_workspace(prefix="aih-hygene-", task=doc)
        try:
            for name in ("hf_local", "baselines", "getting-started", "rogue_dir"):
                (root / name).mkdir()
                (root / name / "x.txt").write_text("x")
            report = scan_workspace_hygiene(root)
            flagged = {f["path"] for f in report["findings"]}
            self.assertNotIn("hf_local", flagged)
            self.assertNotIn("baselines", flagged)
            self.assertIn("getting-started", flagged)
            self.assertIn("rogue_dir", flagged)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_allowlist_cache_invalidates_on_mtime_change(self) -> None:
        doc = task_document()
        doc["task"]["hygiene"] = {"allowed_top_dirs": ["cache_a"]}
        root, task_config = make_workspace(prefix="aih-hygcache-", task=doc)
        try:
            self.assertEqual(task_allowed_dirs(root), frozenset({"cache_a"}))
            doc2 = task_document()
            doc2["task"]["hygiene"] = {"allowed_top_dirs": ["cache_b"]}
            task_config.write_text(
                yaml.safe_dump(doc2, sort_keys=False), encoding="utf-8"
            )
            future = time.time() + 5
            os.utime(task_config, (future, future))
            self.assertEqual(task_allowed_dirs(root), frozenset({"cache_b"}))
            self.assertNotIn(
                "hf_local",
                {f["path"] for f in scan_workspace_hygiene(root)["findings"]},
            )
            (root / "cache_b").mkdir()
            self.assertNotIn(
                "cache_b",
                {f["path"] for f in scan_workspace_hygiene(root)["findings"]},
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestAuditGateOofIds(unittest.TestCase):
    def _oof_df(self, rows_per_key: int = 3, duplicate_row: bool = False):
        rows = []
        for i in range(4):
            for j in range(rows_per_key):
                rows.append(
                    {
                        "id": f"case_{i:04d}",
                        "fold": j,
                        "input_variant": "problem_text",
                        "pred_robustness_prediction": bool(i % 2),
                    }
                )
        if duplicate_row:
            rows.append(dict(rows[0]))
        return pd.DataFrame(rows)

    def test_multi_variant_oof_table_passes_integrity(self) -> None:
        df = self._oof_df()
        self.assertTrue(df["id"].duplicated().any(), "test data must repeat ids")
        self.assertEqual(audit_gate.audit_prediction_integrity(df, id_col="id"), [])

    def test_true_duplicate_within_scope_still_fails(self) -> None:
        df = self._oof_df(duplicate_row=True)
        errors = audit_gate.audit_prediction_integrity(df, id_col="id")
        self.assertTrue(
            any("unique within" in e for e in errors), f"expected scope error, got {errors}"
        )

    def test_plain_table_still_requires_unique_ids(self) -> None:
        df = self._oof_df().drop(columns=["fold", "input_variant"])
        errors = audit_gate.audit_prediction_integrity(df, id_col="id")
        self.assertTrue(
            any("must be non-null and unique" in e for e in errors), f"got {errors}"
        )

    def test_diff_metrics_composite_identity(self) -> None:
        cand = self._oof_df()
        base = cand.copy()
        base["pred_robustness_prediction"] = base["pred_robustness_prediction"] ^ True
        counts, _ = audit_gate.compute_diff_metrics(cand, base, id_col="id")
        self.assertEqual(counts, len(cand))
        base2 = cand.copy()
        counts2, _ = audit_gate.compute_diff_metrics(cand, base2, id_col="id")
        self.assertEqual(counts2, 0)


if __name__ == "__main__":
    unittest.main()
