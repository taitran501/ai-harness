import ssl
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from harness.llm_engine import _safe_urlopen
from harness.resource_ledger import reserve, terminal


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
        from unittest.mock import Mock

        import yaml

        from harness.agent_bridge import AgentBridge
        from tests.test_correctness_contracts import task
        from tools.dashboard_server import DashboardHandler

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
        import concurrent.futures

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
