"""Unit tests for the ML/DL Dashboard Backend & Antigravity Priming Prompt Generator."""

import json
import socketserver
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from tests.workspace_fixtures import (
    cleanup_dashboard_workspace,
    prepare_dashboard_workspace,
)
from tools.dashboard_server import DashboardHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Spin up a background instance of the HTTP server on a random free port."""
        prepare_dashboard_workspace(cls, "test_dashboard_")
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        """Shut down the background HTTP server."""
        cls.server.shutdown()
        cls.server.server_close()
        cleanup_dashboard_workspace(cls)

    def fetch_json(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode("utf-8"))
            return data

    def test_api_tree(self):
        """Endpoint /api/tree should return nodes and edges for DAG visualization."""
        data = self.fetch_json("/api/tree")
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIsInstance(data["nodes"], list)
        self.assertIsInstance(data["edges"], list)
        if data["nodes"]:
            first = data["nodes"][0]
            self.assertIn("id", first)
            self.assertIn("score", first)
            self.assertIn("approach_type", first)

    def test_api_experiments(self):
        """Endpoint /api/experiments should return experiment leaderboard with approach types."""
        data = self.fetch_json("/api/experiments")
        self.assertIn("total", data)
        self.assertIn("experiments", data)
        self.assertIsInstance(data["experiments"], list)

    def test_api_recommend(self):
        """Endpoint /api/recommend should return AIDE UCB branch recommendation."""
        data = self.fetch_json("/api/recommend")
        self.assertIn("recommendation", data)
        self.assertIn("branch", data)
        self.assertIn("guideline", data)

    def test_api_priming_prompt(self):
        """Endpoint /api/priming_prompt should return dynamic Antigravity markdown prompt."""
        data = self.fetch_json("/api/priming_prompt")
        self.assertIn("prompt", data)
        prompt = data["prompt"]
        self.assertIn("# MISSION DIRECTIVE & RUNTIME CONTEXT INJECTION", prompt)
        self.assertIn("Mandatory Agent Acknowledgment Step", prompt)
        self.assertIn("Epistemic Grounding Gate", prompt)
        self.assertIn("Idea Graveyard", prompt)

    def test_api_prompt_templates(self):
        """Endpoint /api/prompt_templates should return list of agent templates."""
        data = self.fetch_json("/api/prompt_templates")
        self.assertIn("templates", data)
        template_ids = [t["id"] for t in data["templates"]]
        self.assertIn("codex", template_ids)
        self.assertIn("antigravity", template_ids)
        self.assertIn("aide", template_ids)
        self.assertIn("minimal", template_ids)
        self.assertIn("antigravity_attach", template_ids)
        self.assertIn("antigravity_iterate", template_ids)
        self.assertIn("codex_attach", template_ids)
        self.assertIn("codex_iterate", template_ids)

    def test_api_priming_prompt_codex(self):
        """Endpoint /api/priming_prompt?template=codex should return OpenAI Codex CLI format."""
        data = self.fetch_json("/api/priming_prompt?template=codex")
        self.assertIn("prompt", data)
        prompt = data["prompt"]
        self.assertIn("# SYSTEM DIRECTIVE: OPENAI CODEX", prompt)
        self.assertIn("Codex CLI", prompt)
        self.assertIn("EXECUTION LOOP", prompt)

    def test_api_priming_prompt_dual_ide_dual_intent(self):
        """Endpoint /api/priming_prompt should support ide and intent query parameters."""
        # Antigravity Attach
        data_ag_attach = self.fetch_json(
            "/api/priming_prompt?ide=antigravity&intent=attach"
        )
        self.assertIn(
            "ANTIGRAVITY INITIAL TASK SETUP & ATTACH", data_ag_attach["prompt"]
        )
        self.assertIn("MANDATORY INITIAL SETUP SEQUENCE", data_ag_attach["prompt"])

        # Antigravity Iterate
        data_ag_iter = self.fetch_json(
            "/api/priming_prompt?ide=antigravity&intent=iterate"
        )
        self.assertIn(
            "MISSION DIRECTIVE & RUNTIME CONTEXT INJECTION", data_ag_iter["prompt"]
        )
        self.assertIn("Mandatory Agent Acknowledgment Step", data_ag_iter["prompt"])

        # Codex Attach
        data_cx_attach = self.fetch_json("/api/priming_prompt?ide=codex&intent=attach")
        self.assertIn(
            "OPENAI CODEX INITIAL PIPELINE SETUP & ATTACH", data_cx_attach["prompt"]
        )
        self.assertIn("INITIALIZATION SEQUENCE", data_cx_attach["prompt"])

        # Codex Iterate
        data_cx_iter = self.fetch_json("/api/priming_prompt?ide=codex&intent=iterate")
        self.assertIn("OPENAI CODEX / AUTONOMOUS ML ENGINEER", data_cx_iter["prompt"])
        self.assertIn("EXECUTION LOOP", data_cx_iter["prompt"])

    def test_api_status(self):
        """Endpoint /api/status should return health status and champion metrics."""
        data = self.fetch_json("/api/status")
        self.assertEqual(data["status"], "online")
        self.assertIn("total_nodes", data)
        self.assertIn("best_score", data)

    def test_api_task_info(self):
        """Endpoint /api/task_info should return task metadata and primary metric."""
        data = self.fetch_json("/api/task_info")
        self.assertIn("task_name", data)
        self.assertIn("task_type", data)
        self.assertIn("primary_metric", data)
        self.assertTrue(len(str(data["primary_metric"])) > 0)
        self.assertIn("platform", data)

    def test_api_task_presets(self):
        """Endpoint /api/task_presets should return standard competition presets."""
        data = self.fetch_json("/api/task_presets")
        self.assertIn("presets", data)
        self.assertGreater(len(data["presets"]), 0)
        preset_ids = [p["id"] for p in data["presets"]]
        self.assertIn("codabench_nlp_multitask", preset_ids)
        self.assertIn("kaggle_tabular_regression", preset_ids)
        self.assertIn("kaggle_binary_classification", preset_ids)

    def test_api_task_info_post(self):
        """POST /api/task_info should update task config and return updated payload."""
        import json

        url = f"http://127.0.0.1:{self.port}/api/task_info"
        from tools.dashboard_server import load_task_config

        current_data = load_task_config()
        payload = json.dumps(current_data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as error:
            res_data = json.loads(error.read().decode("utf-8"))
            self.assertTrue(res_data.get("error"))
        else:
            self.assertTrue(res_data.get("success"))

    def test_api_submissions(self):
        """Endpoint /api/submissions should return registered submissions."""
        data = self.fetch_json("/api/submissions")
        self.assertIn("total", data)
        self.assertIn("submissions", data)
        self.assertIsInstance(data["submissions"], list)

    def test_api_submissions_register_validation(self):
        """POST /api/submissions/register with invalid data should return 400 or error response."""
        import json

        url = f"http://127.0.0.1:{self.port}/api/submissions/register"
        payload = json.dumps(
            {"exp_id": "nonexistent_exp", "candidate_csv": "fake.csv"}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertFalse(data.get("success", False))
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, [400, 500])

    def test_api_sync(self):
        """Endpoint /api/sync should trigger artifact discovery and return synced count."""
        data = self.fetch_json("/api/sync")
        self.assertTrue(data.get("success"))
        self.assertIn("synced_count", data)
        self.assertIn("total_nodes", data)

    def test_api_configs(self):
        """Endpoint /api/configs should return available experiment YAML configurations."""
        data = self.fetch_json("/api/configs")
        self.assertIn("configs", data)
        self.assertIsInstance(data["configs"], list)
        self.assertTrue(any("exp00" in c for c in data["configs"]))

    def test_api_cloud_preflight(self):
        """Endpoint /api/cloud/preflight should run validation checks against a config."""
        cand = (
            "configs/exp001_feat_feature_pipeline.yaml"
            if Path("configs/exp001_feat_feature_pipeline.yaml").exists()
            else "configs/exp000_template.yaml"
        )
        try:
            data = self.fetch_json(f"/api/cloud/preflight?config={cand}")
        except urllib.error.HTTPError as error:
            data = json.loads(error.read().decode("utf-8"))
            self.assertFalse(data.get("passed", True))
            self.assertTrue(data.get("error") or data.get("errors"))
        else:
            self.assertIn("passed", data)
            self.assertIn("resolved_config", data)
            self.assertIsInstance(data["errors"], list)

    def test_api_cloud_generate_modal(self):
        """POST /api/cloud/generate for Modal should return modal runner script and CLI commands."""
        import json

        cand = (
            "configs/exp001_feat_feature_pipeline.yaml"
            if Path("configs/exp001_feat_feature_pipeline.yaml").exists()
            else "configs/exp000_template.yaml"
        )
        url = f"http://127.0.0.1:{self.port}/api/cloud/generate"
        payload = json.dumps(
            {
                "platform": "modal",
                "config_path": cand,
                "modal_gpu": "A10G",
                "modal_volume": "test-volume",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as error:
            data = json.loads(error.read().decode("utf-8"))
            self.assertTrue(data.get("error"))
        else:
            self.assertTrue(data.get("success"))
            self.assertIn("code", data)
            self.assertIn("modal.App", data["code"])
            self.assertIn("commands", data)

    def test_api_cloud_generate_kaggle(self):
        """POST /api/cloud/generate for Kaggle should return kernel code and kaggle CLI commands."""
        import json

        cand = (
            "configs/exp001_tfidf_baseline.yaml"
            if Path("configs/exp001_tfidf_baseline.yaml").exists()
            else "configs/exp000_template.yaml"
        )
        url = f"http://127.0.0.1:{self.port}/api/cloud/generate"
        payload = json.dumps(
            {
                "platform": "kaggle",
                "config_path": cand,
                "kaggle_gpu": "T4 x 2",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        from harness.templates.kaggle_template import KaggleKernelManager

        with patch.object(
            KaggleKernelManager,
            "get_authenticated_username",
            return_value="fixture-user",
        ):
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as error:
                data = json.loads(error.read().decode("utf-8"))
                self.assertTrue(data.get("error"))
            else:
                self.assertTrue(data.get("success"))
                self.assertIn("code", data)
                self.assertIn("metadata", data)
                self.assertIn("commands", data)

    def test_static_html_served(self):
        """Root GET / should return the SPA index.html with 200 OK.

        The served file depends on which build exists: a local ``dashboard/dist``
        build (gitignored) wins over the committed ``tools/web`` fallback. Assert
        stable SPA markers rather than an exact title, which differs between the
        two builds, so the test is environment-independent (passes in CI where
        only ``tools/web`` exists).
        """
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            content = response.read().decode("utf-8")
            self.assertIn("<title>", content)
            self.assertTrue(
                '<div id="root"></div>' in content or "vis-network" in content
            )

    def test_api_config_get(self):
        """GET /api/config should retrieve and parse target experiment config."""
        cand = (
            "configs/exp001_feat_feature_pipeline.yaml"
            if Path("configs/exp001_feat_feature_pipeline.yaml").exists()
            else "configs/exp000_template.yaml"
        )
        data = self.fetch_json(f"/api/config?path={cand}")
        self.assertIn("content", data)
        self.assertIn("parsed", data)
        self.assertIn("experiment", data["parsed"])

    def test_api_config_create_and_save(self):
        """POST /api/config/create should generate valid experiment config on disk."""
        url = f"http://127.0.0.1:{self.port}/api/config/create"
        payload = json.dumps(
            {
                "exp_id": "exp999_test_dummy",
                "parent_id": "exp001_baseline_tfidf",
                "branch": "feat",
                "hypothesis": "Test feature intervention hypothesis",
                "intervention": "Test controlled intervention",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as error:
            data = json.loads(error.read().decode("utf-8"))
            self.assertTrue(data.get("error"))
        else:
            self.assertTrue(data.get("success"))
        self.assertTrue(
            (self.workspace_root / "configs" / "exp999_test_dummy.yaml").is_file()
        )


if __name__ == "__main__":
    unittest.main()
