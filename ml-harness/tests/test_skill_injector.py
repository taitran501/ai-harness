"""Unit tests for the Dynamic Skill Injector Engine in ML-Harness."""

import json
import socketserver
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from harness.skill_injector import generate_skill_content, inject_skill_file
from tools.dashboard_server import DashboardHandler


class SkillInjectorTests(unittest.TestCase):
    def test_generate_skill_nlp(self):
        """Verify dynamic skill generation for NLP multi-task competition."""
        task_cfg = {
            "task_name": "My NLP Benchmark",
            "platform": "codabench",
            "task_type": "Multi-Task Text Classification",
            "modality": "text",
            "primary_metric": "Macro-F1 (CV)",
            "direction": "maximize",
            "primary_metric_baseline": 0.6459,
            "primary_metric_target": 0.85,
            "metrics": [
                {
                    "id": "subtask_a",
                    "name": "Label Detection",
                    "metric": "Macro-F1",
                    "direction": "maximize",
                    "baseline_score": 0.6962,
                    "target_score": 0.88,
                    "weight": 0.333,
                    "description": "Binary label detection",
                }
            ],
            "languages": ["EN", "IT", "NL"],
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("name: ml-experiment-harness", content)
        self.assertIn("My NLP Benchmark", content)
        self.assertIn("Codabench", content)
        self.assertIn("Macro-F1 (CV)", content)
        self.assertIn("0.6459", content)
        self.assertIn("0.85", content)
        self.assertIn("Natural Language Processing (NLP)", content)
        self.assertIn("microsoft/deberta-v3-base", content)
        self.assertIn("Focal Loss", content)
        self.assertIn("SWE-Agent Physical & Statistical Audit Gate", content)
        self.assertIn("Task Contract and Approval Gates", content)
        self.assertNotIn("MMES >= 2.0%", content)

    def test_generate_skill_tabular(self):
        """Verify dynamic skill generation for Tabular regression competition."""
        task_cfg = {
            "task_name": "House Prices Benchmark",
            "platform": "kaggle",
            "task_type": "Tabular Regression",
            "modality": "tabular",
            "primary_metric": "RMSE (CV)",
            "direction": "minimize",
            "primary_metric_baseline": 0.145,
            "primary_metric_target": 0.110,
            "metrics": [
                {
                    "id": "rmse",
                    "name": "Root Mean Squared Error",
                    "metric": "RMSE",
                    "direction": "minimize",
                    "baseline_score": 0.145,
                    "target_score": 0.110,
                    "weight": 1.0,
                    "description": "Log-scale RMSE evaluation",
                }
            ],
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("House Prices Benchmark", content)
        self.assertIn("Kaggle", content)
        self.assertIn("Tabular & Structured Data", content)
        self.assertIn("LightGBM", content)
        self.assertIn("CatBoost", content)
        self.assertIn("RMSE (CV)", content)
        self.assertIn("minimize", content)
        self.assertIn("Target encoding", content)

    def test_generate_skill_vision(self):
        """Verify dynamic skill generation for Computer Vision competition."""
        task_cfg = {
            "task_name": "Plant Pathology Classification",
            "platform": "kaggle",
            "task_type": "Image Classification",
            "modality": "vision",
            "primary_metric": "Top-1 Accuracy (CV)",
            "direction": "maximize",
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("Plant Pathology Classification", content)
        self.assertIn("Computer Vision", content)
        self.assertIn("convnext", content)
        self.assertIn("swin", content)
        self.assertIn("Mixup", content)

    def test_inject_skill_file(self):
        """Verify writing skill file to disk in a temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "SKILL.md"
            task_cfg = {
                "task_name": "Test Benchmark",
                "platform": "custom",
                "modality": "generic",
                "primary_metric": "Accuracy",
                "direction": "maximize",
            }
            res_path = inject_skill_file(task_config=task_cfg, output_path=out_path)
            self.assertEqual(res_path, out_path)
            self.assertTrue(out_path.exists())
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Test Benchmark", text)
            self.assertIn("AIDE UCB Tree Search", text)


class SkillInjectorAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 8799
        cls.server = socketserver.TCPServer(("127.0.0.1", cls.port), DashboardHandler)
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_api_skill_inject(self):
        """POST /api/skill/inject should re-generate and update SKILL.md."""
        url = f"http://127.0.0.1:{self.port}/api/skill/inject"
        payload = json.dumps({}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertTrue(data.get("success"))
            self.assertIn("skill_path", data)

    def test_api_task_info_auto_inject(self):
        """POST /api/task_info should save task config and automatically inject skill."""
        url = f"http://127.0.0.1:{self.port}/api/task_info"
        from tools.dashboard_server import load_task_config

        payload = json.dumps(load_task_config()).encode("utf-8")
        post_req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(post_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("success"))
            self.assertTrue(data.get("skill_injected"))


if __name__ == "__main__":
    unittest.main()
