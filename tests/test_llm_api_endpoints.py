import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from tools.dashboard_server import DashboardHandler
import socketserver

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestLLMAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from unittest.mock import patch
        from harness.llm_engine import LLMEngine
        cls.port = 0  # dynamic: avoid fixed-port conflicts (e.g. under pytest-xdist)
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
        from harness.environment_detector import EnvironmentDetector
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


if __name__ == "__main__":
    unittest.main()

