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
        cls.port = 8995
        socketserver.TCPServer.allow_reuse_address = True
        cls.httpd = socketserver.ThreadingTCPServer(("", cls.port), DashboardHandler)
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_get_llm_providers(self):
        url = f"http://localhost:{self.port}/api/llm/providers"
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
        url = f"http://localhost:{self.port}/api/llm/sample_data"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("columns", data)
            self.assertIn("rows", data)
            self.assertIsInstance(data["columns"], list)
            self.assertIsInstance(data["rows"], list)

    def test_post_llm_generate_peft_script(self):
        url = f"http://localhost:{self.port}/api/llm/generate_peft_script"
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
        url = f"http://localhost:{self.port}/api/environment/llm/key"
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


if __name__ == "__main__":
    unittest.main()
