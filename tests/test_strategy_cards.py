import unittest
import json
import urllib.request
import threading
import time
import socketserver
from pathlib import Path
from tools.dashboard_server import DEFAULT_STRATEGY_CARDS, DashboardHandler, get_next_experiment_id

class TestStrategyCardsAndSpawner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 8789
        cls.server = socketserver.TCPServer(("127.0.0.1", cls.port), DashboardHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_default_strategy_cards_completeness(self):
        self.assertEqual(len(DEFAULT_STRATEGY_CARDS), 6)
        branches = {c["branch"] for c in DEFAULT_STRATEGY_CARDS}
        self.assertIn("feat", branches)
        self.assertIn("model", branches)
        self.assertIn("loss", branches)
        self.assertIn("ens", branches)

        approaches = {c["approach_type"] for c in DEFAULT_STRATEGY_CARDS}
        self.assertIn("ml", approaches)
        self.assertIn("dl", approaches)
        self.assertIn("llm", approaches)

        for card in DEFAULT_STRATEGY_CARDS:
            self.assertIn("id", card)
            self.assertIn("title", card)
            self.assertIn("hypothesis", card)
            self.assertIn("intervention", card)
            self.assertIn("problem_addressed", card)
            self.assertIn("model_family", card)
            self.assertIn("model_name", card)
            self.assertIn("estimated_delta", card)
            self.assertIn("approach_type", card)

    def test_api_strategy_cards_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/api/strategy_cards"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode("utf-8"))
            self.assertIn("strategies", data)
            self.assertEqual(len(data["strategies"]), len(DEFAULT_STRATEGY_CARDS))

    def test_get_next_experiment_id(self):
        res = get_next_experiment_id("feat", "video_title_ctx")
        self.assertIn("next_id", res)
        self.assertTrue(res["next_id"].startswith("exp"))
        self.assertIn("feat", res["next_id"])
        self.assertIn("video_title_ctx", res["next_id"])

if __name__ == "__main__":
    unittest.main()
