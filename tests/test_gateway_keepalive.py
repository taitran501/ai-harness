import json
import unittest
from unittest.mock import MagicMock, patch

from harness.gateway_keepalive import GatewayKeepaliveDaemon


class TestGatewayKeepalive(unittest.TestCase):
    def test_init_defaults(self):
        daemon = GatewayKeepaliveDaemon()
        self.assertEqual(daemon.gateway_url, "http://gateway:8001")
        self.assertEqual(daemon.action, "RESET")
        self.assertGreaterEqual(daemon.ping_interval_s, 10.0)

    def test_init_env_overrides(self):
        with patch.dict("os.environ", {"ARC_BASE_URL": "http://custom-hearth:9000", "ARC_API_KEY": "secret-key"}):
            daemon = GatewayKeepaliveDaemon()
            self.assertEqual(daemon.gateway_url, "http://custom-hearth:9000")
            self.assertEqual(daemon.api_key, "secret-key")
            headers = daemon._get_headers()
            self.assertEqual(headers["X-API-Key"], "secret-key")

    @patch("urllib.request.urlopen")
    def test_discovery_and_heartbeat(self, mock_urlopen):
        def side_effect(req, timeout=10):
            url = req.get_full_url() if hasattr(req, "get_full_url") else str(req)
            resp = MagicMock()
            resp.__enter__.return_value = resp
            if "/api/games" in url:
                resp.status = 200
                resp.read.return_value = json.dumps([{"game_id": "game_alpha_01"}]).encode()
            elif "/api/game/game_alpha_01/step" in url:
                resp.status = 200
                resp.read.return_value = b'{"success": true}'
            else:
                resp.status = 404
                resp.read.return_value = b"{}"
            return resp

        mock_urlopen.side_effect = side_effect

        daemon = GatewayKeepaliveDaemon(
            gateway_url="http://test-gateway:8001",
            ping_interval_s=10.0,
            action="RESET",
        )

        with daemon:
            discovered = daemon._discover_game_id()
            self.assertEqual(discovered, "game_alpha_01")
            success = daemon._send_heartbeat(discovered)
            self.assertTrue(success)
            self.assertEqual(daemon.last_status, 200)
            self.assertGreaterEqual(daemon.pulses_sent, 1)


if __name__ == "__main__":
    unittest.main()
