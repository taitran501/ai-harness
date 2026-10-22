import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

from harness.codabench_compat import CodabenchClient
from harness.codabench_cli import main as codabench_cli_main


class TestCodabenchCompat(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("requests.get")
    def test_get_submission(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": 1007196,
            "status": "Finished",
            "scores": [
                {"column_key": "accuracy", "score": "0.714285", "is_primary": True},
                {"column_key": "coverage", "score": "1.0", "is_primary": False},
            ],
            "description": "Test V30",
        }
        mock_get.return_value = mock_resp

        client = CodabenchClient(session_id="dummy_session_123")
        sub = client.get_submission(1007196)
        self.assertEqual(sub["id"], 1007196)
        self.assertEqual(sub["status"], "Finished")
        self.assertEqual(sub["primary_score"], 0.714285)
        self.assertEqual(sub["description"], "Test V30")

    @patch("requests.get")
    def test_download_submission(self, mock_get):
        import io
        import zipfile

        # Prepare a mock zip in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("predictions.jsonl", '{"id": "case-1", "is_robust": true}\n')
        zip_bytes = zip_buf.getvalue()

        def side_effect(url, headers=None):
            resp = MagicMock()
            if "api/submissions/1007196/get_details/" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "logs": [{"name": "ingestion_log", "data_file": "https://example.com/log.txt"}],
                    "prediction_result": "https://example.com/predictions.zip",
                }
                resp.text = json.dumps(resp.json.return_value)
            elif "api/submissions/1007196/" in url:
                resp.status_code = 200
                resp.json.return_value = {"id": 1007196, "status": "Finished", "scores": []}
                resp.text = json.dumps(resp.json.return_value)
            elif "log.txt" in url:
                resp.status_code = 200
                resp.content = b"All models evaluated successfully."
            elif "predictions.zip" in url:
                resp.status_code = 200
                resp.content = zip_bytes
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        client = CodabenchClient(session_id="dummy_session")
        out_dir = self.tmp / "audit_1007196"
        res = client.download_submission(1007196, output_dir=out_dir)

        self.assertEqual(res["sub_id"], 1007196)

        self.assertTrue((out_dir / "meta.json").is_file())
        self.assertTrue((out_dir / "details.json").is_file())
        self.assertTrue((out_dir / "ingestion_log.txt").is_file())
        self.assertTrue((out_dir / "predictions.jsonl").is_file())
        self.assertIn("case-1", (out_dir / "predictions.jsonl").read_text())

    @patch("requests.get")
    def test_codabench_cli(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": 1007196, "status": "Finished", "scores": []}
        mock_get.return_value = mock_resp

        ret = codabench_cli_main(["status", "1007196", "--session", "test_sess"])
        self.assertEqual(ret, 0)

    def test_resolve_session_id_scratch_fallback(self):
        scratch_dir = self.tmp / "scratch"
        scratch_dir.mkdir(parents=True)
        py_file = scratch_dir / "probe.py"
        py_file.write_text('session_id = "test_scanned_token_abc1234567890"\nprint("ok")', encoding="utf-8")

        resolved = CodabenchClient._resolve_session_id(project_root=self.tmp)
        self.assertEqual(resolved, "test_scanned_token_abc1234567890")

    def test_resolve_session_id_home_dir(self):
        home_dir = self.tmp / "fake_home"
        home_dir.mkdir(parents=True)
        token_file = home_dir / ".codabench_session"
        token_file.write_text("home_token_xyz987654321", encoding="utf-8")

        with patch("pathlib.Path.home", return_value=home_dir):
            resolved = CodabenchClient._resolve_session_id(project_root=self.tmp)
            self.assertEqual(resolved, "home_token_xyz987654321")

    def test_codabench_cli_auth(self):
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmp)
            ret = codabench_cli_main(["auth", "my_secret_token_12345"])
            self.assertEqual(ret, 0)
            self.assertTrue((self.tmp / ".codabench_session").is_file())
            self.assertEqual((self.tmp / ".codabench_session").read_text(encoding="utf-8"), "my_secret_token_12345")
        finally:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
