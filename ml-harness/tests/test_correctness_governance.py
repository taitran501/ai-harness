import tempfile
import unittest
import ssl
import urllib.error
from pathlib import Path
from unittest.mock import patch
from harness.resource_ledger import reserve,terminal
from harness.llm_engine import _safe_urlopen

class GovernanceContracts(unittest.TestCase):
    def test_tls_failure_never_retries_unverified(self):
        with patch("urllib.request.urlopen",side_effect=urllib.error.URLError("certificate verify failed")) as call:
            with self.assertRaises(urllib.error.URLError): _safe_urlopen("https://example.invalid")
            self.assertEqual(call.call_count,1)
            self.assertEqual(call.call_args.kwargs["context"].verify_mode,ssl.CERT_REQUIRED)
    def test_reservation_and_terminal_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"ledger.json"
            token,_=reserve(path,"session","run_full_validation","local",{"max_experiments":1})
            with self.assertRaises(ValueError): reserve(path,"session","run_full_validation","local",{"max_experiments":1})
            self.assertTrue(terminal(path,token,"FAILED",2))
            self.assertFalse(terminal(path,token,"COMPLETED",5))
            reserve(path,"session","promote_candidate","local",{"max_experiments":1})
