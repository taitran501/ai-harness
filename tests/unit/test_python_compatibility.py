"""Artifact hashing supports both Python 3.10 and the native digest API."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.artifacts import _fresh_sha256


class FreshSha256Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "payload.bin"

    def payloads(self):
        for size in (0, 1, 65536, 2 * 65536 + 17):
            yield (bytes(range(256)) * ((size + 255) // 256))[:size]

    @unittest.skipUnless(hasattr(hashlib, "file_digest"), "Native digest requires Python 3.11+")
    def test_native_and_fallback_match_known_digest(self):
        for payload in self.payloads():
            with self.subTest(size=len(payload)):
                self.path.write_bytes(payload)
                expected = hashlib.sha256(payload).hexdigest()
                with patch("harness.artifacts.hashlib.file_digest", wraps=hashlib.file_digest) as native:
                    self.assertEqual(_fresh_sha256(self.path), expected)
                    native.assert_called_once()
                with patch.dict(hashlib.__dict__):
                    del hashlib.file_digest
                    self.assertEqual(_fresh_sha256(self.path), expected)

    def test_fallback_without_file_digest(self):
        with patch.dict(hashlib.__dict__):
            hashlib.__dict__.pop("file_digest", None)
            for payload in self.payloads():
                with self.subTest(size=len(payload)):
                    self.path.write_bytes(payload)
                    self.assertEqual(_fresh_sha256(self.path), hashlib.sha256(payload).hexdigest())
