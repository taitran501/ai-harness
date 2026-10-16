"""Regression tests for the process-wide environment probe cache.

The cache exists because ``ai-harness status`` was measured at 18 s on a
machine with PyTorch installed (torch import 6.5 s, modal CLI 2.6 s,
localhost port probes 2 s each). GPU/driver/credentials state does not
change within a session, so repeated probes are pure waste.
"""

import os
import time
import unittest

from harness import env_cache


class EnvCacheTests(unittest.TestCase):
    def setUp(self):
        env_cache.invalidate()

    def tearDown(self):
        env_cache.invalidate()
        for var in ("AI_HARNESS_ENV_CACHE", "AI_HARNESS_ENV_TTL_GPU"):
            os.environ.pop(var, None)

    def test_set_and_get(self):
        env_cache.set_cached("gpu", {"detected": True})
        self.assertEqual(env_cache.get_cached("gpu"), {"detected": True})

    def test_expired_entry_returns_none(self):
        env_cache.set_cached("gpu", {"detected": True}, ttl=0.01)
        time.sleep(0.03)
        self.assertIsNone(env_cache.get_cached("gpu"))

    def test_ttl_override_env_var(self):
        os.environ["AI_HARNESS_ENV_TTL_GPU"] = "0.01"
        env_cache.set_cached("gpu", {"a": 1})
        time.sleep(0.03)
        self.assertIsNone(env_cache.get_cached("gpu"))

    def test_invalid_ttl_override_falls_back_to_default(self):
        os.environ["AI_HARNESS_ENV_TTL_GPU"] = "not-a-number"
        self.assertEqual(env_cache._ttl_for("gpu"), env_cache.DEFAULT_TTLS["gpu"])

    def test_disabled_cache_never_caches(self):
        os.environ["AI_HARNESS_ENV_CACHE"] = "off"
        env_cache.set_cached("gpu", {"detected": True})
        self.assertIsNone(env_cache.get_cached("gpu"))

    def test_cached_probe_runs_probe_once(self):
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            return {"n": calls["n"]}

        first = env_cache.cached_probe("kaggle", probe)
        second = env_cache.cached_probe("kaggle", probe)
        self.assertEqual(first, {"n": 1})
        self.assertEqual(second, {"n": 1})
        self.assertEqual(calls["n"], 1)

    def test_invalidate_clears_everything(self):
        env_cache.set_cached("modal", 1)
        env_cache.set_cached("kaggle", 2)
        env_cache.invalidate()
        self.assertIsNone(env_cache.get_cached("modal"))
        self.assertIsNone(env_cache.get_cached("kaggle"))


class LocalEndpointNormalizationTests(unittest.TestCase):
    def test_localhost_normalized_to_127_0_0_1(self):
        import urllib.request as urlreq

        from harness.llm_engine import LLMEngine

        # A dead port must fail fast; the point under test is that the
        # request targets 127.0.0.1, not 'localhost' (which on Windows
        # resolves to ::1 + 127.0.0.1 and doubles the connect timeout).
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["timeout"] = timeout
            raise urlreq.URLError("connection refused")

        with unittest.mock.patch.object(urlreq, "urlopen", side_effect=fake_urlopen):
            result = LLMEngine.check_local_endpoint("http://localhost:11434/v1")
        self.assertFalse(result)
        self.assertIn("127.0.0.1", seen["url"])
        self.assertNotIn("localhost", seen["url"])

    def test_explicit_host_untouched(self):
        import urllib.request as urlreq

        from harness.llm_engine import LLMEngine

        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            raise urlreq.URLError("nope")

        with unittest.mock.patch.object(urlreq, "urlopen", side_effect=fake_urlopen):
            LLMEngine.check_local_endpoint("http://192.168.1.5:8000/v1")
        self.assertIn("192.168.1.5", seen["url"])

    def test_dead_localhost_port_fails_within_timeout(self):
        from harness.llm_engine import LLMEngine

        # Live measurement, no mocks: on Windows a dead 'localhost' port
        # used to cost 2x the timeout (2.02 s at timeout=1.0) because the
        # connect is attempted for both ::1 and 127.0.0.1. After the
        # normalization it must stay under one timeout cycle.
        started = time.perf_counter()
        LLMEngine.check_local_endpoint("http://localhost:11434/v1", timeout=0.5)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
