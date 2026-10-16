"""Regression tests for the process-wide environment probe cache.

The cache exists because ``ai-harness status`` was measured at 18 s on a
machine with PyTorch installed (torch import 6.5 s, modal CLI 2.6 s,
localhost port probes 2 s each). GPU/driver/credentials state does not
change within a session, so repeated probes are pure waste.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

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


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "env_probes.json"

    def tearDown(self):
        env_cache.invalidate()
        for var in ("AI_HARNESS_ENV_CACHE", "AI_HARNESS_ENV_TTL_GPU"):
            os.environ.pop(var, None)
        self._tmp.cleanup()

    def test_roundtrip(self):
        env_cache.disk_store(self.cache, {"gpu": {"detected": True}})
        loaded = env_cache.disk_load(self.cache)
        self.assertEqual(loaded, {"gpu": {"detected": True}})

    def test_expired_record_returns_none(self):
        import json as _json

        self.cache.parent.mkdir(parents=True, exist_ok=True)
        record = {"generated_at": time.time() - (env_cache.DISK_TTL + 1), "data": {"gpu": {}}}
        self.cache.write_text(_json.dumps(record), encoding="utf-8")
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_corrupt_record_returns_none(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text("{not json", encoding="utf-8")
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_missing_file_returns_none(self):
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_disabled_cache_skips_disk(self):
        os.environ["AI_HARNESS_ENV_CACHE"] = "off"
        env_cache.disk_store(self.cache, {"gpu": {}})
        self.assertFalse(self.cache.exists())
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_get_full_status_uses_disk_cache_across_processes(self):
        from harness.environment_detector import EnvironmentDetector

        calls = {"n": 0}

        def fake_probe():
            calls["n"] += 1
            return {"fake": calls["n"], "detected": True}

        # First call (cold process): probes run, result lands on disk.
        with unittest.mock.patch.object(
            EnvironmentDetector, "detect_modal", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_kaggle", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_docker", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_gpu", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_llm_status", classmethod(lambda cls: fake_probe())
        ):
            first = EnvironmentDetector.get_full_status(disk_cache_dir=self.cache.parent)
        self.assertEqual(calls["n"], 5)  # all five probes ran

        # Second call simulates a NEW process: wipe the in-process cache
        # (a fresh interpreter has none) — a fresh disk record must prime it,
        # so NO raw probe may run again. The inner _probe_* methods are
        # patched (not the public detect_* wrappers) so the caching layer
        # itself is exercised.
        env_cache.invalidate()
        with unittest.mock.patch.object(
            EnvironmentDetector,
            "_probe_gpu",
            side_effect=AssertionError("probe must not re-run when the disk record is fresh"),
        ), unittest.mock.patch.object(
            EnvironmentDetector, "_probe_llm_status",
            side_effect=AssertionError("probe must not re-run when the disk record is fresh"),
        ):
            second = EnvironmentDetector.get_full_status(disk_cache_dir=self.cache.parent)
        self.assertEqual(calls["n"], 5, "no probe may re-run when the disk record is fresh")
        self.assertEqual(first["gpu"], second["gpu"])

    def test_force_bypasses_disk_cache(self):
        from harness.environment_detector import EnvironmentDetector

        calls = {"n": 0}

        def fake_probe():
            calls["n"] += 1
            return {"n": calls["n"]}

        for _ in range(2):
            with unittest.mock.patch.object(
                EnvironmentDetector, "detect_modal", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_kaggle", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_docker", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_gpu", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_llm_status", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                env_cache, "get_cached", return_value=None
            ):
                EnvironmentDetector.get_full_status(
                    disk_cache_dir=self.cache.parent, force=True
                )
        self.assertEqual(calls["n"], 10, "force=True must re-run every probe")


class ModelRegistryTests(unittest.TestCase):
    """The name registry must serve names WITHOUT importing harness.models
    (that import costs ~1.3 s — numpy/pandas/sklearn — on first import)."""

    def tearDown(self):
        # The registry's cache file is package-level state; tests must not
        # leave synthetic records behind (they would be served to the next
        # test, or to a real CLI run, if the mtime happens to match).
        from harness import model_registry

        try:
            model_registry._CACHE_FILE.unlink()
        except FileNotFoundError:
            pass

    def test_serves_names_without_importing_models_module(self):
        from harness import model_registry

        # Warm the cache (this first call may import harness.models — that
        # is the documented one-time cost).
        model_registry.builtin_model_names()

        sys_mods = {k: v for k, v in sys.modules.items() if k == "harness.models"}
        # Drop the module so the assertion measures what a cache hit does:
        # nothing.
        sys.modules.pop("harness.models", None)
        try:
            names = model_registry.builtin_model_names()
            self.assertNotIn("harness.models", sys.modules,
                             "registry must not import harness.models on a cache hit")
        finally:
            sys.modules.update(sys_mods)
        self.assertTrue(len(names) >= 10)
        self.assertIn("baseline", names)

    def test_cache_invalidates_on_models_py_change(self):
        import json as _json

        from harness import model_registry

        # Warm the cache with real data.
        names_before = model_registry.builtin_model_names()
        record = _json.loads(model_registry._CACHE_FILE.read_text(encoding="utf-8"))
        record["mtime"] = 12345.0  # simulate a code change
        model_registry._CACHE_FILE.write_text(_json.dumps(record), encoding="utf-8")

        # A stale mtime must trigger the real import path (patched here so
        # we observe the delegation without paying the sklearn import).
        with unittest.mock.patch(
            "harness.models.available_model_names", return_value=list(names_before) + ["__sentinel__"]
        ) as patched:
            names_after = model_registry.builtin_model_names()
        self.assertIn("__sentinel__", names_after)
        patched.assert_called_once()
        # and the cache was refreshed with the new mtime
        record2 = _json.loads(model_registry._CACHE_FILE.read_text(encoding="utf-8"))
        self.assertEqual(record2["mtime"], model_registry._MODELS_PY.stat().st_mtime)

    def test_corrupt_cache_falls_back_to_import(self):
        from harness import model_registry

        model_registry._CACHE_FILE.write_text("{broken", encoding="utf-8")
        with unittest.mock.patch("harness.models.available_model_names", return_value=["x"]):
            names = model_registry.builtin_model_names()
        self.assertEqual(names, ["x"])


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
