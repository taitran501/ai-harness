"""Environment and Multi-Account Detection Module for AI-Harness.

Detects and manages credentials, CLI availability, active accounts, and hardware for:
1. Modal.com (Multi-profile switcher, tokens, workspace inspection)
2. Kaggle CLI (Multi-profile switcher, ~/.kaggle/kaggle.json credentials)
3. Docker (Daemon status, version)
4. Local Hardware & GPU (NVIDIA GPU, VRAM, Driver, CUDA)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from harness.env_cache import cached_probe
from harness.io_utils import get_project_root

PROJECT_ROOT = get_project_root()

KEY_TO_PROVIDER: Dict[str, str] = {
    "GEMINI_API_KEY": "gemini",
    "GOOGLE_API_KEY": "gemini",
    "OPENAI_API_KEY": "openai",
    "ANTHROPIC_API_KEY": "anthropic",
    "DEEPSEEK_API_KEY": "deepseek",
    "GROQ_API_KEY": "groq",
    "OPENROUTER_API_KEY": "openrouter",
}
PROVIDER_TO_KEY: Dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}



class EnvironmentDetector:
    """Detects and manages environments and accounts across Modal, Kaggle, Docker, and GPU."""

    @staticmethod
    def _probe_modal() -> Dict[str, Any]:
        """Detect Modal CLI installation, configured profiles, and active workspace."""
        modal_path = shutil.which("modal")
        result: Dict[str, Any] = {
            "installed": modal_path is not None,
            "executable": modal_path,
            "profiles": [],
            "active_profile": None,
            "has_credentials": False,
            "error": None,
        }

        # Check ~/.modal.toml
        modal_toml = Path.home() / ".modal.toml"
        if modal_toml.exists():
            result["has_credentials"] = True

        if not result["installed"]:
            return result

        try:
            cmd = ["modal", "profile", "list"]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            output = proc.stdout.strip()
            profiles = []
            active = None

            for line in output.splitlines():
                if "│" not in line:
                    continue
                parts = [p.strip() for p in line.split("│")]
                # Expected parts: ['', indicator, profile, workspace, '']
                if len(parts) >= 4:
                    indicator = parts[1]
                    name = parts[2]
                    if not name or name.lower() in ("profile", "") or any(c in name for c in "─┼┬┴"):
                        continue
                    profiles.append(name)
                    if "•" in indicator or "*" in indicator or indicator.lower() in ("yes", "active", "true"):
                        active = name

            if not profiles and output:
                words = re.findall(r"\b[a-zA-Z0-9_-]{3,30}\b", output)
                for w in words:
                    if w.lower() not in ("profile", "workspace", "active", "yes", "no", "modal"):
                        profiles.append(w)

            result["profiles"] = list(dict.fromkeys(profiles))
            result["active_profile"] = active or (profiles[0] if profiles else None)
        except Exception as e:
            result["error"] = str(e)

        return result

    @staticmethod
    def parse_modal_token_input(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse raw user input or copy-pasted CLI command to extract token_id and token_secret."""
        if not raw_text:
            return None, None
        token_id = None
        token_secret = None

        m_id = re.search(r"\b(ak-[a-zA-Z0-9_-]+)\b", raw_text)
        if m_id:
            token_id = m_id.group(1)

        m_sec = re.search(r"\b(as-[a-zA-Z0-9_-]+)\b", raw_text)
        if m_sec:
            token_secret = m_sec.group(1)

        return token_id, token_secret

    @staticmethod
    def launch_modal_auth() -> Dict[str, Any]:
        """Open Modal token management web page in browser."""
        auth_url = "https://modal.com/settings/tokens"
        opened = False
        try:
            import webbrowser
            opened = webbrowser.open(auth_url)
        except Exception:
            pass
        return {
            "success": True,
            "opened_browser": opened,
            "auth_url": auth_url,
            "message": "Opened Modal tokens page (https://modal.com/settings/tokens) in your browser.",
        }

    @staticmethod
    def launch_kaggle_auth() -> Dict[str, Any]:
        """Open Kaggle settings and run `kaggle auth login` to authenticate Kaggle CLI."""
        auth_url = "https://www.kaggle.com/settings"
        opened = False
        try:
            import webbrowser
            opened = webbrowser.open(auth_url)
        except Exception:
            pass

        cmd = ["kaggle", "auth", "login"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=2)
                output = (stdout or "") + "\n" + (stderr or "")
                return {
                    "success": True,
                    "auth_url": auth_url,
                    "opened_browser": opened,
                    "command": "kaggle auth login",
                    "output": output.strip(),
                    "message": "Opened Kaggle Account Settings (https://www.kaggle.com/settings). Complete login in your browser.",
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": True,
                    "auth_url": auth_url,
                    "opened_browser": opened,
                    "command": "kaggle auth login",
                    "message": "Running 'kaggle auth login' in background. Complete login in your browser.",
                }
        except Exception as e:
            return {
                "success": True,
                "auth_url": auth_url,
                "opened_browser": opened,
                "command": "kaggle auth login",
                "message": f"Opened Kaggle Account Settings ({auth_url}). Note: {e}",
            }

    @staticmethod
    def _probe_kaggle() -> Dict[str, Any]:
        """Detect Kaggle CLI installation, credentials, and active username."""
        kaggle_path = shutil.which("kaggle")
        result: Dict[str, Any] = {
            "installed": kaggle_path is not None,
            "executable": kaggle_path,
            "authenticated": False,
            "username": None,
            "config_dir": str(Path.home() / ".kaggle"),
            "profiles": [],
            "error": None,
        }

        # Check credentials in ~/.kaggle/ or ~/.config/kaggle/
        search_dirs = [
            Path.home() / ".kaggle",
            Path.home() / ".config" / "kaggle",
        ]
        for k_dir in search_dirs:
            for cred_name in ["credentials.json", "kaggle.json", "access_token"]:
                cred_file = k_dir / cred_name
                if cred_file.exists():
                    try:
                        raw = cred_file.read_text(encoding="utf-8").strip()
                        if cred_name.endswith(".json"):
                            data = json.loads(raw)
                            username = (
                                data.get("username")
                                or data.get("user")
                                or data.get("user_name")
                                or ("OAuth User" if data.get("access_token") or data.get("refresh_token") else None)
                            )
                            if username:
                                result["authenticated"] = True
                                result["username"] = username
                                break
                        elif raw:
                            result["authenticated"] = True
                            result["username"] = "Kaggle OAuth"
                            break
                    except Exception:
                        pass
            if result["authenticated"]:
                break

        # Also check kaggle config view
        if not result["username"] and result["installed"]:
            try:
                proc = subprocess.run(
                    ["kaggle", "config", "view"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if proc.returncode == 0:
                    for line in proc.stdout.splitlines():
                        if "username:" in line.lower():
                            parts = line.split(":", 1)
                            if len(parts) == 2 and parts[1].strip() and parts[1].strip() != "None":
                                result["authenticated"] = True
                                result["username"] = parts[1].strip()
                                break
            except Exception:
                pass

        # Check saved profiles under ~/.kaggle/profiles/
        profiles_dir = Path.home() / ".kaggle" / "profiles"
        if profiles_dir.exists():
            for p_file in profiles_dir.glob("*.json"):
                try:
                    p_data = json.loads(p_file.read_text(encoding="utf-8"))
                    u = p_data.get("username", p_file.stem)
                    result["profiles"].append({
                        "name": p_file.stem,
                        "username": u,
                        "is_active": (result["username"] == u),
                    })
                except Exception:
                    pass

        return result

    @staticmethod
    def switch_kaggle_profile(username: str, key: str, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Save and switch Kaggle credentials."""
        try:
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            kaggle_json = kaggle_dir / "kaggle.json"

            # Write active credentials
            creds = {"username": username, "key": key}
            kaggle_json.write_text(json.dumps(creds, indent=2), encoding="utf-8")

            # Also save to profiles archive
            p_name = profile_name or username
            profiles_dir = kaggle_dir / "profiles"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            (profiles_dir / f"{p_name}.json").write_text(json.dumps(creds, indent=2), encoding="utf-8")

            return {"success": True, "username": username, "profile_name": p_name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _probe_docker() -> Dict[str, Any]:
        """Detect Docker installation and daemon status."""
        docker_path = shutil.which("docker")
        result: Dict[str, Any] = {
            "installed": docker_path is not None,
            "executable": docker_path,
            "running": False,
            "version": None,
            "error": None,
        }

        if not result["installed"]:
            return result

        try:
            proc = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                result["version"] = proc.stdout.strip()
        except Exception as e:
            result["error"] = str(e)

        try:
            # Check daemon
            proc_info = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc_info.returncode == 0:
                result["running"] = True
        except Exception:
            result["running"] = False

        return result

    @staticmethod
    def _probe_gpu() -> Dict[str, Any]:
        """Detect NVIDIA GPU hardware, VRAM, and driver."""
        result: Dict[str, Any] = {
            "detected": False,
            "gpu_name": None,
            "memory_total_mb": None,
            "memory_free_mb": None,
            "driver_version": None,
            "cuda_available": False,
        }

        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                proc = subprocess.run(
                    [
                        smi,
                        "--query-gpu=name,memory.total,memory.free,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    line = proc.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        result["detected"] = True
                        result["gpu_name"] = parts[0]
                        try:
                            result["memory_total_mb"] = int(float(parts[1]))
                            result["memory_free_mb"] = int(float(parts[2]))
                        except ValueError:
                            pass
                        result["driver_version"] = parts[3]
            except Exception:
                pass

        # Fallback: torch cuda or mps — but ONLY when nvidia-smi did not
        # already report a GPU. Importing torch costs ~6s on machines that
        # have it installed, so it must not run on the common path. When
        # torch is already in sys.modules (agent imported it) it is free to
        # use. When skipped, cuda_available is None = "unknown, not probed"
        # (never a false negative that would hide a working GPU).
        smi_reported = result["detected"]
        torch_mod = sys.modules.get("torch")
        if smi_reported and torch_mod is None:
            result["cuda_available"] = None
            result["cuda_probe"] = "skipped (nvidia-smi reported GPU; torch not imported)"
        else:
            try:
                if torch_mod is None:
                    import torch as torch_mod
                result["cuda_available"] = torch_mod.cuda.is_available()
                result["cuda_probe"] = "torch"
                if result["cuda_available"] and not result["gpu_name"]:
                    result["detected"] = True
                    result["gpu_name"] = torch_mod.cuda.get_device_name(0)
                elif hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
                    result["mps_available"] = True
                    if not result["gpu_name"]:
                        result["detected"] = True
                        result["gpu_name"] = "Apple Silicon GPU (MPS)"
            except Exception:
                pass

        return result

    @classmethod
    def detect_modal(cls) -> Dict[str, Any]:
        """Detect Modal CLI installation, profiles, and active workspace (TTL-cached).

        The ``modal profile list`` subprocess can take several seconds on a
        cold start, so repeated calls within the TTL window are served from
        the process cache.
        """
        return cached_probe("modal", lambda: cls._probe_modal())

    @classmethod
    def detect_kaggle(cls) -> Dict[str, Any]:
        """Detect Kaggle CLI installation and authentication (TTL-cached)."""
        return cached_probe("kaggle", lambda: cls._probe_kaggle())

    @classmethod
    def detect_docker(cls) -> Dict[str, Any]:
        """Detect Docker CLI installation and daemon status (TTL-cached)."""
        return cached_probe("docker", lambda: cls._probe_docker())

    @classmethod
    def detect_gpu(cls) -> Dict[str, Any]:
        """Detect NVIDIA GPU hardware, VRAM, and driver (TTL-cached, default 600 s).

        See :meth:`_probe_gpu` for the torch-import policy.
        """
        return cached_probe("gpu", lambda: cls._probe_gpu())

    @classmethod
    def detect_llm_status(cls) -> Dict[str, Any]:
        """Detect LLM API keys, configured providers, and local inference endpoints.

        Cached (TTL default 120 s): the local-endpoint port probes cost
        seconds each on hosts where the ports are closed.
        """
        return cached_probe("llm", lambda: cls._probe_llm_status())

    @classmethod
    def _probe_llm_status(cls) -> Dict[str, Any]:
        env_dict: Dict[str, str] = {}
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_dict[k.strip()] = v.strip("\"' ")

        key_vars = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        providers: Dict[str, Any] = {}
        keys: Dict[str, Any] = {}

        for p_name, env_key in key_vars.items():
            val = os.environ.get(env_key) or env_dict.get(env_key)
            if p_name == "gemini" and not val:
                val = os.environ.get("GOOGLE_API_KEY") or env_dict.get("GOOGLE_API_KEY")

            is_set = bool(val)
            masked = f"{val[:4]}...{val[-4:]}" if val and len(val) > 8 else ("***" if val else None)

            keys[env_key] = {
                "set": is_set,
                "preview": masked,
            }

            providers[p_name] = {
                "configured": is_set,
                "env_var": env_key,
                "masked_key": masked,
                "is_local": False,
                "endpoint": None,
                "reachable": is_set,
            }

        # Local LLM providers (Ollama & vLLM)
        local_providers = {
            "ollama": ("http://localhost:11434/v1", "OLLAMA_BASE_URL"),
            "vllm": ("http://localhost:8000/v1", "VLLM_BASE_URL"),
        }
        for lp_name, (default_url, env_var) in local_providers.items():
            endpoint = os.environ.get(env_var) or env_dict.get(env_var) or default_url
            try:
                from harness.llm_engine import LLMEngine
                is_reachable = LLMEngine.check_local_endpoint(endpoint)
            except Exception:
                is_reachable = False

            providers[lp_name] = {
                "configured": is_reachable,
                "env_var": env_var,
                "masked_key": None,
                "is_local": True,
                "endpoint": endpoint,
                "reachable": is_reachable,
            }

        total_configured = sum(1 for p in providers.values() if p["configured"] or p["reachable"])

        return {
            "keys": keys,
            "providers": providers,
            "active_providers_count": total_configured,
            "total_configured": total_configured,
        }

    @classmethod
    def verify_llm_api_key(cls, provider_or_key: str, key_value: Optional[str] = None) -> Dict[str, Any]:
        """Test whether an API key for the given provider is valid by pinging official endpoints."""
        prov = provider_or_key.strip()
        if prov.upper() in KEY_TO_PROVIDER:
            prov = KEY_TO_PROVIDER[prov.upper()]
        else:
            prov = prov.lower()

        # Resolve key value if not provided
        val = key_value.strip() if key_value else None
        if not val:
            env_var = PROVIDER_TO_KEY.get(prov)
            if env_var:
                val = os.environ.get(env_var)
                if not val and prov == "gemini":
                    val = os.environ.get("GOOGLE_API_KEY")
                if not val:
                    env_file = PROJECT_ROOT / ".env"
                    if env_file.exists():
                        for line in env_file.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if env_var and line.startswith(f"{env_var}="):
                                val = line.split("=", 1)[1].strip("\"' ")
                                break
                            if prov == "gemini" and line.startswith("GOOGLE_API_KEY="):
                                val = line.split("=", 1)[1].strip("\"' ")
                                break

        t0 = time.time()
        ctx = ssl._create_unverified_context()

        # Handle local daemons
        if prov == "ollama":
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "AI-Harness/2.0"})
                with urllib.request.urlopen(req, timeout=3.0, context=ctx) as resp:
                    latency = int((time.time() - t0) * 1000)
                    return {
                        "success": True,
                        "valid": resp.status < 500,
                        "provider": prov,
                        "latency_ms": latency,
                        "message": f"Ollama local daemon responding (HTTP {resp.status})",
                    }
            except Exception as e:
                return {
                    "success": False,
                    "valid": False,
                    "provider": prov,
                    "error": f"Ollama daemon not reachable at localhost:11434 ({e})",
                }

        if prov == "vllm":
            try:
                req = urllib.request.Request("http://localhost:8000/v1/models", headers={"User-Agent": "AI-Harness/2.0"})
                with urllib.request.urlopen(req, timeout=3.0, context=ctx) as resp:
                    latency = int((time.time() - t0) * 1000)
                    return {
                        "success": True,
                        "valid": resp.status < 500,
                        "provider": prov,
                        "latency_ms": latency,
                        "message": f"vLLM server responding (HTTP {resp.status})",
                    }
            except Exception as e:
                return {
                    "success": False,
                    "valid": False,
                    "provider": prov,
                    "error": f"vLLM server not reachable at localhost:8000 ({e})",
                }

        if not val:
            return {
                "success": False,
                "valid": False,
                "provider": prov,
                "error": f"No API key provided or found in environment for provider '{prov}'.",
            }

        headers: Dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Harness/2.0",
        }

        if prov == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote_plus(val)}"
        elif prov == "openai":
            url = "https://api.openai.com/v1/models"
            headers["Authorization"] = f"Bearer {val}"
        elif prov == "anthropic":
            url = "https://api.anthropic.com/v1/models"
            headers["x-api-key"] = val
            headers["anthropic-version"] = "2023-06-01"
        elif prov == "deepseek":
            url = "https://api.deepseek.com/models"
            headers["Authorization"] = f"Bearer {val}"
        elif prov == "groq":
            url = "https://api.groq.com/openai/v1/models"
            headers["Authorization"] = f"Bearer {val}"
        elif prov == "openrouter":
            url = "https://openrouter.ai/api/v1/auth/key"
            headers["Authorization"] = f"Bearer {val}"
        else:
            return {
                "success": False,
                "valid": False,
                "provider": prov,
                "error": f"Unsupported LLM provider '{prov}'.",
            }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8.0, context=ctx) as resp:
                latency = int((time.time() - t0) * 1000)
                return {
                    "success": True,
                    "valid": True,
                    "provider": prov,
                    "latency_ms": latency,
                    "message": f"Active & Verified ({resp.status} OK, {latency}ms)",
                }
        except urllib.error.HTTPError as e:
            latency = int((time.time() - t0) * 1000)
            body = e.read().decode("utf-8", errors="ignore")
            error_msg = f"HTTP {e.code}"
            try:
                parsed = json.loads(body)
                if "error" in parsed:
                    err_obj = parsed["error"]
                    error_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                elif "message" in parsed:
                    error_msg = parsed["message"]
            except Exception:
                pass

            if e.code in (400, 401, 403):
                return {
                    "success": True,
                    "valid": False,
                    "provider": prov,
                    "latency_ms": latency,
                    "error": f"{error_msg} ({e.code})",
                }
            return {
                "success": False,
                "valid": False,
                "provider": prov,
                "latency_ms": latency,
                "error": f"HTTP {e.code}: {error_msg}",
            }
        except Exception as e:
            return {
                "success": False,
                "valid": False,
                "provider": prov,
                "error": str(e),
            }

    @staticmethod
    def save_llm_api_key(key_name: str, key_value: str) -> Dict[str, Any]:
        """Save API key to environment and persist into .env file."""
        clean_key = key_value.strip()
        os.environ[key_name] = clean_key
        env_file = PROJECT_ROOT / ".env"
        lines = []
        if env_file.exists():
            lines = env_file.read_text(encoding="utf-8").splitlines()

        found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(f"{key_name}="):
                new_lines.append(f"{key_name}={clean_key}")
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f"{key_name}={clean_key}")

        env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        preview = f"{clean_key[:6]}...{clean_key[-4:]}" if len(clean_key) > 10 else "***"
        return {"success": True, "key_name": key_name, "preview": preview}

    @classmethod
    def get_full_status(cls, force: bool = False, disk_cache_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Aggregate full environment and hardware status.

        Probes run in parallel (they are subprocess/network-bound and
        independent) and each is TTL-cached, so a repeated call within a
        process is nearly free. ``force=True`` bypasses the caches.

        ``disk_cache_dir`` adds a cross-process layer: when given (and the
        cache is not disabled) a fresh ``env_probes.json`` record inside that
        directory is returned as-is, and freshly probed results are written
        back. This is what makes one-shot CLI commands fast — without it
        every new process re-runs all five probes.
        """
        from concurrent.futures import ThreadPoolExecutor

        from harness import env_cache

        if force:
            env_cache.invalidate()

        # The in-process TTL cache holds only per-probe values, so a fresh
        # process with a warm disk record still re-runs every probe before
        # writing the merged record. Prime the process cache from the disk
        # record first; fully fresh keys are the only ones that re-probe.
        # (A per-key TTL on disk would require per-probe timestamps; the
        # wall-clock cost of one full probe round per 120 s is acceptable.)
        cache_path = disk_cache_dir / "env_probes.json" if disk_cache_dir is not None else None
        if cache_path is not None and not force:
            record = env_cache.disk_load(cache_path)
            if record is not None:
                for key, value in record.items():
                    env_cache.set_cached(key, value)

        # Call the public cached wrappers (not the raw _probe_* methods)
        # so tests can patch the public API and caching stays composed.
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                name: pool.submit(getattr(cls, "detect_" + name))
                for name in ("modal", "kaggle", "docker", "gpu", "llm_status")
            }
        result = {name: fut.result() for name, fut in futures.items()}
        payload = {
            "modal": result["modal"],
            "kaggle": result["kaggle"],
            "docker": result["docker"],
            "gpu": result["gpu"],
            "llm": result["llm_status"],
        }
        if cache_path is not None:
            env_cache.disk_store(cache_path, payload)
        return payload

    # ------------------------------------------------------------------
    # Legacy / agent-facing aliases
    # ------------------------------------------------------------------

    @classmethod
    def detect_kaggle_environment(cls) -> Dict[str, Any]:
        """Alias for :meth:`detect_kaggle`.

        Agents frequently reach for the longer, more descriptive name; the
        underlying behaviour is identical.
        """
        return cls.detect_kaggle()

    @classmethod
    def is_kaggle_environment(cls) -> bool:
        """Return True when a Kaggle CLI is installed and authenticated.

        Convenience predicate that wraps :meth:`detect_kaggle` so agents do
        not have to inspect the returned dict themselves.
        """
        info = cls.detect_kaggle()
        return bool(info.get("installed") and info.get("authenticated"))

