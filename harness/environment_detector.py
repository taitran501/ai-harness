"""Environment and Multi-Account Detection Module for ML-Harness.

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
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    _kaggle_auth_lock = threading.Lock()
    _kaggle_auth_process: Optional[subprocess.Popen] = None
    _kaggle_auth_log: Any = None
    _kaggle_auth_state: Dict[str, Any] = {
        "status": "idle",
        "command": None,
        "started_at": None,
        "finished_at": None,
        "return_code": None,
    }

    @staticmethod
    def detect_modal() -> Dict[str, Any]:
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
    def activate_modal_profile(profile_name: str) -> Dict[str, Any]:
        """Activate a specific Modal profile."""
        try:
            proc = subprocess.run(
                ["modal", "profile", "activate", profile_name],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            success = proc.returncode == 0
            return {
                "success": success,
                "profile": profile_name,
                "output": proc.stdout.strip() or proc.stderr.strip(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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

    @classmethod
    def set_modal_token(
        cls,
        token_id: str,
        token_secret: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set or update Modal credentials with smart parsing."""
        try:
            # If user pasted combined string into token_id or secret is empty
            if not token_secret or "ak-" in token_id and "as-" in token_id:
                extracted_id, extracted_sec = cls.parse_modal_token_input(f"{token_id} {token_secret or ''}")
                if extracted_id and extracted_sec:
                    token_id = extracted_id
                    token_secret = extracted_sec

            if not token_id or not token_secret:
                return {
                    "success": False,
                    "error": "Could not identify valid ak-... token ID and as-... token secret.",
                }

            cmd = ["modal", "token", "set", "--token-id", token_id, "--token-secret", token_secret]
            if profile_name:
                cmd.extend(["--profile", profile_name])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
            return {
                "success": proc.returncode == 0,
                "output": proc.stdout.strip() or proc.stderr.strip(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
        """Start Kaggle's official forced OAuth flow; Kaggle opens its authorization URL."""
        kaggle_path = shutil.which("kaggle")
        command = [kaggle_path or "kaggle", "auth", "login", "--force"]
        if not kaggle_path:
            return {
                "success": False,
                "status": "cli_missing",
                "command": "kaggle auth login --force",
                "message": "Kaggle CLI is not installed or is not on PATH.",
            }

        try:
            help_result = subprocess.run(
                [kaggle_path, "auth", "login", "--help"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            help_text = (help_result.stdout or "") + "\n" + (help_result.stderr or "")
            if help_result.returncode != 0 or "--force" not in help_text:
                return {
                    "success": False,
                    "status": "oauth_unsupported",
                    "command": "kaggle auth login --force",
                    "message": "The installed Kaggle CLI does not expose OAuth login --force. Upgrade the official Kaggle CLI and retry.",
                }
        except Exception as exc:
            return {
                "success": False,
                "status": "oauth_unavailable",
                "command": "kaggle auth login --force",
                "message": f"Could not verify Kaggle OAuth support: {exc}",
            }

        with EnvironmentDetector._kaggle_auth_lock:
            current = EnvironmentDetector._kaggle_auth_process
            if current is not None and current.poll() is None:
                return {
                    "success": True,
                    "status": "running",
                    "command": "kaggle auth login --force",
                    "message": "Kaggle OAuth is already running. Approve access in the browser window it opened.",
                }

            if EnvironmentDetector._kaggle_auth_log is not None:
                try:
                    EnvironmentDetector._kaggle_auth_log.close()
                except Exception:
                    pass
            log_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                log_file.close()
                return {
                    "success": False,
                    "status": "failed",
                    "command": "kaggle auth login --force",
                    "message": f"Could not start Kaggle OAuth: {exc}",
                }

            EnvironmentDetector._kaggle_auth_process = process
            EnvironmentDetector._kaggle_auth_log = log_file
            EnvironmentDetector._kaggle_auth_state = {
                "status": "running",
                "command": "kaggle auth login --force",
                "started_at": time.time(),
                "finished_at": None,
                "return_code": None,
            }

        return {
            "success": True,
            **EnvironmentDetector.get_kaggle_auth_flow_status(),
            "message": "Kaggle OAuth started. Approve access on the Kaggle page opened by the CLI.",
        }

    @classmethod
    def get_kaggle_auth_flow_status(cls) -> Dict[str, Any]:
        """Return the current OAuth subprocess state without exposing its output or credentials."""
        with cls._kaggle_auth_lock:
            process = cls._kaggle_auth_process
            if process is not None:
                return_code = process.poll()
                if return_code is not None and cls._kaggle_auth_state.get("status") == "running":
                    cls._kaggle_auth_state = {
                        **cls._kaggle_auth_state,
                        "status": "completed" if return_code == 0 else "failed",
                        "finished_at": time.time(),
                        "return_code": return_code,
                    }
                    cls._kaggle_auth_process = None
                    if cls._kaggle_auth_log is not None:
                        try:
                            cls._kaggle_auth_log.close()
                        except Exception:
                            pass
                        cls._kaggle_auth_log = None
            return dict(cls._kaggle_auth_state)

    @staticmethod
    def detect_kaggle() -> Dict[str, Any]:
        """Detect Kaggle CLI, active account, configured auth source, and OAuth flow state."""
        kaggle_path = shutil.which("kaggle")
        configured_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        if configured_dir:
            kaggle_dir = Path(configured_dir).expanduser()
        else:
            default_dir = Path.home() / ".kaggle"
            xdg_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "kaggle"
            kaggle_dir = default_dir if default_dir.exists() or os.name == "nt" else xdg_dir
        result: Dict[str, Any] = {
            "installed": kaggle_path is not None,
            "executable": kaggle_path,
            "authenticated": False,
            "username": None,
            "config_dir": str(kaggle_dir.resolve()),
            "cli_version": None,
            "oauth_supported": False,
            "authentication_method": None,
            "connection_status": "cli_missing" if not kaggle_path else "not_connected",
            "profiles": [],
            "error": None,
        }

        # Match Kaggle CLI credential precedence while never returning token values.
        if os.environ.get("KAGGLE_API_TOKEN"):
            result["authenticated"] = True
            result["authentication_method"] = "KAGGLE_API_TOKEN"
        elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            result["authenticated"] = True
            result["authentication_method"] = "KAGGLE_USERNAME / KAGGLE_KEY"

        if kaggle_dir.is_dir():
            access_token = kaggle_dir / "access_token"
            try:
                if access_token.is_file() and access_token.stat().st_size > 0:
                    result["authenticated"] = True
                    result["authentication_method"] = result["authentication_method"] or "OAuth access token file"
            except OSError:
                pass
            for cred_name in ("credentials.json", "kaggle.json"):
                cred_file = kaggle_dir / cred_name
                if not cred_file.is_file():
                    continue
                try:
                    data = json.loads(cred_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict):
                    continue
                username = data.get("username") or data.get("user") or data.get("user_name")
                if username and not result["username"]:
                    result["username"] = str(username)
                if data.get("refresh_token") or data.get("access_token"):
                    result["authenticated"] = True
                    result["authentication_method"] = result["authentication_method"] or "OAuth credentials"
                elif data.get("username") and data.get("key"):
                    result["authenticated"] = True
                    result["authentication_method"] = result["authentication_method"] or "Legacy API key"

        if os.environ.get("KAGGLE_USERNAME"):
            result["username"] = os.environ["KAGGLE_USERNAME"]

        if result["installed"]:
            try:
                version_proc = subprocess.run(
                    [kaggle_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if version_proc.returncode == 0:
                    result["cli_version"] = (version_proc.stdout or version_proc.stderr).strip()[:200]
            except Exception:
                pass

            try:
                auth_help = subprocess.run(
                    [kaggle_path, "auth", "login", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
                help_text = (auth_help.stdout or "") + "\n" + (auth_help.stderr or "")
                result["oauth_supported"] = auth_help.returncode == 0 and "--force" in help_text
            except Exception:
                pass

            # config view also reports the active OAuth account on current Kaggle CLI.
            try:
                proc = subprocess.run(
                    [kaggle_path, "config", "view"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
                if proc.returncode == 0:
                    for line in proc.stdout.splitlines():
                        if ":" not in line:
                            continue
                        key, value = (part.strip().strip("\"'") for part in line.split(":", 1))
                        normalized_key = key.casefold().replace(" ", "_")
                        if normalized_key in {"username", "user", "user_name"} and value and value.casefold() not in {"none", "null"}:
                            result["username"] = value
                        elif normalized_key in {"auth_method", "authentication_method", "auth"} and value and value.casefold() not in {"none", "null"}:
                            result["authentication_method"] = value
                elif not result["error"]:
                    result["error"] = "Kaggle CLI could not read the active account configuration."
            except Exception:
                pass

        if result["authentication_method"] and str(result["authentication_method"]).casefold() not in {
            "none", "null", "unauthenticated", "not connected"
        }:
            result["authenticated"] = True

        if result["authenticated"]:
            result["connection_status"] = "credentials_detected" if not result["username"] else "account_detected"
        if not result["authentication_method"] and result["authenticated"]:
            result["authentication_method"] = "Kaggle CLI credentials"

        result["auth_flow"] = EnvironmentDetector.get_kaggle_auth_flow_status()

        # Preserve legacy profile display for installations that still use archived profiles.
        profiles_dir = kaggle_dir / "profiles"
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
    def detect_docker() -> Dict[str, Any]:
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
    def detect_gpu() -> Dict[str, Any]:
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

        # Also test torch cuda or mps if available
        try:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            if result["cuda_available"] and not result["gpu_name"]:
                result["detected"] = True
                result["gpu_name"] = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                result["mps_available"] = True
                if not result["gpu_name"]:
                    result["detected"] = True
                    result["gpu_name"] = "Apple Silicon GPU (MPS)"
        except Exception:
            pass

        return result

    @classmethod
    def detect_llm_status(cls) -> Dict[str, Any]:
        """Detect LLM API keys, configured providers, and local inference endpoints."""
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
                req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "ML-Harness/2.0"})
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
                req = urllib.request.Request("http://localhost:8000/v1/models", headers={"User-Agent": "ML-Harness/2.0"})
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ML-Harness/2.0",
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
    def get_full_status(cls) -> Dict[str, Any]:
        """Aggregate full environment and hardware status."""
        return {
            "modal": cls.detect_modal(),
            "kaggle": cls.detect_kaggle(),
            "docker": cls.detect_docker(),
            "gpu": cls.detect_gpu(),
            "llm": cls.detect_llm_status(),
        }

