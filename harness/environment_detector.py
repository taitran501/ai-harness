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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent



class EnvironmentDetector:
    """Detects and manages environments and accounts across Modal, Kaggle, Docker, and GPU."""

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
        """Open Kaggle Account settings web page in browser."""
        auth_url = "https://www.kaggle.com/settings"
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
            "message": "Opened Kaggle Account settings (https://www.kaggle.com/settings) in your browser.",
        }

    @staticmethod
    def detect_kaggle() -> Dict[str, Any]:
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

        # Check credentials in ~/.kaggle/credentials.json or ~/.kaggle/kaggle.json
        kaggle_dir = Path.home() / ".kaggle"
        for cred_name in ["credentials.json", "kaggle.json"]:
            cred_file = kaggle_dir / cred_name
            if cred_file.exists():
                try:
                    data = json.loads(cred_file.read_text(encoding="utf-8"))
                    username = data.get("username") or data.get("user")
                    if username:
                        result["authenticated"] = True
                        result["username"] = username
                        break
                except Exception:
                    pass

        # If still not found, execute kaggle config view
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

        # Also test torch cuda if available
        try:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            if result["cuda_available"] and not result["gpu_name"]:
                result["detected"] = True
                result["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return result

    @staticmethod
    def detect_llm_status() -> Dict[str, Any]:
        """Detect LLM API keys and local inference endpoints."""
        keys = {}
        for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]:
            val = os.environ.get(k)
            if not val:
                # check .env
                env_file = PROJECT_ROOT / ".env"
                if env_file.exists():
                    for line in env_file.read_text(encoding="utf-8").splitlines():
                        if line.strip().startswith(f"{k}="):
                            val = line.strip().split("=", 1)[1].strip("\"' ")
                            break
            keys[k] = {
                "set": bool(val),
                "preview": f"{val[:6]}...{val[-4:]}" if val and len(val) > 10 else ("***" if val else None),
            }

        ollama_active = False
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "ML-Harness"})
            with urllib.request.urlopen(req, timeout=0.5) as r:
                ollama_active = r.status < 500
        except Exception:
            pass

        vllm_active = False
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:8000/v1/models", headers={"User-Agent": "ML-Harness"})
            with urllib.request.urlopen(req, timeout=0.5) as r:
                vllm_active = r.status < 500
        except Exception:
            pass

        active_count = sum(1 for v in keys.values() if v["set"])
        if ollama_active:
            active_count += 1
        if vllm_active:
            active_count += 1

        return {
            "keys": keys,
            "ollama_active": ollama_active,
            "vllm_active": vllm_active,
            "active_providers_count": active_count,
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

