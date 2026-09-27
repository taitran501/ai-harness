"""Unified LLM Execution, Prompting, and PEFT Engine for ML-Harness.

Provides comprehensive support for:
1. Multi-Provider API Execution (Gemini, OpenAI, Anthropic, DeepSeek, Groq, OpenRouter).
2. Local / Self-Hosted Models (Ollama, vLLM, HuggingFace transformers).
3. In-Context Learning (Zero-Shot, Few-Shot Demonstrations, Chain-of-Thought Reasoning).
4. Deterministic Output Extraction & Classification Parser (JSON, regex, categorical fallback).
5. Parameter-Efficient Fine-Tuning (PEFT / QLoRA) Standalone Script Generator.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CACHE: Dict[str, Tuple[float, Any]] = {}


def _safe_urlopen(req: Union[urllib.request.Request, str], timeout: float = 12.0):
    """Execute urlopen with default SSL, automatically falling back to unverified context if certificates fail."""
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        err_str = str(e).lower()
        if "certificate" in err_str or "ssl" in err_str or "verify failed" in err_str:
            try:
                unverified = ssl._create_unverified_context()
                return urllib.request.urlopen(req, timeout=timeout, context=unverified)
            except Exception:
                raise e
        raise e
    except Exception as e:
        err_str = str(e).lower()
        if "certificate" in err_str or "ssl" in err_str or "verify failed" in err_str:
            try:
                unverified = ssl._create_unverified_context()
                return urllib.request.urlopen(req, timeout=timeout, context=unverified)
            except Exception:
                pass
        raise e


def _get_cached(key: str, ttl: float = 120.0) -> Optional[Any]:
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < ttl:
            return val
    return None


def _set_cached(key: str, val: Any) -> None:
    _CACHE[key] = (time.time(), val)


@dataclass
class FewShotExample:
    """A few-shot demonstration example."""
    input_text: str
    target_output: str
    explanation: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for an LLM experiment or inference run."""
    provider: str = "gemini"  # gemini, openai, anthropic, deepseek, groq, openrouter, ollama, vllm, local_hf
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512
    system_prompt: str = ""
    prompt_template: str = "{text}"
    reasoning_effort: Optional[str] = None  # low, medium, high (for o-series / deepseek-r1)
    use_cot: bool = False  # Chain of Thought
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    output_format: str = "json"  # json, label_only, free_text
    valid_classes: List[str] = field(default_factory=list)
    api_base_url: Optional[str] = None


class LLMEngine:
    """Unified engine for LLM inference, prompting, and script generation."""

    PROVIDER_PRESETS = {
        "gemini": {
            "name": "Google Gemini",
            "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-pro"],
            "env_key": "GEMINI_API_KEY",
            "default_model": "gemini-2.0-flash",
        },
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "o3-mini", "o1", "gpt-4-turbo"],
            "env_key": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini",
        },
        "deepseek": {
            "name": "DeepSeek",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "env_key": "DEEPSEEK_API_KEY",
            "default_model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
        },
        "anthropic": {
            "name": "Anthropic Claude",
            "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            "env_key": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-5-haiku-20241022",
        },
        "groq": {
            "name": "Groq LPU",
            "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768", "gemma2-9b-it"],
            "env_key": "GROQ_API_KEY",
            "default_model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
        },
        "openrouter": {
            "name": "OpenRouter",
            "models": ["deepseek/deepseek-r1", "deepseek/deepseek-chat", "qwen/qwen-2.5-72b-instruct", "meta-llama/llama-3.3-70b-instruct", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-001"],
            "env_key": "OPENROUTER_API_KEY",
            "default_model": "deepseek/deepseek-r1",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "models": ["llama3.2", "qwen2.5:7b", "mistral", "deepseek-r1:7b"],
            "env_key": None,
            "default_model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
        },
        "vllm": {
            "name": "vLLM (Local / Self-hosted)",
            "models": ["default"],
            "env_key": None,
            "default_model": "default",
            "base_url": "http://localhost:8000/v1",
        },
    }

    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Retrieve API key for provider from environment or .env."""
        preset = cls.PROVIDER_PRESETS.get(provider, {})
        env_key = preset.get("env_key")
        if not env_key:
            return None

        val = os.environ.get(env_key)
        if val:
            return val

        # Secondary fallback: GOOGLE_API_KEY for gemini
        if provider == "gemini":
            val = os.environ.get("GOOGLE_API_KEY")
            if val:
                return val

        # Check local .env file
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{env_key}="):
                    return line.split("=", 1)[1].strip("\"' ")
                if provider == "gemini" and line.startswith("GOOGLE_API_KEY="):
                    return line.split("=", 1)[1].strip("\"' ")

        return None

    @classmethod
    def list_providers(cls) -> List[Dict[str, Any]]:
        """List all providers and their availability status."""
        results = []
        for pid, pdata in cls.PROVIDER_PRESETS.items():
            has_key = bool(cls.get_api_key(pid)) if pdata.get("env_key") else True
            # For local providers, check reachable endpoint
            is_reachable = has_key
            if pid in ("ollama", "vllm"):
                is_reachable = cls.check_local_endpoint(pdata.get("base_url", ""))

            results.append({
                "id": pid,
                "name": pdata["name"],
                "models": pdata["models"],
                "default_model": pdata["default_model"],
                "has_key": has_key,
                "is_reachable": is_reachable,
                "env_key_name": pdata.get("env_key"),
                "is_local": pid in ("ollama", "vllm"),
            })
        return results

    @staticmethod
    def check_local_endpoint(url: str, timeout: float = 1.0) -> bool:
        """Check if local Ollama / vLLM HTTP port is responding."""
        if not url:
            return False
        try:
            req = urllib.request.Request(url.replace("/v1", ""), headers={"User-Agent": "ML-Harness"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status < 500
        except Exception:
            return False

    CURATED_HF_FALLBACK_MODELS = [
        {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Qwen2.5-7B-Instruct",
            "author": "Qwen",
            "downloads": 2400000,
            "likes": 3600,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-11-01",
            "is_unsloth": False,
        },
        {
            "id": "meta-llama/Llama-3.3-70B-Instruct",
            "name": "Llama-3.3-70B-Instruct",
            "author": "meta-llama",
            "downloads": 1820000,
            "likes": 4400,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-12-06",
            "is_unsloth": False,
        },
        {
            "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "name": "DeepSeek-R1-Distill-Qwen-14B",
            "author": "deepseek-ai",
            "downloads": 2600000,
            "likes": 6100,
            "pipeline_tag": "text-generation",
            "last_modified": "2025-01-20",
            "is_unsloth": False,
        },
        {
            "id": "microsoft/phi-4",
            "name": "phi-4",
            "author": "microsoft",
            "downloads": 1250000,
            "likes": 2900,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-12-15",
            "is_unsloth": False,
        },
        {
            "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "name": "Meta-Llama-3.1-8B-Instruct",
            "author": "meta-llama",
            "downloads": 3900000,
            "likes": 7500,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-09-01",
            "is_unsloth": False,
        },
        {
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "name": "Mistral-7B-Instruct-v0.3",
            "author": "mistralai",
            "downloads": 1950000,
            "likes": 4200,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-05-22",
            "is_unsloth": False,
        },
        {
            "id": "google/gemma-2-9b-it",
            "name": "gemma-2-9b-it",
            "author": "google",
            "downloads": 1380000,
            "likes": 3100,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-06-27",
            "is_unsloth": False,
        },
        {
            "id": "FacebookAI/xlm-roberta-base",
            "name": "xlm-roberta-base",
            "author": "FacebookAI",
            "downloads": 5200000,
            "likes": 1950,
            "pipeline_tag": "fill-mask",
            "last_modified": "2024-01-10",
            "is_unsloth": False,
        },
        {
            "id": "microsoft/mdeberta-v3-base",
            "name": "mdeberta-v3-base",
            "author": "microsoft",
            "downloads": 2600000,
            "likes": 1250,
            "pipeline_tag": "fill-mask",
            "last_modified": "2023-11-05",
            "is_unsloth": False,
        },
        {
            "id": "microsoft/deberta-v3-large",
            "name": "deberta-v3-large",
            "author": "microsoft",
            "downloads": 3400000,
            "likes": 1800,
            "pipeline_tag": "fill-mask",
            "last_modified": "2023-10-12",
            "is_unsloth": False,
        },
        {
            "id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "name": "Qwen2.5-7B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 640000,
            "likes": 1950,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-10-15",
            "is_unsloth": True,
        },
        {
            "id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
            "name": "DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
            "author": "unsloth",
            "downloads": 580000,
            "likes": 2200,
            "pipeline_tag": "text-generation",
            "last_modified": "2025-01-22",
            "is_unsloth": True,
        },
        {
            "id": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            "name": "Llama-3.3-70B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 410000,
            "likes": 1600,
            "pipeline_tag": "text-generation",
            "last_modified": "2024-12-08",
            "is_unsloth": True,
        },
    ]

    CURATED_UNSLOTH_MODELS = [
        {
            "id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "name": "Qwen2.5-7B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 640000,
            "likes": 1950,
            "family": "Qwen",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
            "name": "DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
            "author": "unsloth",
            "downloads": 580000,
            "likes": 2200,
            "family": "DeepSeek",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            "name": "Llama-3.3-70B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 410000,
            "likes": 1600,
            "family": "Llama",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "name": "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 820000,
            "likes": 2800,
            "family": "Llama",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
            "name": "DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
            "author": "unsloth",
            "downloads": 490000,
            "likes": 1750,
            "family": "DeepSeek",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
            "name": "Qwen2.5-14B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 380000,
            "likes": 1400,
            "family": "Qwen",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "name": "mistral-7b-instruct-v0.3-bnb-4bit",
            "author": "unsloth",
            "downloads": 290000,
            "likes": 980,
            "family": "Mistral",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/gemma-2-9b-it-bnb-4bit",
            "name": "gemma-2-9b-it-bnb-4bit",
            "author": "unsloth",
            "downloads": 260000,
            "likes": 870,
            "family": "Gemma",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
            "name": "Phi-3.5-mini-instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 195000,
            "likes": 720,
            "family": "Phi",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
        {
            "id": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
            "name": "Qwen2.5-Coder-7B-Instruct-bnb-4bit",
            "author": "unsloth",
            "downloads": 310000,
            "likes": 1100,
            "family": "Qwen",
            "is_bnb_4bit": True,
            "is_gguf": False,
            "is_instruct": True,
            "recommended_for_lora": True,
        },
    ]

    CURATED_OPENROUTER_FALLBACK_MODELS = [
        {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "context_length": 128000, "is_live": True},
        {"id": "anthropic/claude-3.5-sonnet", "name": "Anthropic: Claude 3.5 Sonnet", "context_length": 200000, "is_live": True},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek: DeepSeek-V3", "context_length": 64000, "is_live": True},
        {"id": "deepseek/deepseek-r1", "name": "DeepSeek: DeepSeek-R1 (Reasoning)", "context_length": 64000, "is_live": True},
        {"id": "qwen/qwen-2.5-72b-instruct", "name": "Qwen: Qwen 2.5 72B Instruct", "context_length": 32768, "is_live": True},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Meta: Llama 3.3 70B Instruct", "context_length": 131072, "is_live": True},
        {"id": "google/gemini-2.0-flash-exp:free", "name": "Google: Gemini 2.0 Flash (Free Tier)", "context_length": 1048576, "is_live": True},
        {"id": "google/gemini-pro-1.5", "name": "Google: Gemini 1.5 Pro", "context_length": 2000000, "is_live": True},
        {"id": "mistralai/mistral-large-2407", "name": "Mistral: Mistral Large 2", "context_length": 128000, "is_live": True},
        {"id": "microsoft/phi-4", "name": "Microsoft: Phi-4", "context_length": 16384, "is_live": True},
    ]

    @classmethod
    def _get_fallback_hf_models(
        cls,
        query: str = "",
        pipeline_tag: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve curated fallback Hugging Face models if live API is unreachable or rate-limited."""
        pool = list(cls.CURATED_HF_FALLBACK_MODELS)
        if author == "unsloth":
            pool = [
                {
                    "id": m["id"],
                    "name": m["name"],
                    "author": "unsloth",
                    "downloads": m["downloads"],
                    "likes": m["likes"],
                    "pipeline_tag": "text-generation",
                    "last_modified": "2025-01-01",
                    "is_unsloth": True,
                }
                for m in cls.CURATED_UNSLOTH_MODELS
            ]

        results = []
        q = (query or "").lower().strip()
        for m in pool:
            if author and author.lower() not in (m.get("author") or "").lower():
                continue
            if pipeline_tag and m.get("pipeline_tag") and pipeline_tag.lower() != m.get("pipeline_tag", "").lower():
                continue
            if q and q not in m["id"].lower() and q not in m["name"].lower():
                continue
            results.append(m)

        if not results:
            results = pool

        return results[:limit]

    @classmethod
    def _get_fallback_unsloth_models(cls, limit: int = 40) -> List[Dict[str, Any]]:
        """Retrieve curated fallback Unsloth models."""
        return list(cls.CURATED_UNSLOTH_MODELS)[:limit]

    @staticmethod
    def _extract_model_attributes(m_id: str, raw_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract family, parameter size, and quantization format from model ID and metadata."""
        lower = m_id.lower()
        family = "Other"
        if "qwen" in lower:
            family = "Qwen"
        elif "llama" in lower:
            family = "Llama"
        elif "deepseek" in lower:
            family = "DeepSeek"
        elif "mistral" in lower or "mixtral" in lower:
            family = "Mistral"
        elif "gemma" in lower:
            family = "Gemma"
        elif "phi" in lower:
            family = "Phi"
        elif any(k in lower for k in ["deberta", "roberta", "xlm", "bert", "electra"]):
            family = "Encoder"

        # Extract parameter size (e.g. 7B, 14B, 27B, 0.5B, 1.5B, 70B)
        param_size_b: Optional[float] = None
        param_label: str = "—"
        match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*[bB](?:$|[-_/])", m_id)
        if match:
            try:
                param_size_b = float(match.group(1))
                param_label = f"{match.group(1)}B"
            except ValueError:
                pass
        elif raw_obj and isinstance(raw_obj.get("safetensors"), dict):
            total_params = raw_obj["safetensors"].get("total")
            if isinstance(total_params, (int, float)) and total_params > 0:
                param_size_b = round(total_params / 1e9, 1)
                param_label = f"{param_size_b}B" if param_size_b >= 1.0 else f"{int(total_params / 1e6)}M"

        # Extract quantization format
        is_gguf = "gguf" in lower
        is_4bit = any(k in lower for k in ["bnb-4bit", "4bit", "awq", "gptq", "nvfp4", "fp4", "nf4", "int4"])
        is_8bit = any(k in lower for k in ["8bit", "fp8", "int8"])
        if is_gguf:
            quantization = "GGUF"
        elif is_4bit:
            quantization = "4-bit"
        elif is_8bit:
            quantization = "8-bit"
        else:
            quantization = "FP16/BF16"

        is_instruct = any(k in lower for k in ["instruct", "-it", "_it", "chat"])
        return {
            "family": family,
            "param_size_b": param_size_b,
            "param_label": param_label,
            "quantization": quantization,
            "is_bnb_4bit": is_4bit,
            "is_gguf": is_gguf,
            "is_instruct": is_instruct,
            "recommended_for_lora": is_4bit or (not is_gguf and is_instruct),
        }

    @staticmethod
    def _matches_filters(
        item: Dict[str, Any],
        family: Optional[str] = None,
        param_range: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> bool:
        if family and family.lower() not in ("all", ""):
            if item.get("family", "").lower() != family.lower():
                return False
        if quantization and quantization.lower() not in ("all", ""):
            if item.get("quantization", "").lower() != quantization.lower():
                return False
        if param_range and param_range.lower() not in ("all", ""):
            p = item.get("param_size_b")
            if p is None:
                return False
            if param_range == "<3B" and not (p < 3.0):
                return False
            elif param_range == "3B-8B" and not (3.0 <= p <= 8.5):
                return False
            elif param_range == "8B-14B" and not (8.5 < p <= 15.0):
                return False
            elif param_range == "14B-32B" and not (15.0 < p <= 35.0):
                return False
            elif param_range == ">32B" and not (p > 35.0):
                return False
        return True

    @classmethod
    def fetch_hf_models(
        cls,
        query: str = "",
        pipeline_tag: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 50,
        sort: str = "downloads",
        family: Optional[str] = None,
        param_range: Optional[str] = None,
        quantization: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch real-time model list from Hugging Face Hub API with attribute filtering."""
        cache_key = f"hf:{query}:{pipeline_tag}:{author}:{limit}:{sort}:{family}:{param_range}:{quantization}"
        if not force_refresh:
            cached = _get_cached(cache_key, ttl=20.0)
            if cached is not None:
                return cached

        fetch_limit = min(max(limit * 2, 60), 100)
        params = [f"limit={fetch_limit}", "full=true"]
        if sort:
            params.append(f"sort={sort}&direction=-1")
        search_terms = [t for t in [query, family if family and family.lower() not in ("all", "other", "encoder") else ""] if t]
        if search_terms:
            params.append(f"search={urllib.parse.quote(' '.join(search_terms))}")
        if pipeline_tag and pipeline_tag.lower() not in ("all", ""):
            params.append(f"pipeline_tag={urllib.parse.quote(pipeline_tag)}")
        if author:
            params.append(f"author={urllib.parse.quote(author)}")

        url = f"https://huggingface.co/api/models?{'&'.join(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ML-Harness/2.0"},
        )
        try:
            with _safe_urlopen(req, timeout=12.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            for m in data:
                m_id = m.get("id") or m.get("modelId")
                if not m_id:
                    continue
                attrs = cls._extract_model_attributes(m_id, m)
                item = {
                    "id": m_id,
                    "name": m_id.split("/")[-1],
                    "author": m_id.split("/")[0] if "/" in m_id else None,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "last_modified": (m.get("lastModified") or "")[:10],
                    "is_unsloth": "unsloth" in m_id.lower(),
                    **attrs,
                }
                if cls._matches_filters(item, family=family, param_range=param_range, quantization=quantization):
                    results.append(item)
            if not results and not (query or family or param_range or quantization):
                fallback_raw = cls._get_fallback_hf_models(query=query, pipeline_tag=pipeline_tag, author=author, limit=limit)
                results = [{**m, **cls._extract_model_attributes(m["id"], m)} for m in fallback_raw]
            results = results[:limit]
            _set_cached(cache_key, results)
            return results
        except Exception as e:
            print(f"Warning: Failed to fetch Hugging Face models ({e}), returning fallback catalog.")
            fallback = cls._get_fallback_hf_models(query=query, pipeline_tag=pipeline_tag, author=author, limit=limit)
            enriched = [
                {**m, **cls._extract_model_attributes(m["id"], m)}
                for m in fallback
            ]
            filtered = [
                item for item in enriched
                if cls._matches_filters(item, family=family, param_range=param_range, quantization=quantization)
            ][:limit]
            _set_cached(cache_key, filtered)
            return filtered

    @classmethod
    def fetch_unsloth_models(
        cls,
        limit: int = 50,
        query: str = "",
        family: Optional[str] = None,
        param_range: Optional[str] = None,
        quantization: Optional[str] = None,
        sort: str = "downloads",
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch real-time Unsloth models directly from Hugging Face Hub API with filtering."""
        return cls.fetch_hf_models(
            query=query,
            author="unsloth",
            limit=limit,
            sort=sort,
            family=family,
            param_range=param_range,
            quantization=quantization,
            force_refresh=force_refresh,
        )

    @classmethod
    def fetch_provider_live_models(cls, provider: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Dynamically fetch live supported models from the provider API if key available, else return verified presets."""
        preset = cls.PROVIDER_PRESETS.get(provider, {})
        key = api_key or cls.get_api_key(provider)

        # 1. OpenRouter (No API key strictly required for public catalog!)
        if provider == "openrouter":
            cache_key = "openrouter_models"
            cached = _get_cached(cache_key, ttl=300.0)
            if cached:
                return cached
            try:
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/models",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ML-Harness/2.0"},
                )
                with _safe_urlopen(req, timeout=12.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                models = []
                for m in data.get("data", []):
                    m_id = m.get("id")
                    if m_id:
                        models.append({
                            "id": m_id,
                            "name": m.get("name", m_id),
                            "context_length": m.get("context_length"),
                            "description": m.get("description", "")[:120],
                            "is_live": True,
                        })
                popular_keywords = ["deepseek", "qwen", "claude", "gpt-4o", "llama-3.3", "gemini-2.0"]
                models.sort(key=lambda x: any(k in x["id"].lower() for k in popular_keywords), reverse=True)
                if models:
                    _set_cached(cache_key, models)
                    return models
            except Exception as e:
                print(f"⚠️ Warning: OpenRouter live model catalog fetch: {e}")

            # Return curated OpenRouter models
            return cls.CURATED_OPENROUTER_FALLBACK_MODELS

        # 2. Ollama local daemon
        if provider == "ollama":
            url = f"{preset.get('base_url', 'http://localhost:11434/v1').replace('/v1', '')}/api/tags"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ML-Harness"})
                with _safe_urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                local_models = []
                for m in data.get("models", []):
                    name = m.get("name")
                    if name:
                        local_models.append({
                            "id": name,
                            "name": name,
                            "size_bytes": m.get("size", 0),
                            "is_live": True,
                        })
                if local_models:
                    return local_models
            except Exception:
                pass

        # 3. Google Gemini (if API key present)
        if provider == "gemini" and key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ML-Harness"})
                with _safe_urlopen(req, timeout=8.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                live_models = []
                for m in data.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        m_id = m.get("name", "").replace("models/", "")
                        live_models.append({
                            "id": m_id,
                            "name": m.get("displayName", m_id),
                            "description": m.get("description", "")[:120],
                            "is_live": True,
                        })
                if live_models:
                    return live_models
            except Exception:
                pass

        # 4. Groq (if API key present)
        if provider == "groq" and key:
            try:
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0 ML-Harness"}
                )
                with _safe_urlopen(req, timeout=8.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                live_models = []
                for m in data.get("data", []):
                    m_id = m.get("id")
                    if m_id and m.get("active", True):
                        live_models.append({
                            "id": m_id,
                            "name": m_id,
                            "context_window": m.get("context_window"),
                            "is_live": True,
                        })
                if live_models:
                    return live_models
            except Exception:
                pass

        # 5. OpenAI (if API key present)
        if provider == "openai" and key:
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0 ML-Harness"}
                )
                with _safe_urlopen(req, timeout=8.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                live_models = []
                for m in data.get("data", []):
                    m_id = m.get("id")
                    if m_id and any(p in m_id for p in ["gpt-4", "o1", "o3", "chatgpt"]):
                        live_models.append({
                            "id": m_id,
                            "name": m_id,
                            "is_live": True,
                        })
                if live_models:
                    live_models.sort(key=lambda x: ("gpt-4o" in x["id"] or "o3" in x["id"] or "o1" in x["id"]), reverse=True)
                    return live_models
            except Exception:
                pass

        # Fallback to verified documentation presets
        return [{"id": m, "name": m, "is_live": False} for m in preset.get("models", [])]


    @classmethod
    def format_prompt(
        cls,
        template: str,
        row_data: Dict[str, Any],
        system_prompt: str = "",
        few_shots: Optional[List[Dict[str, str]]] = None,
        use_cot: bool = False,
        valid_classes: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Format complete system prompt and user prompt."""
        # 1. Substitute variables in user template
        formatted_user = template
        for k, v in row_data.items():
            formatted_user = formatted_user.replace(f"{{{k}}}", str(v if v is not None else ""))

        # 2. Append Chain of Thought guidance if enabled
        effective_sys = system_prompt
        if use_cot:
            cot_instruction = (
                "\nThink step by step. First explain your reasoning carefully, then output your final decision."
            )
            effective_sys = (effective_sys + cot_instruction).strip()

        # 3. Add valid classes constraint if provided
        if valid_classes:
            class_list_str = ", ".join(f"'{c}'" for c in valid_classes)
            classes_instruction = (
                f"\nYour final answer MUST strictly be one of the following classes: [{class_list_str}]."
                "\nProvide output formatted as JSON: {\"reasoning\": \"...\", \"prediction\": \"<CLASS>\"}"
            )
            effective_sys = (effective_sys + classes_instruction).strip()

        # 4. Integrate few-shot examples
        if few_shots and len(few_shots) > 0:
            examples_text = "\n\n### Examples:\n"
            for i, ex in enumerate(few_shots, 1):
                inp = ex.get("input", ex.get("input_text", ""))
                out = ex.get("output", ex.get("target_output", ""))
                exp = ex.get("explanation", "")
                examples_text += f"\nExample {i}:\nInput: {inp}\n"
                if exp:
                    examples_text += f"Reasoning: {exp}\n"
                examples_text += f"Output: {out}\n"
            formatted_user = examples_text + "\n### Current Task:\nInput: " + formatted_user

        return effective_sys, formatted_user

    @classmethod
    def generate_completion(
        cls,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 0.95,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response from any supported LLM provider with timing and token usage."""
        key = api_key or cls.get_api_key(provider)
        start_time = time.time()

        # Route by provider
        if provider == "gemini":
            return cls._call_gemini(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=key,
                start_time=start_time,
            )
        elif provider in ("openai", "deepseek", "groq", "openrouter", "ollama", "vllm"):
            return cls._call_openai_compatible(
                provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                api_key=key,
                custom_base_url=base_url,
                start_time=start_time,
            )
        elif provider == "anthropic":
            return cls._call_anthropic(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=key,
                start_time=start_time,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def _call_gemini(
        cls,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Call Google Gemini API via google-genai or direct REST."""
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured. Please set it in Environment Settings.")

        # Try google-genai SDK if available
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            config_dict = {"temperature": temperature, "max_output_tokens": max_tokens}
            if system_prompt:
                config_dict["system_instruction"] = system_prompt

            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config_dict,
            )
            raw_text = response.text or ""
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "text": raw_text,
                "latency_ms": latency_ms,
                "provider": "gemini",
                "model": model_name,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
                },
            }
        except Exception:
            pass

        # Fallback to direct REST API
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _safe_urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                raw_text = parts[0].get("text", "")

        latency_ms = int((time.time() - start_time) * 1000)
        usage = data.get("usageMetadata", {})
        return {
            "text": raw_text,
            "latency_ms": latency_ms,
            "provider": "gemini",
            "model": model_name,
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
            },
        }

    @classmethod
    def _call_openai_compatible(
        cls,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        api_key: Optional[str],
        custom_base_url: Optional[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Call OpenAI or any OpenAI-compatible API (DeepSeek, Groq, OpenRouter, Ollama, vLLM)."""
        preset = cls.PROVIDER_PRESETS.get(provider, {})
        base_url = custom_base_url or preset.get("base_url") or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _safe_urlopen(req, timeout=40.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_text = ""
        choices = data.get("choices", [])
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "text": raw_text,
            "latency_ms": latency_ms,
            "provider": provider,
            "model": model_name,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        }

    @classmethod
    def _call_anthropic(
        cls,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Call Anthropic Messages API."""
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")

        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with _safe_urlopen(req, timeout=40.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_text = ""
        content_parts = data.get("content", [])
        if content_parts and isinstance(content_parts, list):
            raw_text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")

        usage = data.get("usage", {})
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "text": raw_text,
            "latency_ms": latency_ms,
            "provider": "anthropic",
            "model": model_name,
            "usage": {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            },
        }

    @classmethod
    def extract_prediction(cls, response_text: str, valid_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Robustly extract structured prediction, reasoning, and metadata from raw LLM output."""
        clean_text = response_text.strip()
        parsed_label = None
        reasoning = ""

        # 1. Try finding JSON block
        json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", clean_text, re.DOTALL)
        candidate_json = json_match.group(1) if json_match else clean_text

        # Try parsing JSON
        try:
            data = json.loads(candidate_json)
            if isinstance(data, dict):
                for k in ["prediction", "label", "class", "target", "answer", "output"]:
                    if k in data:
                        parsed_label = str(data[k]).strip()
                        break
                reasoning = data.get("reasoning") or data.get("explanation") or ""
        except Exception:
            pass

        # 2. If valid_classes provided, match via regex
        if valid_classes:
            lowered_text = clean_text.lower()
            if not parsed_label or parsed_label.lower() not in [c.lower() for c in valid_classes]:
                # Search exact words in text
                for c in sorted(valid_classes, key=len, reverse=True):
                    pattern = r"\b" + re.escape(c.lower()) + r"\b"
                    if re.search(pattern, lowered_text):
                        parsed_label = c
                        break

        # 3. Fallback: first non-empty word
        if not parsed_label:
            words = clean_text.split()
            parsed_label = words[0] if words else "UNKNOWN"

        return {
            "parsed_label": parsed_label,
            "reasoning": reasoning,
            "raw_text": clean_text,
            "is_valid": bool(valid_classes and parsed_label in valid_classes) if valid_classes else True,
        }

    @classmethod
    def generate_qlora_script(
        cls,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        train_path: str = "dataset/train.csv",
        output_dir: str = "artifacts/exp_qlora",
        text_column: str = "text",
        target_column: str = "target",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        num_train_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
    ) -> str:
        """Generate standalone, production-ready PyTorch + Hugging Face TRL QLoRA fine-tuning script."""
        modules_list = [f'"{m.strip()}"' for m in target_modules.split(",") if m.strip()]
        modules_str = f"[{', '.join(modules_list)}]"

        return f'''#!/usr/bin/env python3
"""Automated QLoRA Parameter-Efficient Fine-Tuning Script.
Model: {model_name} | Dataset: {train_path}
Generated by ML-Harness LLM Studio.
"""

from __future__ import annotations

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    model_id = "{model_name}"
    output_dir = "{output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Initializing QLoRA Fine-Tuning for: {{model_id}}")

    # 1. 4-bit NormalFloat Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Base Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Adapter Configuration
    peft_config = LoraConfig(
        r={lora_r},
        lora_alpha={lora_alpha},
        lora_dropout={lora_dropout},
        bias="none",
        task_type="CAUSAL_LM",
        target_modules={modules_str},
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    df = pd.read_csv("{train_path}") if "{train_path}".endswith(".csv") else pd.read_csv("{train_path}", sep="\\t")
    def format_prompt(row):
        return f"<|im_start|>user\\n{{row.get('{text_column}', '')}}<|im_end|>\\n<|im_start|>assistant\\n{{row.get('{target_column}', '')}}<|im_end|>"

    df["prompt_formatted"] = df.apply(format_prompt, axis=1)
    dataset = Dataset.from_pandas(df[["prompt_formatted"]])

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs={num_train_epochs},
        per_device_train_batch_size={batch_size},
        gradient_accumulation_steps={gradient_accumulation_steps},
        learning_rate={learning_rate},
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="prompt_formatted",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("⚡ Starting Training...")
    trainer.train()

    # Save fine-tuned adapter weights
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 Fine-Tuning Complete! Adapter saved at: {{output_dir}}")

if __name__ == "__main__":
    main()
'''

    @classmethod
    def generate_unsloth_script(
        cls,
        model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        train_path: str = "dataset/train.csv",
        output_dir: str = "artifacts/exp_unsloth_lora",
        text_column: str = "text",
        target_column: str = "target",
        max_seq_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        num_train_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
    ) -> str:
        """Generate standalone, high-performance Unsloth 4-bit LoRA fine-tuning script.
        
        Unsloth achieves 2x-5x faster training and ~80% less memory than standard Hugging Face PEFT.
        """
        modules_list = [f'"{m.strip()}"' for m in target_modules.split(",") if m.strip()]
        modules_str = f"[{', '.join(modules_list)}]"

        return f'''#!/usr/bin/env python3
\"\"\"Automated Unsloth Fast LoRA Parameter-Efficient Fine-Tuning Script.
Model: {model_name} | Dataset: {train_path}
Generated by ML-Harness LLM & Unsloth Studio.
Powered by Unsloth (https://github.com/unslothai/unsloth).
\"\"\"

from __future__ import annotations

import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    model_id = "{model_name}"
    output_dir = "{output_dir}"
    max_seq_length = {max_seq_length}
    dtype = None # Auto detection: Float16 for Tesla T4/V100, Bfloat16 for Ampere+
    load_in_4bit = True

    os.makedirs(output_dir, exist_ok=True)
    print(f"🦥 Initializing Unsloth FastLanguageModel for: {{model_id}}")

    # 1. Load Base Model & Fast Tokenizer (2x faster download & initialization)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # 2. Add Unsloth Fast LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r={lora_r},
        target_modules={modules_str},
        lora_alpha={lora_alpha},
        lora_dropout={lora_dropout},
        bias="none",
        use_gradient_checkpointing="unsloth", # 4x longer context, saves ~80% VRAM!
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Prepare Dataset
    df = pd.read_csv("{train_path}") if "{train_path}".endswith(".csv") else pd.read_csv("{train_path}", sep="\\t")
    def format_prompt(row):
        return f"<|im_start|>user\\n{{row.get('{text_column}', '')}}<|im_end|>\\n<|im_start|>assistant\\n{{row.get('{target_column}', '')}}<|im_end|>"

    df["prompt_formatted"] = df.apply(format_prompt, axis=1)
    dataset = Dataset.from_pandas(df[["prompt_formatted"]])

    # 4. Training Arguments with Unsloth Optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size={batch_size},
        gradient_accumulation_steps={gradient_accumulation_steps},
        warmup_steps=5,
        num_train_epochs={num_train_epochs},
        learning_rate={learning_rate},
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    )

    # 5. Fast SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="prompt_formatted",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("⚡ Starting Unsloth Accelerated Training...")
    trainer_stats = trainer.train()

    # 6. Save Fast LoRA Adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 Unsloth LoRA Fine-Tuning Complete! Adapter saved at: {{output_dir}}")

if __name__ == "__main__":
    main()
'''

