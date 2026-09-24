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
            "models": ["llama3.2", "qwen2.5", "deepseek-r1:7b", "mistral", "phi4"],
            "env_key": None,
            "default_model": "qwen2.5",
            "base_url": "http://localhost:11434/v1",
        },
        "vllm": {
            "name": "vLLM (Local / Cluster)",
            "models": ["default-model"],
            "env_key": None,
            "default_model": "default-model",
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

    @classmethod
    def fetch_hf_models(
        cls,
        query: str = "",
        pipeline_tag: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 20,
        sort: str = "downloads",
    ) -> List[Dict[str, Any]]:
        """Fetch real-time model list from Hugging Face Hub API with caching."""
        cache_key = f"hf:{query}:{pipeline_tag}:{author}:{limit}:{sort}"
        cached = _get_cached(cache_key, ttl=180.0)
        if cached is not None:
            return cached

        params = [f"limit={min(max(limit, 1), 100)}"]
        if sort:
            params.append(f"sort={sort}&direction=-1")
        if query:
            params.append(f"search={urllib.parse.quote(query)}")
        if pipeline_tag:
            params.append(f"pipeline_tag={urllib.parse.quote(pipeline_tag)}")
        if author:
            params.append(f"author={urllib.parse.quote(author)}")

        url = f"https://huggingface.co/api/models?{'&'.join(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ML-Harness"})
        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            for m in data:
                m_id = m.get("id") or m.get("modelId")
                if not m_id:
                    continue
                results.append({
                    "id": m_id,
                    "name": m_id.split("/")[-1],
                    "author": m_id.split("/")[0] if "/" in m_id else None,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "last_modified": m.get("lastModified"),
                    "is_unsloth": "unsloth" in m_id.lower(),
                })
            _set_cached(cache_key, results)
            return results
        except Exception as e:
            print(f"⚠️ Warning: Failed to fetch Hugging Face models: {e}")
            return []

    @classmethod
    def fetch_unsloth_models(cls, limit: int = 40) -> List[Dict[str, Any]]:
        """Fetch curated and latest Unsloth 4-bit and Instruct models directly from HF Hub."""
        cache_key = f"unsloth:{limit}"
        cached = _get_cached(cache_key, ttl=300.0)
        if cached is not None:
            return cached

        raw = cls.fetch_hf_models(author="unsloth", limit=100, sort="downloads")
        results = []
        for m in raw:
            m_id = m["id"]
            is_gguf = "gguf" in m_id.lower()
            is_bnb = "bnb-4bit" in m_id.lower() or "unsloth-bnb" in m_id.lower() or "nvfp4" in m_id.lower()
            is_instruct = "instruct" in m_id.lower() or "it" in m_id.lower()

            family = "Other"
            lower = m_id.lower()
            if "qwen" in lower:
                family = "Qwen"
            elif "llama" in lower:
                family = "Llama"
            elif "deepseek" in lower:
                family = "DeepSeek"
            elif "mistral" in lower:
                family = "Mistral"
            elif "gemma" in lower:
                family = "Gemma"
            elif "phi" in lower:
                family = "Phi"

            results.append({
                **m,
                "family": family,
                "is_bnb_4bit": is_bnb,
                "is_gguf": is_gguf,
                "is_instruct": is_instruct,
                "recommended_for_lora": is_bnb or (not is_gguf and is_instruct),
            })

        results.sort(key=lambda x: (x["recommended_for_lora"], not x["is_gguf"], x["downloads"]), reverse=True)
        res_subset = results[:limit]
        _set_cached(cache_key, res_subset)
        return res_subset

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
                req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "ML-Harness"})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
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
                _set_cached(cache_key, models)
                return models
            except Exception:
                pass

        # 2. Ollama local daemon
        if provider == "ollama":
            url = f"{preset.get('base_url', 'http://localhost:11434/v1').replace('/v1', '')}/api/tags"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ML-Harness"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
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
                req = urllib.request.Request(url, headers={"User-Agent": "ML-Harness"})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
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
                    headers={"Authorization": f"Bearer {key}", "User-Agent": "ML-Harness"}
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
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
                    headers={"Authorization": f"Bearer {key}", "User-Agent": "ML-Harness"}
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
        with urllib.request.urlopen(req, timeout=40) as resp:
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
        with urllib.request.urlopen(req, timeout=40) as resp:
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

