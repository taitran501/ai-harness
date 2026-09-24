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
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-pro"],
            "env_key": "GEMINI_API_KEY",
            "default_model": "gemini-2.0-flash",
        },
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "o3-mini", "o1"],
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
            "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
            "env_key": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-5-haiku-20241022",
        },
        "groq": {
            "name": "Groq LPU",
            "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
            "env_key": "GROQ_API_KEY",
            "default_model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
        },
        "openrouter": {
            "name": "OpenRouter",
            "models": ["qwen/qwen-2.5-72b-instruct", "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-r1"],
            "env_key": "OPENROUTER_API_KEY",
            "default_model": "qwen/qwen-2.5-72b-instruct",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "models": ["llama3.2", "qwen2.5", "mistral", "deepseek-r1:7b"],
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
