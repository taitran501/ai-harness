"""Unit tests for LLMEngine and LLM integration."""

import unittest
import unittest.mock
from harness.llm_engine import LLMEngine, LLMConfig


class TestLLMEngine(unittest.TestCase):
    def test_list_providers(self):
        with unittest.mock.patch.object(LLMEngine, "check_local_endpoint", return_value=False):
            providers = LLMEngine.list_providers()
        self.assertTrue(len(providers) >= 6)
        provider_ids = [p["id"] for p in providers]
        self.assertIn("gemini", provider_ids)
        self.assertIn("openai", provider_ids)
        self.assertIn("anthropic", provider_ids)
        self.assertIn("deepseek", provider_ids)
        self.assertIn("groq", provider_ids)
        self.assertIn("ollama", provider_ids)

    def test_format_prompt(self):
        template = "Comment: {comment} | Title: {title}"
        row = {"comment": "Great video!", "title": "Pride Parade 2026"}
        sys_prompt, user_prompt = LLMEngine.format_prompt(
            template=template,
            row_data=row,
            system_prompt="You are a classifier.",
            use_cot=True,
            valid_classes=["yes", "no"],
        )
        self.assertIn("Great video!", user_prompt)
        self.assertIn("Pride Parade 2026", user_prompt)
        self.assertIn("Think step by step", sys_prompt)
        self.assertIn("['yes', 'no']", sys_prompt)

    def test_extract_prediction_json(self):
        response_text = '```json\n{"prediction": "yes", "reasoning": "Explicit stereotype found"}\n```'
        result = LLMEngine.extract_prediction(response_text, valid_classes=["yes", "no"])
        self.assertEqual(result["parsed_label"], "yes")
        self.assertEqual(result["reasoning"], "Explicit stereotype found")
        self.assertTrue(result["is_valid"])

    def test_extract_prediction_fallback(self):
        response_text = "Based on the input context, the answer is clearly no."
        result = LLMEngine.extract_prediction(response_text, valid_classes=["yes", "no"])
        self.assertEqual(result["parsed_label"], "no")
        self.assertTrue(result["is_valid"])

    def test_generate_qlora_script(self):
        script = LLMEngine.generate_qlora_script(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            train_path="dataset/train.csv",
            lora_r=16,
        )
        self.assertIn("BitsAndBytesConfig", script)
        self.assertIn("LoraConfig", script)
        self.assertIn("SFTTrainer", script)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", script)

    def test_generate_unsloth_script(self):
        script = LLMEngine.generate_unsloth_script(
            model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            train_path="dataset/train.csv",
            lora_r=16,
            lora_alpha=32,
        )
        self.assertIn("FastLanguageModel", script)
        self.assertIn("unsloth", script)
        self.assertIn("SFTTrainer", script)
        self.assertIn("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", script)
        self.assertIn("is_bfloat16_supported", script)

    def test_fetch_provider_live_models_fallback(self):
        models = LLMEngine.fetch_provider_live_models("gemini")
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn("id", models[0])

    def test_fetch_unsloth_models_structure(self):
        models = LLMEngine.fetch_unsloth_models(limit=5)
        self.assertIsInstance(models, list)
        if models:
            self.assertIn("id", models[0])
            self.assertIn("family", models[0])


if __name__ == "__main__":
    unittest.main()

