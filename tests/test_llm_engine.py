"""Unit tests for LLMEngine and LLM integration."""

import unittest
from harness.llm_engine import LLMEngine, LLMConfig


class TestLLMEngine(unittest.TestCase):
    def test_list_providers(self):
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


if __name__ == "__main__":
    unittest.main()
