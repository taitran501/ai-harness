import shutil
import tempfile
import unittest
from pathlib import Path

from harness.tree_journal import Node, TreeJournal
from tools.dashboard_server import DEFAULT_STRATEGY_CARDS


class TestLLMExperimentSpecs(unittest.TestCase):
    def test_node_dataclass_llm_specs(self):
        node = Node(
            id="exp010_model_unsloth_qwen_7b",
            parent_id="exp001_baseline_tfidf",
            branch="model",
            approach_type="llm",
            model_family="llm_peft",
            param_count=7_615_000_000,
            param_size="7B",
            llm_spec={
                "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                "param_size": "7B",
                "quantization": "4-bit NormalFloat (NF4)",
                "paradigm": "unsloth_qlora",
                "lora_r": 16,
                "lora_alpha": 32,
            },
        )
        self.assertEqual(node.approach_type, "llm")
        self.assertEqual(node.param_size, "7B")
        self.assertIsNotNone(node.llm_spec)
        self.assertEqual(node.llm_spec["quantization"], "4-bit NormalFloat (NF4)")

    def test_tree_journal_get_graph_data_includes_llm_specs(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            journal_file = Path(tmp_dir) / "journal.json"
            journal = TreeJournal(storage_path=str(journal_file))
            node = Node(
                id="exp020_model_deepseek_r1_14b",
                parent_id=None,
                branch="model",
                approach_type="llm",
                model_family="llm_reasoning",
                param_count=14_000_000_000,
                param_size="14B",
                llm_spec={
                    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    "param_size": "14B",
                    "quantization": "4-bit AWQ",
                    "paradigm": "zero_shot_cot",
                    "use_cot": True,
                },
            )
            journal.append(node)

            graph = journal.get_graph_data()
            self.assertEqual(len(graph["nodes"]), 1)
            g_node = graph["nodes"][0]
            self.assertEqual(g_node["id"], "exp020_model_deepseek_r1_14b")
            self.assertEqual(g_node["approach_type"], "llm")
            self.assertEqual(g_node["param_size"], "14B")
            self.assertIsNotNone(g_node["llm_spec"])
            self.assertEqual(g_node["llm_spec"]["model_name"], "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_llm_strategy_cards_present(self):
        llm_cards = [c for c in DEFAULT_STRATEGY_CARDS if c.get("approach_type") == "llm"]
        self.assertGreaterEqual(len(llm_cards), 2)
        qwen = next((c for c in llm_cards if "qwen" in c["id"].lower()), None)
        self.assertIsNotNone(qwen)
        self.assertEqual(qwen["param_size"], "7B")

        r1 = next((c for c in llm_cards if "deepseek" in c["id"].lower()), None)
        self.assertIsNotNone(r1)
        self.assertEqual(r1["param_size"], "14B")

if __name__ == "__main__":
    unittest.main()
