import unittest
from tools.agent_bridge_cli import _parse_dynamic_kv_args, _coerce_value


class TestCliErgonomics(unittest.TestCase):
    def test_coerce_value(self):
        self.assertEqual(_coerce_value("true"), True)
        self.assertEqual(_coerce_value("False"), False)
        self.assertIsNone(_coerce_value("None"))
        self.assertIsNone(_coerce_value("null"))
        self.assertEqual(_coerce_value("123"), 123)
        self.assertEqual(_coerce_value("-45"), -45)
        self.assertEqual(_coerce_value("3.1415"), 3.1415)
        self.assertEqual(_coerce_value("exp001"), "exp001")
        self.assertEqual(_coerce_value('{"a": 1}'), {"a": 1})
        self.assertEqual(_coerce_value('[1, 2, 3]'), [1, 2, 3])

    def test_parse_dynamic_kv_args(self):
        args = [
            "--exp-id", "exp001_baseline",
            "--config-path", "configs/exp001.yaml",
            "--num-folds", "5",
            "--subsample", "0.85",
            "--enable-gpu",
            "--custom-opt=adamw",
        ]
        params = _parse_dynamic_kv_args(args)
        self.assertEqual(params["exp_id"], "exp001_baseline")
        self.assertEqual(params["config_path"], "configs/exp001.yaml")
        self.assertEqual(params["num_folds"], 5)
        self.assertEqual(params["subsample"], 0.85)
        self.assertEqual(params["enable_gpu"], True)
        self.assertEqual(params["custom_opt"], "adamw")


if __name__ == "__main__":
    unittest.main()
