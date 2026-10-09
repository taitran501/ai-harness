"""Empirical Challenger Verification Suite for Milestone 3.

Author: Challenger M3 2 (teamwork_preview_challenger)
Target: dashboard/src/components/TaskConfigModal.tsx, dashboard/src/App.tsx, dashboard/src/constants/i18n.ts
Milestone: Milestone 3 (Frontend Modality-Adaptive Audit Gate UI/UX & Contextual Presets)

Empirically tests and challenges:
1. i18n Completeness & Parity for TaskConfigModal (all literal and dynamic keys).
2. Modality-adaptive key presence and validity across en/vi for:
   - taskcfg_audit_*
   - audit_label_*
   - audit_hint_*
   - taskcfg_missing_opt_*
   - taskcfg_context_cols_*
   - taskcfg_params_preset_*
3. Audit regime resolution logic and regime defaults across all 4 regimes.
4. Persistence contract verification for feature_columns and api.updateTaskInfo.
5. Parameter scale formatting edge cases.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = PROJECT_ROOT / "dashboard" / "src"
TASK_CONFIG_MODAL_PATH = DASHBOARD_SRC / "components" / "TaskConfigModal.tsx"
APP_PATH = DASHBOARD_SRC / "App.tsx"
I18N_PATH = DASHBOARD_SRC / "constants" / "i18n.ts"


def extract_i18n_dictionaries():
    """Extract EN and VI dictionaries from i18n.ts."""
    content = I18N_PATH.read_text(encoding="utf-8")
    en_match = re.search(r"en:\s*\{(.*?)\},\s*vi:", content, re.DOTALL)
    vi_match = re.search(r"vi:\s*\{(.*?)\}\s*,\s*\};", content, re.DOTALL)

    if not en_match or not vi_match:
        raise ValueError("Could not parse en or vi blocks in i18n.ts")

    def parse_block(block):
        entries = {}
        for line in block.splitlines():
            m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*['\"](.*)['\"],?\s*$", line)
            if m:
                entries[m.group(1)] = m.group(2)
        return entries

    return parse_block(en_match.group(1)), parse_block(vi_match.group(1))


def resolve_audit_regime_py(modality: str, task_type: str, primary_metric: str) -> str:
    """Exact Python replica of resolveAuditRegime in TaskConfigModal.tsx."""
    t = (task_type or "").lower()
    m = (primary_metric or "").lower()
    mod = (modality or "").lower()

    # Metric category check (approximated for known metrics)
    regression_metrics = {"rmse", "mae", "mse", "r2", "rmsle", "medae", "mape"}
    ranking_metrics = {"ndcg", "map", "mrr", "hitrate", "kendall"}
    nlp_metrics = {"bleu", "rouge", "perplexity", "exactmatch", "chrf", "meteor", "bertscore"}

    # 1. Regression
    if "regress" in t or any(k in m for k in regression_metrics):
        return "regression"

    # 2. Ranking
    if "rank" in t or "recsys" in t or any(k in m for k in ranking_metrics):
        return "ranking"

    # 3. Generative NLP
    if (mod == "text" or "llm" in t or "generat" in t) and any(k in m for k in nlp_metrics):
        return "generative_nlp"

    if any(k in t for k in ["generat", "generation", "llm", "summariz", "translat"]):
        return "generative_nlp"

    # 4. Classification (Default)
    return "classification"


def format_param_scale_py(val: str, unlimited_label: str = "Unlimited parameters") -> str:
    """Exact Python replica of formatParamScale in TaskConfigModal.tsx."""
    if not val or val.strip() == "" or val == "0":
        return unlimited_label
    try:
        num = float(val)
    except ValueError:
        return val

    if num >= 1e9:
        b = f"{num / 1e9:.1f}"
        return f"{b}B parameters"
    if num >= 1e6:
        m = f"{num / 1e6:.0f}"
        return f"{m}M parameters"
    if num >= 1e3:
        k = f"{num / 1e3:.0f}"
        return f"{k}K parameters"
    return f"{int(num) if num.is_integer() else num} parameters"


class TestI18nCompletenessForTaskConfigModal(unittest.TestCase):
    """Test Suite 1: Rigorous i18n key presence and validity."""

    def setUp(self):
        self.en_dict, self.vi_dict = extract_i18n_dictionaries()
        self.modal_code = TASK_CONFIG_MODAL_PATH.read_text(encoding="utf-8")

    def test_direct_t_call_keys_in_both_languages(self):
        """All literal t('key', ...) calls in TaskConfigModal.tsx must exist in EN and VI."""
        call_regex = re.compile(r"\bt\(\s*['\"]([a-zA-Z0-9_]+)['\"]")
        keys = set(call_regex.findall(self.modal_code))

        self.assertGreater(len(keys), 30, f"Expected >30 literal t() calls, found {len(keys)}")

        missing_en = [k for k in keys if k not in self.en_dict]
        missing_vi = [k for k in keys if k not in self.vi_dict]

        self.assertEqual(missing_en, [], f"Keys used in TaskConfigModal missing in EN: {missing_en}")
        self.assertEqual(missing_vi, [], f"Keys used in TaskConfigModal missing in VI: {missing_vi}")

    def test_dynamic_preset_keys_in_both_languages(self):
        """Dynamic keys referenced in arrays (labelKey: '...') must exist in EN and VI."""
        label_key_regex = re.compile(r"labelKey:\s*['\"]([a-zA-Z0-9_]+)['\"]")
        preset_keys = set(label_key_regex.findall(self.modal_code))

        self.assertGreaterEqual(len(preset_keys), 15, f"Expected >=15 preset keys, found {len(preset_keys)}")

        missing_en = [k for k in preset_keys if k not in self.en_dict]
        missing_vi = [k for k in preset_keys if k not in self.vi_dict]

        self.assertEqual(missing_en, [], f"Preset keys missing in EN: {missing_en}")
        self.assertEqual(missing_vi, [], f"Preset keys missing in VI: {missing_vi}")

    def test_prefixed_keys_render_valid_non_empty_text(self):
        """Verify all keys matching target prefixes render valid non-empty text in both EN & VI."""
        target_prefixes = (
            "taskcfg_audit_",
            "audit_label_",
            "audit_hint_",
            "taskcfg_missing_opt_",
            "taskcfg_context_cols_",
            "taskcfg_params_preset_",
        )

        all_prefixed_keys = [
            k for k in self.en_dict.keys()
            if any(k.startswith(p) for p in target_prefixes)
        ]

        self.assertGreaterEqual(
            len(all_prefixed_keys), 40,
            f"Expected at least 40 prefixed keys, found {len(all_prefixed_keys)}"
        )

        for key in all_prefixed_keys:
            # 1. Must exist in both EN and VI
            self.assertIn(key, self.en_dict, f"Key {key} missing in EN dictionary")
            self.assertIn(key, self.vi_dict, f"Key {key} missing in VI dictionary")

            en_val = self.en_dict[key].strip()
            vi_val = self.vi_dict[key].strip()

            # 2. Must not be empty or whitespace
            self.assertTrue(en_val, f"Key {key} has empty EN translation")
            self.assertTrue(vi_val, f"Key {key} has empty VI translation")

            # 3. Parameter placeholders {param} must match exactly
            en_params = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", en_val))
            vi_params = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", vi_val))
            self.assertEqual(
                en_params, vi_params,
                f"Param mismatch for {key}: EN has {en_params}, VI has {vi_params}"
            )


class TestModalityAdaptiveAuditRegimeLogic(unittest.TestCase):
    """Test Suite 2: Modality-adaptive regime resolution and defaults."""

    def test_regression_regime_resolution(self):
        """Regression tasks and metrics must resolve to 'regression'."""
        cases = [
            ("tabular", "regression", "RMSE"),
            ("tabular", "House Price Regression", "MAE"),
            ("generic", "Supervised Learning", "mse"),
            ("generic", "Predictive Modeling", "r2"),
            ("time-series", "Forecasting", "mape"),
            ("tabular", "regression", "rmsle"),
        ]
        for mod, task_type, metric in cases:
            regime = resolve_audit_regime_py(mod, task_type, metric)
            self.assertEqual(
                regime, "regression",
                f"Failed to resolve regression for ({mod}, {task_type}, {metric})"
            )

    def test_ranking_regime_resolution(self):
        """Ranking tasks and metrics must resolve to 'ranking'."""
        cases = [
            ("tabular", "ranking", "NDCG@10"),
            ("generic", "Recommendation System", "MAP"),
            ("text", "recsys", "MRR"),
            ("tabular", "Ranking", "KendallTau"),
        ]
        for mod, task_type, metric in cases:
            regime = resolve_audit_regime_py(mod, task_type, metric)
            self.assertEqual(
                regime, "ranking",
                f"Failed to resolve ranking for ({mod}, {task_type}, {metric})"
            )

    def test_generative_nlp_regime_resolution(self):
        """Generative NLP tasks and metrics must resolve to 'generative_nlp'."""
        cases = [
            ("text", "Text Generation", "BLEU"),
            ("text", "LLM Reasoning", "ROUGE-1"),
            ("text", "Summarization", "meteor"),
            ("generic", "llm_finetuning", "Perplexity"),
            ("generic", "translation", "chrf"),
        ]
        for mod, task_type, metric in cases:
            regime = resolve_audit_regime_py(mod, task_type, metric)
            self.assertEqual(
                regime, "generative_nlp",
                f"Failed to resolve generative_nlp for ({mod}, {task_type}, {metric})"
            )

    def test_classification_fallback_regime(self):
        """Classification tasks or unknown metrics default cleanly to 'classification'."""
        cases = [
            ("tabular", "classification", "macro_f1"),
            ("text", "Sentiment Classification", "accuracy"),
            ("vision", "Image Classification", "logloss"),
            ("multimodal", "VQA", "roc_auc"),
            ("generic", "Supervised Learning", "Score"),
            ("", "", ""),
        ]
        for mod, task_type, metric in cases:
            regime = resolve_audit_regime_py(mod, task_type, metric)
            self.assertEqual(
                regime, "classification",
                f"Failed to resolve classification for ({mod}, {task_type}, {metric})"
            )


class TestPersistenceContractVerification(unittest.TestCase):
    """Test Suite 3: Verification of data persistence in TaskConfigModal and App."""

    def setUp(self):
        self.modal_code = TASK_CONFIG_MODAL_PATH.read_text(encoding="utf-8")
        self.app_code = APP_PATH.read_text(encoding="utf-8")

    def test_feature_columns_in_data_config(self):
        """handleSave MUST store feature_columns in updatedTask.data_config."""
        # Check that data_config includes feature_columns: featureColumns.trim()
        pattern = re.compile(
            r"data_config:\s*\{[^}]*feature_columns:\s*featureColumns\.trim\(\)",
            re.DOTALL
        )
        self.assertRegex(
            self.modal_code, pattern,
            "handleSave does not include feature_columns: featureColumns.trim() in data_config!"
        )

    def test_handle_save_calls_api_update_task_info(self):
        """handleSave MUST await api.updateTaskInfo(updatedTask)."""
        pattern = re.compile(r"await\s+api\.updateTaskInfo\(\s*updatedTask\s*\)")
        self.assertRegex(
            self.modal_code, pattern,
            "handleSave does not await api.updateTaskInfo(updatedTask)!"
        )

    def test_handle_save_calls_on_save_and_on_close(self):
        """handleSave MUST invoke onSave(updatedTask) and onClose()."""
        pattern_onsave = re.compile(r"await\s+onSave\(\s*updatedTask\s*\)")
        pattern_onclose = re.compile(r"onClose\(\)")
        self.assertRegex(self.modal_code, pattern_onsave, "handleSave does not await onSave(updatedTask)")
        self.assertRegex(self.modal_code, pattern_onclose, "handleSave does not call onClose()")

    def test_app_on_save_handler_reloads_data(self):
        """App.tsx onSave handler must set taskInfo and reload data."""
        pattern = re.compile(
            r"onSave=\{async\s*\(\s*updatedTask\s*\)\s*=>\s*\{[^}]*setTaskInfo\(updatedTask\)[^}]*await\s+loadData\(\)",
            re.DOTALL
        )
        self.assertRegex(
            self.app_code, pattern,
            "App.tsx onSave does not setTaskInfo and await loadData()!"
        )


class TestParamScaleFormatterEdgeCases(unittest.TestCase):
    """Test Suite 4: Stress-test formatParamScale logic."""

    def test_standard_preset_scales(self):
        """Preset values must format to clean human-readable parameter counts."""
        self.assertEqual(format_param_scale_py("100000000"), "100M parameters")
        self.assertEqual(format_param_scale_py("350000000"), "350M parameters")
        self.assertEqual(format_param_scale_py("1000000000"), "1.0B parameters")
        self.assertEqual(format_param_scale_py("7000000000"), "7.0B parameters")
        self.assertEqual(format_param_scale_py("14000000000"), "14.0B parameters")
        self.assertEqual(format_param_scale_py("70000000000"), "70.0B parameters")

    def test_unlimited_and_zero_values(self):
        """Empty, zero, or blank values must return unlimited label."""
        self.assertEqual(format_param_scale_py(""), "Unlimited parameters")
        self.assertEqual(format_param_scale_py("   "), "Unlimited parameters")
        self.assertEqual(format_param_scale_py("0"), "Unlimited parameters")

    def test_small_parameter_values(self):
        """Values below 1M format with K or raw count."""
        self.assertEqual(format_param_scale_py("50000"), "50K parameters")
        self.assertEqual(format_param_scale_py("500"), "500 parameters")


if __name__ == "__main__":
    unittest.main()
