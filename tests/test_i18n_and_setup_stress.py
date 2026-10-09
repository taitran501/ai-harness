"""Empirical stress test suite for ML-Harness Web Dashboard UI/UX:
1. Stage 1 Setup KPI Card 2 boundary state stress-testing.
2. Complete absence of (CV) contamination across dashboard/src/.
3. EN/VI i18n completeness, parity, parameter matching, and key usage.
4. Vietnamese technical terminology verification.
"""

import math
import os
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = PROJECT_ROOT / "dashboard" / "src"
I18N_FILE = DASHBOARD_SRC / "constants" / "i18n.ts"


def extract_i18n_dictionaries():
    """Extract EN and VI dictionaries from i18n.ts."""
    content = I18N_FILE.read_text(encoding="utf-8")
    en_match = re.search(r"en:\s*\{(.*?)\},\s*vi:", content, re.DOTALL)
    vi_match = re.search(r"vi:\s*\{(.*?)\}\s*,\s*\};", content, re.DOTALL)

    if not en_match or not vi_match:
        raise ValueError("Could not parse en or vi blocks in i18n.ts")

    def parse_block(block):
        entries = {}
        for line in block.splitlines():
            # match key: 'val' or key: "val"
            m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*['\"](.*)['\"],?\s*$", line)
            if m:
                entries[m.group(1)] = m.group(2)
        return entries

    return parse_block(en_match.group(1)), parse_block(vi_match.group(1))


def simulate_kpi_card_2(
    selected_phase: int,
    phase_index: int,
    baseline_locked: bool,
    experiments: list,
    baseline_score,
    cv_best: dict = None,
    lang: str = "en",
):
    """Exact emulation of the React logic in CompetitionJourneyWorkspace.tsx lines 353-365 & 600-658."""
    en_dict, vi_dict = extract_i18n_dictionaries()
    translations = en_dict if lang == "en" else vi_dict

    def t(key, fallback=""):
        return translations.get(key, fallback)

    # Line 353-358:
    is_valid_baseline_score = (
        isinstance(baseline_score, (int, float))
        and not math.isnan(baseline_score)
    )
    is_baseline_established = bool(
        baseline_locked
        and is_valid_baseline_score
        and len(experiments) > 0
    )

    # Line 359:
    is_awaiting_baseline = (selected_phase == 1 or phase_index < 3) and not is_baseline_established

    # Line 361-364:
    raw_cv_best = cv_best.get("score") if cv_best else None
    if isinstance(raw_cv_best, (int, float)) and not math.isnan(raw_cv_best):
        cv_best_score = raw_cv_best
    elif is_valid_baseline_score:
        cv_best_score = baseline_score
    else:
        cv_best_score = 0.0

    raw_delta = cv_best.get("delta") if cv_best else None
    if isinstance(raw_delta, (int, float)) and not math.isnan(raw_delta):
        cv_delta = raw_delta
    else:
        base_val = baseline_score if is_valid_baseline_score else 0.0
        cv_delta = cv_best_score - base_val

    # Rendered outputs
    card = {}
    if is_awaiting_baseline:
        card["title"] = t("best_score_label", "Best Score")
        card["score_display"] = t("status_not_started", "Not Started")
        card["badge"] = t("status_awaiting_phase3", "Awaiting Phase 3")
        card["subtitle"] = t("status_awaiting_phase3_baseline", "Awaiting Phase 3 Baseline")
        card["footer_anchor"] = f"{t('anchor_label', 'Anchor:')} —"
        card["footer_status"] = t("status_pending_phase3", "Pending Phase 3")
    else:
        card["title"] = t("kpi_best_validation_title", "Best Validation Score")
        card["score_display"] = f"{cv_best_score:.4f}"
        card["badge"] = f"+{cv_delta:.4f}" if cv_delta >= 0 else f"{cv_delta:.4f}"
        model_id = cv_best.get("id") if cv_best else "baseline"
        card["subtitle"] = f"Model: {model_id}"
        anchor_val = f"{baseline_score:.4f}" if is_valid_baseline_score else "0.0000"
        card["footer_anchor"] = f"{t('anchor_label', 'Anchor:')} {anchor_val}"
        card["footer_status"] = t("status_verified", "Verified")

    card["is_baseline_established"] = is_baseline_established
    card["is_awaiting_baseline"] = is_awaiting_baseline
    return card


class TestDashboardSetupStateStress(unittest.TestCase):
    """Stress tests for R2: Stage 1 Setup KPI Card 2 boundary states."""

    def test_stage1_unlocked_baseline_never_shows_zero_or_verified(self):
        """In Stage 1 (Setup), when baseline is NOT locked, 0.0000 and Verified must never appear."""
        experiments_scenarios = [
            [],
            [{"id": "exp001", "score": 0.85}],
            [{"id": "exp001"}, {"id": "exp002"}],
        ]
        score_scenarios = [None, float("nan"), 0.0, 0.75, 1.25]
        cv_best_scenarios = [None, {}, {"score": 0.9, "delta": 0.1, "id": "m1"}]

        for exps in experiments_scenarios:
            for score in score_scenarios:
                for cv_best in cv_best_scenarios:
                    for lang in ["en", "vi"]:
                        card = simulate_kpi_card_2(
                            selected_phase=1,
                            phase_index=1,
                            baseline_locked=False,
                            experiments=exps,
                            baseline_score=score,
                            cv_best=cv_best,
                            lang=lang,
                        )
                        self.assertTrue(
                            card["is_awaiting_baseline"],
                            f"Must be awaiting baseline when baseline_locked=False: {card}",
                        )
                        self.assertFalse(
                            card["is_baseline_established"],
                            "Baseline cannot be established when baseline_locked=False",
                        )
                        # Ensure 0.0000 is not displayed in score
                        self.assertNotEqual(card["score_display"], "0.0000")
                        self.assertNotIn("0.0000", card["score_display"])
                        self.assertNotIn("0.0000", card["footer_anchor"])

                        # Ensure Verified is not displayed in footer
                        self.assertNotEqual(card["footer_status"], "Verified")
                        self.assertNotEqual(card["footer_status"], "Đã xác thực")

                        if lang == "en":
                            self.assertEqual(card["score_display"], "Not Started")
                            self.assertEqual(card["badge"], "Awaiting Phase 3")
                            self.assertEqual(card["subtitle"], "Awaiting Phase 3 Baseline")
                            self.assertEqual(card["footer_anchor"], "Anchor: —")
                            self.assertEqual(card["footer_status"], "Pending Phase 3")
                        else:
                            self.assertEqual(card["score_display"], "Chưa bắt đầu")
                            self.assertEqual(card["badge"], "Chờ Giai đoạn 3")
                            self.assertEqual(card["subtitle"], "Chờ Baseline Giai đoạn 3")
                            self.assertEqual(card["footer_anchor"], "Điểm tựa: —")
                            self.assertEqual(card["footer_status"], "Chờ Giai đoạn 3")

    def test_stage1_locked_baseline_but_empty_experiments(self):
        """If baseline_locked is True but experiments list is empty, baseline is NOT established."""
        for lang in ["en", "vi"]:
            card = simulate_kpi_card_2(
                selected_phase=1,
                phase_index=1,
                baseline_locked=True,
                experiments=[],
                baseline_score=0.82,
                cv_best={"score": 0.82},
                lang=lang,
            )
            self.assertFalse(card["is_baseline_established"])
            self.assertTrue(card["is_awaiting_baseline"])
            self.assertNotEqual(card["score_display"], "0.0000")
            self.assertNotEqual(card["footer_status"], "Verified")
            self.assertNotEqual(card["footer_status"], "Đã xác thực")

    def test_stage1_locked_baseline_with_invalid_score(self):
        """If baseline_locked is True but baseline score is NaN or None, baseline is NOT established."""
        for invalid_score in [None, float("nan")]:
            for lang in ["en", "vi"]:
                card = simulate_kpi_card_2(
                    selected_phase=1,
                    phase_index=1,
                    baseline_locked=True,
                    experiments=[{"id": "exp000"}],
                    baseline_score=invalid_score,
                    cv_best=None,
                    lang=lang,
                )
                self.assertFalse(card["is_baseline_established"])
                self.assertTrue(card["is_awaiting_baseline"])
                self.assertNotEqual(card["score_display"], "0.0000")
                self.assertNotEqual(card["footer_status"], "Verified")

    def test_stage1_fully_established_baseline_shows_verified(self):
        """When baseline is fully established, Card 2 displays actual score and Verified status."""
        for lang in ["en", "vi"]:
            card = simulate_kpi_card_2(
                selected_phase=1,
                phase_index=3,
                baseline_locked=True,
                experiments=[{"id": "exp000", "score": 0.8234}],
                baseline_score=0.8234,
                cv_best={"score": 0.8500, "delta": 0.0266, "id": "exp001"},
                lang=lang,
            )
            self.assertTrue(card["is_baseline_established"])
            self.assertFalse(card["is_awaiting_baseline"])
            self.assertEqual(card["score_display"], "0.8500")
            self.assertEqual(card["badge"], "+0.0266")
            if lang == "en":
                self.assertEqual(card["footer_status"], "Verified")
                self.assertIn("Anchor: 0.8234", card["footer_anchor"])
            else:
                self.assertEqual(card["footer_status"], "Đã xác thực")
                self.assertIn("Điểm tựa: 0.8234", card["footer_anchor"])


class TestZeroCVContamination(unittest.TestCase):
    """Stress tests for R2: Zero occurrences of (CV) anywhere in dashboard/src/."""

    def test_zero_cv_in_dashboard_src(self):
        """Rigorously scan all files in dashboard/src/ for '(CV)' or '(cv)'."""
        matches = []
        for root, _, files in os.walk(DASHBOARD_SRC):
            for file in files:
                path = Path(root) / file
                # Skip binary files if any
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue

                for line_no, line in enumerate(text.splitlines(), start=1):
                    if "(CV)" in line or "(cv)" in line:
                        matches.append(f"{path.relative_to(PROJECT_ROOT)}:{line_no} -> {line.strip()}")

        self.assertEqual(
            len(matches),
            0,
            f"Found (CV) contamination in dashboard/src/:\n" + "\n".join(matches),
        )


class TestI18nCompletenessAndQuality(unittest.TestCase):
    """Stress tests for R3: i18n key parity, parameters, and Vietnamese technical accuracy."""

    def setUp(self):
        self.en_dict, self.vi_dict = extract_i18n_dictionaries()

    def test_key_parity_between_en_and_vi(self):
        """Every key defined in EN must exist in VI and vice versa."""
        en_keys = set(self.en_dict.keys())
        vi_keys = set(self.vi_dict.keys())

        missing_in_vi = en_keys - vi_keys
        missing_in_en = vi_keys - en_keys

        self.assertEqual(missing_in_vi, set(), f"Keys in EN missing in VI: {missing_in_vi}")
        self.assertEqual(missing_in_en, set(), f"Keys in VI missing in EN: {missing_in_en}")
        self.assertGreaterEqual(len(en_keys), 200, "Expected at least 200 i18n keys")

    def test_parameter_interpolation_parity(self):
        """Any {param} in EN must also exist in VI with exact same name."""
        param_regex = re.compile(r"\{([a-zA-Z0-9_]+)\}")
        mismatches = []

        for key, en_text in self.en_dict.items():
            en_params = set(param_regex.findall(en_text))
            vi_text = self.vi_dict.get(key, "")
            vi_params = set(param_regex.findall(vi_text))

            if en_params != vi_params:
                mismatches.append(f"{key}: EN params {en_params} != VI params {vi_params}")

        self.assertEqual(
            mismatches,
            [],
            f"Parameter interpolation mismatches:\n" + "\n".join(mismatches),
        )

    def test_all_used_translation_keys_are_defined(self):
        """Any t('key') call in tsx/ts files must be defined in i18n dictionaries."""
        call_regex = re.compile(r"\bt\(\s*['\"]([a-zA-Z0-9_]+)['\"]")
        missing_keys = {}

        for root, _, files in os.walk(DASHBOARD_SRC):
            for file in files:
                if not (file.endswith(".tsx") or file.endswith(".ts")):
                    continue
                path = Path(root) / file
                text = path.read_text(encoding="utf-8")
                matches = call_regex.findall(text)
                for k in matches:
                    if k not in self.en_dict or k not in self.vi_dict:
                        missing_keys.setdefault(k, []).append(str(path.relative_to(PROJECT_ROOT)))

        self.assertEqual(
            missing_keys,
            {},
            f"Keys referenced in components but missing in i18n dictionary: {missing_keys}",
        )

    def test_vietnamese_technical_terminology_accuracy(self):
        """Verify Vietnamese translations use accurate and natural ML technical terms."""
        terminology_checks = {
            "best_score_label": "Điểm cao nhất",
            "kpi_best_validation_title": "Điểm đánh giá tốt nhất",
            "status_not_started": "Chưa bắt đầu",
            "status_awaiting_phase3": "Chờ Giai đoạn 3",
            "status_awaiting_phase3_baseline": "Chờ Baseline Giai đoạn 3",
            "status_verified": "Đã xác thực",
            "anchor_label": "Điểm tựa:",
            "stage1_metric_card": "Chỉ số đánh giá chính",
            "stage1_constraints_card": "Ràng buộc đã khai báo",
            "stage1_protocol_card": "Giao thức đánh giá",
            "tab_validation_protocol": "3. Phân chia Tập xác thực",
            "inspector_atomic_hypothesis": "Giả thuyết khoa học đơn nguyên",
            "inspector_grounding_pass": "Đã xác thực căn cứ thực nghiệm (ĐẠT)",
            "inspector_assumption_register": "Sổ theo dõi giả định (Mức độ không chắc chắn)",
            "copilot_title": "Bộ gợi ý trợ lý AI tự động",
            "stage4_title": "4. Luồng thử nghiệm mô hình",
            "stage5_strategy_ensemble": "Kết hợp Ensemble đa dạng",
        }

        for key, expected_text in terminology_checks.items():
            self.assertIn(key, self.vi_dict, f"Missing key {key} in VI dictionary")
            self.assertEqual(
                self.vi_dict[key],
                expected_text,
                f"Translation for {key} is unexpected: got '{self.vi_dict[key]}', expected '{expected_text}'",
            )

    def test_no_empty_or_whitespace_only_translations(self):
        """Translations must not be empty or whitespace-only."""
        for lang, d in [("en", self.en_dict), ("vi", self.vi_dict)]:
            for k, v in d.items():
                self.assertTrue(v.strip(), f"Empty translation for key '{k}' in {lang}")

    def test_defined_keys_usage(self):
        """Report defined keys in i18n that are not utilized in components."""
        call_regex = re.compile(r"\bt\(\s*['\"]([a-zA-Z0-9_]+)['\"]")
        used_keys = set()
        for root, _, files in os.walk(DASHBOARD_SRC):
            for file in files:
                if not (file.endswith(".tsx") or file.endswith(".ts")):
                    continue
                path = Path(root) / file
                text = path.read_text(encoding="utf-8")
                used_keys.update(call_regex.findall(text))

        unused_keys = set(self.en_dict.keys()) - used_keys
        # We check specific keys defined for stage4 that should be wired:
        wired_stage4_keys = {"stage4_val_score", "stage4_vs_baseline", "stage4_fold_std", "stage4_promote_btn"}
        unwired_in_stage4 = wired_stage4_keys.intersection(unused_keys)
        # Record findings
        if unwired_in_stage4:
            print(f"\n[Finding] Unwired Stage 4 keys in components: {unwired_in_stage4}")
        if unused_keys:
            print(f"\n[Finding] Total unused keys in i18n ({len(unused_keys)}):")
            for k in sorted(unused_keys):
                print(f"  - {k}")


if __name__ == "__main__":
    unittest.main()
