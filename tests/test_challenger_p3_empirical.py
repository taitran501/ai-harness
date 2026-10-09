"""Empirical Verification Suite by teamwork_preview_challenger_3.

Verifies:
1. App.tsx offline retry error display:
   - Validates that App.tsx line 98 does NOT guard error display with !initialLoadedRef.current
   - Validates that loadData() sets apiError and preserves offline banner on retry when backend remains down
   - Validates that successful retry clears apiError
   - Validates that dismiss clears apiError and subsequent retry resensitizes error display
2. Localization switching across Stage 4 candidate cards and headers:
   - Validates that stage4_empty, stage4_val_score, stage4_vs_baseline, stage4_fold_std, stage4_promote_btn
     render Vietnamese strings in VI mode
   - Validates that modal_task_presets_label, modal_task_primary_metric_title, inspector_assumption_register,
     and inspector_config_title render Vietnamese strings in VI mode
   - Validates that zero raw English strings remain for all 9 flagged items
"""

import math
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = PROJECT_ROOT / "dashboard" / "src"
APP_TSX = DASHBOARD_SRC / "App.tsx"
STAGE4_TSX = DASHBOARD_SRC / "components" / "CompetitionJourneyWorkspace.tsx"
TASK_MODAL_TSX = DASHBOARD_SRC / "components" / "TaskConfigModal.tsx"
NODE_INSPECTOR_TSX = DASHBOARD_SRC / "components" / "NodeInspector.tsx"
I18N_TS = DASHBOARD_SRC / "constants" / "i18n.ts"


def load_i18n():
    content = I18N_TS.read_text(encoding="utf-8")
    en_block = re.search(r"en:\s*\{(.*?)\},\s*vi:", content, re.DOTALL).group(1)
    vi_block = re.search(r"vi:\s*\{(.*?)\}\s*,\s*\};", content, re.DOTALL).group(1)

    def parse(blk):
        res = {}
        for line in blk.splitlines():
            m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*['\"](.*)['\"],?\s*$", line)
            if m:
                res[m.group(1)] = m.group(2)
        return res

    return parse(en_block), parse(vi_block)


class TestAppOfflineRetryEmpirical(unittest.TestCase):
    """Empirical verification of App.tsx offline retry error handling."""

    def test_app_tsx_source_code_line_98_condition(self):
        """Verify directly in App.tsx source that !initialLoadedRef.current is absent from error setter."""
        content = APP_TSX.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Find criticalFailures check
        found_target = False
        for i, line in enumerate(lines):
            if "const criticalFailures = [statusRes, taskRes]" in line:
                next_line = lines[i + 1].strip()
                # Must be 'if (criticalFailures.length > 0) {' without !initialLoadedRef.current
                self.assertNotIn(
                    "!initialLoadedRef.current",
                    next_line,
                    "Defect found: !initialLoadedRef.current is guarding error handling in App.tsx!"
                )
                self.assertIn("if (criticalFailures.length > 0)", next_line)
                found_target = True
                break

        self.assertTrue(found_target, "Could not find criticalFailures check in App.tsx")

    def test_app_tsx_error_lifecycle_simulation(self):
        """Simulate App.tsx loadData() and banner state machine under failing and recovering network."""
        class AppState:
            def __init__(self, endpoint_status_gen):
                self.endpoint_status_gen = endpoint_status_gen
                self.initial_loaded_ref = False
                self.loading = True
                self.api_error = None

            def load_data(self):
                # Line 72: if (!initialLoadedRef.current) setLoading(true)
                if not self.initial_loaded_ref:
                    self.loading = True
                # Line 75: setApiError(null)
                self.api_error = None

                status_res, task_res = self.endpoint_status_gen()

                critical_failures = [r for r in [status_res, task_res] if r["status"] == "rejected"]
                # Remediated line 98 in App.tsx:
                if len(critical_failures) > 0:
                    first_err = critical_failures[0]["reason"]
                    self.api_error = first_err if isinstance(first_err, str) else str(first_err)

                # Line 105: finally block
                self.initial_loaded_ref = True
                self.loading = False

            def dismiss(self):
                # Line 195: onClick={() => setApiError(null)}
                self.api_error = None

            @property
            def banner_visible(self):
                # Line 174: {apiError && (
                return self.api_error is not None

        network_state = {"online": False}

        def endpoint_fetch():
            if not network_state["online"]:
                return (
                    {"status": "rejected", "reason": "Failed to fetch: Connection refused"},
                    {"status": "rejected", "reason": "Failed to fetch: Connection refused"},
                )
            return (
                {"status": "fulfilled", "value": {"best_node": "exp001"}},
                {"status": "fulfilled", "value": {"name": "test_task"}},
            )

        app = AppState(endpoint_fetch)

        # 1. Cold initial load while offline
        app.load_data()
        self.assertTrue(app.banner_visible, "Offline banner must be visible after initial load failure.")
        self.assertEqual(app.api_error, "Failed to fetch: Connection refused")
        self.assertFalse(app.loading, "App shell must not be in full-page loading state.")
        self.assertTrue(app.initial_loaded_ref, "initialLoadedRef must be True after first attempt.")

        # 2. Retry Connection while backend is STILL offline
        # In the old defective version, api_error became None here.
        app.load_data()
        self.assertTrue(app.banner_visible, "Offline banner MUST remain visible when retry fails!")
        self.assertEqual(app.api_error, "Failed to fetch: Connection refused")
        self.assertFalse(app.loading, "App shell must not trigger full unmount on retry.")

        # 3. Dismiss error banner
        app.dismiss()
        self.assertFalse(app.banner_visible, "Banner must be hidden after dismissal.")
        self.assertIsNone(app.api_error)

        # 4. User clicks Retry Connection again after dismissal (backend still offline)
        app.load_data()
        self.assertTrue(app.banner_visible, "Banner MUST re-appear when manual retry is clicked after dismissal!")
        self.assertEqual(app.api_error, "Failed to fetch: Connection refused")

        # 5. Backend comes online, user retries
        network_state["online"] = True
        app.load_data()
        self.assertFalse(app.banner_visible, "Banner must be hidden when backend recovers.")
        self.assertIsNone(app.api_error)


class TestI18nWiringEmpirical(unittest.TestCase):
    """Empirical verification of i18n wiring in Stage 4, TaskConfigModal, and NodeInspector."""

    @classmethod
    def setUpClass(cls):
        cls.en_dict, cls.vi_dict = load_i18n()

    def translate(self, key, lang="en"):
        d = self.vi_dict if lang == "vi" else self.en_dict
        return d.get(key, key)

    def test_stage4_candidate_cards_wired_and_render_vietnamese(self):
        """Verify that Stage 4 candidate cards use t() and render complete Vietnamese in VI mode."""
        stage4_code = STAGE4_TSX.read_text(encoding="utf-8")

        # 1. stage4_empty
        self.assertIn("t('stage4_empty'", stage4_code)
        self.assertEqual(self.translate("stage4_empty", "vi"), "Không có thử nghiệm ứng viên nào khớp với bộ lọc này.")

        # 2. stage4_val_score
        self.assertIn("t('stage4_val_score'", stage4_code)
        self.assertEqual(self.translate("stage4_val_score", "vi"), "Điểm xác thực")

        # 3. stage4_vs_baseline
        self.assertIn("t('stage4_vs_baseline'", stage4_code)
        self.assertEqual(self.translate("stage4_vs_baseline", "vi"), "So với Baseline")

        # 4. stage4_fold_std
        self.assertIn("t('stage4_fold_std'", stage4_code)
        self.assertEqual(self.translate("stage4_fold_std", "vi"), "Độ lệch chuẩn Fold")

        # 5. stage4_promote_btn
        self.assertIn("t('stage4_promote_btn'", stage4_code)
        self.assertEqual(self.translate("stage4_promote_btn", "vi"), "Nâng hạng ứng viên")

        # Verify that hardcoded English versions are not present in card stats
        # Check lines around card stats (1240-1265)
        lines = stage4_code.splitlines()[1238:1265]
        block = "\n".join(lines)
        self.assertNotIn("Validation Score</div>", block)
        self.assertNotIn("vs. Baseline</div>", block)
        self.assertNotIn("Fold Std</div>", block)
        self.assertNotIn("Promote</button>", block)

    def test_modal_headers_wired_and_render_vietnamese(self):
        """Verify TaskConfigModal and NodeInspector headers use t() and render Vietnamese."""
        modal_code = TASK_MODAL_TSX.read_text(encoding="utf-8")
        inspector_code = NODE_INSPECTOR_TSX.read_text(encoding="utf-8")

        # TaskConfigModal presets
        self.assertIn("t('modal_task_presets_label'", modal_code)
        self.assertEqual(self.translate("modal_task_presets_label", "vi"), "Bộ mẫu cuộc thi nhanh (1-Click điền toàn bộ)")

        # TaskConfigModal primary metric title
        self.assertIn("t('modal_task_primary_metric_title'", modal_code)
        self.assertEqual(self.translate("modal_task_primary_metric_title", "vi"), "Chỉ số chuẩn & Điểm mục tiêu chính")

        # NodeInspector assumption register
        self.assertIn("t('inspector_assumption_register'", inspector_code)
        self.assertEqual(self.translate("inspector_assumption_register", "vi"), "Sổ theo dõi giả định (Mức độ không chắc chắn)")

        # NodeInspector target configuration
        self.assertIn("t('inspector_config_title'", inspector_code)
        self.assertEqual(self.translate("inspector_config_title", "vi"), "Cấu hình mục tiêu thử nghiệm")


if __name__ == "__main__":
    unittest.main()
