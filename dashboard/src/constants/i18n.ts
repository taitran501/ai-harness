import React, { createContext, useContext, useState } from 'react';

export type UILanguage = 'en' | 'vi';

export const UI_TRANSLATIONS: Record<UILanguage, Record<string, string>> = {
  en: {
    // App & Nav
    app_title: 'ML Experiment Harness',
    nav_journey: 'Competition Journey (Phases 1–8)',
    nav_experiments_hub: 'Experiments & Solution Tree',
    nav_execution: 'Execution & Cloud',
    nav_llm_studio: 'LLM Studio',
    nav_submissions: 'Submissions History',
    nav_agent_copilot: 'Agent & Prompts Hub',
    nav_experiments: 'Experiments',
    nav_tree: 'AIDE Tree',
    nav_compare: 'Compare',
    nav_cloud_runner: 'Cloud Runner',
    nav_priming: 'Priming Hub',
    new_experiment: 'New Experiment',
    task_settings: 'Task Settings',
    refresh: 'Refresh',
    language_label: 'Language',
    loading_telemetry: 'Loading workspace telemetry & experiment tree…',
    connection_error: 'ML-Harness Connection Error',
    retry_connection: 'Retry Connection',
    recorded_runs: '{count} recorded runs',
    view_leaderboard_title: 'Leaderboard & Experiment History',
    view_tree_title: 'AIDE Solution Tree DAG',
    tab_leaderboard: 'Leaderboard Table',
    tab_tree: 'Solution Tree',
    view_runner_title: '4-Step Competition Execution Pipeline',
    view_cloud_title: 'Cloud GPU Runners & Clusters',
    tab_runner: 'Pipeline Runner',
    tab_cloud: 'Cloud GPU Hub',
    view_agent_title: 'Autonomous Agent Workspace & Diagnostics',
    view_priming_title: 'Copilot Priming Prompts & System Instructions',
    tab_agent_ws: 'Agent Workspace',
    tab_priming: 'Priming Prompts',
    footer_evidence_notice: 'Verified evidence is required for eligible runs. Legacy artifacts are display only.',
    footer_budget_notice: 'Hard runtime budgets: local wall-time; remote allocation limits unavailable.',

    // Header
    header_switch_task: 'Switch Competition Task',
    header_configure_task: 'Configure or Add Task',
    header_best_label: 'Best:',
    header_readiness: 'Readiness',
    header_readiness_tooltip: 'Evaluate 8 Competition Readiness Dimensions',
    header_setup: 'Setup',
    header_setup_tooltip: 'Configure task platform, metrics, and subtasks',
    header_env: 'Env',
    header_env_tooltip: 'Inspect & configure Modal, Kaggle, Docker, and local GPU accounts',
    header_strategy: 'Strategy',
    header_strategy_tooltip: 'Open Strategy Catalog and configure experiment targets',
    header_sync: 'Sync',
    header_synced: 'Synced',
    header_sync_failed: 'Sync Failed',
    header_sync_tooltip: 'Scan experiment artifacts for new runs',

    // Error & Resilience
    err_server_offline_title: 'Backend Service Offline or Initializing',
    err_server_offline_desc: 'The ML-Harness API service is unreachable. If launching, it will connect automatically. Run `ml-harness dashboard` in terminal.',
    err_service_notice_title: 'Workspace Service Notice',
    err_offline_banner_title: 'Backend Service Offline or Initializing',
    err_offline_banner_desc: 'Telemetry unavailable. Ensure `ml-harness dashboard` is running.',
    btn_retry: 'Retry',
    btn_details: 'Details',
    btn_hide_details: 'Hide',
    btn_dismiss: 'Dismiss',

    // Copilot Banner
    copilot_title: 'Autonomous Copilot',
    copilot_ready: 'READY',
    copilot_running: 'RUNNING',
    copilot_stage_target: 'Stage {stage} Target',
    copilot_next_label: 'Next:',
    copilot_approve_action: 'Approve Action',
    copilot_run_loop: 'Run Copilot Loop',
    copilot_is_running: 'Running...',
    copilot_details: 'Details',
    copilot_run_goal: 'Autonomous Run Goal',
    copilot_max_exp: 'Max Experiments',
    copilot_plan_checklist: 'Active Plan Checklist',
    copilot_parallel_tracks: 'Parallel Tracks',
    copilot_no_steps: 'No background task steps recorded. Ready for autonomous cycle.',

    // KPI Cards
    kpi_current_phase: 'Current Phase',
    kpi_runs: 'Runs:',
    kpi_subs: 'Subs:',
    best_score_label: 'Best Score',
    kpi_best_validation_title: 'Best Validation Score',
    status_not_started: 'Not Started',
    status_awaiting_phase3: 'Awaiting Phase 3',
    status_awaiting_phase3_baseline: 'Awaiting Phase 3 Baseline',
    anchor_label: 'Anchor:',
    status_pending_phase3: 'Pending Phase 3',
    status_verified: 'Verified',
    kpi_external_eval: 'External Eval',
    kpi_quota_label: 'Quota: {quota}/day',
    kpi_portfolio_limit: 'Portfolio Limit:',
    kpi_slots_max: '{slots} slots max',
    kpi_sub_id: 'Sub ID:',
    kpi_action_items: 'Action Items',
    kpi_all_checks_passed: 'All checks passed — ready to iterate!',
    kpi_readiness_evaluated: '8 Dimensions Evaluated',

    // Journey Stepper & Stages
    journey_prev_phase: 'Prev Phase',
    journey_next_phase: 'Next Phase',
    journey_stage_counter: 'Stage {current} of 8:',

    // Stage 1: Setup
    stage1_title: '1. Competition Setup: {name}',
    stage1_desc: 'Loaded dynamically from Competition Spec (configs/task.yaml)',
    stage1_edit_spec: 'Edit Spec',
    stage1_metric_card: 'Primary Evaluation Metric',
    stage1_constraints_card: 'Declared Constraints',
    stage1_protocol_card: 'Evaluation Protocol',
    stage1_agent_context_settings: 'Advanced Agent Context Injection Settings',
    stage1_opt_phase_hud: 'Current Stage & Active Parallel Tracks',
    stage1_opt_anchors: '4-Tier Score Anchors (Baseline / CV / External / Final)',
    stage1_opt_blockers: 'Active Blockers & Required Confirmations',
    stage1_opt_next_action: 'Next Recommended Action & Agent Plan',
    stage1_opt_data_risk: 'Spec-Driven Data Risk Warnings',
    stage1_opt_candidate_pool: 'Candidate Pool & Subtask Specialists',
    status_saving: 'Saving...',
    status_saved: 'Saved',
    status_failed_reverted: 'Failed (reverted)',

    // Stage 2: Data & Problem Understanding
    stage2_title: '2. Data & Problem Understanding',
    stage2_data_confirmed: 'Data Readiness Confirmed',
    stage2_review_required: 'Data Review Required',
    stage2_total_samples: 'Total Samples',
    stage2_groups: 'Groups',
    stage2_spec_checks: 'Spec Checks',
    stage2_cross_fold_leakage: 'Cross-Fold Leakage',
    stage2_unlock_baseline_btn: 'Data readiness confirmed — Continue to Baseline',
    stage2_data_check_controls: 'Data Check Configuration Controls',
    stage2_mode_deep: 'Deep Audit (All Spec Checks & Outliers)',
    stage2_mode_light: 'Light Audit (Metadata & Leakage Only)',
    stage2_mode_manual: 'Manual Mode',
    stage2_chk_group_leakage: 'Group Isolation & Split Leakage Check',
    stage2_chk_conditional_labels: 'Conditional Target Rules Check',
    stage2_chk_label_distribution: 'Class Balance & Label Sparsity Check',
    stage2_chk_text_length_outliers: 'Input Length & Context Dilution Check',

    // Stage 3: Baseline
    stage3_title: '3. Establish Official Baseline Anchor',
    stage3_desc: 'Locks the foundational reference model before opening parallel exploration tracks.',
    stage3_lock_btn: 'Establish Baseline Anchor & Activate Parallel Tracks',

    // Stage 4: Experimentation Loop
    stage4_title: '4. Experimentation Track',
    stage4_tab_cards: 'Candidate Cards',
    stage4_tab_table: 'Leaderboard Table',
    stage4_tab_tree: 'Solution Tree',
    stage4_tab_runner: 'Execution Runner',
    stage4_tab_cloud: 'Cloud GPU Hub',
    stage4_filter_all: 'ALL',
    stage4_filter_keep: 'KEEP',
    stage4_filter_reject: 'REJECT',
    stage4_empty: 'No candidate experiments matching this filter.',
    stage4_val_score: 'Validation Score',
    stage4_vs_baseline: 'vs. Baseline',
    stage4_fold_std: 'Fold Std',
    stage4_promote_btn: 'Promote Candidate',

    // Stage 5: Candidate Management Pool
    stage5_title: '5. Candidate Management Pool',
    stage5_strategy_single: 'Single Model Candidate',
    stage5_strategy_ensemble: 'Diverse Ensemble Blend',
    stage5_strategy_routing: 'Multi-Subtask Specialist Routing',
    stage5_strategy_checkpoint: 'Checkpoint Selection',
    stage5_create_btn: 'Create Verified Candidate',

    // Stage 7: Freeze Policy
    stage7_title: '7. Final Selection Policy (Max Allowed Candidates: {max})',
    stage7_slot1: 'Slot 1: Primary Safe Candidate',
    stage7_slot2: 'Slot 2: Secondary Candidate',
    stage7_unfreeze_btn: 'Emergency Unfreeze',
    stage7_freeze_btn: 'Freeze Final Portfolio',
    stage7_audit_trail: 'Audit Trail',

    // Stage 8: Closure
    stage8_title: '8. Competition Closure',
    stage8_export_btn: 'Export Report',

    // Workspace Hub
    wshub_title: 'Agent workspace',
    wshub_desc: 'Prompts and phase history stay in this task workspace. Copy a phase prompt into your IDE agent, then save its response and the sources or artifacts it actually used.',
    wshub_refresh_btn: 'Refresh workspace state',
    wshub_prompt_pack: 'Prompt pack:',
    wshub_next_phase: 'Next phase:',
    wshub_evidence: 'Evidence:',
    wshub_eight_phases: 'Eight competition phases',
    wshub_copy_prompt_btn: 'Copy prompt for IDE agent',
    wshub_agent_response: 'Agent response',
    wshub_agent_response_ph: "Paste the IDE agent's phase response here. The saved output is checked against the phase contract.",
    wshub_sources_inspected: 'Sources actually inspected (JSON array)',
    wshub_artifact_paths: 'Workspace artifact paths (one per line)',
    wshub_save_output_btn: 'Save phase output',
    wshub_saving: 'Saving…',
    wshub_eda_title: 'Exploratory data analysis',
    wshub_eda_desc: "Profiles declared training data locally. Reports omit raw rows and are saved under this task's artifact tree.",
    wshub_run_eda_btn: 'Run EDA',
    wshub_accounts_title: 'Kaggle and Modal accounts',
    wshub_accounts_desc: 'The harness reads official CLI profiles and credentials. Provider quotas and charges depend on the account.',
    wshub_account_setup_btn: 'Account setup',
    wshub_upload_consent: 'I approve uploading these {count} task data files to {provider} for this fingerprint. Keep the upload private.',
    wshub_approve_upload_btn: 'Approve {provider} upload',
    wshub_jobs_title: 'Execution jobs',
    wshub_jobs_desc: "Shared job state from CLI and dashboard. Results remain in this workspace's artifact tree.",
    wshub_refreshes_interval: 'refreshes every 10 seconds',
    wshub_th_job: 'Job',
    wshub_th_provider: 'Provider',
    wshub_th_exp: 'Experiment',
    wshub_th_status: 'Status',
    wshub_th_metrics: 'Metrics',
    wshub_th_artifacts: 'Artifacts',
    wshub_no_jobs: 'No execution jobs recorded yet.',

    // Node Inspector
    inspector_primary_metric: 'Score:',
    inspector_scale: 'Scale:',
    inspector_tab_metrics: 'Metrics',
    inspector_tab_hypothesis: 'Hypothesis',
    inspector_tab_llm: 'LLM Specs',
    inspector_tab_diagnostics: 'Diagnostics',
    inspector_tab_grounding: 'Grounding',
    inspector_tab_config: 'Config',
    inspector_perf_breakdown: 'Performance Breakdown',
    inspector_artifacts_disk: 'Immutable Artifacts on Disk',
    inspector_atomic_hypothesis: 'Atomic Scientific Hypothesis',
    inspector_intervention: 'Intervention (Control vs Treatment)',
    inspector_bottlenecks: 'Diagnosed Bottlenecks',
    inspector_grounding_pass: 'Grounding Verified (PASS)',
    inspector_assumption_register: 'Assumption Register (Uncertainty Level)',
    inspector_config_title: 'Experiment Target Configuration',

    // Task Config Modal
    modal_task_title: 'Competition & Benchmark Setup',
    modal_task_desc: 'Configure task objectives, cross-validation data splits, and competition rules.',
    tab_validation_protocol: '3. Validation Split',
    modal_tab_obj: '1. Objective & Metrics',
    modal_tab_data: '2. Data Schema & Strategy',
    modal_tab_rules: '4. Rules & Constraints',
    modal_tab_sub: '5. Submission & Audit',
    modal_task_presets_label: 'Quick Competition Presets (1-Click Fill All Tabs)',
    modal_task_platform_modality: 'Competition Platform / Problem Modality',
    modal_task_primary_metric_title: 'Primary Benchmark Metric & Target',
    modal_task_pick_metric: 'Pick Standard ML Metric',
    modal_task_subtasks_title: 'Subtasks & Auxiliary Metrics',
    btn_save_task_config: 'Save Benchmark Specification',
    btn_cancel: 'Cancel',
  },
  vi: {
    // App & Nav
    app_title: 'Hệ thống Quản lý Thực nghiệm ML',
    nav_journey: 'Hành trình Cuộc thi (Giai đoạn 1–8)',
    nav_experiments_hub: 'Thực nghiệm & Cây Giải pháp',
    nav_execution: 'Thực thi & Cloud',
    nav_llm_studio: 'LLM Studio',
    nav_submissions: 'Lịch sử Nộp bài',
    nav_agent_copilot: 'Agent & Prompts Hub',
    nav_experiments: 'Thực nghiệm',
    nav_tree: 'Cây tìm kiếm AIDE',
    nav_compare: 'So sánh',
    nav_cloud_runner: 'Cloud Runner',
    nav_priming: 'Trung tâm Khởi tạo',
    new_experiment: 'Thực nghiệm mới',
    task_settings: 'Cấu hình Bài toán',
    refresh: 'Làm mới',
    language_label: 'Ngôn ngữ',
    loading_telemetry: 'Đang tải dữ liệu đo lường & cây thử nghiệm…',
    connection_error: 'Lỗi kết nối ML-Harness',
    retry_connection: 'Thử kết nối lại',
    recorded_runs: '{count} lượt chạy đã ghi nhận',
    view_leaderboard_title: 'Bảng xếp hạng & Lịch sử thử nghiệm',
    view_tree_title: 'Đồ thị Cây giải pháp AIDE',
    tab_leaderboard: 'Bảng xếp hạng',
    tab_tree: 'Cây giải pháp',
    view_runner_title: 'Quy trình thực thi 4 bước',
    view_cloud_title: 'Cụm máy chủ & GPU Cloud',
    tab_runner: 'Bộ chạy quy trình',
    tab_cloud: 'Trung tâm GPU Cloud',
    view_agent_title: 'Không gian làm việc Agent & Chẩn đoán',
    view_priming_title: 'Lời nhắc khởi tạo Copilot & Chỉ dẫn hệ thống',
    tab_agent_ws: 'Không gian Agent',
    tab_priming: 'Lời nhắc khởi tạo',
    footer_evidence_notice: 'Bắt buộc bằng chứng xác thực cho lượt chạy hợp lệ. Dữ liệu cũ chỉ dùng để hiển thị.',
    footer_budget_notice: 'Ngân sách thời gian cứng: thời gian chạy cục bộ; giới hạn cấp phát từ xa không khả dụng.',

    // Header
    header_switch_task: 'Chuyển đổi bài toán cuộc thi',
    header_configure_task: 'Cấu hình hoặc thêm bài toán',
    header_best_label: 'Điểm cao nhất:',
    header_readiness: 'Mức độ sẵn sàng',
    header_readiness_tooltip: 'Đánh giá 8 chiều mức độ sẵn sàng cho cuộc thi',
    header_setup: 'Thiết lập',
    header_setup_tooltip: 'Cấu hình nền tảng, chỉ số và tác vụ phụ',
    header_env: 'Môi trường',
    header_env_tooltip: 'Kiểm tra & cấu hình tài khoản Modal, Kaggle, Docker và GPU cục bộ',
    header_strategy: 'Chiến lược',
    header_strategy_tooltip: 'Mở danh mục chiến lược và cấu hình mục tiêu thử nghiệm',
    header_sync: 'Đồng bộ',
    header_synced: 'Đã đồng bộ',
    header_sync_failed: 'Lỗi đồng bộ',
    header_sync_tooltip: 'Quét thành phần thực nghiệm để tìm lượt chạy mới',

    // Error & Resilience
    err_server_offline_title: 'Dịch vụ nền đang khởi tạo hoặc ngoại tuyến',
    err_server_offline_desc: 'Không thể kết nối đến dịch vụ ML-Harness API. Nếu đang khởi động, hệ thống sẽ tự kết nối lại. Bạn có thể chạy `ml-harness dashboard` trong terminal.',
    err_service_notice_title: 'Thông báo dịch vụ không gian làm việc',
    err_offline_banner_title: 'Dịch vụ nền đang ngoại tuyến hoặc khởi tạo',
    err_offline_banner_desc: 'Dữ liệu đo lường chưa sẵn sàng. Hãy đảm bảo lệnh `ml-harness dashboard` đang chạy.',
    btn_retry: 'Thử lại',
    btn_details: 'Chi tiết',
    btn_hide_details: 'Ẩn',
    btn_dismiss: 'Bỏ qua',

    // Copilot Banner
    copilot_title: 'Bộ gợi ý trợ lý AI tự động',
    copilot_ready: 'SẴN SÀNG',
    copilot_running: 'ĐANG CHẠY',
    copilot_stage_target: 'Mục tiêu Giai đoạn {stage}',
    copilot_next_label: 'Tiếp theo:',
    copilot_approve_action: 'Phê duyệt hành động',
    copilot_run_loop: 'Chạy vòng lặp Copilot',
    copilot_is_running: 'Đang chạy...',
    copilot_details: 'Chi tiết',
    copilot_run_goal: 'Mục tiêu lượt chạy tự động',
    copilot_max_exp: 'Số thử nghiệm tối đa',
    copilot_plan_checklist: 'Danh mục kế hoạch hoạt động',
    copilot_parallel_tracks: 'Các luồng song song',
    copilot_no_steps: 'Chưa có bước tác vụ nền nào. Sẵn sàng cho chu kỳ tự động.',

    // KPI Cards
    kpi_current_phase: 'Giai đoạn hiện tại',
    kpi_runs: 'Thử nghiệm:',
    kpi_subs: 'Lần nộp:',
    best_score_label: 'Điểm cao nhất',
    kpi_best_validation_title: 'Điểm đánh giá tốt nhất',
    status_not_started: 'Chưa bắt đầu',
    status_awaiting_phase3: 'Chờ Giai đoạn 3',
    status_awaiting_phase3_baseline: 'Chờ Baseline Giai đoạn 3',
    anchor_label: 'Điểm tựa:',
    status_pending_phase3: 'Chờ Giai đoạn 3',
    status_verified: 'Đã xác thực',
    kpi_external_eval: 'Đánh giá đối ngoại',
    kpi_quota_label: 'Hạn ngạch: {quota}/ngày',
    kpi_portfolio_limit: 'Giới hạn danh mục:',
    kpi_slots_max: 'Tối đa {slots} suất',
    kpi_sub_id: 'Mã lần nộp:',
    kpi_action_items: 'Danh mục hành động',
    kpi_all_checks_passed: 'Tất cả kiểm tra hợp lệ — sẵn sàng thử nghiệm!',
    kpi_readiness_evaluated: '8 Chiều sẵn sàng đã đánh giá',

    // Journey Stepper & Stages
    journey_prev_phase: 'Giai đoạn trước',
    journey_next_phase: 'Giai đoạn sau',
    journey_stage_counter: 'Giai đoạn {current} / 8:',

    // Stage 1: Setup
    stage1_title: '1. Thiết lập cuộc thi: {name}',
    stage1_desc: 'Tải động từ Đặc tả Cuộc thi (configs/task.yaml)',
    stage1_edit_spec: 'Chỉnh sửa đặc tả',
    stage1_metric_card: 'Chỉ số đánh giá chính',
    stage1_constraints_card: 'Ràng buộc đã khai báo',
    stage1_protocol_card: 'Giao thức đánh giá',
    stage1_agent_context_settings: 'Cấu hình tiêm ngữ cảnh nâng cao cho Agent',
    stage1_opt_phase_hud: 'Giai đoạn hiện tại & Các luồng song song',
    stage1_opt_anchors: '4 Mức điểm tựa (Baseline / CV / Ngoại / Cuối)',
    stage1_opt_blockers: 'Điểm nghẽn hoạt động & Xác nhận bắt buộc',
    stage1_opt_next_action: 'Hành động đề xuất kế tiếp & Kế hoạch Agent',
    stage1_opt_data_risk: 'Cảnh báo rủi ro dữ liệu theo đặc tả',
    stage1_opt_candidate_pool: 'Kho ứng viên & Chuyên gia tác vụ phụ',
    status_saving: 'Đang lưu...',
    status_saved: 'Đã lưu',
    status_failed_reverted: 'Thất bại (đã hoàn tác)',

    // Stage 2: Data & Problem Understanding
    stage2_title: '2. Khám phá dữ liệu & Bài toán',
    stage2_data_confirmed: 'Đã xác nhận sẵn sàng dữ liệu',
    stage2_review_required: 'Yêu cầu rà soát dữ liệu',
    stage2_total_samples: 'Tổng số mẫu',
    stage2_groups: 'Nhóm',
    stage2_spec_checks: 'Kiểm tra đặc tả',
    stage2_cross_fold_leakage: 'Rò rỉ giữa các Fold',
    stage2_unlock_baseline_btn: 'Đã xác nhận sẵn sàng dữ liệu — Tiến tới tạo Baseline',
    stage2_data_check_controls: 'Cấu hình kiểm tra dữ liệu',
    stage2_mode_deep: 'Kiểm toán sâu (Toàn bộ kiểm tra đặc tả & ngoại lai)',
    stage2_mode_light: 'Kiểm toán nhanh (Chỉ siêu dữ liệu & rò rỉ)',
    stage2_mode_manual: 'Chế độ thủ công',
    stage2_chk_group_leakage: 'Kiểm tra cách ly nhóm & rò rỉ chia tách',
    stage2_chk_conditional_labels: 'Kiểm tra quy tắc nhãn có điều kiện',
    stage2_chk_label_distribution: 'Kiểm tra cân bằng lớp & độ thưa của nhãn',
    stage2_chk_text_length_outliers: 'Kiểm tra độ dài đầu vào & pha loãng ngữ cảnh',

    // Stage 3: Baseline
    stage3_title: '3. Thiết lập điểm tựa Baseline chính thức',
    stage3_desc: 'Khóa mô hình chuẩn tham chiếu trước khi mở các luồng thử nghiệm song song.',
    stage3_lock_btn: 'Khóa điểm tựa Baseline & Kích hoạt các luồng song song',

    // Stage 4: Experimentation Loop
    stage4_title: '4. Luồng thử nghiệm mô hình',
    stage4_tab_cards: 'Thẻ ứng viên',
    stage4_tab_table: 'Bảng xếp hạng',
    stage4_tab_tree: 'Cây giải pháp',
    stage4_tab_runner: 'Bộ chạy thực thi',
    stage4_tab_cloud: 'Trung tâm GPU Cloud',
    stage4_filter_all: 'TẤT CẢ',
    stage4_filter_keep: 'GIỮ',
    stage4_filter_reject: 'LOẠI BỎ',
    stage4_empty: 'Không có thử nghiệm ứng viên nào khớp với bộ lọc này.',
    stage4_val_score: 'Điểm xác thực',
    stage4_vs_baseline: 'So với Baseline',
    stage4_fold_std: 'Độ lệch chuẩn Fold',
    stage4_promote_btn: 'Nâng hạng ứng viên',

    // Stage 5: Candidate Management Pool
    stage5_title: '5. Kho quản lý ứng viên',
    stage5_strategy_single: 'Ứng viên đơn mô hình',
    stage5_strategy_ensemble: 'Kết hợp Ensemble đa dạng',
    stage5_strategy_routing: 'Định tuyến chuyên gia theo tác vụ phụ',
    stage5_strategy_checkpoint: 'Chọn lựa Checkpoint',
    stage5_create_btn: 'Tạo ứng viên đã xác thực',

    // Stage 7: Freeze Policy
    stage7_title: '7. Chính sách chọn lựa cuối cùng (Tối đa: {max} ứng viên)',
    stage7_slot1: 'Suất 1: Ứng viên an toàn chính',
    stage7_slot2: 'Suất 2: Ứng viên thứ hai',
    stage7_unfreeze_btn: 'Mở băng khẩn cấp',
    stage7_freeze_btn: 'Đóng băng danh mục cuối cùng',
    stage7_audit_trail: 'Nhật ký kiểm toán',

    // Stage 8: Closure
    stage8_title: '8. Tổng kết cuộc thi & Đóng gói',
    stage8_export_btn: 'Xuất báo cáo tổng kết',

    // Workspace Hub
    wshub_title: 'Không gian làm việc của Agent',
    wshub_desc: 'Lời nhắc và lịch sử các giai đoạn được lưu trong thư mục bài toán này. Sao chép lời nhắc giai đoạn vào IDE Agent, sau đó lưu phản hồi cùng các nguồn hoặc tệp kết quả thực tế.',
    wshub_refresh_btn: 'Làm mới trạng thái không gian làm việc',
    wshub_prompt_pack: 'Gói lời nhắc:',
    wshub_next_phase: 'Giai đoạn kế tiếp:',
    wshub_evidence: 'Bằng chứng:',
    wshub_eight_phases: 'Tám giai đoạn cuộc thi',
    wshub_copy_prompt_btn: 'Sao chép lời nhắc cho IDE Agent',
    wshub_agent_response: 'Phản hồi của Agent',
    wshub_agent_response_ph: 'Dán phản hồi giai đoạn của IDE Agent vào đây. Kết quả lưu sẽ được kiểm tra đối chiếu hợp đồng giai đoạn.',
    wshub_sources_inspected: 'Nguồn đã tham khảo thực tế (Mảng JSON)',
    wshub_artifact_paths: 'Đường dẫn tệp kết quả (mỗi dòng một tệp)',
    wshub_save_output_btn: 'Lưu kết quả giai đoạn',
    wshub_saving: 'Đang lưu…',
    wshub_eda_title: 'Phân tích khám phá dữ liệu (EDA)',
    wshub_eda_desc: 'Lập hồ sơ dữ liệu huấn luyện cục bộ. Báo cáo không chứa dòng dữ liệu thô và được lưu trong cây kết quả bài toán.',
    wshub_run_eda_btn: 'Chạy phân tích EDA',
    wshub_accounts_title: 'Tài khoản Kaggle và Modal',
    wshub_accounts_desc: 'Hệ thống đọc cấu hình và thông tin xác thực CLI chính thức. Hạn ngạch và chi phí phụ thuộc vào tài khoản.',
    wshub_account_setup_btn: 'Cấu hình tài khoản',
    wshub_upload_consent: 'Tôi đồng ý tải {count} tệp dữ liệu bài toán lên {provider} cho mã nhận diện này. Giữ chế độ riêng tư.',
    wshub_approve_upload_btn: 'Phê duyệt tải lên {provider}',
    wshub_jobs_title: 'Các tác vụ thực thi',
    wshub_jobs_desc: 'Trạng thái tác vụ dùng chung từ CLI và bảng điều khiển. Kết quả được lưu trong cây kết quả bài toán.',
    wshub_refreshes_interval: 'tự động làm mới mỗi 10 giây',
    wshub_th_job: 'Tác vụ',
    wshub_th_provider: 'Nền tảng',
    wshub_th_exp: 'Thực nghiệm',
    wshub_th_status: 'Trạng thái',
    wshub_th_metrics: 'Chỉ số',
    wshub_th_artifacts: 'Tệp kết quả',
    wshub_no_jobs: 'Chưa có tác vụ thực thi nào được ghi nhận.',

    // Node Inspector
    inspector_primary_metric: 'Điểm số:',
    inspector_scale: 'Quy mô:',
    inspector_tab_metrics: 'Chỉ số',
    inspector_tab_hypothesis: 'Giả thuyết',
    inspector_tab_llm: 'Cấu hình LLM',
    inspector_tab_diagnostics: 'Chẩn đoán',
    inspector_tab_grounding: 'Căn cứ thực nghiệm',
    inspector_tab_config: 'Cấu hình',
    inspector_perf_breakdown: 'Phân tích hiệu năng chi tiết',
    inspector_artifacts_disk: 'Các thành phần bất biến trên đĩa',
    inspector_atomic_hypothesis: 'Giả thuyết khoa học đơn nguyên',
    inspector_intervention: 'Biện pháp can thiệp (Đối chứng vs Thử nghiệm)',
    inspector_bottlenecks: 'Các điểm nghẽn chẩn đoán',
    inspector_grounding_pass: 'Đã xác thực căn cứ thực nghiệm (ĐẠT)',
    inspector_assumption_register: 'Sổ theo dõi giả định (Mức độ không chắc chắn)',
    inspector_config_title: 'Cấu hình mục tiêu thử nghiệm',

    // Task Config Modal
    modal_task_title: 'Thiết lập Cuộc thi & Chuẩn đánh giá',
    modal_task_desc: 'Cấu hình mục tiêu bài toán, phân chia tập xác thực và quy tắc cuộc thi.',
    tab_validation_protocol: '3. Phân chia Tập xác thực',
    modal_tab_obj: '1. Mục tiêu & Chỉ số',
    modal_tab_data: '2. Lược đồ & Chiến lược Dữ liệu',
    modal_tab_rules: '4. Quy tắc & Ràng buộc',
    modal_tab_sub: '5. Nộp bài & Kiểm toán',
    modal_task_presets_label: 'Bộ mẫu cuộc thi nhanh (1-Click điền toàn bộ)',
    modal_task_platform_modality: 'Nền tảng cuộc thi / Định dạng dữ liệu bài toán',
    modal_task_primary_metric_title: 'Chỉ số chuẩn & Điểm mục tiêu chính',
    modal_task_pick_metric: 'Chọn chỉ số ML tiêu chuẩn',
    modal_task_subtasks_title: 'Các tác vụ phụ & Chỉ số bổ trợ',
    btn_save_task_config: 'Lưu Đặc tả Chuẩn đánh giá',
    btn_cancel: 'Hủy bỏ',
  },
};

interface LanguageContextValue {
  lang: UILanguage;
  setLang: (lang: UILanguage) => void;
  t: (key: string, fallback?: string, params?: Record<string, string | number>) => string;
}

const interpolate = (text: string, params?: Record<string, string | number>): string => {
  if (!params) return text;
  let res = text;
  for (const [k, v] of Object.entries(params)) {
    res = res.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v));
  }
  return res;
};

const LanguageContext = createContext<LanguageContextValue>({
  lang: 'en',
  setLang: () => {},
  t: (key: string, fallback?: string, params?: Record<string, string | number>) => {
    const raw = UI_TRANSLATIONS.en[key] || fallback || key;
    return interpolate(raw, params);
  },
});

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lang, setLangState] = useState<UILanguage>(() => {
    const saved = localStorage.getItem('ml_harness_ui_lang');
    return saved === 'vi' ? 'vi' : 'en';
  });

  const setLang = (next: UILanguage) => {
    setLangState(next);
    localStorage.setItem('ml_harness_ui_lang', next);
  };

  const t = (key: string, fallback?: string, params?: Record<string, string | number>): string => {
    const raw = UI_TRANSLATIONS[lang]?.[key] || UI_TRANSLATIONS.en[key] || fallback || key;
    return interpolate(raw, params);
  };

  return React.createElement(
    LanguageContext.Provider,
    { value: { lang, setLang, t } },
    children
  );
};

export const useLanguage = () => useContext(LanguageContext);
