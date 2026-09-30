export type ApproachType = 'ml' | 'dl' | 'llm' | 'hybrid';
export type BranchType = 'baseline' | 'feat' | 'model' | 'loss' | 'mtl' | 'ens' | string;
export type NodeStatus = 'COMPLETED' | 'BUGGY' | 'PRUNED' | 'RUNNING' | string;

export interface LLMSpec {
  model_name: string;
  param_size?: string | null; // e.g. "7B", "14B", "70B", "8B", "1.5B"
  param_count?: number | null; // e.g. 7000000000
  quantization?: string | null; // e.g. "4-bit NormalFloat (NF4)", "8-bit", "bfloat16"
  paradigm?: 'unsloth_qlora' | 'zero_shot_cot' | 'few_shot_icl' | 'ensemble_blend' | string;
  context_length?: number | null; // e.g. 2048, 8192
  provider?: 'unsloth' | 'huggingface' | 'gemini' | 'openrouter' | 'groq' | 'local' | string;
  lora_r?: number | null; // e.g. 16
  lora_alpha?: number | null; // e.g. 32
  lora_dropout?: number | null; // e.g. 0.05
  target_modules?: string | null; // e.g. "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  temperature?: number | null;
  use_cot?: boolean;
  system_prompt?: string | null;
  prompt_template?: string | null;
  few_shot_count?: number | null;
}

export interface GraphNode {
  id: string;
  label: string;
  parent_id: string | null;
  branch: BranchType;
  approach_type: ApproachType;
  model_family: string;
  score: number;
  parent_delta?: number | null;
  param_count: number | null;
  param_size?: string | null;
  llm_spec?: LLMSpec | null;
  status: NodeStatus;
  is_best: boolean;
  is_buggy: boolean;
  atomic_hypothesis?: string;
  intervention?: string;
  timestamp: number;
  primary_metric_name?: string;
  metric_name?: string;
  metrics?: Record<string, any>;
  diagnostics_summary?: Record<string, any>;
  layoutDirection?: 'TB' | 'LR';
}

export interface GraphEdge {
  from: string;
  to: string;
  delta: number;
  improved: boolean;
}

export interface TreeGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  best_id: string | null;
}

export interface ExperimentSummary {
  id: string;
  branch: BranchType;
  parent_id: string;
  approach_type: ApproachType;
  model_family: string;
  score: number;
  param_count: number | null;
  param_size?: string | null;
  llm_spec?: LLMSpec | null;
  status: NodeStatus;
  is_best: boolean;
  is_buggy: boolean;
  hypothesis?: string;
  intervention?: string;
  timestamp: number;
  official_lb_score?: number | null;
}

export interface NodeDetail {
  id: string;
  parent_id: string | null;
  branch: BranchType;
  approach_type: ApproachType;
  model_family: string;
  score: number;
  param_count: number | null;
  param_size?: string | null;
  llm_spec?: LLMSpec | null;
  status: NodeStatus;
  is_buggy: boolean;
  atomic_hypothesis?: string;
  intervention?: string;
  notes?: string;
  timestamp: number;
  metrics: Record<string, any>;
  diagnostics_summary: Record<string, any>;
  grounding_receipt?: {
    receipt_id?: string;
    experiment_id?: string;
    status?: string;
    grounding?: {
      source_type: string;
      source_id: string;
      retrieval_summary: string;
      verified_via?: string;
    };
    assumption_register?: {
      known_facts?: string[];
      assumptions?: string[];
      uncertainty_level?: string;
      verification_probe?: string;
    };
  } | null;
  config?: {
    path: string;
    content: string;
  } | null;
}

export interface UCBRecommendation {
  recommendation?: string;
  parent_id: string | null;
  parent_score?: number;
  parent_params?: number | null;
  branch?: string;
  recommended_branch?: string;
  ucb_score?: number;
  bottlenecks?: string[];
  diagnostic_cue?: string;
  guideline: string;
}

export interface PromptTemplateInfo {
  id: string;
  ide?: string;
  intent?: string;
  name: string;
  badge: string;
  description: string;
}

export interface PrimingPromptResponse {
  template_id?: string;
  prompt: string;
  character_count: number;
}

export interface SystemStatus {
  status: string;
  total_nodes: number;
  best_node: string | null;
  best_score: number;
  distinct_families: number;
}

export interface TaskMetricDef {
  id: string;
  name: string;
  metric: string;
  type?: string;
  direction?: 'maximize' | 'minimize';
  baseline_score?: number;
  target_score?: number;
  weight?: number;
  threshold?: number;
  description?: string;
}

export interface DataConfig {
  train_path?: string;
  test_path?: string;
  sample_submission_path?: string;
  split_strategy?: 'GroupKFold' | 'StratifiedKFold' | 'KFold' | 'TimeSeriesSplit' | string;
  group_column?: string;
  partition_column?: string;
  n_folds?: number;
  seed?: number;
  id_column?: string;
  target_column?: string;
  split_file?: string;
  fold_file?: string;
  missing_data_policy?: string;
  external_data_policy?: string;
}

export interface CompetitionRules {
  allow_llm?: boolean;
  allow_api?: boolean;
  allow_external_data?: boolean;
  allow_pretrained_models?: boolean;
  max_params_limit?: number;
  max_gpu_hours?: number;
  daily_submission_limit?: number;
  deadline?: string;
  team_policy?: string;
  reproducibility_required?: boolean;
}

export interface SubmissionSpec {
  file_name?: string;
  archive_format?: 'zip' | 'csv' | 'tar.gz';
  required_columns?: string[];
  min_diff_pct?: number; // MMES floor, default 2.0%
  max_kappa?: number; // Ensemble diversity ceiling, default 0.85
  min_disagreement?: number; // Ensemble disagreement floor, default 5.0%
}

export interface CompetitionPillarCheck {
  id: string;
  title: string;
  status: 'passed' | 'warning' | 'pending';
  summary: string;
  details: string;
  tabKey?: 'task' | 'data' | 'rules' | 'submission';
}

export interface TaskInfo {
  platform?: string;
  name?: string;
  task_name: string;
  type?: string;
  task_type: string;
  modality?: string;
  primary_metric: string;
  direction?: 'maximize' | 'minimize';
  primary_metric_type?: string;
  primary_metric_baseline?: number;
  primary_metric_target?: number;
  metrics?: TaskMetricDef[];
  subtasks?: TaskMetricDef[];
  languages?: string[];
  description?: string;
  data_config?: DataConfig;
  competition_rules?: CompetitionRules;
  submission_spec?: SubmissionSpec;
  artifacts_dir?: string;
  task_plugin?: {
    id: string;
    version?: string | null;
    capabilities?: Record<string, boolean>;
    error?: string;
  };
}

export interface TaskPreset {
  id: string;
  preset_name: string;
  platform: string;
  name: string;
  type: string;
  modality: string;
  primary_metric: string;
  direction: 'maximize' | 'minimize';
  primary_metric_type?: string;
  primary_metric_baseline?: number;
  primary_metric_target?: number;
  metrics: TaskMetricDef[];
  languages: string[];
  description: string;
  data_config?: DataConfig;
  competition_rules?: CompetitionRules;
  submission_spec?: SubmissionSpec;
}

export interface CloudPreflightResult {
  passed: boolean;
  errors: string[];
  warnings?: string[];
  resolved_config?: any;
}

export interface CloudGenerateResult {
  success: boolean;
  backend: 'modal' | 'kaggle';
  script_path?: string;
  code: string;
  metadata?: Record<string, any>;
  commands: {
    dataset_cmd?: string;
    push_cmd: string;
    status_cmd?: string;
    output_cmd?: string;
  };
}

export interface RegisterSubmissionPayload {
  exp_id: string;
  candidate_csv?: string;
  baseline_csv?: string | null;
  notes?: string;
  min_diff_pct?: number;
  ensemble_components?: string[];
}

export interface SubmissionEntry {
  id: string;
  exp_id: string;
  created_at: string;
  status: 'LOCAL_VERIFIED' | 'SUBMITTED' | 'SCORED' | string;
  local_cv_scores: Record<string, any>;
  official_scores?: Record<string, any>;
  score_deltas?: Record<string, any>;
  candidate_path: string;
  baseline_path?: string | null;
  archive_path: string;
  receipt_path: string;
  notes: string;
  divergence_warning: boolean;
}

export interface EnvironmentStatus {
  modal: {
    installed: boolean;
    executable: string | null;
    profiles: string[];
    active_profile: string | null;
    has_credentials: boolean;
    error: string | null;
  };
  kaggle: {
    installed: boolean;
    executable: string | null;
    authenticated: boolean;
    username: string | null;
    config_dir: string;
    profiles: { name: string; username: string; is_active: boolean }[];
    error: string | null;
  };
  docker: {
    installed: boolean;
    executable: string | null;
    running: boolean;
    version: string | null;
    error: string | null;
  };
  gpu: {
    detected: boolean;
    gpu_name: string | null;
    memory_total_mb: number | null;
    memory_free_mb: number | null;
    driver_version: string | null;
    cuda_available: boolean;
  };
  llm?: {
    providers: Record<string, {
      configured: boolean;
      env_var: string | null;
      masked_key: string | null;
      is_local: boolean;
      endpoint: string | null;
      reachable: boolean;
    }>;
    total_configured: number;
  };
}

export interface CloudJob {
  job_id: string;
  backend: 'modal' | 'kaggle' | 'local';
  config_path: string;
  exp_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  return_code: number | null;
  started_at: number;
  ended_at: number | null;
  log_file: string;
  pid: number | null;
  auto_audit: boolean;
  auto_sync: boolean;
  baseline_csv: string | null;
  audit_passed: boolean | null;
  audit_receipt: Record<string, any> | null;
  error_message: string | null;
}

export interface TaskSummary {
  id: string;
  name: string;
  platform: string;
  type: string;
  modality: string;
  primary_metric: string;
  direction: string;
  file_path: string;
  is_active: boolean;
}

export interface NextExpIdResponse {
  next_id: string;
  next_number: number;
  branch: string;
  descriptor: string;
  suggested_file: string;
}

export interface LLMProviderInfo {
  id: string;
  name: string;
  models: string[];
  default_model: string;
  has_key: boolean;
  is_reachable: boolean;
  env_key_name: string | null;
  is_local: boolean;
}

export interface LLMSampleDataResponse {
  columns: string[];
  rows: Record<string, any>[];
  total_samples: number;
  source_file: string | null;
}

export interface LLMTestPromptRequest {
  provider: string;
  model: string;
  system_prompt: string;
  user_prompt_template: string;
  input_row: Record<string, any>;
  api_key?: string;
  base_url?: string;
  temperature?: number;
  max_tokens?: number;
  use_cot?: boolean;
  few_shot_examples?: Array<{ input: string; output: string; explanation?: string }>;
  valid_classes?: string[];
}

export interface LLMTestPromptResponse {
  success: boolean;
  provider: string;
  model: string;
  effective_system_prompt: string;
  effective_user_prompt: string;
  raw_text: string;
  extracted: {
    parsed_label: string;
    reasoning?: string;
    raw_text: string;
    is_valid: boolean;
  };
  latency_ms: number;
  usage: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
  error?: string;
}

export interface LLMPeftGenerateRequest {
  model_name: string;
  train_path?: string;
  output_dir?: string;
  text_column?: string;
  target_column?: string;
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  target_modules?: string;
  num_train_epochs?: number;
  batch_size?: number;
  gradient_accumulation_steps?: number;
  learning_rate?: number;
  save_to_disk?: boolean;
}

export interface LLMPeftGenerateResponse {
  success: boolean;
  script_code: string;
  saved_path?: string | null;
  error?: string;
}

export interface HFModelInfo {
  id: string;
  name: string;
  author?: string | null;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  last_modified?: string;
  is_unsloth: boolean;
  family?: string;
  param_size_b?: number | null;
  param_label?: string | null;
  quantization?: string;
  is_bnb_4bit?: boolean;
  is_gguf?: boolean;
  is_instruct?: boolean;
}

export interface UnslothModelInfo extends HFModelInfo {
  family: string;
  is_bnb_4bit: boolean;
  is_gguf: boolean;
  is_instruct: boolean;
  recommended_for_lora: boolean;
}

export interface ProviderModelInfo {
  id: string;
  name: string;
  context_length?: number;
  description?: string;
  is_live: boolean;
  size_bytes?: number;
}

export interface LLMUnslothGenerateRequest {
  model_name: string;
  train_path?: string;
  output_dir?: string;
  text_column?: string;
  target_column?: string;
  max_seq_length?: number;
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  target_modules?: string;
  num_train_epochs?: number;
  batch_size?: number;
  gradient_accumulation_steps?: number;
  learning_rate?: number;
  save_to_disk?: boolean;
}

export interface StrategyCard {
  id: string;
  branch: BranchType;
  badge?: string;
  title: string;
  problem_addressed: string;
  hypothesis: string;
  intervention: string;
  approach_type: ApproachType;
  model_family: string;
  model_name: string;
  descriptor: string;
  rationale: string;
  target_tasks: string[];
  estimated_delta?: string;
  param_size?: string | null;
  llm_spec?: LLMSpec | null;
  uncertainty_level?: string;
  source_type?: string;
  source_id?: string;
  model_params?: Record<string, any>;
}



