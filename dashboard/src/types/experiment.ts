export type ApproachType = 'ml' | 'dl' | 'hybrid';
export type BranchType = 'baseline' | 'feat' | 'model' | 'loss' | 'mtl' | 'ens' | string;
export type NodeStatus = 'COMPLETED' | 'BUGGY' | 'PRUNED' | 'RUNNING' | string;

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
  status: NodeStatus;
  is_best: boolean;
  is_buggy: boolean;
  hypothesis?: string;
  intervention?: string;
  timestamp: number;
  subtask_a?: number | null;
  subtask_b?: number | null;
  subtask_c?: number | null;
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
  n_folds?: number;
  seed?: number;
  id_column?: string;
  target_column?: string;
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
  local_cv_scores: {
    subtask_a_macro?: number | null;
    subtask_b_macro?: number | null;
    subtask_c_type_macro?: number | null;
    subtask_c_id_macro?: number | null;
    [key: string]: any;
  };
  official_scores?: {
    subtask_a_macro?: number | null;
    subtask_b_macro?: number | null;
    subtask_c_type_macro?: number | null;
    subtask_c_id_macro?: number | null;
    split_type?: string;
    [key: string]: any;
  };
  score_deltas?: {
    subtask_a_macro?: number | null;
    subtask_b_macro?: number | null;
    subtask_c_type_macro?: number | null;
    subtask_c_id_macro?: number | null;
    [key: string]: any;
  };
  candidate_path: string;
  baseline_path?: string | null;
  archive_path: string;
  receipt_path: string;
  notes: string;
  divergence_warning: boolean;
}
