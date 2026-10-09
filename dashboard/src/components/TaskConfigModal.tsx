import React, { useState, useEffect } from 'react';
import {
  X,
  Sliders,
  Sparkles,
  Plus,
  Trash2,
  Check,
  Award,
  Globe,
  HelpCircle,
  Layers,
  ArrowUpRight,
  ArrowDownRight,
  ChevronDown,
  ChevronUp,
  Database,
  Scale,
  FileArchive,
  ShieldCheck,
  RotateCcw,
} from 'lucide-react';
import { TaskInfo, TaskMetricDef, TaskPreset, DataConfig, CompetitionRules, SubmissionSpec } from '../types/experiment';
import { api } from '../services/api';
import { ML_METRICS_CATALOG, getMetricById, getSuggestedDirection } from '../constants/metrics';
import { useLanguage } from '../constants/i18n';

interface TaskConfigModalProps {
  isOpen: boolean;
  currentTask: TaskInfo | null;
  onClose: () => void;
  onSave: (updatedTask: TaskInfo) => Promise<void>;
  initialTab?: 'task' | 'data' | 'cv' | 'rules' | 'submission';
}

const PLATFORMS = [
  { id: 'codabench', name: 'Codabench', badge: 'Academic / Shared Tasks' },
  { id: 'kaggle', name: 'Kaggle', badge: 'Kaggle Competitions' },
  { id: 'aicrowd', name: 'AIcrowd', badge: 'Challenge Platform' },
  { id: 'drivendata', name: 'DrivenData', badge: 'Data Science for Good' },
  { id: 'huggingface', name: 'Hugging Face', badge: 'Community Leaderboard' },
  { id: 'custom', name: 'Custom / In-House', badge: 'Internal Benchmark' },
];

const MODALITIES = [
  { id: 'text', label: 'Text / NLP' },
  { id: 'tabular', label: 'Tabular' },
  { id: 'vision', label: 'Vision / Image' },
  { id: 'audio', label: 'Audio / Speech' },
  { id: 'time-series', label: 'Time-Series' },
  { id: 'multimodal', label: 'Multimodal' },
];

const DATA_STRATEGIES = [
  { id: 'full_feature_concat', name: 'Concatenate All Input Columns' },
  { id: 'primary_text_only', name: 'Primary Feature Column Only' },
  { id: 'multimodal_fusion', name: 'Multimodal / Tabular + Text Concatenation' },
  { id: 'standard_normalized', name: 'Standard Scaling & Normalization' },
];

const IMBALANCE_STRATEGIES = [
  { id: 'class_weighted', name: 'Inverse Frequency Class Weighting' },
  { id: 'focal_loss', name: 'Focal Loss / Hard Negative Mining' },
  { id: 'resampling', name: 'Stratified Resampling (Oversample / Undersample)' },
  { id: 'none', name: 'Standard Empirical Distribution (No Resampling)' },
];

const SPLIT_STRATEGIES = [
  { id: 'GroupKFold', name: 'GroupKFold (Grouped by Entity / Group ID)' },
  { id: 'StratifiedKFold', name: 'StratifiedKFold (Stratified by Target Class)' },
  { id: 'KFold', name: 'Standard KFold (Random Split)' },
  { id: 'TimeSeriesSplit', name: 'TimeSeriesSplit (Temporal Forward Split)' },
];

export type AuditRegime = 'classification' | 'regression' | 'ranking' | 'generative_nlp';

export function resolveAuditRegime(
  modality: string,
  taskType: string,
  primaryMetric: string
): AuditRegime {
  const t = (taskType || '').toLowerCase();
  const m = (primaryMetric || '').toLowerCase();
  const mod = (modality || '').toLowerCase();
  const metricObj = getMetricById(primaryMetric || '');
  const cat = (metricObj?.category || '').toLowerCase();

  // 1. Regression detection
  if (
    cat === 'regression' ||
    t.includes('regress') ||
    ['rmse', 'mae', 'mse', 'r2', 'rmsle', 'medae', 'mape'].some((k) => m.includes(k))
  ) {
    return 'regression';
  }

  // 2. Ranking detection
  if (
    cat === 'ranking' ||
    t.includes('rank') ||
    t.includes('recsys') ||
    ['ndcg', 'map', 'mrr', 'hitrate', 'kendall'].some((k) => m.includes(k))
  ) {
    return 'ranking';
  }

  // 3. Generative NLP detection
  if (
    (mod === 'text' || t.includes('llm') || t.includes('generat') || t.includes('causal-lm') || t.includes('language-model')) &&
    (cat === 'nlp' || ['bleu', 'rouge', 'perplexity', 'exactmatch', 'chrf', 'meteor', 'bertscore'].some((k) => m.includes(k)))
  ) {
    return 'generative_nlp';
  }

  if (
    t.includes('generat') ||
    t.includes('generation') ||
    t.includes('llm') ||
    t.includes('causal-lm') ||
    t.includes('language-model') ||
    t.includes('summariz') ||
    t.includes('translat')
  ) {
    return 'generative_nlp';
  }

  // 4. Default: Classification
  return 'classification';
}

const REGIME_DISPLAY_NAMES: Record<AuditRegime, string> = {
  classification: 'Classification',
  regression: 'Continuous Regression',
  ranking: 'Ranking & Recommendation',
  generative_nlp: 'NLP / Generative LLM',
};

export const getRegimeDefaults = (regime: AuditRegime) => {
  switch (regime) {
    case 'regression':
      return { minDiff: '2.0', maxKappa: '0.90', minDisagreement: '5.0' };
    case 'ranking':
      return { minDiff: '2.0', maxKappa: '0.85', minDisagreement: '5.0' };
    case 'generative_nlp':
      return { minDiff: '2.0', maxKappa: '0.92', minDisagreement: '5.0' };
    case 'classification':
    default:
      return { minDiff: '2.0', maxKappa: '0.85', minDisagreement: '5.0' };
  }
};

const CONTEXT_COL_SUGGESTIONS: Record<string, string[]> = {
  text: ['text', 'prompt', 'context', 'question', 'title', 'premise, hypothesis'],
  tabular: ['all_features', 'numerical_features', 'categorical_features', 'feature_cols'],
  vision: ['image_path', 'image_id', 'filepath', 'bbox', 'mask_path'],
  audio: ['audio_path', 'waveform_path', 'sampling_rate', 'transcript'],
  'time-series': ['timestamp', 'series_id', 'value', 'lag_1', 'group_id'],
  multimodal: ['text, image_path', 'features, context'],
};

const getContextColPlaceholder = (mod: string, t: (k: string, fallback?: string) => string): string => {
  switch (mod) {
    case 'text':
      return t('taskcfg_context_cols_ph_text', 'e.g. text, prompt, context, title');
    case 'tabular':
      return t('taskcfg_context_cols_ph_tabular', 'e.g. feat_1, feat_2, ..., numerical & categorical features');
    case 'vision':
      return t('taskcfg_context_cols_ph_vision', 'e.g. image_path, image_id, label');
    case 'time-series':
      return t('taskcfg_context_cols_ph_timeseries', 'e.g. timestamp, entity_id, value, lag_features');
    case 'multimodal':
      return t('taskcfg_context_cols_ph_multimodal', 'e.g. text, image_path, tabular_features');
    case 'audio':
      return t('taskcfg_context_cols_ph_audio', 'e.g. audio_path, sample_rate, transcript');
    default:
      return t('taskcfg_context_cols_ph_text', 'e.g. text, prompt, context, title');
  }
};

interface MissingDataPreset {
  id: string;
  labelKey: string;
  defaultLabel: string;
}

const MISSING_DATA_PRESETS: Record<string, MissingDataPreset[]> = {
  tabular: [
    { id: 'impute_median_mode', labelKey: 'taskcfg_missing_opt_median_mode', defaultLabel: 'Median (Numerical) / Mode (Categorical) Imputation' },
    { id: 'indicator_imputation', labelKey: 'taskcfg_missing_opt_indicator', defaultLabel: 'Mean Imputation + Missingness Indicator' },
    { id: 'drop_invalid_rows', labelKey: 'taskcfg_missing_opt_drop_rows', defaultLabel: 'Drop Incomplete / Corrupted Rows' },
    { id: 'constant_sentinel', labelKey: 'taskcfg_missing_opt_constant', defaultLabel: 'Constant Sentinel Value (-999, "missing")' },
  ],
  text: [
    { id: 'fill_empty_context', labelKey: 'taskcfg_missing_opt_fill_empty', defaultLabel: 'Fill Empty Text String / Default Context' },
    { id: 'truncate_and_pad', labelKey: 'taskcfg_missing_opt_truncate_pad', defaultLabel: 'Truncate Long Context / Pad Short Sequences' },
    { id: 'mark_unknown_token', labelKey: 'taskcfg_missing_opt_mark_unk', defaultLabel: 'Replace with [UNK] / Placeholder Token' },
    { id: 'drop_invalid_rows', labelKey: 'taskcfg_missing_opt_drop_rows', defaultLabel: 'Drop Incomplete / Corrupted Rows' },
  ],
  vision: [
    { id: 'drop_missing_files', labelKey: 'taskcfg_missing_opt_drop_missing_files', defaultLabel: 'Drop Missing Media References' },
    { id: 'substitute_blank_canvas', labelKey: 'taskcfg_missing_opt_blank_canvas', defaultLabel: 'Substitute Neutral Blank Canvas' },
    { id: 'drop_invalid_rows', labelKey: 'taskcfg_missing_opt_drop_rows', defaultLabel: 'Drop Incomplete / Corrupted Rows' },
  ],
  audio: [
    { id: 'drop_missing_files', labelKey: 'taskcfg_missing_opt_drop_missing_files', defaultLabel: 'Drop Missing Media References' },
    { id: 'fill_silence_constant', labelKey: 'taskcfg_missing_opt_constant', defaultLabel: 'Constant Sentinel Value (-999, "missing")' },
    { id: 'drop_invalid_rows', labelKey: 'taskcfg_missing_opt_drop_rows', defaultLabel: 'Drop Incomplete / Corrupted Rows' },
  ],
  'time-series': [
    { id: 'forward_backward_fill', labelKey: 'taskcfg_missing_opt_forward_fill', defaultLabel: 'Forward / Backward Fill Imputation' },
    { id: 'impute_median_mode', labelKey: 'taskcfg_missing_opt_median_mode', defaultLabel: 'Median (Numerical) / Mode (Categorical) Imputation' },
    { id: 'drop_invalid_rows', labelKey: 'taskcfg_missing_opt_drop_rows', defaultLabel: 'Drop Incomplete / Corrupted Rows' },
  ],
  multimodal: [
    { id: 'fill_empty_context', labelKey: 'taskcfg_missing_opt_fill_empty', defaultLabel: 'Fill Empty Text String / Default Context' },
    { id: 'drop_missing_files', labelKey: 'taskcfg_missing_opt_drop_missing_files', defaultLabel: 'Drop Missing Media References' },
    { id: 'drop_invalid_rows', labelKey: 'taskcfg_missing_opt_drop_rows', defaultLabel: 'Drop Incomplete / Corrupted Rows' },
  ],
};

interface ParamPreset {
  id: string;
  value: string;
  labelKey: string;
  defaultLabel: string;
}

const PARAM_PRESETS: ParamPreset[] = [
  { id: '100m', value: '100000000', labelKey: 'taskcfg_params_preset_100m', defaultLabel: '100M (Edge/Base)' },
  { id: '350m', value: '350000000', labelKey: 'taskcfg_params_preset_350m', defaultLabel: '350M (DeBERTa-large)' },
  { id: '1b', value: '1000000000', labelKey: 'taskcfg_params_preset_1b', defaultLabel: '1B (Small LLM / Vision)' },
  { id: '7b', value: '7000000000', labelKey: 'taskcfg_params_preset_7b', defaultLabel: '7B (LoRA/vLLM)' },
  { id: '14b', value: '14000000000', labelKey: 'taskcfg_params_preset_14b', defaultLabel: '14B (Reasoning / CoT)' },
  { id: '70b', value: '70000000000', labelKey: 'taskcfg_params_preset_70b', defaultLabel: '70B (Cluster LLM)' },
  { id: 'unlimited', value: '', labelKey: 'taskcfg_params_preset_unlimited', defaultLabel: 'Unlimited (No Ceiling)' },
];

const formatParamScale = (val: string, unlimitedLabel: string): string => {
  if (!val || val.trim() === '' || val === '0') {
    return unlimitedLabel;
  }
  const num = Number(val);
  if (isNaN(num)) return val;
  if (num >= 1e9) {
    const b = (num / 1e9).toFixed(1);
    return `${b}B parameters`;
  }
  if (num >= 1e6) {
    const m = (num / 1e6).toFixed(0);
    return `${m}M parameters`;
  }
  if (num >= 1e3) {
    return `${(num / 1e3).toFixed(0)}K parameters`;
  }
  return `${num} parameters`;
};

export const TaskConfigModal: React.FC<TaskConfigModalProps> = ({
  isOpen,
  currentTask,
  onClose,
  onSave,
  initialTab = 'task',
}) => {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState<'task' | 'data' | 'cv' | 'rules' | 'submission'>(initialTab);
  const [presets, setPresets] = useState<TaskPreset[]>([]);

  // Tab 1: Task & Objective
  const [platform, setPlatform] = useState<string>('custom');
  const [name, setName] = useState<string>('');
  const [type, setType] = useState<string>('');
  const [modality, setModality] = useState<string>('generic');
  const [primaryMetric, setPrimaryMetric] = useState<string>('Score');
  const [direction, setDirection] = useState<'maximize' | 'minimize'>('maximize');
  const [primaryBaseline, setPrimaryBaseline] = useState<string>('');
  const [primaryTarget, setPrimaryTarget] = useState<string>('');
  const [metrics, setMetrics] = useState<TaskMetricDef[]>([]);
  const [expandedMetricIndices, setExpandedMetricIndices] = useState<Record<number, boolean>>({});
  const [languages, setLanguages] = useState<string>('');
  const [description, setDescription] = useState<string>('');

  // Tab 2: Data Schema & Strategy
  const [trainPath, setTrainPath] = useState<string>('dataset/');
  const [testPath, setTestPath] = useState<string>('dataset/test/');
  const [sampleSubPath, setSampleSubPath] = useState<string>('submissions/sample_submission.csv');
  const [idColumn, setIdColumn] = useState<string>('id');
  const [featureColumns, setFeatureColumns] = useState<string>('text, context');
  const [targetColumn, setTargetColumn] = useState<string>('target');
  const [dataStrategy, setDataStrategy] = useState<string>('full_feature_concat');
  const [imbalanceStrategy, setImbalanceStrategy] = useState<string>('class_weighted');
  const [missingDataPolicy, setMissingDataPolicy] = useState<string>('fill_empty');

  // Tab 3: Validation Protocol
  const [splitStrategy, setSplitStrategy] = useState<string>('KFold');
  const [groupColumn, setGroupColumn] = useState<string>('');
  const [nFolds, setNFolds] = useState<number>(5);
  const [seed, setSeed] = useState<number>(42);
  const [foldFilePath, setFoldFilePath] = useState<string>('splits/folds.csv');

  // Tab 4: Competition Rules & Constraints
  const [allowLlm, setAllowLlm] = useState<boolean>(false);
  const [allowApi, setAllowApi] = useState<boolean>(false);
  const [allowExternalData, setAllowExternalData] = useState<boolean>(false);
  const [allowPretrainedModels, setAllowPretrainedModels] = useState<boolean>(true);
  const [maxParamsLimit, setMaxParamsLimit] = useState<string>('150000000');
  const [maxGpuHours, setMaxGpuHours] = useState<string>('12');
  const [dailySubmissionLimit, setDailySubmissionLimit] = useState<string>('5');
  const [deadline, setDeadline] = useState<string>('');
  const [teamPolicy, setTeamPolicy] = useState<string>('');
  const [reproducibilityRequired, setReproducibilityRequired] = useState<boolean>(true);

  // Tab 5: Submission & Audit Specification
  const [submissionFileName, setSubmissionFileName] = useState<string>('submission.zip');
  const [archiveFormat, setArchiveFormat] = useState<'zip' | 'csv' | 'tar.gz'>('zip');
  const [minDiffPct, setMinDiffPct] = useState<string>('2.0');
  const [maxKappa, setMaxKappa] = useState<string>('0.85');
  const [minDisagreement, setMinDisagreement] = useState<string>('5.0');

  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Active audit regime resolved adaptively from modality, task type, and metric
  const regime: AuditRegime = resolveAuditRegime(modality, type, primaryMetric);

  const getLocalizedRegimeName = (r: AuditRegime): string => {
    switch (r) {
      case 'classification':
        return t('regime_classification', REGIME_DISPLAY_NAMES.classification);
      case 'regression':
        return t('regime_regression', REGIME_DISPLAY_NAMES.regression);
      case 'ranking':
        return t('regime_ranking', REGIME_DISPLAY_NAMES.ranking);
      case 'generative_nlp':
        return t('regime_generative_nlp', REGIME_DISPLAY_NAMES.generative_nlp);
      default:
        return REGIME_DISPLAY_NAMES[r] || r;
    }
  };

  const handleResetRegimeDefaults = (targetRegime?: AuditRegime) => {
    const r = targetRegime || regime;
    const defaults = getRegimeDefaults(r);
    setMinDiffPct(defaults.minDiff);
    setMaxKappa(defaults.maxKappa);
    setMinDisagreement(defaults.minDisagreement);
  };

  const handleAddContextColChip = (chip: string) => {
    if (!featureColumns.trim()) {
      setFeatureColumns(chip);
    } else {
      const parts = featureColumns.split(',').map((p) => p.trim());
      if (!parts.includes(chip)) {
        setFeatureColumns(`${featureColumns.trim()}, ${chip}`);
      }
    }
  };

  // Fetch presets
  useEffect(() => {
    api
      .getTaskPresets()
      .then((res) => {
        if (res.presets) setPresets(res.presets);
      })
      .catch((err) => console.warn('Could not load presets:', err));
  }, []);

  // Initialize form when currentTask changes
  useEffect(() => {
    if (currentTask) {
      setPlatform(currentTask.platform || 'custom');
      setName(currentTask.name || currentTask.task_name || '');
      setType(currentTask.type || currentTask.task_type || '');
      setModality(currentTask.modality || 'text');
      setPrimaryMetric(currentTask.primary_metric ? currentTask.primary_metric.replace(/\s*\(CV\)/i, '') : 'Score');
      setDirection(currentTask.direction || 'maximize');
      setPrimaryBaseline(currentTask.primary_metric_baseline !== undefined ? String(currentTask.primary_metric_baseline) : '');
      setPrimaryTarget(currentTask.primary_metric_target !== undefined ? String(currentTask.primary_metric_target) : '');
      setMetrics(currentTask.metrics || []);
      setLanguages((currentTask.languages || []).join(', '));
      setDescription(currentTask.description || '');

      // Data Config
      const d = currentTask.data_config || {};
      setTrainPath(d.train_path || 'dataset/');
      setTestPath(d.test_path || 'dataset/test/');
      setSampleSubPath(d.sample_submission_path || 'submissions/sample_submission.csv');
      setSplitStrategy(d.split_strategy || 'KFold');
      setGroupColumn(d.group_column || '');
      setNFolds(d.n_folds || 5);
      setSeed(d.seed || 42);
      setIdColumn(d.id_column || 'id');
      setFeatureColumns((d as any).feature_columns || 'text, context');
      setTargetColumn(d.target_column || 'target');
      setMissingDataPolicy(d.missing_data_policy || 'drop_invalid_rows');

      // Rules
      const r = currentTask.competition_rules || {};
      setAllowLlm(!!r.allow_llm);
      setAllowApi(!!r.allow_api);
      setAllowExternalData(!!r.allow_external_data);
      setAllowPretrainedModels(r.allow_pretrained_models !== false);
      setMaxParamsLimit(r.max_params_limit ? String(r.max_params_limit) : '150000000');
      setMaxGpuHours(r.max_gpu_hours ? String(r.max_gpu_hours) : '12');
      setDailySubmissionLimit(r.daily_submission_limit ? String(r.daily_submission_limit) : '5');
      setDeadline(r.deadline || '');
      setTeamPolicy(r.team_policy || '');
      setReproducibilityRequired(r.reproducibility_required !== false);

      // Submission Spec
      const s = currentTask.submission_spec || {};
      setSubmissionFileName(s.file_name || 'submission.zip');
      setArchiveFormat(s.archive_format || 'zip');
      setMinDiffPct(s.min_diff_pct !== undefined ? String(s.min_diff_pct) : '2.0');
      setMaxKappa(s.max_kappa !== undefined ? String(s.max_kappa) : '0.85');
      setMinDisagreement(s.min_disagreement !== undefined ? String(s.min_disagreement) : '5.0');

      setError(null);
    }
  }, [currentTask, isOpen]);

  useEffect(() => {
    if (initialTab) setActiveTab(initialTab);
  }, [initialTab]);

  if (!isOpen) return null;
  const isCustomTaskPlugin = Boolean(currentTask?.task_plugin?.id && currentTask.task_plugin.id !== 'default');

  const handleApplyPreset = (preset: TaskPreset) => {
    setPlatform(preset.platform || 'kaggle');
    setName(preset.name);
    setType(preset.type);
    setModality(preset.modality || 'text');
    setPrimaryMetric(preset.primary_metric ? preset.primary_metric.replace(/\s*\(CV\)/i, '') : 'Score');
    setDirection(preset.direction);
    setPrimaryBaseline(preset.primary_metric_baseline !== undefined ? String(preset.primary_metric_baseline) : '');
    setPrimaryTarget(preset.primary_metric_target !== undefined ? String(preset.primary_metric_target) : '');
    setMetrics(preset.metrics || []);
    setLanguages((preset.languages || []).join(', '));
    setDescription(preset.description || '');

    if (preset.data_config) {
      setTrainPath(preset.data_config.train_path || 'dataset/');
      setTestPath(preset.data_config.test_path || 'dataset/test/');
      setSplitStrategy(preset.data_config.split_strategy || 'KFold');
      setGroupColumn(preset.data_config.group_column || '');
      setNFolds(preset.data_config.n_folds || 5);
      setSeed(preset.data_config.seed || 42);
      setIdColumn(preset.data_config.id_column || 'id');
      setTargetColumn(preset.data_config.target_column || 'target');
      if ((preset.data_config as any)?.feature_columns) {
        setFeatureColumns((preset.data_config as any).feature_columns);
      }
    }

    if (preset.competition_rules) {
      setAllowLlm(!!preset.competition_rules.allow_llm);
      setAllowApi(!!preset.competition_rules.allow_api);
      setAllowExternalData(!!preset.competition_rules.allow_external_data);
      setAllowPretrainedModels(preset.competition_rules.allow_pretrained_models !== false);
      setMaxParamsLimit(preset.competition_rules.max_params_limit ? String(preset.competition_rules.max_params_limit) : '');
      setDailySubmissionLimit(preset.competition_rules.daily_submission_limit ? String(preset.competition_rules.daily_submission_limit) : '5');
      setDeadline(preset.competition_rules.deadline || '');
    }

    if (preset.submission_spec) {
      setSubmissionFileName(preset.submission_spec.file_name || 'submission.csv');
      setArchiveFormat(preset.submission_spec.archive_format || 'csv');
      setMinDiffPct(preset.submission_spec.min_diff_pct !== undefined ? String(preset.submission_spec.min_diff_pct) : '2.0');
      const targetRegime = resolveAuditRegime(preset.modality || 'text', preset.type, preset.primary_metric || 'Score');
      const defaults = getRegimeDefaults(targetRegime);
      setMaxKappa(preset.submission_spec.max_kappa !== undefined ? String(preset.submission_spec.max_kappa) : defaults.maxKappa);
      setMinDisagreement(preset.submission_spec.min_disagreement !== undefined ? String(preset.submission_spec.min_disagreement) : defaults.minDisagreement);
    }
  };

  const handleAddMetric = () => {
    const nextIdx = metrics.length + 1;
    setMetrics([
      ...metrics,
      {
        id: `metric_${nextIdx}`,
        name: `Subtask / Metric ${nextIdx}`,
        metric: primaryMetric.split(' ')[0] || 'Score',
        type: 'macro_f1',
        direction: direction,
        weight: 1.0,
        baseline_score: 0.0,
        target_score: 0.0,
        description: '',
      },
    ]);
    setExpandedMetricIndices({ ...expandedMetricIndices, [metrics.length]: true });
  };

  const handleUpdateMetric = (index: number, updated: Partial<TaskMetricDef>) => {
    const newMetrics = [...metrics];
    newMetrics[index] = { ...newMetrics[index], ...updated };
    setMetrics(newMetrics);
  };

  const handleRemoveMetric = (index: number) => {
    setMetrics(metrics.filter((_, i) => i !== index));
  };

  const toggleMetricExpand = (index: number) => {
    setExpandedMetricIndices((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Please provide a task or competition name.');
      return;
    }

    setSaving(true);
    setError(null);

    const langList = languages
      .split(',')
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    const parseSafeFloat = (v: string, def?: number): number | undefined => {
      if (!v || v.trim() === '') return def;
      const parsed = parseFloat(v);
      return isNaN(parsed) ? def : parsed;
    };

    const updatedTask: TaskInfo = {
      platform,
      task_name: name.trim(),
      name: name.trim(),
      task_type: type.trim() || 'Supervised Learning',
      type: type.trim() || 'Supervised Learning',
      modality,
      primary_metric: primaryMetric.trim() || 'Score',
      direction,
      primary_metric_baseline: parseSafeFloat(primaryBaseline),
      primary_metric_target: parseSafeFloat(primaryTarget),
      metrics,
      languages: langList,
      description: description.trim(),
      data_config: {
        train_path: trainPath.trim(),
        test_path: testPath.trim(),
        sample_submission_path: sampleSubPath.trim(),
        split_strategy: splitStrategy,
        group_column: groupColumn.trim() || undefined,
        n_folds: Number(nFolds) || 5,
        seed: Number(seed) || 42,
        id_column: idColumn.trim() || 'id',
        feature_columns: featureColumns.trim(),
        target_column: targetColumn.trim() || 'target',
        missing_data_policy: missingDataPolicy.trim(),
      } as DataConfig & { feature_columns?: string },
      competition_rules: {
        allow_llm: allowLlm,
        allow_api: allowApi,
        allow_external_data: allowExternalData,
        allow_pretrained_models: allowPretrainedModels,
        max_params_limit: maxParamsLimit ? parseInt(maxParamsLimit, 10) : undefined,
        max_gpu_hours: parseSafeFloat(maxGpuHours),
        daily_submission_limit: dailySubmissionLimit ? parseInt(dailySubmissionLimit, 10) : 5,
        deadline: deadline.trim() || undefined,
        team_policy: teamPolicy.trim() || undefined,
        reproducibility_required: reproducibilityRequired,
      },
      submission_spec: {
        file_name: submissionFileName.trim(),
        archive_format: archiveFormat,
        ...(isCustomTaskPlugin ? {} : {
          min_diff_pct: parseSafeFloat(minDiffPct, 2.0) ?? 2.0,
          max_kappa: parseSafeFloat(maxKappa, 0.85) ?? 0.85,
          min_disagreement: parseSafeFloat(minDisagreement, 5.0) ?? 5.0,
        }),
      },
    };

    try {
      await api.updateTaskInfo(updatedTask);
      await onSave(updatedTask);
      onClose();
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to save competition setup.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-4xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh] animate-in fade-in zoom-in-95 duration-150">
        {/* Top Header */}
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/60">
          <div className="flex items-center space-x-2.5">
            <div className="w-7 h-7 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-300">
              <Sliders className="w-3.5 h-3.5" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <span>{t('modal_task_title', 'Competition & Benchmark Setup')}</span>
                {name && <span className="text-xs text-zinc-400 font-normal">({name})</span>}
              </h2>
              <p className="text-[11px] text-zinc-500">
                {t('modal_task_desc', 'Configure task objectives, cross-validation data splits, competition rules, and audit gates')}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-300 p-1 rounded-md hover:bg-zinc-800 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* 5-Tab Navigation */}
        <div className="px-5 pt-2 pb-0 border-b border-zinc-800 bg-zinc-950/40 flex flex-wrap gap-1">
          <button
            type="button"
            onClick={() => setActiveTab('task')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'task'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Award className="w-3.5 h-3.5" />
            <span>{t('modal_tab_obj', '1. Objective & Metrics')}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('data')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'data'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>{t('modal_tab_data', '2. Data Schema & Strategy')}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('cv')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'cv'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>{t('tab_validation_protocol', '3. Validation Split')}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('rules')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'rules'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Scale className="w-3.5 h-3.5" />
            <span>{t('modal_tab_rules', '4. Rules & Constraints')}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('submission')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'submission'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <FileArchive className="w-3.5 h-3.5" />
            <span>{t('modal_tab_sub', '5. Submission & Audit')}</span>
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSave} className="flex-1 overflow-y-auto p-5 space-y-4">
          {error && (
            <div className="p-3 bg-red-950/40 border border-red-800/80 rounded-lg text-xs text-red-300">
              {error}
            </div>
          )}

          {/* TAB 1: Task & Objective */}
          {activeTab === 'task' && (
            <div className="space-y-4">
              {/* Presets Bar */}
              {presets.length > 0 && (
                <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                  <label className="block text-[11px] font-semibold text-zinc-300 mb-1.5 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    <span>{t('modal_task_presets_label', 'Quick Competition Presets (1-Click Fill All Tabs)')}</span>
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    {presets.map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => handleApplyPreset(p)}
                        className="px-2.5 py-1 text-xs bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-zinc-100 rounded border border-zinc-800 transition"
                      >
                        {p.preset_name}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Platform & Modality */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4" title={t('modal_task_platform_modality', 'Competition Platform / Problem Modality')}>
                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_platform_label', 'Competition Platform')}</label>
                  <select
                    value={platform}
                    onChange={(e) => setPlatform(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  >
                    {PLATFORMS.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({p.badge})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_modality_label', 'Problem Modality')}</label>
                  <select
                    value={modality}
                    onChange={(e) => setModality(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  >
                    {MODALITIES.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Task Name & Type */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_task_name_label', 'Task / Competition Name')}</label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={t('taskcfg_task_name_ph', 'e.g. Kaggle Competition, House Prices, Titanic')}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_task_type_label', 'Task Category / Type')}</label>
                  <input
                    type="text"
                    value={type}
                    onChange={(e) => setType(e.target.value)}
                    placeholder={t('taskcfg_task_type_ph', 'e.g. Multi-Task Text Classification, Tabular Regression')}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>
              </div>

              {/* Primary Metric & Goal Scores */}
              <div className="p-3.5 bg-zinc-950/70 border border-zinc-800/80 rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-semibold text-zinc-200">
                    {t('modal_task_primary_metric_title', 'Primary Benchmark Metric & Target')}
                  </div>
                  <span className="text-[10px] text-zinc-500 font-mono">
                    Standard ML Catalog + Custom Support
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1" title={t('taskcfg_primary_metric_label', 'Primary Metric')}>
                      {t('modal_task_pick_metric', 'Pick Standard ML Metric')}
                    </label>
                    <select
                      value={ML_METRICS_CATALOG.some((m) => m.id === primaryMetric) ? primaryMetric : '__custom__'}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val !== '__custom__') {
                          setPrimaryMetric(val);
                          setDirection(getSuggestedDirection(val));
                        }
                      }}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                    >
                      <optgroup label="Classification">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Classification').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Regression">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Regression').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Ranking & RecSys">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Ranking').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="NLP & Generation">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'NLP').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Computer Vision">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Vision').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Clustering">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Clustering').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </optgroup>
                      <option value="__custom__">{t('taskcfg_custom_metric_free', 'Custom / Type Freely...')}</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      {t('taskcfg_active_metric_name', 'Active Metric Name')}
                    </label>
                    <input
                      type="text"
                      required
                      value={primaryMetric}
                      onChange={(e) => {
                        const val = e.target.value;
                        setPrimaryMetric(val);
                        setDirection(getSuggestedDirection(val));
                      }}
                      placeholder="e.g. Macro-F1, ROC-AUC, RMSE, NDCG@10"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_direction_label', 'Optimization Direction')}</label>
                    <select
                      value={direction}
                      onChange={(e) => setDirection(e.target.value as 'maximize' | 'minimize')}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                    >
                      <option value="maximize">{t('taskcfg_opt_max', 'Maximize (F1, AUC, Acc, R²)')}</option>
                      <option value="minimize">{t('taskcfg_opt_min', 'Minimize (RMSE, Loss, Error)')}</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_baseline_floor_label', 'Baseline Score Floor')}</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={primaryBaseline}
                      onChange={(e) => setPrimaryBaseline(e.target.value)}
                      placeholder="e.g. 0.6459"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_target_goal_label', 'Target Goal Score')}</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={primaryTarget}
                      onChange={(e) => setPrimaryTarget(e.target.value)}
                      placeholder="e.g. 0.8500"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-emerald-400 focus:outline-none focus:border-zinc-700"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_languages_label', 'Languages (Comma separated)')}</label>
                    <input
                      type="text"
                      value={languages}
                      onChange={(e) => setLanguages(e.target.value)}
                      placeholder="e.g. EN, IT, NL (optional)"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                </div>
              </div>

              {/* Subtasks / Criteria List */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-zinc-200">
                    {t('modal_task_subtasks_title', 'Subtasks & Auxiliary Metrics')} ({metrics.length})
                  </label>
                  <button
                    type="button"
                    onClick={handleAddMetric}
                    className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 px-2 py-0.5 rounded transition"
                  >
                    <Plus className="w-3 h-3" />
                    <span>{t('taskcfg_btn_add_subtask', 'Add Subtask')}</span>
                  </button>
                </div>

                {metrics.map((m, idx) => (
                  <div key={idx} className="p-3 bg-zinc-950 border border-zinc-800/80 rounded-lg space-y-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={m.name}
                        onChange={(e) => handleUpdateMetric(idx, { name: e.target.value })}
                        placeholder={t('taskcfg_subtask_name_ph', 'Subtask Name')}
                        className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200"
                      />
                      <input
                        type="text"
                        value={m.id}
                        onChange={(e) => handleUpdateMetric(idx, { id: e.target.value })}
                        placeholder="metric_id"
                        className="w-28 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-400"
                      />
                      <button
                        type="button"
                        onClick={() => toggleMetricExpand(idx)}
                        className="px-2 py-1 text-[11px] bg-zinc-900 text-zinc-400 hover:text-zinc-200 rounded border border-zinc-800"
                      >
                        {expandedMetricIndices[idx] ? 'Less ▲' : 'Details ▼'}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleRemoveMetric(idx)}
                        className="p-1 text-zinc-500 hover:text-red-400 rounded"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {expandedMetricIndices[idx] && (
                      <div className="pt-2 border-t border-zinc-800 grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
                        <div>
                          <label className="block text-zinc-500 mb-0.5">{t('taskcfg_eval_metric_label', 'Evaluation Metric')}</label>
                          <div className="space-y-1">
                            <select
                              value={ML_METRICS_CATALOG.some((c) => c.id === m.metric) ? m.metric : '__custom__'}
                              onChange={(e) => {
                                const val = e.target.value;
                                if (val !== '__custom__') {
                                  handleUpdateMetric(idx, {
                                    metric: val,
                                    direction: getSuggestedDirection(val),
                                  });
                                }
                              }}
                              className="w-full bg-zinc-900 border border-zinc-800 rounded px-1.5 py-0.5 text-[11px] text-zinc-200"
                            >
                              <option value="__custom__">{t('taskcfg_custom_metric_below', 'Custom / Type Below...')}</option>
                              {ML_METRICS_CATALOG.map((c) => (
                                <option key={c.id} value={c.id}>
                                  {c.id} ({c.category})
                                </option>
                              ))}
                            </select>
                            <input
                              type="text"
                              value={m.metric}
                              onChange={(e) => {
                                const val = e.target.value;
                                handleUpdateMetric(idx, {
                                  metric: val,
                                  direction: getSuggestedDirection(val),
                                });
                              }}
                              placeholder={t('taskcfg_metric_name_ph', 'Metric Name')}
                              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-0.5 text-zinc-300 font-mono text-[11px]"
                            />
                          </div>
                        </div>
                        <div>
                          <label className="block text-zinc-500 mb-0.5">{t('taskcfg_baseline_score_label', 'Baseline Score')}</label>
                          <input
                            type="number"
                            step="0.0001"
                            value={m.baseline_score ?? ''}
                            onChange={(e) => handleUpdateMetric(idx, { baseline_score: parseFloat(e.target.value) || 0 })}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 font-mono text-zinc-300"
                          />
                        </div>
                        <div>
                          <label className="block text-zinc-500 mb-0.5">{t('taskcfg_target_goal_label_sub', 'Target Goal')}</label>
                          <input
                            type="number"
                            step="0.0001"
                            value={m.target_score ?? ''}
                            onChange={(e) => handleUpdateMetric(idx, { target_score: parseFloat(e.target.value) || 0 })}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 font-mono text-emerald-400"
                          />
                        </div>
                        <div>
                          <label className="block text-zinc-500 mb-0.5">{t('taskcfg_weight_label', 'Weight (0-1.0)')}</label>
                          <input
                            type="number"
                            step="0.01"
                            value={m.weight ?? 1.0}
                            onChange={(e) => handleUpdateMetric(idx, { weight: parseFloat(e.target.value) || 0 })}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-zinc-300"
                          />
                        </div>
                        <div className="col-span-2 md:col-span-4">
                          <label className="block text-zinc-500 mb-0.5">{t('taskcfg_desc_rules_label', 'Description & Special Rules')}</label>
                          <input
                            type="text"
                            value={m.description ?? ''}
                            onChange={(e) => handleUpdateMetric(idx, { description: e.target.value })}
                            placeholder={t('taskcfg_desc_rules_ph', 'e.g. Evaluated strictly on hateful comments (oracle_hate)')}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-zinc-300"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: Data Schema & Strategy */}
          {activeTab === 'data' && (
            <div className="space-y-4">
              <div className="p-3.5 bg-zinc-950 border border-zinc-800 rounded-lg space-y-3">
                <div className="text-xs font-semibold text-zinc-200">{t('taskcfg_dataset_dirs_title', 'Dataset Directories')}</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_train_path_label', 'Training Data Path')}</label>
                    <input
                      type="text"
                      value={trainPath}
                      onChange={(e) => setTrainPath(e.target.value)}
                      placeholder="e.g. dataset/train.csv or dataset/"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_test_path_label', 'Test Data Path')}</label>
                    <input
                      type="text"
                      value={testPath}
                      onChange={(e) => setTestPath(e.target.value)}
                      placeholder="e.g. dataset/test/"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>
                </div>
              </div>

              <div className="p-3.5 bg-zinc-950 border border-zinc-800 rounded-lg space-y-3">
                <div className="text-xs font-semibold text-zinc-200">{t('taskcfg_column_schema_title', 'Column Schema')}</div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_id_col_label', 'ID Column Name')}</label>
                    <input
                      type="text"
                      value={idColumn}
                      onChange={(e) => setIdColumn(e.target.value)}
                      placeholder="e.g. id, row_id, item_id"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="block text-xs font-medium text-zinc-400">{t('taskcfg_context_cols_label', 'Input / Context Columns')}</label>
                      <span className="text-[10px] text-zinc-500">{t('taskcfg_context_cols_hint', 'Contextual input columns for feature extraction or prompt engineering')}</span>
                    </div>
                    <input
                      type="text"
                      value={featureColumns}
                      onChange={(e) => setFeatureColumns(e.target.value)}
                      placeholder={getContextColPlaceholder(modality, t)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-100"
                    />
                    <div className="flex items-center flex-wrap gap-1 mt-1.5">
                      <span className="text-[10px] text-zinc-500 mr-1">{t('taskcfg_context_preset_label', 'Suggested Columns:')}</span>
                      {(CONTEXT_COL_SUGGESTIONS[modality] || CONTEXT_COL_SUGGESTIONS.text).map((chip) => (
                        <button
                          key={chip}
                          type="button"
                          onClick={() => handleAddContextColChip(chip)}
                          className="px-1.5 py-0.5 text-[10px] font-mono bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 rounded border border-zinc-800 hover:border-zinc-700 transition"
                        >
                          +{chip}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_target_col_label', 'Target Column(s)')}</label>
                    <input
                      type="text"
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      placeholder="e.g. target, label, class"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>
                </div>
              </div>

              <div className="p-3.5 bg-zinc-950 border border-zinc-800 rounded-lg space-y-3">
                <div className="text-xs font-semibold text-zinc-200">{t('taskcfg_data_processing_title', 'Data Processing & Representation Strategy')}</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_feature_strategy_label', 'Feature / Context Strategy')}</label>
                    <select
                      value={dataStrategy}
                      onChange={(e) => setDataStrategy(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100"
                    >
                      {DATA_STRATEGIES.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_imbalance_strategy_label', 'Label Filtering & Imbalance Strategy')}</label>
                    <select
                      value={imbalanceStrategy}
                      onChange={(e) => setImbalanceStrategy(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100"
                    >
                      {IMBALANCE_STRATEGIES.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-xs font-medium text-zinc-400">{t('taskcfg_missing_data_label', 'Missing Data Handling Policy')}</label>
                    <span className="text-[10px] text-zinc-500">{t('taskcfg_missing_data_hint', 'Contextual resolution strategy for null or missing values')}</span>
                  </div>
                  <input
                    type="text"
                    value={missingDataPolicy}
                    onChange={(e) => setMissingDataPolicy(e.target.value)}
                    placeholder={t('taskcfg_missing_data_ph', 'e.g. fill_empty_context, drop_invalid_rows')}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 mb-2"
                  />
                  <div className="flex items-center flex-wrap gap-1.5">
                    {(MISSING_DATA_PRESETS[modality] || MISSING_DATA_PRESETS.tabular).map((preset) => {
                      const isSelected = missingDataPolicy === preset.id;
                      return (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => setMissingDataPolicy(preset.id)}
                          className={`px-2 py-1 text-[11px] rounded border transition ${
                            isSelected
                              ? 'bg-emerald-950/70 text-emerald-300 border-emerald-700 font-medium'
                              : 'bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border-zinc-800'
                          }`}
                        >
                          {t(preset.labelKey, preset.defaultLabel)}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: Validation Split */}
          {activeTab === 'cv' && (
            <div className="space-y-4">
              <div className="p-3.5 bg-zinc-950 border border-zinc-800 rounded-lg space-y-3">
                <div className="text-xs font-semibold text-zinc-200">{t('tab_validation_protocol', 'Validation Split Protocol')}</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_split_strategy_label', 'Validation Split Strategy')}</label>
                    <select
                      value={splitStrategy}
                      onChange={(e) => setSplitStrategy(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100"
                    >
                      {SPLIT_STRATEGIES.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">
                      {t('taskcfg_group_col_label', 'Group Identifier Column')}
                    </label>
                    <input
                      type="text"
                      value={groupColumn}
                      onChange={(e) => setGroupColumn(e.target.value)}
                      placeholder="e.g. group_id, user_id, entity_id"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_n_folds_label', 'Number of Folds (K)')}</label>
                    <input
                      type="number"
                      min={2}
                      max={20}
                      value={nFolds}
                      onChange={(e) => setNFolds(parseInt(e.target.value, 10) || 5)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_seed_label', 'Random Seed')}</label>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_fold_file_label', 'Precomputed Fold File')}</label>
                    <input
                      type="text"
                      value={foldFilePath}
                      onChange={(e) => setFoldFilePath(e.target.value)}
                      placeholder="splits/fold_assignments.tsv"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-100"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: Competition Rules & Constraints */}
          {activeTab === 'rules' && (
            <div className="space-y-4">
              <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 space-y-3">
                <div className="text-xs font-semibold text-zinc-200">{t('taskcfg_permitted_tech_title', 'Permitted Technologies & External Sources')}</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowLlm}
                      onChange={(e) => setAllowLlm(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>{t('taskcfg_allow_llm', 'Allow LLM / Generative Models & Prompting')}</span>
                  </label>

                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowApi}
                      onChange={(e) => setAllowApi(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>{t('taskcfg_allow_api', 'Allow Proprietary Closed APIs (OpenAI / Anthropic)')}</span>
                  </label>

                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowExternalData}
                      onChange={(e) => setAllowExternalData(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>{t('taskcfg_allow_ext_label', 'Allow External Data')}</span>
                  </label>

                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowPretrainedModels}
                      onChange={(e) => setAllowPretrainedModels(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>{t('taskcfg_allow_pretrained', 'Allow Open Pretrained Weights (HuggingFace / timm)')}</span>
                  </label>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-[11px] text-zinc-400">{t('taskcfg_max_params_label', 'Max Parameter Limit')}</label>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-300 border border-zinc-800">
                      {formatParamScale(maxParamsLimit, t('taskcfg_max_params_unlimited', 'Unlimited parameters'))}
                    </span>
                  </div>
                  <input
                    type="number"
                    value={maxParamsLimit}
                    onChange={(e) => setMaxParamsLimit(e.target.value)}
                    placeholder="e.g. 150000000"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                  <div className="flex items-center flex-wrap gap-1 mt-1.5">
                    <span className="text-[10px] text-zinc-500 mr-1">{t('taskcfg_params_preset_label', 'Architectural Parameter Presets:')}</span>
                    {PARAM_PRESETS.map((preset) => {
                      const isSelected = preset.value === '' ? (!maxParamsLimit || maxParamsLimit === '0') : maxParamsLimit === preset.value;
                      return (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => setMaxParamsLimit(preset.value)}
                          className={`px-1.5 py-0.5 text-[10px] rounded border transition ${
                            isSelected
                              ? 'bg-indigo-950/80 text-indigo-300 border-indigo-700 font-medium'
                              : 'bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border-zinc-800'
                          }`}
                        >
                          {t(preset.labelKey, preset.defaultLabel)}
                        </button>
                      );
                    })}
                  </div>
                  <span className="text-[10px] text-zinc-500 block mt-1">{t('taskcfg_gpu_limit_hint', 'Limits GPU memory footprint')}</span>
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_max_gpu_hours_label', 'Max GPU Runtime (Hours)')}</label>
                  <input
                    type="number"
                    value={maxGpuHours}
                    onChange={(e) => setMaxGpuHours(e.target.value)}
                    placeholder="e.g. 9 or 12"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                  <span className="text-[10px] text-zinc-500">{t('taskcfg_gpu_hours_hint', 'Modal/Kaggle kernel cutoff')}</span>
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_daily_quota_label', 'Daily Submission Quota')}</label>
                  <input
                    type="number"
                    value={dailySubmissionLimit}
                    onChange={(e) => setDailySubmissionLimit(e.target.value)}
                    placeholder="e.g. 5"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                  <span className="text-[10px] text-zinc-500">{t('taskcfg_daily_quota_hint', 'Allowed submissions per day')}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_deadline_label', 'Competition Deadline (UTC)')}</label>
                  <input
                    type="text"
                    value={deadline}
                    onChange={(e) => setDeadline(e.target.value)}
                    placeholder="e.g. 2027-01-15T23:59:59Z"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">{t('taskcfg_team_policy_label', 'Team & Collaboration Policy')}</label>
                  <input
                    type="text"
                    value={teamPolicy}
                    onChange={(e) => setTeamPolicy(e.target.value)}
                    placeholder={t('taskcfg_team_policy_ph', 'e.g. Max 4 members, no private sharing')}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-100"
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: Submission & Audit Standards */}
          {activeTab === 'submission' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_sub_filename_label', 'Official Submission File Name')}</label>
                  <input
                    type="text"
                    value={submissionFileName}
                    onChange={(e) => setSubmissionFileName(e.target.value)}
                    placeholder="e.g. submission.csv or submission.zip"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">{t('taskcfg_archive_format_label', 'Archive Packaging Format')}</label>
                  <select
                    value={archiveFormat}
                    onChange={(e) => setArchiveFormat(e.target.value as 'zip' | 'csv' | 'tar.gz')}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  >
                    <option value="zip">{t('taskcfg_fmt_zip', 'ZIP Compressed Archive (.zip)')}</option>
                    <option value="csv">{t('taskcfg_fmt_csv', 'Raw Delimited CSV (.csv)')}</option>
                    <option value="tar.gz">{t('taskcfg_fmt_tar', 'Tarball (.tar.gz)')}</option>
                  </select>
                </div>
              </div>

              {isCustomTaskPlugin ? (
                <div className="p-3.5 bg-zinc-950 rounded-lg border border-zinc-800 text-xs text-zinc-300">
                  The selected task plugin owns submission construction and validation. Its builder is {currentTask?.task_plugin?.capabilities?.submission_builder ? 'available' : 'not implemented'}, and its validator is {currentTask?.task_plugin?.capabilities?.submission_validator ? 'available' : 'not implemented'}. Core audit thresholds are not applied to this plugin.
                </div>
              ) : (
              <div className="p-3.5 bg-zinc-950 rounded-lg border border-zinc-800 space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800/80 pb-3">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                    <div>
                      <div className="text-xs font-semibold text-zinc-200">
                        {regime === 'regression' && t('taskcfg_audit_title_regression', 'Audit Gate Parameters (Continuous Regression)')}
                        {regime === 'ranking' && t('taskcfg_audit_title_ranking', 'Audit Gate Parameters (Ranking & Recommendation)')}
                        {regime === 'generative_nlp' && t('taskcfg_audit_title_generative', 'Audit Gate Parameters (NLP / Generative LLM)')}
                        {regime === 'classification' && t('taskcfg_audit_title_classification', 'Audit Gate Parameters (Classification)')}
                      </div>
                      <div className="text-[11px] text-zinc-400">
                        {regime === 'regression' && t('taskcfg_audit_desc_regression', 'Enforces continuous prediction variance, collinearity rejection, and relative epsilon tolerance.')}
                        {regime === 'ranking' && t('taskcfg_audit_desc_ranking', 'Audits permutation concordances, rank inversion rates, and listwise order deltas.')}
                        {regime === 'generative_nlp' && t('taskcfg_audit_desc_generative', 'Audits embedding cosine diversity, lexical edit distance, and output hash integrity.')}
                        {regime === 'classification' && t('taskcfg_audit_desc_classification', 'Enforces discrete label diversity and chance-adjusted inter-model agreement.')}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/70 text-emerald-300 border border-emerald-800/60 shadow-sm"
                      title={t('taskcfg_audit_adaptive_tooltip', 'Audit Gate metrics and thresholds automatically adapt to the selected Problem Modality and Task Category.')}
                    >
                      {t('taskcfg_audit_regime_badge', `Active Audit Regime: ${getLocalizedRegimeName(regime)}`, { regime: getLocalizedRegimeName(regime) })}
                    </span>

                    <button
                      type="button"
                      onClick={() => handleResetRegimeDefaults(regime)}
                      className="inline-flex items-center gap-1 px-2 py-1 text-[11px] text-zinc-300 bg-zinc-900 hover:bg-zinc-800 hover:text-white rounded border border-zinc-800 hover:border-zinc-700 transition"
                      title={t('taskcfg_audit_reset_defaults', 'Reset to Regime Defaults')}
                    >
                      <RotateCcw className="w-3 h-3 text-zinc-400" />
                      <span>{t('taskcfg_audit_reset_defaults', 'Reset to Regime Defaults')}</span>
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      {regime === 'regression' && t('audit_label_min_diff_regression', 'Min Meaningful Effect Delta (% rows)')}
                      {regime === 'ranking' && t('audit_label_min_diff_ranking', 'Min Top-K Rank Delta (% rows)')}
                      {regime === 'generative_nlp' && t('audit_label_min_diff_nlp', 'Generation Checksum & Output Delta (% rows)')}
                      {regime === 'classification' && t('audit_label_min_diff_classification', 'Min Class Delta (% rows)')}
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={minDiffPct}
                      onChange={(e) => setMinDiffPct(e.target.value)}
                      placeholder="2.0"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                    />
                    <span className="text-[10px] text-zinc-500 block mt-1">
                      {regime === 'regression' && t('audit_hint_min_diff_regression', 'Min % rows where |y_cand - y_base| exceeds relative epsilon tolerance (|Δy|/σy ≥ ε)')}
                      {regime === 'ranking' && t('audit_hint_min_diff_ranking', 'Min % queries with reordered rankings vs baseline')}
                      {regime === 'generative_nlp' && t('audit_hint_min_diff_nlp', 'Rejects identical prompt replays and verifies generation checksum divergence (≥ 2.0%)')}
                      {regime === 'classification' && t('audit_hint_min_diff_classification', 'Minimum % changed prediction classes vs parent baseline (MMES floor 2.0%)')}
                    </span>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      {regime === 'regression' && t('audit_label_max_corr_regression', 'Max Ensemble Pearson/Spearman Corr (r ≤ 0.90)')}
                      {regime === 'ranking' && t('audit_label_max_corr_ranking', 'Max Rank Correlation (Kendall’s Tau ≤ 0.85)')}
                      {regime === 'generative_nlp' && t('audit_label_max_corr_nlp', 'Max Semantic Embedding Similarity (cos θ ≤ 0.92)')}
                      {regime === 'classification' && t('audit_label_max_corr_classification', 'Max Ensemble Correlation (Cohen’s Kappa)')}
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={maxKappa}
                      onChange={(e) => setMaxKappa(e.target.value)}
                      placeholder={regime === 'regression' ? '0.90' : regime === 'generative_nlp' ? '0.92' : '0.85'}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                    />
                    <span className="text-[10px] text-zinc-500 block mt-1">
                      {regime === 'regression' && t('audit_hint_max_corr_regression', 'Ceiling r ≤ 0.90 ensures continuous prediction vectors are genuinely diverse')}
                      {regime === 'ranking' && t('audit_hint_max_corr_ranking', 'Ceiling Kendall’s Tau ≤ 0.85 ensures diverse ranking orders across models')}
                      {regime === 'generative_nlp' && t('audit_hint_max_corr_nlp', 'Ceiling cos θ ≤ 0.92 prevents blending prompts with redundant output distributions')}
                      {regime === 'classification' && t('audit_hint_max_corr_classification', 'Ceiling 0.85 blocks pseudo-ensembles of near-identical classifiers')}
                    </span>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      {regime === 'regression' && t('audit_label_disagree_regression', 'Relative Epsilon Tolerance & Value Disagreement')}
                      {regime === 'ranking' && t('audit_label_disagree_ranking', 'Min Rank Inversion Rate (%)')}
                      {regime === 'generative_nlp' && t('audit_label_disagree_nlp', 'Min Token / Embedding Diversity Rate (%)')}
                      {regime === 'classification' && t('audit_label_disagree_classification', 'Min Class Disagreement Rate (%)')}
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={minDisagreement}
                      onChange={(e) => setMinDisagreement(e.target.value)}
                      placeholder="5.0"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                    />
                    <span className="text-[10px] text-zinc-500 block mt-1">
                      {regime === 'regression' && t('audit_hint_disagree_regression', 'Min % samples where |y1 - y2| exceeds relative error threshold ε (relative to σy)')}
                      {regime === 'ranking' && t('audit_hint_disagree_ranking', 'Floor 5.0% pairwise rank inversions guarantees complementary ranking lists')}
                      {regime === 'generative_nlp' && t('audit_hint_disagree_nlp', 'Floor 5.0% semantic divergence enforces distinct reasoning paths or generation phrasing')}
                      {regime === 'classification' && t('audit_hint_disagree_classification', 'Floor 5.0% enforces orthogonal decision boundaries between ensemble members')}
                    </span>
                  </div>
                </div>

                {/* Mathematical Epistemic Grounding Formula Callout */}
                <div className="mt-2 pt-2 border-t border-zinc-800/60 flex items-center justify-between text-[10px] font-mono text-zinc-500">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-zinc-400 font-semibold">{t('taskcfg_audit_adaptive_badge', 'Modality-Adaptive Gate')}:</span>
                    {regime === 'classification' && (
                      <span>Σ 𝕀(ŷ_cand ≠ ŷ_base) / N ≥ {minDiffPct || '2.0'}% • Kappa ≤ {maxKappa || '0.85'} • Disagreement ≥ {minDisagreement || '5.0'}%</span>
                    )}
                    {regime === 'regression' && (
                      <span>|Δy| / σy ≥ ε ({minDiffPct || '2.0'}% rows) • Pearson/Spearman r ≤ {maxKappa || '0.90'} • |Δ| &gt; ε ({minDisagreement || '5.0'}%)</span>
                    )}
                    {regime === 'ranking' && (
                      <span>Top-K Inversion Rate ≥ {minDiffPct || '2.0'}% • Kendall’s Tau ≤ {maxKappa || '0.85'} • Disagreement ≥ {minDisagreement || '5.0'}%</span>
                    )}
                    {regime === 'generative_nlp' && (
                      <span>Output Checksum &amp; Token Delta ≥ {minDiffPct || '2.0'}% • Cosine Sim ≤ {maxKappa || '0.92'} • Diversity ≥ {minDisagreement || '5.0'}%</span>
                    )}
                  </div>
                </div>
              </div>
              )}
            </div>
          )}

          {/* Footer Action Buttons */}
          <div className="pt-3 border-t border-zinc-800 flex items-center justify-between">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition"
            >
              {t('btn_cancel', 'Cancel')}
            </button>

            <button
              type="submit"
              disabled={saving}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-semibold shadow transition disabled:opacity-50"
            >
              <Check className="w-3.5 h-3.5" />
              <span>{saving ? t('status_saving', 'Saving...') : t('btn_save_task_config', 'Save & Sync Agent Skill')}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
