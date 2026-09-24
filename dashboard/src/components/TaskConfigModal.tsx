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
} from 'lucide-react';
import { TaskInfo, TaskMetricDef, TaskPreset, DataConfig, CompetitionRules, SubmissionSpec } from '../types/experiment';
import { api } from '../services/api';
import { ML_METRICS_CATALOG, getMetricById, getSuggestedDirection } from '../constants/metrics';

interface TaskConfigModalProps {
  isOpen: boolean;
  currentTask: TaskInfo | null;
  onClose: () => void;
  onSave: (updatedTask: TaskInfo) => Promise<void>;
  initialTab?: 'task' | 'data' | 'rules' | 'submission';
}

const PLATFORMS = [
  { id: 'codabench', name: 'Codabench', badge: 'SemEval / Academic' },
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

const SPLIT_STRATEGIES = [
  { id: 'GroupKFold', name: 'GroupKFold', desc: 'Prevents leakage by keeping groups (e.g. video_id, patient_id) strictly in one fold' },
  { id: 'StratifiedKFold', name: 'StratifiedKFold', desc: 'Preserves balanced percentage of samples for each target class' },
  { id: 'KFold', name: 'Standard KFold', desc: 'Standard non-stratified random k-fold cross-validation' },
  { id: 'TimeSeriesSplit', name: 'TimeSeriesSplit', desc: 'Strictly forward-rolling temporal validation without future leakage' },
];

export const TaskConfigModal: React.FC<TaskConfigModalProps> = ({
  isOpen,
  currentTask,
  onClose,
  onSave,
  initialTab = 'task',
}) => {
  const [activeTab, setActiveTab] = useState<'task' | 'data' | 'rules' | 'submission'>(initialTab);
  const [presets, setPresets] = useState<TaskPreset[]>([]);

  // Tab 1: Task & Objective
  const [platform, setPlatform] = useState<string>('codabench');
  const [name, setName] = useState<string>('');
  const [type, setType] = useState<string>('');
  const [modality, setModality] = useState<string>('text');
  const [primaryMetric, setPrimaryMetric] = useState<string>('Macro-F1 (CV)');
  const [direction, setDirection] = useState<'maximize' | 'minimize'>('maximize');
  const [primaryBaseline, setPrimaryBaseline] = useState<string>('');
  const [primaryTarget, setPrimaryTarget] = useState<string>('');
  const [metrics, setMetrics] = useState<TaskMetricDef[]>([]);
  const [expandedMetricIndices, setExpandedMetricIndices] = useState<Record<number, boolean>>({});
  const [languages, setLanguages] = useState<string>('');
  const [description, setDescription] = useState<string>('');

  // Tab 2: Data & Split Strategy
  const [trainPath, setTrainPath] = useState<string>('dataset/');
  const [testPath, setTestPath] = useState<string>('dataset/test/');
  const [sampleSubPath, setSampleSubPath] = useState<string>('submissions/sample_submission.csv');
  const [splitStrategy, setSplitStrategy] = useState<string>('GroupKFold');
  const [groupColumn, setGroupColumn] = useState<string>('yt_title');
  const [nFolds, setNFolds] = useState<number>(5);
  const [seed, setSeed] = useState<number>(42);
  const [idColumn, setIdColumn] = useState<string>('id');
  const [targetColumn, setTargetColumn] = useState<string>('target');
  const [missingDataPolicy, setMissingDataPolicy] = useState<string>('drop_invalid_rows');

  // Tab 3: Competition Rules & Constraints
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

  // Tab 4: Submission & Audit Specification
  const [submissionFileName, setSubmissionFileName] = useState<string>('submission.zip');
  const [archiveFormat, setArchiveFormat] = useState<'zip' | 'csv' | 'tar.gz'>('zip');
  const [minDiffPct, setMinDiffPct] = useState<string>('2.0');
  const [maxKappa, setMaxKappa] = useState<string>('0.85');
  const [minDisagreement, setMinDisagreement] = useState<string>('5.0');

  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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
      setPlatform(currentTask.platform || 'codabench');
      setName(currentTask.name || currentTask.task_name || '');
      setType(currentTask.type || currentTask.task_type || '');
      setModality(currentTask.modality || 'text');
      setPrimaryMetric(currentTask.primary_metric ? currentTask.primary_metric.replace(/\s*\(CV\)/i, '') : 'Macro-F1');
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
      setSplitStrategy(d.split_strategy || 'GroupKFold');
      setGroupColumn(d.group_column || '');
      setNFolds(d.n_folds || 5);
      setSeed(d.seed || 42);
      setIdColumn(d.id_column || 'id');
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

  const handleApplyPreset = (preset: TaskPreset) => {
    setPlatform(preset.platform || 'kaggle');
    setName(preset.name);
    setType(preset.type);
    setModality(preset.modality || 'text');
    setPrimaryMetric(preset.primary_metric ? preset.primary_metric.replace(/\s*\(CV\)/i, '') : 'Macro-F1');
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
      setMaxKappa(preset.submission_spec.max_kappa !== undefined ? String(preset.submission_spec.max_kappa) : '0.85');
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

    const updatedTask: TaskInfo = {
      platform,
      task_name: name.trim(),
      name: name.trim(),
      task_type: type.trim() || 'Supervised Learning',
      type: type.trim() || 'Supervised Learning',
      modality,
      primary_metric: primaryMetric.trim() || 'Score (CV)',
      direction,
      primary_metric_baseline: primaryBaseline ? parseFloat(primaryBaseline) : undefined,
      primary_metric_target: primaryTarget ? parseFloat(primaryTarget) : undefined,
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
        target_column: targetColumn.trim() || 'target',
        missing_data_policy: missingDataPolicy.trim(),
      },
      competition_rules: {
        allow_llm: allowLlm,
        allow_api: allowApi,
        allow_external_data: allowExternalData,
        allow_pretrained_models: allowPretrainedModels,
        max_params_limit: maxParamsLimit ? parseInt(maxParamsLimit, 10) : undefined,
        max_gpu_hours: maxGpuHours ? parseFloat(maxGpuHours) : undefined,
        daily_submission_limit: dailySubmissionLimit ? parseInt(dailySubmissionLimit, 10) : 5,
        deadline: deadline.trim() || undefined,
        team_policy: teamPolicy.trim() || undefined,
        reproducibility_required: reproducibilityRequired,
      },
      submission_spec: {
        file_name: submissionFileName.trim(),
        archive_format: archiveFormat,
        min_diff_pct: minDiffPct ? parseFloat(minDiffPct) : 2.0,
        max_kappa: maxKappa ? parseFloat(maxKappa) : 0.85,
        min_disagreement: minDisagreement ? parseFloat(minDisagreement) : 5.0,
      },
    };

    try {
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
                <span>Competition & Benchmark Setup</span>
                {name && <span className="text-xs text-zinc-400 font-normal">• {name}</span>}
              </h2>
              <p className="text-[11px] text-zinc-500">
                Configure task objectives, cross-validation data splits, competition rules, and audit gates
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

        {/* 4-Tab Navigation */}
        <div className="px-5 pt-2 pb-0 border-b border-zinc-800 bg-zinc-950/40 flex space-x-1">
          <button
            onClick={() => setActiveTab('task')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'task'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Award className="w-3.5 h-3.5" />
            <span>1. Objective & Metrics</span>
          </button>

          <button
            onClick={() => setActiveTab('data')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'data'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>2. Data & Validation Strategy</span>
          </button>

          <button
            onClick={() => setActiveTab('rules')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'rules'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Scale className="w-3.5 h-3.5" />
            <span>3. Rules & Constraints</span>
          </button>

          <button
            onClick={() => setActiveTab('submission')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
              activeTab === 'submission'
                ? 'border-zinc-200 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <FileArchive className="w-3.5 h-3.5" />
            <span>4. Submission & Audit Standards</span>
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
                    <span>Quick Competition Presets (1-Click Fill All Tabs)</span>
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
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Competition Platform</label>
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
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Problem Modality</label>
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
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Task / Competition Name</label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. StereoQueerEval, House Prices, Titanic"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Task Category / Type</label>
                  <input
                    type="text"
                    value={type}
                    onChange={(e) => setType(e.target.value)}
                    placeholder="e.g. Multi-Task Text Classification, Tabular Regression"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>
              </div>

              {/* Primary Metric & Goal Scores */}
              <div className="p-3.5 bg-zinc-950/70 border border-zinc-800/80 rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-semibold text-zinc-200">
                    Primary Benchmark Metric & Target
                  </div>
                  <span className="text-[10px] text-zinc-500 font-mono">
                    Standard ML Catalog + Custom Support
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      Pick Standard ML Metric
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
                            {m.id} — {m.name.slice(0, 32)}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Regression">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Regression').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id} — {m.name.slice(0, 32)}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Ranking & RecSys">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Ranking').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id} — {m.name.slice(0, 32)}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="NLP & Generation">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'NLP').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id} — {m.name.slice(0, 32)}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Computer Vision">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Vision').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id} — {m.name.slice(0, 32)}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Clustering">
                        {ML_METRICS_CATALOG.filter((m) => m.category === 'Clustering').map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id} — {m.name.slice(0, 32)}
                          </option>
                        ))}
                      </optgroup>
                      <option value="__custom__">Custom / Type Freely...</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      Active Metric Name
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
                    <label className="block text-[11px] text-zinc-400 mb-1">Optimization Direction</label>
                    <select
                      value={direction}
                      onChange={(e) => setDirection(e.target.value as 'maximize' | 'minimize')}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                    >
                      <option value="maximize">Maximize (F1, AUC, Acc, R²)</option>
                      <option value="minimize">Minimize (RMSE, Loss, Error)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">Baseline Score Floor</label>
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
                    <label className="block text-[11px] text-zinc-400 mb-1">Target Goal Score</label>
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
                    <label className="block text-[11px] text-zinc-400 mb-1">Languages (Comma separated)</label>
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
                    Subtask Metrics & Evaluation Criteria ({metrics.length})
                  </label>
                  <button
                    type="button"
                    onClick={handleAddMetric}
                    className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 px-2 py-0.5 rounded transition"
                  >
                    <Plus className="w-3 h-3" />
                    <span>Add Subtask</span>
                  </button>
                </div>

                {metrics.map((m, idx) => (
                  <div key={idx} className="p-3 bg-zinc-950 border border-zinc-800/80 rounded-lg space-y-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={m.name}
                        onChange={(e) => handleUpdateMetric(idx, { name: e.target.value })}
                        placeholder="Subtask Name"
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
                          <label className="block text-zinc-500 mb-0.5">Evaluation Metric</label>
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
                              <option value="__custom__">Custom / Type Below...</option>
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
                              placeholder="Metric Name"
                              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-zinc-300 font-mono text-[11px]"
                            />
                          </div>
                        </div>
                        <div>
                          <label className="block text-zinc-500 mb-0.5">Baseline Score</label>
                          <input
                            type="number"
                            step="0.0001"
                            value={m.baseline_score ?? ''}
                            onChange={(e) => handleUpdateMetric(idx, { baseline_score: parseFloat(e.target.value) || 0 })}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 font-mono text-zinc-300"
                          />
                        </div>
                        <div>
                          <label className="block text-zinc-500 mb-0.5">Target Goal</label>
                          <input
                            type="number"
                            step="0.0001"
                            value={m.target_score ?? ''}
                            onChange={(e) => handleUpdateMetric(idx, { target_score: parseFloat(e.target.value) || 0 })}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 font-mono text-emerald-400"
                          />
                        </div>
                        <div>
                          <label className="block text-zinc-500 mb-0.5">Weight (0-1.0)</label>
                          <input
                            type="number"
                            step="0.01"
                            value={m.weight ?? 1.0}
                            onChange={(e) => handleUpdateMetric(idx, { weight: parseFloat(e.target.value) || 0 })}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-zinc-300"
                          />
                        </div>
                        <div className="col-span-2 md:col-span-4">
                          <label className="block text-zinc-500 mb-0.5">Description & Special Rules</label>
                          <input
                            type="text"
                            value={m.description ?? ''}
                            onChange={(e) => handleUpdateMetric(idx, { description: e.target.value })}
                            placeholder="e.g. Evaluated strictly on hateful comments (oracle_hate)"
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

          {/* TAB 2: Data & Validation Strategy */}
          {activeTab === 'data' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Cross-Validation Split Scheme</label>
                  <select
                    value={splitStrategy}
                    onChange={(e) => setSplitStrategy(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  >
                    {SPLIT_STRATEGIES.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name} ({s.desc})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1 flex items-center gap-1.5">
                    <span>Anti-Leakage Group Column</span>
                    <span className="text-[10px] text-amber-400 font-mono">(Crucial for GroupKFold)</span>
                  </label>
                  <input
                    type="text"
                    value={groupColumn}
                    onChange={(e) => setGroupColumn(e.target.value)}
                    placeholder="e.g. video_id, yt_title, patient_id, group_id"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Number of Folds (k)</label>
                  <input
                    type="number"
                    min={2}
                    max={20}
                    value={nFolds}
                    onChange={(e) => setNFolds(parseInt(e.target.value, 10) || 5)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Fixed Random Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">ID Column Name</label>
                  <input
                    type="text"
                    value={idColumn}
                    onChange={(e) => setIdColumn(e.target.value)}
                    placeholder="e.g. id, StereoQueerEval_id"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Target Column(s)</label>
                  <input
                    type="text"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    placeholder="e.g. target, SalePrice"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Training Data Path</label>
                  <input
                    type="text"
                    value={trainPath}
                    onChange={(e) => setTrainPath(e.target.value)}
                    placeholder="e.g. dataset/train.csv or dataset/"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Test Data Path</label>
                  <input
                    type="text"
                    value={testPath}
                    onChange={(e) => setTestPath(e.target.value)}
                    placeholder="e.g. dataset/test.csv"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400 mb-1">Missing Data Handling Policy</label>
                <input
                  type="text"
                  value={missingDataPolicy}
                  onChange={(e) => setMissingDataPolicy(e.target.value)}
                  placeholder="e.g. impute_median_categorical_missing, drop_invalid_rows"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100"
                />
              </div>
            </div>
          )}

          {/* TAB 3: Competition Rules & Constraints */}
          {activeTab === 'rules' && (
            <div className="space-y-4">
              <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 space-y-3">
                <div className="text-xs font-semibold text-zinc-200">Permitted Technologies & External Sources</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowLlm}
                      onChange={(e) => setAllowLlm(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>Allow LLM / Generative Models & Prompting</span>
                  </label>

                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowApi}
                      onChange={(e) => setAllowApi(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>Allow Proprietary Closed APIs (OpenAI / Anthropic)</span>
                  </label>

                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowExternalData}
                      onChange={(e) => setAllowExternalData(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>Allow External Datasets (Must register in grounding)</span>
                  </label>

                  <label className="flex items-center gap-2 text-zinc-300">
                    <input
                      type="checkbox"
                      checked={allowPretrainedModels}
                      onChange={(e) => setAllowPretrainedModels(e.target.checked)}
                      className="rounded bg-zinc-900 border-zinc-700"
                    />
                    <span>Allow Open Pretrained Weights (HuggingFace / timm)</span>
                  </label>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Max Parameter Ceiling</label>
                  <input
                    type="number"
                    value={maxParamsLimit}
                    onChange={(e) => setMaxParamsLimit(e.target.value)}
                    placeholder="e.g. 150000000"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                  <span className="text-[10px] text-zinc-500">Limits GPU memory footprint</span>
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Max GPU Runtime (Hours)</label>
                  <input
                    type="number"
                    value={maxGpuHours}
                    onChange={(e) => setMaxGpuHours(e.target.value)}
                    placeholder="e.g. 9 or 12"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                  <span className="text-[10px] text-zinc-500">Modal/Kaggle kernel cutoff</span>
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Daily Submission Quota</label>
                  <input
                    type="number"
                    value={dailySubmissionLimit}
                    onChange={(e) => setDailySubmissionLimit(e.target.value)}
                    placeholder="e.g. 5"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                  <span className="text-[10px] text-zinc-500">Allowed submissions per day</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Competition Deadline (UTC)</label>
                  <input
                    type="text"
                    value={deadline}
                    onChange={(e) => setDeadline(e.target.value)}
                    placeholder="e.g. 2027-01-15T23:59:59Z"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                  />
                </div>

                <div>
                  <label className="block text-[11px] text-zinc-400 mb-1">Team & Collaboration Policy</label>
                  <input
                    type="text"
                    value={teamPolicy}
                    onChange={(e) => setTeamPolicy(e.target.value)}
                    placeholder="e.g. Max 4 members, no private sharing"
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
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Official Submission File Name</label>
                  <input
                    type="text"
                    value={submissionFileName}
                    onChange={(e) => setSubmissionFileName(e.target.value)}
                    placeholder="e.g. submission.csv or submission.zip"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Archive Packaging Format</label>
                  <select
                    value={archiveFormat}
                    onChange={(e) => setArchiveFormat(e.target.value as 'zip' | 'csv' | 'tar.gz')}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                  >
                    <option value="zip">ZIP Compressed Archive (.zip)</option>
                    <option value="csv">Raw Delimited CSV (.csv)</option>
                    <option value="tar.gz">Tarball (.tar.gz)</option>
                  </select>
                </div>
              </div>

              <div className="p-3.5 bg-zinc-950 rounded-lg border border-zinc-800 space-y-3">
                <div className="text-xs font-semibold text-zinc-200 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span>SWE-Agent Deterministic Audit Gate Parameters</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">MMES Threshold (% rows)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={minDiffPct}
                      onChange={(e) => setMinDiffPct(e.target.value)}
                      placeholder="2.0"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                    />
                    <span className="text-[10px] text-zinc-500">Minimum row delta vs baseline</span>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">Ensemble Max Cohen's Kappa</label>
                    <input
                      type="number"
                      step="0.01"
                      value={maxKappa}
                      onChange={(e) => setMaxKappa(e.target.value)}
                      placeholder="0.85"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                    />
                    <span className="text-[10px] text-zinc-500">Blocks identical pseudo-ensembles</span>
                  </div>

                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">Min Disagreement Rate (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={minDisagreement}
                      onChange={(e) => setMinDisagreement(e.target.value)}
                      placeholder="5.0"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-100"
                    />
                    <span className="text-[10px] text-zinc-500">Ensures distinct prediction patterns</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Footer Action Buttons */}
          <div className="pt-3 border-t border-zinc-800 flex items-center justify-between">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={saving}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-semibold shadow transition disabled:opacity-50"
            >
              <Check className="w-3.5 h-3.5" />
              <span>{saving ? 'Saving & Injecting Skill...' : 'Save & Sync Agent Skill'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
