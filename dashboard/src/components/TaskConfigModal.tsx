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
} from 'lucide-react';
import { TaskInfo, TaskMetricDef, TaskPreset } from '../types/experiment';
import { api } from '../services/api';

interface TaskConfigModalProps {
  isOpen: boolean;
  currentTask: TaskInfo | null;
  onClose: () => void;
  onSave: (updatedTask: TaskInfo) => Promise<void>;
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
  { id: 'multimodal', label: 'Multimodal' },
];

export const TaskConfigModal: React.FC<TaskConfigModalProps> = ({
  isOpen,
  currentTask,
  onClose,
  onSave,
}) => {
  const [presets, setPresets] = useState<TaskPreset[]>([]);
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
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch presets once
  useEffect(() => {
    api
      .getTaskPresets()
      .then((res) => {
        if (res.presets) setPresets(res.presets);
      })
      .catch((err) => console.warn('Could not load presets:', err));
  }, []);

  // Initialize form with currentTask values
  useEffect(() => {
    if (currentTask) {
      setPlatform(currentTask.platform || 'codabench');
      setName(currentTask.name || currentTask.task_name || '');
      setType(currentTask.type || currentTask.task_type || '');
      setModality(currentTask.modality || 'text');
      setPrimaryMetric(currentTask.primary_metric || 'Score (CV)');
      setDirection(currentTask.direction || 'maximize');
      setPrimaryBaseline(currentTask.primary_metric_baseline !== undefined ? String(currentTask.primary_metric_baseline) : '');
      setPrimaryTarget(currentTask.primary_metric_target !== undefined ? String(currentTask.primary_metric_target) : '');
      setMetrics(currentTask.metrics || []);
      setLanguages((currentTask.languages || []).join(', '));
      setDescription(currentTask.description || '');
      setError(null);
    }
  }, [currentTask, isOpen]);

  if (!isOpen) return null;

  const handleApplyPreset = (preset: TaskPreset) => {
    setPlatform(preset.platform || 'kaggle');
    setName(preset.name);
    setType(preset.type);
    setModality(preset.modality || 'text');
    setPrimaryMetric(preset.primary_metric);
    setDirection(preset.direction);
    setPrimaryBaseline(preset.primary_metric_baseline !== undefined ? String(preset.primary_metric_baseline) : '');
    setPrimaryTarget(preset.primary_metric_target !== undefined ? String(preset.primary_metric_target) : '');
    setMetrics(preset.metrics || []);
    setLanguages((preset.languages || []).join(', '));
    setDescription(preset.description || '');
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
        baseline_score: undefined,
        target_score: undefined,
        weight: 1.0,
      },
    ]);
    setExpandedMetricIndices((prev) => ({ ...prev, [metrics.length]: true }));
  };

  const toggleExpandMetric = (index: number) => {
    setExpandedMetricIndices((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  const handleUpdateMetric = (index: number, field: keyof TaskMetricDef, val: any) => {
    const updated = [...metrics];
    updated[index] = { ...updated[index], [field]: val };
    setMetrics(updated);
  };

  const handleRemoveMetric = (index: number) => {
    setMetrics(metrics.filter((_, idx) => idx !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Please provide a task or competition name.');
      return;
    }
    if (!primaryMetric.trim()) {
      setError('Please specify a primary evaluation metric.');
      return;
    }

    setSaving(true);
    setError(null);

    const langList = languages
      .split(',')
      .map((l) => l.trim())
      .filter(Boolean);

    const payload: TaskInfo = {
      platform,
      name: name.trim(),
      task_name: name.trim(),
      type: type.trim() || 'Supervised Learning',
      task_type: type.trim() || 'Supervised Learning',
      modality,
      primary_metric: primaryMetric.trim(),
      direction,
      primary_metric_baseline: primaryBaseline ? parseFloat(primaryBaseline) : undefined,
      primary_metric_target: primaryTarget ? parseFloat(primaryTarget) : undefined,
      metrics,
      languages: langList,
      description: description.trim(),
    };

    try {
      await api.updateTaskInfo(payload);
      await onSave(payload);
      onClose();
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to save task configuration.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden my-8 flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-5 border-b border-slate-800 flex items-start justify-between bg-slate-950/80">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
                <Sliders className="w-4 h-4" />
              </div>
              <h2 className="text-base font-bold text-white tracking-tight">
                Task & Competition Setup Wizard
              </h2>
              <span className="text-[10px] bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded-full font-mono">
                Task-Agnostic
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Configure target platform, evaluation criteria, and subtasks to adapt the harness to any new task.
            </p>
          </div>

          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar text-xs">
          {error && (
            <div className="bg-rose-950/40 border border-rose-800/60 p-3 rounded-lg text-rose-300">
              {error}
            </div>
          )}

          {/* Quick Presets Bar */}
          {presets.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-slate-400 font-semibold text-[11px] uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>1-Click Presets for Newbies:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {presets.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleApplyPreset(p)}
                    className="bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 hover:border-indigo-500/50 px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition active:scale-95 shadow-sm"
                  >
                    {p.preset_name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Section 1: Platform Selection */}
          <div className="space-y-2 border-t border-slate-800/80 pt-4">
            <label className="block text-slate-300 font-semibold">
              1. Competition / Host Platform
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {PLATFORMS.map((plat) => {
                const isSelected = platform === plat.id;
                return (
                  <button
                    key={plat.id}
                    type="button"
                    onClick={() => setPlatform(plat.id)}
                    className={`flex flex-col items-start p-2.5 rounded-xl border text-left transition ${
                      isSelected
                        ? 'bg-indigo-950/40 border-indigo-500 ring-1 ring-indigo-500/40 shadow-sm'
                        : 'bg-slate-950 hover:bg-slate-800/60 border-slate-800 text-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className={`font-semibold text-xs ${isSelected ? 'text-white' : 'text-slate-200'}`}>
                        {plat.name}
                      </span>
                      {isSelected && <Check className="w-3.5 h-3.5 text-indigo-400" />}
                    </div>
                    <span className="text-[10px] text-slate-500 mt-0.5">{plat.badge}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Task Basics & Modality */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800/80 pt-4">
            <div>
              <label className="block text-slate-300 font-semibold mb-1">
                Task / Competition Name
              </label>
              <input
                type="text"
                placeholder="e.g. StereoQueerEval or House Prices"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-slate-300 font-semibold mb-1">
                Problem Type
              </label>
              <input
                type="text"
                placeholder="e.g. Multi-Task Text Classification or Regression"
                value={type}
                onChange={(e) => setType(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-slate-300 font-semibold mb-1">
                Data Modality
              </label>
              <select
                value={modality}
                onChange={(e) => setModality(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                {MODALITIES.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-300 font-semibold mb-1">
                Languages (if NLP, optional)
              </label>
              <input
                type="text"
                placeholder="e.g. EN, IT, NL (leave blank if tabular/vision)"
                value={languages}
                onChange={(e) => setLanguages(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          {/* Section 3: Primary Optimization Metric */}
          <div className="border-t border-slate-800/80 pt-4 space-y-3">
            <label className="block text-slate-300 font-semibold">
              2. Primary Evaluation Metric & Direction
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-slate-400 text-[11px] mb-1">Metric Name</label>
                <input
                  type="text"
                  placeholder="e.g. Macro-F1 (CV), RMSE (CV), ROC-AUC (CV)"
                  value={primaryMetric}
                  onChange={(e) => setPrimaryMetric(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 text-[11px] mb-1">Optimization Direction</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setDirection('maximize')}
                    className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border font-medium transition ${
                      direction === 'maximize'
                        ? 'bg-emerald-950/40 border-emerald-500 text-emerald-300 ring-1 ring-emerald-500/40'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <ArrowUpRight className="w-3.5 h-3.5" />
                    <span>Maximize ↗</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setDirection('minimize')}
                    className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border font-medium transition ${
                      direction === 'minimize'
                        ? 'bg-amber-950/40 border-amber-500 text-amber-300 ring-1 ring-amber-500/40'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <ArrowDownRight className="w-3.5 h-3.5" />
                    <span>Minimize ↘</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Baseline & Target Goal Scores */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
              <div>
                <label className="block text-slate-400 text-[11px] mb-1">Primary Metric Baseline (Reference)</label>
                <input
                  type="number"
                  step="0.0001"
                  placeholder="e.g. 0.6500"
                  value={primaryBaseline}
                  onChange={(e) => setPrimaryBaseline(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 font-mono text-xs focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-400 text-[11px] mb-1">Primary Metric Target / Goal Score</label>
                <input
                  type="number"
                  step="0.0001"
                  placeholder="e.g. 0.8500"
                  value={primaryTarget}
                  onChange={(e) => setPrimaryTarget(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 font-mono text-xs focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>
          </div>

          {/* Section 4: Subtasks & Multi-Target Evaluation */}
          <div className="border-t border-slate-800/80 pt-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="block text-slate-300 font-semibold">
                  3. Subtasks & Expandable Criteria
                </label>
                <p className="text-[11px] text-slate-400">
                  Expand each subtask to configure calculation functions, per-metric direction, baseline & target goals, and composite weights.
                </p>
              </div>

              <button
                type="button"
                onClick={handleAddMetric}
                className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 text-[11px] font-medium transition active:scale-95"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Subtask</span>
              </button>
            </div>

            {metrics.length === 0 ? (
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 text-center text-slate-500 text-[11px]">
                No subtasks defined. The harness will focus exclusively on the primary metric ({primaryMetric}).
              </div>
            ) : (
              <div className="space-y-2.5">
                {metrics.map((m, idx) => {
                  const isExpanded = !!expandedMetricIndices[idx];

                  return (
                    <div
                      key={idx}
                      className="bg-slate-950 rounded-lg border border-slate-800 overflow-hidden"
                    >
                      {/* Compact Header Row */}
                      <div className="grid grid-cols-12 gap-2 items-center p-2.5">
                        <div className="col-span-3">
                          <input
                            type="text"
                            placeholder="ID (e.g. subtask_a)"
                            value={m.id}
                            onChange={(e) => handleUpdateMetric(idx, 'id', e.target.value)}
                            className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-indigo-500"
                          />
                        </div>
                        <div className="col-span-4">
                          <input
                            type="text"
                            placeholder="Display Name (e.g. Hate Speech)"
                            value={m.name}
                            onChange={(e) => handleUpdateMetric(idx, 'name', e.target.value)}
                            className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] focus:outline-none focus:border-indigo-500"
                          />
                        </div>
                        <div className="col-span-2">
                          <input
                            type="text"
                            placeholder="Metric (e.g. F1)"
                            value={m.metric}
                            onChange={(e) => handleUpdateMetric(idx, 'metric', e.target.value)}
                            className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] font-mono focus:outline-none focus:border-indigo-500"
                          />
                        </div>
                        <div className="col-span-2 flex items-center justify-end">
                          <button
                            type="button"
                            onClick={() => toggleExpandMetric(idx)}
                            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition ${
                              isExpanded
                                ? 'bg-indigo-900/60 text-indigo-300 border border-indigo-700/50'
                                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                            }`}
                            title="Expand full score configuration"
                          >
                            <span>Score Details</span>
                            {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                        </div>
                        <div className="col-span-1 text-center">
                          <button
                            type="button"
                            onClick={() => handleRemoveMetric(idx)}
                            className="text-slate-500 hover:text-rose-400 p-1 rounded hover:bg-slate-900 transition"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>

                      {/* Expandable Score Configuration Panel */}
                      {isExpanded && (
                        <div className="p-3 bg-slate-900/70 border-t border-slate-800/80 space-y-2.5 text-xs">
                          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5">
                            <div>
                              <label className="block text-slate-400 text-[10px] mb-0.5">Evaluation Type</label>
                              <select
                                value={m.type || 'macro_f1'}
                                onChange={(e) => handleUpdateMetric(idx, 'type', e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] focus:outline-none focus:border-indigo-500"
                              >
                                <option value="macro_f1">Macro-F1</option>
                                <option value="weighted_f1">Weighted-F1</option>
                                <option value="micro_f1">Micro-F1</option>
                                <option value="roc_auc">ROC-AUC</option>
                                <option value="accuracy">Accuracy</option>
                                <option value="logloss">LogLoss</option>
                                <option value="rmse">RMSE</option>
                                <option value="mae">MAE</option>
                                <option value="r2">R-Squared</option>
                                <option value="top1_acc">Top-1 Accuracy</option>
                                <option value="top5_acc">Top-5 Accuracy</option>
                                <option value="custom">Custom Score</option>
                              </select>
                            </div>

                            <div>
                              <label className="block text-slate-400 text-[10px] mb-0.5">Direction</label>
                              <select
                                value={m.direction || 'maximize'}
                                onChange={(e) => handleUpdateMetric(idx, 'direction', e.target.value as 'maximize' | 'minimize')}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] focus:outline-none focus:border-indigo-500"
                              >
                                <option value="maximize">Maximize ↗ (Higher is better)</option>
                                <option value="minimize">Minimize ↘ (Lower is better)</option>
                              </select>
                            </div>

                            <div>
                              <label className="block text-slate-400 text-[10px] mb-0.5">Baseline Score</label>
                              <input
                                type="number"
                                step="0.0001"
                                placeholder="e.g. 0.6000"
                                value={m.baseline_score !== undefined ? m.baseline_score : ''}
                                onChange={(e) => handleUpdateMetric(idx, 'baseline_score', e.target.value ? parseFloat(e.target.value) : undefined)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-indigo-500"
                              />
                            </div>

                            <div>
                              <label className="block text-slate-400 text-[10px] mb-0.5">Target / Goal Score</label>
                              <input
                                type="number"
                                step="0.0001"
                                placeholder="e.g. 0.8000"
                                value={m.target_score !== undefined ? m.target_score : ''}
                                onChange={(e) => handleUpdateMetric(idx, 'target_score', e.target.value ? parseFloat(e.target.value) : undefined)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-indigo-500"
                              />
                            </div>
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                            <div>
                              <label className="block text-slate-400 text-[10px] mb-0.5">Composite Weight (0.0 - 1.0)</label>
                              <input
                                type="number"
                                step="0.01"
                                placeholder="e.g. 0.33"
                                value={m.weight !== undefined ? m.weight : ''}
                                onChange={(e) => handleUpdateMetric(idx, 'weight', e.target.value ? parseFloat(e.target.value) : undefined)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-indigo-500"
                              />
                            </div>

                            <div className="sm:col-span-2">
                              <label className="block text-slate-400 text-[10px] mb-0.5">Criteria Description / Evaluation Filter</label>
                              <input
                                type="text"
                                placeholder="e.g. Evaluated only on comments where hate_speech != 'no'"
                                value={m.description || ''}
                                onChange={(e) => handleUpdateMetric(idx, 'description', e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] focus:outline-none focus:border-indigo-500"
                              />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Section 5: Description */}
          <div className="border-t border-slate-800/80 pt-4 space-y-1">
            <label className="block text-slate-300 font-semibold mb-1">
              Task Context & Notes
            </label>
            <textarea
              rows={2}
              placeholder="Brief summary of the benchmark, evaluation caveats, or data distribution..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </form>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 flex items-center justify-end gap-3 bg-slate-950/80">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition text-xs"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={saving}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white font-semibold rounded-lg transition shadow-lg shadow-indigo-600/30 text-xs disabled:opacity-50 flex items-center gap-1.5"
          >
            {saving ? 'Saving...' : 'Save & Apply Task Configuration'}
          </button>
        </div>
      </div>
    </div>
  );
};
