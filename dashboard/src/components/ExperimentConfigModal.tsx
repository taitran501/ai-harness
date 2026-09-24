import React, { useState, useEffect } from 'react';
import {
  X,
  FileCode,
  Save,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  Plus,
  Play,
  RotateCcw,
  Sparkles,
  GitBranch,
  ShieldCheck,
  FileText,
} from 'lucide-react';
import { api } from '../services/api';
import { CloudPreflightResult, ExperimentSummary } from '../types/experiment';

interface ExperimentConfigModalProps {
  initialConfigPath?: string;
  initialParentId?: string;
  experiments?: ExperimentSummary[];
  onClose: () => void;
  onConfigSaved?: (savedPath: string) => void;
}

export const ExperimentConfigModal: React.FC<ExperimentConfigModalProps> = ({
  initialConfigPath,
  initialParentId,
  experiments = [],
  onClose,
  onConfigSaved,
}) => {
  const [activeTab, setActiveTab] = useState<'editor' | 'create'>('editor');
  const [configs, setConfigs] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>(
    initialConfigPath || 'configs/exp001_tfidf_baseline.yaml'
  );

  // Editor state
  const [yamlContent, setYamlContent] = useState<string>('');
  const [initialContent, setInitialContent] = useState<string>('');
  const [loadingConfig, setLoadingConfig] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  // Preflight state
  const [runningPreflight, setRunningPreflight] = useState<boolean>(false);
  const [preflightResult, setPreflightResult] = useState<CloudPreflightResult | null>(null);

  // Create form state
  const [newExpId, setNewExpId] = useState<string>('exp002_feat_run');
  const [autoAssigningId, setAutoAssigningId] = useState<boolean>(false);
  const [newParentId, setNewParentId] = useState<string>(initialParentId || 'exp001_baseline_tfidf');
  const [newBranch, setNewBranch] = useState<string>('feat');
  const [newProblem, setNewProblem] = useState<string>('Lack of topical context for comment-level hate classification.');
  const [newHypothesis, setNewHypothesis] = useState<string>('Concatenating yt_title with comment improves semantic context and Macro-F1 by +0.02.');
  const [newIntervention, setNewIntervention] = useState<string>('Prepend title metadata to raw comment text before vectorization.');
  const [newApproachType, setNewApproachType] = useState<string>('ml');
  const [newModelFamily, setNewModelFamily] = useState<string>('linear');
  const [creating, setCreating] = useState<boolean>(false);

  const fetchNextId = async (branchName: string, desc: string = 'run') => {
    setAutoAssigningId(true);
    try {
      const res = await api.getNextExpId(branchName, desc);
      if (res && res.next_id) {
        setNewExpId(res.next_id);
      }
    } catch (err) {
      console.warn('Could not auto-fetch next experiment ID:', err);
    } finally {
      setAutoAssigningId(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'create') {
      fetchNextId(newBranch);
    }
  }, [activeTab]);

  // Load list of available configs
  const loadConfigsList = async () => {
    try {
      const res = await api.getConfigs();
      if (res.configs && res.configs.length > 0) {
        setConfigs(res.configs);
        if (!initialConfigPath && !res.configs.includes(selectedConfig)) {
          setSelectedConfig(res.configs[0]);
        }
      }
    } catch (err) {
      console.warn('Could not fetch configs list:', err);
    }
  };

  useEffect(() => {
    loadConfigsList();
  }, []);

  // Load content of selected config
  useEffect(() => {
    if (!selectedConfig) return;
    setLoadingConfig(true);
    setErrorMessage(null);
    setSaveSuccess(false);
    setPreflightResult(null);

    api
      .getConfig(selectedConfig)
      .then((res) => {
        setYamlContent(res.content || '');
        setInitialContent(res.content || '');
      })
      .catch((err) => {
        console.error(err);
        setErrorMessage(err.message || `Failed to read ${selectedConfig}`);
      })
      .finally(() => setLoadingConfig(false));
  }, [selectedConfig]);

  // Handle saving YAML content
  const handleSave = async () => {
    setSaving(true);
    setErrorMessage(null);
    setSaveSuccess(false);

    try {
      await api.saveConfig(selectedConfig, yamlContent);
      setInitialContent(yamlContent);
      setSaveSuccess(true);
      if (onConfigSaved) onConfigSaved(selectedConfig);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || 'Failed to save configuration file.');
    } finally {
      setSaving(false);
    }
  };

  // Handle preflight verification check
  const handlePreflight = async () => {
    setRunningPreflight(true);
    setErrorMessage(null);
    try {
      // First save if dirty
      if (yamlContent !== initialContent) {
        await api.saveConfig(selectedConfig, yamlContent);
        setInitialContent(yamlContent);
      }
      const res = await api.runCloudPreflight(selectedConfig);
      setPreflightResult(res);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || 'Preflight inspection failed.');
    } finally {
      setRunningPreflight(false);
    }
  };

  // Handle creating a new config
  const handleCreateNew = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setErrorMessage(null);

    try {
      const res = await api.createConfig({
        exp_id: newExpId.trim(),
        parent_id: newParentId.trim(),
        branch: newBranch,
        hypothesis: newHypothesis.trim(),
        intervention: newIntervention.trim(),
        problem_addressed: newProblem.trim(),
        approach_type: newApproachType,
        model_family: newModelFamily,
      });

      await loadConfigsList();
      setSelectedConfig(res.path);
      setActiveTab('editor');
      if (onConfigSaved) onConfigSaved(res.path);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || 'Failed to create new experiment configuration.');
    } finally {
      setCreating(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(yamlContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isDirty = yamlContent !== initialContent;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-4xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh] animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/70">
          <div className="flex items-center space-x-2.5">
            <div className="w-7 h-7 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-300">
              <FileCode className="w-3.5 h-3.5" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <span>Target Experiment Configuration</span>
                <span className="text-[11px] font-mono text-zinc-400 font-normal">
                  • {selectedConfig}
                </span>
              </h2>
              <p className="text-[11px] text-zinc-500">
                Inspect, edit hyperparameters, or create new experiment targets for local & cloud runs
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

        {/* Tab Navigation */}
        <div className="px-5 pt-2 pb-0 border-b border-zinc-800 bg-zinc-950/40 flex items-center justify-between">
          <div className="flex space-x-1">
            <button
              onClick={() => setActiveTab('editor')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
                activeTab === 'editor'
                  ? 'border-zinc-200 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Target Config Editor</span>
              {isDirty && (
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              )}
            </button>

            <button
              onClick={() => setActiveTab('create')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
                activeTab === 'create'
                  ? 'border-zinc-200 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Spawn New Experiment Target</span>
            </button>
          </div>

          {/* Config Quick Picker in Editor Mode */}
          {activeTab === 'editor' && (
            <div className="flex items-center gap-2 pb-2">
              <label className="text-[11px] text-zinc-500">Select File:</label>
              <select
                value={selectedConfig}
                onChange={(e) => setSelectedConfig(e.target.value)}
                className="bg-zinc-900 border border-zinc-800 text-zinc-200 text-xs rounded px-2.5 py-1 focus:outline-none focus:border-zinc-700 font-mono"
              >
                {configs.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Status / Alerts */}
        {errorMessage && (
          <div className="mx-5 mt-4 p-3 bg-red-950/40 border border-red-800/80 rounded-lg text-xs text-red-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {saveSuccess && (
          <div className="mx-5 mt-4 p-3 bg-emerald-950/40 border border-emerald-800/80 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span>Target configuration successfully saved to disk.</span>
          </div>
        )}

        {/* Tab 1: Editor */}
        {activeTab === 'editor' && (
          <div className="flex-1 overflow-hidden flex flex-col p-5 space-y-3">
            {/* Top Toolbar */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="text-zinc-400">File:</span>
                <span className="font-mono text-zinc-200 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                  {selectedConfig}
                </span>
                {isDirty && (
                  <span className="text-[10px] text-amber-400 bg-amber-950/50 border border-amber-800 px-1.5 py-0.5 rounded font-medium">
                    Unsaved Changes
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleCopy}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 transition"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copied ? 'Copied' : 'Copy YAML'}</span>
                </button>

                {isDirty && (
                  <button
                    type="button"
                    onClick={() => setYamlContent(initialContent)}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 rounded border border-zinc-700 transition"
                  >
                    <RotateCcw className="w-3 h-3" />
                    <span>Revert</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={handlePreflight}
                  disabled={runningPreflight}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded border border-zinc-700 transition disabled:opacity-50"
                >
                  <ShieldCheck className="w-3 h-3 text-indigo-400" />
                  <span>{runningPreflight ? 'Checking...' : 'Preflight Test'}</span>
                </button>

                <button
                  type="button"
                  onClick={handleSave}
                  disabled={saving || !isDirty}
                  className="flex items-center gap-1 px-3 py-1 text-xs bg-zinc-100 hover:bg-white text-zinc-900 font-semibold rounded transition disabled:opacity-40"
                >
                  <Save className="w-3.5 h-3.5" />
                  <span>{saving ? 'Saving...' : 'Save Config'}</span>
                </button>
              </div>
            </div>

            {/* Code Textarea Area */}
            <div className="flex-1 min-h-[340px] flex flex-col relative bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden">
              {loadingConfig ? (
                <div className="flex-1 flex items-center justify-center text-xs text-zinc-500">
                  Loading configuration...
                </div>
              ) : (
                <textarea
                  value={yamlContent}
                  onChange={(e) => setYamlContent(e.target.value)}
                  placeholder="# YAML Experiment Configuration..."
                  spellCheck={false}
                  className="w-full flex-1 p-4 bg-transparent text-zinc-200 font-mono text-xs leading-relaxed resize-none focus:outline-none selection:bg-zinc-700"
                />
              )}
            </div>

            {/* Preflight Results Panel (if run) */}
            {preflightResult && (
              <div
                className={`p-3 rounded-lg border text-xs space-y-1.5 ${
                  preflightResult.passed
                    ? 'bg-emerald-950/20 border-emerald-800/60 text-emerald-300'
                    : 'bg-red-950/20 border-red-800/60 text-red-300'
                }`}
              >
                <div className="flex items-center justify-between font-semibold">
                  <div className="flex items-center gap-1.5">
                    {preflightResult.passed ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    )}
                    <span>
                      {preflightResult.passed
                        ? 'Pre-Flight Audit Passed — Valid Experiment Schema'
                        : 'Pre-Flight Validation Errors Detected'}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                    {preflightResult.errors.length} issues
                  </span>
                </div>
                {preflightResult.errors.length > 0 && (
                  <ul className="list-disc pl-5 space-y-0.5 text-[11px] text-red-400">
                    {preflightResult.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Spawn New Experiment Target */}
        {activeTab === 'create' && (
          <form onSubmit={handleCreateNew} className="flex-1 overflow-y-auto p-5 space-y-4">
            <div className="p-3.5 bg-zinc-950 rounded-lg border border-zinc-800 space-y-1">
              <div className="text-xs font-semibold text-zinc-200 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>Hypothesis-First Rule (AGENTS.md Contract)</span>
              </div>
              <p className="text-[11px] text-zinc-400">
                Every new experiment config requires a defined parent, an atomic hypothesis, a single controlled intervention, and grounding verification.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-zinc-400 mb-1">
                  Parent Experiment
                </label>
                <select
                  value={newParentId}
                  onChange={(e) => setNewParentId(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                >
                  <option value="none">none (Root Baseline)</option>
                  {experiments.map((exp) => (
                    <option key={exp.id} value={exp.id}>
                      {exp.id} ({exp.branch}, score: {exp.score.toFixed(4)})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400 mb-1">
                  Branch Taxonomy
                </label>
                <select
                  value={newBranch}
                  onChange={(e) => {
                    const b = e.target.value;
                    setNewBranch(b);
                    fetchNextId(b);
                  }}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                >
                  <option value="feat">feat — Features & Contexts</option>
                  <option value="model">model — Architecture & Pretrained Weights</option>
                  <option value="loss">loss — Loss Objectives & Weights</option>
                  <option value="mtl">mtl — Multi-Task & Routing</option>
                  <option value="ens">ens — Ensembling & Blending</option>
                  <option value="baseline">baseline — Control Benchmarks</option>
                </select>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs font-medium text-zinc-400">
                    New Experiment ID
                  </label>
                  <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.2 rounded">
                    {autoAssigningId ? 'Resolving ID...' : 'Auto-Incremented (Next)'}
                  </span>
                </div>
                <input
                  type="text"
                  required
                  value={newExpId}
                  onChange={(e) => setNewExpId(e.target.value)}
                  placeholder="e.g. exp002_feat_run"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-700"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1">
                Problem Addressed (Failure Mode in Parent)
              </label>
              <input
                type="text"
                required
                value={newProblem}
                onChange={(e) => setNewProblem(e.target.value)}
                placeholder="e.g. High false positive rate on implicit hate speech due to missing video context."
                className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1">
                Scientific Hypothesis
              </label>
              <textarea
                required
                rows={2}
                value={newHypothesis}
                onChange={(e) => setNewHypothesis(e.target.value)}
                placeholder="e.g. Incorporating video title metadata will boost Subtask B implicit hate F1 by +0.03."
                className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1">
                Controlled Intervention (Control vs Treatment)
              </label>
              <input
                type="text"
                required
                value={newIntervention}
                onChange={(e) => setNewIntervention(e.target.value)}
                placeholder="e.g. Prepend yt_title to comment text before n-gram tokenization."
                className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-zinc-400 mb-1">Approach Type</label>
                <select
                  value={newApproachType}
                  onChange={(e) => setNewApproachType(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                >
                  <option value="ml">Classical ML (Linear / Tree / Scikit-learn)</option>
                  <option value="dl">Deep Learning (Transformers / PyTorch)</option>
                  <option value="hybrid">Hybrid (Ensemble / Stacking)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400 mb-1">Model Family</label>
                <select
                  value={newModelFamily}
                  onChange={(e) => setNewModelFamily(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-zinc-700"
                >
                  <option value="linear">linear (Logistic Regression / SVM)</option>
                  <option value="transformer">transformer (DeBERTa / XLM-R / BERT)</option>
                  <option value="tree">tree (LightGBM / XGBoost / CatBoost)</option>
                  <option value="ensemble">ensemble (Weighted Blend / Stacking)</option>
                </select>
              </div>
            </div>

            <div className="pt-2 flex justify-end gap-2 border-t border-zinc-800">
              <button
                type="button"
                onClick={() => setActiveTab('editor')}
                className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={creating}
                className="flex items-center gap-1.5 px-4 py-1.5 text-xs bg-zinc-100 hover:bg-white text-zinc-900 font-semibold rounded-md transition disabled:opacity-50"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>{creating ? 'Creating Config...' : 'Create Target Config File'}</span>
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
