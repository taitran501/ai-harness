import React, { useState, useEffect } from 'react';
import {
  X,
  FileCode,
  Save,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  RotateCcw,
  GitBranch,
  ShieldCheck,
  ArrowRight,
  Eye,
  EyeOff,
} from 'lucide-react';
import { api } from '../services/api';
import { CloudPreflightResult, ExperimentSummary, StrategyCard, TaskInfo } from '../types/experiment';
import { useLanguage } from '../constants/i18n';

interface ExperimentConfigModalProps {
  initialConfigPath?: string;
  initialParentId?: string;
  experiments?: ExperimentSummary[];
  taskInfo?: TaskInfo | null;
  onClose: () => void;
  onConfigSaved?: (savedPath: string) => void;
}

const FALLBACK_STRATEGIES: StrategyCard[] = [
  {
    id: 'feat_representation',
    branch: 'feat',
    title: 'Input Representation & Feature Pipeline',
    problem_addressed: 'Raw feature representations miss higher-order interactions and domain signals.',
    hypothesis: 'Composing structured feature representations and n-gram/polynomial features improves baseline score.',
    intervention: 'Compose enriched input representation and normalize numerical/text features.',
    approach_type: 'ml',
    model_family: 'linear',
    model_name: 'logistic_regression',
    descriptor: 'feat_pipeline',
    rationale: 'Structured representation engineering exposes predictive feature interactions directly to the classifier.',
    target_tasks: ['Primary Task Metric'],
    uncertainty_level: 'low',
    source_type: 'verified_baseline',
    source_id: 'feature_pipeline',
    model_params: {
      C: 1.0,
      max_iter: 1000,
    },
  },
  {
    id: 'model_pretrained_encoder',
    branch: 'model',
    title: 'Pretrained Transformer Backbone',
    problem_addressed: 'Classical shallow models lack contextual representations across challenging test slices.',
    hypothesis: 'Fine-tuning a modern pretrained encoder backbone captures deep context and improves validation performance.',
    intervention: 'Replace shallow baseline with fine-tuned pretrained transformer encoder.',
    approach_type: 'dl',
    model_family: 'transformer',
    model_name: 'pretrained_encoder_base',
    descriptor: 'transformer_backbone',
    rationale: 'Pretrained representations transfer rich semantic abstractions directly to the competition dataset.',
    target_tasks: ['All Prediction Targets'],
    uncertainty_level: 'medium',
    source_type: 'huggingface_hub',
    source_id: 'pretrained_encoder',
    model_params: {
      learning_rate: 2e-5,
      batch_size: 16,
      epochs: 3,
      warmup_ratio: 0.1,
      weight_decay: 0.01,
    },
  },
  {
    id: 'loss_focal_balanced',
    branch: 'loss',
    title: 'Class-Balanced / Focal Loss Optimization',
    problem_addressed: 'Severe class imbalance and label skew degrade recall on minority classes.',
    hypothesis: 'Employing focal loss with dynamically adjusted alpha focuses gradient updates on hard minority examples.',
    intervention: 'Switch cross-entropy objective to Class-Balanced Focal Loss (gamma=2.0, alpha=0.25).',
    approach_type: 'dl',
    model_family: 'transformer',
    model_name: 'pretrained_encoder_base',
    descriptor: 'focal_balanced_loss',
    rationale: 'Focal Loss dynamically scales cross-entropy down for easy negatives, focusing gradients squarely on hard rare minority classes.',
    target_tasks: ['Minority Target Classes'],
    uncertainty_level: 'medium',
    source_type: 'verified_literature',
    source_id: 'lin2017focal',
    model_params: {
      gamma: 2.0,
      alpha: 0.25,
      loss_type: 'focal',
    },
  },
  {
    id: 'ens_threshold_opt',
    branch: 'ens',
    title: 'Threshold Calibration Utility (requires custom integration)',
    problem_addressed: 'Static decision thresholds produce miscalibrated confidence scores across validation partitions.',
    hypothesis: 'Optimizing decision thresholds per slice on out-of-fold validation probabilities via Nelder-Mead maximizes primary metric.',
    intervention: 'Apply harness ThresholdOptimizer on out-of-fold validation probabilities.',
    approach_type: 'ml',
    model_family: 'ensemble',
    model_name: 'threshold_optimizer',
    descriptor: 'threshold_tuning',
    rationale: 'Threshold search shifts the precision-recall trade-off to maximize Macro-F1 directly on validation folds without retraining.',
    target_tasks: ['All Targets'],
    uncertainty_level: 'low',
    source_type: 'verified_baseline',
    source_id: 'harness.tuning_engine',
    model_params: {
      method: 'nelder-mead',
      metric: 'macro_f1',
    },
  },
  {
    id: 'model_unsloth_qwen_7b',
    branch: 'model',
    title: 'Fast QLoRA Fine-Tuning (Qwen2.5 7B)',
    problem_addressed: 'Full fine-tuning large language models exceeds single-GPU memory and compute limits.',
    hypothesis: 'Fine-tuning with 4-bit QLoRA adapts generative knowledge to the competition format with minimal compute.',
    intervention: 'Execute 4-bit QLoRA fine-tuning on quantized base LLM with custom classification formatting.',
    approach_type: 'llm',
    model_family: 'llm_peft',
    model_name: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit',
    descriptor: 'unsloth_qwen_7b_qlora',
    rationale: '4-bit quantized Qwen 7B consumes <9GB VRAM with Unsloth Triton kernels while capturing domain context.',
    target_tasks: ['Primary Targets'],
    param_size: '7B',
    uncertainty_level: 'low',
    source_type: 'unsloth_hub',
    source_id: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit',
    model_params: {
      param_size: '7B',
      quantization: '4-bit NormalFloat (NF4)',
      paradigm: 'unsloth_qlora',
      lora_r: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      epochs: 3,
      learning_rate: 2e-4,
    },
  },
  {
    id: 'model_deepseek_r1_cot_14b',
    branch: 'model',
    title: 'DeepSeek-R1 Distill 14B Chain-of-Thought',
    problem_addressed: 'Complex classification tasks require structured reasoning steps prior to emitting categorical labels.',
    hypothesis: 'Few-shot Chain-of-Thought prompting unpacks subtle nuances and establishes a strong zero-training baseline.',
    intervention: 'Apply structured Chain-of-Thought prompting with JSON classification schema parsing.',
    approach_type: 'llm',
    model_family: 'llm_reasoning',
    model_name: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    descriptor: 'deepseek_r1_cot_14b',
    rationale: 'Reasoning models excel at unpacking implicit subtext and domain tropes without task-specific retraining.',
    target_tasks: ['Target Classifications'],
    param_size: '14B',
    uncertainty_level: 'medium',
    source_type: 'huggingface_hub',
    source_id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    model_params: {
      param_size: '14B',
      quantization: '4-bit AWQ / NF4',
      paradigm: 'zero_shot_cot',
      use_cot: true,
      temperature: 0.2,
      context_length: 8192,
    },
  },
];

export const ExperimentConfigModal: React.FC<ExperimentConfigModalProps> = ({
  initialConfigPath,
  initialParentId,
  experiments = [],
  taskInfo,
  onClose,
  onConfigSaved,
}) => {
  const { t } = useLanguage();
  const bestExp =
    experiments.find((e) => e.is_best) ||
    (experiments.length > 0
      ? [...experiments].sort((a, b) => (b.score || 0) - (a.score || 0))[0]
      : null);
  const defaultParentId = initialParentId || bestExp?.id || 'exp001_baseline_tfidf';

  const [activeTab, setActiveTab] = useState<'strategies' | 'editor'>(
    initialConfigPath ? 'editor' : 'strategies'
  );

  const [selectedParentId, setSelectedParentId] = useState<string>(defaultParentId);
  const [strategyCards, setStrategyCards] = useState<StrategyCard[]>(FALLBACK_STRATEGIES.filter(card => card.model_name !== 'threshold_optimizer'));
  const [strategyCategory, setStrategyCategory] = useState<'all' | 'llm' | 'dl' | 'ml'>('all');
  const [selectedIde, setSelectedIde] = useState<'antigravity' | 'codex'>('antigravity');
  const [generatingCardId, setGeneratingCardId] = useState<string | null>(null);
  const [copiedPromptId, setCopiedPromptId] = useState<string | null>(null);
  const [previewPromptContent, setPreviewPromptContent] = useState<{ title: string; prompt: string } | null>(null);
  const [spawnSuccessMessage, setSpawnSuccessMessage] = useState<string | null>(null);

  const [configs, setConfigs] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>(
    initialConfigPath || 'configs/exp001_tfidf_baseline.yaml'
  );
  const [yamlContent, setYamlContent] = useState<string>('');
  const [initialContent, setInitialContent] = useState<string>('');
  const [loadingConfig, setLoadingConfig] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedYaml, setCopiedYaml] = useState<boolean>(false);

  const [runningPreflight, setRunningPreflight] = useState<boolean>(false);
  const [preflightResult, setPreflightResult] = useState<CloudPreflightResult | null>(null);

  useEffect(() => {
    api
      .getStrategyCards()
      .then((res) => {
        if (res.strategies && res.strategies.length > 0) {
          setStrategyCards(res.strategies.filter(card => card.model_name !== 'threshold_optimizer'));
        }
      })
      .catch((err) => {
        console.warn('Using fallback strategy cards:', err);
      });
  }, []);

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

  const handlePreflight = async () => {
    setRunningPreflight(true);
    setErrorMessage(null);
    try {
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

  const handleSpawnStrategy = async (card: StrategyCard) => {
    setGeneratingCardId(card.id);
    setErrorMessage(null);
    setSpawnSuccessMessage(null);

    try {
      const nextInfo = await api.getNextExpId(card.branch, card.descriptor);
      const nextId = nextInfo.next_id;

      const res = await api.createConfig({
        exp_id: nextId,
        descriptor: card.descriptor,
        parent_id: selectedParentId,
        branch: card.branch,
        hypothesis: card.hypothesis,
        intervention: card.intervention,
        problem_addressed: card.problem_addressed,
        approach_type: card.approach_type,
        model_family: card.model_family,
        model_name: card.model_name,
        model_params: card.model_params,
        param_size: card.param_size,
        source_type: card.source_type,
        source_id: card.source_id,
        uncertainty_level: card.uncertainty_level,
      });

      await loadConfigsList();
      setSelectedConfig(res.path);
      setSpawnSuccessMessage(`Created target configuration: ${res.path} (Parent: ${selectedParentId})`);
      if (onConfigSaved) onConfigSaved(res.path);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || `Failed to spawn strategy: ${card.title}`);
    } finally {
      setGeneratingCardId(null);
    }
  };

  const buildAgentPrompt = (type: 'iterate' | 'explore' | 'card', card?: StrategyCard): string => {
    const parentNode = experiments.find((e) => e.id === selectedParentId);
    const parentScore = parentNode?.score ? parentNode.score.toFixed(4) : 'baseline';
    const primaryMetric = taskInfo?.primary_metric || 'primary metric';
    const artifactDir = taskInfo?.artifacts_dir || '.ml-harness/artifacts/<task-id>';
    const foldCount = taskInfo?.data_config?.n_folds;
    const validationPlan = foldCount ? `${foldCount}-fold configured validation` : 'configured validation';

    if (type === 'iterate') {
      return selectedIde === 'antigravity'
        ? `# ANTIGRAVITY ITERATION DIRECTIVE: OPTIMIZE BOTTLENECK ON ${selectedParentId}

- Target Baseline: ${selectedParentId} (${primaryMetric}: ${parentScore})
- Operating Contract: Strictly adhere to AGENTS.md.
- Rule Invariant: Single Atomic Intervention (Control vs Treatment).

## MANDATORY STEPS:
1. Failure Mode Analysis: Inspect \`${artifactDir}/${selectedParentId}/\` and diagnostics. Identify the primary failure slice or error pattern.
2. Formulate Target Config: Create \`configs/exp<NNN>_<branch>_<descriptor>.yaml\` declaring:
   - \`parent_id: "${selectedParentId}"\`
   - Single atomic intervention addressing the diagnosed bottleneck.
   - Grounding citation from local knowledge base or Hugging Face Hub.
3. Execute Smoke Test:
   \`\`\`powershell
   ml-harness-run --project-root . --config configs/exp<NNN>_<branch>_<descriptor>.yaml --debug --fold 0
   \`\`\`
4. Run Full Validation: Execute the configured ${validationPlan} and record artifacts.
5. Verify the declared task metric and follow the selected task plugin's submission contract. Do not treat validation predictions as test-set predictions.`
        : `# OPENAI CODEX CLI DIRECTIVE: ITERATE & ADDRESS FAILURE MODES

- Parent Baseline: \`${selectedParentId}\` (Score: ${parentScore})
- Repository Rules: Follow \`AGENTS.md\`. Raw data in \`dataset/\` is read-only.
- Task: Implement ONE atomic modification to improve over \`${selectedParentId}\`.
- Action:
  1. Inspect \`${artifactDir}/${selectedParentId}/\` to diagnose errors.
  2. Create a new config in \`configs/\` specifying the intervention.
  3. Run a smoke test, then the configured ${validationPlan}.
  4. Summarize validation metrics and follow the selected task plugin's submission contract. Do not use validation predictions as test-set predictions.`;
    }

    if (type === 'explore') {
      return selectedIde === 'antigravity'
        ? `# ANTIGRAVITY DIRECTIVE: PARADIGM SHIFT & BRANCH EXPLORATION

- Current Record: ${selectedParentId} (Score: ${parentScore})
- Objective: Explore an un-sampled model family per solution tree search.

## SCIENTIFIC EXPLORATION PROTOCOL:
1. Epistemic Grounding: Do not guess hyperparameters. Retrieve verified model architectures and representations matching the dataset modality.
2. Declare Experiment Config: Create \`configs/exp<NNN>_model_<backbone>.yaml\` specifying \`grounding:\` and \`assumption_register:\`.
3. Pre-flight Verification Probe:
   \`\`\`powershell
   python harness/cloud_inspector.py --config configs/exp<NNN>_model_<backbone>.yaml
   \`\`\`
4. Execute Cross-Validation: Run full training, log fold-by-fold results, and update \`EXPERIMENTS.md\`.`
        : `# OPENAI CODEX CLI DIRECTIVE: ARCHITECTURE PARADIGM SHIFT

- Parent Baseline: \`${selectedParentId}\`
- Goal: Shift from linear baseline to deep learning transformer or distinct model family.
- Protocol:
  1. Create \`configs/exp<NNN>_model_<name>.yaml\` with explicit model backbone.
  2. Implement necessary model loader or pipeline in \`src/\`.
  3. Run smoke test on Fold 0: \`ml-harness-run --project-root . --config configs/exp<NNN>_...yaml --debug --fold 0\`.
  4. Run full CV and output evaluation metrics.`;
    }

    if (card) {
      return `# DIRECTIVE: IMPLEMENT RESEARCH STRATEGY [${card.title.toUpperCase()}]

- Target IDE: ${selectedIde === 'antigravity' ? 'Antigravity IDE' : 'OpenAI Codex CLI'}
- Strategy Branch: \`${card.branch}\`
- Model Family: \`${card.model_family}\` (${card.model_name})
- Parent Experiment: \`${selectedParentId}\` (Score: ${parentScore})

## SCIENTIFIC SPECIFICATION:
- Problem Addressed: ${card.problem_addressed}
- Scientific Hypothesis: ${card.hypothesis}
- Controlled Intervention: ${card.intervention}
- Rationale: ${card.rationale}

## EXECUTION SEQUENCE:
1. Create or inspect config \`configs/exp<NNN>_${card.descriptor}.yaml\` with \`parent_id: "${selectedParentId}"\`.
2. Implement the controlled intervention in \`harness/\` or \`src/\`.
3. Run smoke test verification:
   \`\`\`powershell
   ml-harness-run --project-root . --config configs/exp<NNN>_${card.descriptor}.yaml --debug --fold 0
   \`\`\`
4. Execute the configured ${validationPlan} and verify generated artifacts in \`${artifactDir}/\`.`;
    }

    return '';
  };

  const handleCopyPrompt = (type: 'iterate' | 'explore' | 'card', card?: StrategyCard) => {
    const prompt = buildAgentPrompt(type, card);
    navigator.clipboard.writeText(prompt);
    const key = card ? card.id : type;
    setCopiedPromptId(key);
    setTimeout(() => setCopiedPromptId(null), 2500);
  };

  const handlePreviewPrompt = (type: 'iterate' | 'explore' | 'card', card?: StrategyCard) => {
    const prompt = buildAgentPrompt(type, card);
    const title = card
      ? `Strategy Prompt: ${card.title}`
      : type === 'iterate'
      ? 'Iterate & Bottleneck Prompt'
      : 'Explore Paradigm Prompt';

    if (previewPromptContent?.title === title) {
      setPreviewPromptContent(null);
    } else {
      setPreviewPromptContent({ title, prompt });
    }
  };

  const handleCopyYaml = () => {
    navigator.clipboard.writeText(yamlContent);
    setCopiedYaml(true);
    setTimeout(() => setCopiedYaml(false), 2000);
  };

  const isDirty = yamlContent !== initialContent;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-5xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/70">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">
              {t('expcfg_title', 'Strategy Catalog & Experiment Targets')}
            </h2>
            <p className="text-[11px] text-zinc-500">
              {t('expcfg_desc', 'Scaffold verified research interventions or generate prompt directives for Antigravity and OpenAI Codex.')}
            </p>
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
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveTab('strategies')}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition ${
                activeTab === 'strategies'
                  ? 'border-amber-400 text-amber-300'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {t('expcfg_strategy_catalog', 'Strategy Catalog & Directives')}
            </button>

            <button
              onClick={() => setActiveTab('editor')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
                activeTab === 'editor'
                  ? 'border-zinc-200 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <FileCode className="w-3.5 h-3.5" />
              <span>{t('expcfg_tab_editor', 'Inspect & Edit YAML')}</span>
              {isDirty && <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />}
            </button>
          </div>

          {activeTab === 'editor' && (
            <div className="flex items-center gap-2 pb-2">
              <label className="text-[11px] text-zinc-500">{t('expcfg_file_label', 'File:')}</label>
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

        {/* Global Notifications */}
        {errorMessage && (
          <div className="mx-5 mt-3 p-3 bg-red-950/40 border border-red-800/80 rounded-lg text-xs text-red-300 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-200 text-xs">
              {t('btn_dismiss', 'Dismiss')}
            </button>
          </div>
        )}

        {spawnSuccessMessage && (
          <div className="mx-5 mt-3 p-3 bg-emerald-950/40 border border-emerald-800/80 rounded-lg text-xs text-emerald-300 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span>{spawnSuccessMessage}</span>
            </div>
            <button
              onClick={() => {
                setActiveTab('editor');
                setSpawnSuccessMessage(null);
              }}
              className="flex items-center gap-1 px-2.5 py-1 text-xs bg-emerald-900/60 hover:bg-emerald-800 text-emerald-200 rounded border border-emerald-700 transition"
            >
              <span>{t('expcfg_btn_view_yaml', 'View & Inspect YAML')}</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        )}

        {saveSuccess && (
          <div className="mx-5 mt-3 p-3 bg-emerald-950/40 border border-emerald-800/80 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span>{t('expcfg_saved_notice', 'Target configuration saved to disk.')}</span>
          </div>
        )}


        {/* TAB 1: Strategy Catalog & Agent Spawner */}
        {activeTab === 'strategies' && (
          <div className="flex-1 overflow-y-auto p-5 space-y-6">
            {/* Top Lineage Bar: Parent Model & Target IDE */}
            <div className="p-4 bg-zinc-950/80 rounded-xl border border-zinc-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-zinc-900 border border-zinc-700 flex items-center justify-center text-zinc-300 flex-shrink-0">
                  <GitBranch className="w-4 h-4 text-emerald-400" />
                </div>
                <div>
                  <span className="text-xs font-semibold text-zinc-200">{t('expcfg_parent_label', 'Parent Baseline Model')}</span>
                  <div className="mt-1 flex flex-wrap items-center gap-2">
                    <select
                      value={selectedParentId}
                      onChange={(e) => setSelectedParentId(e.target.value)}
                      className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded-md px-3 py-1 font-mono focus:outline-none focus:border-zinc-500"
                    >
                      <option value="exp001_baseline_tfidf">exp001_baseline_tfidf (Root TF-IDF Baseline)</option>
                      {experiments.map((exp) => (
                        <option key={exp.id} value={exp.id}>
                          {exp.id} ({exp.branch}, score: {exp.score ? exp.score.toFixed(4) : 'N/A'}
                          {exp.is_best ? ', Best' : ''})
                        </option>
                      ))}
                    </select>
                    <span className="text-[11px] text-zinc-500">
                      {t('expcfg_parent_hint', 'Candidate runs benchmark directly against this parent baseline.')}
                    </span>
                  </div>
                </div>
              </div>

              {/* IDE Selector */}
              <div className="flex items-center gap-2 self-end md:self-auto bg-zinc-900/90 p-1 rounded-lg border border-zinc-800">
                <span className="text-[11px] text-zinc-400 px-2 font-medium">{t('expcfg_prompt_ide', 'Prompt IDE:')}</span>
                <button
                  type="button"
                  onClick={() => setSelectedIde('antigravity')}
                  className={`px-2.5 py-1 text-xs rounded font-medium transition ${
                    selectedIde === 'antigravity'
                      ? 'bg-zinc-800 text-zinc-100 border border-zinc-700'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  Antigravity
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedIde('codex')}
                  className={`px-2.5 py-1 text-xs rounded font-medium transition ${
                    selectedIde === 'codex'
                      ? 'bg-zinc-800 text-zinc-100 border border-zinc-700'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  OpenAI Codex
                </button>
              </div>
            </div>

            {/* SECTION 1: AI Agent Target Directives */}
            <div className="space-y-3">
              <div>
                <h3 className="text-xs font-semibold text-zinc-200">
                  {t('expcfg_agent_directives_title', 'AI Agent Target Directives')}
                </h3>
                <p className="text-[11px] text-zinc-400">
                  {t('expcfg_agent_directives_desc', 'Generate structured prompt directives pre-injected with failure diagnostics for autonomous coding agents.')}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                <div className="p-4 bg-zinc-950/60 hover:bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl transition flex flex-col justify-between space-y-3">
                  <div className="space-y-1.5">
                    <span className="text-xs font-semibold text-zinc-200">
                      {t('expcfg_iterate_title', 'Iterate & Fix Diagnosed Bottlenecks')}
                    </span>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">
                      {t('expcfg_iterate_desc', 'Agent analyzes failure modes and diagnostic logs of {parent} to apply a single controlled intervention.', { parent: selectedParentId })}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-zinc-900 flex items-center justify-between gap-2">
                    <button
                      type="button"
                      onClick={() => handlePreviewPrompt('iterate')}
                      className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 transition"
                    >
                      {previewPromptContent?.title.includes('Iterate') ? (
                        <>
                          <EyeOff className="w-3 h-3" />
                          <span>{t('btn_hide_preview', 'Hide Preview')}</span>
                        </>
                      ) : (
                        <>
                          <Eye className="w-3 h-3" />
                          <span>{t('btn_preview_directive', 'Preview Directive')}</span>
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleCopyPrompt('iterate')}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded-md border border-zinc-700 transition"
                    >
                      {copiedPromptId === 'iterate' ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-300">{t('btn_directive_copied', 'Directive Copied')}</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-zinc-400" />
                          <span>{t('btn_copy_directive', 'Copy Directive')}</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                <div className="p-4 bg-zinc-950/60 hover:bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl transition flex flex-col justify-between space-y-3">
                  <div className="space-y-1.5">
                    <span className="text-xs font-semibold text-zinc-200">
                      {t('expcfg_explore_title', 'Explore Novel Architecture & Paradigm')}
                    </span>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">
                      {t('expcfg_explore_desc', 'Agent explores a distinct model family (Multilingual Transformer or Multi-Task Learning) guided by tree search with epistemic grounding.')}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-zinc-900 flex items-center justify-between gap-2">
                    <button
                      type="button"
                      onClick={() => handlePreviewPrompt('explore')}
                      className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 transition"
                    >
                      {previewPromptContent?.title.includes('Explore') ? (
                        <>
                          <EyeOff className="w-3 h-3" />
                          <span>{t('btn_hide_preview', 'Hide Preview')}</span>
                        </>
                      ) : (
                        <>
                          <Eye className="w-3 h-3" />
                          <span>{t('btn_preview_directive', 'Preview Directive')}</span>
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleCopyPrompt('explore')}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded-md border border-zinc-700 transition"
                    >
                      {copiedPromptId === 'explore' ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-300">{t('btn_directive_copied', 'Directive Copied')}</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-zinc-400" />
                          <span>{t('btn_copy_directive', 'Copy Directive')}</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>


              {previewPromptContent && (
                <div className="p-4 bg-zinc-950 rounded-xl border border-zinc-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-300 font-mono">
                      {previewPromptContent.title} ({selectedIde})
                    </span>
                    <button
                      type="button"
                      onClick={() => setPreviewPromptContent(null)}
                      className="text-zinc-500 hover:text-zinc-300 text-xs"
                    >
                      {t('btn_close_preview', 'Close Preview')}
                    </button>
                  </div>
                  <pre className="text-[11px] font-mono bg-zinc-900 p-3 rounded-lg border border-zinc-800 text-zinc-300 max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                    {previewPromptContent.prompt}
                  </pre>
                </div>
              )}
            </div>

            {/* SECTION 2: Research Strategy Catalog */}
            <div className="space-y-3 pt-2 border-t border-zinc-800">
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-2.5">
                <div>
                  <h3 className="text-xs font-semibold text-zinc-200">
                    {t('expcfg_strategy_catalog', 'Research Strategy Catalog')}
                  </h3>
                  <p className="text-[11px] text-zinc-400">
                    {t('expcfg_strategy_desc', 'Scientific interventions conforming to repository standards. Scaffold a target config or copy prompt directive.')}
                  </p>
                </div>

                {/* Strategy Category Filter */}
                <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-[11px] font-medium self-end md:self-auto">
                  <button
                    type="button"
                    onClick={() => setStrategyCategory('all')}
                    className={`px-2 py-0.5 rounded transition ${
                      strategyCategory === 'all'
                        ? 'bg-zinc-800 text-zinc-100 font-semibold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    All ({strategyCards.length})
                  </button>
                  <button
                    type="button"
                    onClick={() => setStrategyCategory('llm')}
                    className={`px-2 py-0.5 rounded transition ${
                      strategyCategory === 'llm'
                        ? 'bg-zinc-800 text-zinc-100 font-semibold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    LLM ({strategyCards.filter((c) => c.approach_type === 'llm').length})
                  </button>
                  <button
                    type="button"
                    onClick={() => setStrategyCategory('dl')}
                    className={`px-2 py-0.5 rounded transition ${
                      strategyCategory === 'dl'
                        ? 'bg-zinc-800 text-zinc-100 font-semibold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    DL ({strategyCards.filter((c) => c.approach_type === 'dl').length})
                  </button>
                  <button
                    type="button"
                    onClick={() => setStrategyCategory('ml')}
                    className={`px-2 py-0.5 rounded transition ${
                      strategyCategory === 'ml'
                        ? 'bg-zinc-800 text-zinc-100 font-semibold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    ML ({strategyCards.filter((c) => c.approach_type === 'ml').length})
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {strategyCards
                  .filter((c) => (strategyCategory === 'all' ? true : c.approach_type === strategyCategory))
                  .map((card) => {
                    const isGenerating = generatingCardId === card.id;
                    const isCopied = copiedPromptId === card.id;

                    return (
                      <div
                        key={card.id}
                        className="p-4 bg-zinc-950/70 hover:bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl transition flex flex-col justify-between space-y-3"
                      >
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400">
                            <span>{t('expcfg_branch_label', 'AIDE Branch:')} {card.branch}</span>
                            <span>{t('expcfg_model_family', 'Model Family:')} {card.model_family}: {card.model_name}</span>
                          </div>
                          {card.approach_type && (
                            <div className="text-[10px] font-mono text-zinc-500">
                              <span>{t('expcfg_approach_type', 'Approach Type:')} {card.approach_type.toUpperCase()}</span>
                            </div>
                          )}

                          <h4 className="text-xs font-bold text-zinc-100">{card.title}</h4>

                          <p className="text-[11px] text-zinc-400 leading-relaxed">
                            <span className="font-semibold text-zinc-300">{t('expcfg_problem_label', 'Problem Addressed:')} </span>
                            {card.rationale || card.problem_addressed}
                          </p>

                          {card.hypothesis && (
                            <p className="text-[11px] text-zinc-400 leading-relaxed">
                              <span className="font-semibold text-zinc-300">{t('expcfg_hypothesis_label', 'Atomic Scientific Hypothesis:')} </span>
                              {card.hypothesis}
                            </p>
                          )}

                          {card.intervention && (
                            <p className="text-[11px] text-zinc-400 leading-relaxed">
                              <span className="font-semibold text-zinc-300">{t('expcfg_intervention_label', 'Scientific Intervention:')} </span>
                              {card.intervention}
                            </p>
                          )}

                          {card.target_tasks && (
                            <div className="text-[11px] text-zinc-500 font-mono">
                              Targets: {card.target_tasks.join(', ')}
                            </div>
                          )}
                        </div>

                        <div className="pt-2.5 border-t border-zinc-900 flex items-center justify-between gap-2">
                          <button
                            type="button"
                            onClick={() => handleCopyPrompt('card', card)}
                            className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80 rounded border border-zinc-800 transition"
                          >
                            {isCopied ? (
                              <>
                                <Check className="w-3 h-3 text-emerald-400" />
                                <span className="text-emerald-300">{t('btn_copied', 'Copied!')}</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3 h-3 text-zinc-400" />
                                <span>{t('btn_copy_prompt', 'Copy Prompt')}</span>
                              </>
                            )}
                          </button>

                          <button
                            type="button"
                            disabled={isGenerating}
                            onClick={() => handleSpawnStrategy(card)}
                            className="px-3 py-1.5 text-xs bg-zinc-100 hover:bg-white text-zinc-900 font-semibold rounded-md transition disabled:opacity-50"
                          >
                            {isGenerating ? t('expcfg_creating_config', 'Creating Config...') : t('expcfg_create_config', 'Create Config')}
                          </button>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: YAML Editor & Preflight Inspector */}
        {activeTab === 'editor' && (
          <div className="flex-1 overflow-hidden flex flex-col p-5 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="text-zinc-400">{t('expcfg_config_label', 'Config:')}</span>
                <span className="font-mono text-zinc-200 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                  {selectedConfig}
                </span>
                {isDirty && (
                  <span className="text-[10px] text-amber-400 bg-amber-950/50 border border-amber-800 px-1.5 py-0.5 rounded font-medium">
                    {t('expcfg_unsaved_changes', 'Unsaved Changes')}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleCopyYaml}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 transition"
                >
                  {copiedYaml ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedYaml ? t('btn_copied', 'Copied!') : t('expcfg_copy_yaml', 'Copy YAML')}</span>
                </button>

                {isDirty && (
                  <button
                    type="button"
                    onClick={() => setYamlContent(initialContent)}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 rounded border border-zinc-700 transition"
                  >
                    <RotateCcw className="w-3 h-3" />
                    <span>{t('btn_revert', 'Revert')}</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={handlePreflight}
                  disabled={runningPreflight}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded border border-zinc-700 transition disabled:opacity-50"
                >
                  <ShieldCheck className="w-3 h-3 text-indigo-400" />
                  <span>{runningPreflight ? t('status_checking', 'Checking...') : t('expcfg_btn_preflight', 'Preflight Test')}</span>
                </button>

                <button
                  type="button"
                  onClick={handleSave}
                  disabled={saving || !isDirty}
                  className="flex items-center gap-1 px-3 py-1 text-xs bg-zinc-100 hover:bg-white text-zinc-900 font-semibold rounded transition disabled:opacity-40"
                >
                  <Save className="w-3.5 h-3.5" />
                  <span>{saving ? t('status_saving', 'Saving...') : t('expcfg_btn_save_config', 'Save Config')}</span>
                </button>
              </div>
            </div>

            <div className="flex-1 min-h-[340px] flex flex-col relative bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden">
              {loadingConfig ? (
                <div className="flex-1 flex items-center justify-center text-xs text-zinc-500">
                  {t('expcfg_loading_config', 'Loading configuration...')}
                </div>
              ) : (
                <textarea
                  value={yamlContent}
                  onChange={(e) => setYamlContent(e.target.value)}
                  placeholder={t('expcfg_yaml_ph', '# YAML Experiment Configuration...')}
                  spellCheck={false}
                  className="w-full flex-1 p-4 bg-transparent text-zinc-200 font-mono text-xs leading-relaxed resize-none focus:outline-none selection:bg-zinc-700"
                />
              )}
            </div>


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
                        ? t('expcfg_preflight_passed', 'Pre-Flight Audit Passed — Valid Experiment Schema')
                        : t('expcfg_preflight_failed', 'Pre-Flight Validation Errors Detected')}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                    {t('expcfg_issues_count', '{count} issues', { count: preflightResult.errors.length })}
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
      </div>
    </div>
  );
};
