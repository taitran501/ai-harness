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
  Sparkles,
  GitBranch,
  ShieldCheck,
  FileText,
  Zap,
  Target,
  Compass,
  ArrowRight,
  Eye,
  EyeOff,
  Layers,
  Cpu,
  TrendingUp,
} from 'lucide-react';
import { api } from '../services/api';
import { CloudPreflightResult, ExperimentSummary, StrategyCard } from '../types/experiment';

interface ExperimentConfigModalProps {
  initialConfigPath?: string;
  initialParentId?: string;
  experiments?: ExperimentSummary[];
  onClose: () => void;
  onConfigSaved?: (savedPath: string) => void;
}

const FALLBACK_STRATEGIES: StrategyCard[] = [
  {
    id: 'feat_video_title',
    branch: 'feat',
    badge: 'Feature Engineering',
    title: 'Video Title Context Injection',
    problem_addressed: 'Baseline TF-IDF on raw comments misses topical cues needed for implicit hate speech.',
    hypothesis: 'Prepending YouTube video title (yt_title) to comment text provides topical context, boosting Subtask B implicit hate Macro-F1 by +0.03.',
    intervention: "Prepend yt_title to raw comment text before vectorization: '[TITLE] ' + yt_title + ' [COMMENT] ' + text.",
    approach_type: 'ml',
    model_family: 'linear',
    model_name: 'logistic_regression',
    descriptor: 'video_title_ctx',
    rationale: 'YouTube comments frequently reference video topics implicitly. Injecting yt_title bridges the lexical knowledge gap.',
    target_tasks: ['Subtask A (Stereotype)', 'Subtask B (Hate Speech)'],
    estimated_delta: '+0.02 - 0.03 Macro-F1',
    uncertainty_level: 'low',
    source_type: 'verified_baseline',
    source_id: 'sklearn_tfidf_with_context',
    model_params: {
      C: 1.0,
      max_iter: 1000,
      ngram_range: [1, 2],
      max_features: 25000,
    },
  },
  {
    id: 'model_xlmr_base',
    branch: 'model',
    badge: 'Multilingual LM',
    title: 'Multilingual Transformer (XLM-RoBERTa)',
    problem_addressed: 'Classical n-grams cannot capture semantic nuances and transfer learning across EN, IT, and NL comments.',
    hypothesis: 'Fine-tuning xlm-roberta-base with shared multilingual representations will outperform bag-of-words by +0.05 Macro-F1 across non-English splits.',
    intervention: 'Replace TF-IDF + LogisticRegression backbone with fine-tuned xlm-roberta-base with pooled classification heads.',
    approach_type: 'dl',
    model_family: 'transformer',
    model_name: 'xlm-roberta-base',
    descriptor: 'xlmr_multilingual',
    rationale: 'Cross-lingual pretraining transfers toxic lexical knowledge learned in English directly to low-resource Italian and Dutch subsets.',
    target_tasks: ['All Subtasks (A, B, C)'],
    estimated_delta: '+0.04 - 0.06 Macro-F1',
    uncertainty_level: 'medium',
    source_type: 'huggingface_hub',
    source_id: 'FacebookAI/xlm-roberta-base',
    model_params: {
      learning_rate: 2e-5,
      batch_size: 16,
      epochs: 3,
      warmup_ratio: 0.1,
      weight_decay: 0.01,
    },
  },
  {
    id: 'loss_focal_identities',
    branch: 'loss',
    badge: 'Class Imbalance',
    title: 'Class-Balanced Focal Loss for Subtask C',
    problem_addressed: "Extreme class imbalance in Subtask C target identities ('a', 'i', 'nb' have <20 positive samples), resulting in near-zero recall.",
    hypothesis: 'Replacing standard Binary Cross-Entropy with Focal Loss (gamma=2.0, alpha=0.25) will penalize easy negatives and boost rare identity Macro-F1 by +0.04.',
    intervention: 'Switch Subtask C identity loss objective from BCE to SigmoidFocalLoss(gamma=2.0, alpha=0.25).',
    approach_type: 'dl',
    model_family: 'transformer',
    model_name: 'xlm-roberta-base',
    descriptor: 'focal_rare_identities',
    rationale: 'Focal Loss dynamically scales cross-entropy down for easy negatives, focusing gradients squarely on hard rare minority classes.',
    target_tasks: ['Subtask C (Identities: l, g, b, t, q, i, a, nb)'],
    estimated_delta: '+0.03 - 0.05 Identity F1',
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
    badge: 'Post-Processing',
    title: 'Language-Specific Threshold Optimization',
    problem_addressed: 'Static 0.5 decision thresholds are suboptimal because label priors and classifier calibration diverge significantly across EN, IT, and NL.',
    hypothesis: 'Optimizing decision thresholds per language on out-of-fold validation probabilities via Nelder-Mead will yield +0.015 Macro-F1 with zero extra training cost.',
    intervention: 'Apply harness/tuning_engine.py ThresholdOptimizer per language split on out-of-fold validation logits.',
    approach_type: 'ml',
    model_family: 'ensemble',
    model_name: 'threshold_optimizer',
    descriptor: 'threshold_tuning_per_lang',
    rationale: 'Threshold search shifts the precision-recall trade-off to maximize Macro-F1 directly on validation folds without retraining.',
    target_tasks: ['Subtask A', 'Subtask B', 'Subtask C'],
    estimated_delta: '+0.01 - 0.02 Macro-F1',
    uncertainty_level: 'low',
    source_type: 'verified_baseline',
    source_id: 'harness.tuning_engine',
    model_params: {
      method: 'nelder-mead',
      metric: 'macro_f1',
      per_language: true,
    },
  },
];

export const ExperimentConfigModal: React.FC<ExperimentConfigModalProps> = ({
  initialConfigPath,
  initialParentId,
  experiments = [],
  onClose,
  onConfigSaved,
}) => {
  // Determine reigning champion
  const bestExp =
    experiments.find((e) => e.is_best) ||
    (experiments.length > 0
      ? [...experiments].sort((a, b) => (b.score || 0) - (a.score || 0))[0]
      : null);
  const defaultParentId = initialParentId || bestExp?.id || 'exp001_baseline_tfidf';

  // Primary navigation: Default to 'strategies' unless initialConfigPath was explicitly opened
  const [activeTab, setActiveTab] = useState<'strategies' | 'editor'>(
    initialConfigPath ? 'editor' : 'strategies'
  );

  // Strategy & Agent State
  const [selectedParentId, setSelectedParentId] = useState<string>(defaultParentId);
  const [strategyCards, setStrategyCards] = useState<StrategyCard[]>(FALLBACK_STRATEGIES);
  const [selectedIde, setSelectedIde] = useState<'antigravity' | 'codex'>('antigravity');
  const [generatingCardId, setGeneratingCardId] = useState<string | null>(null);
  const [copiedPromptId, setCopiedPromptId] = useState<string | null>(null);
  const [previewPromptContent, setPreviewPromptContent] = useState<{ title: string; prompt: string } | null>(null);
  const [spawnSuccessMessage, setSpawnSuccessMessage] = useState<string | null>(null);

  // Editor State
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

  // Preflight state
  const [runningPreflight, setRunningPreflight] = useState<boolean>(false);
  const [preflightResult, setPreflightResult] = useState<CloudPreflightResult | null>(null);

  // Load strategy cards from server
  useEffect(() => {
    api
      .getStrategyCards()
      .then((res) => {
        if (res.strategies && res.strategies.length > 0) {
          setStrategyCards(res.strategies);
        }
      })
      .catch((err) => {
        console.warn('Using fallback strategy cards:', err);
      });
  }, []);

  // Load configs list
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

  // Handle 1-Click Spawning of Strategy Card
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

  // Build prompt for intent or strategy
  const buildAgentPrompt = (type: 'iterate' | 'explore' | 'card', card?: StrategyCard): string => {
    const parentNode = experiments.find((e) => e.id === selectedParentId);
    const parentScore = parentNode?.score ? parentNode.score.toFixed(4) : 'baseline';

    if (type === 'iterate') {
      return selectedIde === 'antigravity'
        ? `# ANTIGRAVITY ITERATION DIRECTIVE: OPTIMIZE BOTTLENECK ON ${selectedParentId}

> Target Baseline: **${selectedParentId}** (Macro-F1: ${parentScore})
> Operating Contract: Strictly adhere to AGENTS.md.
> Rule Invariant: Single Atomic Intervention (Control vs Treatment).

## MANDATORY STEPS:
1. **Failure Mode Analysis**: Inspect \`artifacts/${selectedParentId}/run.log\` and diagnostics. Identify whether the primary error is implicit hate speech (Subtask B) or rare identities (Subtask C).
2. **Formulate Target Config**: Create \`configs/exp<NNN>_<branch>_<descriptor>.yaml\` declaring:
   - \`parent_id: "${selectedParentId}"\`
   - Single atomic intervention addressing the diagnosed bottleneck.
   - Grounding citation from local knowledge base or Hugging Face Hub.
3. **Execute Smoke Test**:
   \`\`\`powershell
   python train.py --config configs/exp<NNN>_<branch>_<descriptor>.yaml --debug --fold 0
   \`\`\`
4. **Run Full 5-Fold Cross-Validation**: Execute full CV and record artifacts.
5. **Deterministic Audit Gate**: Validate that validation delta achieves MMES >= 2.0% without regression.`
        : `# OPENAI CODEX CLI DIRECTIVE: ITERATE & ADDRESS FAILURE MODES

- Parent Baseline: \`${selectedParentId}\` (Macro-F1: ${parentScore})
- Repository Rules: Follow \`AGENTS.md\`. Raw data in \`dataset/\` is read-only.
- Task: Implement ONE atomic modification to improve over \`${selectedParentId}\`.
- Action:
  1. Inspect \`artifacts/${selectedParentId}/\` to diagnose errors.
  2. Create a new config in \`configs/\` specifying the intervention.
  3. Run smoke test on Fold 0, then full 5-fold cross-validation.
  4. Summarize validation metrics and verify MMES >= 2.0% row delta.`;
    }

    if (type === 'explore') {
      return selectedIde === 'antigravity'
        ? `# ANTIGRAVITY DIRECTIVE: PARADIGM SHIFT & BRANCH EXPLORATION

> Current Record: **${selectedParentId}** (Macro-F1: ${parentScore})
> Objective: Explore an un-sampled model family per AIDE UCB tree search.

## SCIENTIFIC EXPLORATION PROTOCOL:
1. **Epistemic Grounding**: Do not guess hyperparameters. Retrieve verified model architectures (e.g. \`xlm-roberta-base\` or \`microsoft/mdeberta-v3-base\`) that support multilingual text (EN, IT, NL).
2. **Declare Experiment Config**: Create \`configs/exp<NNN>_model_<backbone>.yaml\` specifying \`grounding:\` and \`assumption_register:\`.
3. **Pre-flight & Verification Probe**: Run pre-flight check:
   \`\`\`powershell
   python harness/cloud_inspector.py --config configs/exp<NNN>_model_<backbone>.yaml
   \`\`\`
4. **Execute Cross-Validation**: Run full training, log fold-by-fold results, and update \`EXPERIMENTS.md\`.`
        : `# OPENAI CODEX CLI DIRECTIVE: ARCHITECTURE PARADIGM SHIFT

- Parent Baseline: \`${selectedParentId}\`
- Goal: Shift from linear baseline to deep learning transformer / distinct model family.
- Protocol:
  1. Create \`configs/exp<NNN>_model_<name>.yaml\` with explicit model backbone.
  2. Implement necessary model loader or pipeline in \`src/\`.
  3. Run smoke test on Fold 0: \`python train.py --config configs/exp<NNN>_...yaml --debug --fold 0\`.
  4. Run full CV and output evaluation metrics.`;
    }

    // Card-specific prompt
    if (card) {
      return `# DIRECTIVE: IMPLEMENT RESEARCH STRATEGY [${card.title.toUpperCase()}]

> Target IDE: **${selectedIde === 'antigravity' ? 'Antigravity IDE' : 'OpenAI Codex CLI'}**
> Strategy Branch: \`${card.branch}\` | Model Family: \`${card.model_family}\` (${card.model_name})
> Parent Experiment: \`${selectedParentId}\` (Score: ${parentScore})
> Estimated Gain: ${card.estimated_delta}

## SCIENTIFIC SPECIFICATION:
- **Problem Addressed**: ${card.problem_addressed}
- **Scientific Hypothesis**: ${card.hypothesis}
- **Controlled Intervention**: ${card.intervention}
- **Rationale**: ${card.rationale}

## EXECUTION SEQUENCE:
1. Create or inspect config \`configs/exp<NNN>_${card.descriptor}.yaml\` with \`parent_id: "${selectedParentId}"\`.
2. Implement the controlled intervention in \`harness/\` or \`src/\`.
3. Run smoke test verification:
   \`\`\`powershell
   python train.py --config configs/exp<NNN>_${card.descriptor}.yaml --debug --fold 0
   \`\`\`
4. Execute full 5-fold cross-validation and verify generated artifacts in \`artifacts/\`.`;
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

  const getBranchBadgeStyle = (branch: string) => {
    switch (branch) {
      case 'feat':
        return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
      case 'model':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
      case 'loss':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
      case 'ens':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      default:
        return 'bg-zinc-800 text-zinc-300 border-zinc-700';
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-5xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh] animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/70">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-300">
              <Sparkles className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <span>Experiment Targets & Strategy Hub</span>
                <span className="text-[11px] font-mono text-zinc-400 font-normal">
                  • 1-Click Strategy Catalog & Agent Spawner
                </span>
              </h2>
              <p className="text-[11px] text-zinc-500">
                Spawn vetted research targets with 1-click or generate prompt directives for Antigravity & Codex
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
              onClick={() => setActiveTab('strategies')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition ${
                activeTab === 'strategies'
                  ? 'border-amber-400 text-amber-300'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>1-Click Strategies & AI Agent</span>
              <span className="ml-1 text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.2 rounded font-mono">
                Guided
              </span>
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
              <span>Inspect & Edit YAML</span>
              {isDirty && (
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              )}
            </button>
          </div>

          {/* Quick Config Picker if in Editor */}
          {activeTab === 'editor' && (
            <div className="flex items-center gap-2 pb-2">
              <label className="text-[11px] text-zinc-500">File:</label>
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
              Dismiss
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
              <span>View & Inspect YAML</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        )}

        {saveSuccess && (
          <div className="mx-5 mt-3 p-3 bg-emerald-950/40 border border-emerald-800/80 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span>Target configuration successfully saved to disk.</span>
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
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-zinc-200">Parent Baseline Model:</span>
                    <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1.5 py-0.2 rounded font-mono">
                      Reigning Champion Auto-Detected
                    </span>
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    <select
                      value={selectedParentId}
                      onChange={(e) => setSelectedParentId(e.target.value)}
                      className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded-md px-3 py-1 font-mono focus:outline-none focus:border-zinc-500"
                    >
                      <option value="exp001_baseline_tfidf">exp001_baseline_tfidf (Root TF-IDF Baseline)</option>
                      {experiments.map((exp) => (
                        <option key={exp.id} value={exp.id}>
                          {exp.id} ({exp.branch}, score: {exp.score ? exp.score.toFixed(4) : 'N/A'}{exp.is_best ? ' 🏆' : ''})
                        </option>
                      ))}
                    </select>
                    <span className="text-[11px] text-zinc-500">
                      All candidate runs will benchmark directly against this Parent baseline.
                    </span>
                  </div>
                </div>
              </div>

              {/* IDE Selector */}
              <div className="flex items-center gap-2 self-end md:self-auto bg-zinc-900/90 p-1 rounded-lg border border-zinc-800">
                <span className="text-[11px] text-zinc-400 px-2 font-medium">Prompt IDE:</span>
                <button
                  type="button"
                  onClick={() => setSelectedIde('antigravity')}
                  className={`px-2.5 py-1 text-xs rounded font-medium transition ${
                    selectedIde === 'antigravity'
                      ? 'bg-zinc-800 text-zinc-100 shadow-sm border border-zinc-700'
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
                      ? 'bg-zinc-800 text-zinc-100 shadow-sm border border-zinc-700'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  OpenAI Codex
                </button>
              </div>
            </div>

            {/* SECTION 1: Delegate Target Design to AI Agent */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-semibold text-zinc-200 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-amber-400" />
                    <span>Delegate Next Target Design to AI Agent</span>
                  </h3>
                  <p className="text-[11px] text-zinc-400">
                    No manual YAML editing required. Select a directive below to generate an epistemic prompt injected with repository diagnostics.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {/* Intent Card 1: Iterate / Optimize diagnosed bottleneck */}
                <div className="p-4 bg-zinc-950/60 hover:bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl transition flex flex-col justify-between space-y-3">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                          <Target className="w-3.5 h-3.5" />
                        </div>
                        <span className="text-xs font-semibold text-zinc-200">
                          Iterate & Fix Diagnosed Failure Modes
                        </span>
                      </div>
                      <span className="text-[10px] font-mono bg-zinc-900 border border-zinc-800 px-1.5 py-0.5 rounded text-zinc-400">
                        Atomic Patch
                      </span>
                    </div>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">
                      AI Agent analyzes failure modes and diagnostic logs of <code className="text-zinc-300 font-mono">{selectedParentId}</code> (e.g. implicit hate speech, rare identities) to apply a single atomic patch.
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
                          <span>Hide Preview</span>
                        </>
                      ) : (
                        <>
                          <Eye className="w-3 h-3" />
                          <span>Preview Directive</span>
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
                          <span className="text-emerald-300">Directive Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-zinc-400" />
                          <span>Copy Agent Directive</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Intent Card 2: Explore / Novel architecture paradigm */}
                <div className="p-4 bg-zinc-950/60 hover:bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl transition flex flex-col justify-between space-y-3">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                          <Compass className="w-3.5 h-3.5" />
                        </div>
                        <span className="text-xs font-semibold text-zinc-200">
                          Explore Novel Architecture & Paradigm
                        </span>
                      </div>
                      <span className="text-[10px] font-mono bg-purple-500/10 text-purple-400 border border-purple-500/20 px-1.5 py-0.5 rounded">
                        UCB Tree Search
                      </span>
                    </div>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">
                      AI Agent explores a completely new model family (Multilingual Transformer / Multi-Task Learning) guided by UCB tree search with epistemic grounding.
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
                          <span>Hide Preview</span>
                        </>
                      ) : (
                        <>
                          <Eye className="w-3 h-3" />
                          <span>Preview Directive</span>
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
                          <span className="text-emerald-300">Directive Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-zinc-400" />
                          <span>Copy Agent Directive</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Collapsible Prompt Preview */}
              {previewPromptContent && (
                <div className="p-4 bg-zinc-950 rounded-xl border border-zinc-800 space-y-2 animate-in fade-in duration-150">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-300 font-mono">
                      {previewPromptContent.title} ({selectedIde})
                    </span>
                    <button
                      type="button"
                      onClick={() => setPreviewPromptContent(null)}
                      className="text-zinc-500 hover:text-zinc-300 text-xs"
                    >
                      Close Preview
                    </button>
                  </div>
                  <pre className="text-[11px] font-mono bg-zinc-900 p-3 rounded-lg border border-zinc-800 text-zinc-300 max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                    {previewPromptContent.prompt}
                  </pre>
                </div>
              )}
            </div>

            {/* SECTION 2: Grounded Research Strategy Catalog */}
            <div className="space-y-3 pt-2 border-t border-zinc-800">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-semibold text-zinc-200 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Grounded Research Strategy Catalog (1-Click)</span>
                  </h3>
                  <p className="text-[11px] text-zinc-400">
                    Vetted scientific interventions conforming to AGENTS.md standards. 1-click to scaffold a compliant config or copy an agent directive.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {strategyCards.map((card) => {
                  const isGenerating = generatingCardId === card.id;
                  const isCopied = copiedPromptId === card.id;

                  return (
                    <div
                      key={card.id}
                      className="p-4 bg-zinc-950/70 hover:bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl transition flex flex-col justify-between space-y-3.5 shadow-sm"
                    >
                      <div className="space-y-2">
                        {/* Header Badge & Delta */}
                        <div className="flex items-center justify-between">
                          <span
                            className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border font-semibold ${getBranchBadgeStyle(
                              card.branch
                            )}`}
                          >
                            {card.badge || card.branch}
                          </span>
                          <span className="text-[11px] font-semibold text-emerald-400 flex items-center gap-1">
                            <TrendingUp className="w-3 h-3" />
                            <span>{card.estimated_delta}</span>
                          </span>
                        </div>

                        {/* Title */}
                        <h4 className="text-xs font-bold text-zinc-100">{card.title}</h4>

                        {/* Problem & Rationale */}
                        <p className="text-[11px] text-zinc-400 leading-relaxed">
                          {card.rationale || card.problem_addressed}
                        </p>

                        {/* Metadata Pills */}
                        <div className="flex flex-wrap gap-1.5 pt-1 text-[10px] font-mono text-zinc-400">
                          <span className="bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded flex items-center gap-1">
                            <Cpu className="w-2.5 h-2.5 text-zinc-500" />
                            <span>{card.model_family} ({card.model_name})</span>
                          </span>
                          {card.target_tasks && (
                            <span className="bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded text-zinc-400">
                              Targets: {card.target_tasks.join(', ')}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Action Buttons */}
                      <div className="pt-2.5 border-t border-zinc-900 flex items-center justify-between gap-2">
                        <button
                          type="button"
                          onClick={() => handleCopyPrompt('card', card)}
                          className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80 rounded border border-zinc-800 transition"
                        >
                          {isCopied ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-400" />
                              <span className="text-emerald-300">Copied!</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3 text-zinc-400" />
                              <span>Copy Prompt</span>
                            </>
                          )}
                        </button>

                        <button
                          type="button"
                          disabled={isGenerating}
                          onClick={() => handleSpawnStrategy(card)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-100 hover:bg-white text-zinc-900 font-semibold rounded-md transition shadow-sm disabled:opacity-50"
                        >
                          {isGenerating ? (
                            <span>Scaffolding Config...</span>
                          ) : (
                            <>
                              <Zap className="w-3 h-3 text-amber-500 fill-amber-500" />
                              <span>Scaffold Config (1-Click)</span>
                            </>
                          )}
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
            {/* Top Toolbar */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="text-zinc-400">Config:</span>
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
                  onClick={handleCopyYaml}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 transition"
                >
                  {copiedYaml ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedYaml ? 'Copied' : 'Copy YAML'}</span>
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
      </div>
    </div>
  );
};
