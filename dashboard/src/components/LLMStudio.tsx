import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Play,
  Cpu,
  Settings,
  Code2,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  RefreshCw,
  Plus,
  Trash2,
  HelpCircle,
  FileCode,
  Flame,
  Key,
  Database,
  ArrowRight,
  Layers,
  ChevronDown,
  Search,
  ExternalLink,
  Download,
  ThumbsUp,
  Zap,
} from 'lucide-react';
import { api } from '../services/api';
import {
  TaskInfo,
  LLMProviderInfo,
  LLMSampleDataResponse,
  LLMTestPromptResponse,
  LLMPeftGenerateResponse,
  HFModelInfo,
  UnslothModelInfo,
  ProviderModelInfo,
} from '../types/experiment';

const FALLBACK_UNSLOTH_MODELS: UnslothModelInfo[] = [
  {
    id: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit',
    name: 'Qwen2.5-7B-Instruct-bnb-4bit',
    downloads: 640000,
    likes: 1950,
    family: 'Qwen',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit',
    name: 'DeepSeek-R1-Distill-Qwen-14B-bnb-4bit',
    downloads: 580000,
    likes: 2200,
    family: 'DeepSeek',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit',
    name: 'Llama-3.3-70B-Instruct-bnb-4bit',
    downloads: 410000,
    likes: 1600,
    family: 'Llama',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    name: 'Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    downloads: 820000,
    likes: 2800,
    family: 'Llama',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit',
    name: 'DeepSeek-R1-Distill-Llama-8B-bnb-4bit',
    downloads: 490000,
    likes: 1750,
    family: 'DeepSeek',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/Qwen2.5-14B-Instruct-bnb-4bit',
    name: 'Qwen2.5-14B-Instruct-bnb-4bit',
    downloads: 380000,
    likes: 1400,
    family: 'Qwen',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
    name: 'mistral-7b-instruct-v0.3-bnb-4bit',
    downloads: 290000,
    likes: 980,
    family: 'Mistral',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/gemma-2-9b-it-bnb-4bit',
    name: 'gemma-2-9b-it-bnb-4bit',
    downloads: 260000,
    likes: 870,
    family: 'Gemma',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/Phi-3.5-mini-instruct-bnb-4bit',
    name: 'Phi-3.5-mini-instruct-bnb-4bit',
    downloads: 195000,
    likes: 720,
    family: 'Phi',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
  {
    id: 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit',
    name: 'Qwen2.5-Coder-7B-Instruct-bnb-4bit',
    downloads: 310000,
    likes: 1100,
    family: 'Qwen',
    is_bnb_4bit: true,
    is_gguf: false,
    is_instruct: true,
    recommended_for_lora: true,
    is_unsloth: true,
  },
];

const FALLBACK_HF_MODELS: HFModelInfo[] = [
  {
    id: 'Qwen/Qwen2.5-7B-Instruct',
    name: 'Qwen2.5-7B-Instruct',
    downloads: 2400000,
    likes: 3600,
    pipeline_tag: 'text-generation',
    is_unsloth: false,
  },
  {
    id: 'meta-llama/Llama-3.3-70B-Instruct',
    name: 'Llama-3.3-70B-Instruct',
    downloads: 1820000,
    likes: 4400,
    pipeline_tag: 'text-generation',
    is_unsloth: false,
  },
  {
    id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    name: 'DeepSeek-R1-Distill-Qwen-14B',
    downloads: 2600000,
    likes: 6100,
    pipeline_tag: 'text-generation',
    is_unsloth: false,
  },
  {
    id: 'microsoft/phi-4',
    name: 'phi-4',
    downloads: 1250000,
    likes: 2900,
    pipeline_tag: 'text-generation',
    is_unsloth: false,
  },
  {
    id: 'FacebookAI/xlm-roberta-base',
    name: 'xlm-roberta-base',
    downloads: 5200000,
    likes: 1950,
    pipeline_tag: 'fill-mask',
    is_unsloth: false,
  },
  {
    id: 'microsoft/deberta-v3-large',
    name: 'deberta-v3-large',
    downloads: 3400000,
    likes: 1800,
    pipeline_tag: 'fill-mask',
    is_unsloth: false,
  },
];

interface LLMStudioProps {
  taskInfo: TaskInfo | null;
  onRefreshSolutionTree?: () => void;
  onOpenConfigModal?: (actionOrPath?: string, parentId?: string) => void;
}

export const LLMStudio: React.FC<LLMStudioProps> = ({
  taskInfo,
  onRefreshSolutionTree,
  onOpenConfigModal,
}) => {
  // Mode selection: Prompting vs Unsloth/PEFT vs HF Hub Explorer
  const [studioMode, setStudioMode] = useState<'prompting' | 'unsloth' | 'hf_hub'>('prompting');

  // Providers & keys state
  const [providers, setProviders] = useState<LLMProviderInfo[]>([]);
  const [selectedProviderId, setSelectedProviderId] = useState<string>('gemini');
  const [selectedModel, setSelectedModel] = useState<string>('gemini-2.0-flash');
  const [customModelInput, setCustomModelInput] = useState<string>('');
  const [isCustomModel, setIsCustomModel] = useState<boolean>(false);
  const [loadingProviders, setLoadingProviders] = useState(false);

  // Dynamic Provider Models state
  const [providerModels, setProviderModels] = useState<ProviderModelInfo[]>([]);
  const [loadingProviderModels, setLoadingProviderModels] = useState(false);

  // Key configuration modal state
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [keyToConfigure, setKeyToConfigure] = useState<string>('');
  const [apiKeyInput, setApiKeyInput] = useState<string>('');
  const [savingKey, setSavingKey] = useState(false);
  const [keySuccessMsg, setKeySuccessMsg] = useState<string | null>(null);

  // Prompt settings
  const [temperature, setTemperature] = useState<number>(0.0);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [useCot, setUseCot] = useState<boolean>(true);
  const [systemPrompt, setSystemPrompt] = useState<string>(
    'You are an expert NLP classification model specializing in nuance, social context, and hate speech analysis.'
  );
  const [promptTemplate, setPromptTemplate] = useState<string>(
    'Analyze the following content:\n\nComment: "{text}"\n\nDetermine whether this comment contains stereotype or hate speech. Provide your final classification label strictly as requested.'
  );

  // Few-Shot examples
  const [fewShots, setFewShots] = useState<Array<{ input: string; output: string; explanation?: string }>>([]);
  const [showFewShots, setShowFewShots] = useState(false);

  // Sample data & testing
  const [sampleData, setSampleData] = useState<LLMSampleDataResponse | null>(null);
  const [selectedSampleIndex, setSelectedSampleIndex] = useState<number>(0);
  const [customInputText, setCustomInputText] = useState<string>('');
  const [useCustomInput, setUseCustomInput] = useState<boolean>(false);

  // Test execution state
  const [testingPrompt, setTestingPrompt] = useState(false);
  const [testResult, setTestResult] = useState<LLMTestPromptResponse | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  // Unsloth & LoRA Fine-Tuning state
  const [fineTuningFramework, setFineTuningFramework] = useState<'unsloth' | 'peft'>('unsloth');
  const [peftModelName, setPeftModelName] = useState('unsloth/Qwen2.5-7B-Instruct-bnb-4bit');
  const [peftTextCol, setPeftTextCol] = useState('text');
  const [peftTargetCol, setPeftTargetCol] = useState('target');
  const [peftLoraR, setPeftLoraR] = useState(16);
  const [peftLoraAlpha, setPeftLoraAlpha] = useState(32);
  const [peftMaxSeqLength, setPeftMaxSeqLength] = useState(2048);
  const [peftEpochs, setPeftEpochs] = useState(3);
  const [peftBatchSize, setPeftBatchSize] = useState(4);
  const [peftLR, setPeftLR] = useState(0.0002);
  const [generatingPeft, setGeneratingPeft] = useState(false);
  const [peftResult, setPeftResult] = useState<LLMPeftGenerateResponse | null>(null);
  const [copiedScript, setCopiedScript] = useState(false);

  // Unsloth Live Models list (initialized with curated models)
  const [unslothModels, setUnslothModels] = useState<UnslothModelInfo[]>(FALLBACK_UNSLOTH_MODELS);
  const [loadingUnslothModels, setLoadingUnslothModels] = useState(false);
  const [unslothFamilyFilter, setUnslothFamilyFilter] = useState<string>('All');

  // Hugging Face Live Hub Search (initialized with curated models)
  const [hfSearchQuery, setHfSearchQuery] = useState('');
  const [hfModels, setHfModels] = useState<HFModelInfo[]>(FALLBACK_HF_MODELS);
  const [loadingHfModels, setLoadingHfModels] = useState(false);
  const [hfSortFilter, setHfSortFilter] = useState<'downloads' | 'trendingScore' | 'likes'>('downloads');

  // Experiment Registration State
  const [registeringExp, setRegisteringExp] = useState(false);
  const [registeredExpMsg, setRegisteredExpMsg] = useState<string | null>(null);

  // Fetch providers
  const loadProviders = async () => {
    setLoadingProviders(true);
    try {
      const data = await api.getLLMProviders();
      setProviders(data.providers);
      if (data.providers.length > 0) {
        const initial = data.providers.find((p) => p.id === selectedProviderId) || data.providers[0];
        setSelectedProviderId(initial.id);
        await loadProviderModels(initial.id);
      }
    } catch (err) {
      console.warn('Failed to load LLM providers:', err);
    } finally {
      setLoadingProviders(false);
    }
  };

  // Fetch dynamic models for selected provider
  const loadProviderModels = async (providerId: string) => {
    setLoadingProviderModels(true);
    try {
      const res = await api.getProviderModels(providerId);
      setProviderModels(res.models);
      if (res.models.length > 0 && !isCustomModel) {
        setSelectedModel(res.models[0].id);
      }
    } catch (err) {
      console.warn(`Failed to fetch live models for ${providerId}:`, err);
    } finally {
      setLoadingProviderModels(false);
    }
  };

  // Fetch live Unsloth models
  const loadUnslothModels = async () => {
    setLoadingUnslothModels(true);
    try {
      const res = await api.getUnslothModels(50);
      if (res.models && res.models.length > 0) {
        setUnslothModels(res.models);
      }
    } catch (err) {
      console.warn('Failed to load Unsloth models from HF Hub:', err);
    } finally {
      setLoadingUnslothModels(false);
    }
  };

  // Search Hugging Face Hub (do not restrict pipeline tag when user searches specifically)
  const searchHfHub = async (query = '', sort = hfSortFilter) => {
    setLoadingHfModels(true);
    try {
      const trimmedQuery = query.trim();
      const res = await api.getHFModels({
        search: trimmedQuery,
        pipeline: trimmedQuery ? undefined : 'text-generation',
        sort,
        limit: 20,
      });
      if (res.models && res.models.length > 0) {
        setHfModels(res.models);
      }
    } catch (err) {
      console.warn('Failed to search Hugging Face Hub:', err);
    } finally {
      setLoadingHfModels(false);
    }
  };

  // Fetch sample data
  const loadSampleData = async () => {
    try {
      const data = await api.getLLMSampleData();
      setSampleData(data);
      if (data.columns && data.columns.length > 0) {
        setPeftTextCol(data.columns[0]);
        if (data.columns.length > 1) {
          setPeftTargetCol(data.columns[data.columns.length - 1]);
        }
      }
    } catch (err) {
      console.warn('Failed to load sample dataset:', err);
    }
  };

  useEffect(() => {
    loadProviders();
    loadSampleData();
    loadUnslothModels();
    searchHfHub('', 'downloads');
  }, []);

  // Update models when provider changes
  const handleSelectProvider = async (provId: string) => {
    setSelectedProviderId(provId);
    setIsCustomModel(false);
    await loadProviderModels(provId);
  };

  const currentProvider = providers.find((p) => p.id === selectedProviderId);

  // Open key configuration
  const handleOpenKeyConfig = (envKeyName: string | null) => {
    if (!envKeyName) return;
    setKeyToConfigure(envKeyName);
    setApiKeyInput('');
    setKeySuccessMsg(null);
    setShowKeyModal(true);
  };

  // Save API Key
  const handleSaveApiKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!keyToConfigure || !apiKeyInput.trim()) return;
    setSavingKey(true);
    setKeySuccessMsg(null);
    try {
      const res = await api.saveLLMKey(keyToConfigure, apiKeyInput.trim());
      if (res.success) {
        setKeySuccessMsg(`Key '${keyToConfigure}' saved successfully!`);
        await loadProviders();
        await loadProviderModels(selectedProviderId);
        setTimeout(() => {
          setShowKeyModal(false);
          setKeySuccessMsg(null);
        }, 1500);
      }
    } catch (err: any) {
      setKeySuccessMsg(`Error: ${err.message}`);
    } finally {
      setSavingKey(false);
    }
  };

  // Insert column token into template
  const insertColumnToken = (col: string) => {
    setPromptTemplate((prev) => `${prev} {${col}}`);
  };

  // Add few-shot item
  const handleAddFewShot = () => {
    setFewShots((prev) => [
      ...prev,
      { input: 'Sample comment text', output: 'no', explanation: 'No stereotype is detected here.' },
    ]);
  };

  // Remove few-shot item
  const handleRemoveFewShot = (index: number) => {
    setFewShots((prev) => prev.filter((_, i) => i !== index));
  };

  // Run test inference
  const handleRunInference = async () => {
    setTestingPrompt(true);
    setTestError(null);
    setTestResult(null);

    const activeModelName = isCustomModel ? customModelInput.trim() : selectedModel;
    if (!activeModelName) {
      setTestError('Please select or specify a model name.');
      setTestingPrompt(false);
      return;
    }

    let rowData: Record<string, any> = {};
    if (useCustomInput) {
      rowData = { text: customInputText || 'Sample test input' };
    } else if (sampleData && sampleData.rows.length > selectedSampleIndex) {
      rowData = sampleData.rows[selectedSampleIndex];
    } else {
      rowData = { text: 'Test input comment' };
    }

    const validClasses: string[] = [];
    if (taskInfo?.subtasks && Array.isArray(taskInfo.subtasks)) {
      taskInfo.subtasks.forEach((st: any) => {
        if (st.classes && Array.isArray(st.classes)) {
          validClasses.push(...st.classes);
        }
      });
    }

    try {
      const res = await api.testLLMPrompt({
        provider: selectedProviderId,
        model: activeModelName,
        system_prompt: systemPrompt,
        user_prompt_template: promptTemplate,
        input_row: rowData,
        temperature,
        max_tokens: maxTokens,
        use_cot: useCot,
        few_shot_examples: fewShots.length > 0 ? fewShots : undefined,
        valid_classes: validClasses.length > 0 ? validClasses : undefined,
      });
      setTestResult(res);
    } catch (err: any) {
      setTestError(err.message || 'LLM execution failed.');
    } finally {
      setTestingPrompt(false);
    }
  };

  // Generate Fine-Tuning Script (Unsloth or Standard PEFT)
  const handleGenerateScript = async (saveToDisk = false) => {
    setGeneratingPeft(true);
    try {
      const trainPath = taskInfo?.data_config?.train_path || 'dataset/train.csv';
      const cleanModelStem = peftModelName.split('/')[1]?.toLowerCase() || 'model';

      if (fineTuningFramework === 'unsloth') {
        const res = await api.generateUnslothScript({
          model_name: peftModelName,
          train_path: trainPath,
          output_dir: `artifacts/exp_unsloth_${cleanModelStem}`,
          text_column: peftTextCol,
          target_column: peftTargetCol,
          max_seq_length: peftMaxSeqLength,
          lora_r: peftLoraR,
          lora_alpha: peftLoraAlpha,
          lora_dropout: 0.0,
          num_train_epochs: peftEpochs,
          batch_size: peftBatchSize,
          learning_rate: peftLR,
          save_to_disk: saveToDisk,
        });
        setPeftResult(res);
      } else {
        const res = await api.generateLLMPeftScript({
          model_name: peftModelName,
          train_path: trainPath,
          output_dir: `artifacts/exp_peft_${cleanModelStem}`,
          text_column: peftTextCol,
          target_column: peftTargetCol,
          lora_r: peftLoraR,
          lora_alpha: peftLoraAlpha,
          num_train_epochs: peftEpochs,
          batch_size: peftBatchSize,
          learning_rate: peftLR,
          save_to_disk: saveToDisk,
        });
        setPeftResult(res);
      }
    } catch (err: any) {
      alert(`Fine-tuning script generation failed: ${err.message}`);
    } finally {
      setGeneratingPeft(false);
    }
  };

  // Register as Solution Tree Experiment
  const handleRegisterAsExperiment = async () => {
    setRegisteringExp(true);
    setRegisteredExpMsg(null);
    try {
      const activeModelName = isCustomModel ? customModelInput.trim() : selectedModel;
      const cleanSlug = activeModelName.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
      const nextIdRes = await api.getNextExpId('model', `llm_${cleanSlug}`);
      const expId = nextIdRes.next_id;

      const payload = {
        exp_id: expId,
        parent_id: 'exp001_baseline_tfidf',
        branch: 'model',
        approach_type: 'llm',
        model_family: 'llm_api',
        model_name: activeModelName,
        provider: selectedProviderId,
        hypothesis: `Zero-shot / Few-shot ICL with ${activeModelName} (CoT=${useCot}) provides high contextual generalization.`,
        intervention: `Inference pipeline via ${selectedProviderId}:${activeModelName} with temperature=${temperature}.`,
        problem_addressed: 'Complex implicit nuances require large language model pre-training and reasoning.',
        source_type: 'api_provider',
        source_id: `${selectedProviderId}/${activeModelName}`,
        retrieval_summary: `LLM provider API validated via interactive studio test.`,
        model_params: {
          temperature,
          max_tokens: maxTokens,
          use_cot: useCot,
          system_prompt: systemPrompt,
          prompt_template: promptTemplate,
        },
      };

      const saveRes = await api.createExperimentConfig(payload);
      if (saveRes.success) {
        setRegisteredExpMsg(`Registered experiment '${expId}'! Solution tree synchronized.`);
        onRefreshSolutionTree?.();
        setTimeout(() => setRegisteredExpMsg(null), 4000);
      }
    } catch (err: any) {
      setRegisteredExpMsg(`Error: ${err.message}`);
    } finally {
      setRegisteringExp(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedScript(true);
    setTimeout(() => setCopiedScript(false), 2000);
  };

  // Filter Unsloth models
  const filteredUnslothModels = unslothModels.filter((m) => {
    if (unslothFamilyFilter === 'All') return true;
    return m.family.toLowerCase() === unslothFamilyFilter.toLowerCase();
  });

  return (
    <div className="w-full flex-1 flex flex-col bg-zinc-950 text-zinc-100 overflow-y-auto">
      {/* Top Header / Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4 px-6 py-4 border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded-xl text-purple-400">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-zinc-100">LLM & Unsloth Fine-Tuning Studio</h1>
            </div>
            <p className="text-xs text-zinc-400">
              Live model discovery from Hugging Face Hub, official Unsloth 4-bit checkpoints, and multi-provider APIs.
            </p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center bg-zinc-900 border border-zinc-800 p-1 rounded-xl">
          <button
            onClick={() => setStudioMode('prompting')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              studioMode === 'prompting'
                ? 'bg-purple-600 text-white shadow'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Prompt Playground</span>
          </button>

          <button
            onClick={() => setStudioMode('unsloth')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              studioMode === 'unsloth'
                ? 'bg-amber-600 text-white shadow'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Zap className="w-3.5 h-3.5 text-amber-300" />
            <span>🦥 Unsloth LoRA Studio</span>
          </button>

          <button
            onClick={() => setStudioMode('hf_hub')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              studioMode === 'hf_hub'
                ? 'bg-blue-600 text-white shadow'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>🤗 Hugging Face Hub</span>
          </button>
        </div>
      </div>

      {/* Main Workspace Body */}
      <div className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
        {studioMode === 'prompting' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Provider & Prompt Configuration (7 cols) */}
            <div className="lg:col-span-7 space-y-5">
              {/* Provider & Model Selector Card */}
              <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-semibold text-zinc-200">Model & API Provider</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={isCustomModel}
                        onChange={(e) => setIsCustomModel(e.target.checked)}
                        className="w-3 h-3 accent-purple-500 rounded"
                      />
                      <span>Custom ID</span>
                    </label>
                    <button
                      onClick={() => loadProviderModels(selectedProviderId)}
                      disabled={loadingProviderModels}
                      className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1 transition"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${loadingProviderModels ? 'animate-spin' : ''}`} />
                      <span>Fetch Live</span>
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Provider Selection */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-400">Inference Provider</label>
                    <select
                      value={selectedProviderId}
                      onChange={(e) => handleSelectProvider(e.target.value)}
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-medium text-zinc-200 focus:outline-none focus:border-purple-500"
                    >
                      {providers.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name} {p.is_local ? '(Local Daemon)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Model Selection */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-zinc-400">Model Name</label>
                      {providerModels.length > 0 && providerModels[0].is_live && (
                        <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                          <CheckCircle2 className="w-2.5 h-2.5" />
                          <span>Live from API</span>
                        </span>
                      )}
                    </div>

                    {!isCustomModel ? (
                      <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-medium text-zinc-200 font-mono focus:outline-none focus:border-purple-500"
                      >
                        {providerModels.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name || m.id} {m.is_live ? '(Live)' : '(Docs)'}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={customModelInput}
                        onChange={(e) => setCustomModelInput(e.target.value)}
                        placeholder="e.g. gemini-2.0-flash or gpt-4o"
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-medium text-zinc-200 font-mono focus:outline-none focus:border-purple-500"
                      />
                    )}
                  </div>
                </div>

                {/* API Key / Daemon Status Row */}
                <div className="flex items-center justify-between p-3 bg-zinc-950/70 border border-zinc-800/80 rounded-xl text-xs">
                  <div className="flex items-center gap-2">
                    {currentProvider?.has_key ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                    )}
                    <span className="text-zinc-300">
                      {currentProvider?.has_key
                        ? `${currentProvider.env_key_name || 'Provider'} is verified`
                        : `Missing ${currentProvider?.env_key_name || 'API Key'}`}
                    </span>
                  </div>

                  {currentProvider?.env_key_name && (
                    <button
                      onClick={() => handleOpenKeyConfig(currentProvider.env_key_name)}
                      className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-[11px] font-medium transition flex items-center gap-1.5"
                    >
                      <Key className="w-3 h-3 text-amber-400" />
                      <span>{currentProvider.has_key ? 'Update Key' : 'Configure Key'}</span>
                    </button>
                  )}
                </div>

                {/* Hyperparameters Controls */}
                <div className="grid grid-cols-2 gap-4 pt-1">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-zinc-400">
                      <span>Temperature</span>
                      <span className="font-mono text-zinc-200">{temperature.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full accent-purple-500 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-zinc-400">
                      <span>Max Tokens</span>
                      <span className="font-mono text-zinc-200">{maxTokens}</span>
                    </div>
                    <input
                      type="range"
                      min="64"
                      max="2048"
                      step="64"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-full accent-purple-500 cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Prompt Engineering Canvas Card */}
              <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-semibold text-zinc-200">Prompt Engineering Canvas</span>
                  </div>

                  <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-zinc-300">
                    <input
                      type="checkbox"
                      checked={useCot}
                      onChange={(e) => setUseCot(e.target.checked)}
                      className="w-3.5 h-3.5 accent-purple-500 rounded cursor-pointer"
                    />
                    <span>Chain-of-Thought (CoT)</span>
                  </label>
                </div>

                {/* System Prompt */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">System Instruction</label>
                  <textarea
                    rows={2}
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl p-3 text-xs text-zinc-200 font-mono focus:outline-none focus:border-purple-500 resize-y"
                    placeholder="Enter system instruction..."
                  />
                </div>

                {/* User Prompt Template */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-zinc-400">User Prompt Template</label>
                    {sampleData?.columns && sampleData.columns.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-[10px] text-zinc-500">Insert tag:</span>
                        {sampleData.columns.slice(0, 5).map((col) => (
                          <button
                            key={col}
                            onClick={() => insertColumnToken(col)}
                            className="px-1.5 py-0.5 bg-zinc-800 hover:bg-zinc-700 text-purple-300 rounded text-[10px] font-mono border border-zinc-700 transition"
                          >
                            {`{${col}}`}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  <textarea
                    rows={4}
                    value={promptTemplate}
                    onChange={(e) => setPromptTemplate(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl p-3 text-xs text-zinc-200 font-mono focus:outline-none focus:border-purple-500 resize-y leading-relaxed"
                    placeholder="Enter template with {text} or {column_name} placeholders..."
                  />
                </div>

                {/* Few-Shot Demonstrations Accordion */}
                <div className="border border-zinc-800/80 rounded-xl overflow-hidden bg-zinc-950/40">
                  <button
                    onClick={() => setShowFewShots(!showFewShots)}
                    className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-medium text-zinc-300 hover:bg-zinc-900/60 transition"
                  >
                    <span className="flex items-center gap-2">
                      <Layers className="w-3.5 h-3.5 text-purple-400" />
                      <span>Few-Shot Demonstrations ({fewShots.length})</span>
                    </span>
                    <span className="text-[11px] text-zinc-500">{showFewShots ? 'Collapse' : 'Expand'}</span>
                  </button>

                  {showFewShots && (
                    <div className="p-3.5 space-y-3 border-t border-zinc-800 bg-zinc-950/80">
                      {fewShots.map((ex, idx) => (
                        <div key={idx} className="p-3 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2 relative">
                          <button
                            onClick={() => handleRemoveFewShot(idx)}
                            className="absolute top-2.5 right-2.5 text-zinc-500 hover:text-red-400 transition"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                          <div className="text-[10px] font-semibold text-purple-400">Example #{idx + 1}</div>
                          <input
                            type="text"
                            value={ex.input}
                            onChange={(e) => {
                              const updated = [...fewShots];
                              updated[idx].input = e.target.value;
                              setFewShots(updated);
                            }}
                            placeholder="Input text..."
                            className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono"
                          />
                          <div className="grid grid-cols-2 gap-2">
                            <input
                              type="text"
                              value={ex.output}
                              onChange={(e) => {
                                const updated = [...fewShots];
                                updated[idx].output = e.target.value;
                                setFewShots(updated);
                              }}
                              placeholder="Expected output label..."
                              className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono"
                            />
                            <input
                              type="text"
                              value={ex.explanation || ''}
                              onChange={(e) => {
                                const updated = [...fewShots];
                                updated[idx].explanation = e.target.value;
                                setFewShots(updated);
                              }}
                              placeholder="Reasoning explanation (optional)..."
                              className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono"
                            />
                          </div>
                        </div>
                      ))}

                      <button
                        onClick={handleAddFewShot}
                        className="w-full py-1.5 border border-dashed border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900 rounded-xl text-xs text-zinc-400 hover:text-zinc-200 flex items-center justify-center gap-1.5 transition"
                      >
                        <Plus className="w-3.5 h-3.5" />
                        <span>Add Few-Shot Exemplar</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column: Interactive Test Console & Results (5 cols) */}
            <div className="lg:col-span-5 space-y-5">
              {/* Test Input Card */}
              <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-semibold text-zinc-200">Input Data for Test</span>
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <button
                      onClick={() => setUseCustomInput(false)}
                      className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                        !useCustomInput ? 'bg-purple-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      Dataset Sample
                    </button>
                    <button
                      onClick={() => setUseCustomInput(true)}
                      className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                        useCustomInput ? 'bg-purple-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      Custom Text
                    </button>
                  </div>
                </div>

                {!useCustomInput ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-zinc-400">
                      <span>Pick Sample Row ({sampleData?.rows.length || 0} loaded):</span>
                      <select
                        value={selectedSampleIndex}
                        onChange={(e) => setSelectedSampleIndex(parseInt(e.target.value))}
                        className="bg-zinc-950 border border-zinc-800 rounded px-2 py-0.5 text-xs text-zinc-200"
                      >
                        {sampleData?.rows.map((_, i) => (
                          <option key={i} value={i}>
                            Row #{i + 1}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-3 text-xs font-mono text-zinc-300 max-h-36 overflow-y-auto space-y-1">
                      {sampleData && sampleData.rows.length > selectedSampleIndex ? (
                        Object.entries(sampleData.rows[selectedSampleIndex]).map(([k, v]) => (
                          <div key={k} className="truncate">
                            <span className="text-purple-400">{k}:</span> {String(v)}
                          </div>
                        ))
                      ) : (
                        <span className="text-zinc-500 italic">No dataset samples available.</span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-400">Custom Test Comment</label>
                    <textarea
                      rows={4}
                      value={customInputText}
                      onChange={(e) => setCustomInputText(e.target.value)}
                      placeholder="Type custom text to evaluate with prompt..."
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl p-3 text-xs text-zinc-200 font-mono focus:outline-none focus:border-purple-500 resize-y"
                    />
                  </div>
                )}

                {/* Run Test Button */}
                <button
                  onClick={handleRunInference}
                  disabled={testingPrompt || (!currentProvider?.has_key && !currentProvider?.is_local)}
                  className={`w-full py-2.5 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition shadow-lg ${
                    testingPrompt || (!currentProvider?.has_key && !currentProvider?.is_local)
                      ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                      : 'bg-purple-600 hover:bg-purple-500 text-white shadow-purple-950/50'
                  }`}
                >
                  {testingPrompt ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Running Inference on {isCustomModel ? customModelInput : selectedModel}...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Run Test Inference</span>
                    </>
                  )}
                </button>
              </div>

              {/* Inference Results Display Card */}
              <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                  <span className="text-sm font-semibold text-zinc-200">Execution Output</span>
                  {testResult && (
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-zinc-800 border border-zinc-700 text-zinc-300 text-[10px] font-mono rounded">
                        {testResult.latency_ms} ms
                      </span>
                      {testResult.usage.completion_tokens && (
                        <span className="px-2 py-0.5 bg-zinc-800 border border-zinc-700 text-zinc-300 text-[10px] font-mono rounded">
                          {testResult.usage.completion_tokens} tokens
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {testError && (
                  <div className="p-3 bg-red-950/40 border border-red-800/80 rounded-xl text-xs text-red-300 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                    <span>{testError}</span>
                  </div>
                )}

                {testResult ? (
                  <div className="space-y-3">
                    <div className="p-3 bg-zinc-950 border border-zinc-800 rounded-xl flex items-center justify-between">
                      <span className="text-xs text-zinc-400 font-medium">Extracted Prediction:</span>
                      <span className="px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono font-bold text-xs rounded-lg">
                        {testResult.extracted.parsed_label || 'UNKNOWN'}
                      </span>
                    </div>

                    {testResult.extracted.reasoning && (
                      <div className="p-3 bg-zinc-950/70 border border-zinc-800 rounded-xl space-y-1">
                        <span className="text-[10px] text-purple-400 font-semibold uppercase tracking-wider">
                          CoT Reasoning:
                        </span>
                        <p className="text-xs text-zinc-300 font-mono leading-relaxed">
                          {testResult.extracted.reasoning}
                        </p>
                      </div>
                    )}

                    <div className="p-3 bg-zinc-950/70 border border-zinc-800 rounded-xl space-y-1">
                      <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">
                        Raw Text Response:
                      </span>
                      <pre className="text-[11px] text-zinc-300 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto leading-relaxed">
                        {testResult.raw_text}
                      </pre>
                    </div>

                    <div className="pt-2">
                      <button
                        onClick={handleRegisterAsExperiment}
                        disabled={registeringExp}
                        className="w-full py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition shadow-md"
                      >
                        <Sparkles className="w-4 h-4" />
                        <span>{registeringExp ? 'Registering...' : 'Register as LLM Experiment in Tree'}</span>
                      </button>
                      {registeredExpMsg && (
                        <p className="text-[11px] text-center pt-2 text-emerald-400 font-medium">
                          {registeredExpMsg}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-xs text-zinc-500 italic">
                    Configure your prompt on the left and click "Run Test Inference" to evaluate.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Mode 2: Unsloth & QLoRA Fine-Tuning Studio */}
        {studioMode === 'unsloth' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-6 space-y-6">
              {/* Unsloth Hero Banner */}
              <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-gradient-to-r from-amber-950/40 via-purple-950/30 to-zinc-900/60 border border-amber-800/40 rounded-xl">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-400 text-xl font-bold">
                    🦥
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                      <span>Unsloth High-Performance LoRA & QLoRA</span>
                      <span className="px-2 py-0.5 text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded-full font-mono">
                        2x - 5x Faster • 80% Less VRAM
                      </span>
                    </h2>
                    <p className="text-xs text-zinc-400">
                      Unsloth implements manual backprop math and FlashAttention kernels to dramatically accelerate fine-tuning for competitive LLM tasks.
                    </p>
                  </div>
                </div>

                {/* Framework Selector */}
                <div className="flex items-center bg-zinc-950 border border-zinc-800 p-1 rounded-xl">
                  <button
                    onClick={() => setFineTuningFramework('unsloth')}
                    className={`px-3 py-1 text-xs font-semibold rounded-lg transition ${
                      fineTuningFramework === 'unsloth'
                        ? 'bg-amber-600 text-white shadow'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    Unsloth (Recommended)
                  </button>
                  <button
                    onClick={() => setFineTuningFramework('peft')}
                    className={`px-3 py-1 text-xs font-semibold rounded-lg transition ${
                      fineTuningFramework === 'peft'
                        ? 'bg-purple-600 text-white shadow'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    Standard PEFT
                  </button>
                </div>
              </div>

              {/* Model Selection & Unsloth Model Hub Picker */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-zinc-300 flex items-center gap-2">
                    <FileCode className="w-4 h-4 text-amber-400" />
                    <span>Select Base Model (HF Hub ID)</span>
                  </label>
                  <button
                    onClick={loadUnslothModels}
                    disabled={loadingUnslothModels}
                    className="text-xs text-amber-400 hover:text-amber-300 flex items-center gap-1 transition"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loadingUnslothModels ? 'animate-spin' : ''}`} />
                    <span>Refresh Unsloth Hub API</span>
                  </button>
                </div>

                <input
                  type="text"
                  value={peftModelName}
                  onChange={(e) => setPeftModelName(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-amber-500"
                  placeholder="e.g. unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
                />

                {/* Unsloth Curated Model Pills */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-[10px] text-zinc-500 font-mono">Filter Family:</span>
                      {['All', 'Qwen', 'Llama', 'DeepSeek', 'Mistral'].map((fam) => (
                        <button
                          key={fam}
                          onClick={() => setUnslothFamilyFilter(fam)}
                          className={`px-2 py-0.5 rounded text-[10px] font-medium transition ${
                            unslothFamilyFilter === fam
                              ? 'bg-amber-600 text-white'
                              : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                          }`}
                        >
                          {fam}
                        </button>
                      ))}
                    </div>
                    <span className="text-[10px] text-zinc-500 font-mono">
                      {filteredUnslothModels.length} models fetched
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto p-1 bg-zinc-950/60 border border-zinc-800/80 rounded-xl">
                    {filteredUnslothModels.slice(0, 15).map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setPeftModelName(m.id)}
                        className={`text-left p-2 rounded-lg border text-xs font-mono transition truncate ${
                          peftModelName === m.id
                            ? 'bg-amber-950/50 border-amber-600 text-amber-200'
                            : 'bg-zinc-900/60 border-zinc-800/60 text-zinc-300 hover:border-zinc-700 hover:bg-zinc-900'
                        }`}
                      >
                        <div className="font-semibold truncate text-[11px]">{m.id}</div>
                        <div className="flex items-center justify-between text-[10px] text-zinc-500 mt-0.5">
                          <span>{m.family}</span>
                          <span>{m.downloads.toLocaleString()} dls</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Dataset Columns Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">Dataset Text Column</label>
                  <input
                    type="text"
                    value={peftTextCol}
                    onChange={(e) => setPeftTextCol(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-mono text-zinc-200"
                    placeholder="e.g. text or comment"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">Dataset Target Column</label>
                  <input
                    type="text"
                    value={peftTargetCol}
                    onChange={(e) => setPeftTargetCol(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-mono text-zinc-200"
                    placeholder="e.g. target or label"
                  />
                </div>
              </div>

              {/* Hyperparameters Grid */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3 p-4 bg-zinc-950/70 border border-zinc-800/80 rounded-xl text-xs font-mono">
                <div>
                  <label className="text-[10px] text-zinc-500">LoRA Rank (r)</label>
                  <select
                    value={peftLoraR}
                    onChange={(e) => setPeftLoraR(parseInt(e.target.value))}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-200 mt-1"
                  >
                    <option value={8}>r = 8</option>
                    <option value={16}>r = 16 (Optimal)</option>
                    <option value={32}>r = 32</option>
                    <option value={64}>r = 64</option>
                  </select>
                </div>

                <div>
                  <label className="text-[10px] text-zinc-500">LoRA Alpha</label>
                  <select
                    value={peftLoraAlpha}
                    onChange={(e) => setPeftLoraAlpha(parseInt(e.target.value))}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-200 mt-1"
                  >
                    <option value={16}>alpha = 16</option>
                    <option value={32}>alpha = 32 (Optimal)</option>
                    <option value={64}>alpha = 64</option>
                  </select>
                </div>

                <div>
                  <label className="text-[10px] text-zinc-500">Max Seq Length</label>
                  <select
                    value={peftMaxSeqLength}
                    onChange={(e) => setPeftMaxSeqLength(parseInt(e.target.value))}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-200 mt-1"
                  >
                    <option value={1024}>1024 tokens</option>
                    <option value={2048}>2048 tokens</option>
                    <option value={4096}>4096 tokens</option>
                  </select>
                </div>

                <div>
                  <label className="text-[10px] text-zinc-500">Epochs</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={peftEpochs}
                    onChange={(e) => setPeftEpochs(parseInt(e.target.value))}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-200 mt-1"
                  />
                </div>

                <div>
                  <label className="text-[10px] text-zinc-500">Batch Size</label>
                  <input
                    type="number"
                    min="1"
                    max="32"
                    value={peftBatchSize}
                    onChange={(e) => setPeftBatchSize(parseInt(e.target.value))}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-200 mt-1"
                  />
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => handleGenerateScript(false)}
                  disabled={generatingPeft}
                  className={`px-5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition shadow-lg ${
                    fineTuningFramework === 'unsloth'
                      ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-950/50'
                      : 'bg-purple-600 hover:bg-purple-500 text-white shadow-purple-950/50'
                  }`}
                >
                  <Zap className="w-4 h-4" />
                  <span>
                    {generatingPeft
                      ? 'Generating...'
                      : `Generate ${fineTuningFramework === 'unsloth' ? 'Unsloth Fast' : 'PEFT'} Script`}
                  </span>
                </button>

                <button
                  onClick={() => handleGenerateScript(true)}
                  disabled={generatingPeft}
                  className="px-5 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-xl text-xs font-medium flex items-center gap-2 transition"
                >
                  <Database className="w-4 h-4 text-amber-400" />
                  <span>Save Script to scripts/llm/</span>
                </button>
              </div>

              {/* Script Result Drawer */}
              {peftResult && (
                <div className="space-y-3 pt-4 border-t border-zinc-800">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-300 flex items-center gap-2">
                      <Code2 className="w-4 h-4 text-amber-400" />
                      <span>Generated Training Script ({fineTuningFramework.toUpperCase()}):</span>
                      {peftResult.saved_path && (
                        <span className="text-emerald-400 font-mono text-[11px]">
                          Saved to: {peftResult.saved_path}
                        </span>
                      )}
                    </span>

                    <button
                      onClick={() => copyToClipboard(peftResult.script_code)}
                      className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-mono flex items-center gap-1.5 transition"
                    >
                      {copiedScript ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copiedScript ? 'Copied!' : 'Copy Code'}</span>
                    </button>
                  </div>

                  <pre className="p-4 bg-zinc-950 border border-zinc-800/90 rounded-2xl text-[11px] font-mono text-zinc-300 whitespace-pre overflow-x-auto max-h-96 leading-relaxed">
                    {peftResult.script_code}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Mode 3: Hugging Face Live Hub Explorer */}
        {studioMode === 'hf_hub' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-6 space-y-6">
              <div>
                <h2 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                  <Search className="w-4 h-4 text-blue-400" />
                  <span>Hugging Face Hub Live Model Browser</span>
                </h2>
                <p className="text-xs text-zinc-400">
                  Query the official Hugging Face API directly to discover, inspect, and select the highest-performing models without guessing.
                </p>
              </div>

              {/* Search Bar & Filters */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex-1 relative min-w-[280px]">
                  <Search className="w-4 h-4 text-zinc-400 absolute left-3.5 top-3 pointer-events-none" />
                  <input
                    type="text"
                    value={hfSearchQuery}
                    onChange={(e) => {
                      setHfSearchQuery(e.target.value);
                      searchHfHub(e.target.value, hfSortFilter);
                    }}
                    placeholder="Search Hugging Face models (e.g. qwen2.5, llama-3.2, deberta, phi-4)..."
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl pl-10 pr-4 py-2.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="flex items-center gap-1.5 bg-zinc-950 border border-zinc-800 p-1 rounded-xl text-xs">
                  <button
                    onClick={() => {
                      setHfSortFilter('downloads');
                      searchHfHub(hfSearchQuery, 'downloads');
                    }}
                    className={`px-3 py-1 rounded-lg transition ${
                      hfSortFilter === 'downloads' ? 'bg-blue-600 text-white font-semibold' : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    📥 Downloads
                  </button>
                  <button
                    onClick={() => {
                      setHfSortFilter('trendingScore');
                      searchHfHub(hfSearchQuery, 'trendingScore');
                    }}
                    className={`px-3 py-1 rounded-lg transition ${
                      hfSortFilter === 'trendingScore' ? 'bg-blue-600 text-white font-semibold' : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    🔥 Trending
                  </button>
                  <button
                    onClick={() => {
                      setHfSortFilter('likes');
                      searchHfHub(hfSearchQuery, 'likes');
                    }}
                    className={`px-3 py-1 rounded-lg transition ${
                      hfSortFilter === 'likes' ? 'bg-blue-600 text-white font-semibold' : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    ❤️ Likes
                  </button>
                </div>
              </div>

              {/* Models Result Grid */}
              {loadingHfModels ? (
                <div className="py-16 text-center text-xs text-zinc-400 flex items-center justify-center gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin text-blue-400" />
                  <span>Fetching models from Hugging Face Hub API...</span>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {hfModels.map((m) => (
                    <div
                      key={m.id}
                      className="p-4 bg-zinc-950/70 border border-zinc-800/80 rounded-xl space-y-3 hover:border-zinc-700 transition flex flex-col justify-between"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-start justify-between gap-2">
                          <span className="font-mono text-xs font-bold text-zinc-200 break-all leading-snug">
                            {m.id}
                          </span>
                          <a
                            href={`https://huggingface.co/${m.id}`}
                            target="_blank"
                            rel="noreferrer"
                            className="text-zinc-500 hover:text-blue-400 transition shrink-0"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        </div>

                        <div className="flex items-center gap-3 text-[10px] text-zinc-400 font-mono">
                          <span className="flex items-center gap-1">
                            <Download className="w-3 h-3 text-emerald-400" />
                            <span>{m.downloads.toLocaleString()}</span>
                          </span>
                          <span className="flex items-center gap-1">
                            <ThumbsUp className="w-3 h-3 text-rose-400" />
                            <span>{m.likes.toLocaleString()}</span>
                          </span>
                          {m.is_unsloth && (
                            <span className="px-1.5 py-0.2 bg-amber-500/20 text-amber-300 rounded font-semibold">
                              🦥 Unsloth
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Action buttons */}
                      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-zinc-800/60">
                        <button
                          onClick={() => {
                            setPeftModelName(m.id);
                            setStudioMode('unsloth');
                          }}
                          className="px-2 py-1.5 bg-amber-950/40 hover:bg-amber-900/60 border border-amber-800/40 text-amber-300 rounded-lg text-[11px] font-semibold transition text-center"
                        >
                          Use in LoRA
                        </button>
                        <button
                          onClick={() => {
                            setCustomModelInput(m.id);
                            setIsCustomModel(true);
                            setStudioMode('prompting');
                          }}
                          className="px-2 py-1.5 bg-purple-950/40 hover:bg-purple-900/60 border border-purple-800/40 text-purple-300 rounded-lg text-[11px] font-semibold transition text-center"
                        >
                          Use in Prompt
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* API Key Configuration Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-md bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2">
                <Key className="w-4 h-4 text-purple-400" />
                <h3 className="text-sm font-bold text-zinc-100">Set {keyToConfigure}</h3>
              </div>
              <button
                onClick={() => setShowKeyModal(false)}
                className="text-zinc-500 hover:text-zinc-300 text-sm"
              >
                ✕
              </button>
            </div>

            <p className="text-xs text-zinc-400">
              This API key will be saved securely to your local <code className="text-purple-300">.env</code> file and activated in the running process.
            </p>

            <form onSubmit={handleSaveApiKey} className="space-y-4">
              <input
                type="password"
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder={`Paste your ${keyToConfigure} here...`}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-mono text-zinc-200 focus:outline-none focus:border-purple-500"
              />

              {keySuccessMsg && (
                <div className="text-xs text-emerald-400 font-medium flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>{keySuccessMsg}</span>
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowKeyModal(false)}
                  className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-xs"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={savingKey || !apiKeyInput.trim()}
                  className="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-semibold"
                >
                  {savingKey ? 'Saving...' : 'Save Key'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
