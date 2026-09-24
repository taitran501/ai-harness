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
} from 'lucide-react';
import { api } from '../services/api';
import {
  TaskInfo,
  LLMProviderInfo,
  LLMSampleDataResponse,
  LLMTestPromptResponse,
  LLMPeftGenerateResponse,
} from '../types/experiment';

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
  // Mode selection: Prompting vs PEFT
  const [studioMode, setStudioMode] = useState<'prompting' | 'peft'>('prompting');

  // Providers & keys state
  const [providers, setProviders] = useState<LLMProviderInfo[]>([]);
  const [selectedProviderId, setSelectedProviderId] = useState<string>('gemini');
  const [selectedModel, setSelectedModel] = useState<string>('gemini-2.0-flash');
  const [loadingProviders, setLoadingProviders] = useState(false);

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

  // QLoRA PEFT state
  const [peftModelName, setPeftModelName] = useState('Qwen/Qwen2.5-7B-Instruct');
  const [peftTextCol, setPeftTextCol] = useState('text');
  const [peftTargetCol, setPeftTargetCol] = useState('target');
  const [peftLoraR, setPeftLoraR] = useState(16);
  const [peftLoraAlpha, setPeftLoraAlpha] = useState(32);
  const [peftEpochs, setPeftEpochs] = useState(3);
  const [peftBatchSize, setPeftBatchSize] = useState(4);
  const [peftLR, setPeftLR] = useState(0.0002);
  const [generatingPeft, setGeneratingPeft] = useState(false);
  const [peftResult, setPeftResult] = useState<LLMPeftGenerateResponse | null>(null);
  const [copiedScript, setCopiedScript] = useState(false);

  // Experiment Registration State
  const [registeringExp, setRegisteringExp] = useState(false);
  const [registeredExpMsg, setRegisteredExpMsg] = useState<string | null>(null);

  // Fetch providers
  const loadProviders = async () => {
    setLoadingProviders(true);
    try {
      const data = await api.getLLMProviders();
      setProviders(data.providers);
      const current = data.providers.find((p) => p.id === selectedProviderId);
      if (current && !current.models.includes(selectedModel)) {
        setSelectedModel(current.default_model);
      }
    } catch (err) {
      console.warn('Failed to load LLM providers:', err);
    } finally {
      setLoadingProviders(false);
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
  }, []);

  // Update models when provider changes
  const handleSelectProvider = (provId: string) => {
    setSelectedProviderId(provId);
    const p = providers.find((x) => x.id === provId);
    if (p) {
      setSelectedModel(p.default_model);
    }
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

    // Build row data
    let rowData: Record<string, any> = {};
    if (useCustomInput) {
      rowData = { text: customInputText || 'Sample test input' };
    } else if (sampleData && sampleData.rows.length > selectedSampleIndex) {
      rowData = sampleData.rows[selectedSampleIndex];
    } else {
      rowData = { text: 'Test input comment' };
    }

    // Extract valid classes from taskInfo
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
        model: selectedModel,
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

  // Generate PEFT Script
  const handleGeneratePeft = async (saveToDisk = false) => {
    setGeneratingPeft(true);
    try {
      const res = await api.generateLLMPeftScript({
        model_name: peftModelName,
        train_path: taskInfo?.data_config?.train_path || 'dataset/train.csv',
        output_dir: `artifacts/exp_qlora_${peftModelName.split('/')[1]?.toLowerCase() || 'model'}`,
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
    } catch (err: any) {
      alert(`PEFT generation failed: ${err.message}`);
    } finally {
      setGeneratingPeft(false);
    }
  };

  // Register as Solution Tree Experiment
  const handleRegisterAsExperiment = async () => {
    setRegisteringExp(true);
    setRegisteredExpMsg(null);
    try {
      const nextIdRes = await api.getNextExpId('model', `llm_${selectedModel.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}`);
      const expId = nextIdRes.next_id;

      const payload = {
        exp_id: expId,
        parent_id: 'exp001_baseline_tfidf',
        branch: 'model',
        approach_type: 'llm',
        model_family: 'llm_api',
        model_name: selectedModel,
        provider: selectedProviderId,
        hypothesis: `Zero-shot / Few-shot ICL with ${selectedModel} (CoT=${useCot}) provides high contextual generalization.`,
        intervention: `Prompting pipeline using ${selectedProviderId}:${selectedModel} with temperature=${temperature}.`,
        problem_addressed: 'Complex implicit nuances require large language model pre-training and reasoning.',
        source_type: 'api_provider',
        source_id: `${selectedProviderId}/${selectedModel}`,
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
              <h1 className="text-base font-bold text-zinc-100">LLM Studio & Prompting Playground</h1>
              <span className="px-2 py-0.5 text-[10px] font-mono bg-purple-950/60 border border-purple-700/50 text-purple-300 rounded-full">
                Zero-Shot • Few-Shot • CoT • QLoRA
              </span>
            </div>
            <p className="text-xs text-zinc-400">
              Universal multi-provider LLM workbench for APIs (Gemini, Claude, GPT, DeepSeek, Groq) and local PEFT fine-tuning.
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
            onClick={() => setStudioMode('peft')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              studioMode === 'peft'
                ? 'bg-purple-600 text-white shadow'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Flame className="w-3.5 h-3.5" />
            <span>QLoRA / PEFT Script Studio</span>
          </button>
        </div>
      </div>

      {/* Main Workspace Body */}
      <div className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
        {studioMode === 'prompting' ? (
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
                  <button
                    onClick={loadProviders}
                    disabled={loadingProviders}
                    className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1 transition"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loadingProviders ? 'animate-spin' : ''}`} />
                    <span>Refresh</span>
                  </button>
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
                          {p.name} {p.is_local ? '(Local)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Model Selection */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-400">Model Name</label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-medium text-zinc-200 font-mono focus:outline-none focus:border-purple-500"
                    >
                      {currentProvider?.models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      )) || <option value={selectedModel}>{selectedModel}</option>}
                    </select>
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
                        ? `${currentProvider.env_key_name || 'Provider'} is configured`
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

                  {/* Chain of Thought Toggle */}
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
                    {/* Available Column Insertion Chips */}
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
                      <span>Running Inference on {selectedModel}...</span>
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
                    {/* Extracted Label Badge */}
                    <div className="p-3 bg-zinc-950 border border-zinc-800 rounded-xl flex items-center justify-between">
                      <span className="text-xs text-zinc-400 font-medium">Extracted Prediction:</span>
                      <span className="px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono font-bold text-xs rounded-lg">
                        {testResult.extracted.parsed_label || 'UNKNOWN'}
                      </span>
                    </div>

                    {/* Step-by-Step Reasoning */}
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

                    {/* Full Raw Output Accordion */}
                    <div className="p-3 bg-zinc-950/70 border border-zinc-800 rounded-xl space-y-1">
                      <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">
                        Raw Text Response:
                      </span>
                      <pre className="text-[11px] text-zinc-300 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto leading-relaxed">
                        {testResult.raw_text}
                      </pre>
                    </div>

                    {/* Register in Solution Tree Button */}
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
        ) : (
          /* Mode 2: QLoRA & PEFT Script Studio */
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-6 space-y-6">
              <div>
                <h2 className="text-sm font-bold text-zinc-100">QLoRA / PEFT Parameter-Efficient Fine-Tuning</h2>
                <p className="text-xs text-zinc-400">
                  Generate high-performance standalone PyTorch scripts utilizing 4-bit NF4 quantization (bitsandbytes), PEFT LoRA, and Hugging Face TRL SFTTrainer.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">HuggingFace Model Hub ID</label>
                  <input
                    type="text"
                    value={peftModelName}
                    onChange={(e) => setPeftModelName(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs font-mono text-zinc-200"
                    placeholder="e.g. Qwen/Qwen2.5-7B-Instruct"
                  />
                </div>

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
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-zinc-950/70 border border-zinc-800/80 rounded-xl text-xs font-mono">
                <div>
                  <label className="text-[10px] text-zinc-500">LoRA Rank (r)</label>
                  <select
                    value={peftLoraR}
                    onChange={(e) => setPeftLoraR(parseInt(e.target.value))}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-200 mt-1"
                  >
                    <option value={8}>r = 8</option>
                    <option value={16}>r = 16 (Recommended)</option>
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
                    <option value={32}>alpha = 32</option>
                    <option value={64}>alpha = 64</option>
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
                  onClick={() => handleGeneratePeft(false)}
                  disabled={generatingPeft}
                  className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition shadow-lg shadow-purple-950/50"
                >
                  <FileCode className="w-4 h-4" />
                  <span>{generatingPeft ? 'Generating...' : 'Generate QLoRA Script'}</span>
                </button>

                <button
                  onClick={() => handleGeneratePeft(true)}
                  disabled={generatingPeft}
                  className="px-5 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-xl text-xs font-medium flex items-center gap-2 transition"
                >
                  <Database className="w-4 h-4 text-purple-400" />
                  <span>Save to scripts/llm/</span>
                </button>
              </div>

              {/* Script Result Drawer */}
              {peftResult && (
                <div className="space-y-3 pt-4 border-t border-zinc-800">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-300 flex items-center gap-2">
                      <Code2 className="w-4 h-4 text-purple-400" />
                      <span>Generated PyTorch + PEFT Script:</span>
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
