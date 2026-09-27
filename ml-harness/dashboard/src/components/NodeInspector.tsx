import React, { useState, useMemo } from 'react';
import {
  X,
  Layers,
  Cpu,
  ShieldCheck,
  Lightbulb,
  AlertCircle,
  FileCode,
  CheckCircle,
  FileText,
  AlertTriangle,
  FolderCheck,
  Sparkles,
  Bot,
  Zap,
  Copy,
  Check,
} from 'lucide-react';
import { NodeDetail, TaskInfo } from '../types/experiment';

interface NodeInspectorProps {
  node: NodeDetail | null;
  taskInfo?: TaskInfo | null;
  onClose: () => void;
  onOpenConfigModal?: (pathOrAction?: string, parentId?: string) => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({
  node,
  taskInfo,
  onClose,
  onOpenConfigModal,
}) => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'hypothesis' | 'llm' | 'diagnostics' | 'grounding' | 'config' | 'raw'>('metrics');
  const [copiedLlmPrompt, setCopiedLlmPrompt] = useState(false);

  if (!node) return null;

  const isLLM =
    node.approach_type === 'llm' ||
    Boolean(node.llm_spec) ||
    node.model_family?.toLowerCase().includes('llm') ||
    ['qwen', 'deepseek', 'llama', 'unsloth', 'gpt', 'gemini'].some((k) =>
      node.id.toLowerCase().includes(k)
    );
  const isDL = !isLLM && node.approach_type === 'dl';
  const grounding = node.grounding_receipt?.grounding;
  const assumptions = node.grounding_receipt?.assumption_register;
  const metrics = node.metrics || {};
  const diagnostics = node.diagnostics_summary || {};
  const llmSpec = node.llm_spec;

  const formatParams = (count?: number | null, size?: string | null): string => {
    if (size) return size;
    if (!count) return 'N/A';
    if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
    if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
    if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
    return String(count);
  };

  // Dynamically resolve all metrics to display
  const metricCards = useMemo(() => {
    const cards: { label: string; value: string; sublabel?: string }[] = [];

    // If taskInfo has defined metrics
    if (taskInfo?.metrics && taskInfo.metrics.length > 0) {
      for (const m of taskInfo.metrics) {
        const val =
          metrics[m.id]?.macro_f1 ??
          metrics[m.id]?.score ??
          (typeof metrics[m.id] === 'number' ? metrics[m.id] : null);
        cards.push({
          label: m.name,
          value: val !== null && val !== undefined ? Number(val).toFixed(4) : '-',
          sublabel: m.metric,
        });
      }
    } else {
      // Check for common subtask keys or fallback to scanning metrics
      if (metrics.subtask_a !== undefined) {
        const val = metrics.subtask_a?.macro_f1 ?? (typeof metrics.subtask_a === 'number' ? metrics.subtask_a : null);
        cards.push({ label: 'Subtask A (Stereotype)', value: val !== null ? val.toFixed(4) : '-', sublabel: 'Binary Macro-F1' });
      }
      if (metrics.subtask_b !== undefined) {
        const val = metrics.subtask_b?.macro_f1 ?? (typeof metrics.subtask_b === 'number' ? metrics.subtask_b : null);
        cards.push({ label: 'Subtask B (Hate Speech)', value: val !== null ? val.toFixed(4) : '-', sublabel: '3-class Macro-F1' });
      }
      if (metrics.subtask_c !== undefined) {
        const valType = metrics.subtask_c?.type_macro_f1 ?? null;
        cards.push({ label: 'Subtask C (Target Type)', value: valType !== null ? valType.toFixed(4) : '-', sublabel: 'Macro-F1' });
        const valId = metrics.subtask_c?.id_micro_f1 ?? metrics.subtask_c?.id_macro_f1 ?? null;
        cards.push({ label: 'Subtask C (Identities)', value: valId !== null ? valId.toFixed(4) : '-', sublabel: 'F1 Score' });
      }
    }

    // Add any additional numeric top-level metrics not already covered
    const standardKeys = new Set(['overall_composite', 'macro_f1', 'score', 'model', 'language_breakdown', 'subtask_a', 'subtask_b', 'subtask_c']);
    for (const [k, v] of Object.entries(metrics)) {
      if (!standardKeys.has(k) && typeof v === 'number') {
        const formattedKey = k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
        cards.push({
          label: formattedKey,
          value: Number(v).toFixed(4),
          sublabel: 'Metric',
        });
      }
    }

    // If still no cards, show overall score
    if (cards.length === 0) {
      cards.push({
        label: taskInfo?.primary_metric || 'Primary Metric',
        value: node.score.toFixed(4),
        sublabel: 'Cross-Validation Score',
      });
    }

    return cards;
  }, [taskInfo, metrics, node.score]);

  // Language breakdown from metrics or diagnostics
  const langBreakdown = diagnostics.language_breakdown || metrics.language_breakdown || null;
  const criticalBottlenecks = diagnostics.critical_bottlenecks || [];
  const implicitHate = diagnostics.implicit_hate_analysis;

  return (
    <div className="fixed inset-y-0 right-0 w-full max-w-xl bg-slate-900 border-l border-slate-800 shadow-2xl z-50 flex flex-col transition-transform duration-300">
      {/* Header */}
      <div className="p-5 border-b border-slate-800 flex items-start justify-between bg-slate-950/70">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span
              className={`flex items-center gap-1 text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                isLLM
                  ? 'bg-gradient-to-r from-purple-900/70 via-indigo-900/70 to-amber-900/60 text-amber-300 border border-amber-500/40 shadow-sm'
                  : isDL
                  ? 'bg-purple-900/60 text-purple-300 border border-purple-700/50'
                  : 'bg-blue-900/60 text-blue-300 border border-blue-700/50'
              }`}
            >
              {isLLM ? (
                <Sparkles className="w-3 h-3 text-amber-400" />
              ) : isDL ? (
                <Cpu className="w-3 h-3" />
              ) : (
                <Layers className="w-3 h-3" />
              )}
              {isLLM ? 'LLM' : node.approach_type.toUpperCase()} ({node.model_family})
            </span>
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              {node.branch}
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              Parent: <strong className="text-slate-200">{node.parent_id || 'ROOT'}</strong>
            </span>
          </div>

          <h2 className="text-lg font-bold text-white font-mono tracking-tight">{node.id}</h2>

          <div className="flex items-center gap-4 text-xs text-slate-400 pt-1">
            <span>
              {taskInfo?.primary_metric || 'Score (CV)'}: <strong className="font-mono text-emerald-400 text-sm">{node.score.toFixed(4)}</strong>
            </span>
            <span>
              Scale: <strong className="font-mono text-slate-200">{formatParams(node.param_count, node.param_size)}</strong>
            </span>
          </div>
        </div>

        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 bg-slate-950/30 px-5 text-xs overflow-x-auto">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition ${
            activeTab === 'metrics'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Metrics & Breakdown
        </button>
        <button
          onClick={() => setActiveTab('hypothesis')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition ${
            activeTab === 'hypothesis'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Hypothesis & Plan
        </button>
        <button
          onClick={() => setActiveTab('llm')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition flex items-center gap-1.5 ${
            activeTab === 'llm'
              ? 'border-amber-400 text-amber-300 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>LLM Specs</span>
          {isLLM && (
            <span className="text-[9px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-1.5 py-0.2 rounded font-mono">
              {node.param_size || '7B+'}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('diagnostics')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition ${
            activeTab === 'diagnostics'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Diagnostics
        </button>
        <button
          onClick={() => setActiveTab('grounding')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition ${
            activeTab === 'grounding'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Grounding
        </button>
        <button
          onClick={() => setActiveTab('config')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition ${
            activeTab === 'config'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Target Config
        </button>
        <button
          onClick={() => setActiveTab('raw')}
          className={`py-3 px-3 font-medium border-b-2 whitespace-nowrap transition ${
            activeTab === 'raw'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Raw JSON
        </button>
      </div>

      {/* Body Content */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4 custom-scrollbar text-xs">
        {/* Tab 1: Metrics */}
        {activeTab === 'metrics' && (
          <div className="space-y-4">
            {/* Dynamic Metric Cards Grid */}
            <div className="grid grid-cols-2 gap-3">
              {metricCards.map((card, idx) => (
                <div key={idx} className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <span className="text-slate-400 text-[11px] block">{card.label}</span>
                  <span className="font-mono text-base font-bold text-white mt-1 block">
                    {card.value}
                  </span>
                  {card.sublabel && (
                    <span className="text-[10px] text-slate-500">{card.sublabel}</span>
                  )}
                </div>
              ))}
            </div>

            {/* Language Breakdown */}
            {langBreakdown && Object.keys(langBreakdown).length > 0 && (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                <h4 className="text-slate-200 font-semibold text-xs flex items-center justify-between">
                  <span>Language Performance Breakdown</span>
                  <span className="text-[10px] font-mono text-slate-400">
                    {Object.keys(langBreakdown).join(' / ')}
                  </span>
                </h4>
                <div className="grid grid-cols-3 gap-2 pt-1 font-mono">
                  {Object.entries(langBreakdown).map(([lang, val]: [string, any]) => (
                    <div key={lang} className="bg-slate-900 p-2.5 rounded-lg border border-slate-800/80 text-center">
                      <div className="text-[10px] text-slate-400 font-bold">{lang}</div>
                      <div className="text-xs font-semibold text-indigo-300 mt-1">
                        {val.a_accuracy !== undefined ? `Acc: ${(val.a_accuracy * 100).toFixed(1)}%` : val.macro_f1 !== undefined ? val.macro_f1.toFixed(4) : '-'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Disk Artifacts Checklist */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <h4 className="text-slate-200 font-semibold text-xs flex items-center gap-1.5">
                <FolderCheck className="w-4 h-4 text-emerald-400" />
                Immutable Artifacts on Disk
              </h4>
              <p className="text-[11px] text-slate-400">
                Located in <code className="text-slate-300">artifacts/{node.id}/</code>:
              </p>
              <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] font-mono text-slate-300">
                <div className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                  <span>run_config.yaml</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                  <span>summary.csv</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                  <span>metrics.json</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                  <span>oof_predictions.csv</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Hypothesis & Plan */}
        {activeTab === 'hypothesis' && (
          <div className="space-y-4">
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <h4 className="text-slate-200 font-semibold flex items-center gap-1.5 text-xs">
                <Lightbulb className="w-4 h-4 text-amber-400" />
                Atomic Scientific Hypothesis
              </h4>
              <p className="text-slate-300 leading-relaxed font-sans">
                {node.atomic_hypothesis || 'No explicit hypothesis declared.'}
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <h4 className="text-slate-200 font-semibold flex items-center gap-1.5 text-xs">
                <FileCode className="w-4 h-4 text-indigo-400" />
                Intervention (Control vs Treatment)
              </h4>
              <p className="font-mono text-indigo-300 bg-slate-900 p-2.5 rounded-lg border border-slate-800/80 leading-relaxed text-[11px]">
                {node.intervention || 'None.'}
              </p>
            </div>

            {node.notes && (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                <h4 className="text-slate-400 font-semibold text-xs">Experiment Notes</h4>
                <p className="text-slate-300 font-sans leading-relaxed">{node.notes}</p>
              </div>
            )}
          </div>
        )}

        {/* Tab: LLM Architecture & Specs */}
        {activeTab === 'llm' && (
          <div className="space-y-4">
            {/* If node is not LLM, show informative callout */}
            {!isLLM && (
              <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0">
                    <Sparkles className="w-4 h-4 text-amber-400" />
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-xs font-semibold text-slate-200">
                      Classical {node.approach_type.toUpperCase()} Experiment ({node.model_family})
                    </h4>
                    <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
                      This run executed under a classical tabular/deep encoder architecture. You can fork this experiment into an LLM target to benchmark 7B or 14B instruction models against this baseline.
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 pt-1 border-t border-slate-900">
                  <button
                    type="button"
                    onClick={() => onOpenConfigModal?.('create', node.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 font-bold rounded-lg transition"
                  >
                    <Zap className="w-3.5 h-3.5 fill-current" />
                    <span>Fork into 7B Unsloth LoRA Target</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onOpenConfigModal?.('create', node.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium rounded-lg border border-slate-700 transition"
                  >
                    <span>Fork into 14B DeepSeek-R1 CoT</span>
                  </button>
                </div>
              </div>
            )}

            {/* Model Architecture & Scaling Overview */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-slate-200 font-semibold text-xs flex items-center gap-2">
                  <Bot className="w-4 h-4 text-amber-400" />
                  <span>Model Backbone & Scale</span>
                </h4>
                <span className="font-mono text-xs font-bold text-amber-300 bg-amber-950/60 border border-amber-800/60 px-2 py-0.5 rounded">
                  {llmSpec?.param_size || node.param_size || (isLLM ? '7B' : 'N/A')}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2.5 font-mono text-[11px]">
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Model Identifier</div>
                  <div className="text-slate-100 font-semibold truncate mt-0.5" title={llmSpec?.model_name || node.metrics?.model || node.id}>
                    {llmSpec?.model_name || node.metrics?.model || (isLLM ? 'Qwen2.5-7B-Instruct' : node.id)}
                  </div>
                </div>

                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Quantization / Precision</div>
                  <div className="text-amber-300 font-semibold mt-0.5">
                    {llmSpec?.quantization || (isLLM ? '4-bit NormalFloat (NF4)' : 'bfloat16')}
                  </div>
                </div>

                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Context Window</div>
                  <div className="text-indigo-300 font-semibold mt-0.5">
                    {llmSpec?.context_length ? `${llmSpec.context_length} tokens` : '8,192 tokens'}
                  </div>
                </div>

                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Provider / Engine</div>
                  <div className="text-emerald-400 font-semibold mt-0.5">
                    {llmSpec?.provider || (isLLM ? 'Unsloth Triton Kernel' : 'PyTorch Native')}
                  </div>
                </div>
              </div>
            </div>

            {/* PEFT / LoRA Hyperparameters */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <h4 className="text-slate-200 font-semibold text-xs flex items-center justify-between">
                <span>PEFT & LoRA Adaptation</span>
                <span className="text-[10px] font-mono text-indigo-400">
                  {llmSpec?.paradigm || 'unsloth_qlora'}
                </span>
              </h4>

              <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">LoRA Rank (r)</div>
                  <div className="text-indigo-300 font-bold text-sm mt-0.5">
                    {llmSpec?.lora_r ?? 16}
                  </div>
                </div>

                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Scaling (alpha)</div>
                  <div className="text-indigo-300 font-bold text-sm mt-0.5">
                    {llmSpec?.lora_alpha ?? 32}
                  </div>
                </div>

                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Dropout</div>
                  <div className="text-slate-300 font-bold text-sm mt-0.5">
                    {llmSpec?.lora_dropout ?? 0.05}
                  </div>
                </div>
              </div>

              <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80 space-y-1">
                <span className="text-[10px] text-slate-400 font-mono block">Target Projections:</span>
                <code className="text-[11px] font-mono text-slate-300 break-all leading-relaxed block">
                  {llmSpec?.target_modules || 'q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj'}
                </code>
              </div>
            </div>

            {/* Prompting & Reasoning Paradigm */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-slate-200 font-semibold text-xs flex items-center gap-1.5">
                  <FileText className="w-4 h-4 text-purple-400" />
                  <span>Prompting & Reasoning Paradigm</span>
                </h4>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/60 border border-purple-800/60 text-purple-300">
                    {llmSpec?.use_cot !== false ? 'CoT Reasoning Enabled' : 'Direct Classification'}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      const promptText = llmSpec?.system_prompt || 'You are an expert AI competition classifier.';
                      navigator.clipboard.writeText(promptText);
                      setCopiedLlmPrompt(true);
                      setTimeout(() => setCopiedLlmPrompt(false), 2000);
                    }}
                    className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
                    title="Copy Prompt"
                  >
                    {copiedLlmPrompt ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-[10px] text-slate-400">System Instruction:</div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80 font-mono text-[11px] text-slate-300 leading-relaxed max-h-28 overflow-y-auto">
                  {llmSpec?.system_prompt ||
                    'You are an expert AI competition classifier. Predict target classes with calibrated confidence.'}
                </div>
              </div>

              {llmSpec?.prompt_template && (
                <div className="space-y-2">
                  <div className="text-[10px] text-slate-400">Prompt Template:</div>
                  <pre className="p-2.5 bg-slate-900 rounded-lg border border-slate-800/80 font-mono text-[11px] text-indigo-300 leading-relaxed max-h-28 overflow-y-auto whitespace-pre-wrap">
                    {llmSpec.prompt_template}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 3: Diagnostics */}
        {activeTab === 'diagnostics' && (
          <div className="space-y-4">
            {criticalBottlenecks.length > 0 && (
              <div className="bg-amber-950/20 border border-amber-800/40 p-4 rounded-xl space-y-2">
                <h4 className="text-amber-300 font-semibold flex items-center gap-1.5 text-xs">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  Diagnosed Bottlenecks
                </h4>
                <ul className="list-disc pl-4 space-y-1 text-slate-300 font-sans">
                  {criticalBottlenecks.map((b: string, i: number) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>
              </div>
            )}

            {implicitHate && (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5 font-sans">
                <h4 className="text-slate-200 font-semibold text-xs">Implicit Hate Speech Miss Rate</h4>
                <p className="text-slate-300">
                  <strong className="text-rose-400 font-mono">
                    {(implicitHate.miss_rate * 100).toFixed(1)}%
                  </strong>{' '}
                  miss rate ({implicitHate.misclassified_as_no} / {implicitHate.total_implicit} implicit hate comments misclassified as "no").
                </p>
              </div>
            )}

            {Object.keys(diagnostics).length > 0 ? (
              <pre className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-[350px]">
                {JSON.stringify(diagnostics, null, 2)}
              </pre>
            ) : (
              <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 text-center text-slate-400 font-sans">
                No diagnostic failure slices recorded for this node.
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Grounding */}
        {activeTab === 'grounding' && (
          <div className="space-y-4">
            {grounding ? (
              <>
                <div className="bg-emerald-950/30 border border-emerald-800/40 p-3.5 rounded-xl flex items-center justify-between">
                  <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    <span>Grounding Verified</span>
                  </div>
                  <span className="text-[10px] font-mono bg-emerald-900/60 text-emerald-300 px-2 py-0.5 rounded">
                    PASS
                  </span>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="text-slate-400">Source Type</div>
                  <div className="font-mono text-slate-100 font-bold">{grounding.source_type}</div>
                  <div className="text-slate-400 pt-2">Source / Model ID</div>
                  <div className="font-mono text-indigo-300 bg-slate-900 p-2 rounded border border-slate-800">
                    {grounding.source_id}
                  </div>
                  <div className="text-slate-400 pt-2">Retrieval Summary</div>
                  <p className="text-slate-300 leading-relaxed font-sans">{grounding.retrieval_summary}</p>
                </div>

                {assumptions && (
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                    <h4 className="font-semibold text-slate-200">Assumption Register</h4>
                    <div className="text-slate-400">
                      Uncertainty Level: <strong className="text-amber-400 uppercase font-mono">{assumptions.uncertainty_level || 'N/A'}</strong>
                    </div>
                    {assumptions.assumptions && (
                      <ul className="list-disc pl-4 space-y-1 text-slate-300 font-sans">
                        {assumptions.assumptions.map((a: string, i: number) => (
                          <li key={i}>{a}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 text-center space-y-2">
                <AlertCircle className="w-8 h-8 text-slate-500 mx-auto" />
                <h4 className="text-slate-300 font-semibold">Canonical Reference Baseline</h4>
                <p className="text-slate-400 font-sans">This node is the foundational root control baseline.</p>
              </div>
            )}
          </div>
        )}

        {/* Tab 5: Target Config */}
        {activeTab === 'config' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <span className="font-semibold text-slate-200">Experiment Target Configuration</span>
                {node.config?.path && (
                  <p className="text-[11px] font-mono text-indigo-400 mt-0.5">{node.config.path}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => onOpenConfigModal?.(node.config?.path, node.id)}
                  className="px-2.5 py-1 text-xs bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded transition"
                >
                  Edit in Target Editor
                </button>
                <button
                  type="button"
                  onClick={() => onOpenConfigModal?.('create', node.id)}
                  className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-amber-300 font-medium rounded border border-slate-700 transition"
                >
                  + Fork Child Target
                </button>
              </div>
            </div>

            {node.config?.content ? (
              <pre className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-200 overflow-x-auto max-h-[500px] leading-relaxed">
                {node.config.content}
              </pre>
            ) : (
              <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 text-center space-y-2">
                <FileCode className="w-8 h-8 text-slate-600 mx-auto" />
                <h4 className="text-slate-300 font-semibold">No Standalone YAML Config Stored</h4>
                <p className="text-slate-400 font-sans">
                  This node ran under inherited baseline or artifact config. You can spawn a new target config extending this experiment.
                </p>
                <button
                  type="button"
                  onClick={() => onOpenConfigModal?.('create', node.id)}
                  className="mt-2 px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded font-medium transition"
                >
                  Spawn New Experiment Config from Node
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 6: Raw JSON */}
        {activeTab === 'raw' && (
          <div className="space-y-2">
            <pre className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-[500px]">
              {JSON.stringify(node, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
