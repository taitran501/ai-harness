import React, { useState, useEffect } from 'react';
import {
  Copy,
  Check,
  Terminal,
  Cpu,
  GitBranch,
  Zap,
  Info,
  Layers,
} from 'lucide-react';
import { PromptTemplateInfo, TaskInfo } from '../types/experiment';
import { api } from '../services/api';

interface PrimingCenterProps {
  initialPrompt?: string;
  taskInfo?: TaskInfo | null;
}

const DEFAULT_TEMPLATES: PromptTemplateInfo[] = [
  {
    id: 'codex',
    name: 'OpenAI Codex CLI',
    badge: 'CLI Agent',
    description:
      'Rigorous execution loop (Plan -> Single Atomic Change -> CV Validation -> Git Diff) following OpenAI Codex & AGENTS.md conventions.',
  },
  {
    id: 'antigravity',
    name: 'Antigravity Agent',
    badge: 'Pair Programmer',
    description:
      'DeepMind Antigravity framework injecting reigning champion, diagnostic bottlenecks, and mandatory 4-step Acknowledgment Gate.',
  },
  {
    id: 'aide',
    name: 'AIDE Solution Tree',
    badge: 'Tree Search',
    description:
      'Tailored for iterative MCTS/UCB tree search, hypothesis verification, and MMES statistical audit checks.',
  },
  {
    id: 'minimal',
    name: 'Minimal Direct Patch',
    badge: 'Fast Probe',
    description:
      'High-density concise prompt for quick debugging, smoke testing, and single-variable probing.',
  },
];

export const PrimingCenter: React.FC<PrimingCenterProps> = ({ initialPrompt = '', taskInfo }) => {
  const [templates, setTemplates] = useState<PromptTemplateInfo[]>(DEFAULT_TEMPLATES);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('codex');
  const [promptText, setPromptText] = useState<string>(initialPrompt);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Fetch templates from API
  useEffect(() => {
    let isMounted = true;
    api
      .getPromptTemplates()
      .then((res) => {
        if (isMounted && res.templates && res.templates.length > 0) {
          setTemplates(res.templates);
        }
      })
      .catch((err) => {
        console.warn('Could not fetch templates list from API, using defaults:', err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Fetch prompt when selected template changes
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    api
      .getPrimingPrompt(selectedTemplate)
      .then((res) => {
        if (isMounted) {
          setPromptText(res.prompt);
        }
      })
      .catch((err) => {
        console.error('Failed to load priming prompt for template:', selectedTemplate, err);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [selectedTemplate]);

  const handleCopy = () => {
    navigator.clipboard.writeText(promptText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const currentTemplate =
    templates.find((t) => t.id === selectedTemplate) || templates[0];

  const getTemplateIcon = (id: string) => {
    switch (id) {
      case 'codex':
        return <Terminal className="w-4 h-4 text-emerald-400" />;
      case 'antigravity':
        return <Cpu className="w-4 h-4 text-indigo-400" />;
      case 'aide':
        return <GitBranch className="w-4 h-4 text-sky-400" />;
      case 'minimal':
        return <Zap className="w-4 h-4 text-amber-400" />;
      default:
        return <Terminal className="w-4 h-4 text-slate-400" />;
    }
  };

  const charCount = promptText.length;
  const tokenEst = Math.round(charCount / 4);

  return (
    <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
      {/* Top Header & Copy */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 tracking-tight flex items-center gap-2">
            <span>Agent Prompts</span>
            <span className="text-[10px] bg-zinc-900 text-zinc-400 border border-zinc-800 px-1.5 py-0.2 rounded font-mono">
              {currentTemplate.name}
            </span>
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Copy the structured priming prompt below into your agent chat (Codex, Antigravity, or AIDE).
          </p>
        </div>

        {/* Copy Button */}
        <button
          onClick={handleCopy}
          disabled={loading || !promptText}
          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition active:scale-95 shadow-sm ${
            copied
              ? 'bg-emerald-600 text-white'
              : 'bg-zinc-100 hover:bg-white text-zinc-950 disabled:opacity-50'
          }`}
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied to Clipboard!' : `Copy Prompt`}</span>
        </button>
      </div>

      {/* Template Switcher Tabs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {templates.map((tmpl) => {
          const isActive = selectedTemplate === tmpl.id;
          return (
            <button
              key={tmpl.id}
              onClick={() => setSelectedTemplate(tmpl.id)}
              className={`flex flex-col text-left p-2.5 rounded-lg border transition ${
                isActive
                  ? 'bg-zinc-900 border-zinc-700 text-zinc-100 shadow-sm'
                  : 'bg-zinc-950 hover:bg-zinc-900 border-zinc-800/80 text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <div className="flex items-center justify-between gap-1 mb-1">
                <div className="flex items-center gap-1.5 font-medium text-xs text-zinc-200">
                  {getTemplateIcon(tmpl.id)}
                  <span>{tmpl.name}</span>
                </div>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-zinc-900 text-zinc-400 border border-zinc-800">
                  {tmpl.badge}
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 line-clamp-2 leading-relaxed">
                {tmpl.description}
              </p>
            </button>
          );
        })}
      </div>

      {/* Context Summary Banner */}
      <div className="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg flex items-center justify-between gap-3 text-xs text-zinc-400">
        <div className="flex items-center gap-2 min-w-0">
          <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span className="truncate">
            Task: <strong className="text-zinc-200 font-mono">{taskInfo?.task_name || 'Active Task'}</strong> • Metric: <strong className="text-zinc-200 font-mono">{taskInfo?.primary_metric || 'Score'}</strong>
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] shrink-0 text-zinc-500">
          <span>{charCount.toLocaleString()} chars</span>
          <span>•</span>
          <span>~{tokenEst.toLocaleString()} tokens</span>
        </div>
      </div>

      {/* Textarea Viewport */}
      <div className="relative">
        {loading ? (
          <div className="w-full h-[460px] bg-zinc-900/80 border border-zinc-800 rounded-lg flex items-center justify-center text-zinc-500 text-xs font-mono">
            Generating agent prompt...
          </div>
        ) : (
          <textarea
            readOnly
            value={promptText}
            className="w-full h-[460px] bg-zinc-900/70 border border-zinc-800 rounded-lg p-3.5 font-mono text-xs text-zinc-300 leading-relaxed custom-scrollbar focus:outline-none focus:border-zinc-700 select-all"
          />
        )}
      </div>
    </div>
  );
};
