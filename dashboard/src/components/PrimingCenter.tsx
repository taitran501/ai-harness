import React, { useState, useEffect } from 'react';
import {
  Copy,
  Check,
  Terminal,
  Cpu,
  Info,
  Play,
  RotateCw,
} from 'lucide-react';
import { TaskInfo } from '../types/experiment';
import { api } from '../services/api';

interface PrimingCenterProps {
  initialPrompt?: string;
  taskInfo?: TaskInfo | null;
}

export const PrimingCenter: React.FC<PrimingCenterProps> = ({ initialPrompt = '', taskInfo }) => {
  const [selectedIde, setSelectedIde] = useState<'antigravity' | 'codex'>('antigravity');
  const [selectedIntent, setSelectedIntent] = useState<'attach' | 'iterate'>('iterate');
  const [promptText, setPromptText] = useState<string>(initialPrompt);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Fetch prompt whenever selected IDE or intent changes
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    api
      .getPrimingPrompt(selectedIde, selectedIntent)
      .then((res) => {
        if (isMounted) {
          setPromptText(res.prompt);
        }
      })
      .catch((err) => {
        console.error('Failed to load priming prompt for IDE/intent:', selectedIde, selectedIntent, err);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [selectedIde, selectedIntent]);

  const handleCopy = () => {
    navigator.clipboard.writeText(promptText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const charCount = promptText.length;
  const tokenEst = Math.round(charCount / 4);

  return (
    <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-4">
      {/* Top Header & Copy Action */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 tracking-tight flex items-center gap-2">
            <span>Agent Prompts Hub</span>
            <span className="text-[10px] bg-zinc-900 text-zinc-400 border border-zinc-800 px-1.5 py-0.2 rounded font-mono uppercase">
              {selectedIde} • {selectedIntent}
            </span>
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Copy tailored system priming prompts for Antigravity or OpenAI Codex CLI sessions.
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
          <span>{copied ? 'Copied to Clipboard!' : 'Copy Prompt'}</span>
        </button>
      </div>

      {/* Level 1: Target IDE Selector (Antigravity vs Codex) */}
      <div className="space-y-1.5">
        <label className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
          1. Select Coding Environment
        </label>
        <div className="grid grid-cols-2 gap-2.5">
          <button
            onClick={() => setSelectedIde('antigravity')}
            className={`flex items-start gap-2.5 p-3 rounded-lg border transition text-left ${
              selectedIde === 'antigravity'
                ? 'bg-zinc-900 border-indigo-500/70 text-zinc-100 shadow-sm ring-1 ring-indigo-500/30'
                : 'bg-zinc-950 hover:bg-zinc-900 border-zinc-800/80 text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <div className="p-1.5 rounded-md bg-indigo-500/10 text-indigo-400 mt-0.5 shrink-0">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-medium text-xs text-zinc-200">
                <span>Google DeepMind Antigravity</span>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-indigo-950/60 text-indigo-300 border border-indigo-800/50">
                  IDE / Pair
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-0.5 leading-relaxed">
                Antigravity IDE assistant with interactive Alignment Gate, slash commands, and UCB tree search.
              </p>
            </div>
          </button>

          <button
            onClick={() => setSelectedIde('codex')}
            className={`flex items-start gap-2.5 p-3 rounded-lg border transition text-left ${
              selectedIde === 'codex'
                ? 'bg-zinc-900 border-emerald-500/70 text-zinc-100 shadow-sm ring-1 ring-emerald-500/30'
                : 'bg-zinc-950 hover:bg-zinc-900 border-zinc-800/80 text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 mt-0.5 shrink-0">
              <Terminal className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-medium text-xs text-zinc-200">
                <span>OpenAI Codex CLI</span>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/50">
                  CLI Agent
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-0.5 leading-relaxed">
                Autonomous CLI engineer executing terminal commands, file diffs, and rapid validation loops.
              </p>
            </div>
          </button>
        </div>
      </div>

      {/* Level 2: Intent Selector (Attach / Setup vs New Chat / Iterate) */}
      <div className="space-y-1.5">
        <label className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
          2. Select Session Intent
        </label>
        <div className="grid grid-cols-2 gap-2.5">
          <button
            onClick={() => setSelectedIntent('attach')}
            className={`flex items-start gap-2.5 p-2.5 rounded-lg border transition text-left ${
              selectedIntent === 'attach'
                ? 'bg-zinc-900 border-sky-500/70 text-zinc-100 shadow-sm ring-1 ring-sky-500/30'
                : 'bg-zinc-950 hover:bg-zinc-900 border-zinc-800/80 text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <div className="p-1.5 rounded-md bg-sky-500/10 text-sky-400 mt-0.5 shrink-0">
              <Play className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-medium text-xs text-zinc-200">
                <span>Attach & Setup Pipeline</span>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-sky-950/60 text-sky-300 border border-sky-800/50">
                  Initial
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-0.5 leading-relaxed">
                First-time onboarding: Reads task specs, checks data leakage rules, and establishes exp001 baseline.
              </p>
            </div>
          </button>

          <button
            onClick={() => setSelectedIntent('iterate')}
            className={`flex items-start gap-2.5 p-2.5 rounded-lg border transition text-left ${
              selectedIntent === 'iterate'
                ? 'bg-zinc-900 border-purple-500/70 text-zinc-100 shadow-sm ring-1 ring-purple-500/30'
                : 'bg-zinc-950 hover:bg-zinc-900 border-zinc-800/80 text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <div className="p-1.5 rounded-md bg-purple-500/10 text-purple-400 mt-0.5 shrink-0">
              <RotateCw className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-medium text-xs text-zinc-200">
                <span>Continue Iteration (New Chat)</span>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-purple-950/60 text-purple-300 border border-purple-800/50">
                  Active
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-0.5 leading-relaxed">
                Ongoing development: Injects reigning champion, diagnosed failure slices, UCB branch, and Idea Graveyard.
              </p>
            </div>
          </button>
        </div>
      </div>

      {/* Context Summary Banner */}
      <div className="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg flex items-center justify-between gap-3 text-xs text-zinc-400">
        <div className="flex items-center gap-2 min-w-0">
          <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span className="truncate">
            Task: <strong className="text-zinc-200 font-mono">{taskInfo?.task_name || 'Active Task'}</strong> • Target:{' '}
            <strong className="text-zinc-200 font-mono">{taskInfo?.primary_metric || 'Score'}</strong>
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
