import React, { memo, useEffect } from 'react';
import { Handle, Position, NodeProps, Node, useUpdateNodeInternals } from '@xyflow/react';
import { AlertTriangle, Sparkles } from 'lucide-react';
import { GraphNode } from '../types/experiment';
import { useLanguage } from '../constants/i18n';

export type ExperimentNodeType = Node<GraphNode & Record<string, any>, 'experimentNode'>;

export const CustomExperimentNode = memo(({ id, data, selected }: NodeProps<ExperimentNodeType>) => {
  const { t } = useLanguage();
  const updateNodeInternals = useUpdateNodeInternals();
  const isBest = Boolean(data?.is_best);
  const isBuggy = Boolean(data?.is_buggy || data?.status === 'BUGGY');
  const isLR = data?.layoutDirection === 'LR';
  const delta = data?.parent_delta;


  // Ensure React Flow updates handle positions whenever layout direction toggles
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, isLR, updateNodeInternals]);

  const branchStyles: Record<string, string> = {
    baseline: 'bg-slate-800 text-slate-300 border-slate-700',
    feat: 'bg-sky-950/70 text-sky-300 border-sky-800/60',
    model: 'bg-purple-950/70 text-purple-300 border-purple-800/60',
    loss: 'bg-amber-950/70 text-amber-300 border-amber-800/60',
    mtl: 'bg-indigo-950/70 text-indigo-300 border-indigo-800/60',
    ens: 'bg-emerald-950/70 text-emerald-300 border-emerald-800/60',
  };

  const branchKey = (data?.branch || '').toLowerCase();
  const branchStyle = branchStyles[branchKey] || 'bg-slate-800 text-slate-300 border-slate-700';

  const hasScore = typeof data?.score === 'number' && !isNaN(data.score);
  const hasDelta = typeof delta === 'number' && !isNaN(delta);

  // Guard against floating-point epsilon artifacts (e.g. -0.0000 displaying in red)
  const isPositiveDelta = hasDelta && delta >= 0.00005;
  const isNegativeDelta = hasDelta && delta <= -0.00005;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-selected={selected}
      className={`relative w-[240px] h-[80px] rounded-lg border backdrop-blur-sm p-2.5 flex flex-col justify-between transition-colors duration-150 cursor-pointer shadow-lg select-none ${
        selected
          ? isBest
            ? 'bg-slate-900/95 border-amber-400 ring-2 ring-amber-400/50 shadow-amber-500/20'
            : isBuggy
            ? 'bg-slate-900/95 border-rose-500 ring-2 ring-rose-500/40 shadow-rose-500/20'
            : 'bg-slate-900/95 border-indigo-500 ring-2 ring-indigo-500/40 shadow-indigo-500/20'
          : isBest
          ? 'bg-slate-900/95 border-amber-500/70 shadow-amber-500/5 hover:border-amber-400'
          : isBuggy
          ? 'bg-slate-900/90 border-rose-800/70 hover:border-rose-700'
          : 'bg-slate-900/90 hover:bg-slate-900 border-slate-800/80 hover:border-slate-700'
      }`}
    >
      {/* Target Handle (Incoming) */}
      <Handle
        key={isLR ? 'target-lr' : 'target-tb'}
        type="target"
        position={isLR ? Position.Left : Position.Top}
        isConnectable={false}
        className="!w-2 !h-2 !rounded-full !bg-indigo-400 !border-2 !border-slate-950 pointer-events-none"
      />

      {/* Header Row: Model / Branch Tag + Delta */}
      <div className="flex items-center justify-between gap-1.5 min-w-0">
        <div className="flex items-center gap-1 min-w-0 flex-1">
          <span
            className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded border leading-none shrink-0 ${branchStyle}`}
          >
            {data?.branch || 'exp'}
          </span>
          {data?.approach_type === 'llm' && (
            <span className="text-[8px] bg-amber-500/20 text-amber-300 border border-amber-500/40 px-1 py-0.2 rounded font-mono font-semibold">
              LLM
            </span>
          )}
          {data?.approach_type === 'dl' && (
            <span className="text-[8px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 px-1 py-0.2 rounded font-mono font-semibold">
              DL
            </span>
          )}
          {data?.approach_type === 'ml' && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-1 py-0.2 rounded font-mono font-semibold">
              ML
            </span>
          )}
          {data?.approach_type === 'hybrid' && (
            <span className="text-[8px] bg-purple-500/20 text-purple-300 border border-purple-500/40 px-1 py-0.2 rounded font-mono font-semibold">
              HYBRID
            </span>
          )}
          {data?.model_family && (
            <span
              className="text-[10px] font-mono text-slate-400 truncate leading-none min-w-0"
              title={data.model_family}
            >
              {data.model_family}
            </span>
          )}
        </div>

        {/* Delta vs parent */}
        <div className="shrink-0">
          {hasDelta ? (
            <span
              className={`text-[10px] font-mono font-semibold leading-none ${
                isPositiveDelta
                  ? 'text-emerald-400'
                  : isNegativeDelta
                  ? 'text-rose-400'
                  : 'text-slate-400'
              }`}
            >
              {isPositiveDelta
                ? `+${delta.toFixed(4)}`
                : isNegativeDelta
                ? delta.toFixed(4)
                : '0.0000'}
            </span>
          ) : (
            <span className="text-[10px] font-mono text-slate-400 leading-none">{t('node_root', 'root')}</span>
          )}
        </div>
      </div>

      {/* Experiment ID */}
      <div className="flex items-center justify-between gap-1 min-w-0">
        <span
          className="font-mono text-xs font-semibold text-slate-200 truncate leading-tight flex-1 min-w-0"
          title={data?.id || id}
        >
          {data?.id || id}
        </span>
        {isBuggy && (
          <span title={t('node_buggy_title', 'Flagged with issues/bugs')} className="shrink-0 flex items-center">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
          </span>
        )}
      </div>

      {/* Primary Score Row */}
      <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider leading-none flex items-center gap-1">
          {isBest && <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block shadow-sm" />}
          <span>{data?.metric_name || data?.primary_metric_name || t('node_score_fallback', 'Score')}</span>
          {data?.param_size && (
            <span className="text-[9px] bg-slate-800/90 text-amber-300 border border-slate-700/60 px-1 py-0.2 rounded font-mono font-semibold">
              {data.param_size}
            </span>
          )}
        </span>
        <span
          className={`font-mono text-xs font-bold leading-none ${
            isBest ? 'text-amber-400' : 'text-slate-100'
          }`}
        >
          {hasScore ? data.score.toFixed(4) : '—'}
        </span>
      </div>


      {/* Source Handle (Outgoing) */}
      <Handle
        key={isLR ? 'source-lr' : 'source-tb'}
        type="source"
        position={isLR ? Position.Right : Position.Bottom}
        isConnectable={false}
        className="!w-2 !h-2 !rounded-full !bg-indigo-400 !border-2 !border-slate-950 pointer-events-none"
      />
    </div>
  );
});

CustomExperimentNode.displayName = 'CustomExperimentNode';
