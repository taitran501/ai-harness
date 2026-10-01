import React, { useState } from 'react';
import {
  RefreshCw,
  Layers,
  Sliders,
  ShieldCheck,
  Server,
} from 'lucide-react';
import { SystemStatus, TaskInfo } from '../types/experiment';
import { useLanguage } from '../constants/i18n';

interface HeaderProps {
  status: SystemStatus | null;
  taskInfo: TaskInfo | null;
  onRefresh: () => Promise<void>;
  onOpenTaskConfig?: () => void;
  onOpenReadinessModal?: () => void;
  onOpenConfigModal?: () => void;
  onOpenEnvironmentModal?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  taskInfo,
  onRefresh,
  onOpenTaskConfig,
  onOpenReadinessModal,
  onOpenConfigModal,
  onOpenEnvironmentModal,
}) => {
  const { lang, setLang } = useLanguage();
  const [syncing, setSyncing] = useState(false);

  const handleSync = async () => {
    setSyncing(true);
    try {
      await onRefresh();
    } finally {
      setSyncing(false);
    }
  };
  const taskName = taskInfo?.task_name || taskInfo?.name || 'Experiment Harness';
  const platform = taskInfo?.platform && taskInfo.platform !== 'unconfirmed' ? taskInfo.platform : undefined;

  return (
    <header className="bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/80 sticky top-0 z-40 px-5 py-2.5 flex items-center justify-between">
      {/* This dashboard is attached to the task workspace that launched it. */}
      <div className="flex items-center space-x-3">
        <div className="w-7 h-7 rounded-lg bg-zinc-900 border border-zinc-700/80 flex items-center justify-center text-zinc-300">
          <Layers className="w-3.5 h-3.5" />
        </div>

        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold tracking-tight text-zinc-100">{taskName}</span>
            {platform && <span className="rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 font-mono text-[10px] uppercase text-zinc-400">{platform}</span>}
            {taskInfo?.primary_metric && taskInfo.primary_metric !== 'unconfirmed' && <span className="hidden font-mono text-[11px] text-zinc-500 sm:inline">({taskInfo.primary_metric})</span>}
          </div>
          <div className="text-[10px] text-zinc-500">One task per workspace</div>
        </div>
      </div>

      {/* Keep routine controls visible; less frequent setup actions share one menu. */}
      <div className="flex items-center space-x-2">
        {status?.best_node && (
          <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded-md text-xs font-mono">
            <span className="text-zinc-400">Best:</span>
            <span className="font-semibold text-zinc-200">{status.best_node}</span>
            <span className="text-emerald-400 font-bold">{status.best_score.toFixed(4)}</span>
          </div>
        )}

        <details className="relative">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-800 hover:text-white">
            <Sliders className="h-3.5 w-3.5 text-zinc-400" />
            <span>Settings</span>
          </summary>
          <div className="absolute right-0 top-full z-50 mt-1 w-52 rounded-lg border border-zinc-700 bg-zinc-950 p-1 shadow-xl">
            {onOpenTaskConfig && <button onClick={onOpenTaskConfig} className="w-full rounded px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-800">Task configuration</button>}
            {onOpenEnvironmentModal && <button onClick={onOpenEnvironmentModal} className="flex w-full items-center gap-2 rounded px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-800"><Server className="h-3.5 w-3.5 text-sky-400" />Cloud accounts</button>}
            {onOpenReadinessModal && <button onClick={onOpenReadinessModal} className="flex w-full items-center gap-2 rounded px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-800"><ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />Readiness details</button>}
            {onOpenConfigModal && <button onClick={onOpenConfigModal} className="w-full rounded px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-800">Strategy catalog</button>}
            <button disabled={syncing} onClick={handleSync} className="flex w-full items-center gap-2 rounded px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"><RefreshCw className={`h-3.5 w-3.5 ${syncing ? 'animate-spin' : ''}`} />{syncing ? 'Refreshing experiment results…' : 'Refresh experiment results'}</button>
          </div>
        </details>

        {/* Explicit Language Toggle (EN default / VI option) */}
        <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-md p-0.5 text-[11px] font-mono">
          <button
            type="button"
            onClick={() => setLang('en')}
            className={`px-2 py-0.5 rounded transition ${
              lang === 'en' ? 'bg-zinc-800 text-zinc-100 font-semibold' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            EN
          </button>
          <button
            type="button"
            onClick={() => setLang('vi')}
            className={`px-2 py-0.5 rounded transition ${
              lang === 'vi' ? 'bg-zinc-800 text-zinc-100 font-semibold' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            VI
          </button>
        </div>
      </div>
    </header>
  );
};
