import React, { useState } from 'react';
import { Crown, RefreshCw, Layers, Sliders } from 'lucide-react';
import { SystemStatus, TaskInfo } from '../types/experiment';

interface HeaderProps {
  status: SystemStatus | null;
  taskInfo: TaskInfo | null;
  onRefresh: () => Promise<void>;
  onOpenTaskConfig?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ status, taskInfo, onRefresh, onOpenTaskConfig }) => {
  const [syncing, setSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  const handleSync = async () => {
    setSyncing(true);
    try {
      await onRefresh();
      setSyncMessage('Artifacts Synced!');
      setTimeout(() => setSyncMessage(null), 2500);
    } catch (err) {
      console.error(err);
      setSyncMessage('Sync Failed');
      setTimeout(() => setSyncMessage(null), 3000);
    } finally {
      setSyncing(false);
    }
  };

  const taskName = taskInfo?.task_name || taskInfo?.name || 'Experiment Harness';
  const platform = taskInfo?.platform;

  return (
    <header className="bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/80 sticky top-0 z-40 px-5 py-2.5 flex items-center justify-between">
      {/* Brand & Task Identity */}
      <div className="flex items-center space-x-3">
        <div className="w-7 h-7 rounded-lg bg-zinc-900 border border-zinc-700/80 flex items-center justify-center text-zinc-300">
          <Layers className="w-3.5 h-3.5" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-100 tracking-tight">
            {taskName}
          </span>
          {platform && (
            <span className="text-[10px] font-mono uppercase font-medium bg-zinc-900 text-zinc-400 border border-zinc-800 px-1.5 py-0.5 rounded">
              {platform}
            </span>
          )}
          {taskInfo?.primary_metric && (
            <span className="text-[11px] font-mono text-zinc-500 hidden sm:inline">
              • {taskInfo.primary_metric}
            </span>
          )}
        </div>
      </div>

      {/* Right Controls: Champion, Settings & Sync */}
      <div className="flex items-center space-x-2">
        {status?.best_node && (
          <div className="flex items-center gap-1.5 bg-amber-500/5 border border-amber-500/20 px-2.5 py-1 rounded-md text-xs text-amber-300/90 font-mono">
            <Crown className="w-3.5 h-3.5 text-amber-400" />
            <span className="font-semibold text-zinc-200">{status.best_node}</span>
            <span className="text-amber-400 font-bold">{status.best_score.toFixed(4)}</span>
          </div>
        )}

        {onOpenTaskConfig && (
          <button
            onClick={onOpenTaskConfig}
            className="flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-95 text-zinc-300 hover:text-white px-2.5 py-1 rounded-md border border-zinc-800 text-xs font-medium transition"
            title="Configure task platform, metrics, and subtasks"
          >
            <Sliders className="w-3.5 h-3.5 text-zinc-400" />
            <span>Configure</span>
          </button>
        )}

        <button
          onClick={handleSync}
          disabled={syncing}
          className="flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-95 text-zinc-300 hover:text-white p-1.5 sm:px-2.5 sm:py-1 rounded-md border border-zinc-800 text-xs font-medium transition"
          title="Scan artifacts/ directory for new runs"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-zinc-400 ${syncing ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline">{syncMessage || (syncing ? 'Syncing...' : 'Sync')}</span>
        </button>
      </div>
    </header>
  );
};
