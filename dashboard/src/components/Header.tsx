import React, { useState, useEffect, useRef } from 'react';
import {
  RefreshCw,
  Layers,
  Sliders,
  ShieldCheck,
  Server,
  ChevronDown,
  Plus,
  Check,
  Trophy,
  Sparkles,
} from 'lucide-react';
import { SystemStatus, TaskInfo, TaskSummary } from '../types/experiment';
import { api } from '../services/api';
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
  const { lang, setLang, t } = useLanguage();
  const [syncing, setSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [isTaskMenuOpen, setIsTaskMenuOpen] = useState(false);
  const [switchingTask, setSwitchingTask] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const loadTasks = async () => {
    try {
      const res = await api.getTasks();
      if (res.tasks) {
        setTasks(res.tasks);
      }
    } catch (err) {
      console.warn('Could not load task list:', err);
    }
  };

  useEffect(() => {
    loadTasks();
  }, [taskInfo]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsTaskMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleTaskSwitch = async (taskId: string) => {
    if (switchingTask) return;
    setSwitchingTask(true);
    setIsTaskMenuOpen(false);
    try {
      await api.switchTask(taskId);
      await onRefresh();
      await loadTasks();
    } catch (err) {
      console.error('Failed to switch task:', err);
    } finally {
      setSwitchingTask(false);
    }
  };

  const handleSync = async () => {
    setSyncing(true);
    try {
      await onRefresh();
      setSyncMessage(t('header_synced', 'Synced'));
      setTimeout(() => setSyncMessage(null), 2500);
    } catch (err) {
      console.error(err);
      setSyncMessage(t('header_sync_failed', 'Sync Failed'));
      setTimeout(() => setSyncMessage(null), 3000);
    } finally {
      setSyncing(false);
    }
  };

  const taskName = taskInfo?.task_name || taskInfo?.name || 'Experiment Harness';
  const platform = taskInfo?.platform;

  return (
    <header className="bg-zinc-950/90 backdrop-blur-md border-b border-zinc-800/80 sticky top-0 z-40 px-4 sm:px-6 py-2.5 flex items-center justify-between gap-3">
      {/* Brand & Multi-Task Project Switcher */}
      <div className="flex items-center space-x-3 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-indigo-950/80 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-sm shadow-indigo-500/10">
          <Layers className="w-4 h-4" />
        </div>

        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setIsTaskMenuOpen(!isTaskMenuOpen)}
            className="flex items-center gap-2 hover:bg-zinc-900/80 px-2.5 py-1.5 rounded-lg border border-transparent hover:border-zinc-800 transition text-left"
          >
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-semibold text-zinc-100 tracking-tight">
                {taskName}
              </span>
              <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
            </div>
            {platform && (
              <span className="text-[10px] font-mono uppercase font-semibold bg-zinc-900 text-zinc-400 border border-zinc-800 px-1.5 py-0.5 rounded">
                {platform}
              </span>
            )}
            {taskInfo?.primary_metric && (
              <span className="text-[11px] font-mono text-emerald-400/90 bg-emerald-950/40 border border-emerald-800/40 px-1.5 py-0.5 rounded hidden sm:inline">
                {taskInfo.primary_metric}
              </span>
            )}
          </button>

          {isTaskMenuOpen && (
            <div className="absolute top-full left-0 mt-1.5 w-72 bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl p-2 z-50 space-y-1">
              <div className="px-2 py-1 text-[10px] font-mono uppercase text-zinc-500 font-semibold border-b border-zinc-800/80 flex justify-between items-center">
                <span>{t('header_switch_task', 'Switch Competition Task')}</span>
                {switchingTask && <RefreshCw className="w-3 h-3 animate-spin text-zinc-400" />}
              </div>

              <div className="max-h-56 overflow-y-auto space-y-1">
                {tasks.map((t) => {
                  const isActive = t.is_active || t.name === taskName;
                  return (
                    <button
                      key={t.id}
                      onClick={() => handleTaskSwitch(t.id)}
                      className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs text-left transition ${
                        isActive
                          ? 'bg-zinc-900 border border-zinc-700 text-zinc-100 font-medium'
                          : 'hover:bg-zinc-900/60 text-zinc-300 hover:text-white border border-transparent'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-1.5">
                          <span className="truncate">{t.name}</span>
                          <span className="text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/60">
                            {t.platform}
                          </span>
                        </div>
                        <div className="text-[10px] text-zinc-500 font-mono mt-0.5">
                          {t.modality} ({t.primary_metric})
                        </div>
                      </div>
                      {isActive && <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                    </button>
                  );
                })}
              </div>

              <div className="border-t border-zinc-800/80 pt-1">
                <button
                  onClick={() => {
                    setIsTaskMenuOpen(false);
                    onOpenTaskConfig?.();
                  }}
                  className="w-full flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-zinc-300 hover:text-white hover:bg-zinc-900 transition"
                >
                  <Plus className="w-3.5 h-3.5 text-zinc-400" />
                  <span>{t('header_configure_task', 'Configure or Add Task')}</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Center: Best Performance Pill (Highlights top score at a glance) */}
      {status?.best_node && (
        <div className="hidden md:flex items-center gap-2 bg-zinc-900/90 border border-amber-500/30 px-3 py-1 rounded-full text-xs font-mono shadow-sm">
          <Trophy className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-zinc-400">{t('header_best_label', 'Best:')}</span>
          <span className="font-semibold text-zinc-200">{status.best_node}</span>
          <span className="text-emerald-400 font-bold bg-emerald-950/60 border border-emerald-500/30 px-1.5 py-0.5 rounded text-[11px]">
            {status.best_score.toFixed(4)}
          </span>
        </div>
      )}

      {/* Right Controls: Readiness, Setup, Environment, Strategy Hub, Sync, Language */}
      <div className="flex items-center space-x-1.5 sm:space-x-2">
        {onOpenReadinessModal && (
          <button
            onClick={onOpenReadinessModal}
            className="flex items-center gap-1.5 bg-zinc-900/90 hover:bg-zinc-800 text-emerald-400 hover:text-emerald-300 px-2.5 py-1 rounded-lg border border-emerald-500/30 text-xs font-medium transition shadow-sm"
            title={t('header_readiness_tooltip', 'Evaluate 8 Competition Readiness Dimensions')}
          >
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span className="hidden lg:inline">{t('header_readiness', 'Readiness')}</span>
          </button>
        )}

        {onOpenTaskConfig && (
          <button
            onClick={onOpenTaskConfig}
            className="flex items-center gap-1.5 bg-zinc-900/90 hover:bg-zinc-800 text-zinc-300 hover:text-white px-2.5 py-1 rounded-lg border border-zinc-800 text-xs font-medium transition"
            title={`${t('task_settings', 'Task Settings')} — ${t('header_setup_tooltip', 'Configure task platform, metrics, and subtasks')}`}
          >
            <Sliders className="w-3.5 h-3.5 text-zinc-400" />
            <span className="hidden xl:inline">{t('header_setup', 'Setup')}</span>
          </button>
        )}

        {onOpenEnvironmentModal && (
          <button
            onClick={onOpenEnvironmentModal}
            className="flex items-center gap-1.5 bg-zinc-900/90 hover:bg-zinc-800 text-zinc-300 hover:text-white px-2.5 py-1 rounded-lg border border-zinc-800 text-xs font-medium transition"
            title={t('header_env_tooltip', 'Inspect & configure Modal, Kaggle, Docker, and local GPU accounts')}
          >
            <Server className="w-3.5 h-3.5 text-sky-400" />
            <span className="hidden xl:inline">{t('header_env', 'Env')}</span>
          </button>
        )}

        {onOpenConfigModal && (
          <button
            onClick={onOpenConfigModal}
            className="flex items-center gap-1.5 bg-zinc-900/90 hover:bg-zinc-800 text-zinc-300 hover:text-white px-2.5 py-1 rounded-lg border border-zinc-800 text-xs font-medium transition"
            title={`${t('new_experiment', 'New Experiment')} — ${t('header_strategy_tooltip', 'Open Strategy Catalog and configure experiment targets')}`}
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden xl:inline">{t('header_strategy', 'Strategy')}</span>
          </button>
        )}

        <button
          onClick={handleSync}
          disabled={syncing}
          className="flex items-center gap-1.5 bg-zinc-900/90 hover:bg-zinc-800 text-zinc-300 hover:text-white px-2.5 py-1 rounded-lg border border-zinc-800 text-xs font-medium transition"
          title={`${t('refresh', 'Refresh')} — ${t('header_sync_tooltip', 'Scan experiment artifacts for new runs')}`}
        >
          <RefreshCw className={`w-3.5 h-3.5 ${syncing ? 'animate-spin text-indigo-400' : 'text-zinc-400'}`} />
          <span className="hidden sm:inline">{syncMessage || t('header_sync', 'Sync')}</span>
        </button>

        {/* Language Toggle (EN default / VI option) */}
        <div
          className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg p-0.5 text-[11px] font-mono"
          title={t('language_label', 'Language')}
        >
          <button
            type="button"
            onClick={() => setLang('en')}
            className={`px-1.5 py-0.5 rounded transition ${
              lang === 'en' ? 'bg-zinc-800 text-zinc-100 font-semibold shadow-sm' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            EN
          </button>
          <button
            type="button"
            onClick={() => setLang('vi')}
            className={`px-1.5 py-0.5 rounded transition ${
              lang === 'vi' ? 'bg-zinc-800 text-zinc-100 font-semibold shadow-sm' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            VI
          </button>
        </div>
      </div>
    </header>
  );
};
