import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  GitBranch,
  Table,
  Cpu,
  Award,
  Cloud,
  Compass,
  Play,
  Sparkles,
  RefreshCw,
  Layers,
  Bot,
  FileText,
  AlertTriangle,
} from 'lucide-react';
import { api } from './services/api';
import { Header } from './components/Header';
import { TreeCanvas } from './components/TreeCanvas';
import { NodeInspector } from './components/NodeInspector';
import { LeaderboardTable } from './components/LeaderboardTable';
import { SubmissionsTab } from './components/SubmissionsTab';
import { PrimingCenter } from './components/PrimingCenter';
import { TaskConfigModal } from './components/TaskConfigModal';
import { CloudRunnerHub } from './components/CloudRunnerHub';
import { LLMStudio } from './components/LLMStudio';
import { CompetitionReadinessModal } from './components/CompetitionReadinessModal';
import { ExperimentConfigModal } from './components/ExperimentConfigModal';
import { CloudEnvironmentModal } from './components/CloudEnvironmentModal';
import { CompetitionPipelineHub } from './components/CompetitionPipelineHub';
import { CompetitionJourneyWorkspace } from './components/CompetitionJourneyWorkspace';
import { WorkspaceHub } from './components/WorkspaceHub';
import { LanguageProvider, useLanguage } from './constants/i18n';
import {
  TreeGraphResponse,
  ExperimentSummary,
  NodeDetail,
  UCBRecommendation,
  SystemStatus,
  TaskInfo,
  SubmissionEntry,
} from './types/experiment';

const DashboardContent: React.FC = () => {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState<'journey' | 'experiments' | 'tree' | 'pipeline' | 'llm' | 'submissions' | 'workspace' | 'cloud' | 'priming'>('journey');
  const [experimentsSubView, setExperimentsSubView] = useState<'table' | 'tree'>('table');
  const [executionSubView, setExecutionSubView] = useState<'runner' | 'cloud'>('runner');
  const [agentSubView, setAgentSubView] = useState<'workspace' | 'priming'>('workspace');
  const [compState, setCompState] = useState<any>(null);
  const [graphData, setGraphData] = useState<TreeGraphResponse>({ nodes: [], edges: [], best_id: null });
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [submissions, setSubmissions] = useState<SubmissionEntry[]>([]);
  const [recommendation, setRecommendation] = useState<UCBRecommendation | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [taskInfo, setTaskInfo] = useState<TaskInfo | null>(null);
  const [primingPrompt, setPrimingPrompt] = useState<string>('');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNodeDetail, setSelectedNodeDetail] = useState<NodeDetail | null>(null);
  const [isTaskConfigOpen, setIsTaskConfigOpen] = useState(false);
  const [taskConfigTab, setTaskConfigTab] = useState<'task' | 'data' | 'cv' | 'rules' | 'submission'>('task');
  const [isReadinessOpen, setIsReadinessOpen] = useState(false);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);
  const [isEnvModalOpen, setIsEnvModalOpen] = useState(false);
  const [configModalTarget, setConfigModalTarget] = useState<string | undefined>(undefined);
  const [configModalParentId, setConfigModalParentId] = useState<string | undefined>(undefined);
  const initialLoadedRef = useRef(false);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    if (!initialLoadedRef.current) {
      setLoading(true);
    }
    setApiError(null);
    try {
      const [treeRes, expRes, recRes, statusRes, promptRes, taskRes, subRes, compRes] = await Promise.allSettled([
        api.getTree(),
        api.getExperiments(),
        api.getRecommendation(),
        api.getStatus(),
        api.getPrimingPrompt(),
        api.getTaskInfo(),
        api.getSubmissions(),
        api.getCompetitionState(),
      ]);

      if (treeRes.status === 'fulfilled') setGraphData(treeRes.value);
      if (expRes.status === 'fulfilled') setExperiments(expRes.value.experiments);
      if (recRes.status === 'fulfilled') setRecommendation(recRes.value);
      if (statusRes.status === 'fulfilled') setStatus(statusRes.value);
      if (promptRes.status === 'fulfilled') setPrimingPrompt(promptRes.value.prompt);
      if (taskRes.status === 'fulfilled') setTaskInfo(taskRes.value);
      if (subRes.status === 'fulfilled') setSubmissions(subRes.value.submissions);
      if (compRes.status === 'fulfilled') setCompState(compRes.value);

      const criticalFailures = [statusRes, taskRes].filter((r) => r.status === 'rejected');
      if (criticalFailures.length > 0) {
        const firstErr = (criticalFailures[0] as PromiseRejectedResult).reason;
        setApiError(firstErr instanceof Error ? firstErr.message : String(firstErr));
      }
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    } finally {
      initialLoadedRef.current = true;
      setLoading(false);
    }
  }, []);

  const handleSyncArtifacts = async () => {
    try {
      await api.syncArtifacts();
    } catch (err) {
      console.warn('Sync call failed, continuing to reload:', err);
    }
    await loadData();
  };

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSelectNode = async (nodeId: string) => {
    setSelectedNodeId(nodeId);
    try {
      const detail = await api.getNode(nodeId);
      setSelectedNodeDetail(detail);
    } catch (err) {
      console.error('Failed to load node detail:', err);
    }
  };

  const handleCloseInspector = () => {
    setSelectedNodeId(null);
    setSelectedNodeDetail(null);
  };

  const handleOpenConfigModal = (targetPathOrAction?: string, parentId?: string) => {
    if (targetPathOrAction === 'create') {
      setConfigModalTarget(undefined);
      setConfigModalParentId(parentId || status?.best_node || undefined);
    } else {
      setConfigModalTarget(targetPathOrAction);
      setConfigModalParentId(parentId);
    }
    setIsConfigModalOpen(true);
  };

  if (loading && !initialLoadedRef.current && !taskInfo && !status) {
    return (
      <main className="min-h-screen bg-zinc-950 text-zinc-100 p-8 flex flex-col items-center justify-center space-y-3">
        <RefreshCw className="w-6 h-6 animate-spin text-indigo-400" />
        <h1 className="text-lg font-bold">{t('app_title', 'ML-Harness Workspace')}</h1>
        <p role="status" className="text-xs text-zinc-400">{t('loading_telemetry', 'Loading workspace telemetry & experiment tree…')}</p>
      </main>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-indigo-600 selection:text-white">
      <Header
        status={status}
        taskInfo={taskInfo}
        onRefresh={handleSyncArtifacts}
        onOpenTaskConfig={() => {
          setTaskConfigTab('task');
          setIsTaskConfigOpen(true);
        }}
        onOpenReadinessModal={() => setIsReadinessOpen(true)}
        onOpenConfigModal={() => handleOpenConfigModal()}
        onOpenEnvironmentModal={() => setIsEnvModalOpen(true)}
      />

      {apiError && (
        <div className="px-4 sm:px-6 pt-3">
          <div role="alert" className="bg-red-950/40 border border-red-800/60 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs text-red-200">
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <div>
                <strong className="font-semibold text-rose-200">{t('err_offline_banner_title', 'Backend Service Offline or Initializing')}</strong>
                <p className="text-[11px] text-red-300/80 mt-0.5">{t('err_offline_banner_desc', 'Telemetry unavailable. Ensure `ml-harness dashboard` is running.')} {apiError && `(${apiError})`}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={loadData}
                className="px-3 py-1.5 bg-rose-600 hover:bg-rose-500 text-white font-semibold rounded-lg text-xs transition"
              >
                {t('retry_connection', 'Retry Connection')}
              </button>
              <button
                type="button"
                onClick={() => setApiError(null)}
                className="px-2 py-1.5 text-red-300/70 hover:text-red-100 text-xs transition"
              >
                {t('btn_dismiss', 'Dismiss')}
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="flex-1 w-full px-4 sm:px-6 py-3 space-y-3.5 flex flex-col">
        {/* Streamlined Segmented Navigation Bar */}
        <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
          <nav className="flex flex-wrap gap-1 bg-zinc-900/80 p-1 rounded-xl border border-zinc-800/80 shadow-sm">
            <button
              onClick={() => setActiveTab('journey')}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg transition ${
                activeTab === 'journey'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span>{t('nav_journey', 'Competition Journey (Phases 1–8)')}</span>
            </button>

            <button
              onClick={() => {
                setActiveTab('experiments');
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                activeTab === 'experiments' || activeTab === 'tree'
                  ? 'bg-zinc-800 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>{t('nav_experiments_hub', 'Experiments & Solution Tree')}</span>
              {experiments.length > 0 && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-zinc-900 text-zinc-400 border border-zinc-700/60">
                  {experiments.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('pipeline')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                activeTab === 'pipeline' || activeTab === 'cloud'
                  ? 'bg-zinc-800 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <Play className="w-3.5 h-3.5" />
              <span>{t('nav_execution', 'Execution & Cloud')}</span>
            </button>

            <button
              onClick={() => setActiveTab('llm')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                activeTab === 'llm'
                  ? 'bg-zinc-800 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>{t('nav_llm_studio', 'LLM Studio')}</span>
            </button>

            <button
              onClick={() => setActiveTab('submissions')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                activeTab === 'submissions'
                  ? 'bg-zinc-800 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <Award className="w-3.5 h-3.5 text-yellow-400" />
              <span>{t('nav_submissions', 'Submissions History')}</span>
              {submissions.length > 0 && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-zinc-900 text-zinc-400 border border-zinc-700/60">
                  {submissions.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('workspace')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                activeTab === 'workspace' || activeTab === 'priming'
                  ? 'bg-zinc-800 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <Bot className="w-3.5 h-3.5 text-indigo-400" />
              <span>{t('nav_agent_copilot', 'Agent & Prompts Hub')}</span>
            </button>
          </nav>
        </div>

        {/* Primary 8-Phase Competition Journey */}
        {activeTab === 'journey' && (
          <CompetitionJourneyWorkspace
            compState={compState}
            taskInfo={taskInfo}
            experiments={experiments}
            submissions={submissions}
            graphData={graphData}
            recommendation={recommendation}
            primingPrompt={primingPrompt}
            selectedNodeId={selectedNodeId}
            onSelectNode={handleSelectNode}
            onOpenTaskConfig={(tab) => {
              setTaskConfigTab(tab || 'task');
              setIsTaskConfigOpen(true);
            }}
            onOpenConfigModal={(targetPath, parentId) => handleOpenConfigModal(targetPath, parentId)}
            onOpenEnvModal={() => setIsEnvModalOpen(true)}
            onRefreshAll={loadData}
          />
        )}

        {/* Unified Experiments Hub: Leaderboard Table & Solution Tree DAG */}
        {(activeTab === 'experiments' || activeTab === 'tree') && (
          <div className="space-y-3 flex-1 flex flex-col">
            {/* View Switcher Pill Bar */}
            <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-800 px-4 py-2 rounded-xl">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-zinc-200">
                  {experimentsSubView === 'table' ? t('view_leaderboard_title', 'Leaderboard & Experiment History') : t('view_tree_title', 'AIDE Solution Tree DAG')}
                </span>
                <span className="text-[11px] font-mono text-zinc-500 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                  {t('recorded_runs', '{count} recorded runs', { count: experiments.length })}
                </span>
              </div>

              <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-medium">
                <button
                  onClick={() => setExperimentsSubView('table')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition ${
                    experimentsSubView === 'table'
                      ? 'bg-zinc-800 text-white font-bold shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Table className="w-3.5 h-3.5" />
                  <span>{t('tab_leaderboard', 'Leaderboard Table')}</span>
                </button>
                <button
                  onClick={() => setExperimentsSubView('tree')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition ${
                    experimentsSubView === 'tree'
                      ? 'bg-indigo-600 text-white font-bold shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <GitBranch className="w-3.5 h-3.5" />
                  <span>{t('tab_tree', 'Solution Tree')}</span>
                </button>
              </div>
            </div>

            {experimentsSubView === 'table' ? (
              <LeaderboardTable
                experiments={experiments}
                submissions={submissions}
                taskInfo={taskInfo}
                onSelectNode={(id) => {
                  handleSelectNode(id);
                  setExperimentsSubView('tree');
                }}
              />
            ) : (
              <div className="w-full flex-1 flex flex-col min-h-[600px]">
                <TreeCanvas
                  graphData={graphData}
                  onSelectNode={handleSelectNode}
                  selectedNodeId={selectedNodeId}
                  recommendation={recommendation}
                />
              </div>
            )}
          </div>
        )}

        {/* Unified Execution & Cloud Hub */}
        {(activeTab === 'pipeline' || activeTab === 'cloud') && (
          <div className="space-y-3 flex-1 flex flex-col">
            <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-800 px-4 py-2 rounded-xl">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-zinc-200">
                  {executionSubView === 'runner' ? t('view_runner_title', '4-Step Competition Execution Pipeline') : t('view_cloud_title', 'Cloud GPU Runners & Clusters')}
                </span>
                <span className="text-[11px] font-mono text-zinc-500 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                  Modal • Kaggle • Local GPU
                </span>
              </div>

              <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-medium">
                <button
                  onClick={() => setExecutionSubView('runner')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition ${
                    executionSubView === 'runner'
                      ? 'bg-zinc-800 text-white font-bold shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{t('tab_runner', 'Pipeline Runner')}</span>
                </button>
                <button
                  onClick={() => setExecutionSubView('cloud')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition ${
                    executionSubView === 'cloud'
                      ? 'bg-indigo-600 text-white font-bold shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Cloud className="w-3.5 h-3.5" />
                  <span>{t('tab_cloud', 'Cloud GPU Hub')}</span>
                </button>
              </div>
            </div>

            {executionSubView === 'runner' ? (
              <CompetitionPipelineHub
                taskInfo={taskInfo}
                experiments={experiments as any}
                submissions={submissions}
                initialPrompt={primingPrompt}
                onOpenTaskConfig={(tab) => {
                  setTaskConfigTab(tab || 'task');
                  setIsTaskConfigOpen(true);
                }}
                onOpenConfigModal={(targetPath, parentId) => handleOpenConfigModal(targetPath, parentId)}
                onOpenEnvModal={() => setIsEnvModalOpen(true)}
                onRefreshAll={loadData}
              />
            ) : (
              <CloudRunnerHub
                taskInfo={taskInfo}
                onRefreshSolutionTree={loadData}
                onOpenConfigModal={(actionOrPath) => handleOpenConfigModal(actionOrPath)}
              />
            )}
          </div>
        )}

        {/* LLM Studio */}
        {activeTab === 'llm' && (
          <LLMStudio
            taskInfo={taskInfo}
            onRefreshSolutionTree={loadData}
            onOpenConfigModal={(actionOrPath, parentId) => handleOpenConfigModal(actionOrPath, parentId)}
          />
        )}

        {/* Submissions Hub */}
        {activeTab === 'submissions' && (
          <SubmissionsTab
            submissions={submissions}
            taskInfo={taskInfo}
            onRefresh={loadData}
            onSelectExperiment={(expId) => {
              handleSelectNode(expId);
              setActiveTab('experiments');
              setExperimentsSubView('tree');
            }}
          />
        )}

        {/* Unified Agent & Prompts Hub */}
        {(activeTab === 'workspace' || activeTab === 'priming') && (
          <div className="space-y-3 flex-1 flex flex-col">
            <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-800 px-4 py-2 rounded-xl">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-zinc-200">
                  {agentSubView === 'workspace' ? t('view_agent_title', 'Autonomous Agent Workspace & Diagnostics') : t('view_priming_title', 'Copilot Priming Prompts & System Instructions')}
                </span>
                <span className="text-[11px] font-mono text-zinc-500 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                  Layer 2 Agent Bridge
                </span>
              </div>

              <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-medium">
                <button
                  onClick={() => setAgentSubView('workspace')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition ${
                    agentSubView === 'workspace'
                      ? 'bg-zinc-800 text-white font-bold shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  <span>{t('tab_agent_ws', 'Agent Workspace')}</span>
                </button>
                <button
                  onClick={() => setAgentSubView('priming')}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition ${
                    agentSubView === 'priming'
                      ? 'bg-indigo-600 text-white font-bold shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>{t('tab_priming', 'Priming Prompts')}</span>
                </button>
              </div>
            </div>

            {agentSubView === 'workspace' ? (
              <WorkspaceHub onOpenEnvironmentModal={() => setIsEnvModalOpen(true)} />
            ) : (
              <PrimingCenter initialPrompt={primingPrompt} taskInfo={taskInfo} />
            )}
          </div>
        )}
      </main>

      {/* Subtle Bottom Contract Footer */}
      <footer className="border-t border-zinc-900 bg-zinc-950 px-6 py-2 flex items-center justify-between text-[11px] text-zinc-500">
        <span>{t('footer_evidence_notice', 'Verified evidence is required for eligible runs. Legacy artifacts are display only.')}</span>
        <span>{t('footer_budget_notice', 'Hard runtime budgets: local wall-time; remote allocation limits unavailable.')}</span>
      </footer>

      {/* Modals & Slide-Over Inspector */}
      {selectedNodeDetail && (
        <NodeInspector
          node={selectedNodeDetail}
          taskInfo={taskInfo}
          onClose={handleCloseInspector}
          onOpenConfigModal={(actionOrPath, parentId) => handleOpenConfigModal(actionOrPath, parentId)}
        />
      )}

      <TaskConfigModal
        isOpen={isTaskConfigOpen}
        initialTab={taskConfigTab}
        currentTask={taskInfo}
        onClose={() => setIsTaskConfigOpen(false)}
        onSave={async (updatedTask) => {
          setTaskInfo(updatedTask);
          await loadData();
        }}
      />

      <CompetitionReadinessModal
        isOpen={isReadinessOpen}
        onClose={() => setIsReadinessOpen(false)}
        taskInfo={taskInfo}
        status={status}
        submissions={submissions}
        onOpenTaskConfig={(tab) => {
          setTaskConfigTab(tab || 'task');
          setIsTaskConfigOpen(true);
        }}
      />

      {isConfigModalOpen && (
        <ExperimentConfigModal
          initialConfigPath={configModalTarget}
          initialParentId={configModalParentId}
          experiments={experiments}
          taskInfo={taskInfo}
          onClose={() => setIsConfigModalOpen(false)}
          onConfigSaved={async () => {
            await loadData();
          }}
        />
      )}

      <CloudEnvironmentModal
        isOpen={isEnvModalOpen}
        onClose={() => setIsEnvModalOpen(false)}
        onStatusChanged={loadData}
      />
    </div>
  );
};

export const App: React.FC = () => (
  <LanguageProvider>
    <DashboardContent />
  </LanguageProvider>
);
