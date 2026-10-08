import React, { useState, useEffect, useCallback } from 'react';
import { GitBranch, Table, Cpu, Award, Cloud } from 'lucide-react';
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
  const [activeTab, setActiveTab] = useState<'journey' | 'workspace' | 'pipeline' | 'tree' | 'experiments' | 'submissions' | 'cloud' | 'llm' | 'priming'>('journey');
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
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setApiError(null);
    try {
      const [treeRes, expRes, recRes, statusRes, promptRes, taskRes, subRes, compRes] = await Promise.all([
        api.getTree(),
        api.getExperiments(),
        api.getRecommendation(),
        api.getStatus(),
        api.getPrimingPrompt(),
        api.getTaskInfo(),
        api.getSubmissions(),
        api.getCompetitionState(),
      ]);

      setGraphData(treeRes);
      setExperiments(expRes.experiments);
      setRecommendation(recRes);
      setStatus(statusRes);
      setPrimingPrompt(promptRes.prompt);
      setTaskInfo(taskRes);
      setSubmissions(subRes.submissions);
      setCompState(compRes);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    } finally {
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

  if (loading) return <main className="min-h-screen bg-zinc-950 text-zinc-100 p-8"><h1>ML-Harness</h1><p role="status">Loading workspace…</p></main>;
  if (apiError) return <main className="min-h-screen bg-zinc-950 text-zinc-100 p-8"><h1>ML-Harness</h1><p role="alert">API error: {apiError}</p><button onClick={loadData}>Retry</button></main>;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans">
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

      <main className="flex-1 w-full px-5 py-3 space-y-3 flex flex-col">
        {loading && <p role="status">Loading workspace…</p>}
        {apiError && <div role="alert" className="border border-red-700 bg-red-950 p-3">API error: {apiError} <button onClick={loadData}>Retry</button></div>}
        {!loading && !apiError && graphData.nodes.length === 0 && <p role="status">Workspace has no recorded runs.</p>}
        {!loading && !apiError && <p className="text-xs text-zinc-400">Verified evidence is required for eligible runs. Legacy artifacts are display only. Hard runtime budgets: local wall-time; remote allocation limits unavailable.</p>}
        {/* Segmented Navigation */}
        <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
          <nav className="flex flex-wrap gap-1 bg-zinc-900/70 p-1 rounded-lg border border-zinc-800">
            <button
              onClick={() => setActiveTab('journey')}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-bold rounded-md transition ${
                activeTab === 'journey'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-zinc-300 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <span>🧭 Competition Journey (Phases 1–8)</span>
            </button>

            <button
              onClick={() => setActiveTab('workspace')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
                activeTab === 'workspace'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Agent workspace</span>
            </button>

            <button
              onClick={() => setActiveTab('pipeline')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
                activeTab === 'pipeline'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <span>Execution Runner</span>
            </button>

            <button
              onClick={() => setActiveTab('tree')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
                activeTab === 'tree'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>{t('nav_tree', 'Solution Tree')}</span>
            </button>

            <button
              onClick={() => setActiveTab('experiments')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
                activeTab === 'experiments'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <Table className="w-3.5 h-3.5" />
              <span>{t('nav_experiments', 'Leaderboard')}</span>
            </button>

            <button
              onClick={() => setActiveTab('llm')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
                activeTab === 'llm'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <span>{t('nav_llm_studio', 'LLM Studio')}</span>
            </button>

            <button
              onClick={() => setActiveTab('submissions')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
                activeTab === 'submissions'
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <Award className="w-3.5 h-3.5" />
              <span>{t('nav_submissions', 'Submissions History')}</span>
              {submissions.length > 0 && (
                <span className="text-zinc-400 text-[10px] font-mono">
                  ({submissions.length})
                </span>
              )}
            </button>
          </nav>
        </div>

        {/* Primary 8-Phase Competition Journey + 4-Question Hero HUD */}
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

        {activeTab === 'workspace' && (
          <WorkspaceHub onOpenEnvironmentModal={() => setIsEnvModalOpen(true)} />
        )}

        {/* Unified 4-Step Competition Pipeline Hub */}
        {activeTab === 'pipeline' && (
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
        )}

        {/* Tab: Solution Tree */}
        {activeTab === 'tree' && (
          <div className="w-full flex-1 flex flex-col">
            <TreeCanvas
              graphData={graphData}
              onSelectNode={handleSelectNode}
              selectedNodeId={selectedNodeId}
              recommendation={recommendation}
            />
          </div>
        )}

        {/* Tab 2: Leaderboard */}
        {activeTab === 'experiments' && (
          <LeaderboardTable
            experiments={experiments}
            submissions={submissions}
            taskInfo={taskInfo}
            onSelectNode={(id) => {
              handleSelectNode(id);
              setActiveTab('tree');
            }}
          />
        )}

        {/* Tab 3: Submissions Hub */}
        {activeTab === 'submissions' && (
          <SubmissionsTab
            submissions={submissions}
            taskInfo={taskInfo}
            onRefresh={loadData}
            onSelectExperiment={(expId) => {
              handleSelectNode(expId);
              setActiveTab('tree');
            }}
          />
        )}

        {/* Tab 4: Cloud Runners Hub */}
        {activeTab === 'cloud' && (
          <CloudRunnerHub
            taskInfo={taskInfo}
            onRefreshSolutionTree={loadData}
            onOpenConfigModal={(actionOrPath) => handleOpenConfigModal(actionOrPath)}
          />
        )}

        {/* Tab 5: LLM Studio & Playground */}
        {activeTab === 'llm' && (
          <LLMStudio
            taskInfo={taskInfo}
            onRefreshSolutionTree={loadData}
            onOpenConfigModal={(actionOrPath, parentId) => handleOpenConfigModal(actionOrPath, parentId)}
          />
        )}

        {/* Tab 6: Priming Prompt Center */}
        {activeTab === 'priming' && (
          <PrimingCenter initialPrompt={primingPrompt} taskInfo={taskInfo} />
        )}
      </main>

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
