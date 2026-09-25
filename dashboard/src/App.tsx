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
  const [activeTab, setActiveTab] = useState<'pipeline' | 'tree' | 'experiments' | 'submissions' | 'cloud' | 'llm' | 'priming'>('pipeline');
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
  const [, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [treeRes, expRes, recRes, statusRes, promptRes, taskRes, subRes] = await Promise.all([
        api.getTree().catch(() => ({ nodes: [], edges: [], best_id: null })),
        api.getExperiments().catch(() => ({ total: 0, experiments: [] })),
        api.getRecommendation().catch(() => ({ guideline: 'No recommendations available.', parent_id: null })),
        api.getStatus().catch(() => null),
        api.getPrimingPrompt().catch(() => ({ prompt: '', character_count: 0 })),
        api.getTaskInfo().catch(() => null),
        api.getSubmissions().catch(() => ({ total: 0, submissions: [] })),
      ]);

      setGraphData(treeRes);
      setExperiments(expRes.experiments);
      setRecommendation(recRes);
      setStatus(statusRes);
      setPrimingPrompt(promptRes.prompt);
      setTaskInfo(taskRes);
      setSubmissions(subRes.submissions);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
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
        {/* Segmented Navigation */}
        <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
          <nav className="flex flex-wrap gap-1 bg-zinc-900/70 p-1 rounded-lg border border-zinc-800">
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-md transition ${
                activeTab === 'pipeline'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-zinc-300 hover:text-white hover:bg-zinc-800/60'
              }`}
            >
              <span>Competition Pipeline (Steps 1–4)</span>
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
