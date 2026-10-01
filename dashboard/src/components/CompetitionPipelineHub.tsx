import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Sliders,
  Play,
  Square,
  Upload,
  Download,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  ArrowRight,
  RotateCcw,
  Terminal,
  Cloud,
  Cpu,
  Globe,
  FileArchive,
} from 'lucide-react';
import {
  TaskInfo,
  ExperimentSummary,
  SubmissionEntry,
  EnvironmentStatus,
  CloudJob,
} from '../types/experiment';
import { api } from '../services/api';
import { PrimingCenter } from './PrimingCenter';

interface CompetitionPipelineHubProps {
  taskInfo: TaskInfo | null;
  experiments: ExperimentSummary[];
  submissions: SubmissionEntry[];
  initialPrompt: string;
  onOpenTaskConfig: (tab?: 'task' | 'data' | 'cv' | 'rules' | 'submission') => void;
  onOpenConfigModal: (targetPath?: string, parentId?: string) => void;
  onOpenEnvModal: () => void;
  onRefreshAll: () => Promise<void>;
}

export const CompetitionPipelineHub: React.FC<CompetitionPipelineHubProps> = ({
  taskInfo,
  experiments,
  submissions,
  initialPrompt,
  onOpenTaskConfig,
  onOpenConfigModal,
  onOpenEnvModal,
  onRefreshAll,
}) => {
  const [activeStep, setActiveStep] = useState<1 | 2 | 3 | 4>(1);

  //Configs & target selection
  const [configs, setConfigs] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>('configs/exp001_tfidf_baseline.yaml');

  // Step 1: Competition portal URL
  const [portalUrl, setPortalUrl] = useState<string>(
    () => localStorage.getItem('competition_portal_url') || ''
  );

  // Step 3: Execution & Cloud sync state
  const [execMode, setExecMode] = useState<'local' | 'kaggle' | 'modal'>('local');
  const [modalGpu, setModalGpu] = useState<string>('A10G');
  const [kaggleGpu, setKaggleGpu] = useState<string>('T4 x 2');
  const [envStatus, setEnvStatus] = useState<EnvironmentStatus | null>(null);
  const [cloudDataStatus, setCloudDataStatus] = useState<{
    ready: boolean;
    paths: string[];
    error?: string | null;
  } | null>(null);
  const [checkingCloudData, setCheckingCloudData] = useState(false);

  // Dataset push state
  const [pushingDataset, setPushingDataset] = useState<boolean>(false);
  const [datasetPushLog, setDatasetPushLog] = useState<string | null>(null);

  // Running job & logs
  const [activeJob, setActiveJob] = useState<CloudJob | null>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [isLaunching, setIsLaunching] = useState<boolean>(false);
  const [isStopping, setIsStopping] = useState<boolean>(false);

  // Step 4: Packaging & LB score recording
  const [packagingSub, setPackagingSub] = useState<boolean>(false);
  const [packageMsg, setPackageMsg] = useState<string | null>(null);
  const [packageErr, setPackageErr] = useState<string | null>(null);
  const [activeSubmissionId, setActiveSubmissionId] = useState<string>('');
  const [lbPrimaryScore, setLbPrimaryScore] = useState<string>('');
  const [subtaskScores, setSubtaskScores] = useState<Record<string, string>>({});
  const [lbNotes, setLbNotes] = useState<string>('');
  const [savingScore, setSavingScore] = useState<boolean>(false);
  const [scoreSavedMsg, setScoreSavedMsg] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const refreshCloudDataStatus = useCallback(async () => {
    setCheckingCloudData(true);
    try {
      const status = await api.getCloudDataStatus();
      setCloudDataStatus(status);
      return status;
    } catch (err: any) {
      const status = {
        ready: false,
        paths: [],
        error: err.message || 'Could not inspect task data paths.',
      };
      setCloudDataStatus(status);
      return status;
    } finally {
      setCheckingCloudData(false);
    }
  }, []);

  const currentExpId = selectedConfig
    .replace(/^configs\//, '')
    .replace(/\.ya?ml$/, '');
  const artifactDir = taskInfo?.artifacts_dir || 'artifacts';
  const isCustomTaskPlugin = Boolean(taskInfo?.task_plugin?.id && taskInfo.task_plugin.id !== 'default');
  const pluginCanSubmit = Boolean(
    taskInfo?.task_plugin?.capabilities?.submission_builder &&
    taskInfo?.task_plugin?.capabilities?.submission_validator,
  );

  useEffect(() => {
    api
      .getConfigs()
      .then((res) => {
        if (res.configs && res.configs.length > 0) {
          setConfigs(res.configs);
          setSelectedConfig(res.configs[res.configs.length - 1]);
        }
      })
      .catch(() => {});

    api
      .getEnvironmentStatus()
      .then((st) => setEnvStatus(st))
      .catch(() => {});
  }, [taskInfo?.name]);

  useEffect(() => {
    if (execMode === 'local' || isCustomTaskPlugin) {
      setCloudDataStatus(null);
      return;
    }
    void refreshCloudDataStatus();
  }, [execMode, isCustomTaskPlugin, refreshCloudDataStatus, taskInfo?.name]);

  useEffect(() => {
    if (submissions.length > 0 && !activeSubmissionId) {
      setActiveSubmissionId(submissions[submissions.length - 1].id);
    }
  }, [submissions, activeSubmissionId]);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [jobLogs]);

  const handleSavePortalUrl = (val: string) => {
    setPortalUrl(val);
    localStorage.setItem('competition_portal_url', val);
  };

  // Step 3A: Push Dataset to Cloud
  const handlePushDataset = async () => {
    const targetPlatform = execMode === 'modal' ? 'modal' : 'kaggle';
    setPushingDataset(true);
    setDatasetPushLog('Checking task data paths before upload...');
    try {
      const dataStatus = await refreshCloudDataStatus();
      if (!dataStatus.ready) {
        setDatasetPushLog(`Nothing was uploaded. ${dataStatus.error || 'Fix the task data paths before using cloud execution.'}`);
        return;
      }
      setDatasetPushLog(`Syncing ${dataStatus.paths.join(', ')} privately to ${targetPlatform.toUpperCase()}...`);
      const res = await api.pushDatasetToCloud({
        platform: targetPlatform,
      });
      setDatasetPushLog(res.logs || `Dataset synced to ${res.dataset_slug || res.volume_name}`);
    } catch (err: any) {
      setDatasetPushLog(`Error pushing dataset: ${err.message}`);
    } finally {
      setPushingDataset(false);
    }
  };

  // Step 3B: Run Experiment (Local CPU or Cloud GPU)
  const handleRunExperiment = async () => {
    setIsLaunching(true);
    setActiveJob(null);
    setJobLogs([execMode === 'local'
      ? `>>> Starting local execution for ${selectedConfig}...`
      : '>>> Checking task data before starting a private cloud run...']);

    try {
      if (execMode !== 'local') {
        const dataStatus = await refreshCloudDataStatus();
        if (!dataStatus.ready) {
          const missingPath = (dataStatus.error || '').replace('Declared task data path does not exist: ', '');
          setJobLogs([
            'Cloud run stopped before upload; no job was started.',
            missingPath ? `Missing task data path: ${missingPath}` : (dataStatus.error || 'Task data paths are not ready.'),
            /tf.?idf/i.test(selectedConfig)
              ? 'This TF-IDF baseline is a CPU workload. Choose Local CPU to run it without cloud sync.'
              : 'To use cloud execution, update Step 1 > Task & Data Setup so every declared data path exists.',
          ]);
          return;
        }
      }

      setJobLogs([`>>> Starting [${execMode.toUpperCase()}] execution for ${selectedConfig}...`]);
      const gpuToUse =
        execMode === 'modal' ? modalGpu : execMode === 'kaggle' ? kaggleGpu : 'CPU';

      const res = await api.runCloudJob({
        backend: execMode,
        config_path: selectedConfig,
        gpu: gpuToUse,
        auto_audit: true,
        auto_sync: true,
      });

      if (res.job) {
        setActiveJob(res.job);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
        const es = new EventSource(`/api/cloud/logs/stream?job_id=${encodeURIComponent(res.job.job_id)}`);
        eventSourceRef.current = es;
        es.onmessage = (ev) => {
          try {
            const payload = JSON.parse(ev.data);
            if (payload.type === 'log') {
              setJobLogs((prev) => [...prev, payload.line]);
            } else if (payload.type === 'status') {
              setActiveJob((prev) =>
                prev ? { ...prev, status: payload.status, return_code: payload.return_code } : null
              );
              es.close();
              if (payload.status === 'COMPLETED') {
                onRefreshAll();
              }
            }
          } catch {}
        };
        es.onerror = () => es.close();
      }
    } catch (err: any) {
      setJobLogs((prev) => [...prev, `[ERROR] ${err.message}`]);
    } finally {
      setIsLaunching(false);
    }
  };

  const handleStopJob = async () => {
    if (!activeJob) return;
    setIsStopping(true);
    try {
      await api.stopCloudJob(activeJob.job_id);
      setJobLogs((prev) => [...prev, '>>> Sent termination signal to running job.']);
    } catch (err: any) {
      setJobLogs((prev) => [...prev, `[ERROR] ${err.message}`]);
    } finally {
      setIsStopping(false);
    }
  };

  // Step 4A: Package Submission according to Step 1 Spec
  const handlePackageSubmission = async () => {
    setPackagingSub(true);
    setPackageMsg(null);
    setPackageErr(null);
    try {
      const minDiff = taskInfo?.submission_spec?.min_diff_pct ?? 2.0;
      const res = await api.registerSubmission({
        exp_id: currentExpId,
        notes: `Packaged from ${currentExpId} for ${taskInfo?.platform || 'the configured task'}`,
        min_diff_pct: minDiff,
      });
      setActiveSubmissionId(res.submission.id);
      setPackageMsg(
        `Packaged ${res.submission.archive_path || 'submission.zip'} (${res.submission.id}) with the active submission checks.`
      );
      await onRefreshAll();
    } catch (err: any) {
      setPackageErr(err.message || 'Failed to package submission.');
    } finally {
      setPackagingSub(false);
    }
  };

  // Step 4B: Record Official Leaderboard Score & Loop Back to Step 2
  const handleSaveScoreAndLoop = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeSubmissionId) {
      setPackageErr('Please package or select a submission first.');
      return;
    }
    setSavingScore(true);
    setPackageErr(null);
    try {
      const payload: Record<string, any> = { split_type: 'public' };
      if (lbPrimaryScore.trim() !== '') {
        const val = parseFloat(lbPrimaryScore);
        if (!isNaN(val)) {
          payload.primary_metric = val;
          payload.score = val;
        }
      }
      Object.entries(subtaskScores).forEach(([subtaskId, val]) => {
        if (val.trim() !== '') {
          const num = parseFloat(val);
          if (!isNaN(num)) {
            payload[subtaskId] = num;
          }
        }
      });

      await api.recordOfficialScore(activeSubmissionId, payload, lbNotes.trim());
      await onRefreshAll();
      setScoreSavedMsg('Score & Note saved! Updating Agent Prompt in Step 2...');
      setTimeout(() => {
        setScoreSavedMsg(null);
        setActiveStep(2);
      }, 900);
    } catch (err: any) {
      setPackageErr(err.message || 'Failed to record official score.');
    } finally {
      setSavingScore(false);
    }
  };

  const dataCfg = taskInfo?.data_config || {};
  const subSpec = taskInfo?.submission_spec || {};
  const activeSubObj = submissions.find((s) => s.id === activeSubmissionId);
  const missingRemoteDataPath = cloudDataStatus?.error?.match(/^Declared task data path does not exist: (.+)$/)?.[1];

  return (
    <div className="space-y-5">
      {/* 4-Step Pipeline Stepper Bar */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-3 shadow-lg">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5">
          {[
            {
              step: 1 as const,
              title: 'Step 1. Task & Data Setup',
              desc: `${taskInfo?.platform?.toUpperCase() || 'Platform not set'} / ${dataCfg.split_strategy || 'Validation not set'}`,
            },
            {
              step: 2 as const,
              title: 'Step 2. Agent Prompt & Propose',
              desc: `Target: ${currentExpId}`,
            },
            {
              step: 3 as const,
              title: 'Step 3. Run & Retrieve Outputs',
              desc: isCustomTaskPlugin ? 'Run locally through the installed task plugin' : 'Local CPU or Cloud GPU (Kaggle / Modal)',
            },
            {
              step: 4 as const,
              title: 'Step 4. Submission & Score',
              desc: isCustomTaskPlugin && !pluginCanSubmit
                ? 'Submission hooks are missing from the selected task plugin'
                : `Build .${subSpec.archive_format || 'zip'} package, record score & continue`,
            },
          ].map((item) => {
            const isActive = activeStep === item.step;
            return (
              <button
                key={item.step}
                type="button"
                onClick={() => setActiveStep(item.step)}
                className={`text-left p-3 rounded-lg border transition flex flex-col justify-between ${
                  isActive
                    ? 'bg-zinc-900 border-indigo-500 text-zinc-100 shadow-md'
                    : 'bg-zinc-950 hover:bg-zinc-900/70 border-zinc-800 text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <div className="font-semibold text-sm">{item.title}</div>
                <div className="text-xs text-zinc-400 mt-1 font-mono truncate">{item.desc}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* =====================================================================
          STEP 1: TASK, DATA STRATEGY & VALIDATION SETUP
         ===================================================================== */}
      {activeStep === 1 && (
        <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5 space-y-5 shadow-xl">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-4">
            <div>
              <h2 className="text-base font-semibold text-zinc-100">
                Step 1. Competition Task, Data Schema & Validation Setup
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                Verify dataset paths, input/target columns, validation split, and submission format before generating the Agent prompt.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onOpenTaskConfig('task')}
                className="flex items-center gap-1.5 px-3.5 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 rounded-lg border border-zinc-700 text-xs font-medium transition"
              >
                <Sliders className="w-4 h-4 text-indigo-400" />
                <span>Edit Full Setup</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveStep(2)}
                className="flex items-center gap-1.5 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition"
              >
                <span>Next: Agent Prompt & Propose</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Summary Cards for Task, Data Schema, CV Protocol, and Submission Portal */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Card 1: Task & Metric */}
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200">1. Task & Evaluation Objective</span>
                <button
                  type="button"
                  onClick={() => onOpenTaskConfig('task')}
                  className="text-xs text-indigo-400 hover:underline"
                >
                  Configure
                </button>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">Task: </span>
                  <span className="text-zinc-200">{taskInfo?.name || 'Competition Task'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Platform: </span>
                  <span className="text-zinc-200 uppercase">{taskInfo?.platform || 'not configured'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Primary Metric: </span>
                  <span className="text-emerald-400">{taskInfo?.primary_metric || 'Not configured'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Submission File: </span>
                  <span className="text-zinc-200">
                    {subSpec.file_name || 'submission.zip'} (.{subSpec.archive_format || 'zip'})
                  </span>
                </div>
              </div>
            </div>

            {/* Card 2: Data Schema & Strategy */}
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200">2. Data Schema & Strategy</span>
                <button
                  type="button"
                  onClick={() => onOpenTaskConfig('data')}
                  className="text-xs text-indigo-400 hover:underline"
                >
                  Configure
                </button>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">Train / Test Path: </span>
                  <span className="text-zinc-200">
                    {dataCfg.train_path || 'dataset/'} , {dataCfg.test_path || 'dataset/test/'}
                  </span>
                </div>
                <div>
                  <span className="text-zinc-500">ID Column: </span>
                  <span className="text-zinc-200">{dataCfg.id_column || 'id'}</span>
                </div>
                <div className="truncate" title={dataCfg.target_column}>
                  <span className="text-zinc-500">Target Columns: </span>
                  <span className="text-zinc-200">
                    {dataCfg.target_column || 'target'}
                  </span>
                </div>
                <div>
                  <span className="text-zinc-500">Missing Policy: </span>
                  <span className="text-zinc-200">{dataCfg.missing_data_policy || 'fill_empty_context'}</span>
                </div>
              </div>
            </div>

            {/* Card 3: Validation Split (CV) */}
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200">3. Validation Split (CV)</span>
                <button
                  type="button"
                  onClick={() => onOpenTaskConfig('cv')}
                  className="text-xs text-indigo-400 hover:underline"
                >
                  Configure
                </button>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">Split Method: </span>
                  <span className="text-zinc-200">{dataCfg.split_strategy || 'Not declared'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Group Column: </span>
                  <span className="text-amber-300">{dataCfg.group_column || 'Not declared'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Folds / Seed: </span>
                  <span className="text-zinc-200">
                    {dataCfg.n_folds ?? 'Not declared'} folds (seed={dataCfg.seed ?? 'Not declared'})
                  </span>
                </div>
                <div>
                  <span className="text-zinc-500">Fold File: </span>
                  <span className="text-zinc-200">{dataCfg.split_file || dataCfg.fold_file || 'Not declared'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Competition Portal URL Bar */}
          <div className="p-4 bg-zinc-900/40 border border-zinc-800 rounded-xl flex flex-wrap items-center justify-between gap-3">
            <div className="flex-1 min-w-[260px]">
              <label className="block text-xs font-semibold text-zinc-300 mb-1">
                Competition Submission Portal URL:
              </label>
              <input
                type="url"
                value={portalUrl}
                onChange={(e) => handleSavePortalUrl(e.target.value)}
                placeholder="https://..."
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200"
              />
            </div>
            <button
              type="button"
              onClick={() => setActiveStep(2)}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition flex items-center gap-2"
            >
              <span>Proceed to Step 2: Agent Prompt & Propose</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* =====================================================================
          STEP 2: AGENT PROMPT & PROPOSE TARGET CONFIG
         ===================================================================== */}
      {activeStep === 2 && (
        <div className="space-y-5">
          {/* Target Experiment Config Selection / Creation Bar */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
            <div className="flex-1 min-w-[260px] space-y-1">
              <div className="text-sm font-semibold text-zinc-100">
                Active Proposed Experiment Config (`configs/*.yaml`)
              </div>
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <select
                  value={selectedConfig}
                  onChange={(e) => setSelectedConfig(e.target.value)}
                  className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200 min-w-[280px]"
                >
                  {configs.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => onOpenConfigModal(selectedConfig)}
                  className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 rounded-lg border border-zinc-800 text-xs font-medium transition"
                >
                  Inspect / Edit YAML
                </button>
                <button
                  type="button"
                  onClick={() => onOpenConfigModal('create')}
                  className="px-3 py-1.5 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 rounded-lg border border-indigo-500/40 text-xs font-semibold transition"
                >
                  + Propose New Config
                </button>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setActiveStep(3)}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition flex items-center gap-2"
            >
              <span>Next: Run Experiment (Step 3)</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Agent Context & Prompt Generator (Includes Validation & External Score Feedback) */}
          <PrimingCenter initialPrompt={initialPrompt} taskInfo={taskInfo} />
        </div>
      )}

      {/* =====================================================================
          STEP 3: RUN LOCALLY OR ON A CLOUD PROVIDER
         ===================================================================== */}
      {activeStep === 3 && (
        <div className="space-y-4">
          {execMode !== 'local' && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-zinc-300">
                {execMode === 'kaggle'
                  ? <>Kaggle account: <strong>{envStatus?.kaggle.username || 'Not connected'}</strong></>
                  : <>Modal profile: <strong>{envStatus?.modal.active_profile || 'Not configured'}</strong></>}
              </div>
              <button
                type="button"
                onClick={onOpenEnvModal}
                className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 rounded-lg border border-zinc-800 text-xs font-medium transition"
              >
                Provider account settings
              </button>
            </div>
          )}

          <div className="grid grid-cols-1 gap-3">

            {/* Card 3.2: Choose Execution Mode & Run */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 space-y-3">
              <div className="text-sm font-semibold text-zinc-100 border-b border-zinc-800 pb-2">
                Run experiment: {currentExpId}
              </div>

              {execMode === 'local' ? (
                <div className="p-2.5 bg-emerald-950/30 border border-emerald-900/70 rounded-lg text-xs text-emerald-200">
                  This runs on your computer. No Kaggle login, dataset upload, GPU selection, or output download is needed.
                  {/tf.?idf/i.test(selectedConfig) && ' A TF-IDF baseline is a good fit for Local CPU.'}
                </div>
              ) : (
                <div className="space-y-2">
                  {checkingCloudData ? (
                    <div className="p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-xs text-zinc-300">
                      Checking the task data paths before enabling cloud execution…
                    </div>
                  ) : cloudDataStatus?.ready ? (
                    <div className="p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-xs text-zinc-300">
                      Data is ready. The Run action uploads it privately if needed, tracks the job, and saves results under <code>{artifactDir}/{currentExpId}/</code>.
                    </div>
                  ) : (
                    <div className="p-3 bg-amber-950/40 border border-amber-800/70 rounded-lg space-y-2 text-xs text-amber-100">
                      <div className="font-semibold">Cloud run is blocked before upload.</div>
                      <div>
                        {missingRemoteDataPath
                          ? <>The task setup expects <code>{missingRemoteDataPath}</code>, but that path is missing here. Nothing has been uploaded.</>
                          : cloudDataStatus?.error || 'Task data paths could not be checked yet.'}
                      </div>
                      {/tf.?idf/i.test(selectedConfig) && (
                        <div>This TF-IDF baseline is a CPU workload. Switch to Local CPU to run it without cloud sync.</div>
                      )}
                      <div className="flex flex-wrap items-center gap-3">
                        <button
                          type="button"
                          onClick={() => setExecMode('local')}
                          className="px-3 py-1.5 bg-amber-200 text-zinc-900 rounded-md font-semibold"
                        >
                          Use Local CPU
                        </button>
                      </div>
                      <details>
                        <summary className="cursor-pointer text-amber-200">Set up cloud data instead</summary>
                        <p className="mt-2">Update the declared data paths in Task & Data Setup so they point to files or folders that exist locally.</p>
                        <button
                          type="button"
                          onClick={() => onOpenTaskConfig('data')}
                          className="mt-2 underline underline-offset-2"
                        >
                          Review task data paths
                        </button>
                      </details>
                    </div>
                  )}
                </div>
              )}

              <div className="grid grid-cols-3 gap-1.5 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
                <button
                  type="button"
                  onClick={() => setExecMode('local')}
                  className={`py-1.5 px-2 rounded text-xs font-medium transition flex items-center justify-center gap-1 ${
                    execMode === 'local'
                      ? 'bg-zinc-800 text-white shadow'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Local CPU</span>
                </button>
                <button
                  type="button"
                  onClick={() => setExecMode('kaggle')}
                  disabled={isCustomTaskPlugin}
                  className={`py-1.5 px-2 rounded text-xs font-medium transition flex items-center justify-center gap-1 ${
                    execMode === 'kaggle'
                      ? 'bg-zinc-800 text-white shadow'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Globe className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Kaggle GPU</span>
                </button>
                <button
                  type="button"
                  onClick={() => setExecMode('modal')}
                  disabled={isCustomTaskPlugin}
                  className={`py-1.5 px-2 rounded text-xs font-medium transition flex items-center justify-center gap-1 ${
                    execMode === 'modal'
                      ? 'bg-zinc-800 text-white shadow'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Cloud className="w-3.5 h-3.5 text-sky-400" />
                  <span>Modal GPU</span>
                </button>
              </div>

              {isCustomTaskPlugin && (
                <p className="p-2.5 bg-amber-950/30 border border-amber-800/60 rounded-lg text-xs text-amber-200">
                  Remote runs are disabled for task plugins until the adapter is installed in the remote runtime. Local runs use the selected adapter.
                </p>
              )}

              {execMode === 'kaggle' ? (
                <div>
                  <label className="block text-xs text-zinc-400 mb-1">Kaggle Cloud GPU:</label>
                  <select
                    value={kaggleGpu}
                    onChange={(e) => setKaggleGpu(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200"
                  >
                    <option value="T4 x 2">2 x NVIDIA T4 (Free 30h/week)</option>
                    <option value="P100">1 x NVIDIA P100 (16GB VRAM)</option>
                  </select>
                </div>
              ) : (
                <div>
                  <label className="block text-xs text-zinc-400 mb-1">Modal Serverless GPU:</label>
                  <select
                    value={modalGpu}
                    onChange={(e) => setModalGpu(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200"
                  >
                    <option value="A10G">NVIDIA A10G (24GB VRAM)</option>
                    <option value="A100-40GB">NVIDIA A100 (40GB VRAM)</option>
                    <option value="H100">NVIDIA H100 (80GB VRAM)</option>
                    <option value="T4">NVIDIA T4 (16GB VRAM)</option>
                  </select>
                </div>
              )}

              {activeJob?.status === 'RUNNING' ? (
                <button
                  type="button"
                  onClick={handleStopJob}
                  disabled={isStopping}
                  className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>{isStopping ? 'Stopping...' : 'Stop Job'}</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleRunExperiment}
                  disabled={isLaunching || (execMode !== 'local' && (checkingCloudData || cloudDataStatus?.ready !== true))}
                  className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>
                    {isLaunching
                      ? 'Starting...'
                      : execMode === 'local'
                        ? 'Run locally'
                        : `Run on ${execMode === 'kaggle' ? 'Kaggle' : 'Modal'}`}
                  </span>
                </button>
              )}
            </div>

            {execMode !== 'local' && !isCustomTaskPlugin && (
              <details className="bg-zinc-950 border border-zinc-800 rounded-xl p-4">
                <summary className="cursor-pointer text-xs font-semibold text-zinc-200">
                  Advanced: sync task data separately
                </summary>
                <div className="mt-3 space-y-3">
                  <p className="text-xs text-zinc-400">
                    Usually you can skip this. Run will sync the approved task data, monitor the remote job, and save its output locally.
                  </p>
                  <button
                    type="button"
                    onClick={handlePushDataset}
                    disabled={pushingDataset || checkingCloudData || cloudDataStatus?.ready !== true}
                    className="px-3 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-700 rounded-lg text-xs font-medium disabled:opacity-50"
                  >
                    <Upload className="inline w-3.5 h-3.5 mr-1.5" />
                    {pushingDataset ? 'Syncing…' : `Sync separately to ${execMode === 'kaggle' ? 'Kaggle' : 'Modal'}`}
                  </button>
                  {datasetPushLog && (
                    <pre className="p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-xs font-mono text-zinc-300 max-h-28 overflow-y-auto whitespace-pre-wrap">
                      {datasetPushLog}
                    </pre>
                  )}
                </div>
              </details>
            )}
          </div>

          {/* Live Console Logs */}
          {jobLogs.length > 0 && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
              <div className="px-4 py-2 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-mono text-zinc-200">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  <span>Run progress ({currentExpId})</span>
                </div>
                <span className="font-mono text-zinc-400">{activeJob?.status || 'IDLE'}</span>
              </div>
              <div className="p-4 max-h-72 overflow-y-auto font-mono text-xs text-zinc-300 space-y-1 bg-black/70">
                {jobLogs.map((line, idx) => (
                  <div key={idx} className="whitespace-pre-wrap break-all">
                    {line}
                  </div>
                ))}
                <div ref={logEndRef} />
              </div>
            </div>
          )}

          {activeJob?.status === 'COMPLETED' && (
            <div className="p-3 bg-emerald-950/30 border border-emerald-800/60 rounded-lg text-xs text-emerald-200">
              Run complete. Results were saved to <code>{artifactDir}/{activeJob.exp_id}/</code>. Open the leaderboard or experiment details to review the metrics before deciding what to try next.
            </div>
          )}
        </div>
      )}

      {/* =====================================================================
          STEP 4: BUILD SUBMISSION, RECORD SCORE & LOOP TO STEP 2
         ===================================================================== */}
      {activeStep === 4 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Left Card: Package Submission according to the task adapter */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5 space-y-4 shadow-xl">
            <div className="border-b border-zinc-800 pb-3">
              <h2 className="text-base font-semibold text-zinc-100">
                1. Build Submission for {taskInfo?.platform?.toUpperCase() || 'Configured Platform'}
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                {isCustomTaskPlugin
                  ? pluginCanSubmit
                    ? 'Use Submissions Hub to provide test-set predictions; the task plugin builds and validates the official package.'
                    : `Task plugin “${taskInfo?.task_plugin?.id}” does not expose both submission builder and validator hooks yet.`
                  : <>Formats the configured predictions from <code>{artifactDir}/{currentExpId}/</code> using the declared task contract and runs the configured audit.</>}
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs text-zinc-400 mb-1">Experiment to Package:</label>
                <select
                  value={selectedConfig}
                  onChange={(e) => setSelectedConfig(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-zinc-200"
                >
                  {configs.map((c) => (
                    <option key={c} value={c}>
                      {c.replace(/^configs\//, '').replace(/\.ya?ml$/, '')}
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                onClick={handlePackageSubmission}
                disabled={packagingSub || isCustomTaskPlugin}
                className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
              >
                <FileArchive className="w-4 h-4" />
                <span>
                  {packagingSub
                    ? 'Building & validating package...'
                    : isCustomTaskPlugin
                      ? 'Use Submissions Hub for plugin submissions'
                      : `Build Submission (${subSpec.file_name || 'submission.zip'})`}
                </span>
              </button>

              {packageMsg && (
                <div className="p-3 bg-emerald-950/30 border border-emerald-800/60 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>{packageMsg}</span>
                </div>
              )}

              {packageErr && (
                <div className="p-3 bg-rose-950/30 border border-rose-800/60 rounded-lg text-xs text-rose-300 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{packageErr}</span>
                </div>
              )}

              {/* Download & Open Competition Portal Buttons */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                {activeSubmissionId && (
                  <a
                    href={api.getSubmissionDownloadUrl(activeSubmissionId)}
                    download
                    className="py-2.5 px-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-100 border border-zinc-700 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
                  >
                    <Download className="w-4 h-4 text-emerald-400" />
                    <span>Download {activeSubObj?.archive_path?.split(/[\\/]/).pop() || 'submission.zip'}</span>
                  </a>
                )}

                <a
                  href={portalUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="py-2.5 px-3 bg-zinc-900 hover:bg-zinc-800 text-cyan-300 border border-cyan-500/30 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>Open {taskInfo?.platform ? taskInfo.platform.toUpperCase() : 'Competition'} Portal</span>
                </a>
              </div>
            </div>
          </div>

          {/* Right Card: Record Official LB Score & Note -> Loop Back to Step 2 */}
          <form
            onSubmit={handleSaveScoreAndLoop}
            className="bg-zinc-950 border border-zinc-800 rounded-xl p-5 space-y-4 shadow-xl"
          >
            <div className="border-b border-zinc-800 pb-3">
              <h2 className="text-base font-semibold text-zinc-100">
                2. Record {taskInfo?.platform ? taskInfo.platform.toUpperCase() : 'Official'} LB Score & Loop to Next Proposal
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                Enter the score returned by the evaluation platform and your observations. Saving updates <code>SUBMISSIONS.md</code> and immediately injects the feedback into Step 2 for the Agent's next proposal.
              </p>
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">Select Submission Package:</label>
              <select
                value={activeSubmissionId}
                onChange={(e) => setActiveSubmissionId(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-zinc-200"
              >
                {submissions.length === 0 && <option value="">No packaged submissions yet</option>}
                {submissions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id} — {s.exp_id} ({s.created_at})
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <div>
                <label className="block text-xs text-zinc-400 mb-1">
                  {taskInfo?.primary_metric || 'Primary Metric'} Score
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={lbPrimaryScore}
                  onChange={(e) => setLbPrimaryScore(e.target.value)}
                  placeholder="0.0000"
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-emerald-400"
                />
              </div>
              {taskInfo?.subtasks?.map((st) => (
                <div key={st.id}>
                  <label className="block text-xs text-zinc-400 mb-1 truncate" title={st.name}>
                    {st.name || st.id}
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={subtaskScores[st.id] || ''}
                    onChange={(e) => setSubtaskScores((prev) => ({ ...prev, [st.id]: e.target.value }))}
                    placeholder="Optional"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-zinc-200"
                  />
                </div>
              ))}
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">
                Diagnostic Note for Next Agent Proposal:
              </label>
              <textarea
                rows={3}
                value={lbNotes}
                onChange={(e) => setLbNotes(e.target.value)}
                placeholder="e.g. Leaderboard reached 0.842 (+0.018 vs baseline). Slices with rare classes still low recall — propose focal loss or feature weighting next."
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            {scoreSavedMsg && (
              <div className="p-3 bg-emerald-950/30 border border-emerald-800/60 rounded-lg text-xs text-emerald-300">
                {scoreSavedMsg}
              </div>
            )}

            <button
              type="submit"
              disabled={savingScore}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
            >
              <RotateCcw className="w-4 h-4" />
              <span>
                {savingScore
                  ? 'Saving Score & Updating Prompt...'
                  : 'Save Score & Loop to Step 2 (Propose Next)'}
              </span>
            </button>
          </form>
        </div>
      )}
    </div>
  );
};
