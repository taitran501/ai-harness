import React, { useState, useEffect, useRef } from 'react';
import {
  Sliders,
  Play,
  Square,
  Upload,
  Download,
  RefreshCw,
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
import { useLanguage } from '../constants/i18n';

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
  const { t } = useLanguage();
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
  const [smokeTestOnly, setSmokeTestOnly] = useState<boolean>(false);
  const [modalGpu, setModalGpu] = useState<string>('A10G');
  const [kaggleGpu, setKaggleGpu] = useState<string>('T4 x 2');
  const [envStatus, setEnvStatus] = useState<EnvironmentStatus | null>(null);

  // Dataset push state
  const [pushingDataset, setPushingDataset] = useState<boolean>(false);
  const [datasetPushLog, setDatasetPushLog] = useState<string | null>(null);

  // Running job & logs
  const [activeJob, setActiveJob] = useState<CloudJob | null>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [isLaunching, setIsLaunching] = useState<boolean>(false);
  const [isStopping, setIsStopping] = useState<boolean>(false);

  // Kaggle status & Pull output state
  const [checkingStatus, setCheckingStatus] = useState<boolean>(false);
  const [kernelStatusText, setKernelStatusText] = useState<string | null>(null);
  const [pullingOutput, setPullingOutput] = useState<boolean>(false);
  const [pulledResult, setPulledResult] = useState<{
    exp_id: string;
    migrated_files: string[];
    metrics?: Record<string, any>;
    raw_output: string;
  } | null>(null);

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
    setDatasetPushLog(`Syncing dataset/ and splits/ to ${targetPlatform.toUpperCase()}...`);
    try {
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
    setJobLogs([`>>> Starting [${execMode.toUpperCase()}] execution for ${selectedConfig}...`]);
    setPulledResult(null);

    try {
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

  // Step 3C: Check Kaggle Status & Pull Output from Cloud
  const handleCheckStatus = async () => {
    setCheckingStatus(true);
    try {
      const res = await api.getKaggleKernelStatus({
        config_path: selectedConfig,
        exp_id: currentExpId,
      });
      setKernelStatusText(`[${res.status}] ${res.raw_output}`);
    } catch (err: any) {
      setKernelStatusText(`Status check failed: ${err.message}`);
    } finally {
      setCheckingStatus(false);
    }
  };

  const handlePullOutput = async () => {
    setPullingOutput(true);
    try {
      const targetPlatform = execMode === 'modal' ? 'modal' : 'kaggle';
      const res = await api.pullCloudOutput({
        platform: targetPlatform,
        job_id: activeJob?.job_id || '',
        config_path: selectedConfig,
        exp_id: currentExpId,
      });
      setPulledResult(res);
      await onRefreshAll();
    } catch (err: any) {
      setKernelStatusText(`Pull output failed: ${err.message}`);
    } finally {
      setPullingOutput(false);
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

  return (
    <div className="space-y-5">
      {/* 4-Step Pipeline Stepper Bar */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-3 shadow-lg">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5">
          {[
            {
              step: 1 as const,
              title: t('pipe_step1_title', 'Step 1. Task & Data Setup'),
              desc: `${taskInfo?.platform?.toUpperCase() || 'Platform not set'} / ${dataCfg.split_strategy || 'Validation not set'}`,
            },
            {
              step: 2 as const,
              title: t('pipe_step2_title', 'Step 2. Agent Prompt & Propose'),
              desc: t('pipe_target_label', 'Target: {target}', { target: currentExpId }),
            },
            {
              step: 3 as const,
              title: t('pipe_step3_title', 'Step 3. Run & Retrieve Outputs'),
              desc: isCustomTaskPlugin ? t('pipe_step3_desc_custom', 'Run locally through the installed task plugin') : t('pipe_step3_desc_general', 'Local CPU or Cloud GPU (Kaggle / Modal)'),
            },
            {
              step: 4 as const,
              title: t('pipe_step4_title', 'Step 4. Submission & Score'),
              desc: isCustomTaskPlugin && !pluginCanSubmit
                ? t('pipe_step4_desc_missing', 'Submission hooks are missing from the selected task plugin')
                : t('pipe_step4_desc_ready', 'Build .{format} package, record score & continue', { format: subSpec.archive_format || 'zip' }),
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
                {t('pipe_step1_heading', 'Step 1. Competition Task, Data Schema & Validation Setup')}
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                {t('pipe_step1_desc', 'Verify dataset paths, input/target columns, validation split, and submission format before generating the Agent prompt.')}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onOpenTaskConfig('task')}
                className="flex items-center gap-1.5 px-3.5 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 rounded-lg border border-zinc-700 text-xs font-medium transition"
              >
                <Sliders className="w-4 h-4 text-indigo-400" />
                <span>{t('pipe_edit_full_setup', 'Edit Full Setup')}</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveStep(2)}
                className="flex items-center gap-1.5 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition"
              >
                <span>{t('pipe_next_step2', 'Next: Agent Prompt & Propose')}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Summary Cards for Task, Data Schema, CV Protocol, and Submission Portal */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Card 1: Task & Metric */}
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200">{t('pipe_task_eval_obj', '1. Task & Evaluation Objective')}</span>
                <button
                  type="button"
                  onClick={() => onOpenTaskConfig('task')}
                  className="text-xs text-indigo-400 hover:underline"
                >
                  {t('pipe_btn_configure', 'Configure')}
                </button>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">{t('pipe_field_task', 'Task:')} </span>
                  <span className="text-zinc-200">{taskInfo?.name || 'Competition Task'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_platform', 'Platform:')} </span>
                  <span className="text-zinc-200 uppercase">{taskInfo?.platform || 'not configured'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_primary_metric', 'Primary Metric:')} </span>
                  <span className="text-emerald-400">{taskInfo?.primary_metric || 'Not configured'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_sub_file', 'Submission File:')} </span>
                  <span className="text-zinc-200">
                    {subSpec.file_name || 'submission.zip'} (.{subSpec.archive_format || 'zip'})
                  </span>
                </div>
              </div>
            </div>

            {/* Card 2: Data Schema & Strategy */}
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200">{t('pipe_data_schema_strat', '2. Data Schema & Strategy')}</span>
                <button
                  type="button"
                  onClick={() => onOpenTaskConfig('data')}
                  className="text-xs text-indigo-400 hover:underline"
                >
                  {t('pipe_btn_configure', 'Configure')}
                </button>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">{t('pipe_field_train_test', 'Train / Test Path:')} </span>
                  <span className="text-zinc-200">
                    {dataCfg.train_path || 'dataset/'} , {dataCfg.test_path || 'dataset/test/'}
                  </span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_id_col', 'ID Column:')} </span>
                  <span className="text-zinc-200">{dataCfg.id_column || 'id'}</span>
                </div>
                <div className="truncate" title={dataCfg.target_column}>
                  <span className="text-zinc-500">{t('pipe_field_target_col', 'Target Columns:')} </span>
                  <span className="text-zinc-200">
                    {dataCfg.target_column || 'target'}
                  </span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_missing_policy', 'Missing Policy:')} </span>
                  <span className="text-zinc-200">{dataCfg.missing_data_policy || 'fill_empty_context'}</span>
                </div>
              </div>
            </div>

            {/* Card 3: Validation Split */}
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-xl space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200">{t('pipe_validation_split', '3. Validation Split')}</span>
                <button
                  type="button"
                  onClick={() => onOpenTaskConfig('cv')}
                  className="text-xs text-indigo-400 hover:underline"
                >
                  {t('pipe_btn_configure', 'Configure')}
                </button>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">{t('pipe_field_split_method', 'Split Method:')} </span>
                  <span className="text-zinc-200">{dataCfg.split_strategy || 'Not declared'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_group_col', 'Group Column:')} </span>
                  <span className="text-amber-300">{dataCfg.group_column || 'Not declared'}</span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_folds_seed', 'Folds / Seed:')} </span>
                  <span className="text-zinc-200">
                    {dataCfg.n_folds ?? 'Not declared'} folds (seed={dataCfg.seed ?? 'Not declared'})
                  </span>
                </div>
                <div>
                  <span className="text-zinc-500">{t('pipe_field_fold_file', 'Fold File:')} </span>
                  <span className="text-zinc-200">{dataCfg.split_file || dataCfg.fold_file || 'Not declared'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Competition Portal URL Bar */}
          <div className="p-4 bg-zinc-900/40 border border-zinc-800 rounded-xl flex flex-wrap items-center justify-between gap-3">
            <div className="flex-1 min-w-[260px]">
              <label className="block text-xs font-semibold text-zinc-300 mb-1">
                {t('pipe_portal_url_label', 'Competition Submission Portal URL:')}
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
              <span>{t('pipe_proceed_step2', 'Proceed to Step 2: Agent Prompt & Propose')}</span>
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
                {t('pipe_step2_active_config', 'Active Proposed Experiment Config (configs/*.yaml)')}
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
                  {t('pipe_inspect_edit_yaml', 'Inspect / Edit YAML')}
                </button>
                <button
                  type="button"
                  onClick={() => onOpenConfigModal('create')}
                  className="px-3 py-1.5 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 rounded-lg border border-indigo-500/40 text-xs font-semibold transition"
                >
                  {t('pipe_propose_new_config', '+ Propose New Config')}
                </button>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setActiveStep(3)}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition flex items-center gap-2"
            >
              <span>{t('pipe_next_step3', 'Next: Run Experiment (Step 3)')}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Agent Context & Prompt Generator (Includes Validation & External Score Feedback) */}
          <PrimingCenter initialPrompt={initialPrompt} taskInfo={taskInfo} />
        </div>
      )}

      {/* =====================================================================
          STEP 3: EXECUTE (LOCAL CPU / CLOUD GPU), PUSH DATASET & PULL OUTPUT
         ===================================================================== */}
      {activeStep === 3 && (
        <div className="space-y-4">
          {/* Top Cloud Account Status Bar */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <span className="font-semibold text-zinc-300">{t('pipe_cloud_accounts', 'Optional Cloud Accounts:')}</span>
              <span className="px-2.5 py-1 rounded border bg-zinc-900 border-zinc-800 text-zinc-300 font-mono">
                Kaggle: <strong>{envStatus?.kaggle.username || t('pipe_not_logged_in', 'Not logged in')}</strong>
              </span>
              <span className="px-2.5 py-1 rounded border bg-zinc-900 border-zinc-800 text-zinc-300 font-mono">
                Modal: <strong>{envStatus?.modal.active_profile || t('pipe_not_configured', 'Not configured')}</strong>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onOpenEnvModal}
                className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 rounded-lg border border-zinc-800 text-xs font-medium transition"
              >
                {t('pipe_btn_account_settings', 'Kaggle Login / Account Settings')}
              </button>
              <button
                type="button"
                onClick={() => setActiveStep(4)}
                className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition flex items-center gap-1.5"
              >
                <span>{t('pipe_next_step4', 'Next: Package & Submit (Step 4)')}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Card 3.1: Push Dataset to Cloud (1-Click Sync) */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 space-y-3">
              <div className="text-sm font-semibold text-zinc-100 border-b border-zinc-800 pb-2">
                {t('pipe_sync_data_cloud', '1. Sync Dataset & Splits to Cloud')}
              </div>
              {isCustomTaskPlugin && (
                <p className="text-xs text-amber-200">
                  {t('pipe_dataset_upload_disabled', 'Dataset upload is disabled until the selected task adapter is available in the remote runtime.')}
                </p>
              )}
              <p className="text-xs text-zinc-400">
                {t('pipe_sync_data_desc', 'Push declared task data and splits to a private resource named from this task and its current data fingerprint. Every subsequent run mounts the matching resource.')}
              </p>

              <button
                type="button"
                onClick={handlePushDataset}
                disabled={pushingDataset || isCustomTaskPlugin}
                className="w-full py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-100 border border-zinc-700 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
              >
                <Upload className={`w-3.5 h-3.5 ${pushingDataset ? 'animate-bounce' : ''}`} />
                <span>
                  {pushingDataset
                    ? t('pipe_btn_pushing_data', 'Pushing Dataset & Splits...')
                    : t('pipe_btn_push_data', 'Push Dataset to {target}', { target: execMode === 'modal' ? 'Modal Volume' : 'Kaggle Datasets' })}
                </span>
              </button>

              {datasetPushLog && (
                <pre className="p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-xs font-mono text-zinc-300 max-h-28 overflow-y-auto whitespace-pre-wrap">
                  {datasetPushLog}
                </pre>
              )}
            </div>

            {/* Card 3.2: Choose Execution Mode & Run */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 space-y-3">
              <div className="text-sm font-semibold text-zinc-100 border-b border-zinc-800 pb-2">
                {t('pipe_exec_title', '2. Execute Experiment ({exp_id})', { exp_id: currentExpId })}
              </div>

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
                  <span>{t('pipe_mode_local', 'Local CPU')}</span>
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
                  <span>{t('pipe_mode_kaggle', 'Kaggle GPU')}</span>
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
                  <span>{t('pipe_mode_modal', 'Modal GPU')}</span>
                </button>
              </div>

              {isCustomTaskPlugin && (
                <p className="p-2.5 bg-amber-950/30 border border-amber-800/60 rounded-lg text-xs text-amber-200">
                  {t('pipe_remote_disabled_note', 'Remote runs are disabled for task plugins until the adapter is installed in the remote runtime. Local runs use the selected adapter.')}
                </p>
              )}

              {execMode === 'local' ? (
                <div className="p-2.5 bg-zinc-900/60 border border-zinc-800 rounded-lg space-y-2 text-xs text-zinc-300">
                  <div>
                    {t('pipe_local_desc', 'Runs the configured experiment through the selected task adapter on this machine.')}
                  </div>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={smokeTestOnly}
                      onChange={(e) => setSmokeTestOnly(e.target.checked)}
                      className="rounded bg-zinc-950 border-zinc-700"
                    />
                    <span>{t('pipe_smoke_test_chk', 'Quick Smoke Test (--debug --fold 0)')}</span>
                  </label>
                </div>
              ) : execMode === 'kaggle' ? (
                <div>
                  <label className="block text-xs text-zinc-400 mb-1">{t('pipe_kaggle_gpu_label', 'Kaggle Cloud GPU:')}</label>
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
                  <label className="block text-xs text-zinc-400 mb-1">{t('pipe_modal_gpu_label', 'Modal Serverless GPU:')}</label>
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
                  <span>{isStopping ? t('pipe_btn_stopping', 'Stopping...') : t('pipe_btn_stop_job', 'Stop Job')}</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleRunExperiment}
                  disabled={isLaunching}
                  className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isLaunching ? t('pipe_btn_starting', 'Starting...') : t('pipe_btn_run', 'Run')}</span>
                </button>
              )}
            </div>

            {/* Card 3.3: Monitor Cloud Status & Pull Output to Local */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 space-y-3">
              <div className="text-sm font-semibold text-zinc-100 border-b border-zinc-800 pb-2">
                {t('pipe_monitor_pull_title', '3. Monitor & Pull Output to Local')}
              </div>
              <p className="text-xs text-zinc-400">
                {t('pipe_monitor_pull_desc', 'Check remote job progress and pull run outputs into {dir}/{exp_id}/.', { dir: artifactDir, exp_id: currentExpId })}
              </p>

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={handleCheckStatus}
                  disabled={checkingStatus}
                  className="py-2 px-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${checkingStatus ? 'animate-spin' : ''}`} />
                  <span>{t('pipe_btn_check_status', 'Check Status')}</span>
                </button>

                <button
                  type="button"
                  onClick={handlePullOutput}
                  disabled={pullingOutput}
                  className="py-2 px-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>{pullingOutput ? t('pipe_btn_pulling', 'Pulling...') : t('pipe_btn_pull_output', 'Pull Output')}</span>
                </button>
              </div>

              {kernelStatusText && (
                <div className="p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-xs font-mono text-zinc-300">
                  {kernelStatusText}
                </div>
              )}

              {pulledResult && (
                <div className="p-3 bg-emerald-950/30 border border-emerald-800/60 rounded-lg space-y-1.5 text-xs">
                  <div className="font-semibold text-emerald-300 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Pulled {pulledResult.migrated_files.length} files to {artifactDir}/{pulledResult.exp_id}/</span>
                  </div>
                  <div className="font-mono text-zinc-300">
                    Files: {pulledResult.migrated_files.join(', ') || 'None'}
                  </div>
                  {pulledResult.metrics && (
                    <div className="font-mono text-emerald-400">
                      Local CV Score: {JSON.stringify(pulledResult.metrics.overall || pulledResult.metrics)}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Live Console Logs */}
          {jobLogs.length > 0 && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
              <div className="px-4 py-2 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-mono text-zinc-200">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  <span>{t('pipe_log_stream_title', 'Execution Log Stream ({exp_id})', { exp_id: currentExpId })}</span>
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
                {t('pipe_step4_heading', '1. Build Submission for {platform}', { platform: taskInfo?.platform?.toUpperCase() || 'Configured Platform' })}
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                {isCustomTaskPlugin
                  ? pluginCanSubmit
                    ? t('pipe_step4_plugin_can_submit', 'Use Submissions Hub to provide test-set predictions; the task plugin builds and validates the official package.')
                    : t('pipe_step4_plugin_cannot_submit', 'Task plugin “{plugin}” does not expose both submission builder and validator hooks yet.', { plugin: taskInfo?.task_plugin?.id || '' })
                  : t('pipe_step4_default_desc', 'Formats the configured predictions from {dir}/{exp_id}/ using the declared task contract and runs the configured audit.', { dir: artifactDir, exp_id: currentExpId })}
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs text-zinc-400 mb-1">{t('pipe_exp_to_package', 'Experiment to Package:')}</label>
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
                    ? t('pipe_btn_building_package', 'Building & validating package...')
                    : isCustomTaskPlugin
                      ? t('pipe_btn_use_sub_hub', 'Use Submissions Hub for plugin submissions')
                      : t('pipe_btn_build_sub', 'Build Submission ({filename})', { filename: subSpec.file_name || 'submission.zip' })}
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
                    <span>{t('pipe_btn_download_sub', 'Download {filename}', { filename: activeSubObj?.archive_path?.split(/[\\/]/).pop() || 'submission.zip' })}</span>
                  </a>
                )}

                <a
                  href={portalUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="py-2.5 px-3 bg-zinc-900 hover:bg-zinc-800 text-cyan-300 border border-cyan-500/30 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>{t('pipe_btn_open_portal', 'Open {platform} Portal', { platform: taskInfo?.platform ? taskInfo.platform.toUpperCase() : 'Competition' })}</span>
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
                {t('pipe_record_score_heading', '2. Record {platform} LB Score & Loop to Next Proposal', { platform: taskInfo?.platform ? taskInfo.platform.toUpperCase() : 'Official' })}
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                {t('pipe_record_score_desc', 'Enter the score returned by the evaluation platform and your observations. Saving updates SUBMISSIONS.md and immediately injects the feedback into Step 2 for the Agent next proposal.')}
              </p>
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">{t('pipe_select_sub_package', 'Select Submission Package:')}</label>
              <select
                value={activeSubmissionId}
                onChange={(e) => setActiveSubmissionId(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-zinc-200"
              >
                {submissions.length === 0 && <option value="">{t('pipe_no_packaged_subs', 'No packaged submissions yet')}</option>}
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
                  {t('pipe_metric_score_label', '{metric} Score', { metric: taskInfo?.primary_metric || 'Primary Metric' })}
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
                {t('pipe_diagnostic_note_label', 'Diagnostic Note for Next Agent Proposal:')}
              </label>
              <textarea
                rows={3}
                value={lbNotes}
                onChange={(e) => setLbNotes(e.target.value)}
                placeholder={t('pipe_diagnostic_note_ph', 'e.g. Leaderboard reached 0.842 (+0.018 vs baseline). Slices with rare classes still low recall — propose focal loss or feature weighting next.')}
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
                  ? t('pipe_btn_saving_score', 'Saving Score & Updating Prompt...')
                  : t('pipe_btn_save_score_loop', 'Save Score & Loop to Step 2 (Propose Next)')}
              </span>
            </button>
          </form>
        </div>
      )}
    </div>
  );
};
