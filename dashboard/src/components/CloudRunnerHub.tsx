import React, { useState, useEffect, useRef } from 'react';
import {
  Cloud,
  Cpu,
  Globe,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  Play,
  Terminal,
  RefreshCw,
  FileCode,
  HardDrive,
  FolderSync,
  Square,
  Server,
  ShieldCheck,
  ShieldAlert,
  Sliders,
  Sparkles,
} from 'lucide-react';
import {
  TaskInfo,
  CloudGenerateResult,
  CloudPreflightResult,
  CloudJob,
  EnvironmentStatus,
} from '../types/experiment';
import { api } from '../services/api';
import { CloudEnvironmentModal } from './CloudEnvironmentModal';

interface CloudRunnerHubProps {
  taskInfo?: TaskInfo | null;
  onRefreshSolutionTree: () => Promise<void>;
  onOpenConfigModal?: (pathOrAction?: string) => void;
}

export const CloudRunnerHub: React.FC<CloudRunnerHubProps> = ({
  taskInfo,
  onRefreshSolutionTree,
  onOpenConfigModal,
}) => {
  const [platform, setPlatform] = useState<'modal' | 'kaggle' | 'local'>('modal');
  const [configs, setConfigs] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>('configs/exp001_tfidf_baseline.yaml');

  // Environment & Multi-Account state
  const [envStatus, setEnvStatus] = useState<EnvironmentStatus | null>(null);
  const [isEnvModalOpen, setIsEnvModalOpen] = useState(false);

  // Modal Settings
  const [modalGpu, setModalGpu] = useState<string>('A10G');
  const [modalVolume, setModalVolume] = useState<string>('dataset-volume');

  // Kaggle Settings
  const [kaggleGpu, setKaggleGpu] = useState<string>('T4 x 2');
  const [kaggleDatasetSlug, setKaggleDatasetSlug] = useState<string>('antigravity/dataset');

  // Execution & Audit Controls
  const [autoAudit, setAutoAudit] = useState<boolean>(true);
  const [autoSync, setAutoSync] = useState<boolean>(true);

  // Running Job State
  const [activeJob, setActiveJob] = useState<CloudJob | null>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [isLaunching, setIsLaunching] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [auditReceipt, setAuditReceipt] = useState<any | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Script Generator & Preflight State
  const [generating, setGenerating] = useState(false);
  const [generatedResult, setGeneratedResult] = useState<CloudGenerateResult | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [preflightResult, setPreflightResult] = useState<CloudPreflightResult | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [syncingTree, setSyncingTree] = useState(false);
  const [syncSuccessMsg, setSyncSuccessMsg] = useState<string | null>(null);

  const terminalEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const timerRef = useRef<any>(null);

  const fetchEnvironment = async () => {
    try {
      const data = await api.getEnvironmentStatus();
      setEnvStatus(data);
    } catch (err) {
      console.warn('Failed to load environment status:', err);
    }
  };

  useEffect(() => {
    fetchEnvironment();
    api
      .getConfigs()
      .then((res) => {
        if (res.configs && res.configs.length > 0) {
          setConfigs(res.configs);
          setSelectedConfig(res.configs[0]);
        }
      })
      .catch((err) => console.warn('Could not load configs:', err));

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [jobLogs]);

  // Elapsed timer
  useEffect(() => {
    if (activeJob?.status === 'RUNNING') {
      timerRef.current = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [activeJob?.status]);

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2500);
  };

  const handlePreflight = async () => {
    setInspecting(true);
    try {
      const res = await api.runCloudPreflight(selectedConfig);
      setPreflightResult(res);
    } catch (err: any) {
      setPreflightResult({
        passed: false,
        errors: [err.message || 'Inspection failed.'],
      });
    } finally {
      setInspecting(false);
    }
  };

  const handleGenerate = async () => {
    if (platform === 'local') return;
    setGenerating(true);
    setGenerateError(null);
    try {
      const res = await api.generateCloudRunner({
        backend: platform,
        config_path: selectedConfig,
        gpu: platform === 'modal' ? modalGpu : kaggleGpu,
        dataset_slug: kaggleDatasetSlug,
        volume_name: modalVolume,
      });
      setGeneratedResult(res);
    } catch (err: any) {
      console.error(err);
      setGenerateError(err.message || 'Failed to generate cloud runner script.');
    } finally {
      setGenerating(false);
    }
  };

  // Launch 1-Click Execution & Stream Logs
  const handleLaunchExecution = async () => {
    setIsLaunching(true);
    setJobLogs([]);
    setAuditReceipt(null);
    setElapsedSeconds(0);

    try {
      // 1. Run Pre-flight first
      const preflight = await api.runCloudPreflight(selectedConfig);
      setPreflightResult(preflight);
      if (!preflight.passed) {
        setIsLaunching(false);
        return;
      }

      // 2. Launch background job via API
      const res = await api.runCloudJob({
        backend: platform,
        config_path: selectedConfig,
        gpu: platform === 'modal' ? modalGpu : kaggleGpu,
        auto_audit: autoAudit,
        auto_sync: autoSync,
      });

      const job = res.job;
      setActiveJob(job);
      setIsLaunching(false);

      // 3. Connect to live SSE log stream
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource(`/api/cloud/logs/stream?job_id=${encodeURIComponent(job.job_id)}`);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.text) {
            setJobLogs((prev) => [...prev, parsed.text]);
          }
          if (parsed.status) {
            setActiveJob((prev) => (prev ? { ...prev, status: parsed.status } : null));
          }
        } catch {
          setJobLogs((prev) => [...prev, event.data]);
        }
      };

      es.addEventListener('end', (event: any) => {
        try {
          const endData = JSON.parse(event.data);
          setActiveJob((prev) =>
            prev
              ? {
                  ...prev,
                  status: endData.status,
                  return_code: endData.return_code,
                  audit_passed: endData.audit_passed,
                  audit_receipt: endData.audit_receipt,
                }
              : null
          );
          if (endData.audit_receipt) {
            setAuditReceipt(endData.audit_receipt);
          }
        } catch (e) {
          console.warn('Error parsing end event:', e);
        }
        es.close();
        onRefreshSolutionTree();
      });

      es.onerror = () => {
        es.close();
      };
    } catch (err: any) {
      console.error('Launch failed:', err);
      setIsLaunching(false);
      setJobLogs((prev) => [...prev, `[ERROR] Launch failed: ${err.message}`]);
    }
  };

  const handleStopExecution = async () => {
    if (!activeJob) return;
    setIsStopping(true);
    try {
      await api.stopCloudJob(activeJob.job_id);
      setActiveJob((prev) => (prev ? { ...prev, status: 'CANCELLED' } : null));
    } catch (err: any) {
      console.error('Stop error:', err);
    } finally {
      setIsStopping(false);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    }
  };

  const handleManualAudit = async () => {
    if (!activeJob) return;
    try {
      const res = await api.runCloudAudit(activeJob.exp_id);
      if (res.receipt) {
        setAuditReceipt(res.receipt);
      }
      onRefreshSolutionTree();
    } catch (err) {
      console.error('Manual audit failed:', err);
    }
  };

  const handleSyncTree = async () => {
    setSyncingTree(true);
    try {
      await onRefreshSolutionTree();
      setSyncSuccessMsg('Artifacts Synced to Solution Tree!');
      setTimeout(() => setSyncSuccessMsg(null), 3000);
    } catch (err) {
      console.error(err);
    } finally {
      setSyncingTree(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Environment Readiness Banner */}
      <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-3 shadow-xl flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4 text-indigo-400" />
          <span className="text-xs font-semibold text-zinc-200">System Environment & Hardware:</span>
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
            {/* Modal Pill */}
            <span
              className={`px-2 py-0.5 rounded border ${
                envStatus?.modal.installed
                  ? 'bg-sky-500/10 text-sky-300 border-sky-500/30'
                  : 'bg-zinc-900 text-zinc-500 border-zinc-800'
              }`}
            >
              Modal: <strong>{envStatus?.modal.active_profile || 'None'}</strong>
            </span>

            {/* Kaggle Pill */}
            <span
              className={`px-2 py-0.5 rounded border ${
                envStatus?.kaggle.authenticated
                  ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30'
                  : 'bg-zinc-900 text-zinc-500 border-zinc-800'
              }`}
            >
              Kaggle: <strong>{envStatus?.kaggle.username || 'Not auth'}</strong>
            </span>

            {/* GPU Pill */}
            <span
              className={`px-2 py-0.5 rounded border ${
                envStatus?.gpu.detected
                  ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                  : 'bg-zinc-900 text-zinc-500 border-zinc-800'
              }`}
            >
              Local GPU: <strong>{envStatus?.gpu.gpu_name ? `${envStatus.gpu.gpu_name.split(' ')[0]} ${envStatus.gpu.memory_total_mb}MB` : 'CPU'}</strong>
            </span>

            {/* Docker Pill */}
            <span
              className={`px-2 py-0.5 rounded border ${
                envStatus?.docker.running
                  ? 'bg-blue-500/10 text-blue-300 border-blue-500/30'
                  : 'bg-zinc-900 text-zinc-500 border-zinc-800'
              }`}
            >
              Docker: <strong>{envStatus?.docker.running ? 'Running' : 'Standby'}</strong>
            </span>
          </div>
        </div>

        <button
          onClick={() => setIsEnvModalOpen(true)}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white rounded-lg border border-zinc-800 text-xs font-medium transition"
        >
          <Sliders className="w-3.5 h-3.5 text-zinc-400" />
          <span>Account Settings</span>
        </button>
      </div>

      {/* Top Header & Platform Selector */}
      <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <Cloud className="w-4 h-4 text-zinc-400" />
            <span>Cloud Runner Hub</span>
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Launch GPU experiments to Modal.com, Kaggle Kernels, or Local Hardware with real-time log streaming and automated audit gate.
          </p>
        </div>

        {/* Platform Switcher */}
        <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
          <button
            onClick={() => setPlatform('modal')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition ${
              platform === 'modal'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Cloud className="w-3.5 h-3.5 text-sky-400" />
            <span>Modal.com</span>
          </button>
          <button
            onClick={() => setPlatform('kaggle')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition ${
              platform === 'kaggle'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Globe className="w-3.5 h-3.5 text-cyan-400" />
            <span>Kaggle Kernels</span>
          </button>
          <button
            onClick={() => setPlatform('local')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition ${
              platform === 'local'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>Local GPU</span>
          </button>
        </div>
      </div>

      {/* 3 Step Interactive Workflow */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Step 1: Target Experiment Config */}
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 flex items-center justify-center text-xs font-mono font-bold">
                1
              </span>
              <span className="text-xs font-semibold text-zinc-200">Select Target Config</span>
            </div>
            <FileCode className="w-4 h-4 text-zinc-500" />
          </div>

          <div className="space-y-2">
            <label className="block text-[11px] text-zinc-400">Target Config YAML:</label>
            <select
              value={selectedConfig}
              onChange={(e) => setSelectedConfig(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
            >
              {configs.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <div className="flex gap-2 pt-1">
              <button
                onClick={() => onOpenConfigModal?.(selectedConfig)}
                className="flex-1 text-center py-1 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white rounded border border-zinc-800 text-xs font-medium transition"
              >
                Inspect / Edit YAML
              </button>
              <button
                onClick={() => onOpenConfigModal?.('create')}
                className="flex items-center justify-center gap-1 px-3 py-1 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 rounded border border-indigo-500/30 text-xs font-medium transition"
              >
                <span>+ New Target</span>
              </button>
            </div>
          </div>
        </div>

        {/* Step 2: Compute & Options */}
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 flex items-center justify-center text-xs font-mono font-bold">
                2
              </span>
              <span className="text-xs font-semibold text-zinc-200">Hardware & Audit Policy</span>
            </div>
            <Cpu className="w-4 h-4 text-zinc-500" />
          </div>

          <div className="space-y-2.5">
            <div>
              <label className="block text-[11px] text-zinc-400 mb-1">
                Hardware Accelerator:
              </label>
              {platform === 'modal' ? (
                <select
                  value={modalGpu}
                  onChange={(e) => setModalGpu(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                >
                  <option value="A10G">NVIDIA A10G (24GB VRAM) — Recommended</option>
                  <option value="T4">NVIDIA T4 (16GB VRAM) — Budget</option>
                  <option value="A100-40GB">NVIDIA A100 (40GB VRAM) — Fast</option>
                  <option value="H100">NVIDIA H100 (80GB VRAM) — Maximum</option>
                </select>
              ) : platform === 'kaggle' ? (
                <select
                  value={kaggleGpu}
                  onChange={(e) => setKaggleGpu(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                >
                  <option value="T4 x 2">2 x NVIDIA T4 (Free 30h/week)</option>
                  <option value="P100">1 x NVIDIA P100 (16GB)</option>
                </select>
              ) : (
                <div className="p-2 bg-zinc-900 border border-zinc-800 rounded text-xs text-zinc-300 font-mono">
                  {envStatus?.gpu.detected
                    ? `${envStatus.gpu.gpu_name} (${envStatus.gpu.memory_total_mb}MB)`
                    : 'Local CPU Host'}
                </div>
              )}
            </div>

            {/* Checkboxes for Audit Gate & DAG Sync */}
            <div className="space-y-1.5 pt-1 border-t border-zinc-900">
              <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoAudit}
                  onChange={(e) => setAutoAudit(e.target.checked)}
                  className="rounded bg-zinc-900 border-zinc-700 text-indigo-600 focus:ring-0"
                />
                <span>Auto-run Audit Gate (Min Diff &ge; 2%)</span>
              </label>
              <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoSync}
                  onChange={(e) => setAutoSync(e.target.checked)}
                  className="rounded bg-zinc-900 border-zinc-700 text-indigo-600 focus:ring-0"
                />
                <span>Auto-sync Solution Tree DAG on Pass</span>
              </label>
            </div>
          </div>
        </div>

        {/* Step 3: 1-Click Launch & Pre-flight */}
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 flex items-center justify-center text-xs font-mono font-bold">
                3
              </span>
              <span className="text-xs font-semibold text-zinc-200">1-Click Execution</span>
            </div>
            <Terminal className="w-4 h-4 text-zinc-500" />
          </div>

          <div className="space-y-2.5">
            {/* Preflight Check Button */}
            <button
              onClick={handlePreflight}
              disabled={inspecting}
              className="w-full flex items-center justify-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-95 text-zinc-300 hover:text-white py-1 rounded-lg text-xs font-medium border border-zinc-800 transition"
            >
              <RefreshCw className={`w-3 h-3 text-zinc-400 ${inspecting ? 'animate-spin' : ''}`} />
              <span>{inspecting ? 'Verifying Pre-Flight...' : 'Run Pre-Flight Check'}</span>
            </button>

            {preflightResult && (
              <div
                className={`p-2 rounded border text-[11px] font-mono ${
                  preflightResult.passed
                    ? 'bg-emerald-950/30 border-emerald-800/60 text-emerald-300'
                    : 'bg-rose-950/30 border-rose-800/60 text-rose-300'
                }`}
              >
                <div className="flex items-center gap-1.5 font-bold">
                  {preflightResult.passed ? (
                    <>
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Pre-flight Passed</span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                      <span>Pre-flight Failed</span>
                    </>
                  )}
                </div>
                {preflightResult.errors && preflightResult.errors.length > 0 && (
                  <ul className="mt-1 list-disc list-inside space-y-0.5 text-[10px]">
                    {preflightResult.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* Big Launch Button */}
            {activeJob?.status === 'RUNNING' ? (
              <button
                onClick={handleStopExecution}
                disabled={isStopping}
                className="w-full flex items-center justify-center gap-2 bg-rose-600 hover:bg-rose-500 active:scale-95 text-white py-2 rounded-lg text-xs font-semibold shadow-lg transition"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>{isStopping ? 'Terminating Process...' : 'Stop / Terminate Job'}</span>
              </button>
            ) : (
              <button
                onClick={handleLaunchExecution}
                disabled={isLaunching}
                className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-500 to-sky-500 hover:from-indigo-400 hover:to-sky-400 active:scale-95 text-white py-2 rounded-lg text-xs font-semibold shadow-lg shadow-indigo-500/20 transition"
              >
                <Play className={`w-3.5 h-3.5 ${isLaunching ? 'animate-spin' : 'fill-current'}`} />
                <span>
                  {isLaunching
                    ? 'Launching Remote Job...'
                    : `Run 1-Click (${platform.toUpperCase()})`}
                </span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Integrated Live Log Console & Audit Gate Viewer */}
      {(activeJob || jobLogs.length > 0) && (
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl shadow-2xl overflow-hidden space-y-0">
          {/* Console Header Bar */}
          <div className="flex items-center justify-between px-4 py-2.5 bg-zinc-900 border-b border-zinc-800">
            <div className="flex items-center gap-3">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-zinc-100">
                  {activeJob?.job_id || 'Execution Terminal'}
                </span>
                {activeJob?.status && (
                  <span
                    className={`text-[10px] font-mono uppercase font-bold px-1.5 py-0.5 rounded border ${
                      activeJob.status === 'RUNNING'
                        ? 'bg-sky-500/20 text-sky-300 border-sky-500/40 animate-pulse'
                        : activeJob.status === 'COMPLETED'
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                        : activeJob.status === 'FAILED'
                        ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                        : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                    }`}
                  >
                    {activeJob.status}
                  </span>
                )}
                {activeJob?.status === 'RUNNING' && (
                  <span className="text-[10px] font-mono text-zinc-400">
                    ⏱ {elapsedSeconds}s
                  </span>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {activeJob?.status === 'RUNNING' && (
                <button
                  onClick={handleStopExecution}
                  disabled={isStopping}
                  className="flex items-center gap-1 px-2.5 py-1 bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 border border-rose-800 text-xs rounded transition"
                >
                  <Square className="w-3 h-3 fill-current" />
                  <span>Stop</span>
                </button>
              )}

              {activeJob?.status === 'COMPLETED' && (
                <button
                  onClick={handleManualAudit}
                  className="flex items-center gap-1 px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 text-xs transition"
                >
                  <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Re-Audit</span>
                </button>
              )}

              <button
                onClick={handleSyncTree}
                disabled={syncingTree}
                className="flex items-center gap-1 px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 text-xs transition"
              >
                <FolderSync className={`w-3.5 h-3.5 text-zinc-400 ${syncingTree ? 'animate-spin' : ''}`} />
                <span>{syncSuccessMsg || 'Sync DAG'}</span>
              </button>
            </div>
          </div>

          {/* Audit Gate Receipt Summary Card (if completed) */}
          {auditReceipt && (
            <div className="p-3 bg-zinc-900/80 border-b border-zinc-800 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                {auditReceipt.status === 'PASS' ? (
                  <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
                )}
                <div>
                  <span className="font-semibold text-zinc-200">Audit Gate: </span>
                  <span
                    className={
                      auditReceipt.status === 'PASS'
                        ? 'text-emerald-400 font-bold'
                        : 'text-rose-400 font-bold'
                    }
                  >
                    {auditReceipt.status}
                  </span>
                  <span className="text-zinc-400 ml-2 font-mono text-[11px]">
                    Effect Size: {(auditReceipt.differing_fraction * 100).toFixed(1)}% (Floor &ge; 2.0%), Differing rows: {auditReceipt.differing_rows}/{auditReceipt.total_rows}
                  </span>
                </div>
              </div>
              <div className="text-[10px] font-mono text-zinc-500">
                Receipt SHA: {auditReceipt.receipt_hash?.slice(0, 16)}...
              </div>
            </div>
          )}

          {/* Terminal Console Output */}
          <div className="p-4 bg-zinc-950 text-zinc-300 font-mono text-[11px] leading-relaxed max-h-96 overflow-y-auto custom-scrollbar select-text">
            {jobLogs.length === 0 ? (
              <p className="text-zinc-600 italic">Waiting for process logs...</p>
            ) : (
              jobLogs.map((line, idx) => (
                <div key={idx} className="whitespace-pre-wrap hover:bg-zinc-900/40">
                  {line}
                </div>
              ))
            )}
            <div ref={terminalEndRef} />
          </div>
        </div>
      )}

      {/* Cloud Environment Settings Modal */}
      <CloudEnvironmentModal
        isOpen={isEnvModalOpen}
        onClose={() => setIsEnvModalOpen(false)}
        onStatusChanged={fetchEnvironment}
      />
    </div>
  );
};
