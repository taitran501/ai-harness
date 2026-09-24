import React, { useState, useEffect } from 'react';
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
  ExternalLink,
} from 'lucide-react';
import { TaskInfo, CloudGenerateResult, CloudPreflightResult } from '../types/experiment';
import { api } from '../services/api';

interface CloudRunnerHubProps {
  taskInfo?: TaskInfo | null;
  onRefreshSolutionTree: () => Promise<void>;
}

export const CloudRunnerHub: React.FC<CloudRunnerHubProps> = ({
  taskInfo,
  onRefreshSolutionTree,
}) => {
  const [platform, setPlatform] = useState<'modal' | 'kaggle'>('modal');
  const [configs, setConfigs] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>('configs/exp001_tfidf_baseline.yaml');

  // Modal Settings
  const [modalGpu, setModalGpu] = useState<string>('A10G');
  const [modalVolume, setModalVolume] = useState<string>('stereoqueer-dataset-volume');
  const [modalDataPath, setModalDataPath] = useState<string>('dataset/');

  // Kaggle Settings
  const [kaggleGpu, setKaggleGpu] = useState<string>('T4 x 2');
  const [kaggleDatasetSlug, setKaggleDatasetSlug] = useState<string>('antigravity/stereoqueer-eval-data');
  const [kaggleDataDir, setKaggleDataDir] = useState<string>('dataset/');

  // State
  const [generating, setGenerating] = useState(false);
  const [generatedResult, setGeneratedResult] = useState<CloudGenerateResult | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const [inspecting, setInspecting] = useState(false);
  const [preflightResult, setPreflightResult] = useState<CloudPreflightResult | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [syncingTree, setSyncingTree] = useState(false);
  const [syncSuccessMsg, setSyncSuccessMsg] = useState<string | null>(null);

  useEffect(() => {
    api
      .getConfigs()
      .then((res) => {
        if (res.configs && res.configs.length > 0) {
          setConfigs(res.configs);
          setSelectedConfig(res.configs[0]);
        }
      })
      .catch((err) => console.warn('Could not load configs:', err));
  }, []);

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2500);
  };

  const handleGenerate = async () => {
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

  const handlePreflight = async () => {
    setInspecting(true);
    try {
      const res = await api.runCloudPreflight(selectedConfig);
      setPreflightResult(res);
    } catch (err: any) {
      console.error(err);
      setPreflightResult({
        passed: false,
        errors: [err.message || 'Inspection failed.'],
      });
    } finally {
      setInspecting(false);
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
      {/* Top Header & Platform Selector */}
      <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <Cloud className="w-4 h-4 text-zinc-400" />
            <span>Cloud Execution Hub (Modal.com & Kaggle CLI)</span>
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Configure datasets, generate runner scripts, and dispatch headless GPU experiments to serverless cloud infrastructure.
          </p>
        </div>

        {/* Platform Switcher */}
        <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
          <button
            onClick={() => {
              setPlatform('modal');
              setGeneratedResult(null);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
              platform === 'modal'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Modal.com (A10G/A100)</span>
          </button>

          <button
            onClick={() => {
              setPlatform('kaggle');
              setGeneratedResult(null);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
              platform === 'kaggle'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Globe className="w-3.5 h-3.5 text-sky-400" />
            <span>Kaggle Kernels (Free 30h)</span>
          </button>
        </div>
      </div>

      {/* Grid: 3 Workflow Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Step 1: Dataset Management */}
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl flex flex-col justify-between space-y-3">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 flex items-center justify-center text-xs font-mono font-bold">
                  1
                </span>
                <span className="text-xs font-semibold text-zinc-200">
                  {platform === 'modal' ? 'Modal Persistent Volume' : 'Kaggle Dataset Management'}
                </span>
              </div>
              <HardDrive className="w-4 h-4 text-zinc-500" />
            </div>

            <p className="text-[11px] text-zinc-400">
              {platform === 'modal'
                ? 'Durable distributed storage mounted to your container functions to avoid redownloading data.'
                : 'Manage raw training data bundles via Kaggle Datasets CLI.'}
            </p>

            {platform === 'modal' ? (
              <div className="space-y-2">
                <div>
                  <label className="block text-[10px] text-zinc-400 mb-0.5">Volume Name</label>
                  <input
                    type="text"
                    value={modalVolume}
                    onChange={(e) => setModalVolume(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-zinc-400 mb-0.5">Local Source Directory</label>
                  <input
                    type="text"
                    value={modalDataPath}
                    onChange={(e) => setModalDataPath(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                  />
                </div>

                {/* Commands */}
                <div className="bg-zinc-900/80 border border-zinc-800 rounded p-2.5 space-y-1.5 font-mono text-[10px] text-zinc-300">
                  <div className="text-[10px] font-sans text-zinc-500 uppercase font-semibold">Terminal Commands:</div>
                  <div className="flex items-center justify-between bg-zinc-950 p-1.5 rounded border border-zinc-800">
                    <span className="truncate">modal volume create {modalVolume}</span>
                    <button
                      onClick={() => handleCopy(`modal volume create ${modalVolume}`, 'vol_create')}
                      className="text-zinc-500 hover:text-zinc-300 ml-1"
                    >
                      {copiedKey === 'vol_create' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                  </div>
                  <div className="flex items-center justify-between bg-zinc-950 p-1.5 rounded border border-zinc-800">
                    <span className="truncate">modal volume put {modalVolume} {modalDataPath} /data/</span>
                    <button
                      onClick={() => handleCopy(`modal volume put ${modalVolume} ${modalDataPath} /data/`, 'vol_put')}
                      className="text-zinc-500 hover:text-zinc-300 ml-1"
                    >
                      {copiedKey === 'vol_put' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div>
                  <label className="block text-[10px] text-zinc-400 mb-0.5">Dataset Slug</label>
                  <input
                    type="text"
                    value={kaggleDatasetSlug}
                    onChange={(e) => setKaggleDatasetSlug(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-zinc-400 mb-0.5">Local Directory</label>
                  <input
                    type="text"
                    value={kaggleDataDir}
                    onChange={(e) => setKaggleDataDir(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                  />
                </div>

                {/* Commands */}
                <div className="bg-zinc-900/80 border border-zinc-800 rounded p-2.5 space-y-1.5 font-mono text-[10px] text-zinc-300">
                  <div className="text-[10px] font-sans text-zinc-500 uppercase font-semibold">Terminal Commands:</div>
                  <div className="flex items-center justify-between bg-zinc-950 p-1.5 rounded border border-zinc-800">
                    <span className="truncate">kaggle datasets init -p {kaggleDataDir}</span>
                    <button
                      onClick={() => handleCopy(`kaggle datasets init -p ${kaggleDataDir}`, 'kg_init')}
                      className="text-zinc-500 hover:text-zinc-300 ml-1"
                    >
                      {copiedKey === 'kg_init' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                  </div>
                  <div className="flex items-center justify-between bg-zinc-950 p-1.5 rounded border border-zinc-800">
                    <span className="truncate">kaggle datasets create -p {kaggleDataDir} -r zip --public</span>
                    <button
                      onClick={() => handleCopy(`kaggle datasets create -p ${kaggleDataDir} -r zip --public`, 'kg_create')}
                      className="text-zinc-500 hover:text-zinc-300 ml-1"
                    >
                      {copiedKey === 'kg_create' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Step 2: Code & Kernel Runner Configuration */}
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl flex flex-col justify-between space-y-3">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 flex items-center justify-center text-xs font-mono font-bold">
                  2
                </span>
                <span className="text-xs font-semibold text-zinc-200">
                  {platform === 'modal' ? 'Modal App Generator' : 'Kaggle Kernel Staging'}
                </span>
              </div>
              <FileCode className="w-4 h-4 text-zinc-500" />
            </div>

            <div>
              <label className="block text-[10px] text-zinc-400 mb-0.5">Target Experiment Config</label>
              <select
                value={selectedConfig}
                onChange={(e) => setSelectedConfig(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
              >
                {configs.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[10px] text-zinc-400 mb-0.5">GPU Acceleration</label>
              {platform === 'modal' ? (
                <select
                  value={modalGpu}
                  onChange={(e) => setModalGpu(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                >
                  <option value="A10G">NVIDIA A10G (24GB - Fast & Cheap)</option>
                  <option value="A100-40GB">NVIDIA A100 (40GB High-Mem)</option>
                  <option value="A100-80GB">NVIDIA A100 (80GB Large Model)</option>
                  <option value="H100">NVIDIA H100 (Hopper Max Performance)</option>
                  <option value="L4">NVIDIA L4 (24GB Ada Lovelace)</option>
                  <option value="T4">NVIDIA T4 (Budget 16GB)</option>
                </select>
              ) : (
                <select
                  value={kaggleGpu}
                  onChange={(e) => setKaggleGpu(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                >
                  <option value="T4 x 2">2 x NVIDIA T4 (Free 30h/week)</option>
                  <option value="P100">1 x NVIDIA P100 (16GB)</option>
                </select>
              )}
            </div>

            <button
              onClick={handleGenerate}
              disabled={generating}
              className="w-full flex items-center justify-center gap-1.5 bg-zinc-100 hover:bg-white active:scale-95 text-zinc-950 py-1.5 rounded-lg text-xs font-semibold shadow-sm transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${generating ? 'animate-spin' : ''}`} />
              <span>{generating ? 'Generating Script...' : 'Generate Runner Package'}</span>
            </button>

            {generateError && (
              <p className="text-[11px] text-rose-400 bg-rose-950/40 p-2 rounded border border-rose-800/60">
                {generateError}
              </p>
            )}
          </div>
        </div>

        {/* Step 3: Dispatch, Monitor & Sync */}
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl flex flex-col justify-between space-y-3">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 flex items-center justify-center text-xs font-mono font-bold">
                  3
                </span>
                <span className="text-xs font-semibold text-zinc-200">
                  Pre-Flight & Dispatch
                </span>
              </div>
              <Terminal className="w-4 h-4 text-zinc-500" />
            </div>

            <p className="text-[11px] text-zinc-400">
              Run deterministic local pre-flight checks (schema, TSV files, Python syntax) before committing GPU resources.
            </p>

            <button
              onClick={handlePreflight}
              disabled={inspecting}
              className="w-full flex items-center justify-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-95 text-zinc-300 hover:text-white py-1.5 rounded-lg text-xs font-medium border border-zinc-800 transition"
            >
              <Play className={`w-3 h-3 text-zinc-400 ${inspecting ? 'animate-spin' : ''}`} />
              <span>{inspecting ? 'Inspecting...' : 'Run Pre-Flight Check'}</span>
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

            {/* Launch Command */}
            <div className="space-y-1">
              <label className="block text-[10px] text-zinc-400">Headless Dispatch Command:</label>
              <div className="flex items-center justify-between bg-zinc-900 border border-zinc-800 p-2 rounded text-[11px] font-mono text-zinc-200">
                <span className="truncate">
                  python harness/cloud_runner.py --backend {platform} --config {selectedConfig}
                </span>
                <button
                  onClick={() =>
                    handleCopy(
                      `python harness/cloud_runner.py --backend ${platform} --config ${selectedConfig}`,
                      'dispatch_cmd'
                    )
                  }
                  className="text-zinc-500 hover:text-zinc-300 ml-2"
                >
                  {copiedKey === 'dispatch_cmd' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                </button>
              </div>
            </div>

            {/* Artifact Sync Button */}
            <button
              onClick={handleSyncTree}
              disabled={syncingTree}
              className="w-full flex items-center justify-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-95 text-zinc-300 hover:text-white py-1.5 rounded-lg text-xs font-medium border border-zinc-800 transition"
            >
              <FolderSync className={`w-3.5 h-3.5 text-zinc-400 ${syncingTree ? 'animate-spin' : ''}`} />
              <span>{syncSuccessMsg || (syncingTree ? 'Syncing...' : 'Sync Artifacts to Solution Tree')}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Generated Code & Manifest Preview Drawer */}
      {generatedResult && (
        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <FileCode className="w-4 h-4 text-zinc-400" />
              <span className="text-xs font-semibold text-zinc-200">
                Generated Runner Script ({generatedResult.script_path})
              </span>
            </div>
            <button
              onClick={() => handleCopy(generatedResult.code, 'full_code')}
              className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded bg-zinc-900 border border-zinc-800"
            >
              {copiedKey === 'full_code' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedKey === 'full_code' ? 'Copied!' : 'Copy Code'}</span>
            </button>
          </div>

          <pre className="bg-zinc-900/70 border border-zinc-800 rounded-lg p-3 font-mono text-[11px] text-zinc-300 max-h-[300px] overflow-y-auto leading-relaxed custom-scrollbar select-all">
            {generatedResult.code}
          </pre>
        </div>
      )}
    </div>
  );
};
