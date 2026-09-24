import React, { useState, useEffect } from 'react';
import {
  X,
  Server,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Cpu,
  HardDrive,
  UserCheck,
  Key,
  Shield,
  Layers,
  ChevronRight,
  Sparkles,
  ExternalLink,
  Upload,
} from 'lucide-react';
import { EnvironmentStatus } from '../types/experiment';
import { api } from '../services/api';

interface CloudEnvironmentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStatusChanged?: () => void;
}

export const CloudEnvironmentModal: React.FC<CloudEnvironmentModalProps> = ({
  isOpen,
  onClose,
  onStatusChanged,
}) => {
  const [loading, setLoading] = useState(false);
  const [envStatus, setEnvStatus] = useState<EnvironmentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Modal Profile action
  const [activatingProfile, setActivatingProfile] = useState<string | null>(null);
  const [modalSuccessMsg, setModalSuccessMsg] = useState<string | null>(null);

  // Modal Token input
  const [showModalTokenForm, setShowModalTokenForm] = useState(false);
  const [modalTokenId, setModalTokenId] = useState('');
  const [modalTokenSecret, setModalTokenSecret] = useState('');
  const [savingModalToken, setSavingModalToken] = useState(false);

  // Kaggle Profile action
  const [showKaggleForm, setShowKaggleForm] = useState(false);
  const [kaggleUser, setKaggleUser] = useState('');
  const [kaggleKey, setKaggleKey] = useState('');
  const [savingKaggle, setSavingKaggle] = useState(false);
  const [kaggleSuccessMsg, setKaggleSuccessMsg] = useState<string | null>(null);

  // LLM Key action
  const [showLLMKeyForm, setShowLLMKeyForm] = useState(false);
  const [selectedLLMKeyName, setSelectedLLMKeyName] = useState('GEMINI_API_KEY');
  const [llmKeyValue, setLLMKeyValue] = useState('');
  const [savingLLMKey, setSavingLLMKey] = useState(false);
  const [llmKeySuccessMsg, setLLMKeySuccessMsg] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getEnvironmentStatus();
      setEnvStatus(data);
    } catch (err: any) {
      setError(err.message || 'Failed to detect environment.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchStatus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleActivateModalProfile = async (profile: string) => {
    setActivatingProfile(profile);
    setModalSuccessMsg(null);
    try {
      const res = await api.activateModalProfile(profile);
      if (res.success) {
        setModalSuccessMsg(`Active profile set to '${profile}'`);
        await fetchStatus();
        onStatusChanged?.();
        setTimeout(() => setModalSuccessMsg(null), 3000);
      } else {
        setError(res.error || 'Failed to switch profile.');
      }
    } catch (err: any) {
      setError(err.message || 'Error switching profile.');
    } finally {
      setActivatingProfile(null);
    }
  };

  const handleLaunchModalAuth = async () => {
    window.open('https://modal.com/settings/tokens', '_blank');
    try {
      await api.launchModalAuth();
    } catch (_) {}
  };

  const handleLaunchKaggleAuth = async () => {
    window.open('https://www.kaggle.com/settings', '_blank');
    try {
      await api.launchKaggleAuth();
    } catch (_) {}
  };

  const handleSaveModalToken = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    let id = modalTokenId.trim();
    let secret = modalTokenSecret.trim();

    // Smart regex parsing if user pasted entire command or combined text into ID
    const akMatch = id.match(/ak-[a-zA-Z0-9_-]+/);
    const asMatch = id.match(/as-[a-zA-Z0-9_-]+/) || secret.match(/as-[a-zA-Z0-9_-]+/);
    if (akMatch && asMatch) {
      id = akMatch[0];
      secret = asMatch[0];
    }

    if (!id || !secret) {
      setError('Please provide both token ID (ak-...) and token secret (as-...).');
      return;
    }
    setSavingModalToken(true);
    try {
      const res = await api.setModalToken(id, secret);
      if (res.success) {
        setModalSuccessMsg('Modal token updated and authenticated successfully.');
        setShowModalTokenForm(false);
        setModalTokenId('');
        setModalTokenSecret('');
        await fetchStatus();
        onStatusChanged?.();
        setTimeout(() => setModalSuccessMsg(null), 3000);
      } else {
        setError(res.error || 'Failed to save Modal token.');
      }
    } catch (err: any) {
      setError(err.message || 'Error saving token.');
    } finally {
      setSavingModalToken(false);
    }
  };

  const handleKaggleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const content = event.target?.result as string;
        const parsed = JSON.parse(content);
        const username = parsed.username || parsed.user;
        const key = parsed.key;
        if (!username || !key) {
          setError('Invalid kaggle.json: "username" and "key" properties are required.');
          return;
        }
        setSavingKaggle(true);
        const res = await api.switchKaggleProfile(username, key);
        if (res.success) {
          setKaggleSuccessMsg(`Kaggle authenticated for '${res.username}' via 1-click kaggle.json!`);
          await fetchStatus();
          onStatusChanged?.();
          setTimeout(() => setKaggleSuccessMsg(null), 3500);
        } else {
          setError('Failed to apply Kaggle credentials from file.');
        }
      } catch (err: any) {
        setError(`Failed to read kaggle.json: ${err.message}`);
      } finally {
        setSavingKaggle(false);
      }
    };
    reader.readAsText(file);
  };

  const handleSaveKaggleCreds = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!kaggleUser || !kaggleKey) return;
    setSavingKaggle(true);
    setError(null);
    try {
      const res = await api.switchKaggleProfile(kaggleUser, kaggleKey);
      if (res.success) {
        setKaggleSuccessMsg(`Kaggle credentials set for '${res.username}'.`);
        setShowKaggleForm(false);
        setKaggleUser('');
        setKaggleKey('');
        await fetchStatus();
        onStatusChanged?.();
        setTimeout(() => setKaggleSuccessMsg(null), 3000);
      } else {
        setError('Failed to save Kaggle credentials.');
      }
    } catch (err: any) {
      setError(err.message || 'Error saving Kaggle credentials.');
    } finally {
      setSavingKaggle(false);
    }
  };

  const handleSaveLLMKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedLLMKeyName || !llmKeyValue.trim()) return;
    setSavingLLMKey(true);
    setLLMKeySuccessMsg(null);
    try {
      const res = await api.saveLLMKey(selectedLLMKeyName, llmKeyValue.trim());
      if (res.success) {
        setLLMKeySuccessMsg(`Key '${selectedLLMKeyName}' saved!`);
        setLLMKeyValue('');
        await fetchStatus();
        onStatusChanged?.();
        setTimeout(() => {
          setShowLLMKeyForm(false);
          setLLMKeySuccessMsg(null);
        }, 2000);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to save LLM key.');
    } finally {
      setSavingLLMKey(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className="bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 bg-zinc-900/50">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700/80 flex items-center justify-center text-zinc-300">
              <Server className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <span>Hardware & Cloud Environment</span>
                <span className="text-[10px] font-mono uppercase bg-zinc-800 text-zinc-400 px-1.5 py-0.5 rounded border border-zinc-700">
                  Multi-Account Manager
                </span>
              </h2>
              <p className="text-xs text-zinc-400">
                Live status detection and multi-profile switcher for Modal.com, Kaggle CLI, Docker, and local GPU.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchStatus}
              disabled={loading}
              className="p-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border border-zinc-800 transition"
              title="Refresh environment status"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Global Feedback */}
        {error && (
          <div className="mx-5 mt-3 p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}
        {(modalSuccessMsg || kaggleSuccessMsg) && (
          <div className="mx-5 mt-3 p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{modalSuccessMsg || kaggleSuccessMsg}</span>
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 custom-scrollbar text-xs">
          {/* 1. Modal.com Environment */}
          <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 pb-2.5">
              <div className="flex items-center gap-2">
                <Cloud className="w-4 h-4 text-sky-400" />
                <span className="font-semibold text-zinc-100">Modal.com Serverless GPU</span>
                {envStatus?.modal.installed ? (
                  <span className="text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1.5 py-0.5 rounded">
                    CLI Available
                  </span>
                ) : (
                  <span className="text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 rounded">
                    Not Installed
                  </span>
                )}
                {envStatus?.modal.active_profile && (
                  <span className="text-[10px] font-mono text-zinc-400">
                    Active: <strong className="text-sky-300 font-semibold">{envStatus.modal.active_profile}</strong>
                  </span>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleLaunchModalAuth}
                  className="flex items-center gap-1.5 px-2.5 py-1 bg-sky-950/40 hover:bg-sky-900/60 text-sky-300 border border-sky-800/40 rounded-lg text-xs font-medium transition"
                  title="Open Modal token creation page in browser"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>Authorize on Modal.com</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowModalTokenForm(!showModalTokenForm)}
                  className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1 font-mono px-2 py-1 bg-zinc-900 hover:bg-zinc-800 rounded-lg border border-zinc-800 transition"
                >
                  <Key className="w-3 h-3" />
                  <span>{showModalTokenForm ? 'Close' : 'Paste Token'}</span>
                </button>
              </div>
            </div>

            {/* Modal Token Form */}
            {showModalTokenForm && (
              <form onSubmit={handleSaveModalToken} className="p-3 bg-zinc-900 border border-zinc-800 rounded-lg space-y-2">
                <div className="text-[11px] text-zinc-400">
                  Paste Token ID + Secret, or the full command from Modal (e.g.{' '}
                  <code className="text-sky-300 font-mono text-[10px]">modal token set --token-id ak-... --token-secret as-...</code>):
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">Token ID / Command</label>
                    <input
                      type="text"
                      value={modalTokenId}
                      onChange={(e) => setModalTokenId(e.target.value)}
                      placeholder="ak-... or modal token set ..."
                      className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">Token Secret (optional if pasted in left box)</label>
                    <input
                      type="password"
                      value={modalTokenSecret}
                      onChange={(e) => setModalTokenSecret(e.target.value)}
                      placeholder="as-..."
                      className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={savingModalToken || !modalTokenId.trim()}
                  className="px-3 py-1 bg-sky-600 hover:bg-sky-500 text-white rounded text-xs font-semibold"
                >
                  {savingModalToken ? 'Saving Token...' : 'Save & Authenticate'}
                </button>
              </form>
            )}

            {/* Profile List & Switcher */}
            <div>
              <div className="text-[11px] text-zinc-400 mb-2">
                <span>Configured Accounts:</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {envStatus?.modal.profiles && envStatus.modal.profiles.length > 0 ? (
                  envStatus.modal.profiles.map((profile) => {
                    const isActive = profile === envStatus.modal.active_profile;
                    return (
                      <button
                        key={profile}
                        onClick={() => handleActivateModalProfile(profile)}
                        disabled={isActive || activatingProfile === profile}
                        className={`flex items-center justify-between p-2.5 rounded-lg border text-left transition ${
                          isActive
                            ? 'bg-sky-950/30 border-sky-600/50 text-sky-200'
                            : 'bg-zinc-900/60 border-zinc-800 hover:border-zinc-700 text-zinc-300 hover:text-white'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <UserCheck className={`w-3.5 h-3.5 ${isActive ? 'text-sky-400' : 'text-zinc-500'}`} />
                          <span className="font-mono text-xs font-medium">{profile}</span>
                        </div>
                        {isActive ? (
                          <span className="text-[10px] font-mono font-bold uppercase text-sky-400 bg-sky-900/40 px-1.5 py-0.5 rounded">
                            Active
                          </span>
                        ) : activatingProfile === profile ? (
                          <RefreshCw className="w-3 h-3 animate-spin text-zinc-400" />
                        ) : (
                          <span className="text-[10px] font-mono text-zinc-500 hover:text-zinc-300">
                            Switch →
                          </span>
                        )}
                      </button>
                    );
                  })
                ) : (
                  <p className="text-[11px] text-zinc-500 italic col-span-2">
                    No Modal profiles configured. Click "Authorize on Modal.com" above to generate a token or run `modal setup` in terminal.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* 2. Kaggle CLI Environment */}
          <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 pb-2.5">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span className="font-semibold text-zinc-100">Kaggle CLI & Kernels</span>
                {envStatus?.kaggle.authenticated ? (
                  <span className="text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1.5 py-0.5 rounded">
                    Authenticated: {envStatus.kaggle.username}
                  </span>
                ) : (
                  <span className="text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 rounded">
                    Not Authenticated
                  </span>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                {/* 1-Click Upload kaggle.json */}
                <label className="flex items-center gap-1.5 px-2.5 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold cursor-pointer transition shadow-sm">
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload kaggle.json</span>
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleKaggleFileUpload}
                    className="hidden"
                  />
                </label>

                {/* Direct Link to Kaggle Settings */}
                <button
                  type="button"
                  onClick={handleLaunchKaggleAuth}
                  className="flex items-center gap-1.5 px-2.5 py-1 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white rounded-lg border border-zinc-800 text-xs font-medium transition"
                  title="Open Kaggle Account Settings to create/download token"
                >
                  <ExternalLink className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Get Token on Kaggle</span>
                </button>

                <button
                  type="button"
                  onClick={() => setShowKaggleForm(!showKaggleForm)}
                  className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1 font-mono px-2 py-1 bg-zinc-900 hover:bg-zinc-800 rounded-lg border border-zinc-800 transition"
                >
                  <Key className="w-3 h-3" />
                  <span>{showKaggleForm ? 'Close' : 'Manual Key'}</span>
                </button>
              </div>
            </div>

            {/* Kaggle Key Form */}
            {showKaggleForm && (
              <form onSubmit={handleSaveKaggleCreds} className="p-3 bg-zinc-900 border border-zinc-800 rounded-lg space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">Username</label>
                    <input
                      type="text"
                      value={kaggleUser}
                      onChange={(e) => setKaggleUser(e.target.value)}
                      placeholder="e.g. locminh504"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">API Key</label>
                    <input
                      type="password"
                      value={kaggleKey}
                      onChange={(e) => setKaggleKey(e.target.value)}
                      placeholder="32-character key"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={savingKaggle || !kaggleUser || !kaggleKey}
                  className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-xs font-semibold"
                >
                  {savingKaggle ? 'Saving...' : 'Save Kaggle Credentials'}
                </button>
              </form>
            )}

            <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono">
              <span>Config Directory:</span>
              <span className="text-zinc-300">{envStatus?.kaggle.config_dir || '~/.kaggle'}</span>
            </div>
          </div>

          {/* 3. Local Hardware & Docker */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Local GPU */}
            <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span className="font-semibold text-zinc-200">Local NVIDIA GPU</span>
              </div>
              {envStatus?.gpu.detected ? (
                <div className="space-y-1 text-[11px] font-mono">
                  <div className="text-zinc-200 font-medium">{envStatus.gpu.gpu_name}</div>
                  <div className="flex justify-between text-zinc-400">
                    <span>VRAM Total:</span>
                    <span className="text-zinc-200">{envStatus.gpu.memory_total_mb} MiB</span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>VRAM Free:</span>
                    <span className="text-emerald-400">{envStatus.gpu.memory_free_mb} MiB</span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>Driver Version:</span>
                    <span className="text-zinc-300">{envStatus.gpu.driver_version}</span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>PyTorch CUDA:</span>
                    <span className="text-emerald-400">{envStatus.gpu.cuda_available ? 'Available' : 'No'}</span>
                  </div>
                </div>
              ) : (
                <p className="text-[11px] text-zinc-500 italic">No NVIDIA GPU detected on local host.</p>
              )}
            </div>

            {/* Docker Environment */}
            <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
                <HardDrive className="w-4 h-4 text-blue-400" />
                <span className="font-semibold text-zinc-200">Docker Containerization</span>
              </div>
              <div className="space-y-1 text-[11px] font-mono">
                <div className="flex justify-between text-zinc-400">
                  <span>Installed:</span>
                  <span className={envStatus?.docker.installed ? 'text-emerald-400' : 'text-zinc-500'}>
                    {envStatus?.docker.installed ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between text-zinc-400">
                  <span>Daemon Status:</span>
                  <span className={envStatus?.docker.running ? 'text-emerald-400' : 'text-amber-400'}>
                    {envStatus?.docker.running ? 'Running' : 'Stopped / Standby'}
                  </span>
                </div>
                {envStatus?.docker.version && (
                  <div className="text-[10px] text-zinc-500 truncate pt-1">
                    {envStatus.docker.version}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 4. LLM API Providers & Local Daemons */}
          <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-400" />
                <span className="font-semibold text-zinc-200">LLM API Providers & Local Daemons</span>
              </div>
              <button
                onClick={() => setShowLLMKeyForm(!showLLMKeyForm)}
                className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-purple-300 rounded text-xs font-medium transition flex items-center gap-1.5"
              >
                <Key className="w-3 h-3 text-purple-400" />
                <span>{showLLMKeyForm ? 'Cancel' : 'Set API Key'}</span>
              </button>
            </div>

            {showLLMKeyForm && (
              <form onSubmit={handleSaveLLMKey} className="p-3 bg-zinc-950/80 border border-zinc-800 rounded-lg space-y-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">Provider Key</label>
                    <select
                      value={selectedLLMKeyName}
                      onChange={(e) => setSelectedLLMKeyName(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono"
                    >
                      <option value="GEMINI_API_KEY">GEMINI_API_KEY (Google Gemini)</option>
                      <option value="OPENAI_API_KEY">OPENAI_API_KEY (OpenAI)</option>
                      <option value="ANTHROPIC_API_KEY">ANTHROPIC_API_KEY (Anthropic Claude)</option>
                      <option value="DEEPSEEK_API_KEY">DEEPSEEK_API_KEY (DeepSeek)</option>
                      <option value="GROQ_API_KEY">GROQ_API_KEY (Groq LPU)</option>
                      <option value="OPENROUTER_API_KEY">OPENROUTER_API_KEY (OpenRouter)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">Key Secret</label>
                    <input
                      type="password"
                      value={llmKeyValue}
                      onChange={(e) => setLLMKeyValue(e.target.value)}
                      placeholder="Paste API secret..."
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between pt-1">
                  {llmKeySuccessMsg ? (
                    <span className="text-xs text-emerald-400 font-medium">{llmKeySuccessMsg}</span>
                  ) : (
                    <span className="text-[10px] text-zinc-500">Key will be saved to .env</span>
                  )}
                  <button
                    type="submit"
                    disabled={savingLLMKey || !llmKeyValue.trim()}
                    className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded text-xs font-semibold"
                  >
                    {savingLLMKey ? 'Saving...' : 'Save API Key'}
                  </button>
                </div>
              </form>
            )}

            {/* Providers Status Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
              {envStatus?.llm?.providers ? (
                Object.entries(envStatus.llm.providers).map(([pName, pInfo]) => (
                  <div key={pName} className="p-2 bg-zinc-950/60 border border-zinc-800/60 rounded-lg space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-300 font-medium capitalize">{pName}</span>
                      {pInfo.configured || pInfo.reachable ? (
                        <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
                      )}
                    </div>
                    <div className="text-[10px] text-zinc-500 truncate">
                      {pInfo.is_local
                        ? pInfo.reachable
                          ? 'Daemon Live'
                          : 'Not Running'
                        : pInfo.configured
                        ? pInfo.masked_key || 'Configured'
                        : 'No Key'}
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-4 text-xs text-zinc-500 italic">No LLM provider status loaded.</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-zinc-800 bg-zinc-900/50">
          <span className="text-[11px] text-zinc-500 font-mono">
            Environment verified across Modal, Kaggle, Docker & Local GPU.
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-medium transition"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
