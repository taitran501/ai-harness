import React, { useState, useEffect } from 'react';
import {
  X,
  Server,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  UserCheck,
  Key,
  Layers,
  ExternalLink,
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
  const [cloudStatus, setCloudStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Modal Profile action
  const [activatingProfile, setActivatingProfile] = useState<string | null>(null);
  const [modalSuccessMsg, setModalSuccessMsg] = useState<string | null>(null);

  // Modal Token input
  const [showModalTokenForm, setShowModalTokenForm] = useState(false);
  const [modalTokenId, setModalTokenId] = useState('');
  const [modalTokenSecret, setModalTokenSecret] = useState('');
  const [savingModalToken, setSavingModalToken] = useState(false);

  const [savingKaggle, setSavingKaggle] = useState(false);
  const [kaggleSuccessMsg, setKaggleSuccessMsg] = useState<string | null>(null);

  // LLM Key action
  const [showLLMKeyForm, setShowLLMKeyForm] = useState(false);
  const [selectedLLMKeyName, setSelectedLLMKeyName] = useState('GEMINI_API_KEY');
  const [llmKeyValue, setLLMKeyValue] = useState('');
  const [savingLLMKey, setSavingLLMKey] = useState(false);
  const [llmKeySuccessMsg, setLLMKeySuccessMsg] = useState<string | null>(null);

  // LLM Key verification state
  const [testingInputKey, setTestingInputKey] = useState(false);
  const [inputKeyTestResult, setInputKeyTestResult] = useState<{
    valid: boolean;
    latency_ms?: number;
    message?: string;
    error?: string;
  } | null>(null);
  const [verifyingProviders, setVerifyingProviders] = useState<Record<string, boolean>>({});
  const [providerVerificationResults, setProviderVerificationResults] = useState<Record<string, {
    valid: boolean;
    latency_ms?: number;
    message?: string;
    error?: string;
  }>>({});

  const fetchStatus = async (): Promise<EnvironmentStatus | null> => {
    setLoading(true);
    setError(null);
    const [environmentResult, cloudResult] = await Promise.allSettled([
      api.getEnvironmentStatus(),
      api.getCloudStatus(),
    ]);
    let environment: EnvironmentStatus | null = null;
    if (environmentResult.status === 'fulfilled') {
      environment = environmentResult.value;
      setEnvStatus(environment);
    } else {
      setError((environmentResult.reason as Error)?.message || 'Failed to detect environment.');
    }
    if (cloudResult.status === 'fulfilled') {
      setCloudStatus(cloudResult.value);
    } else {
      setCloudStatus(null);
    }
    setLoading(false);
    return environment;
  };

  useEffect(() => {
    if (isOpen) {
      fetchStatus();
    }
  }, [isOpen]);

  const kaggleProvider = cloudStatus?.providers?.kaggle;
  const kaggleQuota = kaggleProvider?.quota;
  const kaggleQuotaValue = kaggleQuota?.data ?? kaggleQuota?.raw;
  const kaggleQuotaText = kaggleQuotaValue !== undefined
    ? typeof kaggleQuotaValue === 'string' ? kaggleQuotaValue : JSON.stringify(kaggleQuotaValue)
    : kaggleQuota?.reason || kaggleQuota?.status || 'Not checked';
  const kaggleApiVerified = Boolean(kaggleProvider?.api_verified);
  const kaggleUploadReady = kaggleProvider?.dataset?.status === 'ready' && kaggleProvider?.dataset?.upload_consent;

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
    setError(null);
    setSavingKaggle(true);
    setKaggleSuccessMsg(null);
    try {
      const res = await api.launchKaggleAuth();
      if (!res.success) {
        throw new Error(res.message || 'Kaggle OAuth could not start.');
      }
      if (res.status === 'failed') {
        throw new Error('Kaggle CLI exited before the authorization flow could start. Check the installed CLI version.');
      }
      setKaggleSuccessMsg(res.message || 'Kaggle OAuth started. Approve access in the browser.');
      const finishLogin = async () => {
        // Stop presenting the browser approval step as active as soon as the
        // CLI exits. Account detection and provider quota checks can take
        // longer, so do not keep the OAuth button in its waiting state.
        setSavingKaggle(false);
        setEnvStatus((previous) => previous ? {
          ...previous,
          kaggle: {
            ...previous.kaggle,
            auth_flow: {
              ...previous.kaggle.auth_flow,
              status: 'completed',
              return_code: 0,
            },
          },
        } : previous);
        setKaggleSuccessMsg('Kaggle OAuth completed. Checking account status...');

        try {
          const refreshed = await api.getEnvironmentStatus();
          setEnvStatus(refreshed);
          if (refreshed.kaggle.authenticated) {
            setKaggleSuccessMsg(
              'Kaggle connected' + (refreshed.kaggle.username ? ' as ' + refreshed.kaggle.username : '') + '.'
            );
            onStatusChanged?.();
            // Quotas and API verification are useful follow-up information,
            // but must not delay confirmation that OAuth itself completed.
            void api.getCloudStatus().then(setCloudStatus).catch(() => undefined);
          } else {
            setError('Kaggle CLI finished login, but the active account could not be verified. Refresh status and check CLI configuration.');
          }
        } catch (err: any) {
          setError('Kaggle OAuth completed, but account status could not be refreshed: ' + (err.message || 'Unknown error.'));
        }
      };

      if (res.status === 'completed') {
        await finishLogin();
      } else if (res.status === 'running') {
        for (let attempt = 0; attempt < 120; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 1500));
          const flow = await api.getKaggleAuthFlowStatus();
          if (flow.status !== 'running') {
            if (flow.status === 'completed') {
              await finishLogin();
            } else if (flow.status === 'failed') {
              await fetchStatus();
              setError('Kaggle OAuth did not complete. Retry the login or check the Kaggle CLI installation.');
            } else {
              await fetchStatus();
              setError('Kaggle CLI finished login, but the active account could not be verified. Refresh status and check CLI configuration.');
            }
            return;
          }
        }
        await fetchStatus();
        setKaggleSuccessMsg('Kaggle authorization is still running. Approve access in the browser, then refresh status.');
      }
    } catch (err: any) {
      setError(err.message || "Failed to execute 'kaggle auth login --force'.");
    } finally {
      setSavingKaggle(false);
    }
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
        setInputKeyTestResult(null);
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

  const handleTestInputKey = async () => {
    if (!llmKeyValue.trim()) return;
    setTestingInputKey(true);
    setInputKeyTestResult(null);
    try {
      const res = await api.verifyLLMKey(selectedLLMKeyName, llmKeyValue.trim());
      setInputKeyTestResult(res);
    } catch (err: any) {
      setInputKeyTestResult({ valid: false, error: err.message || 'Verification failed.' });
    } finally {
      setTestingInputKey(false);
    }
  };

  const handleVerifyProvider = async (provider: string) => {
    setVerifyingProviders((prev) => ({ ...prev, [provider]: true }));
    try {
      const res = await api.verifyLLMKey(provider);
      setProviderVerificationResults((prev) => ({ ...prev, [provider]: res }));
    } catch (err: any) {
      setProviderVerificationResults((prev) => ({
        ...prev,
        [provider]: { valid: false, error: err.message || 'Verification failed.' },
      }));
    } finally {
      setVerifyingProviders((prev) => ({ ...prev, [provider]: false }));
    }
  };

  const handleVerifyAllConfigured = async () => {
    if (!envStatus?.llm?.providers) return;
    const providersToTest = Object.entries(envStatus.llm.providers)
      .filter(([_, info]) => info.configured || info.reachable)
      .map(([name]) => name);

    for (const p of providersToTest) {
      handleVerifyProvider(p);
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
                <span>Cloud & LLM Environment</span>
              </h2>
              <p className="text-xs text-zinc-400">
                Live status and credentials for Modal.com, Kaggle, and LLM API providers.
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
                  <span>Authorize</span>
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
                    No Modal profiles configured. Click "Authorize" above to generate a token or run `modal setup` in terminal.
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
                <span className="font-semibold text-zinc-100">Kaggle account</span>
                <span className={`text-[10px] font-mono border px-1.5 py-0.5 rounded ${
                  kaggleApiVerified
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    : envStatus?.kaggle.authenticated
                      ? 'bg-amber-500/10 text-amber-300 border-amber-500/20'
                      : 'bg-zinc-800 text-zinc-400 border-zinc-700'
                }`}>
                  {kaggleApiVerified
                    ? 'Connected'
                    : envStatus?.kaggle.authenticated
                      ? 'Credentials found · API not verified'
                      : 'Not connected'}
                </span>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleLaunchKaggleAuth}
                  disabled={savingKaggle || envStatus?.kaggle.auth_flow?.status === 'running'}
                  className="flex items-center gap-1.5 px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold transition"
                  title="Run kaggle auth login --force and approve access in the browser"
                >
                  {savingKaggle || envStatus?.kaggle.auth_flow?.status === 'running'
                    ? <RefreshCw className="w-3 h-3 animate-spin" />
                    : <ExternalLink className="w-3 h-3" />}
                  <span>{savingKaggle || envStatus?.kaggle.auth_flow?.status === 'running'
                    ? 'Waiting for approval...'
                    : envStatus?.kaggle.authenticated ? 'Switch / reconnect' : 'Connect Kaggle'}</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-5 gap-y-2 text-[11px]">
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">Account</span>
                {envStatus?.kaggle.username ? (
                  <a
                    href={`https://www.kaggle.com/${encodeURIComponent(envStatus.kaggle.username)}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-cyan-300 hover:text-cyan-200 inline-flex items-center gap-1"
                  >
                    {envStatus.kaggle.username}<ExternalLink className="w-3 h-3" />
                  </a>
                ) : (
                  <span className="text-zinc-400">{envStatus?.kaggle.authenticated ? 'Name unavailable' : 'Not signed in'}</span>
                )}
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">CLI</span>
                <span className="text-zinc-300 text-right">{envStatus?.kaggle.cli_version || (envStatus?.kaggle.installed ? 'Installed · version unavailable' : 'Not installed')}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">OAuth login support</span>
                <span className={envStatus?.kaggle.oauth_supported ? 'text-emerald-400' : 'text-amber-300'}>
                  {envStatus?.kaggle.oauth_supported ? 'Available' : 'Upgrade Kaggle CLI'}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 sm:col-span-2">
                <span className="text-zinc-500">CLI path</span>
                <span className="text-zinc-400 font-mono text-right break-all">{envStatus?.kaggle.executable || 'Not found on PATH'}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">Authentication</span>
                <span className="text-zinc-300">{envStatus?.kaggle.authentication_method || 'None detected'}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">Task run readiness</span>
                <span className={kaggleProvider?.ready ? 'text-emerald-400' : 'text-amber-300'}>
                  {kaggleProvider?.ready ? 'Ready' : 'Needs setup'}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">OAuth flow</span>
                <span className={envStatus?.kaggle.auth_flow?.status === 'running' ? 'text-amber-300' : 'text-zinc-300'}>
                  {envStatus?.kaggle.auth_flow?.status === 'running'
                    ? 'Waiting for browser approval'
                    : envStatus?.kaggle.auth_flow?.status === 'failed'
                      ? `Failed (exit ${envStatus.kaggle.auth_flow.return_code ?? 'unknown'})`
                      : envStatus?.kaggle.auth_flow?.status === 'completed' ? 'Completed' : 'Idle'}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-zinc-500">Account API check</span>
                <span className={kaggleApiVerified ? 'text-emerald-400' : 'text-zinc-400'}>
                  {kaggleApiVerified ? 'Verified' : kaggleQuota?.reason || 'Unavailable'}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 sm:col-span-2">
                <span className="text-zinc-500">Accelerator quota</span>
                <span className="text-zinc-300 text-right break-all">{kaggleQuotaText}</span>
              </div>
              <div className="flex items-center justify-between gap-3 sm:col-span-2">
                <span className="text-zinc-500">Current task upload</span>
                <span className={kaggleUploadReady ? 'text-emerald-400' : 'text-amber-300'}>
                  {kaggleUploadReady
                    ? `Approved · ${kaggleProvider.dataset.file_count} files · ${String(kaggleProvider.dataset.fingerprint).slice(0, 12)}`
                    : kaggleProvider?.dataset?.status === 'not_ready'
                      ? kaggleProvider.dataset.reason || 'Dataset paths are not ready'
                      : 'Consent required for this dataset'}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 sm:col-span-2">
                <span className="text-zinc-500">Config directory</span>
                <span className="text-zinc-400 font-mono text-right break-all">{envStatus?.kaggle.config_dir || '~/.kaggle'}</span>
              </div>
              {envStatus?.kaggle.error && (
                <div className="sm:col-span-2 text-rose-300">{envStatus.kaggle.error}</div>
              )}
              {Array.isArray(kaggleProvider?.warnings) && kaggleProvider.warnings.length > 0 && (
                <div className="sm:col-span-2 text-zinc-500 space-y-1">
                  {kaggleProvider.warnings.map((warning: string, index: number) => (
                    <div key={`${index}-${warning}`}>{warning}</div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 3. LLM API Providers */}
          <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-zinc-200">LLM API Providers</span>
              </div>
              <div className="flex items-center gap-2">
                {envStatus?.llm?.providers && (
                  <button
                    type="button"
                    onClick={handleVerifyAllConfigured}
                    className="px-2 py-1 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 rounded text-xs font-mono transition flex items-center gap-1.5"
                    title="Test all configured provider keys"
                  >
                    <RefreshCw className="w-3 h-3 text-cyan-400" />
                    <span>Test All</span>
                  </button>
                )}
                <button
                  onClick={() => {
                    setShowLLMKeyForm(!showLLMKeyForm);
                    setInputKeyTestResult(null);
                  }}
                  className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-purple-300 rounded text-xs font-medium transition flex items-center gap-1.5"
                >
                  <Key className="w-3 h-3 text-purple-400" />
                  <span>{showLLMKeyForm ? 'Cancel' : 'Set API Key'}</span>
                </button>
              </div>
            </div>

            {showLLMKeyForm && (
              <form onSubmit={handleSaveLLMKey} className="p-3 bg-zinc-950/80 border border-zinc-800 rounded-lg space-y-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-zinc-400 font-mono">Provider Key</label>
                    <select
                      value={selectedLLMKeyName}
                      onChange={(e) => {
                        setSelectedLLMKeyName(e.target.value);
                        setInputKeyTestResult(null);
                      }}
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
                      onChange={(e) => {
                        setLLMKeyValue(e.target.value);
                        setInputKeyTestResult(null);
                      }}
                      placeholder="Paste API secret..."
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200 font-mono"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                  <div className="flex items-center gap-2">
                    {inputKeyTestResult && (
                      <span
                        className={`text-[11px] font-mono flex items-center gap-1.5 px-2 py-0.5 rounded border ${
                          inputKeyTestResult.valid
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                            : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                        }`}
                      >
                        {inputKeyTestResult.valid ? (
                          <>
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                            <span>Valid ({inputKeyTestResult.latency_ms}ms)</span>
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                            <span className="truncate max-w-[200px]" title={inputKeyTestResult.error}>
                              {inputKeyTestResult.error || 'Invalid key'}
                            </span>
                          </>
                        )}
                      </span>
                    )}
                    {llmKeySuccessMsg && (
                      <span className="text-xs text-emerald-400 font-medium">{llmKeySuccessMsg}</span>
                    )}
                    {!inputKeyTestResult && !llmKeySuccessMsg && (
                      <span className="text-[10px] text-zinc-500">Key will be saved to .env</span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={testingInputKey || !llmKeyValue.trim()}
                      onClick={handleTestInputKey}
                      className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 rounded text-xs font-medium transition flex items-center gap-1.5 disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3 h-3 ${testingInputKey ? 'animate-spin text-purple-400' : 'text-zinc-400'}`} />
                      <span>{testingInputKey ? 'Testing...' : 'Test Key'}</span>
                    </button>
                    <button
                      type="submit"
                      disabled={savingLLMKey || !llmKeyValue.trim()}
                      className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded text-xs font-semibold disabled:opacity-50"
                    >
                      {savingLLMKey ? 'Saving...' : 'Save API Key'}
                    </button>
                  </div>
                </div>
              </form>
            )}

            {/* Providers Status Grid (Cloud API Providers only, no ollama/vllm) */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] font-mono">
              {envStatus?.llm?.providers ? (
                Object.entries(envStatus.llm.providers)
                  .filter(([pName, pInfo]) => !pInfo.is_local && pName !== 'ollama' && pName !== 'vllm')
                  .map(([pName, pInfo]) => {
                    const isVerifying = verifyingProviders[pName];
                    const testResult = providerVerificationResults[pName];
                    const isConfigured = pInfo.configured;

                    return (
                      <div
                        key={pName}
                        className={`p-2.5 rounded-lg border space-y-1.5 transition ${
                          testResult?.valid
                            ? 'bg-emerald-950/20 border-emerald-700/50'
                            : testResult && !testResult.valid
                            ? 'bg-rose-950/20 border-rose-700/50'
                            : 'bg-zinc-950/60 border-zinc-800/60'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-zinc-200 font-medium capitalize">{pName}</span>
                          {isVerifying ? (
                            <RefreshCw className="w-3 h-3 animate-spin text-purple-400" />
                          ) : testResult ? (
                            testResult.valid ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                            ) : (
                              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                            )
                          ) : isConfigured ? (
                            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                          ) : (
                            <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
                          )}
                        </div>

                        <div className="flex items-center justify-between gap-1 text-[10px]">
                          <span className="truncate text-zinc-400">
                            {testResult ? (
                              testResult.valid ? (
                                <span className="text-emerald-400">Valid ({testResult.latency_ms}ms)</span>
                              ) : (
                                <span className="text-rose-400 truncate" title={testResult.error}>
                                  {testResult.error?.includes('401') ? 'Invalid (401)' : testResult.error?.includes('400') ? 'Invalid (400)' : 'Failed'}
                                </span>
                              )
                            ) : isConfigured ? (
                              <span className="text-zinc-300">{pInfo.masked_key || 'Configured'}</span>
                            ) : (
                              <span className="text-zinc-500">No Key</span>
                            )}
                          </span>

                          {isConfigured && (
                            <button
                              type="button"
                              disabled={isVerifying}
                              onClick={() => handleVerifyProvider(pName)}
                              className="px-1.5 py-0.5 bg-zinc-900 hover:bg-zinc-800 text-[10px] text-zinc-300 hover:text-white rounded border border-zinc-800 transition shrink-0"
                              title="Test this provider API key live"
                            >
                              {isVerifying ? '...' : 'Check'}
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })
              ) : (
                <div className="col-span-3 text-xs text-zinc-500 italic">No LLM provider status loaded.</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-zinc-800 bg-zinc-900/50">
          <span className="text-[11px] text-zinc-500 font-mono">
            Environment verified across Modal, Kaggle, and LLM API providers.
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
