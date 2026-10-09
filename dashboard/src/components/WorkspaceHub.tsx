import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import { useLanguage } from '../constants/i18n';

type ProviderStatus = {
  provider: 'kaggle' | 'modal';
  cli_installed?: boolean;
  authenticated?: boolean;
  account?: string | null;
  cli_version?: string | null;
  ready?: boolean;
  dataset?: { fingerprint?: string; file_count?: number; upload_consent?: boolean; visibility?: string };
  quota?: { status?: string; reason?: string; observed_at?: string; data?: unknown; raw?: string; note?: string };
  billing?: { status?: string; reason?: string; period?: string; observed_at?: string; data?: unknown; raw?: string; note?: string };
  warnings?: string[];
};

const prettyStatus = (value?: string) => (value || 'UNKNOWN').replace(/_/g, ' ');

export const WorkspaceHub: React.FC<{ onOpenEnvironmentModal: () => void }> = ({ onOpenEnvironmentModal }) => {
  const { t } = useLanguage();
  const [phaseState, setPhaseState] = useState<any>(null);
  const [selectedPhase, setSelectedPhase] = useState<string>('');
  const [phasePrompt, setPhasePrompt] = useState('');
  const [phaseResponse, setPhaseResponse] = useState('');
  const [sourceJson, setSourceJson] = useState('[]');
  const [artifactText, setArtifactText] = useState('');
  const [eda, setEda] = useState<any>(null);
  const [cloudState, setCloudState] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [uploadChecks, setUploadChecks] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  const [showErrorDetails, setShowErrorDetails] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [notice, setNotice] = useState('');
  const dismissedErrorRef = useRef<string | null>(null);
  dismissedErrorRef.current = dismissedError;

  const refresh = useCallback(async (isManual = false) => {
    if (isManual) {
      setError('');
      setDismissedError(null);
    }
    const results = await Promise.allSettled([
      api.getPhases(),
      api.getLatestEda(),
      api.getCloudJobs(),
    ]);
    const [phaseResult, edaResult, jobsResult] = results;
    if (phaseResult.status === 'fulfilled') {
      const state = phaseResult.value;
      setPhaseState(state);
      setSelectedPhase((current) => {
        if (current && state.phases?.some((phase: any) => phase.id === current)) return current;
        return state.next_phase || state.phases?.find((phase: any) => phase.can_revisit)?.id || '';
      });
      if (isManual) setError('');
    } else {
      const msg = phaseResult.reason?.message || 'Could not read the prompt pack. Run `ml-harness phase init` in this workspace.';
      if (isManual || dismissedErrorRef.current !== msg) {
        setError(msg);
      }
    }
    if (edaResult.status === 'fulfilled') setEda(edaResult.value?.report ?? edaResult.value);
    if (jobsResult.status === 'fulfilled') setJobs(jobsResult.value.jobs || []);
  }, []);

  const refreshProviders = useCallback(async (isManual = false) => {
    try {
      setCloudState(await api.getCloudStatus());
      if (isManual && !error) setError('');
    } catch (reason: any) {
      const msg = reason?.message || 'Could not inspect Kaggle and Modal account status.';
      if (isManual || dismissedErrorRef.current !== msg) {
        setError(msg);
      }
    }
  }, [error]);

  const handleManualRetry = async () => {
    setIsRetrying(true);
    try {
      await Promise.allSettled([refresh(true), refreshProviders(true)]);
    } finally {
      setIsRetrying(false);
    }
  };

  const handleDismissError = () => {
    setDismissedError(error);
    setError('');
  };

  useEffect(() => {
    refresh();
    refreshProviders();
    const timer = window.setInterval(() => {
      refresh(false);
    }, 10000);
    return () => window.clearInterval(timer);
  }, [refresh, refreshProviders]);

  const selectedPhaseData = useMemo(
    () => phaseState?.phases?.find((phase: any) => phase.id === selectedPhase),
    [phaseState, selectedPhase],
  );

  const renderPrompt = async (phaseId: string) => {
    setSelectedPhase(phaseId);
    setPhasePrompt('');
    setError('');
    try {
      const result = await api.getPhasePrompt(phaseId);
      setPhasePrompt(result.prompt);
    } catch (reason: any) {
      setError(reason?.message || 'Could not render this phase prompt.');
    }
  };

  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(phasePrompt);
      setNotice('Phase prompt copied. Continue in your IDE agent; this dashboard does not run an LLM.');
      setTimeout(() => setNotice(''), 4000);
    } catch {
      setError('Clipboard access is unavailable. Select and copy the prompt text below.');
    }
  };

  const savePhaseResponse = async () => {
    if (!selectedPhase) return;
    setBusy('phase');
    setError('');
    setNotice('');
    try {
      const sources = JSON.parse(sourceJson || '[]');
      if (!Array.isArray(sources)) throw new Error('Sources must be a JSON array.');
      const result = await api.recordPhaseOutput({
        phase_id: selectedPhase,
        response: phaseResponse,
        sources,
        artifacts: artifactText.split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
      });
      setNotice(`Saved ${selectedPhase} output to ${result.record?.response_path || 'phase history'}.`);
      setPhaseResponse('');
      await refresh();
    } catch (reason: any) {
      setError(reason?.message || 'Could not save this phase output.');
    } finally {
      setBusy('');
    }
  };

  const runEda = async () => {
    setBusy('eda');
    setError('');
    try {
      const result = await api.runEda();
      const report = result.report || result;
      setEda(report);
      setNotice(`EDA report saved to ${report.paths?.markdown || report.paths?.json || 'the task artifact directory'}.`);
      await refresh();
    } catch (reason: any) {
      setError(reason?.message || 'EDA failed.');
    } finally {
      setBusy('');
    }
  };

  const grantConsent = async (provider: 'kaggle' | 'modal') => {
    setBusy(`consent-${provider}`);
    setError('');
    try {
      await api.grantUploadConsent(provider);
      setUploadChecks((current) => ({ ...current, [provider]: false }));
      setNotice(`Private upload consent saved for ${provider} and this dataset fingerprint.`);
      await refresh();
      await refreshProviders();
    } catch (reason: any) {
      setError(reason?.message || `Could not save ${provider} upload consent.`);
    } finally {
      setBusy('');
    }
  };

  const providers: ProviderStatus[] = Object.values(cloudState?.providers || {}) as ProviderStatus[];

  return (
    <div className="flex-1 overflow-y-auto space-y-5 pb-8">
      <section className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">{t('wshub_title', 'Agent workspace')}</h2>
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-zinc-400">
              {t('wshub_desc', 'Prompts and phase history stay in this task workspace. Copy a phase prompt into your IDE agent, then save its response and the sources or artifacts it actually used.')}
            </p>
          </div>
          <button onClick={handleManualRetry} disabled={isRetrying} className="rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800 transition flex items-center gap-1.5">
            <RefreshCw className={`w-3.5 h-3.5 ${isRetrying ? 'animate-spin' : ''}`} />
            <span>{t('wshub_refresh_btn', 'Refresh workspace state')}</span>
          </button>
        </div>
        {phaseState && (
          <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-zinc-400">
            <span className="rounded bg-zinc-950 px-2 py-1">{t('wshub_prompt_pack', 'Prompt pack:')} <code>{phaseState.pack}</code></span>
            <span className="rounded bg-zinc-950 px-2 py-1">{t('wshub_next_phase', 'Next phase:')} <strong className="text-zinc-200">{phaseState.next_phase || t('wshub_initial_phases_complete', 'initial phases complete')}</strong></span>
            <span className="rounded bg-zinc-950 px-2 py-1">{t('wshub_evidence', 'Evidence:')} <code>{String(phaseState.current_evidence_revision || '').slice(0, 12)}</code></span>
          </div>
        )}
        {error && (
          <div role="alert" className="mt-3 rounded-xl border border-red-900/70 bg-red-950/40 p-3.5 text-xs text-red-200 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span className="font-semibold text-red-100">
                  {t('err_service_notice_title', 'Workspace Service Notice')}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleManualRetry}
                  disabled={isRetrying}
                  className="flex items-center gap-1 px-2.5 py-1 bg-red-900/60 hover:bg-red-800 text-red-100 rounded-md text-[11px] font-semibold transition disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
                  <span>{t('btn_retry', 'Retry')}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowErrorDetails(!showErrorDetails)}
                  className="px-2 py-1 bg-zinc-900/80 hover:bg-zinc-800 text-zinc-300 rounded-md text-[11px] transition border border-zinc-800"
                >
                  {showErrorDetails ? t('btn_hide_details', 'Hide') : t('btn_details', 'Details')}
                </button>
                <button
                  type="button"
                  onClick={handleDismissError}
                  className="px-2 py-1 text-zinc-400 hover:text-zinc-200 transition text-[11px]"
                >
                  {t('btn_dismiss', 'Dismiss')}
                </button>
              </div>
            </div>
            {showErrorDetails && (
              <div className="p-2.5 bg-black/50 rounded-lg font-mono text-[11px] text-red-300 whitespace-pre-wrap break-all border border-red-950">
                {error}
              </div>
            )}
          </div>
        )}
        {notice && <p role="status" className="mt-3 rounded border border-emerald-900 bg-emerald-950/30 p-3 text-xs text-emerald-300">{notice}</p>}
      </section>

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(250px,0.8fr)_minmax(0,1.6fr)]">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <h3 className="text-sm font-bold text-white">{t('wshub_eight_phases', 'Eight competition phases')}</h3>
          <div className="mt-3 space-y-2">
            {(phaseState?.phases || []).map((phase: any, index: number) => {
              const selectable = !phase.completed || phase.can_revisit;
              return (
                <button
                  key={phase.id}
                  disabled={!selectable}
                  onClick={() => renderPrompt(phase.id)}
                  className={`flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-2 text-left text-xs transition ${
                    selectedPhase === phase.id
                      ? 'border-indigo-500 bg-indigo-950/50 text-white'
                      : selectable
                        ? 'border-zinc-800 bg-zinc-950/60 text-zinc-300 hover:border-zinc-600'
                        : 'cursor-not-allowed border-zinc-900 bg-zinc-950/30 text-zinc-600'
                  }`}
                >
                  <span><span className="mr-2 font-mono text-zinc-500">{index + 1}.</span>{phase.title}</span>
                  <span className="whitespace-nowrap text-[10px]">
                    {phase.completed ? (phase.can_revisit ? t('wshub_new_evidence', 'new evidence') : t('wshub_runs_saved', '{runs} saved', { runs: phase.runs })) : t('wshub_pending', 'pending')}
                  </span>
                </button>
              );
            })}
          </div>
          {selectedPhaseData?.latest_response_path && (
            <p className="mt-3 break-all text-[10px] text-zinc-500">{t('wshub_latest_output', 'Latest output: {path}', { path: selectedPhaseData.latest_response_path })}</p>
          )}
          {selectedPhaseData?.record_blocked_reason && (
            <p className="mt-3 rounded border border-amber-900 bg-amber-950/30 p-2 text-[10px] leading-relaxed text-amber-300">{selectedPhaseData.record_blocked_reason}</p>
          )}
        </div>

        <div className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-white">{selectedPhaseData?.title || t('wshub_phase_prompt_fallback', 'Phase prompt')}</h3>
              <p className="mt-1 text-[11px] text-zinc-500">{t('wshub_edit_prompt_hint', 'Edit prompt files in the workspace and version them with Git.')}</p>
            </div>
            <button disabled={!phasePrompt} onClick={copyPrompt} className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-40">
              {t('wshub_copy_prompt_btn', 'Copy prompt for IDE agent')}
            </button>
          </div>
          <textarea
            aria-label={t('wshub_rendered_prompt_aria', 'Rendered phase prompt')}
            value={phasePrompt}
            readOnly
            placeholder={t('wshub_select_phase_ph', 'Select an available phase to render its prompt and current task context.')}
            className="h-64 w-full resize-y rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-300 outline-none"
          />
          <div className="grid grid-cols-1 gap-3">
            <label className="text-xs text-zinc-400">
              {t('wshub_agent_response', 'Agent response')}
              <textarea value={phaseResponse} onChange={(event) => setPhaseResponse(event.target.value)} className="mt-1 h-40 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-200 outline-none focus:border-indigo-600" placeholder={t('wshub_agent_response_ph', "Paste the IDE agent's phase response here. The saved output is checked against the phase contract.")} />
            </label>
            <label className="text-xs text-zinc-400">
              {t('wshub_sources_inspected', 'Sources actually inspected (JSON array)')}
              <textarea value={sourceJson} onChange={(event) => setSourceJson(event.target.value)} className="mt-1 h-20 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] text-zinc-200 outline-none focus:border-indigo-600" placeholder={'[{"url":"https://...","title":"..."}]'} />
            </label>
            <label className="text-xs text-zinc-400">
              {t('wshub_artifact_paths', 'Workspace artifact paths (one per line)')}
              <textarea value={artifactText} onChange={(event) => setArtifactText(event.target.value)} className="mt-1 h-16 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] text-zinc-200 outline-none focus:border-indigo-600" placeholder="artifacts/exp001/DIAGNOSTICS.md" />
            </label>
            <div className="flex justify-end">
              <button disabled={!selectedPhase || !phaseResponse.trim() || busy === 'phase' || Boolean(selectedPhaseData?.record_blocked_reason)} onClick={savePhaseResponse} className="rounded-md bg-emerald-700 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-600 disabled:opacity-40">
                {busy === 'phase' ? t('wshub_saving', 'Saving…') : t('wshub_save_output_btn', 'Save phase output')}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white">{t('wshub_eda_title', 'Exploratory data analysis')}</h3>
              <p className="mt-1 text-xs text-zinc-400">{t('wshub_eda_desc', "Profiles declared training data locally. Reports omit raw rows and are saved under this task's artifact tree.")}</p>
            </div>
            <button onClick={runEda} disabled={busy === 'eda'} className="shrink-0 rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">
              {busy === 'eda' ? t('wshub_running_eda', 'Running…') : t('wshub_run_eda_btn', 'Run EDA')}
            </button>
          </div>
          {eda?.paths && <p className="mt-3 break-all text-[10px] text-zinc-500">{t('wshub_latest_eda_report', 'Latest report: {path}', { path: eda.paths.markdown || eda.paths.json })}</p>}
          <div className="mt-3 space-y-3">
            {(eda?.reports || []).map((report: any) => (
              <div key={report.source} className="rounded-lg border border-zinc-800 bg-zinc-950/70 p-3">
                <div className="flex flex-wrap justify-between gap-2 text-xs">
                  <span className="font-mono text-zinc-200">{report.source}</span>
                  <span className="text-zinc-400">{t('wshub_eda_stats', '{rows} rows · {cols} columns · {dups} duplicates', { rows: report.row_count?.toLocaleString(), cols: report.column_count, dups: report.duplicate_row_count })}</span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(report.columns || []).slice(0, 12).map((column: any) => (
                    <span key={column.name} className="rounded border border-zinc-800 px-2 py-1 text-[10px] text-zinc-400">
                      {t('wshub_eda_missing', '{name}: {dtype}, {count} missing', { name: column.name, dtype: column.dtype, count: column.missing_count })}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {!eda?.reports?.length && <p className="text-xs text-zinc-500">{t('wshub_no_eda_report', 'No EDA report saved yet.')}</p>}
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white">{t('wshub_accounts_title', 'Kaggle and Modal accounts')}</h3>
              <p className="mt-1 text-xs text-zinc-400">{t('wshub_accounts_desc', 'The harness reads official CLI profiles and credentials. Provider quotas and charges depend on the account.')}</p>
            </div>
            <button onClick={onOpenEnvironmentModal} className="shrink-0 rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800">{t('wshub_account_setup_btn', 'Account setup')}</button>
          </div>
          <div className="mt-4 space-y-3">
            {providers.map((provider) => {
              const fingerprint = provider.dataset?.fingerprint || '';
              const consent = Boolean(provider.dataset?.upload_consent);
              const usage = provider.provider === 'kaggle' ? provider.quota : provider.billing;
              const usageLabel = provider.provider === 'kaggle' ? t('wshub_quota_accel', 'Accelerator quota') : t('wshub_billing_summary', 'Current month billing summary');
              return (
                <div key={provider.provider} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <strong className="text-xs capitalize text-zinc-100">{provider.provider}</strong>
                    <span className={`rounded px-2 py-1 text-[10px] ${provider.authenticated ? 'bg-emerald-950 text-emerald-300' : 'bg-amber-950 text-amber-300'}`}>
                      {provider.authenticated ? `${t('status_connected', 'connected')}${provider.account ? ` · ${provider.account}` : ''}` : t('status_not_connected', 'not connected')}
                    </span>
                  </div>
                  <p className="mt-2 text-[10px] text-zinc-500">CLI: {provider.cli_installed ? (provider.cli_version || t('status_installed', 'installed')) : t('status_not_installed', 'not installed')} · {t('wshub_data_files', '{count} data files', { count: provider.dataset?.file_count ?? 0 })} · {t('wshub_fingerprint', 'fingerprint {fp}', { fp: fingerprint.slice(0, 12) })}</p>
                  <p className="mt-1 text-[10px] text-zinc-500">{t('wshub_upload_policy_desc', 'Current upload policy: private. Consent is tied to this task and dataset fingerprint.')}</p>
                  <div className="mt-2 rounded border border-zinc-800 bg-zinc-900/70 p-2">
                    <p className="text-[10px] font-semibold text-zinc-300">{usageLabel}</p>
                    {usage?.status === 'available' || usage?.status === 'available_text' ? (
                      <details className="mt-1">
                        <summary className="cursor-pointer text-[10px] text-indigo-300">CLI report · {usage.observed_at ? new Date(usage.observed_at).toLocaleString() : 'available'}</summary>
                        <pre className="mt-2 max-h-36 overflow-auto whitespace-pre-wrap break-words text-[10px] text-zinc-400">{usage.raw || JSON.stringify(usage.data, null, 2)}</pre>
                        {usage.note && <p className="mt-1 text-[10px] text-amber-300">{usage.note}</p>}
                      </details>
                    ) : (
                      <p className="mt-1 text-[10px] text-zinc-500">{usage?.reason || (usage?.status === 'not_exposed' ? t('wshub_no_numeric_quota', 'No numeric quota is reported by this CLI.') : t('wshub_quota_unavailable', 'Unavailable; connect the official CLI account and refresh.'))}</p>
                    )}
                  </div>
                  {provider.warnings?.map((warning) => <p key={warning} className="mt-1 text-[10px] text-amber-300">{warning}</p>)}
                  {consent ? (
                    <p className="mt-2 text-[10px] text-emerald-300">{t('wshub_upload_approved', 'Upload approved for this fingerprint.')}</p>
                  ) : (
                    <div className="mt-2">
                      <label className="flex items-start gap-2 text-[10px] leading-relaxed text-zinc-400">
                        <input type="checkbox" checked={Boolean(uploadChecks[provider.provider])} onChange={(event) => setUploadChecks((current) => ({ ...current, [provider.provider]: event.target.checked }))} className="mt-0.5 accent-indigo-500" />
                        {t('wshub_upload_consent', 'I approve uploading these {count} task data files to {provider} for this fingerprint. Keep the upload private.', { count: provider.dataset?.file_count ?? 0, provider: provider.provider })}
                      </label>
                      <button disabled={!uploadChecks[provider.provider] || busy === `consent-${provider.provider}`} onClick={() => grantConsent(provider.provider)} className="mt-2 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">
                        {busy === `consent-${provider.provider}` ? t('wshub_saving', 'Saving…') : t('wshub_approve_upload_btn', 'Approve {provider} upload', { provider: provider.provider })}
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
            {!providers.length && <p className="text-xs text-zinc-500">{t('wshub_provider_unavailable', 'Provider status is unavailable. Refresh after connecting an official CLI account.')}</p>}
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-white">{t('wshub_jobs_title', 'Execution jobs')}</h3>
            <p className="mt-1 text-xs text-zinc-400">{t('wshub_jobs_desc', "Shared job state from CLI and dashboard. Results remain in this workspace's artifact tree.")}</p>
          </div>
          <span className="text-[10px] text-zinc-500">{t('wshub_refreshes_interval', 'refreshes every 10 seconds')}</span>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[620px] text-left text-xs">
            <thead className="text-[10px] uppercase text-zinc-500">
              <tr>
                <th className="py-2 pr-3">{t('wshub_th_job', 'Job')}</th>
                <th className="py-2 pr-3">{t('wshub_th_provider', 'Provider')}</th>
                <th className="py-2 pr-3">{t('wshub_th_exp', 'Experiment')}</th>
                <th className="py-2 pr-3">{t('wshub_th_status', 'Status')}</th>
                <th className="py-2 pr-3">{t('wshub_th_metrics', 'Metrics')}</th>
                <th className="py-2">{t('wshub_th_artifacts', 'Artifacts')}</th>
              </tr>
            </thead>
            <tbody>
              {jobs.slice(0, 12).map((job) => (
                <tr key={job.job_id} className="border-t border-zinc-800 text-zinc-300">
                  <td className="py-2 pr-3 font-mono">{job.job_id}</td>
                  <td className="py-2 pr-3 capitalize">{job.backend}</td>
                  <td className="py-2 pr-3 font-mono">{job.exp_id}</td>
                  <td className="py-2 pr-3">{prettyStatus(job.status)}{job.provider_status && job.provider_status !== job.status && <span className="ml-1 text-[10px] text-zinc-500">({prettyStatus(job.provider_status)})</span>}</td>
                  <td className="py-2 pr-3 font-mono text-[10px]">
                    {Object.entries(job.result_metrics || {}).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)).slice(0, 2).map(([name, value]) => name + '=' + Number(value).toFixed(4)).join(' · ') || '—'}
                  </td>
                  <td className="py-2 font-mono text-[10px]">{job.artifact_dir || '—'}{job.error_message && <span className="ml-2 text-red-300">{job.error_message}</span>}</td>
                </tr>
              ))}
              {!jobs.length && <tr><td colSpan={6} className="py-4 text-center text-zinc-500">{t('wshub_no_jobs', 'No execution jobs recorded yet.')}</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
