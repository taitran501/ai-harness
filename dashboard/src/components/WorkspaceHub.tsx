import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';

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

type DatasetSource = {
  platform: string;
  kind: string;
  slug: string;
  url: string;
  local_path: string;
  imported_at: string;
  file_count: number;
  license_review_required?: boolean;
};

type LocalDataset = {
  label: string;
  role: string;
  path: string;
  kind: string;
  file_count: number;
  file_count_capped?: boolean;
  size_bytes: number;
};

const prettyStatus = (value?: string) => (value || 'UNKNOWN').replace(/_/g, ' ');

export const WorkspaceHub: React.FC<{ onOpenEnvironmentModal: () => void }> = ({ onOpenEnvironmentModal }) => {
  const [phaseState, setPhaseState] = useState<any>(null);
  const [selectedPhase, setSelectedPhase] = useState<string>('');
  const [phasePrompt, setPhasePrompt] = useState('');
  const [hasLoadedInitialPrompt, setHasLoadedInitialPrompt] = useState(false);
  const [phaseResponse, setPhaseResponse] = useState('');
  const [sourceJson, setSourceJson] = useState('[]');
  const [artifactText, setArtifactText] = useState('');
  const [eda, setEda] = useState<any>(null);
  const [localDatasets, setLocalDatasets] = useState<{ task_data: LocalDataset[]; local_files: LocalDataset[] }>({ task_data: [], local_files: [] });
  const [datasetSources, setDatasetSources] = useState<DatasetSource[]>([]);
  const [kaggleSlug, setKaggleSlug] = useState('');
  const [edaDataPath, setEdaDataPath] = useState('');
  const [datasetResearchPrompt, setDatasetResearchPrompt] = useState('');
  const [cloudState, setCloudState] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [uploadChecks, setUploadChecks] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const refresh = useCallback(async () => {
    setError('');
    const results = await Promise.allSettled([
      api.getPhases(),
      api.getLatestEda(),
      api.getCloudJobs(),
      api.getDatasetSources(),
    ]);
    const [phaseResult, edaResult, jobsResult, dataSourcesResult] = results;
    if (phaseResult.status === 'fulfilled') {
      const state = phaseResult.value;
      setPhaseState(state);
      const initialPhase = state.next_phase || state.phases?.find((phase: any) => phase.can_revisit)?.id || '';
      setSelectedPhase((current) => {
        if (current && state.phases?.some((phase: any) => phase.id === current)) return current;
        return initialPhase;
      });
      if (initialPhase && !hasLoadedInitialPrompt) {
        setHasLoadedInitialPrompt(true);
        api.getPhasePrompt(initialPhase).then((result) => setPhasePrompt(result.prompt)).catch(() => {});
      }
    } else {
      setError(phaseResult.reason?.message || 'Could not read the prompt pack. Run `ml-harness phase init` in this workspace.');
    }
    if (edaResult.status === 'fulfilled') setEda(edaResult.value?.report ?? edaResult.value);
    if (jobsResult.status === 'fulfilled') setJobs(jobsResult.value.jobs || []);
    if (dataSourcesResult.status === 'fulfilled') {
      const sources = dataSourcesResult.value.sources || [];
      setDatasetSources(sources);
    }
  }, [hasLoadedInitialPrompt]);

  const refreshLocalDatasets = useCallback(async () => {
    try {
      setLocalDatasets(await api.getLocalDatasets());
    } catch (reason: any) {
      setError(reason?.message || 'Could not inspect local workspace datasets.');
    }
  }, []);

  const refreshProviders = useCallback(async () => {
    try {
      setCloudState(await api.getCloudStatus());
    } catch (reason: any) {
      setError(reason?.message || 'Could not inspect Kaggle and Modal account status.');
    }
  }, []);

  useEffect(() => {
    refresh();
    refreshLocalDatasets();
    refreshProviders();
    const timer = window.setInterval(refresh, 10000);
    return () => window.clearInterval(timer);
  }, [refresh, refreshLocalDatasets, refreshProviders]);

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

  const runEda = async (dataPath = edaDataPath) => {
    setBusy('eda');
    setError('');
    try {
      const result = await api.runEda(dataPath || undefined);
      const report = result.report || result;
      setEda(report);
      setNotice(`Profiled ${dataPath || 'the task training data'}. EDA report saved to ${report.paths?.markdown || report.paths?.json || 'the task artifact directory'}.`);
      await refresh();
      if (selectedPhase) {
        try {
          const rendered = await api.getPhasePrompt(selectedPhase);
          setPhasePrompt(rendered.prompt);
        } catch {
          // The current phase may be gated; the user can render the next available phase.
        }
      }
    } catch (reason: any) {
      setError(reason?.message || 'EDA failed.');
    } finally {
      setBusy('');
    }
  };

  const importKaggleDataset = async () => {
    setBusy('dataset-import');
    setError('');
    setNotice('');
    try {
      const result = await api.importKaggleDataset(kaggleSlug);
      setDatasetSources(result.sources || []);
      setEdaDataPath(result.source.local_path);
      await refreshLocalDatasets();
      setKaggleSlug('');
      setNotice(
        result.status === 'EXISTS'
          ? `This Kaggle dataset is already in the workspace at ${result.source.local_path}.`
          : `Downloaded ${result.source.slug} to ${result.source.local_path}. The files are local and have not been uploaded to a provider. Review the dataset license before use.`,
      );
      if (selectedPhase) {
        try {
          const rendered = await api.getPhasePrompt(selectedPhase);
          setPhasePrompt(rendered.prompt);
        } catch {
          // The current phase may be gated; the user can render the next available phase.
        }
      }
    } catch (reason: any) {
      setError(reason?.message || 'Could not download this Kaggle dataset.');
    } finally {
      setBusy('');
    }
  };

  const prepareDatasetResearchPrompt = async () => {
    setBusy('dataset-research');
    setError('');
    setNotice('');
    try {
      const result = await api.getDatasetResearchPrompt();
      setDatasetResearchPrompt(result.prompt);
      try {
        await navigator.clipboard.writeText(result.prompt);
        setNotice('Research prompt copied. Paste it into your IDE agent; it will inspect local data first, then research outside sources if needed.');
      } catch {
        setNotice('Research prompt is ready below. Copy it into your IDE agent to continue.');
      }
    } catch (reason: any) {
      setError(reason?.message || 'Could not prepare the dataset research prompt.');
    } finally {
      setBusy('');
    }
  };

  const copyDatasetResearchPrompt = async () => {
    try {
      await navigator.clipboard.writeText(datasetResearchPrompt);
      setNotice('Research prompt copied. Continue in your IDE agent; this dashboard does not run an LLM.');
    } catch {
      setError('Clipboard access is unavailable. Select and copy the prompt below.');
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
            <h2 className="text-lg font-bold text-white">Agent workspace</h2>
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-zinc-400">
              Prompts and phase history stay in this task workspace. Copy a phase prompt into your IDE agent,
              then save its response and the sources or artifacts it actually used.
            </p>
          </div>
          <button onClick={async () => { await refresh(); await refreshProviders(); await refreshLocalDatasets(); }} className="rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800">
            Refresh workspace state
          </button>
        </div>
        {phaseState && (
          <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-zinc-400">
            <span className="rounded bg-zinc-950 px-2 py-1">Prompt pack: <code>{phaseState.pack}</code></span>
            <span className="rounded bg-zinc-950 px-2 py-1">Next phase: <strong className="text-zinc-200">{phaseState.next_phase || 'initial phases complete'}</strong></span>
            <span className="rounded bg-zinc-950 px-2 py-1">Evidence: <code>{String(phaseState.current_evidence_revision || '').slice(0, 12)}</code></span>
          </div>
        )}
        {error && <p role="alert" className="mt-3 rounded border border-red-900 bg-red-950/40 p-3 text-xs text-red-300">{error}</p>}
        {notice && <p role="status" className="mt-3 rounded border border-emerald-900 bg-emerald-950/30 p-3 text-xs text-emerald-300">{notice}</p>}
      </section>

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(250px,0.8fr)_minmax(0,1.6fr)]">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <h3 className="text-sm font-bold text-white">Eight competition phases</h3>
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
                    {phase.completed ? (phase.can_revisit ? 'new evidence' : `${phase.runs} saved`) : 'pending'}
                  </span>
                </button>
              );
            })}
          </div>
          {selectedPhaseData?.latest_response_path && (
            <p className="mt-3 break-all text-[10px] text-zinc-500">Latest output: {selectedPhaseData.latest_response_path}</p>
          )}
          {selectedPhaseData?.record_blocked_reason && (
            <p className="mt-3 rounded border border-amber-900 bg-amber-950/30 p-2 text-[10px] leading-relaxed text-amber-300">{selectedPhaseData.record_blocked_reason}</p>
          )}
        </div>

        <div className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-white">{selectedPhaseData?.title || 'Phase prompt'}</h3>
              <p className="mt-1 text-[11px] text-zinc-500">Edit prompt files in the workspace and version them with Git.</p>
            </div>
            <button disabled={!phasePrompt} onClick={copyPrompt} className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-40">
              Copy prompt for IDE agent
            </button>
          </div>
          <textarea
            aria-label="Rendered phase prompt"
            value={phasePrompt}
            readOnly
            placeholder="Select an available phase to render its prompt and current task context."
            className="h-64 w-full resize-y rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-300 outline-none"
          />
          <div className="grid grid-cols-1 gap-3">
            <label className="text-xs text-zinc-400">
              Agent response
              <textarea value={phaseResponse} onChange={(event) => setPhaseResponse(event.target.value)} className="mt-1 h-40 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-200 outline-none focus:border-indigo-600" placeholder="Paste the IDE agent's phase response here. The saved output is checked against the phase contract." />
            </label>
            <label className="text-xs text-zinc-400">
              Sources actually inspected (JSON array)
              <textarea value={sourceJson} onChange={(event) => setSourceJson(event.target.value)} className="mt-1 h-20 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] text-zinc-200 outline-none focus:border-indigo-600" placeholder={'[{"url":"https://...","title":"..."}]'} />
            </label>
            <label className="text-xs text-zinc-400">
              Workspace artifact paths (one per line)
              <textarea value={artifactText} onChange={(event) => setArtifactText(event.target.value)} className="mt-1 h-16 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] text-zinc-200 outline-none focus:border-indigo-600" placeholder="artifacts/exp001/DIAGNOSTICS.md" />
            </label>
            <div className="flex justify-end">
              <button disabled={!selectedPhase || !phaseResponse.trim() || busy === 'phase' || Boolean(selectedPhaseData?.record_blocked_reason)} onClick={savePhaseResponse} className="rounded-md bg-emerald-700 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-600 disabled:opacity-40">
                {busy === 'phase' ? 'Saving…' : 'Save phase output'}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <div className="rounded-xl border border-emerald-900/60 bg-emerald-950/10 p-5 xl:col-span-2">
          <div>
            <h3 className="text-sm font-bold text-white">1. Start with data already in this workspace</h3>
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-zinc-400">
              ML-Harness checks the task's configured files and common local folders first. Pick a file to profile it; nothing is copied or uploaded.
            </p>
          </div>
          <div className="mt-4 space-y-2">
            {localDatasets.task_data.map((source) => (
              <div key={`task:${source.path}`} className="flex flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <span className="text-xs font-semibold text-emerald-300">{source.label}</span>
                  <p className="mt-0.5 break-all font-mono text-[10px] text-zinc-500">{source.path} · {source.file_count_capped ? `${source.file_count}+ files` : `${source.file_count} files`} · {(source.size_bytes / (1024 * 1024)).toFixed(1)} MB{source.file_count_capped ? ' sampled' : ''}</p>
                </div>
                <button onClick={() => { setEdaDataPath(source.path); void runEda(source.path); }} disabled={busy === 'eda' || busy === 'dataset-import'} className="shrink-0 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">Profile local data</button>
              </div>
            ))}
            {localDatasets.local_files.map((source) => (
              <div key={`local:${source.path}`} className="flex flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <span className="text-xs font-semibold text-zinc-200">{source.label}</span>
                  <p className="mt-0.5 break-all font-mono text-[10px] text-zinc-500">{source.path} · {(source.size_bytes / (1024 * 1024)).toFixed(1)} MB</p>
                </div>
                <button onClick={() => { setEdaDataPath(source.path); void runEda(source.path); }} disabled={busy === 'eda' || busy === 'dataset-import'} className="shrink-0 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">Profile local data</button>
              </div>
            ))}
            {!localDatasets.task_data.length && !localDatasets.local_files.length && (
              <p className="rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 text-xs text-zinc-400">No dataset files found in this workspace yet. You can add files to a local data folder or ask your IDE agent to research suitable sources.</p>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-indigo-900/60 bg-indigo-950/15 p-5">
          <h3 className="text-sm font-bold text-white">2. Ask your IDE agent to research data</h3>
          <p className="mt-1 text-xs leading-relaxed text-zinc-400">
            The prompt asks the agent to inspect local files first, then search public sources if there is a gap. It asks for fit, source links, license checks, and leakage risks before recommending anything.
          </p>
          <button onClick={prepareDatasetResearchPrompt} disabled={busy === 'dataset-research'} className="mt-3 rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-40">
            {busy === 'dataset-research' ? 'Preparing prompt…' : 'Prepare and copy research prompt'}
          </button>
          {datasetResearchPrompt && (
            <details className="mt-3">
              <summary className="cursor-pointer text-[10px] text-indigo-300">Review the prompt copied to the clipboard</summary>
              <textarea readOnly value={datasetResearchPrompt} className="mt-2 h-48 w-full rounded-md border border-zinc-800 bg-zinc-950 p-2 font-mono text-[10px] leading-relaxed text-zinc-300" />
              <button onClick={copyDatasetResearchPrompt} className="mt-2 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800">Copy prompt again</button>
            </details>
          )}
          <p className="mt-3 text-[10px] text-zinc-500">The dashboard prepares the prompt; your IDE agent performs the web search and asks before downloading or uploading data.</p>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <details>
            <summary className="cursor-pointer text-sm font-bold text-zinc-100">3. Import a Kaggle public dataset you already found</summary>
            <p className="mt-2 text-xs leading-relaxed text-zinc-400">This also supports Kaggle datasets published outside competitions. The download stays local and does not change the task's train/test config.</p>
            <label htmlFor="kaggle-dataset-slug" className="mb-1 mt-3 block text-[11px] font-medium text-zinc-300">Dataset page slug</label>
            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                id="kaggle-dataset-slug"
                value={kaggleSlug}
                onChange={(event) => setKaggleSlug(event.target.value)}
                onKeyDown={(event) => { if (event.key === 'Enter' && kaggleSlug.trim() && busy !== 'dataset-import') void importKaggleDataset(); }}
                placeholder="owner/dataset-name"
                className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-indigo-500"
              />
              <button onClick={importKaggleDataset} disabled={!kaggleSlug.trim() || busy === 'dataset-import'} className="rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">
                {busy === 'dataset-import' ? 'Downloading…' : 'Download here'}
              </button>
            </div>
            <p className="mt-1.5 text-[10px] text-zinc-500">
              {cloudState?.providers?.kaggle?.account ? `Connected as ${cloudState.providers.kaggle.account}. ` : 'Connect Kaggle in Account setup if the CLI requests sign-in. '}
              Check the source license before using it.
            </p>
            <div className="mt-3 space-y-2">
              {datasetSources.map((source) => (
                <div key={`${source.platform}:${source.slug}`} className="flex flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <a href={source.url} target="_blank" rel="noreferrer" className="text-xs font-semibold text-indigo-300 hover:underline">{source.slug}</a>
                    <p className="mt-0.5 break-all font-mono text-[10px] text-zinc-500">{source.local_path} · {source.file_count} files</p>
                  </div>
                  <button onClick={() => { setEdaDataPath(source.local_path); void runEda(source.local_path); }} disabled={busy === 'eda' || busy === 'dataset-import'} className="shrink-0 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">Profile dataset</button>
                </div>
              ))}
              {!datasetSources.length && <p className="text-[10px] text-zinc-500">No Kaggle datasets downloaded to this workspace.</p>}
            </div>
          </details>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white">Exploratory data analysis</h3>
              <p className="mt-1 text-xs text-zinc-400">Profiles the selected data locally. Reports omit raw rows and are saved under this task's artifact tree; the selected source is included in the next phase prompt.</p>
            </div>
            <button onClick={() => void runEda()} disabled={busy === 'eda' || busy === 'dataset-import'} className="shrink-0 rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">
              {busy === 'eda' ? 'Running…' : 'Run EDA'}
            </button>
          </div>
          <label className="mt-3 block text-[10px] text-zinc-500">
            Data to profile
            <select value={edaDataPath} onChange={(event) => setEdaDataPath(event.target.value)} className="mt-1 w-full rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-300 outline-none focus:border-indigo-600">
              <option value="">Task's declared training data</option>
              {localDatasets.task_data.map((source) => <option key={`task:${source.path}`} value={source.path}>{source.label} · {source.path}</option>)}
              {localDatasets.local_files.map((source) => <option key={`local:${source.path}`} value={source.path}>{source.label} · {source.path}</option>)}
              {datasetSources.map((source) => <option key={`kaggle:${source.slug}`} value={source.local_path}>{source.slug} · {source.local_path}</option>)}
            </select>
          </label>
          {eda?.paths && <p className="mt-3 break-all text-[10px] text-zinc-500">Latest report: {eda.paths.markdown || eda.paths.json}</p>}
          <div className="mt-3 space-y-3">
            {(eda?.reports || []).map((report: any) => (
              <div key={report.source} className="rounded-lg border border-zinc-800 bg-zinc-950/70 p-3">
                <div className="flex flex-wrap justify-between gap-2 text-xs">
                  <span className="font-mono text-zinc-200">{report.source}</span>
                  <span className="text-zinc-400">{report.row_count?.toLocaleString()} rows · {report.column_count} columns · {report.duplicate_row_count} duplicates</span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(report.columns || []).slice(0, 12).map((column: any) => (
                    <span key={column.name} className="rounded border border-zinc-800 px-2 py-1 text-[10px] text-zinc-400">
                      {column.name}: {column.dtype}, {column.missing_count} missing
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {!eda?.reports?.length && <p className="text-xs text-zinc-500">No EDA report saved yet.</p>}
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white">Kaggle and Modal accounts</h3>
              <p className="mt-1 text-xs text-zinc-400">The harness reads official CLI profiles and credentials. Provider quotas and charges depend on the account.</p>
            </div>
            <button onClick={onOpenEnvironmentModal} className="shrink-0 rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800">Account setup</button>
          </div>
          <div className="mt-4 space-y-3">
            {providers.map((provider) => {
              const fingerprint = provider.dataset?.fingerprint || '';
              const consent = Boolean(provider.dataset?.upload_consent);
              const usage = provider.provider === 'kaggle' ? provider.quota : provider.billing;
              const usageLabel = provider.provider === 'kaggle' ? 'Accelerator quota' : 'Current month billing summary';
              return (
                <div key={provider.provider} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <strong className="text-xs capitalize text-zinc-100">{provider.provider}</strong>
                    <span className={`rounded px-2 py-1 text-[10px] ${provider.authenticated ? 'bg-emerald-950 text-emerald-300' : 'bg-amber-950 text-amber-300'}`}>
                      {provider.authenticated ? `connected${provider.account ? ` · ${provider.account}` : ''}` : 'not connected'}
                    </span>
                  </div>
                  <p className="mt-2 text-[10px] text-zinc-500">CLI: {provider.cli_installed ? (provider.cli_version || 'installed') : 'not installed'} · {provider.dataset?.file_count ?? 0} data files · fingerprint <code>{fingerprint.slice(0, 12)}</code></p>
                  <p className="mt-1 text-[10px] text-zinc-500">Current upload policy: private. Consent is tied to this task and dataset fingerprint.</p>
                  <div className="mt-2 rounded border border-zinc-800 bg-zinc-900/70 p-2">
                    <p className="text-[10px] font-semibold text-zinc-300">{usageLabel}</p>
                    {usage?.status === 'available' || usage?.status === 'available_text' ? (
                      <details className="mt-1">
                        <summary className="cursor-pointer text-[10px] text-indigo-300">CLI report · {usage.observed_at ? new Date(usage.observed_at).toLocaleString() : 'available'}</summary>
                        <pre className="mt-2 max-h-36 overflow-auto whitespace-pre-wrap break-words text-[10px] text-zinc-400">{usage.raw || JSON.stringify(usage.data, null, 2)}</pre>
                        {usage.note && <p className="mt-1 text-[10px] text-amber-300">{usage.note}</p>}
                      </details>
                    ) : (
                      <p className="mt-1 text-[10px] text-zinc-500">{usage?.reason || (usage?.status === 'not_exposed' ? 'No numeric quota is reported by this CLI.' : 'Unavailable; connect the official CLI account and refresh.')}</p>
                    )}
                  </div>
                  {provider.warnings?.map((warning) => <p key={warning} className="mt-1 text-[10px] text-amber-300">{warning}</p>)}
                  {consent ? (
                    <p className="mt-2 text-[10px] text-emerald-300">Upload approved for this fingerprint.</p>
                  ) : (
                    <div className="mt-2">
                      <label className="flex items-start gap-2 text-[10px] leading-relaxed text-zinc-400">
                        <input type="checkbox" checked={Boolean(uploadChecks[provider.provider])} onChange={(event) => setUploadChecks((current) => ({ ...current, [provider.provider]: event.target.checked }))} className="mt-0.5 accent-indigo-500" />
                        I approve uploading these {provider.dataset?.file_count ?? 0} task data files to {provider.provider} for this fingerprint. Keep the upload private.
                      </label>
                      <button disabled={!uploadChecks[provider.provider] || busy === `consent-${provider.provider}`} onClick={() => grantConsent(provider.provider)} className="mt-2 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">
                        {busy === `consent-${provider.provider}` ? 'Saving…' : `Approve ${provider.provider} upload`}
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
            {!providers.length && <p className="text-xs text-zinc-500">Provider status is unavailable. Refresh after connecting an official CLI account.</p>}
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-white">Execution jobs</h3>
            <p className="mt-1 text-xs text-zinc-400">Shared job state from CLI and dashboard. Results remain in this workspace's artifact tree.</p>
          </div>
          <span className="text-[10px] text-zinc-500">refreshes every 10 seconds</span>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[620px] text-left text-xs">
            <thead className="text-[10px] uppercase text-zinc-500"><tr><th className="py-2 pr-3">Job</th><th className="py-2 pr-3">Provider</th><th className="py-2 pr-3">Experiment</th><th className="py-2 pr-3">Status</th><th className="py-2 pr-3">Metrics</th><th className="py-2">Artifacts</th></tr></thead>
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
              {!jobs.length && <tr><td colSpan={6} className="py-4 text-center text-zinc-500">No execution jobs recorded yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
