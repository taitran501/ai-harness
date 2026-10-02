import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Check, Circle, Database, FlaskConical, ListChecks, Search, Send, ShieldCheck } from 'lucide-react';
import { api } from '../services/api';
import { TaskInfo } from '../types/experiment';

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

type WorkspaceStep = 'task' | 'data' | 'experiments' | 'results';

const WORKSPACE_STEPS: Array<{ id: WorkspaceStep; label: string; detail: string; icon: React.ReactNode }> = [
  { id: 'task', label: 'Task', detail: 'Rules and metric', icon: <ListChecks className="h-4 w-4" /> },
  { id: 'data', label: 'Data', detail: 'Local files and EDA', icon: <Database className="h-4 w-4" /> },
  { id: 'experiments', label: 'Experiments', detail: 'Baseline and runs', icon: <FlaskConical className="h-4 w-4" /> },
  { id: 'results', label: 'Results', detail: 'Compare and submit', icon: <Send className="h-4 w-4" /> },
];

const PHASES_BY_STEP: Record<WorkspaceStep, string[]> = {
  task: ['setup', 'understand'],
  data: ['baseline'],
  experiments: ['experiment', 'select', 'public_lb'],
  results: ['freeze', 'finalization'],
};

const stepForPhase = (phase?: string): WorkspaceStep => {
  if (phase === 'baseline') return 'data';
  if (phase === 'experiment' || phase === 'select' || phase === 'public_lb') return 'experiments';
  if (phase === 'freeze' || phase === 'finalization') return 'results';
  return 'task';
};

type PhaseAgentPanelProps = {
  phaseState: any;
  phase: any;
  selectedPhase: string;
  phasePrompt: string;
  phaseResponse: string;
  sourceJson: string;
  artifactText: string;
  busy: string;
  onSelectPhase: (phaseId: string) => void;
  onCopyPrompt: () => void;
  onPhaseResponseChange: (value: string) => void;
  onSourceJsonChange: (value: string) => void;
  onArtifactTextChange: (value: string) => void;
  onSaveResponse: () => void;
};

const PhaseAgentPanel: React.FC<PhaseAgentPanelProps> = ({
  phaseState,
  phase,
  selectedPhase,
  phasePrompt,
  phaseResponse,
  sourceJson,
  artifactText,
  busy,
  onSelectPhase,
  onCopyPrompt,
  onPhaseResponseChange,
  onSourceJsonChange,
  onArtifactTextChange,
  onSaveResponse,
}) => {
  if (!phase) return null;

  const selectablePhases = (phaseState?.phases || []).filter((item: any) =>
    (!item.completed || item.can_revisit) && !(item.id === 'understand' && item.record_blocked_reason),
  );

  return (
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-5 sm:p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-2xl">
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-300">IDE agent · current step</p>
          <h3 className="mt-1 text-lg font-bold text-white">{phase.title}</h3>
          <p className="mt-1 text-sm text-zinc-400">Copy the task-aware prompt, continue with your IDE agent, then save its response here to keep the workspace in sync.</p>
        </div>
        <button
          disabled={!phasePrompt}
          onClick={onCopyPrompt}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-indigo-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Copy prompt <ArrowRight className="h-4 w-4" />
        </button>
      </div>

      {phase.record_blocked_reason && (
        <p role="status" className="mt-4 rounded-lg border border-amber-800/70 bg-amber-950/30 p-3 text-sm leading-relaxed text-amber-200">
          This step is waiting: {phase.record_blocked_reason}
        </p>
      )}

      <div className="mt-5 grid gap-2 sm:grid-cols-3">
        {['Copy the prompt', 'Work with your IDE agent', 'Save the result here'].map((label, index) => (
          <div key={label} className="flex items-center gap-2 rounded-lg bg-zinc-950/70 px-3 py-2.5 text-xs text-zinc-300">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-800 font-mono text-[10px] text-indigo-200">{index + 1}</span>
            {label}
          </div>
        ))}
      </div>

      <details className="mt-4 rounded-lg border border-zinc-800 bg-zinc-950/60">
        <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-zinc-300 hover:text-white">Preview prompt text</summary>
        <textarea
          aria-label="Rendered phase prompt"
          value={phasePrompt}
          readOnly
          className="h-56 w-full resize-y border-t border-zinc-800 bg-zinc-950 p-4 font-mono text-xs leading-relaxed text-zinc-300 outline-none"
          placeholder="Preparing the current task prompt…"
        />
      </details>

      <details className="mt-2 rounded-lg border border-zinc-800 bg-zinc-950/60">
        <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-zinc-300 hover:text-white">Save the IDE agent response and sources</summary>
        <div className="space-y-3 border-t border-zinc-800 p-4">
          <label className="block text-sm text-zinc-300">
            Agent response
            <textarea
              value={phaseResponse}
              onChange={(event) => onPhaseResponseChange(event.target.value)}
              className="mt-2 h-40 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-sm text-zinc-100 outline-none focus:border-indigo-500"
              placeholder="Paste the agent's summary, decisions, and next steps."
            />
          </label>
          <details>
            <summary className="cursor-pointer text-xs text-zinc-400">Add sources and output files</summary>
            <div className="mt-3 grid gap-3">
              <label className="text-xs text-zinc-400">
                Sources inspected (JSON array)
                <textarea value={sourceJson} onChange={(event) => onSourceJsonChange(event.target.value)} className="mt-1 h-20 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs text-zinc-200 outline-none focus:border-indigo-500" placeholder={'[{"url":"https://...","title":"..."}]'} />
              </label>
              <label className="text-xs text-zinc-400">
                Workspace artifact paths (one per line)
                <textarea value={artifactText} onChange={(event) => onArtifactTextChange(event.target.value)} className="mt-1 h-16 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs text-zinc-200 outline-none focus:border-indigo-500" placeholder="artifacts/exp001/DIAGNOSTICS.md" />
              </label>
            </div>
          </details>
          <div className="flex justify-end">
            <button disabled={!selectedPhase || !phaseResponse.trim() || busy === 'phase' || Boolean(phase.record_blocked_reason)} onClick={onSaveResponse} className="rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-40">
              {busy === 'phase' ? 'Saving…' : 'Save phase result'}
            </button>
          </div>
        </div>
      </details>

      {selectablePhases.length > 1 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-300">Choose another available phase</summary>
          <div className="mt-2 flex flex-wrap gap-2">
            {selectablePhases.map((item: any) => (
              <button key={item.id} onClick={() => onSelectPhase(item.id)} className={`rounded-md border px-2.5 py-1.5 text-xs ${selectedPhase === item.id ? 'border-indigo-500 bg-indigo-950/50 text-indigo-200' : 'border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-200'}`}>
                {item.title}
              </button>
            ))}
          </div>
        </details>
      )}
    </section>
  );
};

export const WorkspaceHub: React.FC<{
  onOpenEnvironmentModal: () => void;
  onOpenExperiments: () => void;
  onOpenResults: () => void;
  onOpenSubmissions: () => void;
  taskInfo?: TaskInfo | null;
}> = ({ onOpenEnvironmentModal, onOpenExperiments, onOpenResults, onOpenSubmissions, taskInfo }) => {
  const [phaseState, setPhaseState] = useState<any>(null);
  const [activeStep, setActiveStep] = useState<WorkspaceStep>('task');
  const [selectedPhase, setSelectedPhase] = useState<string>('');
  const [phasePrompt, setPhasePrompt] = useState('');
  const promptRequestSequence = useRef(0);
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
        const selected = state.phases?.find((phase: any) => phase.id === current);
        if (selected && (!selected.completed || selected.can_revisit)) return current;
        return initialPhase;
      });
    } else {
      setError(phaseResult.reason?.message || 'Could not read the prompt pack. Run `ml-harness phase init` in this workspace.');
    }
    if (edaResult.status === 'fulfilled') setEda(edaResult.value?.report ?? edaResult.value);
    if (jobsResult.status === 'fulfilled') setJobs(jobsResult.value.jobs || []);
    if (dataSourcesResult.status === 'fulfilled') {
      const sources = dataSourcesResult.value.sources || [];
      setDatasetSources(sources);
    }
    return phaseResult.status === 'fulfilled' ? phaseResult.value : null;
  }, []);

  const refreshLocalDatasets = useCallback(async () => {
    try {
      const inventory = await api.getLocalDatasets();
      setLocalDatasets(inventory);
      setEdaDataPath((current) => current || inventory.task_data[0]?.path || inventory.local_files[0]?.path || '');
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

  useEffect(() => {
    const nextPhase = phaseState?.next_phase;
    if (!nextPhase) return;
    setActiveStep(stepForPhase(nextPhase));
    setSelectedPhase(nextPhase);
    setPhasePrompt('');
    const request = ++promptRequestSequence.current;
    api.getPhasePrompt(nextPhase).then((result) => {
      if (request === promptRequestSequence.current) setPhasePrompt(result.prompt);
    }).catch(() => {});
  }, [phaseState?.next_phase]);

  const renderPrompt = async (phaseId: string) => {
    setSelectedPhase(phaseId);
    setPhasePrompt('');
    setError('');
    try {
      const request = ++promptRequestSequence.current;
      const result = await api.getPhasePrompt(phaseId);
      if (request === promptRequestSequence.current) setPhasePrompt(result.prompt);
    } catch (reason: any) {
      setError(reason?.message || 'Could not render this phase prompt.');
    }
  };

  const openPhase = (phaseId: string) => {
    setActiveStep(stepForPhase(phaseId));
    void renderPrompt(phaseId);
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
      const state = await refresh();
      if (state?.next_phase || selectedPhase) await renderPrompt(state?.next_phase || selectedPhase);
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
      if (selectedPhase) await renderPrompt(selectedPhase);
    } catch (reason: any) {
      setError(reason?.message || 'Could not download this Kaggle dataset.');
    } finally {
      setBusy('');
    }
  };

  const openStep = (step: WorkspaceStep) => {
    setActiveStep(step);
    const phaseIds = PHASES_BY_STEP[step];
    const phases = phaseState?.phases || [];
    const phaseToOpen = phases.find((phase: any) => phase.id === phaseState?.next_phase && phaseIds.includes(phase.id))
      || phases.find((phase: any) => phaseIds.includes(phase.id) && phase.can_revisit);
    if (phaseToOpen && phaseToOpen.id !== selectedPhase) void renderPrompt(phaseToOpen.id);
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
  const contractNeedsReview = !taskInfo || ['needs_review', 'unverified', 'draft'].includes(
    String(taskInfo?.contract_status || '').toLowerCase(),
  );

  const currentStepIndex = WORKSPACE_STEPS.findIndex((step) => step.id === activeStep);
  const phaseForStep = useMemo(() => {
    const phaseIds = PHASES_BY_STEP[activeStep];
    const phases = phaseState?.phases || [];
    const selected = phases.find((phase: any) => phaseIds.includes(phase.id) && phase.id === selectedPhase);
    return (selected && (!selected.completed || selected.can_revisit) ? selected : undefined)
      || phases.find((phase: any) => phaseIds.includes(phase.id) && phase.id === phaseState?.next_phase)
      || phases.find((phase: any) => phaseIds.includes(phase.id) && phase.can_revisit);
  }, [activeStep, phaseState, selectedPhase]);

  const selectedDatasetLabel = localDatasets.task_data.concat(localDatasets.local_files).find((source) => source.path === edaDataPath)?.label
    || datasetSources.find((source) => source.local_path === edaDataPath)?.slug
    || edaDataPath
    || 'Choose a local file first';

  const continueToNextStep = () => {
    const nextStep = WORKSPACE_STEPS[Math.min(currentStepIndex + 1, WORKSPACE_STEPS.length - 1)];
    if (nextStep) openStep(nextStep.id);
  };

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-5 overflow-y-auto px-4 pb-10 sm:px-6">
      <section className="flex flex-col gap-4 border-b border-zinc-800/80 pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-300">Task workspace</p>
            <div className="mt-1 flex flex-wrap items-center gap-3">
              <h1 className="text-xl font-bold tracking-tight text-white sm:text-2xl">Competition workbench</h1>
              <span className={`rounded-full border px-2.5 py-1 text-xs font-medium ${contractNeedsReview ? 'border-amber-800 bg-amber-950/50 text-amber-200' : 'border-emerald-800 bg-emerald-950/40 text-emerald-200'}`}>
                {contractNeedsReview ? 'Task review needed' : 'Task contract confirmed'}
              </span>
            </div>
            <p className="mt-1 max-w-3xl text-sm text-zinc-400">Work with your IDE agent from the competition rules to a reviewed submission. This workspace keeps prompts, data, runs, and outputs together.</p>
          </div>
        </div>
        <button onClick={async () => { const state = await refresh(); await refreshProviders(); await refreshLocalDatasets(); if (state?.next_phase || selectedPhase) await renderPrompt(state?.next_phase || selectedPhase); }} className="inline-flex shrink-0 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 px-3.5 py-2.5 text-sm font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800">
          Refresh workspace
        </button>
      </section>

      {error && <p role="alert" className="rounded-lg border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">{error}</p>}
      {notice && <p role="status" className="rounded-lg border border-emerald-900 bg-emerald-950/30 p-3 text-sm text-emerald-200">{notice}</p>}

      <nav aria-label="Competition workflow" className="grid grid-cols-2 gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/70 p-2 lg:grid-cols-4">
        {WORKSPACE_STEPS.map((step, index) => {
          const active = step.id === activeStep;
          return (
            <button key={step.id} type="button" aria-current={active ? 'step' : undefined} onClick={() => openStep(step.id)} className={`flex min-h-[68px] items-center gap-3 rounded-xl border px-3 py-3 text-left transition sm:px-4 ${active ? 'border-indigo-500/70 bg-indigo-500/10 text-white' : 'border-transparent text-zinc-400 hover:border-zinc-700 hover:bg-zinc-800/70 hover:text-zinc-100'}`}>
              <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${active ? 'bg-indigo-500 text-white' : 'bg-zinc-800 text-zinc-400'}`}>{step.icon}</span>
              <span className="min-w-0">
                <span className="flex items-center gap-2 text-sm font-semibold"><span className="font-mono text-xs text-zinc-500">0{index + 1}</span>{step.label}</span>
                <span className="mt-0.5 block truncate text-xs text-zinc-500">{step.detail}</span>
              </span>
            </button>
          );
        })}
      </nav>

      {activeStep === 'task' && <>
        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 sm:p-6">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-amber-500/10 text-amber-300"><ShieldCheck className="h-5 w-5" /></span>
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-zinc-400">Step 1 of 4 · Task</p>
              <h2 className="mt-1 text-lg font-bold text-white">Confirm what the competition expects</h2>
              <p className="mt-1 max-w-3xl text-sm leading-relaxed text-zinc-400">Start with the official task rules and scoring contract. The setup prompt carries the supplied competition context and local files this workspace found.</p>
            </div>
          </div>
        <div className="mt-5 rounded-xl border border-zinc-800 bg-zinc-950/70 p-4 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <strong className="text-zinc-100">{taskInfo?.task_name || taskInfo?.name || 'Task workspace'}</strong>
            <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${contractNeedsReview ? 'bg-amber-950/70 text-amber-200' : 'bg-emerald-950/60 text-emerald-200'}`}>
              {contractNeedsReview ? 'Official review not recorded' : taskInfo?.contract_status || 'Confirmed'}
            </span>
          </div>
          {contractNeedsReview ? (
            <p className="mt-2 max-w-4xl leading-relaxed text-amber-100/90">
              Before a baseline or cloud run, ask your agent to verify the official metric, task rules, validation setup, and submission format. Draft values are not treated as confirmed facts.
            </p>
          ) : (
            <p className="mt-2 text-zinc-400">
              {taskInfo?.platform || 'Platform not listed'} · {taskInfo?.task_type || 'Task type'} · metric: {taskInfo?.primary_metric || 'not recorded'}
            </p>
          )}
        </div>
        </section>
        {phaseForStep ? <PhaseAgentPanel phaseState={phaseState} phase={phaseForStep} selectedPhase={selectedPhase} phasePrompt={phasePrompt} phaseResponse={phaseResponse} sourceJson={sourceJson} artifactText={artifactText} busy={busy} onSelectPhase={openPhase} onCopyPrompt={copyPrompt} onPhaseResponseChange={setPhaseResponse} onSourceJsonChange={setSourceJson} onArtifactTextChange={setArtifactText} onSaveResponse={savePhaseResponse} /> : <p className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-sm text-zinc-400">No task-review phase is waiting. Continue to inspect local data.</p>}
        <div className="flex justify-end"><button onClick={continueToNextStep} className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 text-sm font-semibold text-zinc-200 hover:bg-zinc-800">Continue to data <ArrowRight className="h-4 w-4" /></button></div>
      </>}

      {activeStep === 'data' && <>
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 sm:p-6">
        <div className="rounded-xl border border-emerald-900/60 bg-emerald-950/10 p-5 xl:col-span-2">
          <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-zinc-400">Step 2 of 4 · Data</p>
          <h2 className="mt-1 text-lg font-bold text-white">Start with data already on your computer</h2>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-zinc-400">
              Choose a local file to profile. This runs on your computer and does not copy or upload the dataset.
            </p>
          </div>
          <div className="mt-4 space-y-2">
            {localDatasets.task_data.map((source) => (
              <button type="button" key={`task:${source.path}`} onClick={() => setEdaDataPath(source.path)} className={`flex w-full flex-col gap-2 rounded-lg border p-3 text-left sm:flex-row sm:items-center sm:justify-between ${edaDataPath === source.path ? 'border-emerald-600 bg-emerald-950/20' : 'border-zinc-800 bg-zinc-950/70 hover:border-zinc-600'}`}>
                <div className="min-w-0">
                <span className="text-sm font-semibold text-emerald-300">{source.label}</span>
                  <p className="mt-0.5 break-all font-mono text-xs text-zinc-500">{source.path} · {source.file_count_capped ? `${source.file_count}+ files` : `${source.file_count} files`} · {(source.size_bytes / (1024 * 1024)).toFixed(1)} MB{source.file_count_capped ? ' sampled' : ''}</p>
                </div>
                <span className="shrink-0 text-xs text-emerald-300">{edaDataPath === source.path ? 'Selected' : 'Select'}</span>
              </button>
            ))}
            {localDatasets.local_files.map((source) => (
              <button type="button" key={`local:${source.path}`} onClick={() => setEdaDataPath(source.path)} className={`flex w-full flex-col gap-2 rounded-lg border p-3 text-left sm:flex-row sm:items-center sm:justify-between ${edaDataPath === source.path ? 'border-indigo-600 bg-indigo-950/20' : 'border-zinc-800 bg-zinc-950/70 hover:border-zinc-600'}`}>
                <div className="min-w-0">
                <span className="text-sm font-semibold text-zinc-200">{source.label}</span>
                  <p className="mt-0.5 break-all font-mono text-xs text-zinc-500">{source.path} · {(source.size_bytes / (1024 * 1024)).toFixed(1)} MB</p>
                </div>
                <span className="shrink-0 text-xs text-indigo-300">{edaDataPath === source.path ? 'Selected' : 'Select'}</span>
              </button>
            ))}
            {!localDatasets.task_data.length && !localDatasets.local_files.length && (
              <p className="rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 text-xs text-zinc-400">No dataset files found in this workspace yet. You can add files to a local data folder or ask your IDE agent to research suitable sources.</p>
            )}
          </div>
          <div className="mt-4 flex flex-col gap-3 rounded-xl border border-emerald-900/50 bg-zinc-950/50 p-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0"><p className="text-xs text-zinc-500">Selected file</p><p className="mt-0.5 truncate text-sm font-medium text-zinc-200">{selectedDatasetLabel}</p></div>
            <button onClick={() => void runEda()} disabled={!edaDataPath || busy === 'eda' || busy === 'dataset-import'} className="shrink-0 rounded-lg bg-emerald-700 px-3.5 py-2 text-sm font-semibold text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-40">{busy === 'eda' ? 'Profiling…' : 'Profile selected data'}</button>
          </div>
        </div>

        <details className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-zinc-200">Need another source? Ask your IDE agent to research</summary>
          <p className="mt-2 text-sm leading-relaxed text-zinc-400">The prompt asks the agent to inspect local files first, then search public sources if there is a real data gap. It asks for fit, source links, licenses, and leakage risks before recommending a dataset.</p>
          <button onClick={prepareDatasetResearchPrompt} disabled={busy === 'dataset-research'} className="mt-3 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-40">
            {busy === 'dataset-research' ? 'Preparing prompt…' : 'Prepare and copy research prompt'}
          </button>
          {datasetResearchPrompt && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-indigo-300">Preview the copied prompt</summary>
              <textarea readOnly value={datasetResearchPrompt} className="mt-2 h-40 w-full rounded-md border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-300" />
              <button onClick={copyDatasetResearchPrompt} className="mt-2 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-800">Copy prompt again</button>
            </details>
          )}
          <p className="mt-3 text-xs text-zinc-500">Your IDE agent performs the web search. You review recommendations before downloading or uploading data.</p>
        </details>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <details>
          <summary className="cursor-pointer text-sm font-semibold text-zinc-200">Or import a public Kaggle dataset you already found</summary>
            <p className="mt-2 text-sm leading-relaxed text-zinc-400">This also supports datasets published outside competitions. The download stays local and does not change the task's train/test config.</p>
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
              {cloudState?.providers?.kaggle?.account ? `Connected as ${cloudState.providers.kaggle.account}. ` : 'Connect Kaggle in Settings → Cloud accounts if the CLI requests sign-in. '}
              Check the source license before using it.
            </p>
            <div className="mt-3 space-y-2">
              {datasetSources.map((source) => (
                <div key={`${source.platform}:${source.slug}`} className="flex flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <a href={source.url} target="_blank" rel="noreferrer" className="text-xs font-semibold text-indigo-300 hover:underline">{source.slug}</a>
                    <p className="mt-0.5 break-all font-mono text-[10px] text-zinc-500">{source.local_path} · {source.file_count} files</p>
                  </div>
                  <button type="button" onClick={() => setEdaDataPath(source.local_path)} className="shrink-0 rounded-md border border-zinc-700 px-3 py-1.5 text-[10px] text-zinc-200 hover:bg-zinc-800">Select for EDA</button>
                </div>
              ))}
              {!datasetSources.length && <p className="text-[10px] text-zinc-500">No Kaggle datasets downloaded to this workspace.</p>}
            </div>
          </details>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white">Latest data profile</h3>
              <p className="mt-1 text-sm text-zinc-400">The local report is saved under this task's artifact tree; raw rows are not included.</p>
            </div>
          </div>
          <label className="mt-3 block text-[10px] text-zinc-500">
            Data to profile
            <select value={edaDataPath} onChange={(event) => setEdaDataPath(event.target.value)} className="mt-1 w-full rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-300 outline-none focus:border-indigo-600">
              <option value="">Choose a local data source</option>
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

      </section>
      {phaseForStep && <PhaseAgentPanel phaseState={phaseState} phase={phaseForStep} selectedPhase={selectedPhase} phasePrompt={phasePrompt} phaseResponse={phaseResponse} sourceJson={sourceJson} artifactText={artifactText} busy={busy} onSelectPhase={openPhase} onCopyPrompt={copyPrompt} onPhaseResponseChange={setPhaseResponse} onSourceJsonChange={setSourceJson} onArtifactTextChange={setArtifactText} onSaveResponse={savePhaseResponse} />}
      <div className="flex justify-end"><button onClick={continueToNextStep} className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 text-sm font-semibold text-zinc-200 hover:bg-zinc-800">Continue to experiments <ArrowRight className="h-4 w-4" /></button></div>
      </>}

      {activeStep === 'experiments' && <>
        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-zinc-400">Step 3 of 4 · Experiments</p>
          <h2 className="mt-1 text-lg font-bold text-white">Build a baseline, then compare focused experiments</h2>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-zinc-400">Start with a measured local run. Use Kaggle or Modal when more compute is useful; jobs and outputs stay attached to this task workspace.</p>
          {contractNeedsReview && <div className="mt-4 flex flex-col gap-3 rounded-xl border border-amber-800/70 bg-amber-950/20 p-4 sm:flex-row sm:items-center sm:justify-between"><div><p className="text-sm font-semibold text-amber-100">Experiments are waiting for the task review</p><p className="mt-1 text-sm text-amber-200/70">Confirm the official metric, validation rules, and submission contract before running a baseline.</p></div><button onClick={() => openStep('task')} className="shrink-0 rounded-lg border border-amber-800 px-3 py-2 text-sm font-medium text-amber-100 hover:bg-amber-950/60">Review task</button></div>}
          <div className="mt-5 flex flex-col gap-3 rounded-xl border border-indigo-800/60 bg-indigo-950/20 p-4 sm:flex-row sm:items-center sm:justify-between"><div><p className="text-sm font-semibold text-white">Experiment runner</p><p className="mt-1 text-sm text-zinc-400">Configure a run, do a quick local probe, then choose local CPU, Kaggle, or Modal.</p></div><button disabled={contractNeedsReview} onClick={onOpenExperiments} className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-indigo-500 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-40">Open experiment runner <ArrowRight className="h-4 w-4" /></button></div>
        </section>
        {phaseForStep && <PhaseAgentPanel phaseState={phaseState} phase={phaseForStep} selectedPhase={selectedPhase} phasePrompt={phasePrompt} phaseResponse={phaseResponse} sourceJson={sourceJson} artifactText={artifactText} busy={busy} onSelectPhase={openPhase} onCopyPrompt={copyPrompt} onPhaseResponseChange={setPhaseResponse} onSourceJsonChange={setSourceJson} onArtifactTextChange={setArtifactText} onSaveResponse={savePhaseResponse} />}
        {!phaseForStep && <p className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-sm text-zinc-400">Complete task review and establish a measured baseline first. The experiment prompt appears when those results are recorded.</p>}
        <details className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
            <summary className="cursor-pointer text-sm font-bold text-zinc-100">Cloud connections and upload consent</summary>
          <div className="mt-4">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-white">Kaggle and Modal accounts</h3>
              <p className="mt-1 text-xs text-zinc-400">The harness reads official CLI profiles and credentials. Provider quotas and charges depend on the account.</p>
            </div>
            <button onClick={onOpenEnvironmentModal} className="shrink-0 rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800">Manage connections</button>
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {(['kaggle', 'modal'] as const).map((providerName) => {
              const provider = providers.find((item) => item.provider === providerName);
              return (
                <div key={providerName} className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                  <div>
                    <p className="text-sm font-semibold capitalize text-zinc-100">{providerName}</p>
                    <p className="mt-0.5 text-xs text-zinc-500">{provider?.cli_installed ? 'CLI installed · ' + (provider.cli_version || 'version unknown') : 'CLI not installed'}</p>
                  </div>
                  <span className={`max-w-[55%] truncate rounded-full px-2.5 py-1 text-xs ${provider?.authenticated ? 'bg-emerald-950 text-emerald-300' : 'bg-zinc-800 text-zinc-400'}`}>
                    {provider?.authenticated ? 'Connected · ' + (provider.account || 'account ready') : 'Not connected'}
                  </span>
                </div>
              );
            })}
          </div>
          <details className="mt-3">
            <summary className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-300">Upload permission and usage details</summary>
          <div className="mt-3 space-y-3">
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
          </details>
        </div>
        </div>
        </details>

        <div className="flex justify-end"><button onClick={continueToNextStep} className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 text-sm font-semibold text-zinc-200 hover:bg-zinc-800">Continue to results <ArrowRight className="h-4 w-4" /></button></div>
      </>}

      {activeStep === 'results' && <>
        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-zinc-400">Step 4 of 4 · Results</p>
          <h2 className="mt-1 text-lg font-bold text-white">Compare measured runs and prepare a submission</h2>
          <p className="mt-1 max-w-3xl text-sm text-zinc-400">Select a candidate from recorded results. Submission checks follow the task contract, and you approve before a candidate is frozen or sent.</p>
          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <button onClick={onOpenResults} className="group rounded-xl border border-zinc-800 bg-zinc-950/60 p-4 text-left transition hover:border-indigo-700/70 hover:bg-zinc-950"><span className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-300"><FlaskConical className="h-5 w-5" /></span><span className="mt-3 block text-sm font-semibold text-zinc-100">Review experiment results</span><span className="mt-1 block text-sm text-zinc-500">Compare measured metrics and inspect experiment artifacts.</span><span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-indigo-300">Open results <ArrowRight className="h-3.5 w-3.5" /></span></button>
            <button onClick={onOpenSubmissions} className="group rounded-xl border border-zinc-800 bg-zinc-950/60 p-4 text-left transition hover:border-emerald-700/70 hover:bg-zinc-950"><span className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-300"><Send className="h-5 w-5" /></span><span className="mt-3 block text-sm font-semibold text-zinc-100">Build and audit submission</span><span className="mt-1 block text-sm text-zinc-500">Check output files against the task plugin's submission contract.</span><span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-emerald-300">Open submissions <ArrowRight className="h-3.5 w-3.5" /></span></button>
          </div>
        </section>
        {phaseForStep && <PhaseAgentPanel phaseState={phaseState} phase={phaseForStep} selectedPhase={selectedPhase} phasePrompt={phasePrompt} phaseResponse={phaseResponse} sourceJson={sourceJson} artifactText={artifactText} busy={busy} onSelectPhase={openPhase} onCopyPrompt={copyPrompt} onPhaseResponseChange={setPhaseResponse} onSourceJsonChange={setSourceJson} onArtifactTextChange={setArtifactText} onSaveResponse={savePhaseResponse} />}

      <details open className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
        <summary className="cursor-pointer text-sm font-bold text-zinc-100">Recent cloud jobs <span className="ml-2 text-xs font-normal text-zinc-500">{jobs.length} total</span></summary>
        <section className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/80 p-5">
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
      </details>
      </>}
    </div>
  );
};
