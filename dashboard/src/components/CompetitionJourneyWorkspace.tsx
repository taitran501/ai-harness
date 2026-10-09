import React, { useState, useEffect, useRef } from 'react';
import {
  Compass,
  Trophy,
  AlertOctagon,
  Rocket,
  CheckCircle2,
  Lock,
  Unlock,
  Layers,
  ShieldAlert,
  Sparkles,
  FileText,
  GitBranch,
  Cloud,
  Award,
  Settings2,
  Play,
  RefreshCw,
  Check,
  ArrowRight,
  ArrowLeft,
  Pin,
  Bot,
  Activity,
  History,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Globe,
  Sliders,
  Table,
} from 'lucide-react';
import { api } from '../services/api';
import { useLanguage } from '../constants/i18n';
import {
  TaskInfo,
  ExperimentSummary,
  SubmissionEntry,
  TreeGraphResponse,
  UCBRecommendation,
} from '../types/experiment';
import { TreeCanvas } from './TreeCanvas';
import { LeaderboardTable } from './LeaderboardTable';
import { SubmissionsTab } from './SubmissionsTab';
import { CloudRunnerHub } from './CloudRunnerHub';
import { CompetitionPipelineHub } from './CompetitionPipelineHub';
import { PrimingCenter } from './PrimingCenter';

interface CompetitionJourneyWorkspaceProps {
  compState: any;
  taskInfo: TaskInfo | null;
  experiments: ExperimentSummary[];
  submissions: SubmissionEntry[];
  graphData: TreeGraphResponse;
  recommendation: UCBRecommendation | null;
  primingPrompt: string;
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  onOpenTaskConfig: (tab?: 'task' | 'data' | 'cv' | 'rules' | 'submission') => void;
  onOpenConfigModal: (targetPath?: string, parentId?: string) => void;
  onOpenEnvModal: () => void;
  onRefreshAll: () => Promise<void>;
}

export const CompetitionJourneyWorkspace: React.FC<CompetitionJourneyWorkspaceProps> = ({
  compState,
  taskInfo,
  experiments,
  submissions,
  graphData,
  recommendation,
  primingPrompt,
  selectedNodeId,
  onSelectNode,
  onOpenTaskConfig,
  onOpenConfigModal,
  onOpenEnvModal,
  onRefreshAll,
}) => {
  const { t } = useLanguage();
  const specSummary = compState?.spec_summary || {};
  const baselineScore = Number(specSummary.baseline_goal ?? 0);
  const evalProto = specSummary.evaluation_protocol || {};
  const finalPolicy = specSummary.final_selection_policy || { max_candidates: 2, slots: [] };
  const hud = compState?.hud || {};
  const whereAmI = hud.where_am_i || {
    phase_index: 4,
    phase_title: '4. Experiment Loop',
    phase_question: 'What are we testing, against what benchmark, and what is the next hypothesis?',
    mode: 'EXPLORE',
    active_tracks: {
      experimentation: 'active',
      candidate_management: 'active',
      external_evaluation: 'active',
    },
  };
  const currentBest = hud.what_is_current_best || {
    baseline: { id: 'baseline', score: baselineScore },
    cv_best: { id: 'baseline', score: baselineScore, delta: 0 },
    public_lb_best: { enabled: true, label: 'Public Leaderboard', sub_id: null, score: null, quota_used_today: 0, daily_quota: 5 },
    private_frozen: { label: 'Private Evaluation', max_candidates: 2, is_frozen: false, slot_1_safe_cv: null, slot_2_aggressive_lb: null },
  };
  const blockers: any[] = hud.what_is_blocking_me || [];
  const nextAction = hud.what_should_i_do_next || {
    target_phase: 4,
    title: 'Investigate errors in current best candidate',
    detail: 'Compare branches and synthesize next candidate.',
    cta_label: 'Open phase',
    agent_plan: [
      'Inspect parent diagnostics & failure slices',
      'Compare recent experiments in solution tree',
      'Formulate atomic hypothesis & build next candidate',
    ],
  };

  const agentCopilot = compState?.agent_copilot || { status: 'IDLE', steps: [], history: [] };
  const auditTrail: any[] = compState?.audit_trail || [];
  const phases: any[] = compState?.phases || [];
  const gates = compState?.gates || {};
  const dataReport = compState?.data_readiness_report || {};
  const dataAuditCfg = compState?.data_audit_config || {};
  const agentCtxCfg = compState?.agent_context_config || {};

  // --- Local State for Context Injection (Optimistic UI) ---
  const [localAgentCtxCfg, setLocalAgentCtxCfg] = useState<Record<string, boolean>>(() => ({
    inject_phase_hud: true,
    inject_current_best_anchors: true,
    inject_active_blockers: true,
    inject_next_action: true,
    inject_data_risk_warnings: true,
    inject_candidate_pool: true,
    inject_submission_quota_and_divergence: true,
    inject_idea_graveyard: true,
    ...(compState?.agent_context_config || {}),
  }));

  const [ctxSaveStatus, setCtxSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const previousSyncedCtxRef = useRef<Record<string, boolean>>(localAgentCtxCfg);
  const ctxSaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isSavingCtxRef = useRef<boolean>(false);

  useEffect(() => {
    if (compState?.agent_context_config && !isSavingCtxRef.current) {
      setLocalAgentCtxCfg((prev) => ({
        ...prev,
        ...compState.agent_context_config,
      }));
      previousSyncedCtxRef.current = {
        ...previousSyncedCtxRef.current,
        ...compState.agent_context_config,
      };
    }
  }, [compState?.agent_context_config]);

  useEffect(() => {
    return () => {
      if (ctxSaveTimeoutRef.current) {
        clearTimeout(ctxSaveTimeoutRef.current);
      }
    };
  }, []);

  const handleToggleAgentContext = (key: string, checked: boolean) => {
    const updated = { ...localAgentCtxCfg, [key]: checked };
    setLocalAgentCtxCfg(updated);
    setCtxSaveStatus('saving');

    if (ctxSaveTimeoutRef.current) {
      clearTimeout(ctxSaveTimeoutRef.current);
    }

    ctxSaveTimeoutRef.current = setTimeout(async () => {
      isSavingCtxRef.current = true;
      try {
        const res = await api.postCompetitionAction('agent-context', { config: updated });
        if (res?.success) {
          previousSyncedCtxRef.current = updated;
          setCtxSaveStatus('saved');
          setTimeout(() => {
            setCtxSaveStatus((curr) => (curr === 'saved' ? 'idle' : curr));
          }, 2000);
        } else {
          throw new Error('Server responded with unsuccessful status');
        }
      } catch (err: any) {
        console.error('Failed to persist agent context configuration:', err);
        setLocalAgentCtxCfg(previousSyncedCtxRef.current);
        setCtxSaveStatus('error');
        setStatusMsg(`Failed to save settings: ${err.message || String(err)}. Reverted.`);
        setTimeout(() => {
          setCtxSaveStatus('idle');
        }, 4000);
      } finally {
        isSavingCtxRef.current = false;
      }
    }, 300);
  };

  // --- Local State for Data Audit Configuration (Optimistic UI) ---
  const [localDataAuditCfg, setLocalDataAuditCfg] = useState<Record<string, any>>(() => ({
    auto_scan_mode: 'deep',
    check_group_leakage: true,
    check_conditional_labels: true,
    check_label_distribution: true,
    check_text_length_outliers: true,
    ...(compState?.data_audit_config || {}),
  }));

  const previousSyncedAuditRef = useRef<Record<string, any>>(localDataAuditCfg);
  const auditSaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (compState?.data_audit_config) {
      setLocalDataAuditCfg((prev) => ({
        ...prev,
        ...compState.data_audit_config,
      }));
      previousSyncedAuditRef.current = {
        ...previousSyncedAuditRef.current,
        ...compState.data_audit_config,
      };
    }
  }, [compState?.data_audit_config]);

  useEffect(() => {
    return () => {
      if (auditSaveTimeoutRef.current) {
        clearTimeout(auditSaveTimeoutRef.current);
      }
    };
  }, []);

  const handleUpdateDataAuditCfg = (key: string, val: any) => {
    const updated = { ...localDataAuditCfg, [key]: val };
    setLocalDataAuditCfg(updated);

    if (auditSaveTimeoutRef.current) {
      clearTimeout(auditSaveTimeoutRef.current);
    }

    auditSaveTimeoutRef.current = setTimeout(async () => {
      try {
        await api.postCompetitionAction('data-audit', { config: updated });
        previousSyncedAuditRef.current = updated;
      } catch (e: any) {
        console.error('Data audit config update failed:', e);
        setLocalDataAuditCfg(previousSyncedAuditRef.current);
        setStatusMsg(`Failed to save audit settings: ${e.message || String(e)}. Reverted.`);
      }
    }, 300);
  };
  const candidatePool = compState?.candidate_pool || {
    available_strategies: [],
    all_candidates: [],
    subtask_specialists: {},
    stable_candidates: [],
    rejected_experiments: [],
    synthesized_candidates: [],
  };
  const freezeSelection = compState?.freeze_selection || {};
  const finalization = compState?.finalization || {};

  const [selectedPhase, setSelectedPhase] = useState<number>(whereAmI.phase_index || 4);
  const [phase4SubView, setPhase4SubView] = useState<'cards' | 'table' | 'tree' | 'cloud' | 'runner'>('cards');
  const [candidateFilter, setCandidateFilter] = useState<'ALL' | 'KEEP' | 'REJECT'>('ALL');
  const [busy, setBusy] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  // Progressive Disclosure states
  const [isCopilotExpanded, setIsCopilotExpanded] = useState<boolean>(false);
  const [isAgentContextOpen, setIsAgentContextOpen] = useState<boolean>(false);
  const [isDataCheckCfgOpen, setIsDataCheckCfgOpen] = useState<boolean>(false);

  // Phase 5 Universal Candidate Strategy state
  const [synthName, setSynthName] = useState('cand_v1');
  const [synthMode, setSynthMode] = useState<string>('specialist_router');
  const [singleSourceExp, setSingleSourceExp] = useState(
    currentBest.cv_best?.id || 'baseline'
  );
  const [routerTargets, setRouterTargets] = useState<Record<string, string>>({});
  const declaredSubtasks = taskInfo?.subtasks?.length ? taskInfo.subtasks : taskInfo?.metrics || [];
  const [ensembleChecked, setEnsembleChecked] = useState<string[]>([]);

  // Phase 7 Dynamic N-Slot Freeze state
  const maxFinalSlots = Number(finalPolicy.max_candidates || 2);
  const [slot1Safe, setSlot1Safe] = useState(
    freezeSelection.slot_1_safe_cv || currentBest.cv_best?.id || 'baseline'
  );
  const [slot2Aggr, setSlot2Aggr] = useState(
    freezeSelection.slot_2_aggressive_lb ||
      candidatePool.synthesized_candidates?.[0]?.id ||
      currentBest.cv_best?.id ||
      'baseline'
  );
  const [freezeNotes, setFreezeNotes] = useState(
    freezeSelection.freeze_notes || 'Selected according to Final Selection Policy'
  );

  // Phase 8 Closure state
  const [privScore1, setPrivScore1] = useState<string>(
    finalization.private_score_slot_1 !== null && finalization.private_score_slot_1 !== undefined
      ? String(finalization.private_score_slot_1)
      : ''
  );
  const [privScore2, setPrivScore2] = useState<string>(
    finalization.private_score_slot_2 !== null && finalization.private_score_slot_2 !== undefined
      ? String(finalization.private_score_slot_2)
      : ''
  );
  const [finalRank, setFinalRank] = useState<string>(
    finalization.final_rank ? String(finalization.final_rank) : ''
  );
  const [totalTeams, setTotalTeams] = useState<string>(
    finalization.total_teams ? String(finalization.total_teams) : '45'
  );
  const [lessonsLearned, setLessonsLearned] = useState<string>(
    finalization.lessons_learned ||
      '1. Subtask-specific context routing avoids signal dilution.\n2. Conditional target constraints must be enforced before evaluation.'
  );
  const [generatedReportMd, setGeneratedReportMd] = useState<string | null>(null);

  const runAction = async (action: string, payload: Record<string, any>, targetPhaseAfter?: number) => {
    setBusy(true);
    setStatusMsg(null);
    try {
      const res = await api.postCompetitionAction(action, payload);
      await onRefreshAll();
      if (targetPhaseAfter) {
        setSelectedPhase(targetPhaseAfter);
      } else if (action === 'agent-run' && res.state?.hud?.where_am_i?.phase_index) {
        setSelectedPhase(res.state.hud.where_am_i.phase_index);
      }
      setStatusMsg('Status updated successfully');
    } catch (e: any) {
      setStatusMsg(`Error: ${e.message || String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const [agentGoal, setAgentGoal] = useState<string>(
    agentCopilot.goal || (taskInfo?.primary_metric ? `Improve ${taskInfo.primary_metric}` : t('copilot_default_goal', 'Improve primary validation metric'))
  );
  const [maxExpBudget, setMaxExpBudget] = useState<number>(
    agentCopilot.budget?.max_experiments || 3
  );

  const activeTracks = whereAmI.active_tracks || {};
  const subtasksList: any[] = specSummary.subtasks || taskInfo?.subtasks || [];
  const rulesObj = specSummary.competition_rules || taskInfo?.competition_rules || {};

  const isBaselineEstablished = Boolean(
    gates?.baseline_locked &&
    typeof currentBest.baseline?.score === 'number' &&
    !isNaN(currentBest.baseline.score) &&
    experiments.length > 0
  );
  const isAwaitingBaseline = (selectedPhase === 1 || whereAmI.phase_index < 3) && !isBaselineEstablished;

  const rawCvBest = currentBest.cv_best?.score;
  const cvBestScore = typeof rawCvBest === 'number' && !isNaN(rawCvBest) ? rawCvBest : (typeof baselineScore === 'number' && !isNaN(baselineScore) ? baselineScore : 0);
  const rawDelta = currentBest.cv_best?.delta;
  const cvDelta = typeof rawDelta === 'number' && !isNaN(rawDelta) ? rawDelta : (cvBestScore - (typeof baselineScore === 'number' && !isNaN(baselineScore) ? baselineScore : 0));

  const filteredCandidates = (candidatePool.all_candidates || []).filter((exp: any) => {
    if (candidateFilter === 'ALL') return true;
    return exp.verdict === candidateFilter;
  });

  return (
    <div className="flex flex-col gap-4 max-w-[1600px] mx-auto w-full">
      {/* =================================================================== */}
      {/* 1. AGENT COPILOT: COMPACT PROGRESSIVE DISCLOSURE BANNER             */}
      {/* =================================================================== */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl overflow-hidden shadow-lg transition-all">
        {/* Compact Header Ribbon */}
        <div className="px-4 py-3 flex flex-wrap items-center justify-between gap-3 bg-gradient-to-r from-indigo-950/40 via-zinc-900 to-zinc-900">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-white tracking-wide">
                  {t('copilot_title', 'Autonomous Copilot')}
                </span>
                <span
                  className={`px-2 py-0.2 rounded text-[10px] font-mono uppercase font-semibold ${
                    agentCopilot.status === 'RUNNING'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'
                      : agentCopilot.pending_approval_action
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                      : 'bg-zinc-800 text-zinc-400'
                  }`}
                >
                  {agentCopilot.status === 'RUNNING' ? t('copilot_running', 'RUNNING') : (agentCopilot.status || t('copilot_ready', 'Ready'))}
                </span>
                <span className="text-[10px] text-zinc-500 font-mono hidden sm:inline">
                  {t('copilot_stage_target', 'Stage {stage} Target', { stage: nextAction.target_phase })}
                </span>
              </div>
              <p className="text-xs text-zinc-300 truncate max-w-xl mt-0.5">
                <span className="text-indigo-400 font-medium">{t('copilot_next_label', 'Next:')} </span>
                {nextAction.title}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {agentCopilot.pending_approval_action && (
              <button
                onClick={() =>
                  runAction('agent-approve', { run_id: agentCopilot.last_run_id })
                }
                disabled={busy}
                className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs flex items-center gap-1.5 shadow animate-pulse"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>{t('copilot_approve_action', 'Approve Action')}</span>
              </button>
            )}

            <button
              onClick={() =>
                runAction('agent-run', { goal: agentGoal, max_experiments: maxExpBudget })
              }
              disabled={busy}
              className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-1.5 shadow transition"
            >
              <Rocket className={`w-3.5 h-3.5 ${busy ? 'animate-spin' : ''}`} />
              <span>{busy ? t('copilot_is_running', 'Running...') : t('copilot_run_loop', 'Run Copilot Loop')}</span>
            </button>

            <button
              onClick={() => setIsCopilotExpanded(!isCopilotExpanded)}
              className="px-2.5 py-1.5 rounded-lg bg-zinc-800/80 hover:bg-zinc-700 text-zinc-300 hover:text-white text-xs font-medium flex items-center gap-1 transition border border-zinc-700/60"
              title={t('copilot_expand_tooltip', 'Expand Copilot controls and step trace')}
            >
              <Sliders className="w-3.5 h-3.5 text-zinc-400" />
              <span className="hidden sm:inline">{t('copilot_details', 'Details')}</span>
              {isCopilotExpanded ? (
                <ChevronUp className="w-3.5 h-3.5 text-zinc-400" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
              )}
            </button>
          </div>
        </div>

        {/* Collapsible Expanded Panel (Goal, Budget, Checklist, Active Tracks, Trace) */}
        {isCopilotExpanded && (
          <div className="p-4 border-t border-zinc-800 bg-zinc-950/60 space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
              {/* Left Column: Goal & Plan */}
              <div className="lg:col-span-7 space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs">
                  <div className="sm:col-span-3">
                    <label className="block text-[10px] uppercase text-zinc-400 font-bold mb-1">
                      {t('copilot_run_goal', 'Autonomous Run Goal')}
                    </label>
                    <input
                      type="text"
                      value={agentGoal}
                      onChange={(e) => setAgentGoal(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-white text-xs"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase text-zinc-400 font-bold mb-1">
                      {t('copilot_max_exp', 'Max Experiments')}
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={15}
                      value={maxExpBudget}
                      onChange={(e) => setMaxExpBudget(Number(e.target.value))}
                      className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-white font-mono text-xs"
                    />
                  </div>
                </div>

                <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 space-y-1.5">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-300 flex items-center justify-between">
                    <span>{t('copilot_plan_checklist', 'Active Plan Checklist')}</span>
                    <span className="text-[10px] font-mono text-zinc-500">
                      AUTO / USER_APPROVAL
                    </span>
                  </div>
                  <ul className="grid grid-cols-1 md:grid-cols-2 gap-1 text-xs text-zinc-300">
                    {(nextAction.agent_plan || []).map((stepText: string, i: number) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                        <span>{stepText}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Right Column: Parallel Tracks & Execution Trace */}
              <div className="lg:col-span-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-zinc-300 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-emerald-400" /> {t('copilot_parallel_tracks', 'Parallel Tracks')}
                  </span>
                  <span className="text-[10px] font-mono text-zinc-500">
                    {agentCopilot.last_run_at ? t('copilot_last_run', 'Run at {time}', { time: new Date(agentCopilot.last_run_at).toLocaleTimeString() }) : t('copilot_ready_status', 'Ready')}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-1.5 text-[11px]">
                  {[
                    { label: t('copilot_track_exp', 'Experiments'), state: activeTracks.experimentation || 'active', phase: 4 },
                    { label: t('copilot_track_cand', 'Candidates'), state: activeTracks.candidate_management || 'active', phase: 5 },
                    { label: evalProto.external_eval_label || t('copilot_track_external', 'External'), state: activeTracks.external_evaluation || 'active', phase: 6 },
                  ].map((tr, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedPhase(tr.phase)}
                      className={`p-2 rounded-lg border text-left transition ${
                        tr.state === 'active'
                          ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300'
                          : tr.state === 'frozen'
                          ? 'bg-amber-950/20 border-amber-500/30 text-amber-300'
                          : 'bg-zinc-900 border-zinc-800 text-zinc-500'
                      }`}
                    >
                      <div className="font-semibold truncate">{tr.label}</div>
                      <div className="text-[9px] uppercase mt-0.5 flex items-center gap-1">
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            tr.state === 'active' ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'
                          }`}
                        />
                        <span>{tr.state}</span>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Step-by-Step Execution Trace */}
                <div className="space-y-1 max-h-28 overflow-y-auto pr-1 text-xs">
                  {(agentCopilot.steps || []).length > 0 ? (
                    agentCopilot.steps.map((st: any, idx: number) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between bg-zinc-900 border border-zinc-800/80 rounded px-2 py-1"
                      >
                        <span className="text-zinc-300 truncate text-[11px]">
                          {idx + 1}. {st.step}
                        </span>
                        <span className="text-emerald-400 font-mono text-[10px] font-bold shrink-0 ml-2">
                          ✓ {st.status}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="text-zinc-500 text-[11px] italic py-1">
                      {t('copilot_no_steps', 'No background task steps recorded. Ready for autonomous cycle.')}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* =================================================================== */}
      {/* 2. AT-A-GLANCE EXECUTIVE KPI SUMMARY CARDS (4 CLEAN TILES)          */}
      {/* =================================================================== */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Card 1: Current Stage & Focus */}
        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-indigo-400 mb-1">
              <span className="flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5" /> {t('kpi_current_phase', 'Current Phase')}
              </span>
              <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-bold">
                {whereAmI.mode}
              </span>
            </div>
            <div className="text-sm font-bold text-white mt-1 truncate">
              {whereAmI.phase_title}
            </div>
            <p className="text-[11px] text-zinc-400 mt-0.5 line-clamp-2">
              “{whereAmI.phase_question}”
            </p>
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-800/80 flex items-center justify-between text-[11px] text-zinc-400">
            <span>{t('kpi_runs', 'Runs:')} <strong className="text-zinc-200">{whereAmI.total_experiments || 0}</strong></span>
            <span>{t('kpi_subs', 'Subs:')} <strong className="text-zinc-200">{whereAmI.total_submissions || 0}</strong></span>
          </div>
        </div>

        {/* Card 2: Current Best Validation */}
        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-emerald-400 mb-1">
              <span className="flex items-center gap-1.5">
                <Trophy className="w-3.5 h-3.5" /> {isAwaitingBaseline ? t('best_score_label', 'Best Score') : t('kpi_best_validation_title', 'Best Validation Score')}
              </span>
              <span className="text-[10px] text-zinc-400 truncate max-w-[120px]">
                {specSummary.primary_metric || t('kpi_metric_fallback', 'Metric')}
              </span>
            </div>
            {isAwaitingBaseline ? (
              <div>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className="text-lg font-bold text-zinc-300">
                    {t('status_not_started', 'Not Started')}
                  </span>
                  <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                    {t('status_awaiting_phase3', 'Awaiting Phase 3')}
                  </span>
                </div>
                <div className="text-[11px] text-zinc-400 mt-1 truncate">
                  {t('status_awaiting_phase3_baseline', 'Awaiting Phase 3 Baseline')}
                </div>
              </div>
            ) : (
              <div>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className="text-xl font-bold font-mono text-white">
                    {cvBestScore.toFixed(4)}
                  </span>
                  <span
                    className={`text-xs font-mono font-semibold px-1.5 py-0.5 rounded ${
                      cvDelta >= 0
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-rose-500/20 text-rose-300'
                    }`}
                  >
                    {cvDelta >= 0 ? `+${cvDelta.toFixed(4)}` : cvDelta.toFixed(4)}
                  </span>
                </div>
                <div className="text-[11px] text-zinc-400 font-mono mt-0.5 truncate">
                  {t('kpi_model_label', 'Model:')} {currentBest.cv_best?.id || 'baseline'}
                </div>
              </div>
            )}
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-800/80 text-[11px] text-zinc-400 flex items-center justify-between font-mono">
            {isAwaitingBaseline ? (
              <>
                <span>{t('anchor_label', 'Anchor:')} —</span>
                <span className="text-zinc-500 font-normal">{t('status_pending_phase3', 'Pending Phase 3')}</span>
              </>
            ) : (
              <>
                <span>{t('anchor_label', 'Anchor:')} {typeof currentBest.baseline?.score === 'number' && !isNaN(currentBest.baseline.score) ? currentBest.baseline.score.toFixed(4) : (typeof baselineScore === 'number' && !isNaN(baselineScore) ? baselineScore.toFixed(4) : '0.0000')}</span>
                <span className="text-emerald-400 font-semibold">{t('status_verified', 'Verified')}</span>
              </>
            )}
          </div>
        </div>

        {/* Card 3: External & Public Leaderboard */}
        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-sky-400 mb-1">
              <span className="flex items-center gap-1.5">
                <Award className="w-3.5 h-3.5" /> {t('kpi_external_eval', 'External Eval')}
              </span>
              <span className="text-[10px] text-zinc-400 font-mono">
                {t('kpi_quota_label', 'Quota: {quota}/day', { quota: evalProto.daily_quota ?? 5 })}
              </span>
            </div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-xl font-bold font-mono text-white">
                {typeof currentBest.public_lb_best?.score === 'number' && !isNaN(currentBest.public_lb_best.score) ? currentBest.public_lb_best.score.toFixed(4) : (currentBest.public_lb_best?.score ?? '—')}
              </span>
              <span className="text-[10px] text-zinc-400 uppercase">
                {currentBest.public_lb_best?.label || t('kpi_public_lb_fallback', 'Public LB')}
              </span>
            </div>
            <div className="text-[11px] text-zinc-400 font-mono mt-0.5 truncate">
              {t('kpi_sub_id', 'Sub ID:')} {currentBest.public_lb_best?.sub_id || t('kpi_none_scored', 'None scored')}
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-800/80 text-[11px] text-zinc-400 flex items-center justify-between">
            <span>{t('kpi_portfolio_limit', 'Portfolio Limit:')}</span>
            <span className="font-semibold text-zinc-200">{t('kpi_slots_max', '{slots} slots max', { slots: maxFinalSlots })}</span>
          </div>
        </div>

        {/* Card 4: Action Attention & Health */}
        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-amber-400 mb-1">
              <span className="flex items-center gap-1.5">
                <AlertOctagon className="w-3.5 h-3.5" /> {t('kpi_action_items', 'Action Items')}
              </span>
              <span
                className={`px-1.5 py-0.2 rounded text-[10px] font-semibold ${
                  blockers.length === 0
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-amber-500/20 text-amber-300'
                }`}
              >
                {t('kpi_items_count', '{count} items', { count: blockers.length })}
              </span>
            </div>
            {blockers.length === 0 ? (
              <div className="text-xs text-emerald-400 flex items-center gap-1.5 mt-2">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>{t('kpi_all_checks_passed', 'All checks passed — ready to iterate!')}</span>
              </div>
            ) : (
              <div className="space-y-1 mt-1 max-h-16 overflow-y-auto">
                {blockers.slice(0, 2).map((b, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedPhase(b.phase || 2)}
                    className="w-full text-left text-[11px] bg-zinc-950/80 border border-zinc-800 hover:border-amber-500/50 rounded px-2 py-1 flex items-center justify-between gap-1 transition"
                  >
                    <span className="text-zinc-200 truncate">{b.message}</span>
                    <span className="text-amber-300 font-semibold shrink-0 text-[10px]">
                      {b.action_label || t('kpi_fix_btn', 'Fix')} →
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-800/80 text-[11px] text-zinc-400 flex items-center justify-between">
            <span>{t('header_readiness', 'Readiness')}:</span>
            <span className="text-zinc-200 font-medium">{t('kpi_readiness_evaluated', '8 Dimensions Evaluated')}</span>
          </div>
        </div>
      </div>

      {/* =================================================================== */}
      {/* 3. STREAMLINED 8-STAGE HORIZONTAL PROGRESS TIMELINE                 */}
      {/* =================================================================== */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-3 shadow-md space-y-2">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-1.5">
          {phases.map((p) => {
            const isSelected = selectedPhase === p.index;
            const isActiveTrack = p.status === 'active_track' || p.status === 'current';
            const isDone = p.status === 'completed';
            return (
              <button
                key={p.id}
                onClick={() => setSelectedPhase(p.index)}
                className={`relative text-left p-2 rounded-lg border transition flex flex-col justify-between ${
                  isSelected
                    ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-sm ring-1 ring-indigo-500/50'
                    : isActiveTrack
                    ? 'bg-zinc-800/90 border-emerald-500/40 text-zinc-100 hover:border-emerald-500/70'
                    : isDone
                    ? 'bg-zinc-950/70 border-zinc-800 text-zinc-300 hover:border-zinc-700'
                    : 'bg-zinc-950/40 border-zinc-900 text-zinc-500 hover:border-zinc-800'
                }`}
              >
                <div className="flex items-center justify-between text-[10px] font-bold uppercase">
                  <span>P{p.index}</span>
                  {p.gate_locked ? (
                    <Lock className="w-3 h-3 text-amber-400" />
                  ) : isDone ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : isActiveTrack ? (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  ) : null}
                </div>
                <div className="text-xs font-semibold mt-1 truncate">{p.short_title}</div>
              </button>
            );
          })}
        </div>

        {/* Stage Stepper Navigation Bar */}
        <div className="flex items-center justify-between pt-1 border-t border-zinc-800/60 text-xs">
          <button
            onClick={() => setSelectedPhase(Math.max(1, selectedPhase - 1))}
            disabled={selectedPhase <= 1}
            className="flex items-center gap-1 text-zinc-400 hover:text-white disabled:opacity-30 disabled:pointer-events-none transition px-2 py-1 rounded hover:bg-zinc-800"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>{t('journey_prev_phase', 'Prev Phase')}</span>
          </button>

          <span className="text-zinc-400 text-[11px] font-medium">
            {t('journey_stage_counter', 'Stage {current} of 8:', { current: selectedPhase })} <strong className="text-zinc-200">{phases.find((p) => p.index === selectedPhase)?.short_title || ''}</strong>
          </span>

          <button
            onClick={() => setSelectedPhase(Math.min(8, selectedPhase + 1))}
            disabled={selectedPhase >= 8}
            className="flex items-center gap-1 text-zinc-400 hover:text-white disabled:opacity-30 disabled:pointer-events-none transition px-2 py-1 rounded hover:bg-zinc-800"
          >
            <span>{t('journey_next_phase', 'Next Phase')}</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {statusMsg && (
          <div className="px-3 py-1.5 rounded bg-indigo-500/10 border border-indigo-500/30 text-xs text-indigo-300 flex items-center justify-between">
            <span>{statusMsg}</span>
            <button onClick={() => setStatusMsg(null)} className="text-zinc-400 hover:text-white">
              ✕
            </button>
          </div>
        )}
      </div>

      {/* =================================================================== */}
      {/* STAGE 1: SPEC-DRIVEN COMPETITION SETUP                              */}
      {/* =================================================================== */}
      {selectedPhase === 1 && (
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white">
                {t('stage1_title', '1. Competition Setup: {name}', { name: specSummary.name || 'Benchmark Specification' })}
              </h2>
              <p className="text-xs text-zinc-400">
                {t('stage1_desc', 'Loaded dynamically from Competition Spec (configs/task.yaml)')}
              </p>
            </div>
            <button
              onClick={() => onOpenTaskConfig('task')}
              className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs font-medium text-zinc-200 flex items-center gap-1.5 border border-zinc-700 transition"
            >
              <Settings2 className="w-3.5 h-3.5" /> {t('stage1_edit_spec', 'Edit Spec')}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage1_metric_card', 'Primary Evaluation Metric')}</div>
              <div className="text-base font-bold text-emerald-400 mt-0.5">
                {specSummary.primary_metric || 'Primary metric'} ({specSummary.direction === 'minimize' ? t('direction_minimize', 'minimize') : t('direction_maximize', 'maximize')})
              </div>
              <div className="text-zinc-500 mt-1 font-mono text-[11px]">
                {t('stage1_baseline_target', 'Baseline: {baseline} → Target: {target}', { baseline: specSummary.baseline_goal ?? 'not set', target: specSummary.target_goal ?? 'not set' })}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage1_constraints_card', 'Declared Constraints')}</div>
              <div className="text-base font-bold text-indigo-400 mt-0.5">
                {rulesObj.max_params_limit
                  ? t('stage1_max_params', '≤ {params}M Params', { params: (Number(rulesObj.max_params_limit) / 1e6).toFixed(0) })
                  : t('stage1_no_param_limit', 'No Parameter Limit')}
              </div>
              <div className="text-zinc-500 mt-1 text-[11px]">
                {t('stage1_external_data', 'External Data: {status}', { status: rulesObj.allow_external_data ? t('status_allowed', 'Allowed') : t('status_forbidden', 'Forbidden') })}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage1_protocol_card', 'Evaluation Protocol')}</div>
              <div className="text-base font-bold text-amber-400 mt-0.5">
                {evalProto.evaluation_mode || 'public_private_leaderboard'}
              </div>
              <div className="text-zinc-500 mt-1 text-[11px]">
                {t('stage1_daily_quota_slots', 'Daily Quota: {quota}/day • Max Final Slots: {slots}', { quota: evalProto.daily_quota ?? 5, slots: maxFinalSlots })}
              </div>
            </div>
          </div>

          <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3.5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300 mb-2">
              {t('stage1_declared_subtasks', 'Declared Subtasks & Target Conditioning ({count} subtasks)', { count: subtasksList.length })}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
              {subtasksList.map((st: any) => (
                <div key={st.id} className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
                  <div className="font-bold text-white">
                    {st.name} (`{st.id}` • {t('stage1_subtask_weight', 'Weight {pct}%', { pct: Math.round((st.weight ?? 1) * 100) })})
                  </div>
                  <div className="text-zinc-400 text-[11px] mt-0.5">{st.description || st.task_type}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Progressive Disclosure: Agent Context Injection Settings */}
          <div className="border-t border-zinc-800/80 pt-3">
            <div className="flex items-center justify-between w-full py-1">
              <button
                type="button"
                onClick={() => setIsAgentContextOpen(!isAgentContextOpen)}
                className="flex items-center gap-1.5 text-xs font-semibold text-zinc-300 hover:text-white transition"
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                <span>{t('stage1_agent_context_settings', 'Advanced Agent Context Injection Settings')}</span>
                {isAgentContextOpen ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
              </button>

              {/* Non-blocking status indicator */}
              <div className="flex items-center gap-1.5 text-[11px] font-mono">
                {ctxSaveStatus === 'saving' && (
                  <span className="text-zinc-400 flex items-center gap-1">
                    <RefreshCw className="w-2.5 h-2.5 animate-spin text-indigo-400" /> {t('status_saving', 'Saving...')}
                  </span>
                )}
                {ctxSaveStatus === 'saved' && (
                  <span className="text-emerald-400 flex items-center gap-1">
                    <Check className="w-3 h-3 text-emerald-400" /> {t('status_saved', 'Saved')}
                  </span>
                )}
                {ctxSaveStatus === 'error' && (
                  <span className="text-rose-400 flex items-center gap-1">
                    {t('status_failed_reverted', 'Failed (reverted)')}
                  </span>
                )}
              </div>
            </div>

            {isAgentContextOpen && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3 pt-2 border-t border-zinc-800/60 text-xs">
                {[
                  { key: 'inject_phase_hud', label: t('stage1_opt_phase_hud', 'Current Stage & Active Parallel Tracks') },
                  { key: 'inject_current_best_anchors', label: t('stage1_opt_anchors', '4-Tier Score Anchors (Baseline / CV / External / Final)') },
                  { key: 'inject_active_blockers', label: t('stage1_opt_blockers', 'Active Blockers & Required Confirmations') },
                  { key: 'inject_next_action', label: t('stage1_opt_next_action', 'Next Recommended Action & Agent Plan') },
                  { key: 'inject_data_risk_warnings', label: t('stage1_opt_data_risk', 'Spec-Driven Data Risk Warnings') },
                  { key: 'inject_candidate_pool', label: t('stage1_opt_candidate_pool', 'Candidate Pool & Subtask Specialists') },
                ].map((item) => (
                  <label
                    key={item.key}
                    className="flex items-center gap-2 p-2 rounded bg-zinc-950 border border-zinc-800 cursor-pointer text-zinc-300 hover:text-white transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={Boolean(localAgentCtxCfg[item.key])}
                      onChange={(e) => handleToggleAgentContext(item.key, e.target.checked)}
                      className="rounded border-zinc-700 bg-zinc-900 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-zinc-950 cursor-pointer"
                    />
                    <span>{item.label}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 2: DECLARATIVE DATA & PROBLEM UNDERSTANDING                   */}
      {/* =================================================================== */}
      {selectedPhase === 2 && (
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white">{t('stage2_title', '2. Data & Problem Understanding')}</h2>
              <p className="text-xs text-zinc-400">
                {t('stage2_desc', 'Executes the data_checks declared in Competition Spec (configs/task.yaml).')}
              </p>
            </div>
            <span
              className={`px-2.5 py-1 rounded text-xs font-bold ${
                gates.data_risks_confirmed
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
              }`}
            >
              {gates.data_risks_confirmed ? '✓ ' + t('stage2_data_confirmed', 'Data Readiness Confirmed') : t('stage2_review_required', 'Data Review Required')}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage2_total_samples', 'Total Samples')}</div>
              <div className="text-lg font-bold text-white mt-0.5 font-mono">
                {dataReport.dataset_overview?.total_rows?.toLocaleString() || '12,027'}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">
                {t('stage2_groups', 'Groups')} (`{dataReport.dataset_overview?.group_column || 'group_key'}`)
              </div>
              <div className="text-lg font-bold text-indigo-400 mt-0.5 font-mono">
                {dataReport.dataset_overview?.unique_groups?.toLocaleString() || '412'}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage2_spec_checks', 'Spec Checks')}</div>
              <div className="text-lg font-bold text-amber-400 mt-0.5 font-mono">
                {t('stage2_checks_count', '{count} checks', { count: dataReport.declared_checks_count ?? 4 })}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage2_cross_fold_leakage', 'Cross-Fold Leakage')}</div>
              <div className="text-lg font-bold text-emerald-400 mt-0.5 font-mono">
                {t('stage2_leaked_count', '{count} leaked', { count: dataReport.leakage_audit?.cross_fold_leaked_groups ?? 0 })}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            {(dataReport.warnings || []).map((w: any, i: number) => (
              <div
                key={i}
                className={`p-3 rounded-lg border text-xs ${
                  w.severity === 'critical'
                    ? 'bg-rose-950/30 border-rose-500/40 text-rose-200'
                    : w.severity === 'warning'
                    ? 'bg-amber-950/30 border-amber-500/40 text-amber-200'
                    : 'bg-zinc-950/80 border-zinc-800 text-zinc-300'
                }`}
              >
                <div className="font-bold text-sm flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 shrink-0" />
                  <span>{w.title}</span>
                </div>
                <p className="mt-1 text-zinc-300 leading-relaxed text-[11px]">{w.detail}</p>
              </div>
            ))}
          </div>

          <div className="pt-3 border-t border-zinc-800 flex items-center justify-between">
            <button
              onClick={() =>
                runAction(
                  'gate',
                  { gates: { data_risks_confirmed: !gates.data_risks_confirmed } },
                  !gates.data_risks_confirmed ? 3 : 2
                )
              }
              disabled={busy}
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow"
            >
              <Check className="w-4 h-4" />
              <span>
                {gates.data_risks_confirmed
                  ? t('stage2_unlock_baseline_btn', 'Data readiness confirmed — Continue to Baseline')
                  : t('stage2_confirm_risks_btn', 'Confirm data risks & unlock Baseline establishment')}
              </span>
            </button>
          </div>

          {/* Progressive Disclosure: Data Check Configuration */}
          <div className="border-t border-zinc-800/80 pt-3">
            <button
              onClick={() => setIsDataCheckCfgOpen(!isDataCheckCfgOpen)}
              className="flex items-center justify-between w-full text-xs font-semibold text-zinc-300 hover:text-white transition py-1"
            >
              <span className="flex items-center gap-1.5">
                <Settings2 className="w-3.5 h-3.5 text-emerald-400" />
                <span>{t('stage2_data_check_controls', 'Data Check Configuration Controls')}</span>
              </span>
              {isDataCheckCfgOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {isDataCheckCfgOpen && (
              <div className="space-y-3 mt-3 pt-2 border-t border-zinc-800/60 text-xs">
                <select
                  value={localDataAuditCfg.auto_scan_mode || 'deep'}
                  onChange={(e) => handleUpdateDataAuditCfg('auto_scan_mode', e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200"
                >
                  <option value="deep">{t('stage2_mode_deep', 'Deep Audit (All Spec Checks & Outliers)')}</option>
                  <option value="light">{t('stage2_mode_light', 'Light Audit (Metadata & Leakage Only)')}</option>
                  <option value="manual">{t('stage2_mode_manual', 'Manual Mode')}</option>
                </select>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {[
                    { key: 'check_group_leakage', label: t('stage2_chk_group_leakage', 'Group Isolation & Split Leakage Check') },
                    { key: 'check_conditional_labels', label: t('stage2_chk_conditional_labels', 'Conditional Target Rules Check') },
                    { key: 'check_label_distribution', label: t('stage2_chk_label_distribution', 'Class Balance & Label Sparsity Check') },
                    { key: 'check_text_length_outliers', label: t('stage2_chk_text_length_outliers', 'Input Length & Context Dilution Check') },
                  ].map((chk) => (
                    <label
                      key={chk.key}
                      className="flex items-center gap-2 p-2 rounded bg-zinc-950 border border-zinc-800 cursor-pointer text-zinc-300 hover:text-white transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={Boolean(localDataAuditCfg[chk.key])}
                        onChange={(e) => handleUpdateDataAuditCfg(chk.key, e.target.checked)}
                        className="rounded border-zinc-700 bg-zinc-900 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-zinc-950 cursor-pointer"
                      />
                      <span>{chk.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 3: OFFICIAL BASELINE ANCHOR                                   */}
      {/* =================================================================== */}
      {selectedPhase === 3 && (
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white">{t('stage3_title', '3. Establish Official Baseline Anchor')}</h2>
              <p className="text-xs text-zinc-400">
                {t('stage3_desc', 'Locks the foundational reference model before opening parallel exploration tracks.')}
              </p>
            </div>
            <button
              onClick={() =>
                runAction(
                  'gate',
                  {
                    gates: {
                      data_risks_confirmed: true,
                      baseline_locked: true,
                      baseline_exp_id: currentBest.baseline?.id || 'baseline',
                    },
                  },
                  4
                )
              }
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow transition"
            >
              <Lock className="w-3.5 h-3.5" />
              <span>{t('stage3_lock_btn', 'Establish Baseline Anchor & Activate Parallel Tracks')}</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage3_baseline_model', 'Baseline Model')}</div>
              <div className="font-mono font-bold text-white text-sm mt-0.5">
                {currentBest.baseline?.id || 'baseline'}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage3_score_anchor', 'Score Anchor')}</div>
              <div className="font-mono font-bold text-emerald-400 text-sm mt-0.5">
                {Number(currentBest.baseline?.score ?? baselineScore).toFixed(4)}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3">
              <div className="text-zinc-400 text-[11px]">{t('stage3_gate_status', 'Gate Status')}</div>
              <div className="font-bold text-zinc-200 text-sm mt-0.5">
                {gates.baseline_locked ? t('stage3_locked_active', '✓ Locked & Active') : t('stage3_pending_lock', 'Pending Lock')}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 4: EXPERIMENT LOOP (KEEP / REJECT + DRILL-DOWN)               */}
      {/* =================================================================== */}
      {selectedPhase === 4 && (
        <div className="space-y-4">
          <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-bold text-white">
                {t('stage4_title', '4. Experimentation Track')}
              </h2>
              <p className="text-xs text-zinc-400">
                {t('stage4_desc', 'Hypothesis → Train → Diagnose → Keep or Reject. Compare against baseline anchor.')}
              </p>
            </div>
            <div className="flex items-center gap-1.5 bg-zinc-950 p-1 rounded-lg border border-zinc-800">
              {[
                { id: 'cards', label: t('stage4_tab_cards', 'Candidate Cards'), icon: Layers },
                { id: 'table', label: t('stage4_tab_table', 'Leaderboard Table'), icon: Table },
                { id: 'tree', label: t('stage4_tab_tree', 'Solution Tree'), icon: GitBranch },
                { id: 'runner', label: t('stage4_tab_runner', 'Execution Runner'), icon: Play },
                { id: 'cloud', label: t('stage4_tab_cloud', 'Cloud GPU Hub'), icon: Cloud },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setPhase4SubView(tab.id as any)}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition ${
                      phase4SubView === tab.id
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {phase4SubView === 'cards' && (
            <div className="space-y-3">
              {/* Filter Row */}
              <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-800 px-3 py-2 rounded-lg text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="text-zinc-400 text-[11px] uppercase font-bold mr-1">{t('stage4_filter_label', 'Filter:')}</span>
                  {(['ALL', 'KEEP', 'REJECT'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setCandidateFilter(mode)}
                      className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                        candidateFilter === mode
                          ? 'bg-zinc-800 text-white font-bold border border-zinc-700'
                          : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      {mode === 'ALL' ? t('stage4_filter_all', 'ALL') : mode === 'KEEP' ? t('stage4_filter_keep', 'KEEP') : t('stage4_filter_reject', 'REJECT')}
                    </button>
                  ))}
                </div>
                <span className="text-zinc-500 font-mono text-[11px]">
                  {t('stage4_candidate_runs_count', '{count} candidate runs', { count: filteredCandidates.length })}
                </span>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {filteredCandidates.length === 0 ? (
                  <div className="bg-zinc-950/60 border border-zinc-800 rounded-xl p-8 text-center text-zinc-400">
                    <Layers className="w-8 h-8 mx-auto text-zinc-600 mb-2" />
                    <p className="text-sm font-medium">{t('stage4_empty', 'No candidate experiments matching this filter.')}</p>
                    <p className="text-xs text-zinc-500 mt-1">{t('stage4_empty_hint', 'Switch filter to ALL or run an experiment to see candidate cards.')}</p>
                  </div>
                ) : (
                  filteredCandidates.map((exp: any) => (
                    <div
                      key={exp.id}
                      className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 grid grid-cols-1 lg:grid-cols-4 gap-3 items-center hover:border-zinc-700 transition"
                    >
                      <div>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            exp.verdict === 'KEEP'
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : 'bg-rose-500/20 text-rose-300'
                          }`}
                        >
                          {exp.verdict}
                        </span>
                        <div
                          onClick={() => onSelectNode(exp.id)}
                          className="text-sm font-bold font-mono text-white mt-1 cursor-pointer hover:text-indigo-400"
                        >
                          {exp.id}
                        </div>
                        <div className="text-[11px] text-zinc-400">{t('stage4_parent_label', 'Parent: {id}', { id: exp.parent_id || 'root' })}</div>
                      </div>
                      <div className="lg:col-span-2 grid grid-cols-3 gap-2 text-xs">
                        <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                          <div className="text-zinc-400 text-[10px]">{t('stage4_val_score', 'Validation Score')}</div>
                          <div className="font-mono font-bold text-white">{Number(exp.composite_score).toFixed(4)}</div>
                        </div>
                        <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                          <div className="text-zinc-400 text-[10px]">{t('stage4_vs_baseline', 'vs. Baseline')}</div>
                          <div className="font-mono font-bold text-emerald-400">
                            {exp.delta_vs_baseline >= 0 ? '+' : ''}
                            {Number(exp.delta_vs_baseline).toFixed(4)}
                          </div>
                        </div>
                        <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                          <div className="text-zinc-400 text-[10px]">{t('stage4_fold_std', 'Fold Std')}</div>
                          <div className="font-mono font-bold text-zinc-300">{exp.fold_std}</div>
                        </div>
                      </div>
                      <div className="flex justify-end gap-2">
                        <button
                          onClick={() => runAction('gate', { toggle_pin_candidate: exp.id }, 5)}
                          className="px-3 py-1.5 rounded-lg bg-indigo-600/20 border border-indigo-500/40 text-indigo-300 hover:text-white hover:bg-indigo-600 text-xs font-semibold flex items-center gap-1 transition"
                        >
                          <Pin className="w-3.5 h-3.5" /> {t('stage4_promote_btn', 'Promote Candidate')}
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {phase4SubView === 'table' && (
            <LeaderboardTable
              experiments={experiments}
              submissions={submissions}
              taskInfo={taskInfo}
              onSelectNode={onSelectNode}
            />
          )}

          {phase4SubView === 'tree' && (
            <div className="h-[620px] bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden flex flex-col">
              <TreeCanvas
                graphData={graphData}
                onSelectNode={onSelectNode}
                selectedNodeId={selectedNodeId}
                recommendation={recommendation}
              />
            </div>
          )}

          {phase4SubView === 'runner' && (
            <CompetitionPipelineHub
              taskInfo={taskInfo}
              experiments={experiments}
              submissions={submissions}
              initialPrompt={primingPrompt}
              onOpenTaskConfig={onOpenTaskConfig}
              onOpenConfigModal={onOpenConfigModal}
              onOpenEnvModal={onOpenEnvModal}
              onRefreshAll={onRefreshAll}
            />
          )}

          {phase4SubView === 'cloud' && (
            <CloudRunnerHub
              taskInfo={taskInfo}
              onRefreshSolutionTree={onRefreshAll}
              onOpenConfigModal={(actionOrPath) => onOpenConfigModal(actionOrPath)}
            />
          )}
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 5: UNIVERSAL CANDIDATE MANAGEMENT                             */}
      {/* =================================================================== */}
      {selectedPhase === 5 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
            <h2 className="text-base font-bold text-white">{t('stage5_title', '5. Candidate Management Pool')}</h2>
            <p className="text-xs text-zinc-400">
              {t('stage5_desc', 'Supports Single Model promotion, Diverse Ensemble Blending, Checkpoint Selection, and Subtask Routing.')}
            </p>
            {(candidatePool.synthesized_candidates || []).map((sc: any) => (
              <div
                key={sc.id}
                className="p-3.5 rounded-lg bg-indigo-950/30 border border-indigo-500/40 flex items-center justify-between text-xs"
              >
                <div>
                  <span className="font-mono font-bold text-white">{sc.id}</span>
                  <span className="ml-2 px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px]">
                    {sc.mode}
                  </span>
                  <div className="text-zinc-400 mt-0.5 text-[11px]">{sc.oof_path}</div>
                </div>
                <button
                  onClick={() => setSelectedPhase(6)}
                  className="px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs"
                >
                  {t('stage5_eval_stage6_btn', 'Evaluate in Stage 6 →')}
                </button>
              </div>
            ))}
          </div>

          <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3 text-xs">
            <h3 className="text-sm font-bold text-white">{t('stage5_build_cand_title', 'Build / Promote Candidate')}</h3>
            <div>
              <label className="block text-zinc-400 mb-1">{t('stage5_cand_name_label', 'Candidate Name:')}</label>
              <input
                type="text"
                value={synthName}
                onChange={(e) => setSynthName(e.target.value)}
                className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
            </div>
            <div>
              <label className="block text-zinc-400 mb-1">{t('stage5_cand_strategy_label', 'Candidate Strategy:')}</label>
              <select
                value={synthMode}
                onChange={(e) => setSynthMode(e.target.value)}
                className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white"
              >
                <option value="single_model">{t('stage5_strategy_single', 'Single Model Candidate')}</option>
                <option value="ensemble_vote">{t('stage5_strategy_ensemble', 'Diverse Ensemble Blend')}</option>
                <option value="specialist_router">{t('stage5_strategy_routing', 'Multi-Subtask Specialist Routing')}</option>
                <option value="checkpoint_selection">{t('stage5_strategy_checkpoint', 'Checkpoint Selection')}</option>
              </select>
            </div>

            {synthMode === 'single_model' || synthMode === 'checkpoint_selection' ? (
              <div>
                <label className="block text-zinc-400 mb-1">{t('stage5_source_exp_label', 'Select Source Experiment:')}</label>
                <select
                  value={singleSourceExp}
                  onChange={(e) => setSingleSourceExp(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
                >
                  {(candidatePool.all_candidates || []).map((c: any) => (
                    <option key={c.id} value={c.id}>
                      {c.id} ({c.composite_score})
                    </option>
                  ))}
                </select>
              </div>
            ) : synthMode === 'specialist_router' ? (
              <div className="space-y-2">
                {declaredSubtasks.map((subtask: any) => (
                  <label key={subtask.id} className="block space-y-1 text-zinc-400">
                    {subtask.name || subtask.id}
                    <select
                      value={
                        routerTargets[subtask.id] ||
                        candidatePool.subtask_specialists?.[subtask.id]?.id ||
                        singleSourceExp
                      }
                      onChange={(e) =>
                        setRouterTargets((prev) => ({ ...prev, [subtask.id]: e.target.value }))
                      }
                      className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
                    >
                      {(candidatePool.all_candidates || []).map((candidate: any) => (
                        <option key={candidate.id} value={candidate.id}>
                          {candidate.id}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
            ) : (
              <div className="space-y-1">
                {(candidatePool.all_candidates || []).map((c: any) => (
                  <label key={c.id} className="flex items-center gap-2 p-1.5 bg-zinc-950 rounded cursor-pointer">
                    <input
                      type="checkbox"
                      checked={ensembleChecked.includes(c.id)}
                      onChange={(e) =>
                        setEnsembleChecked((prev) =>
                          e.target.checked ? [...prev, c.id] : prev.filter((x) => x !== c.id)
                        )
                      }
                    />
                    <span className="font-mono text-zinc-200">{c.id}</span>
                  </label>
                ))}
              </div>
            )}

            <button
              onClick={() =>
                runAction('synthesize-candidate', {
                  name: synthName,
                  mode: synthMode,
                  source_map:
                    synthMode === 'single_model' || synthMode === 'checkpoint_selection'
                      ? { source_exp: singleSourceExp }
                      : synthMode === 'specialist_router'
                      ? {
                          routing: Object.fromEntries(
                            declaredSubtasks.map((subtask: any) => [
                              `pred_${subtask.column || subtask.id}`,
                              routerTargets[subtask.id] ||
                                candidatePool.subtask_specialists?.[subtask.id]?.id ||
                                singleSourceExp,
                            ])
                          ),
                        }
                      : { components: ensembleChecked },
                })
              }
              disabled={busy}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition shadow"
            >
              {t('stage5_create_btn', 'Create Verified Candidate')}
            </button>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 6: EXTERNAL / PUBLIC EVALUATION                               */}
      {/* =================================================================== */}
      {selectedPhase === 6 && (
        <SubmissionsTab
          submissions={submissions}
          taskInfo={taskInfo}
          onRefresh={onRefreshAll}
          onSelectExperiment={onSelectNode}
        />
      )}

      {/* =================================================================== */}
      {/* STAGE 7: POLICY-DRIVEN FINAL SELECTION & AUDITABLE FREEZE           */}
      {/* =================================================================== */}
      {selectedPhase === 7 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
            <h2 className="text-base font-bold text-white">
              {t('stage7_title', '7. Final Selection Policy (Max Allowed Candidates: {max})', { max: maxFinalSlots })}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="bg-zinc-950 border border-emerald-500/40 rounded-xl p-3.5 space-y-2">
                <div className="text-emerald-400 font-bold">
                  {finalPolicy.slots?.[0]?.label || t('stage7_slot1', 'Slot 1: Primary Safe Candidate')}
                </div>
                <select
                  value={slot1Safe}
                  onChange={(e) => setSlot1Safe(e.target.value)}
                  disabled={gates.freeze_locked}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-white font-mono"
                >
                  {(candidatePool.all_candidates || []).map((c: any) => (
                    <option key={c.id} value={c.id}>
                      {c.id} (CV: {c.composite_score})
                    </option>
                  ))}
                </select>
              </div>
              {maxFinalSlots >= 2 && (
                <div className="bg-zinc-950 border border-indigo-500/40 rounded-xl p-3.5 space-y-2">
                  <div className="text-indigo-400 font-bold">
                    {finalPolicy.slots?.[1]?.label || t('stage7_slot2', 'Slot 2: Secondary Candidate')}
                  </div>
                  <select
                    value={slot2Aggr}
                    onChange={(e) => setSlot2Aggr(e.target.value)}
                    disabled={gates.freeze_locked}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-white font-mono"
                  >
                    {(candidatePool.synthesized_candidates || []).map((sc: any) => (
                      <option key={sc.id} value={sc.id}>
                        {sc.id} ({sc.mode})
                      </option>
                    ))}
                    {(candidatePool.all_candidates || []).map((c: any) => (
                      <option key={c.id} value={c.id}>
                        {c.id} (CV: {c.composite_score})
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-zinc-800">
              {gates.freeze_locked && (
                <button
                  onClick={() =>
                    runAction('unfreeze', { reason: 'Emergency Unfreeze logged in Audit Trail' })
                  }
                  className="px-3.5 py-2 rounded-lg bg-rose-600 text-white text-xs font-semibold flex items-center gap-1.5"
                >
                  <Unlock className="w-4 h-4" /> {t('stage7_unfreeze_btn', 'Emergency Unfreeze')}
                </button>
              )}
              <button
                onClick={() =>
                  runAction(
                    'freeze',
                    { slot_1_safe_cv: slot1Safe, slot_2_aggressive_lb: slot2Aggr, notes: freezeNotes },
                    8
                  )
                }
                className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold flex items-center gap-1.5 shadow"
              >
                <Lock className="w-4 h-4" /> {t('stage7_freeze_btn', 'Freeze Final Portfolio')} {t('stage7_slots_count', '({max} slots)', { max: maxFinalSlots })}
              </button>
            </div>
          </div>

          {/* Immutable Audit Trail Panel */}
          <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3 text-xs">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <History className="w-4 h-4 text-amber-400" /> {t('stage7_audit_trail', 'Audit Trail')}
            </h3>
            <div className="space-y-2 max-h-56 overflow-y-auto">
              {auditTrail.map((ev, idx) => (
                <div key={idx} className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                  <div className="font-mono font-bold text-amber-300">{ev.event_type}</div>
                  <div className="text-zinc-300 mt-0.5">{ev.summary}</div>
                  <div className="text-[10px] text-zinc-500 mt-0.5">{ev.timestamp}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 8: COMPETITION CLOSURE & REPRODUCIBILITY                      */}
      {/* =================================================================== */}
      {selectedPhase === 8 && (
        <div className="space-y-5">
          <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
            <h2 className="text-base font-bold text-white">
              {t('stage8_title', '8. Competition Closure')} ({t('stage8_motto', 'Local Validation ≠ External Score ≠ Final Score ≠ Rank')})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
              <input
                type="number"
                step="0.0001"
                placeholder={t('stage8_score_slot1_ph', 'Final Score (Slot 1)')}
                value={privScore1}
                onChange={(e) => setPrivScore1(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
              <input
                type="number"
                step="0.0001"
                placeholder={t('stage8_score_slot2_ph', 'Final Score (Slot 2)')}
                value={privScore2}
                onChange={(e) => setPrivScore2(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
              <input
                type="number"
                placeholder={t('stage8_rank_ph', 'Final Rank')}
                value={finalRank}
                onChange={(e) => setFinalRank(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
              <button
                onClick={async () => {
                  const res = await api.postCompetitionAction('finalize', {
                    private_score_slot_1: privScore1 ? Number(privScore1) : null,
                    private_score_slot_2: privScore2 ? Number(privScore2) : null,
                    final_rank: finalRank ? Number(finalRank) : null,
                    total_teams: totalTeams ? Number(totalTeams) : null,
                    lessons_learned: lessonsLearned,
                  });
                  await onRefreshAll();
                  if (res.state?.finalization) {
                    setGeneratedReportMd(t('stage8_exported_notice', 'Exported to {path}/FINAL_COMPETITION_REPORT.md', { path: taskInfo?.artifacts_dir || 'artifacts' }));
                  }
                }}
                className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold flex items-center justify-center gap-1.5 shadow"
              >
                <FileText className="w-4 h-4" /> {t('stage8_export_btn', 'Export Report')}
              </button>
            </div>
            {generatedReportMd && (
              <div className="text-xs text-emerald-400 font-mono">{generatedReportMd}</div>
            )}
          </div>
          <PrimingCenter initialPrompt={primingPrompt} taskInfo={taskInfo} />
        </div>
      )}
    </div>
  );
};
