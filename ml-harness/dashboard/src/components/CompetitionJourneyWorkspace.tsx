import React, { useState } from 'react';
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
  Pin,
  Bot,
  Activity,
  History,
} from 'lucide-react';
import { api } from '../services/api';
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
  const specSummary = compState?.spec_summary || {};
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
    baseline: { id: 'baseline_v1_comment', score: 0.6459 },
    cv_best: { id: 'baseline_v1_comment', score: 0.6459, delta: 0 },
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
  const [phase4SubView, setPhase4SubView] = useState<'cards' | 'tree' | 'cloud' | 'runner'>('cards');
  const [busy, setBusy] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  // Phase 5 Universal Candidate Strategy state
  const [synthName, setSynthName] = useState('cand_v1');
  const [synthMode, setSynthMode] = useState<string>('specialist_router');
  const [singleSourceExp, setSingleSourceExp] = useState(
    currentBest.cv_best?.id || 'baseline_v1_comment'
  );
  const [routerSubA, setRouterSubA] = useState(
    candidatePool.subtask_specialists?.subtask_a?.id || 'baseline_v1_comment'
  );
  const [routerSubB, setRouterSubB] = useState(
    candidatePool.subtask_specialists?.subtask_b?.id || 'baseline_v1_comment'
  );
  const [routerSubC, setRouterSubC] = useState(
    candidatePool.subtask_specialists?.subtask_c_id?.id || 'baseline_v1_full_context'
  );
  const [ensembleChecked, setEnsembleChecked] = useState<string[]>([]);

  // Phase 7 Dynamic N-Slot Freeze state
  const maxFinalSlots = Number(finalPolicy.max_candidates || 2);
  const [slot1Safe, setSlot1Safe] = useState(
    freezeSelection.slot_1_safe_cv || currentBest.cv_best?.id || 'baseline_v1_comment'
  );
  const [slot2Aggr, setSlot2Aggr] = useState(
    freezeSelection.slot_2_aggressive_lb ||
      candidatePool.synthesized_candidates?.[0]?.id ||
      'baseline_v1_full_context'
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
      setStatusMsg('Đã cập nhật trạng thái Competition & đồng bộ Agent thành công!');
    } catch (e: any) {
      setStatusMsg(`Lỗi: ${e.message || String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const [agentGoal, setAgentGoal] = useState<string>(
    agentCopilot.goal || (taskInfo?.primary_metric ? `Improve ${taskInfo.primary_metric}` : 'Improve primary validation metric')
  );
  const [maxExpBudget, setMaxExpBudget] = useState<number>(
    agentCopilot.budget?.max_experiments || 3
  );

  const activeTracks = whereAmI.active_tracks || {};
  const subtasksList: any[] = specSummary.subtasks || taskInfo?.subtasks || [];
  const rulesObj = specSummary.competition_rules || taskInfo?.competition_rules || {};

  return (
    <div className="flex flex-col gap-5 p-6 max-w-[1600px] mx-auto w-full">
      {/* =================================================================== */}
      {/* PRIORITY 1: AGENT RUNTIME CONTROLLER (GOAL -> PLAN -> ACT -> STOP)  */}
      {/* =================================================================== */}
      <div className="bg-gradient-to-r from-indigo-950/90 via-zinc-900 to-zinc-900 border border-indigo-500/50 rounded-2xl p-5 shadow-xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Left: Goal, Budget & Next Action */}
          <div className="lg:col-span-7 space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-300">
              <Rocket className="w-4 h-4 text-indigo-400" />
              <span>AGENT RUNTIME CONTROLLER</span>
              {agentCopilot.last_run_id && (
                <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[10px]">
                  Run ID: {agentCopilot.last_run_id} ({agentCopilot.status})
                </span>
              )}
              <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-200 text-[10px]">
                Stage {nextAction.target_phase}
              </span>
            </div>
            <h1 className="text-xl font-extrabold text-white leading-snug">{nextAction.title}</h1>
            <p className="text-xs text-zinc-300 leading-relaxed">{nextAction.detail}</p>

            {/* Goal & Run Budget Inputs */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 pt-1 text-xs">
              <div className="sm:col-span-3">
                <label className="block text-[10px] uppercase text-indigo-300 font-bold mb-1">
                  Autonomous Run Goal:
                </label>
                <input
                  type="text"
                  value={agentGoal}
                  onChange={(e) => setAgentGoal(e.target.value)}
                  className="w-full bg-zinc-950/90 border border-indigo-500/30 rounded-lg px-3 py-1.5 text-white text-xs"
                />
              </div>
              <div>
                <label className="block text-[10px] uppercase text-indigo-300 font-bold mb-1">
                  Budget (Max Exps):
                </label>
                <input
                  type="number"
                  min={1}
                  max={15}
                  value={maxExpBudget}
                  onChange={(e) => setMaxExpBudget(Number(e.target.value))}
                  className="w-full bg-zinc-950/90 border border-indigo-500/30 rounded-lg px-3 py-1.5 text-white font-mono text-xs"
                />
              </div>
            </div>

            {/* Agent Will Checklist */}
            <div className="bg-zinc-950/80 border border-indigo-500/20 rounded-xl p-3.5 space-y-1.5">
              <div className="text-[11px] font-bold uppercase tracking-wider text-indigo-300 flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <Bot className="w-3.5 h-3.5" /> Typed Action Contract & Plan:
                </span>
                <span className="text-[10px] font-mono text-zinc-400">
                  AUTO / AUTO_WITH_LIMIT / USER_APPROVAL
                </span>
              </div>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-1.5 text-xs text-zinc-300">
                {(nextAction.agent_plan || []).map((stepText: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                    <span>{stepText}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                onClick={() =>
                  runAction('agent-run', { goal: agentGoal, max_experiments: maxExpBudget })
                }
                disabled={busy}
                className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs flex items-center gap-2 shadow-lg transition"
              >
                <Bot className={`w-4 h-4 ${busy ? 'animate-spin' : ''}`} />
                <span>{busy ? 'Agent Loop Running...' : 'Launch AgentRun Loop'}</span>
              </button>

              {agentCopilot.pending_approval_action && (
                <button
                  onClick={() =>
                    runAction('agent-approve', { run_id: agentCopilot.last_run_id })
                  }
                  disabled={busy}
                  className="px-4 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs flex items-center gap-1.5 shadow-lg animate-pulse"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  <span>
                    Approve `{agentCopilot.pending_approval_action.action}` ({agentCopilot.last_run_id})
                  </span>
                </button>
              )}

              <button
                onClick={() => setSelectedPhase(nextAction.target_phase || 4)}
                className="px-4 py-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-semibold text-xs flex items-center gap-1.5 transition"
              >
                <span>{nextAction.cta_label}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Right: Live Agent Activity Feed & Parallel Tracks Status */}
          <div className="lg:col-span-5 bg-zinc-950/90 border border-zinc-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                <Activity className="w-4 h-4" /> Agent Activity & Iteration Tracks
              </span>
              <span className="text-[10px] font-mono text-zinc-400">
                {agentCopilot.last_run_at ? `Last run: ${new Date(agentCopilot.last_run_at).toLocaleTimeString()}` : 'Ready'}
              </span>
            </div>

            {/* Parallel Active Tracks Bar */}
            <div className="grid grid-cols-3 gap-2 text-[11px]">
              {[
                { label: 'Experiment Track', state: activeTracks.experimentation || 'active', phase: 4 },
                { label: 'Candidate Track', state: activeTracks.candidate_management || 'active', phase: 5 },
                { label: evalProto.external_eval_label || 'External Eval', state: activeTracks.external_evaluation || 'active', phase: 6 },
              ].map((tr, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedPhase(tr.phase)}
                  className={`p-2 rounded-lg border text-left transition ${
                    tr.state === 'active'
                      ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300'
                      : tr.state === 'frozen'
                      ? 'bg-amber-950/30 border-amber-500/40 text-amber-300'
                      : 'bg-zinc-900 border-zinc-800 text-zinc-500'
                  }`}
                >
                  <div className="font-bold truncate">{tr.label}</div>
                  <div className="text-[10px] uppercase mt-0.5 flex items-center gap-1">
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
            <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1 text-xs">
              {(agentCopilot.steps || []).length > 0 ? (
                agentCopilot.steps.map((st: any, idx: number) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between bg-zinc-900/90 border border-zinc-800/80 rounded px-2.5 py-1.5"
                  >
                    <span className="text-zinc-200 truncate">
                      {idx + 1}. {st.step}
                    </span>
                    <span className="text-emerald-400 font-mono text-[10px] font-bold shrink-0 ml-2">
                      ✓ {st.status}
                    </span>
                  </div>
                ))
              ) : (
                <div className="text-zinc-500 text-xs italic py-2">
                  Click “Run with Agent” to execute the next scientific step automatically.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* =================================================================== */}
      {/* PRIORITY 2: 3 COMPACT HUD CARDS (WHERE / CURRENT BEST / BLOCKERS)   */}
      {/* =================================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* WHERE AM I? */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-indigo-400">
              <span className="flex items-center gap-1.5">
                <Compass className="w-4 h-4" /> Where am I?
              </span>
              <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-bold">
                {whereAmI.mode}
              </span>
            </div>
            <div className="text-base font-bold text-white mt-1">{whereAmI.phase_title}</div>
            <p className="text-xs text-zinc-400 mt-0.5">“{whereAmI.phase_question}”</p>
          </div>
          <div className="mt-3 pt-2 border-t border-zinc-800 flex items-center justify-between text-xs text-zinc-400">
            <span>Experiments: <strong className="text-white">{whereAmI.total_experiments || 0}</strong></span>
            <span>Submissions: <strong className="text-white">{whereAmI.total_submissions || 0}</strong></span>
          </div>
        </div>

        {/* WHAT IS THE CURRENT BEST? */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-emerald-400 mb-2">
            <span className="flex items-center gap-1.5">
              <Trophy className="w-4 h-4" /> What is the current best?
            </span>
            <span className="text-[10px] text-zinc-400">{specSummary.primary_metric || 'Macro-F1'}</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2">
              <div className="text-zinc-400 text-[10px]">Baseline Anchor</div>
              <div className="font-mono font-bold text-zinc-200 text-sm">
                {Number(currentBest.baseline?.score || 0.6459).toFixed(4)}
              </div>
            </div>
            <div className="bg-zinc-950 border border-emerald-500/30 rounded-lg p-2">
              <div className="text-emerald-400 text-[10px]">Best Local CV</div>
              <div className="font-mono font-bold text-emerald-300 text-sm">
                {Number(currentBest.cv_best?.score || 0.6459).toFixed(4)}
              </div>
            </div>
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2">
              <div className="text-sky-400 text-[10px] truncate">{currentBest.public_lb_best?.label}</div>
              <div className="font-mono font-bold text-sky-300 text-sm">
                {currentBest.public_lb_best?.score ?? '—'}
              </div>
            </div>
          </div>
        </div>

        {/* WHAT IS BLOCKING ME? (Human-Centric Action Language) */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-amber-400 mb-2">
            <span className="flex items-center gap-1.5">
              <AlertOctagon className="w-4 h-4" /> What requires attention?
            </span>
            <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[10px]">
              {blockers.length} items
            </span>
          </div>
          {blockers.length === 0 ? (
            <div className="text-xs text-emerald-400 flex items-center gap-1.5 mt-3">
              <CheckCircle2 className="w-4 h-4" /> All checks passed — ready to iterate!
            </div>
          ) : (
            <ul className="space-y-1.5 max-h-24 overflow-y-auto pr-1">
              {blockers.map((b, idx) => (
                <li
                  key={idx}
                  onClick={() => setSelectedPhase(b.phase || 2)}
                  className="text-xs bg-zinc-950 border border-zinc-800 hover:border-amber-500/50 cursor-pointer rounded p-1.5 flex items-center justify-between gap-2"
                >
                  <span className="text-zinc-200 truncate">
                    <strong className="text-amber-300">{b.title || 'Attention'}:</strong> {b.message}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[10px] font-semibold shrink-0">
                    {b.action_label || 'Resolve'}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* =================================================================== */}
      {/* PRIORITY 3: 8-STAGE JOURNEY BAR (WITH NON-LINEAR TRACK HIGHLIGHTS)  */}
      {/* =================================================================== */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-3">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
          {phases.map((p) => {
            const isSelected = selectedPhase === p.index;
            const isActiveTrack = p.status === 'active_track' || p.status === 'current';
            const isDone = p.status === 'completed';
            return (
              <button
                key={p.id}
                onClick={() => setSelectedPhase(p.index)}
                className={`relative text-left p-2.5 rounded-lg border transition flex flex-col justify-between ${
                  isSelected
                    ? 'bg-indigo-600/20 border-indigo-500 text-white shadow'
                    : isActiveTrack
                    ? 'bg-zinc-800/90 border-emerald-500/40 text-zinc-100'
                    : isDone
                    ? 'bg-zinc-950/70 border-zinc-800 text-zinc-300 hover:border-zinc-700'
                    : 'bg-zinc-950/40 border-zinc-900 text-zinc-500 hover:border-zinc-800'
                }`}
              >
                <div className="flex items-center justify-between text-[10px] font-bold uppercase">
                  <span>Stage {p.index}</span>
                  {p.gate_locked ? (
                    <Lock className="w-3 h-3 text-amber-400" />
                  ) : isDone ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : isActiveTrack ? (
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  ) : null}
                </div>
                <div className="text-xs font-semibold mt-1 truncate">{p.short_title}</div>
              </button>
            );
          })}
        </div>
        {statusMsg && (
          <div className="mt-2 px-3 py-1.5 rounded bg-indigo-500/10 border border-indigo-500/30 text-xs text-indigo-300 flex items-center justify-between">
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white">
                  1. Competition Setup: {specSummary.name || 'Benchmark Specification'}
                </h2>
                <p className="text-xs text-zinc-400">
                  Loaded dynamically from Competition Spec (`configs/task.yaml`).
                </p>
              </div>
              <button
                onClick={() => onOpenTaskConfig('task')}
                className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs font-medium text-zinc-200 flex items-center gap-1.5"
              >
                <Settings2 className="w-3.5 h-3.5" /> Edit Competition Spec
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">Primary Evaluation Metric</div>
                <div className="text-base font-bold text-emerald-400 mt-0.5">
                  {specSummary.primary_metric || 'Macro-F1'} ({specSummary.direction || 'maximize'})
                </div>
                <div className="text-zinc-500 mt-1">
                  Baseline: {specSummary.baseline_goal ?? 0.6459} → Target: {specSummary.target_goal ?? 0.85}
                </div>
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">Declared Constraints</div>
                <div className="text-base font-bold text-indigo-400 mt-0.5">
                  {rulesObj.max_params_limit
                    ? `≤ ${(Number(rulesObj.max_params_limit) / 1e6).toFixed(0)}M Params`
                    : 'No Parameter Limit'}
                </div>
                <div className="text-zinc-500 mt-1">
                  External Data: {rulesObj.allow_external_data ? 'Allowed' : 'Forbidden'}
                </div>
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">Evaluation Protocol</div>
                <div className="text-base font-bold text-amber-400 mt-0.5">
                  {evalProto.evaluation_mode || 'public_private_leaderboard'}
                </div>
                <div className="text-zinc-500 mt-1">
                  Daily Quota: {evalProto.daily_quota ?? 5}/day • Max Final Slots: {maxFinalSlots}
                </div>
              </div>
            </div>

            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300 mb-3">
                Declared Subtasks & Target Conditioning ({subtasksList.length} subtasks)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {subtasksList.map((st: any) => (
                  <div key={st.id} className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
                    <div className="font-bold text-white">
                      {st.name} (`{st.id}` • Weight {Math.round((st.weight || 0.25) * 100)}%)
                    </div>
                    <div className="text-zinc-400 mt-0.5">{st.description || st.task_type}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Configurable Agent Context Injection */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-4">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-indigo-400" /> Agent Context Injection Settings
            </h3>
            <div className="space-y-2 text-xs">
              {[
                { key: 'inject_phase_hud', label: 'Current Stage & Active Parallel Tracks' },
                { key: 'inject_current_best_anchors', label: '4-Tier Score Anchors (Baseline / CV / External / Final)' },
                { key: 'inject_active_blockers', label: 'Active Blockers & Required Confirmations' },
                { key: 'inject_next_action', label: 'Next Recommended Action & Agent Plan' },
                { key: 'inject_data_risk_warnings', label: 'Spec-Driven Data Risk Warnings' },
                { key: 'inject_candidate_pool', label: 'Candidate Pool & Subtask Specialists' },
              ].map((item) => (
                <label
                  key={item.key}
                  className="flex items-center gap-2.5 p-2 rounded bg-zinc-950 border border-zinc-800 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={Boolean(agentCtxCfg[item.key])}
                    onChange={(e) =>
                      runAction('agent-context', {
                        config: { ...agentCtxCfg, [item.key]: e.target.checked },
                      })
                    }
                  />
                  <span className="text-zinc-200">{item.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 2: DECLARATIVE DATA & PROBLEM UNDERSTANDING                   */}
      {/* =================================================================== */}
      {selectedPhase === 2 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white">2. Data & Problem Understanding</h2>
                <p className="text-xs text-zinc-400">
                  Executes the `data_checks` declared in Competition Spec (`configs/task.yaml`).
                </p>
              </div>
              <span
                className={`px-2.5 py-1 rounded text-xs font-bold ${
                  gates.data_risks_confirmed
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                }`}
              >
                {gates.data_risks_confirmed ? '✓ Data Readiness Confirmed' : 'Data Review Required'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">Total Samples</div>
                <div className="text-lg font-bold text-white mt-0.5">
                  {dataReport.dataset_overview?.total_rows?.toLocaleString() || '12,027'}
                </div>
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">
                  Isolation Groups (`{dataReport.dataset_overview?.group_column || 'group_key'}`)
                </div>
                <div className="text-lg font-bold text-indigo-400 mt-0.5">
                  {dataReport.dataset_overview?.unique_groups?.toLocaleString() || '412'}
                </div>
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">Declared Spec Checks</div>
                <div className="text-lg font-bold text-amber-400 mt-0.5">
                  {dataReport.declared_checks_count ?? 4} checks
                </div>
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
                <div className="text-zinc-400">Cross-Fold Leakage</div>
                <div className="text-lg font-bold text-emerald-400 mt-0.5">
                  {dataReport.leakage_audit?.cross_fold_leaked_groups ?? 0} leaked
                </div>
              </div>
            </div>

            <div className="space-y-3">
              {(dataReport.warnings || []).map((w: any, i: number) => (
                <div
                  key={i}
                  className={`p-3.5 rounded-lg border text-xs ${
                    w.severity === 'critical'
                      ? 'bg-rose-950/30 border-rose-500/40 text-rose-200'
                      : w.severity === 'warning'
                      ? 'bg-amber-950/30 border-amber-500/40 text-amber-200'
                      : 'bg-zinc-950 border-zinc-800 text-zinc-300'
                  }`}
                >
                  <div className="font-bold text-sm flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 shrink-0" />
                    <span>{w.title}</span>
                  </div>
                  <p className="mt-1 text-zinc-300 leading-relaxed">{w.detail}</p>
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
                    ? 'Data readiness confirmed — Continue to Baseline'
                    : 'Confirm data risks & unlock Baseline establishment'}
                </span>
              </button>
            </div>
          </div>

          {/* Configurable Data Audit Controls */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-4 text-xs">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <Settings2 className="w-4 h-4 text-emerald-400" /> Data Check Configuration
            </h3>
            <select
              value={dataAuditCfg.auto_scan_mode || 'deep'}
              onChange={(e) =>
                runAction('data-audit', {
                  config: { ...dataAuditCfg, auto_scan_mode: e.target.value },
                })
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200"
            >
              <option value="deep">Deep Audit (All Spec Checks & Outliers)</option>
              <option value="light">Light Audit (Metadata & Leakage Only)</option>
              <option value="manual">Manual Mode</option>
            </select>
            {[
              { key: 'check_group_leakage', label: 'Group Isolation & Split Leakage Check' },
              { key: 'check_subtask_c_conditioning', label: 'Conditional Target Rules Check' },
              { key: 'check_label_distribution', label: 'Class Balance & Label Sparsity Check' },
              { key: 'check_text_length_outliers', label: 'Input Length & Context Dilution Check' },
            ].map((chk) => (
              <label
                key={chk.key}
                className="flex items-center gap-2.5 p-2 rounded bg-zinc-950 border border-zinc-800 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={Boolean(dataAuditCfg[chk.key])}
                  onChange={(e) =>
                    runAction('data-audit', {
                      config: { ...dataAuditCfg, [chk.key]: e.target.checked },
                    })
                  }
                />
                <span className="text-zinc-200">{chk.label}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 3: OFFICIAL BASELINE ANCHOR                                   */}
      {/* =================================================================== */}
      {selectedPhase === 3 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white">3. Establish Official Baseline Anchor</h2>
              <p className="text-xs text-zinc-400">
                Locks the foundational reference model before opening the Experiment, Candidate, and Evaluation tracks.
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
                      baseline_exp_id: currentBest.baseline?.id || 'baseline_v1_comment',
                    },
                  },
                  4
                )
              }
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5"
            >
              <Lock className="w-3.5 h-3.5" />
              <span>Establish Baseline Anchor & Activate Parallel Tracks</span>
            </button>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STAGE 4: EXPERIMENT LOOP (KEEP / REJECT + DRILL-DOWN)               */}
      {/* =================================================================== */}
      {selectedPhase === 4 && (
        <div className="space-y-5">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-bold text-white">
                4. Experimentation Track (Hypothesis → Train → Diagnose → Keep / Reject)
              </h2>
              <p className="text-xs text-zinc-400">
                What are we testing? Against which parent? Did it improve validation? Should we promote it to Candidates?
              </p>
            </div>
            <div className="flex items-center gap-2">
              {[
                { id: 'cards', label: 'Experiment Cards', icon: Layers },
                { id: 'tree', label: 'Solution Tree', icon: GitBranch },
                { id: 'runner', label: 'Execution Runner', icon: Play },
                { id: 'cloud', label: 'Cloud GPU', icon: Cloud },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setPhase4SubView(tab.id as any)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                      phase4SubView === tab.id
                        ? 'bg-indigo-600 text-white'
                        : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
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
            <div className="grid grid-cols-1 gap-4">
              {(candidatePool.all_candidates || []).map((exp: any) => (
                <div
                  key={exp.id}
                  className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 grid grid-cols-1 lg:grid-cols-4 gap-4 items-center"
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
                    <div className="text-xs text-zinc-400">Parent: {exp.parent_id || 'root'}</div>
                  </div>
                  <div className="lg:col-span-2 grid grid-cols-3 gap-2 text-xs">
                    <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                      <div className="text-zinc-400 text-[10px]">Validation Score</div>
                      <div className="font-mono font-bold text-white">{Number(exp.composite_score).toFixed(4)}</div>
                    </div>
                    <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                      <div className="text-zinc-400 text-[10px]">vs. Baseline</div>
                      <div className="font-mono font-bold text-emerald-400">
                        {exp.delta_vs_baseline >= 0 ? '+' : ''}
                        {Number(exp.delta_vs_baseline).toFixed(4)}
                      </div>
                    </div>
                    <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                      <div className="text-zinc-400 text-[10px]">Fold Std</div>
                      <div className="font-mono font-bold text-zinc-300">{exp.fold_std}</div>
                    </div>
                  </div>
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => runAction('gate', { toggle_pin_candidate: exp.id }, 5)}
                      className="px-3 py-1.5 rounded-lg bg-indigo-600/20 border border-indigo-500/40 text-indigo-300 text-xs font-semibold flex items-center gap-1"
                    >
                      <Pin className="w-3.5 h-3.5" /> Promote to Candidates
                    </button>
                  </div>
                </div>
              ))}
              <LeaderboardTable
                experiments={experiments}
                submissions={submissions}
                taskInfo={taskInfo}
                onSelectNode={onSelectNode}
              />
            </div>
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-bold text-white">5. Candidate Management Pool</h2>
            <p className="text-xs text-zinc-400">
              Supports Single Model promotion, Diverse Ensemble Blending, Checkpoint Selection, and Subtask Routing.
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
                  <div className="text-zinc-400 mt-0.5">{sc.oof_path}</div>
                </div>
                <button
                  onClick={() => setSelectedPhase(6)}
                  className="px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-semibold"
                >
                  Evaluate in Stage 6 →
                </button>
              </div>
            ))}
          </div>

          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-4 text-xs">
            <h3 className="text-sm font-bold text-white">Build / Promote Candidate</h3>
            <div>
              <label className="block text-zinc-400 mb-1">Candidate Name:</label>
              <input
                type="text"
                value={synthName}
                onChange={(e) => setSynthName(e.target.value)}
                className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
            </div>
            <div>
              <label className="block text-zinc-400 mb-1">Candidate Strategy:</label>
              <select
                value={synthMode}
                onChange={(e) => setSynthMode(e.target.value)}
                className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white"
              >
                <option value="single_model">Single Model Candidate</option>
                <option value="ensemble_vote">Diverse Ensemble Blend (EnsembleGuard)</option>
                <option value="specialist_router">Multi-Subtask Specialist Routing</option>
                <option value="checkpoint_selection">Checkpoint / Fold Selection</option>
              </select>
            </div>

            {synthMode === 'single_model' || synthMode === 'checkpoint_selection' ? (
              <div>
                <label className="block text-zinc-400 mb-1">Select Source Experiment:</label>
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
                <select
                  value={routerSubA}
                  onChange={(e) => setRouterSubA(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
                >
                  {(candidatePool.all_candidates || []).map((c: any) => (
                    <option key={c.id} value={c.id}>
                      Primary A: {c.id}
                    </option>
                  ))}
                </select>
                <select
                  value={routerSubC}
                  onChange={(e) => setRouterSubC(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
                >
                  {(candidatePool.all_candidates || []).map((c: any) => (
                    <option key={c.id} value={c.id}>
                      Secondary C: {c.id}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="space-y-1">
                {(candidatePool.all_candidates || []).map((c: any) => (
                  <label key={c.id} className="flex items-center gap-2 p-1.5 bg-zinc-950 rounded">
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
                      ? { subtask_a_exp: routerSubA, subtask_b_exp: routerSubB, subtask_c_exp: routerSubC }
                      : { components: ensembleChecked },
                })
              }
              disabled={busy}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold"
            >
              Create Verified Candidate
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-5">
            <h2 className="text-lg font-bold text-white">
              7. Final Selection Policy (Max Allowed Candidates: {maxFinalSlots})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="bg-zinc-950 border border-emerald-500/40 rounded-xl p-4 space-y-2">
                <div className="text-emerald-400 font-bold">
                  {finalPolicy.slots?.[0]?.label || 'Slot 1: Primary Safe Candidate'}
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
                <div className="bg-zinc-950 border border-indigo-500/40 rounded-xl p-4 space-y-2">
                  <div className="text-indigo-400 font-bold">
                    {finalPolicy.slots?.[1]?.label || 'Slot 2: Secondary Candidate'}
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
                  className="px-4 py-2 rounded-lg bg-rose-600 text-white text-xs font-semibold flex items-center gap-1.5"
                >
                  <Unlock className="w-4 h-4" /> Emergency Unfreeze (Logged in Audit Trail)
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
                className="px-5 py-2.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold flex items-center gap-1.5"
              >
                <Lock className="w-4 h-4" /> Freeze Final Portfolio ({maxFinalSlots} slots max)
              </button>
            </div>
          </div>

          {/* Immutable Audit Trail Panel */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-3 text-xs">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <History className="w-4 h-4 text-amber-400" /> Freeze & Governance Audit Trail
            </h3>
            <div className="space-y-2 max-h-60 overflow-y-auto">
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
        <div className="space-y-6">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-bold text-white">
              8. Competition Closure (`Local Validation ≠ External Score ≠ Final Score ≠ Rank`)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
              <input
                type="number"
                step="0.0001"
                placeholder="Final Score (Slot 1)"
                value={privScore1}
                onChange={(e) => setPrivScore1(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
              <input
                type="number"
                step="0.0001"
                placeholder="Final Score (Slot 2)"
                value={privScore2}
                onChange={(e) => setPrivScore2(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded p-2 text-white font-mono"
              />
              <input
                type="number"
                placeholder="Final Rank"
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
                    setGeneratedReportMd('Exported to artifacts/FINAL_COMPETITION_REPORT.md');
                  }
                }}
                className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold flex items-center justify-center gap-1.5"
              >
                <FileText className="w-4 h-4" /> Export Closure & Reproducibility Package
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
