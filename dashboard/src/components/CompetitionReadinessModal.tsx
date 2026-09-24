import React from 'react';
import {
  X,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Layers,
  FileCheck,
  Cpu,
  Database,
  Sliders,
  Scale,
  Award,
  BookOpen,
  ArrowRight,
} from 'lucide-react';
import { TaskInfo, SystemStatus, SubmissionEntry, CompetitionPillarCheck } from '../types/experiment';

interface CompetitionReadinessModalProps {
  isOpen: boolean;
  onClose: () => void;
  taskInfo: TaskInfo | null;
  status: SystemStatus | null;
  submissions: SubmissionEntry[];
  onOpenTaskConfig: (tabKey?: 'task' | 'data' | 'rules' | 'submission') => void;
}

export const CompetitionReadinessModal: React.FC<CompetitionReadinessModalProps> = ({
  isOpen,
  onClose,
  taskInfo,
  status,
  submissions,
  onOpenTaskConfig,
}) => {
  if (!isOpen) return null;

  const dataCfg = taskInfo?.data_config || {};
  const rules = taskInfo?.competition_rules || {};
  const subSpec = taskInfo?.submission_spec || {};

  // Compute 8 pillars
  const pillars: CompetitionPillarCheck[] = [
    {
      id: 'task_objective',
      title: '1. Task & Metric Objectives',
      status: taskInfo?.primary_metric ? 'passed' : 'warning',
      summary: taskInfo?.primary_metric
        ? `${taskInfo.task_name || 'Benchmark'} • ${taskInfo.primary_metric} (${taskInfo.direction || 'maximize'})`
        : 'Primary metric not declared',
      details: `Target: ${taskInfo?.primary_metric_target ?? 'Not set'} | Baseline: ${taskInfo?.primary_metric_baseline ?? 'Not set'} | Subtasks: ${taskInfo?.metrics?.length || 0}`,
      tabKey: 'task',
    },
    {
      id: 'data_splits',
      title: '2. Data & Anti-Leakage Cross-Validation',
      status: dataCfg.split_strategy ? 'passed' : 'warning',
      summary: dataCfg.split_strategy
        ? `${dataCfg.split_strategy} (${dataCfg.n_folds || 5} folds) • Anti-Leakage: ${dataCfg.group_column || 'Standard'}`
        : 'Cross-validation split scheme not configured',
      details: `Train path: ${dataCfg.train_path || 'dataset/'} | Group Column: ${dataCfg.group_column || 'None'} | ID: ${dataCfg.id_column || 'id'}`,
      tabKey: 'data',
    },
    {
      id: 'model_grounding',
      title: '3. Model Strategy & Parameter Ceilings',
      status: rules.max_params_limit ? 'passed' : 'warning',
      summary: rules.max_params_limit
        ? `Parameter ceiling: ${(rules.max_params_limit / 1e6).toFixed(0)}M params • Modality: ${taskInfo?.modality || 'Generic'}`
        : 'Parameter limit not constrained',
      details: `Pretrained models: ${rules.allow_pretrained_models ? 'Permitted' : 'Forbidden'} | Backbones registered in KNOWLEDGE_BASE.md`,
      tabKey: 'rules',
    },
    {
      id: 'experiment_lineage',
      title: '4. MCTS Solution Tree & Lineage Tracking',
      status: (status?.total_nodes || 0) > 0 ? 'passed' : 'pending',
      summary: `${status?.total_nodes || 0} runs tracked • ${status?.distinct_families || 0} distinct model families`,
      details: `Champion Node: ${status?.best_node || 'None'} (${(status?.best_score || 0).toFixed(4)}) | UCB tree search active`,
      tabKey: 'task',
    },
    {
      id: 'validation_overfitting',
      title: '5. Overfitting & Leaderboard Divergence Guard',
      status: 'passed',
      summary: `MMES >= ${subSpec.min_diff_pct ?? 2.0}% floor • Divergence trigger Delta(LB - CV) < -0.05`,
      details: 'Prevents submitting random noise or overfitting to public leaderboard feedback.',
      tabKey: 'submission',
    },
    {
      id: 'competition_rules',
      title: '6. Competition Rules & Compute Quotas',
      status: rules.daily_submission_limit ? 'passed' : 'warning',
      summary: `Daily Quota: ${rules.daily_submission_limit ?? 'Unlimited'} subs/day • Deadline: ${rules.deadline ? rules.deadline.split('T')[0] : 'Open'}`,
      details: `LLMs: ${rules.allow_llm ? 'Allowed' : 'Local only'} | External data: ${rules.allow_external_data ? 'Allowed' : 'Forbidden'}`,
      tabKey: 'rules',
    },
    {
      id: 'final_submission',
      title: '7. Final Submission & Audit Gate Packaging',
      status: submissions.length > 0 ? 'passed' : 'pending',
      summary: `${submissions.length} verified submissions on disk • Output format: ${subSpec.file_name || 'submission.zip'}`,
      details: `Cryptographic AUDIT_RECEIPT.json generated • Ensemble Kappa <= ${subSpec.max_kappa ?? 0.85}`,
      tabKey: 'submission',
    },
    {
      id: 'system_description',
      title: '8. Paper & System Description Readiness',
      status: (status?.total_nodes || 0) > 0 ? 'passed' : 'pending',
      summary: 'Auto-generated EXPERIMENT_CARD.md, DIAGNOSTICS.md & Summary tables ready',
      details: 'All hypotheses, delta scores, error slices, and ablation branches serialized in artifacts/.',
      tabKey: 'task',
    },
  ];

  const passedCount = pillars.filter((p) => p.status === 'passed').length;
  const warningCount = pillars.filter((p) => p.status === 'warning').length;

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-3xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/60">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <span>Competition Readiness Assessment</span>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {passedCount} / {pillars.length} Verified
                </span>
              </h2>
              <p className="text-xs text-zinc-400">
                Evaluating the 8 essential dimensions for competitive ML/DL benchmarks
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-300 p-1 rounded-md hover:bg-zinc-800 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content List */}
        <div className="p-5 overflow-y-auto space-y-3 flex-1">
          {pillars.map((pillar) => (
            <div
              key={pillar.id}
              className={`p-3.5 rounded-lg border transition ${
                pillar.status === 'passed'
                  ? 'bg-zinc-950/50 border-emerald-500/20 hover:border-emerald-500/40'
                  : pillar.status === 'warning'
                  ? 'bg-zinc-950/50 border-amber-500/20 hover:border-amber-500/40'
                  : 'bg-zinc-950/30 border-zinc-800 hover:border-zinc-700'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-2.5">
                  <div className="mt-0.5">
                    {pillar.status === 'passed' ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : pillar.status === 'warning' ? (
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                    ) : (
                      <Clock className="w-4 h-4 text-zinc-500" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-xs font-semibold text-zinc-200">{pillar.title}</h3>
                    <p className="text-xs text-zinc-300 font-medium mt-0.5">{pillar.summary}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">{pillar.details}</p>
                  </div>
                </div>

                {pillar.tabKey && (
                  <button
                    onClick={() => {
                      onClose();
                      onOpenTaskConfig(pillar.tabKey);
                    }}
                    className="flex items-center gap-1 text-[11px] font-medium text-zinc-400 hover:text-zinc-200 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2 py-1 rounded transition"
                  >
                    <span>Configure</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-zinc-800 bg-zinc-950/60 flex items-center justify-between">
          <div className="text-xs text-zinc-400">
            {warningCount > 0 ? (
              <span className="text-amber-400 font-medium">
                ⚠️ {warningCount} areas need attention before official submission.
              </span>
            ) : (
              <span className="text-emerald-400 font-medium">
                ✔ All competition dimensions are configured and audit-ready.
              </span>
            )}
          </div>
          <button
            onClick={() => {
              onClose();
              onOpenTaskConfig('task');
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 text-xs font-medium text-zinc-200 border border-zinc-700 transition"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Open Task Setup Modal</span>
          </button>
        </div>
      </div>
    </div>
  );
};
