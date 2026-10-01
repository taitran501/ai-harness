import React from 'react';
import {
  X,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Sliders,
  ArrowRight,
} from 'lucide-react';
import { TaskInfo, SystemStatus, SubmissionEntry } from '../types/experiment';

interface CompetitionReadinessModalProps {
  isOpen: boolean;
  onClose: () => void;
  taskInfo: TaskInfo | null;
  status: SystemStatus | null;
  submissions: SubmissionEntry[];
  onOpenTaskConfig: (tabKey?: 'task' | 'data' | 'cv' | 'rules' | 'submission') => void;
  onOpenWorkspace?: () => void;
}

interface PillarDisplayItem {
  id: string;
  title: string;
  status: 'passed' | 'warning' | 'pending';
  summary: string;
  attributes: Array<{ label: string; value: string }>;
  tabKey?: 'task' | 'data' | 'cv' | 'rules' | 'submission';
}

export const CompetitionReadinessModal: React.FC<CompetitionReadinessModalProps> = ({
  isOpen,
  onClose,
  taskInfo,
  status,
  submissions,
  onOpenTaskConfig,
  onOpenWorkspace,
}) => {
  if (!isOpen) return null;

  const contractNeedsReview = ['needs_review', 'unverified', 'draft'].includes(
    String(taskInfo?.contract_status || '').toLowerCase(),
  );
  if (contractNeedsReview) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
        <section className="w-full max-w-xl rounded-xl border border-amber-700/50 bg-zinc-900 p-5 shadow-2xl">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-wider text-amber-300">Contract not verified</p>
              <h2 className="mt-1 text-base font-semibold text-white">Readiness cannot be assessed yet</h2>
            </div>
            <button onClick={onClose} className="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-white" aria-label="Close readiness details"><X className="h-4 w-4" /></button>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-zinc-300">
            The task config is an onboarding draft. The dashboard will not label guessed settings as verified. Ask your IDE agent to inspect official instructions and record the source URLs before confirming task setup.
          </p>
          <div className="mt-4 flex justify-end gap-2">
            <button onClick={() => { onClose(); onOpenTaskConfig('task'); }} className="rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800">Review task config</button>
            {onOpenWorkspace && <button onClick={() => { onClose(); onOpenWorkspace(); }} className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500">Open guided workflow</button>}
          </div>
        </section>
      </div>
    );
  }

  const dataCfg = taskInfo?.data_config || {};
  const rules = taskInfo?.competition_rules || {};
  const subSpec = taskInfo?.submission_spec || {};
  const isCustomTaskPlugin = Boolean(taskInfo?.task_plugin?.id && taskInfo.task_plugin.id !== 'default');
  const pluginCanSubmit = Boolean(
    taskInfo?.task_plugin?.capabilities?.submission_builder &&
    taskInfo?.task_plugin?.capabilities?.submission_validator,
  );

  const pillars: PillarDisplayItem[] = [
    {
      id: 'task_objective',
      title: '1. Task & Metric Objectives',
      status: taskInfo?.primary_metric ? 'passed' : 'warning',
      summary: taskInfo?.primary_metric
        ? `${taskInfo.task_name || 'Benchmark'} (${taskInfo.primary_metric}, ${taskInfo.direction || 'maximize'})`
        : 'Primary metric not declared',
      attributes: [
        { label: 'Target Score', value: String(taskInfo?.primary_metric_target ?? 'Not set') },
        { label: 'Baseline Score', value: String(taskInfo?.primary_metric_baseline ?? 'Not set') },
        { label: 'Subtasks', value: String(taskInfo?.metrics?.length || 0) },
      ],
      tabKey: 'task',
    },
    {
      id: 'data_splits',
      title: '2. Dataset Schema & Preprocessing Strategy',
      status: dataCfg.train_path ? 'passed' : 'warning',
      summary: `Train: ${dataCfg.train_path || 'dataset/'} — Declared inputs and targets`,
      attributes: [
        { label: 'ID Column', value: dataCfg.id_column || 'id' },
        { label: 'Target Columns', value: dataCfg.target_column || 'target' },
        { label: 'Validation Split', value: dataCfg.n_folds
          ? `${dataCfg.split_strategy || 'Not declared'} (${dataCfg.n_folds} folds)`
          : `${dataCfg.split_strategy || 'Not declared'} (fold count not set)` },
      ],
      tabKey: 'data',
    },
    {
      id: 'model_grounding',
      title: '3. Model Strategy & Parameter Ceilings',
      status: rules.max_params_limit ? 'passed' : 'warning',
      summary: rules.max_params_limit
        ? `Parameter ceiling: ${(rules.max_params_limit / 1e6).toFixed(0)}M parameters`
        : 'Parameter limit not constrained',
      attributes: [
        { label: 'Modality', value: taskInfo?.modality || 'Generic' },
        { label: 'Pretrained Models', value: rules.allow_pretrained_models ? 'Permitted' : 'Forbidden' },
      ],
      tabKey: 'rules',
    },
    {
      id: 'experiment_lineage',
      title: '4. Solution Tree & Lineage Tracking',
      status: (status?.total_nodes || 0) > 0 ? 'passed' : 'pending',
      summary: `${status?.total_nodes || 0} runs tracked across ${status?.distinct_families || 0} model families`,
      attributes: [
        { label: 'Best Node', value: status?.best_node || 'None' },
        { label: 'Best Score', value: (status?.best_score || 0).toFixed(4) },
      ],
      tabKey: 'task',
    },
    {
      id: 'validation_overfitting',
      title: '5. Evaluation & Submission Checks',
      status: isCustomTaskPlugin && !pluginCanSubmit ? 'warning' : 'pending',
      summary: isCustomTaskPlugin
        ? pluginCanSubmit
          ? `Submission checks are delegated to task plugin ${taskInfo?.task_plugin?.id}.`
          : `Task plugin ${taskInfo?.task_plugin?.id} has no complete submission validator and builder.`
        : 'Record external scores separately from local validation metrics.',
      attributes: isCustomTaskPlugin
        ? [{ label: 'Submission Hooks', value: pluginCanSubmit ? 'Builder and validator available' : 'Incomplete' }]
        : [
            { label: 'Minimum Prediction Delta', value: subSpec.min_diff_pct !== undefined ? `${subSpec.min_diff_pct}%` : 'Not configured' },
            { label: 'External Score', value: 'Recorded separately from local validation' },
          ],
      tabKey: 'submission',
    },
    {
      id: 'competition_rules',
      title: '6. Competition Rules & Compute Quotas',
      status: rules.daily_submission_limit ? 'passed' : 'warning',
      summary: `Daily quota: ${rules.daily_submission_limit ?? 'Unlimited'} submissions per day`,
      attributes: [
        { label: 'LLMs Policy', value: rules.allow_llm ? 'Allowed' : 'Local only' },
        { label: 'External Data', value: rules.allow_external_data ? 'Allowed' : 'Forbidden' },
        { label: 'Deadline', value: rules.deadline ? rules.deadline.split('T')[0] : 'Open' },
      ],
      tabKey: 'rules',
    },
    {
      id: 'final_submission',
      title: '7. Submission Packaging',
      status: submissions.length > 0 ? 'passed' : isCustomTaskPlugin && !pluginCanSubmit ? 'warning' : 'pending',
      summary: isCustomTaskPlugin && !pluginCanSubmit
        ? 'Packaging is blocked until the task plugin implements both submission hooks.'
        : `${submissions.length} verified packages recorded (${subSpec.file_name || 'task-declared format'})`,
      attributes: isCustomTaskPlugin
        ? [
            { label: 'Task Plugin', value: taskInfo?.task_plugin?.id || 'Not selected' },
            { label: 'Submission Hooks', value: pluginCanSubmit ? 'Builder and validator available' : 'Incomplete' },
          ]
        : [
            { label: 'Verification Receipt', value: 'Generated by configured checks' },
            { label: 'Ensemble Diversity', value: subSpec.max_kappa !== undefined ? `Kappa <= ${subSpec.max_kappa}` : 'Not configured' },
          ],
      tabKey: 'submission',
    },
    {
      id: 'system_description',
      title: '8. Paper & System Description Readiness',
      status: (status?.total_nodes || 0) > 0 ? 'passed' : 'pending',
      summary: 'EXPERIMENT_CARD.md and summary tables generated per run',
      attributes: [
        { label: 'Artifact Storage', value: `${taskInfo?.artifacts_dir || 'artifacts'}/<exp_id>/` },
        { label: 'Lineage Ledger', value: '.ml-harness/EXPERIMENTS.md' },
      ],
      tabKey: 'task',
    },
  ];

  const passedCount = pillars.filter((p) => p.status === 'passed').length;
  const warningCount = pillars.filter((p) => p.status === 'warning').length;

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-3xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/60">
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-sm font-semibold text-zinc-100">
                Competition Readiness Assessment
              </h2>
              <span className="text-[11px] font-mono text-emerald-400">
                ({passedCount} of {pillars.length} verified)
              </span>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Evaluating the 8 essential dimensions for competitive ML/DL benchmarks.
            </p>
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
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 flex-1">
                  <div className="mt-0.5">
                    {pillar.status === 'passed' ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : pillar.status === 'warning' ? (
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                    ) : (
                      <Clock className="w-4 h-4 text-zinc-500" />
                    )}
                  </div>
                  <div className="flex-1 space-y-1.5">
                    <h3 className="text-xs font-semibold text-zinc-200">{pillar.title}</h3>
                    <p className="text-xs text-zinc-300">{pillar.summary}</p>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1">
                      {pillar.attributes.map((attr) => (
                        <div
                          key={attr.label}
                          className="bg-zinc-900/80 border border-zinc-800/80 rounded px-2.5 py-1 text-[11px]"
                        >
                          <span className="text-zinc-500 block text-[10px]">{attr.label}</span>
                          <span className="text-zinc-300 font-mono font-medium">{attr.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {pillar.tabKey && (
                  <button
                    onClick={() => {
                      onClose();
                      onOpenTaskConfig(pillar.tabKey);
                    }}
                    className="flex items-center gap-1 text-[11px] font-medium text-zinc-400 hover:text-zinc-200 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2.5 py-1 rounded transition shrink-0"
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
                {warningCount} areas need attention before official submission.
              </span>
            ) : (
              <span className="text-emerald-400 font-medium">
                All competition dimensions are configured and audit-ready.
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
