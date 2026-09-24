import React, { useState } from 'react';
import {
  Send,
  CheckCircle2,
  AlertTriangle,
  FileArchive,
  Award,
  PlusCircle,
  ExternalLink,
  HelpCircle,
  X,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import { SubmissionEntry, TaskInfo } from '../types/experiment';
import { api } from '../services/api';

interface SubmissionsTabProps {
  submissions: SubmissionEntry[];
  taskInfo?: TaskInfo | null;
  onRefresh: () => Promise<void>;
  onSelectExperiment: (expId: string) => void;
}

export const SubmissionsTab: React.FC<SubmissionsTabProps> = ({
  submissions,
  taskInfo,
  onRefresh,
  onSelectExperiment,
}) => {
  const [selectedSubForScore, setSelectedSubForScore] = useState<SubmissionEntry | null>(null);
  const [submittingScore, setSubmittingScore] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  // Form State
  const [dynamicScores, setDynamicScores] = useState<Record<string, string>>({});
  const [macroA, setMacroA] = useState<string>('');
  const [macroB, setMacroB] = useState<string>('');
  const [macroCType, setMacroCType] = useState<string>('');
  const [macroCID, setMacroCID] = useState<string>('');
  const [splitType, setSplitType] = useState<string>('public');
  const [notes, setNotes] = useState<string>('');

  // Register New Submission State
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false);
  const [regExpId, setRegExpId] = useState('');
  const [regCandidateCsv, setRegCandidateCsv] = useState('');
  const [regBaselineCsv, setRegBaselineCsv] = useState('');
  const [regNotes, setRegNotes] = useState('');
  const [regMinDiffPct, setRegMinDiffPct] = useState('2.0');
  const [registering, setRegistering] = useState(false);
  const [regError, setRegError] = useState<string | null>(null);
  const [regSuccess, setRegSuccess] = useState<string | null>(null);

  const handleRegisterSubmission = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!regExpId.trim()) {
      setRegError('Please provide an Experiment ID.');
      return;
    }
    setRegistering(true);
    setRegError(null);
    setRegSuccess(null);
    try {
      const res = await api.registerSubmission({
        exp_id: regExpId.trim(),
        candidate_csv: regCandidateCsv.trim() || undefined,
        baseline_csv: regBaselineCsv.trim() || null,
        notes: regNotes.trim(),
        min_diff_pct: parseFloat(regMinDiffPct) || 2.0,
      });
      setRegSuccess(`Submission ${res.submission.id} registered and audit-verified!`);
      await onRefresh();
      setTimeout(() => {
        setIsRegisterModalOpen(false);
        setRegSuccess(null);
        setRegExpId('');
        setRegCandidateCsv('');
        setRegBaselineCsv('');
        setRegNotes('');
      }, 1500);
    } catch (err: any) {
      console.error(err);
      setRegError(err.message || 'Audit Gate failed to register submission.');
    } finally {
      setRegistering(false);
    }
  };

  const openScoreModal = (sub: SubmissionEntry) => {
    setSelectedSubForScore(sub);
    const dyn: Record<string, string> = {};
    if (taskInfo?.metrics && taskInfo.metrics.length > 0) {
      for (const m of taskInfo.metrics) {
        const val = sub.official_scores?.[m.id] ?? sub.official_scores?.[`${m.id}_macro`];
        dyn[m.id] = val !== undefined && val !== null ? String(val) : '';
      }
    } else {
      const primary =
        sub.official_scores?.primary_metric ??
        sub.official_scores?.score ??
        sub.official_scores?.subtask_a_macro;
      dyn['primary_metric'] = primary !== undefined && primary !== null ? String(primary) : '';
    }
    setDynamicScores(dyn);
    setMacroA(sub.official_scores?.subtask_a_macro !== undefined && sub.official_scores?.subtask_a_macro !== null ? String(sub.official_scores.subtask_a_macro) : '');
    setMacroB(sub.official_scores?.subtask_b_macro !== undefined && sub.official_scores?.subtask_b_macro !== null ? String(sub.official_scores.subtask_b_macro) : '');
    setMacroCType(sub.official_scores?.subtask_c_type_macro !== undefined && sub.official_scores?.subtask_c_type_macro !== null ? String(sub.official_scores.subtask_c_type_macro) : '');
    setMacroCID(sub.official_scores?.subtask_c_id_macro !== undefined && sub.official_scores?.subtask_c_id_macro !== null ? String(sub.official_scores.subtask_c_id_macro) : '');
    setSplitType(sub.official_scores?.split_type || 'public');
    setNotes(sub.notes || '');
    setFormError(null);
  };

  const closeScoreModal = () => {
    setSelectedSubForScore(null);
    setFormError(null);
  };

  const handleSubmitScore = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSubForScore) return;

    setSubmittingScore(true);
    setFormError(null);

    try {
      const scoresPayload: Record<string, any> = {
        split_type: splitType,
      };

      if (macroA !== '') scoresPayload.subtask_a_macro = parseFloat(macroA);
      if (macroB !== '') scoresPayload.subtask_b_macro = parseFloat(macroB);
      if (macroCType !== '') scoresPayload.subtask_c_type_macro = parseFloat(macroCType);
      if (macroCID !== '') scoresPayload.subtask_c_id_macro = parseFloat(macroCID);

      for (const [k, v] of Object.entries(dynamicScores)) {
        if (v !== '') {
          const num = parseFloat(v);
          if (!isNaN(num)) {
            scoresPayload[k] = num;
            if (k === 'subtask_a' && !scoresPayload.subtask_a_macro) {
              scoresPayload.subtask_a_macro = num;
            }
          }
        }
      }

      await api.recordOfficialScore(selectedSubForScore.id, scoresPayload, notes);
      await onRefresh();
      closeScoreModal();
    } catch (err: any) {
      console.error(err);
      setFormError(err.message || 'Failed to record official score.');
    } finally {
      setSubmittingScore(false);
    }
  };

  const renderScoreSet = (scores: Record<string, any> | undefined) => {
    if (!scores || Object.keys(scores).length === 0) return <span className="text-slate-500">-</span>;
    if (taskInfo?.metrics && taskInfo.metrics.length > 0) {
      const items = taskInfo.metrics.map((m) => {
        const val = scores[m.id] ?? scores[`${m.id}_macro`];
        return val !== undefined && val !== null ? Number(val).toFixed(4) : '-';
      });
      return (
        <div className="space-x-1.5 font-mono">
          {items.map((val, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 && <span className="text-slate-500">/</span>}
              <span className={idx === 0 ? "text-slate-200 font-semibold" : "text-slate-400"}>
                {val}
              </span>
            </React.Fragment>
          ))}
        </div>
      );
    }
    if (scores.subtask_a_macro !== undefined || scores.subtask_b_macro !== undefined) {
      return (
        <div className="space-x-1.5 font-mono">
          <span className="text-slate-300 font-semibold" title="Subtask A">
            {scores.subtask_a_macro !== undefined && scores.subtask_a_macro !== null ? Number(scores.subtask_a_macro).toFixed(4) : '-'}
          </span>
          <span className="text-slate-500">/</span>
          <span className="text-slate-400" title="Subtask B">
            {scores.subtask_b_macro !== undefined && scores.subtask_b_macro !== null ? Number(scores.subtask_b_macro).toFixed(4) : '-'}
          </span>
          <span className="text-slate-500">/</span>
          <span className="text-slate-400" title="Subtask C Type">
            {scores.subtask_c_type_macro !== undefined && scores.subtask_c_type_macro !== null ? Number(scores.subtask_c_type_macro).toFixed(4) : '-'}
          </span>
        </div>
      );
    }
    const numeric = Object.entries(scores).filter(([k, v]) => typeof v === 'number');
    if (numeric.length === 0) return <span className="text-slate-500">-</span>;
    return (
      <div className="space-x-1.5 font-mono">
        {numeric.slice(0, 3).map(([k, v], idx) => (
          <React.Fragment key={k}>
            {idx > 0 && <span className="text-slate-500">/</span>}
            <span className={idx === 0 ? "text-slate-200 font-semibold" : "text-slate-400"} title={k}>
              {Number(v).toFixed(4)}
            </span>
          </React.Fragment>
        ))}
      </div>
    );
  };

  const totalSubs = submissions.length;
  const scoredSubs = submissions.filter((s) => s.status === 'SCORED').length;
  const divergenceCount = submissions.filter((s) => s.divergence_warning).length;

  return (
    <div className="space-y-3">
      {/* Compact Stats Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-zinc-900/50 border border-zinc-800/80 px-3.5 py-2 rounded-lg text-xs">
        <div className="flex items-center gap-3 text-zinc-400 font-mono text-[11px]">
          <span>Registered: <strong className="text-zinc-100">{totalSubs}</strong></span>
          <span>•</span>
          <span>Scored: <strong className="text-zinc-100">{scoredSubs}</strong></span>
          <span>•</span>
          <span>Verified Packages: <strong className="text-zinc-100">{submissions.filter((s) => s.archive_path).length}</strong></span>
        </div>
        {divergenceCount > 0 ? (
          <div className="flex items-center gap-1.5 text-rose-400 text-xs font-medium">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{divergenceCount} Divergence Alert (LB drops below CV)</span>
          </div>
        ) : (
          <div className="text-zinc-500 font-mono text-[11px]">
            CV & LB aligned
          </div>
        )}
      </div>

      {/* Main Submissions Table */}
      <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              <Award className="w-4 h-4 text-zinc-400" />
              <span>
                {taskInfo?.platform ? `${taskInfo.platform.toUpperCase()} Submissions` : 'Submissions'}
              </span>
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Verified submission packages, cross-validation scores, and official leaderboard results.
            </p>
          </div>

          <button
            onClick={() => setIsRegisterModalOpen(true)}
            className="flex items-center gap-1.5 bg-zinc-100 hover:bg-white active:scale-95 text-zinc-950 px-3 py-1.5 rounded-lg text-xs font-semibold shadow-sm transition"
          >
            <PlusCircle className="w-3.5 h-3.5" />
            <span>Register Submission</span>
          </button>
        </div>

        {submissions.length === 0 ? (
          <div className="p-8 text-center space-y-3">
            <FileArchive className="w-10 h-10 text-slate-600 mx-auto" />
            <h3 className="text-sm font-semibold text-slate-300">No Submissions Registered Yet</h3>
            <p className="text-xs text-slate-400 max-w-lg mx-auto leading-relaxed">
              Submissions must pass the deterministic SWE-agent Audit Gate before being registered. Package a candidate via:
            </p>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300 max-w-xl mx-auto text-left">
              python tools/audit_gate.py --candidate artifacts/expXXX/oof_predictions.csv --baseline artifacts/exp001/oof_predictions.csv --output-zip submission.zip
              <br />
              python tools/submission_tool.py register --exp-id expXXX --candidate artifacts/expXXX/oof_predictions.csv
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                  <th className="py-3 px-3">Submission ID</th>
                  <th className="py-3 px-3">Exp ID</th>
                  <th className="py-3 px-3">Status</th>
                  <th className="py-3 px-3 text-right">Local CV ({taskInfo?.primary_metric || 'Metric'})</th>
                  <th className="py-3 px-3 text-right">Remote LB ({taskInfo?.primary_metric || 'Metric'})</th>
                  <th className="py-3 px-3 text-right">Delta (Δ LB - CV)</th>
                  <th className="py-3 px-3">Audit Receipt</th>
                  <th className="py-3 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {submissions.map((sub) => {
                  const hasScores = sub.status === 'SCORED' && sub.official_scores;
                  const deltaValues = sub.score_deltas
                    ? Object.values(sub.score_deltas).filter((v) => typeof v === 'number')
                    : [];
                  const mainDelta =
                    sub.score_deltas?.subtask_a_macro ??
                    sub.score_deltas?.primary_metric ??
                    (deltaValues.length > 0 ? (deltaValues[0] as number) : undefined);

                  return (
                    <tr key={sub.id} className="hover:bg-slate-800/40 transition">
                      <td className="py-3.5 px-3 font-semibold text-slate-200">
                        <div className="flex items-center gap-1.5">
                          {sub.divergence_warning && (
                            <span title="Divergence Alert: LB drop > 0.05">
                              <AlertTriangle className="w-3.5 h-3.5 text-rose-400 flex-shrink-0" />
                            </span>
                          )}
                          <span>{sub.id}</span>
                        </div>
                        <div className="text-[10px] text-slate-500 font-sans mt-0.5">
                          {sub.created_at}
                        </div>
                      </td>

                      <td className="py-3.5 px-3">
                        <button
                          onClick={() => onSelectExperiment(sub.exp_id)}
                          className="text-indigo-400 hover:text-indigo-300 hover:underline font-mono"
                        >
                          {sub.exp_id}
                        </button>
                      </td>

                      <td className="py-3.5 px-3">
                        {sub.status === 'SCORED' ? (
                          <span className="bg-indigo-900/60 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded text-[10px] font-semibold">
                            SCORED ({sub.official_scores?.split_type || 'public'})
                          </span>
                        ) : (
                          <span className="bg-emerald-900/60 text-emerald-300 border border-emerald-700/50 px-2 py-0.5 rounded text-[10px] font-semibold">
                            LOCAL_VERIFIED
                          </span>
                        )}
                      </td>

                      {/* Local CV */}
                      <td className="py-3.5 px-3 text-right">
                        {renderScoreSet(sub.local_cv_scores)}
                      </td>

                      {/* Official Remote LB */}
                      <td className="py-3.5 px-3 text-right">
                        {hasScores ? (
                          renderScoreSet(sub.official_scores)
                        ) : (
                          <span className="text-slate-500 font-sans italic text-[11px]">
                            Pending Remote
                          </span>
                        )}
                      </td>

                      {/* Deltas */}
                      <td className="py-3.5 px-3 text-right">
                        {mainDelta !== undefined && mainDelta !== null ? (
                          <div
                            className={`flex items-center justify-end gap-1 font-bold ${
                              mainDelta >= 0 ? 'text-emerald-400' : mainDelta < -0.05 ? 'text-rose-400' : 'text-amber-400'
                            }`}
                          >
                            {mainDelta >= 0 ? (
                              <TrendingUp className="w-3.5 h-3.5" />
                            ) : (
                              <TrendingDown className="w-3.5 h-3.5" />
                            )}
                            <span>
                              {mainDelta > 0 ? `+${mainDelta.toFixed(4)}` : mainDelta.toFixed(4)}
                            </span>
                          </div>
                        ) : (
                          <span className="text-slate-600">-</span>
                        )}
                      </td>

                      {/* Audit Receipt */}
                      <td className="py-3.5 px-3 text-slate-400 font-sans text-[11px]">
                        <span className="font-mono text-emerald-400">✓ Verified</span>
                        {sub.notes && (
                          <div className="text-slate-500 text-[10px] truncate max-w-[180px]">
                            {sub.notes}
                          </div>
                        )}
                      </td>

                      {/* Action */}
                      <td className="py-3.5 px-3 text-right">
                        <button
                          onClick={() => openScoreModal(sub)}
                          className="bg-indigo-600/80 hover:bg-indigo-600 active:scale-95 text-white px-2.5 py-1 rounded text-[11px] font-sans font-medium transition shadow-sm"
                        >
                          {sub.status === 'SCORED' ? 'Update Score' : 'Record LB Score'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Record Score Modal */}
      {selectedSubForScore && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 w-full max-w-md shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white">Record Remote / Leaderboard Score</h3>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  {selectedSubForScore.id}
                </p>
              </div>
              <button
                onClick={closeScoreModal}
                className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {formError && (
              <div className="bg-rose-950/40 border border-rose-800/60 p-3 rounded-lg text-xs text-rose-300">
                {formError}
              </div>
            )}

            <form onSubmit={handleSubmitScore} className="space-y-3.5 text-xs">
              {taskInfo?.metrics && taskInfo.metrics.length > 0 ? (
                <div className="grid grid-cols-2 gap-3">
                  {taskInfo.metrics.map((m) => (
                    <div key={m.id}>
                      <label className="block text-slate-400 font-medium mb-1 truncate" title={m.name}>
                        {m.name} ({m.metric})
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        placeholder="e.g. 0.6850"
                        value={dynamicScores[m.id] ?? ''}
                        onChange={(e) =>
                          setDynamicScores((prev) => ({ ...prev, [m.id]: e.target.value }))
                        }
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 font-medium mb-1">Subtask A Macro-F1</label>
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="e.g. 0.6850"
                      value={macroA}
                      onChange={(e) => setMacroA(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-400 font-medium mb-1">Subtask B Macro-F1</label>
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="e.g. 0.5340"
                      value={macroB}
                      onChange={(e) => setMacroB(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-400 font-medium mb-1">Subtask C Type Macro-F1</label>
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="e.g. 0.7200"
                      value={macroCType}
                      onChange={(e) => setMacroCType(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-400 font-medium mb-1">Subtask C Identities Macro-F1</label>
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="Optional"
                      value={macroCID}
                      onChange={(e) => setMacroCID(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-slate-400 font-medium mb-1">Split Type</label>
                <select
                  value={splitType}
                  onChange={(e) => setSplitType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="public">Public Leaderboard</option>
                  <option value="private">Private Test Leaderboard</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1">Evaluation Notes / Run ID</label>
                <input
                  type="text"
                  placeholder="e.g. Codabench submission #42981"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={closeScoreModal}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submittingScore}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white font-medium rounded-lg transition shadow-lg shadow-indigo-600/30 disabled:opacity-50"
                >
                  {submittingScore ? 'Saving...' : 'Save Official Score'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Register New Submission Modal */}
      {isRegisterModalOpen && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 w-full max-w-lg shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Award className="w-4 h-4 text-indigo-400" />
                  <span>Register & Audit New Submission</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Package candidate predictions via the SWE Audit Gate (verifies MMES &ge; 2.0% and formats archive).
                </p>
              </div>
              <button
                onClick={() => {
                  setIsRegisterModalOpen(false);
                  setRegError(null);
                  setRegSuccess(null);
                }}
                className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {regError && (
              <div className="bg-rose-950/40 border border-rose-800/60 p-3 rounded-lg text-xs text-rose-300">
                {regError}
              </div>
            )}

            {regSuccess && (
              <div className="bg-emerald-950/40 border border-emerald-800/60 p-3 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>{regSuccess}</span>
              </div>
            )}

            <form onSubmit={handleRegisterSubmission} className="space-y-3.5 text-xs">
              <div>
                <label className="block text-slate-300 font-semibold mb-1">
                  Experiment ID <span className="text-rose-400">*</span>
                </label>
                <input
                  type="text"
                  placeholder="e.g. exp001_baseline_tfidf"
                  value={regExpId}
                  onChange={(e) => {
                    const id = e.target.value;
                    setRegExpId(id);
                    if (!regCandidateCsv || regCandidateCsv.startsWith('artifacts/')) {
                      setRegCandidateCsv(id ? `artifacts/${id}/oof_predictions.csv` : '');
                    }
                  }}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-300 font-semibold mb-1">
                  Candidate Predictions CSV Path
                </label>
                <input
                  type="text"
                  placeholder="artifacts/<exp_id>/oof_predictions.csv"
                  value={regCandidateCsv}
                  onChange={(e) => setRegCandidateCsv(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                />
                <p className="text-[10px] text-slate-500 mt-0.5">
                  Leave blank to auto-detect <code className="text-slate-400">artifacts/{regExpId || '&lt;exp_id&gt;'}/oof_predictions.csv</code>
                </p>
              </div>

              <div>
                <label className="block text-slate-300 font-semibold mb-1">
                  Parent Baseline CSV Path (for MMES delta check)
                </label>
                <input
                  type="text"
                  placeholder="e.g. artifacts/exp000_template/oof_predictions.csv (leave blank if initial baseline)"
                  value={regBaselineCsv}
                  onChange={(e) => setRegBaselineCsv(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                />
                <p className="text-[10px] text-slate-500 mt-0.5">
                  Optional. If omitted, candidate is registered as the initial baseline package.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 font-semibold mb-1">
                    MMES Threshold (%)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    value={regMinDiffPct}
                    onChange={(e) => setRegMinDiffPct(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                  />
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    Requires &ge; 2.0% row variance vs baseline.
                  </p>
                </div>

                <div>
                  <label className="block text-slate-300 font-semibold mb-1">
                    Target Platform
                  </label>
                  <div className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-300 font-mono">
                    {taskInfo?.platform ? taskInfo.platform.toUpperCase() : 'CODABENCH / KAGGLE'}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    From Task Config.
                  </p>
                </div>
              </div>

              <div>
                <label className="block text-slate-300 font-semibold mb-1">
                  Submission Notes / Rationale
                </label>
                <input
                  type="text"
                  placeholder="e.g. LightGBM baseline with feature selection"
                  value={regNotes}
                  onChange={(e) => setRegNotes(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsRegisterModalOpen(false)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={registering}
                  className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white font-semibold rounded-lg transition shadow-lg shadow-indigo-600/30 disabled:opacity-50"
                >
                  {registering ? 'Auditing & Registering...' : 'Verify Audit & Register'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
