import React, { useState, useMemo } from 'react';
import { Crown, Search, Filter, Cpu, Layers, AlertTriangle, ArrowUpDown, ChevronDown, ChevronUp } from 'lucide-react';
import { ExperimentSummary, SubmissionEntry, TaskInfo } from '../types/experiment';

interface LeaderboardTableProps {
  experiments: ExperimentSummary[];
  submissions?: SubmissionEntry[];
  taskInfo?: TaskInfo | null;
  onSelectNode: (nodeId: string) => void;
}

export const LeaderboardTable: React.FC<LeaderboardTableProps> = ({
  experiments,
  submissions = [],
  taskInfo,
  onSelectNode,
}) => {
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<'score' | 'timestamp' | 'params'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>(taskInfo?.direction === 'minimize' ? 'asc' : 'desc');
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});

  const primaryMetricLabel = `${taskInfo?.primary_metric || 'Primary Metric (CV)'} (${taskInfo?.direction || 'maximize'})`;

  // Map experiment ID to latest official submission score if any
  const subScoreMap = useMemo(() => {
    const map: Record<string, { primary?: number }> = {};
    for (const sub of submissions) {
      if (sub.status === 'SCORED' && sub.official_scores) {
        const scores = sub.official_scores;
        const primaryScore = [
          scores.primary_metric,
          taskInfo?.primary_metric ? scores[taskInfo.primary_metric] : undefined,
          scores.primary,
          scores.score,
          scores.overall_composite,
          ...Object.values(scores),
        ].find((value) => typeof value === 'number');
        map[sub.exp_id] = {
          primary: typeof primaryScore === 'number' ? primaryScore : undefined,
        };
      }
    }
    return map;
  }, [submissions, taskInfo?.primary_metric]);

  const toggleExpand = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedRows((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filtered = useMemo(() => {
    return experiments
      .filter((exp) => {
        const matchesSearch =
          exp.id.toLowerCase().includes(search.toLowerCase()) ||
          (exp.hypothesis && exp.hypothesis.toLowerCase().includes(search.toLowerCase())) ||
          exp.branch.toLowerCase().includes(search.toLowerCase());
        return matchesSearch;
      })
      .sort((a, b) => {
        if (sortBy === 'score' && a.eligible !== b.eligible) return a.eligible ? -1 : 1;
        let valA = a.score;
        let valB = b.score;
        if (sortBy === 'timestamp') {
          valA = a.timestamp;
          valB = b.timestamp;
        } else if (sortBy === 'params') {
          valA = a.param_count || 0;
          valB = b.param_count || 0;
        }
        return sortOrder === 'desc' ? valB - valA : valA - valB;
      });
  }, [experiments, search, sortBy, sortOrder]);

  const toggleSort = (field: 'score' | 'timestamp' | 'params') => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  return (
    <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-xl space-y-3">
      {/* Controls Bar */}
      <div className="flex items-center justify-between gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
          <input
            type="text"
            placeholder="Search experiments, branches, hypotheses..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-zinc-700 transition"
          />
        </div>
        <div className="text-xs text-zinc-500 font-mono">
          {filtered.length} of {experiments.length} runs
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-zinc-800 text-zinc-400 uppercase tracking-wider text-[10px]">
              <th className="py-2.5 px-3">Experiment</th>
              <th className="py-2.5 px-3">Branch</th>
              <th className="py-2.5 px-3">Parent</th>
              <th
                onClick={() => toggleSort('score')}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-zinc-200 transition"
              >
                <div className="flex items-center justify-end gap-1 font-semibold text-zinc-300">
                  {primaryMetricLabel}
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-2.5 px-3 text-right text-zinc-400">
                Remote LB
              </th>
              <th
                onClick={() => toggleSort('params')}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-zinc-200 transition"
              >
                <div className="flex items-center justify-end gap-1">
                  Params
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-2.5 px-3 min-w-[280px]">Hypothesis & Atomic Intervention</th>
              <th className="py-2.5 px-3 text-center">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60 font-mono">
            {filtered.map((exp) => {
              const isExpanded = !!expandedRows[exp.id];
              const text = exp.hypothesis || exp.intervention || '-';
              const official = subScoreMap[exp.id];

              return (
                <tr
                  key={exp.id}
                  onClick={() => onSelectNode(exp.id)}
                  className="hover:bg-zinc-900/60 cursor-pointer transition group"
                >
                  <td className="py-2.5 px-3 font-semibold text-zinc-100 flex items-center gap-1.5">
                    {exp.is_best && <Crown className="w-3.5 h-3.5 text-amber-400 fill-amber-400 flex-shrink-0" />}
                    <span className="group-hover:text-zinc-200 transition">{exp.id}</span>
                  </td>
                  <td className="py-2.5 px-3 text-zinc-400">
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] bg-zinc-900 text-zinc-300 px-1.5 py-0.5 rounded border border-zinc-800">
                        {exp.branch}
                      </span>
                      {exp.approach_type === 'llm' && (
                        <span className="text-[9px] bg-amber-500/15 text-amber-300 border border-amber-500/30 px-1 py-0.2 rounded font-mono font-semibold">
                          LLM
                        </span>
                      )}
                      {exp.approach_type === 'dl' && (
                        <span className="text-[9px] bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 px-1 py-0.2 rounded font-mono font-semibold">
                          DL
                        </span>
                      )}
                      {exp.approach_type === 'ml' && (
                        <span className="text-[9px] bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 px-1 py-0.2 rounded font-mono font-semibold">
                          ML
                        </span>
                      )}
                      {exp.approach_type === 'hybrid' && (
                        <span className="text-[9px] bg-purple-500/15 text-purple-300 border border-purple-500/30 px-1 py-0.2 rounded font-mono font-semibold">
                          HYBRID
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-2.5 px-3 text-zinc-500">{exp.parent_id}</td>
                  <td
                    className={`py-2.5 px-3 text-right font-bold ${
                      exp.is_best ? 'text-amber-400 text-xs' : 'text-emerald-400'
                    }`}
                  >
                    {exp.score.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right text-zinc-300 font-bold">
                    {official?.primary !== undefined ? `${official.primary.toFixed(4)}` : '-'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-zinc-400 font-mono">
                    {exp.param_size ? (
                      <span className="font-semibold text-amber-300">{exp.param_size}</span>
                    ) : exp.param_count ? (
                      exp.param_count >= 1e9
                        ? `${(exp.param_count / 1e9).toFixed(1)}B`
                        : `${(exp.param_count / 1e6).toFixed(1)}M`
                    ) : (
                      '-'
                    )}
                  </td>
                  <td className="py-3 px-3 text-slate-300 font-sans text-xs">
                    <div className="flex items-start justify-between gap-2">
                      <p className={`leading-relaxed ${isExpanded ? '' : 'line-clamp-2'}`}>
                        {text}
                      </p>
                      {text.length > 90 && (
                        <button
                          onClick={(e) => toggleExpand(exp.id, e)}
                          className="text-slate-400 hover:text-indigo-300 flex-shrink-0 p-0.5"
                          title={isExpanded ? 'Collapse' : 'Expand full hypothesis'}
                        >
                          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                        </button>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-3 text-center">
                    {exp.is_buggy ? (
                      <span className="text-rose-400 flex items-center justify-center gap-1 text-[11px]">
                        <AlertTriangle className="w-3.5 h-3.5" /> ERR
                      </span>
                    ) : (
                      <span className={exp.eligible ? "text-emerald-400 text-[11px]" : "text-amber-400 text-[11px]"}>{exp.eligible ? "Verified · eligible" : exp.status === "LEGACY_UNVERIFIED" ? "LEGACY_UNVERIFIED" : "Ineligible"}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
