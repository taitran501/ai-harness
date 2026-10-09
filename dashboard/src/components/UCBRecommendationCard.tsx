import React from 'react';
import { Compass, Sparkles, ShieldCheck } from 'lucide-react';
import { UCBRecommendation } from '../types/experiment';
import { useLanguage } from '../constants/i18n';

interface UCBRecommendationCardProps {
  recommendation: UCBRecommendation | null;
}

export const UCBRecommendationCard: React.FC<UCBRecommendationCardProps> = ({ recommendation }) => {
  const { t } = useLanguage();
  if (!recommendation) return null;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
      <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs uppercase tracking-wider">
        <Compass className="w-4 h-4 text-indigo-400" />
        {t('ucb_card_title', 'UCB Tree Recommendation')}
      </div>

      <div className="space-y-1">
        <h3 className="text-sm font-bold text-white">
          {recommendation.recommendation || t('ucb_rec_active', 'Recommendation Active')}
        </h3>
        {recommendation.ucb_score && (
          <div className="text-xs text-slate-400">
            {t('ucb_score_label', 'UCB Potential Score:')} <strong className="font-mono text-indigo-300">{recommendation.ucb_score.toFixed(4)}</strong>
          </div>
        )}
      </div>

      <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 text-xs space-y-2 font-mono">
        <div className="flex justify-between">
          <span className="text-slate-400 font-sans">{t('ucb_rec_parent', 'Recommended Parent:')}</span>
          <span className="text-indigo-300 font-bold">{recommendation.parent_id || 'ROOT'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400 font-sans">{t('ucb_branch_label', 'Branch:')}</span>
          <span className="text-purple-300 font-bold uppercase">{recommendation.branch || recommendation.recommended_branch || 'baseline'}</span>
        </div>
      </div>

      <div className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 italic">
        "{recommendation.guideline}"
      </div>
    </div>
  );
};
