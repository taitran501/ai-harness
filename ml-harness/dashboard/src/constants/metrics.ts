export interface MetricDefinition {
  id: string;
  name: string;
  category: 'Classification' | 'Regression' | 'Ranking' | 'NLP' | 'Vision' | 'Clustering';
  direction: 'maximize' | 'minimize';
  description: string;
  commonIn: string;
}

export const ML_METRICS_CATALOG: MetricDefinition[] = [
  // --- Classification ---
  {
    id: 'Macro-F1',
    name: 'Macro-F1 (Unweighted Average F1 across all classes)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Calculates F1 per class and takes unweighted mean; critical for imbalanced classes.',
    commonIn: 'NLP, Multi-Class, Hate Speech Detection',
  },
  {
    id: 'Micro-F1',
    name: 'Micro-F1 (Global Aggregated Precision & Recall)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Calculates global TP, FP, FN across all classes; favors common classes.',
    commonIn: 'Multi-label classification, Entity Recognition',
  },
  {
    id: 'Weighted-F1',
    name: 'Weighted-F1 (Support-Weighted Average F1)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Weights per-class F1 by true class frequency/support.',
    commonIn: 'General classification with skewed priors',
  },
  {
    id: 'Binary-F1',
    name: 'Binary-F1 (Positive Class F1)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Standard harmonic mean of precision and recall for binary positive class.',
    commonIn: 'Fraud detection, Medical diagnosis',
  },
  {
    id: 'Accuracy',
    name: 'Accuracy (Fraction of Correct Predictions)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Overall exact match ratio across all test instances.',
    commonIn: 'Balanced benchmarks, ImageNet Top-1',
  },
  {
    id: 'Balanced-Accuracy',
    name: 'Balanced Accuracy (Macro Average Recall)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Average of recall obtained on each class to prevent bias towards majority class.',
    commonIn: 'Highly imbalanced medical & sensor datasets',
  },
  {
    id: 'ROC-AUC',
    name: 'ROC-AUC (Area Under Receiver Operating Characteristic)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Probability that model ranks a random positive above a random negative.',
    commonIn: 'Kaggle Tabular, Ad Click-Through, Credit Risk',
  },
  {
    id: 'PR-AUC',
    name: 'PR-AUC (Average Precision / Precision-Recall AUC)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Area under Precision-Recall curve; more informative than ROC-AUC when positives are very rare.',
    commonIn: 'Anomaly detection, Search relevance',
  },
  {
    id: 'LogLoss',
    name: 'LogLoss / Cross-Entropy Loss',
    category: 'Classification',
    direction: 'minimize',
    description: 'Measures confidence calibration of predicted probabilities; lower is better.',
    commonIn: 'Probability estimation, Multiclass tournaments',
  },
  {
    id: 'Precision',
    name: 'Precision (Positive Predictive Value)',
    category: 'Classification',
    direction: 'maximize',
    description: 'TP / (TP + FP). Minimizes false alarms.',
    commonIn: 'Spam filters, Content moderation',
  },
  {
    id: 'Recall',
    name: 'Recall (Sensitivity / True Positive Rate)',
    category: 'Classification',
    direction: 'maximize',
    description: 'TP / (TP + FN). Minimizes missed positives.',
    commonIn: 'Disease screening, Safety monitoring',
  },
  {
    id: 'Cohen-Kappa',
    name: "Cohen's Kappa (Agreement Above Chance)",
    category: 'Classification',
    direction: 'maximize',
    description: 'Measures inter-annotator or model-groundtruth agreement adjusted for chance.',
    commonIn: 'Human ratings, Essay scoring, Audits',
  },
  {
    id: 'MCC',
    name: 'Matthews Correlation Coefficient (MCC)',
    category: 'Classification',
    direction: 'maximize',
    description: 'Correlation coefficient between true and predicted binary classifications (-1 to +1).',
    commonIn: 'Bioinformatics, Imbalanced benchmark tests',
  },
  {
    id: 'Brier-Score',
    name: 'Brier Score (Mean Squared Probability Error)',
    category: 'Classification',
    direction: 'minimize',
    description: 'Mean squared difference between predicted probability and actual binary outcome.',
    commonIn: 'Weather forecasting, Probability calibration',
  },

  // --- Regression ---
  {
    id: 'RMSE',
    name: 'RMSE (Root Mean Squared Error)',
    category: 'Regression',
    direction: 'minimize',
    description: 'Square root of mean squared deviations; penalizes large outlier errors heavily.',
    commonIn: 'Price forecasting, Energy demand, Kaggle Regression',
  },
  {
    id: 'MAE',
    name: 'MAE (Mean Absolute Error)',
    category: 'Regression',
    direction: 'minimize',
    description: 'Average magnitude of errors in raw target units; robust against extreme outliers.',
    commonIn: 'Supply chain forecasting, Delivery time estimation',
  },
  {
    id: 'MSE',
    name: 'MSE (Mean Squared Error)',
    category: 'Regression',
    direction: 'minimize',
    description: 'Mean of squared differences; standard optimization loss.',
    commonIn: 'Signal processing, Statistical modeling',
  },
  {
    id: 'R2',
    name: 'R² Score (Coefficient of Determination)',
    category: 'Regression',
    direction: 'maximize',
    description: 'Proportion of target variance explained by the model (1.0 is perfect).',
    commonIn: 'Scientific modeling, Tabular regression',
  },
  {
    id: 'RMSLE',
    name: 'RMSLE (Root Mean Squared Logarithmic Error)',
    category: 'Regression',
    direction: 'minimize',
    description: 'Penalizes relative percentage errors; ideal for right-skewed positive targets.',
    commonIn: 'House Prices, Store sales, Revenue',
  },
  {
    id: 'MedAE',
    name: 'MedAE (Median Absolute Error)',
    category: 'Regression',
    direction: 'minimize',
    description: 'Median of absolute errors; completely immune to extreme outliers.',
    commonIn: 'Robust regression with noisy sensor labels',
  },
  {
    id: 'MAPE',
    name: 'MAPE (Mean Absolute Percentage Error)',
    category: 'Regression',
    direction: 'minimize',
    description: 'Relative percentage error expressed as a fraction.',
    commonIn: 'Financial forecasting, Capacity planning',
  },
  {
    id: 'Pearson-r',
    name: 'Pearson Correlation Coefficient (r)',
    category: 'Regression',
    direction: 'maximize',
    description: 'Linear correlation between continuous ground truth and prediction.',
    commonIn: 'Semantic textual similarity (STS), Bio assays',
  },
  {
    id: 'Spearman-rho',
    name: 'Spearman Rank Correlation (ρ)',
    category: 'Regression',
    direction: 'maximize',
    description: 'Monotonic relationship correlation based on rank orders.',
    commonIn: 'Subjective human evaluations, Sentiment ranking',
  },

  // --- Ranking & Retrieval ---
  {
    id: 'NDCG@5',
    name: 'NDCG@5 (Normalized Discounted Cumulative Gain at rank 5)',
    category: 'Ranking',
    direction: 'maximize',
    description: 'Measures quality of top-5 ranking, discounting relevant items appearing lower.',
    commonIn: 'Search engines, E-commerce recommendation',
  },
  {
    id: 'NDCG@10',
    name: 'NDCG@10 (Normalized Discounted Cumulative Gain at rank 10)',
    category: 'Ranking',
    direction: 'maximize',
    description: 'Standard information retrieval benchmark ranking metric at top-10.',
    commonIn: 'RecSys, Learning-to-Rank tournaments',
  },
  {
    id: 'MAP',
    name: 'MAP (Mean Average Precision)',
    category: 'Ranking',
    direction: 'maximize',
    description: 'Mean of average precision scores across queries.',
    commonIn: 'Information retrieval, Document matching',
  },
  {
    id: 'MRR',
    name: 'MRR (Mean Reciprocal Rank)',
    category: 'Ranking',
    direction: 'maximize',
    description: 'Average reciprocal rank (1/rank) of first relevant document.',
    commonIn: 'Question answering, Keyword query search',
  },
  {
    id: 'HitRate@10',
    name: 'HitRate@10 (Top-10 Recommendation Hit Rate)',
    category: 'Ranking',
    direction: 'maximize',
    description: 'Fraction of test cases where true item is present in top 10.',
    commonIn: 'Candidate generation in recommender systems',
  },

  // --- NLP & Text Generation ---
  {
    id: 'BLEU-4',
    name: 'BLEU-4 (Bilingual Evaluation Understudy 4-gram)',
    category: 'NLP',
    direction: 'maximize',
    description: 'Geometric average of modified n-gram precisions up to 4-grams with brevity penalty.',
    commonIn: 'Machine translation, Paraphrase generation',
  },
  {
    id: 'ROUGE-1',
    name: 'ROUGE-1 (Unigram Overlap F1)',
    category: 'NLP',
    direction: 'maximize',
    description: 'Measures word-level overlap between generated and reference summaries.',
    commonIn: 'Text summarization, Question answering',
  },
  {
    id: 'ROUGE-2',
    name: 'ROUGE-2 (Bigram Overlap F1)',
    category: 'NLP',
    direction: 'maximize',
    description: 'Measures fluency and word-pair co-occurrences.',
    commonIn: 'Text summarization, Abstract generation',
  },
  {
    id: 'ROUGE-L',
    name: 'ROUGE-L (Longest Common Subsequence F1)',
    category: 'NLP',
    direction: 'maximize',
    description: 'Sentence-level structure similarity based on longest common subsequence.',
    commonIn: 'Summarization, Dialogue generation',
  },
  {
    id: 'ExactMatch',
    name: 'Exact Match (EM / String Equality)',
    category: 'NLP',
    direction: 'maximize',
    description: 'Percentage of model outputs that match gold reference text character-for-character.',
    commonIn: 'SQuAD QA, Math problem solving, Code generation',
  },
  {
    id: 'Perplexity',
    name: 'Perplexity (PPL)',
    category: 'NLP',
    direction: 'minimize',
    description: 'Exponentiated cross-entropy of token sequences; measures language modeling surprise.',
    commonIn: 'Pretraining, Language model benchmarking',
  },
  {
    id: 'WER',
    name: 'WER (Word Error Rate)',
    category: 'NLP',
    direction: 'minimize',
    description: 'Substitutions + Deletions + Insertions divided by total words.',
    commonIn: 'Speech recognition (ASR), OCR evaluation',
  },
  {
    id: 'CER',
    name: 'CER (Character Error Rate)',
    category: 'NLP',
    direction: 'minimize',
    description: 'Levenshtein edit distance at character level normalized by reference length.',
    commonIn: 'Handwriting OCR, Typo correction',
  },

  // --- Computer Vision ---
  {
    id: 'Top-1-Accuracy',
    name: 'Top-1 Accuracy',
    category: 'Vision',
    direction: 'maximize',
    description: 'Model top predicted class matches gold target exactly.',
    commonIn: 'Image classification, CIFAR, ImageNet',
  },
  {
    id: 'Top-5-Accuracy',
    name: 'Top-5 Accuracy',
    category: 'Vision',
    direction: 'maximize',
    description: 'Target class is within top 5 highest confidence predictions.',
    commonIn: 'Large-scale fine-grained visual classification',
  },
  {
    id: 'mAP@0.5',
    name: 'mAP@0.5 (Mean Average Precision at IoU=0.5)',
    category: 'Vision',
    direction: 'maximize',
    description: 'PASCAL VOC standard detection metric at IoU threshold 0.50.',
    commonIn: 'Object detection, Bounding box regression',
  },
  {
    id: 'mAP@0.5:0.95',
    name: 'mAP@0.5:0.95 (COCO Detection / Segmentation mAP)',
    category: 'Vision',
    direction: 'maximize',
    description: 'Average mAP across 10 IoU thresholds from 0.50 to 0.95 with step 0.05.',
    commonIn: 'MS COCO, Autonomous driving detection',
  },
  {
    id: 'IoU',
    name: 'IoU (Intersection over Union / Jaccard Index)',
    category: 'Vision',
    direction: 'maximize',
    description: 'Area of overlap divided by area of union between predicted and gold masks.',
    commonIn: 'Semantic segmentation, Medical imaging',
  },
  {
    id: 'Dice',
    name: 'Dice Similarity Coefficient (F1 Mask Score)',
    category: 'Vision',
    direction: 'maximize',
    description: '2 * |X ∩ Y| / (|X| + |Y|); standard for foreground segmentation.',
    commonIn: 'Biomedical image segmentation',
  },
  {
    id: 'PSNR',
    name: 'PSNR (Peak Signal-to-Noise Ratio)',
    category: 'Vision',
    direction: 'maximize',
    description: 'Logarithmic ratio between peak signal power and corrupting noise power.',
    commonIn: 'Super-resolution, Image denoising',
  },
  {
    id: 'SSIM',
    name: 'SSIM (Structural Similarity Index Measure)',
    category: 'Vision',
    direction: 'maximize',
    description: 'Perceived visual similarity considering luminance, contrast, and structure (-1 to 1).',
    commonIn: 'Image reconstruction, Inpainting, Generative models',
  },

  // --- Clustering ---
  {
    id: 'Silhouette',
    name: 'Silhouette Coefficient',
    category: 'Clustering',
    direction: 'maximize',
    description: 'Measures how well separated clusters are (-1 to +1).',
    commonIn: 'Unsupervised clustering, Customer segmentation',
  },
  {
    id: 'ARI',
    name: 'Adjusted Rand Index (ARI)',
    category: 'Clustering',
    direction: 'maximize',
    description: 'Measures similarity between two clusterings adjusted for chance agreement.',
    commonIn: 'Community detection, Document clustering',
  },
  {
    id: 'NMI',
    name: 'Normalized Mutual Information (NMI)',
    category: 'Clustering',
    direction: 'maximize',
    description: 'Normalized mutual information between cluster assignments and ground truth.',
    commonIn: 'Topic modeling, Unsupervised representation learning',
  },
];

export function getMetricById(id: string): MetricDefinition | undefined {
  const cleanId = id.trim().toLowerCase();
  return ML_METRICS_CATALOG.find(
    (m) => m.id.toLowerCase() === cleanId || m.name.toLowerCase().startsWith(cleanId)
  );
}

export function getSuggestedDirection(metricName: string): 'maximize' | 'minimize' {
  const found = getMetricById(metricName);
  if (found) return found.direction;

  const lower = metricName.toLowerCase();
  if (
    lower.includes('loss') ||
    lower.includes('rmse') ||
    lower.includes('mse') ||
    lower.includes('mae') ||
    lower.includes('error') ||
    lower.includes('wer') ||
    lower.includes('cer') ||
    lower.includes('perplexity') ||
    lower.includes('ppl')
  ) {
    return 'minimize';
  }
  return 'maximize';
}
