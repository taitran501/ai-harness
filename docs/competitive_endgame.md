# Competitive ML Endgame Suite

Documentation for high-tier competition ensembling, knowledge transfer, and submission auditing in `ai-harness`.

---

## 1. Motivation & Architectural Blueprint

In competitive machine learning (e.g., Kaggle, DrivenData), single tree-based models (LightGBM, XGBoost, CatBoost) inevitably plateau around an inductive bias ceiling. To achieve top-tier finishes (e.g., Top 100 on Kaggle), teams must execute a **two-layer competitive endgame**:

1. **Tri-Family Synergy (Layer 1)**: Combine orthogonal paradigms:
   - Axis-aligned decision trees (custom feature-engineered GBDTs).
   - In-context Tabular Foundation Models (TabPFN, LimiX, TabFM).
   - Deep tabular neural architectures (RealMLP).
2. **Meta-Stacking with Shrinkage (Layer 2)**:
   - Standard linear regression or $C=1.0$ logistic regression oscillates violently over 10-100 correlated inputs ($r > 0.95$).
   - `L2LogitStacker` applies heavy $L_2$ regularization ($C=0.003$) on bounded logits $\ln(p / (1-p)) \in [-15, 15]$.
3. **Scale-Invariant Rank Averaging**:
   - `RankAverager` combines final meta-stack predictions with diverse top-scoring public submissions to eliminate calibration drift and minimize test variance.

---

## 2. Quickstart Usage Examples

### A. Prediction Alignment (`align_prediction_frames`)
Guarantees Out-of-Fold (OOF) predictions are strictly indexed by natural dataset `id`, preventing devastating row-order mismatch bugs:

```python
from harness.stacking import align_prediction_frames

# Align candidate models to ground truth reference table
X_oof, model_names = align_prediction_frames(
    prediction_frames=[oof_exp001, oof_exp007, oof_tabpfn],
    reference_df=train_df,
    id_column="id",
)
```

### B. Regularized Logit Stacking (`L2LogitStacker`)
Fits a 5-fold nested L2 shrinkage stacker on bounded logits:

```python
from harness.stacking import L2LogitStacker

stacker = L2LogitStacker(C=0.003, n_splits=5, random_state=42)
oof_pred, test_pred = stacker.fit_predict(
    X_oof=X_oof,
    y=y_train,
    X_test=X_test,
    feature_names=model_names,
)

# Inspect model coefficients (positive vs contrastive negative weights)
importance_df = stacker.get_feature_importance()
print(importance_df.head(10))
```

### C. Scale-Invariant Rank Averaging (`RankAverager`)
Averages percentiles monotonically:

```python
from harness.stacking import RankAverager

final_test_pred = RankAverager.blend(
    prediction_arrays=[ours_stack_test, grandmaster_model_a, grandmaster_model_b],
    weights=[0.40, 0.35, 0.25],
)
```

### D. External Teacher Model Knowledge Transfer (`TeacherLogitPrior`)
Transfers conditional probability priors from auxiliary reference datasets without generator drift:

```python
from harness.foundation_models import TeacherLogitPrior

teacher = TeacherLogitPrior(
    target_column="target",
    model_type="lightgbm",
)
teacher.fit(auxiliary_original_df)

# Extract smooth priors for train and test sets
train_priors = teacher.transform(train_df, prefix="orig_teacher")
test_priors = teacher.transform(test_df, prefix="orig_teacher")
```

### E. Preflight Submission Auditor (`SubmissionPreflightValidator`)
Audits schema, row count, sequence integrity, and probability bounds prior to submission:

```python
from harness.endgame_submission import SubmissionPreflightValidator

report = SubmissionPreflightValidator.audit_submission(
    submission_path="submission.csv",
    reference_test_path="dataset/test.csv",
    id_column="id",
    probability_mode=True,
)
if not report.is_valid:
    raise ValueError(f"Submission rejected: {report.errors}")
```
