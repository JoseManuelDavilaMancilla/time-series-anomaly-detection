# Submission History — ANM2026 Time Series Anomaly Detection

This file documents every submission pipeline built, in chronological order, including what method each one used and the key design decisions.

---

## Submission 1: `submission.json` (Initial Unsupervised Ensemble)

**When**: First pipeline run.

**Core idea**: Ensemble of classical unsupervised anomaly detectors with per-window threshold grid-search.

**Methods included**:
- **Value-based statistical**: Z-score, MAD (Median Absolute Deviation), IQR
- **ML / density**: Isolation Forest, Local Outlier Factor (LOF)
- **Deep learning**: 1D fully-connected autoencoder (PyTorch)

**Ensemble strategy**:
1. For each window, compute anomaly scores from every model on both train and test.
2. Normalize each model's test scores using **test** min/max (later found to be a bug).
3. Average normalized scores → ensemble score.
4. Grid-search threshold on training ensemble scores to maximize training F1.
5. Apply threshold to test ensemble scores.

**Threshold handling**:
- Windows with 0 training anomalies → use `info.json` test anomaly ratio for percentile-based threshold.
- Constant training data → return all zeros (buggy, later fixed).

**Key bugs in this version**:
- Scorers like `score_zscore` take 2 args but were called with 3 args, causing `TypeError` silently caught by `try/except`. Only Isolation Forest and autoencoder actually contributed.
- Normalization used **test** min/max instead of **train** min/max, making thresholds non-transferable.
- Constant-data edge case used quantile thresholding which flagged everything when scores had heavy zero mass.

**Estimated score**: ~0.33 (leaderboard).

---

## Submission 2: `submission.json` (Bug Fixes + Weighted Ensemble)

**When**: After discovering the TypeError bug and normalization bug.

**Changes from v1**:
- **Fixed scorer calls**: wrapped 2-arg scorers in 3-arg lambdas so all models actually run.
- **Fixed normalization**: test scores now normalized using **train** statistics (`(test - train_min) / (train_max - train_min)`), then clipped to `[-5, 5]` to prevent outliers from dominating.
- **Added temporal scorers**:
  - `score_diff_zscore` — z-score of first differences
  - `score_diff_mad` — MAD of first differences
  - `score_diff_iqr` — IQR of first differences
  - `score_jerk` — z-score of second differences
  - `score_rolling_zscore` — z-score relative to last training window
  - `score_percentile` — extreme percentile (5th/95th) based score
- **Weighted ensemble**: each model gets a weight = max(training F1, 0.01). Ensemble is a weighted average of normalized scores.
- **Fixed constant-data edge case**: switched from quantile thresholding to exact **top-k** selection (`np.argpartition`).
- **Threshold regularization**: `find_best_threshold` skips thresholds predicting <10% or >90% anomalies; falls back to top-k matching training rate.
- **Test-rate fallback**: if grid-search threshold predicts too extreme a rate on test, fall back to top-k using training anomaly rate.

**Estimated score**: improved but still limited by unsupervised methods on contextual anomalies.

---

## Submission 3: `submission.json` (Use Test Ratio for All Windows)

**When**: After analysis showed severe under/over-prediction when using training rate for windows with train/test distribution shift.

**Key change**:
- For **all** windows, the final prediction uses **top-k selection** where `k = round(len(test) * test_ratio)` from `info.json`.
- For windows with training anomalies, the supervised/ensemble model provides the **ranking**, and top-k ensures exactly the expected number of anomalies.
- For windows with 0 training anomalies, unsupervised ensemble provides the ranking.

**Impact**:
- Mean abs diff from test ratio dropped from ~0.08 to **0.024**.
- Eliminated 100% prediction windows.
- Score improved from ~0.33 to **~0.47**.

**Limitation**: still relies on unsupervised models for ranking, which miss contextual anomalies.

---

## Submission 4: `submission.json` (Supervised RandomForest — Primary Method)

**When**: After discovering that temporal contextual anomalies (sudden dips/spikes) are invisible to value-based methods but detectable with feature engineering.

**Core idea**: Replace the unsupervised ensemble with a **supervised tree-based classifier** trained on engineered features, using training labels directly.

**Feature engineering** (per point):
- `value` — raw value
- Differences & ratios at lags 1, 2, 3, 5, 10
- Rolling mean / std / min / max / z-score at windows 3, 5, 10
- Second difference (jerk)
- EMA and EMA deviation

**Model**: `RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced')`

**Pipeline per window**:
1. Extract features for train + test combined (so lag features are continuous across the boundary).
2. Train RF on training features + labels.
3. Predict anomaly probability on test features.
4. Select top-k using `info.json` test ratio.

**Fallback**: for windows with 0 training anomalies or constant training data, use the unsupervised ensemble.

**Why this is powerful**:
- Train oracle F1 ≈ **1.0** on almost all windows with training anomalies (the model can perfectly separate training anomalies from normal points given the features).
- Captures **contextual / shape anomalies** (sudden changes, volatility spikes) that pure value-based methods miss.

**Score**: **~0.47** (similar to v3, but with much better ranking quality on many windows).

---

## Submission 5: `submission_xgboost.json` (Boosted Trees — XGBoost + LightGBM)

**When**: After researching Kaggle-winning solutions for similar competitions.

**Core idea**: Same engineered features as the RF submission, but use **gradient-boosted trees** (XGBoost + LightGBM) instead of RandomForest, then average their predicted probabilities.

**Models**:
- **XGBoost**: `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=neg/pos)`
- **LightGBM**: `LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, class_weight='balanced')`

**Ensemble**: average of XGBoost and LightGBM predicted probabilities.

**Fallback**: identical to RF version (unsupervised ensemble for 0-anomaly windows).

**Result**: improvement over RF was **negligible** (~0.47). This indicates the bottleneck is **not model capacity** but **generalization** — train F1 is already ~1.0, so the gap is caused by train/test distribution shift.

---

## Deep Dive: Experiments & Analysis Framework

**When**: After submission v5 plateaued, we built a reusable `experiments/` framework to systematically test ideas and collect evidence for the report.

**Framework created**:
- `experiments/utils.py` — shared utilities (data loading, F1 computation, score normalization, result persistence)
- `experiments/scorers.py` — 17+ anomaly scorers organized by type (statistical, temporal, online, distance, density, isolation, supervised, deep learning)
- `experiments/exp_*.py` — individual experiment scripts for each approach
- `experiments/run_all_experiments.py` — orchestrates all experiments and produces a summary table
- `experiments/analyze_dataset.py` — dataset-wide statistics
- `experiments/visualize_window.py` — plot individual windows
- `experiments/generate_report_plots.py` — generate report figures

---

### Experiment 1: Baseline Unsupervised Ensemble (v1 revisited)

**Result**: mean val F1 = **0.200** (time-based 70/30 split, n=137/200 windows with anomalies)

**Insight**: unsupervised methods alone are too weak for this dataset.

---

### Experiment 2: Supervised RandomForest with Features (v3 revisited)

**Result**: mean val F1 = **0.252** (time-based split)

**Insight**: supervised features help, but the time-based validation is very hard.

---

### Experiment 3: XGBoost + LightGBM Ensemble (v4 revisited)

**Result**: XGBoost failed on macOS due to missing `libomp.dylib`; LightGBM alone not tested separately.

**Note**: on platforms where XGBoost works, this should be re-run for a fair comparison.

---

### Experiment 4: Per-Window Model Selection

**Idea**: for each window, evaluate all scorers on training data and pick the one with best train F1.

**Result**: mean val F1 = **0.180**. Isolation Forest was selected for **100%** of windows in the sample.

**Insight**: model selection based on training performance does **not** transfer to validation. The best scorer on training data is not the best on test data.

---

### Experiment 5: Stacking / Learned Ensemble Weighting

**Strategies tested**:
- Mean ensemble: 0.180
- Weighted softmax (by train F1): 0.180
- Top-3 ensemble: 0.180
- Stack with LogisticRegression: 0.187
- Stack with small RF: 0.182

**Insight**: learned combinations of scorers do not outperform simple averaging. The scorers are not complementary enough, or the training signal is too weak.

---

### Experiment 6: Online / Adaptive Scorers

**Idea**: for windows with train/test distribution shift, use scorers that adapt to the test data itself (rolling z-score, diff z-score, jerk).

**Result**:
- Online only: **0.127**
- Global only (train-based): **0.207**
- Hybrid equal: **0.209**
- Hybrid global-heavy: **0.204**

**Insight**: online adaptive scorers are worse than global train-based scorers on time-based validation. However, this may be because time-based validation still comes from the same distribution as training. On actual disjoint test windows, online methods might be better.

---

### Experiment 7: Subsequence / Discord Detection

**Idea**: anomalies are contiguous segments; detect anomalous subsequences by comparing test subsequences to training subsequences.

**Result**: m3=0.143, m5=0.150, m7=0.151, m10=0.152

**Insight**: subsequence-based methods underperform pointwise methods. The anomaly signature is not strongly shape-based at the subsequence level.

---

### Experiment 8: Cross-Window Training

**Idea**: train a single RF on all windows' training data, then do leave-one-out validation.

**Result**: mean val F1 = **0.633** (n=76 windows with anomalies)

**Insight**: anomaly patterns **do generalize across windows**! A cross-window model achieves 0.633 F1 when evaluated on held-out windows' training data. This is very close to the top leaderboard score (0.635), suggesting the top performers may be using cross-window models.

**Caveat**: 0.633 is on training data. The actual test gap (0.47) is caused by train/test distribution shift within individual windows.

---

## Key Dataset Findings

From `experiments/analyze_dataset.py`:

- **1000 windows**, train/test lengths: 128–512 (mean ~315)
- **245 windows (24.5%)** have **zero training anomalies**
- **Train anomaly rate**: mean 9.1%, median 4.7%, max 71.3%
- **Test anomaly ratio**: mean 9.5%, median 5.1%, max 66.3%
- **Anomalies are contiguous segments**: mean 1.98 segments per anomalous window, mean segment length 16.9, median 10.0
- **Train-test value overlap**:
  - Mean overlap ratio: **0.297**
  - Median overlap ratio: **0.226**
  - **268 windows (26.8%)** have **zero overlap** (disjoint ranges)

---

## Critical Insight: Validation Strategy Matters Enormously

We compared two validation strategies on the same model (supervised RF with engineered features):

| Validation Strategy | Mean F1 | Notes |
|---|---|---|
| Random stratified split (30% holdout) | **0.717** | Train and val from same distribution |
| Time-based split (70/30 chronological) | **0.270** | Val comes after train in time |
| Train oracle (predict on training data) | **0.962** | Model perfectly fits training |
| Actual leaderboard test | **~0.47** | Unknown test labels |

**Interpretation**:
- The model has **more than enough capacity** (train F1 = 0.96).
- Random split validation (0.72) shows the model **can generalize** when test data comes from the same distribution.
- Time-based split validation (0.27) shows that **distribution shift** over time is severe.
- The actual test score (0.47) is between these extremes, suggesting the test set has **moderate distribution shift** — not as severe as the last 30% of training, but more than a random split.

---

## Submission 6: `submission_v6.json` (Category-Aware Strategy Selection)

**When**: After discovering that windows fall into distinct categories based on train-test value range overlap, and that different categories need different strategies.

**Window categorization**:
| Category | Count | % | Strategy |
|---|---|---|---|
| `constant_train` | 70 | 7.0% | Online adaptive ensemble only |
| `disjoint` | 198 | 19.8% | Global distance + online adaptive, weighted by test ratio |
| `partial_overlap` | 552 | 55.2% | Supervised RF + online hybrid (60/40) |
| `test_within_train` | 180 | 18.0% | Supervised RF only |

**Key design decisions**:
1. **Constant training data**: supervised models cannot learn (no variation), so we fall back to online scorers that detect local anomalies within the test window.
2. **Disjoint ranges**: train statistics are irrelevant for the test distribution. We use global distance from train (catches level shifts) combined with online local scorers (catches unusual points within the new level). The weighting favors global distance when test ratio is high (suggesting many anomalies / level shift).
3. **Partial overlap**: hybrid approach combines supervised model (good for overlapping region) with online scorers (good for new regions in test).
4. **Test within train**: supervised RF should generalize well because test values are within the training distribution.

**All predictions use top-k selection with `info.json` test anomaly ratio**.

**File**: `generate_submission_v6.py`

---

## Summary of Techniques Tried

| Technique | Type | In submissions |
|---|---|---|
| Z-score | Value-based unsupervised | v1, v2, fallback |
| MAD | Value-based unsupervised | v1, v2, fallback |
| IQR | Value-based unsupervised | v1, v2, fallback |
| Percentile | Value-based unsupervised | v2, fallback |
| Rolling Z-score | Value-based unsupervised | v2, fallback |
| Isolation Forest | Density-based ML | v1, v2, fallback |
| LOF | Density-based ML | v1, v2, fallback |
| Autoencoder (1D FC) | Reconstruction DL | v1, v2 |
| First-diff Z-score / MAD / IQR | Temporal unsupervised | v2, v3, fallback |
| Jerk (second-diff Z-score) | Temporal unsupervised | v2, v3, fallback |
| RandomForest on engineered features | Supervised ML | v4, v6 |
| XGBoost + LightGBM ensemble | Supervised ML | v5 |
| Online rolling z-score / diff / jerk | Adaptive unsupervised | v6, experiments |
| Global distance score | Train-test comparison | v6 |
| Subsequence discord (k-NN) | Shape-based | experiment |
| Cross-window RF | Multi-window supervised | experiment |
| Model selection (per window) | Meta-learning | experiment |
| Stacking (LR / RF meta-models) | Meta-learning | experiment |

---

## Current Files in Repo

| File | Description |
|---|---|
| `generate_submission.py` | RandomForest supervised pipeline (submission v4) — produces `submission.json` |
| `generate_submission_xgboost.py` | XGBoost + LightGBM pipeline (submission v5) — produces `submission_xgboost.json` |
| `generate_submission_v6.py` | Category-aware strategy pipeline (submission v6) — produces `submission_v6.json` |
| `submission.json` | RF-based submission |
| `submission_xgboost.json` | Boosted-tree variant |
| `submission_v6.json` | Category-aware variant |
| `validate_submission.py` | Format validator |
| `submissions_log.md` | This file |
| `experiments/` | Reusable experiment framework |
| `results/` | Experiment results (JSON) |
| `report_plots/` | Generated figures for report |

---

## Submissions v10–v11: Refinement Variants

After v8 (0.5485) became the best, we generated several lightweight variants to explore the weight space and test alternative hypotheses.

### v10_light: Lighter Augmentation
- Single augmented variant per window (2x data), single RF model
- Faster to train than full v10 (which timed out after 15 min)
- File: `submission_v10_light.json`

### v11a: Higher Cross-Window Weight
- test_within_train: 65/35
- partial_overlap: 45/35/20
- disjoint: 40/25/35
- constant_train: 60/40

### v11b: Higher Online Weight for Disjoint
- disjoint: 20/20/60 (more online, less cross-window/global)
- Other categories moderately adjusted

### v11c: Balanced Middle Ground
- Moderate adjustments across all categories

### v11d: No Per-Window RF
- Hypothesis: per-window RF overfits and hurts test generalization
- Uses ONLY cross-window model + online scorers
- test_within_train: 100% cross-window
- partial_overlap: 60/40 cross-window/online
- disjoint: 40/30/30 cross-window/global/online
- constant_train: 60/40 cross-window/online
- File: `submission_v11d.json`

---

## Current Best Submissions (Ready to Test)

| File | Approach | Score (if tested) |
|---|---|---|
| `submission_v8.json` | Scale-invariant cross-window + per-window RF + online | **0.5485** ★ |
| `submission_v9.json` | Metric-aware cross-window | 0.5367 |
| `submission_v7.json` | Full cross-window (with raw value) | 0.527 |
| `submission_v10_light.json` | Light augmentation + cross-window | untested |
| `submission_v11a.json` | Higher CW weight | untested |
| `submission_v11b.json` | Higher online for disjoint | untested |
| `submission_v11c.json` | Balanced weights | untested |
| `submission_v11d.json` | No per-window RF | untested |


---

## CRITICAL BUG FIX: Zero-Ratio Windows

**Discovery**: 235 windows have `test set anomaly ratio = 0.0` in `info.json`.

**Bug**: All submission generators used `k = max(1, int(round(len(test_x) * test_ratio)))`, which forced at least **1 predicted anomaly** even when the expected number was **0**.

**Impact**: For windows with 0 true anomalies, predicting 1 anomaly gives:
- Precision = 0 / (0 + 1) = 0
- Recall = 0 / (0 + 0) = undefined (but effectively 0)
- F1 = 0

This means **235 windows were guaranteed to get F1 = 0** regardless of how good the rest of the pipeline was.

**Fix**: Changed to `k = int(round(len(test_x) * test_ratio))` in all submission generators. For `test_ratio = 0`, this correctly predicts **all zeros**.

**Expected improvement**: If the leaderboard gives F1 = 1.0 for perfect predictions on zero-anomaly windows, the overall score could improve by up to **0.106** (from 0.5485 to ~0.654). Even with a conservative estimate, this should be a significant boost.

**Files regenerated with fix**:
- `submission_v8.json` (best so far, 0.5485)
- `submission_v10_light.json`
- `submission_v11a.json`, `submission_v11b.json`, `submission_v11c.json`
- `submission_v11d.json`
- `submission_v12.json` (3-RF ensemble + enhanced online)

