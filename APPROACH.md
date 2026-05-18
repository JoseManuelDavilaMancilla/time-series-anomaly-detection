# Approach Summary (THOMASSO CODE)

This folder implements a **time-series anomaly detection** pipeline built around a consistent pattern:

1. **Load each “window”** (a single time-series case) from a dataset directory.
2. **Compute pointwise features** (rolling stats, diffs, robust z-scores, percentiles, time features) plus a small set of **window-static features**.
3. **Train pooled supervised models** by aggregating labeled points across many windows.
4. **Score each test window** with model probabilities.
5. **Optionally smooth** probabilities to reduce noise.
6. **Calibrate to hard labels** by predicting **exactly top‑k anomalies per window**, where *k* comes from metadata in `info.json`.

A second recurring theme is **iteration by versioning**: `metric_pooled_v2`, `metric_pooled_v5`, `metric_pooled_v8` represent successive attempts that add features, improve robustness on edge cases, and combine complementary pipelines.

---

## Expected Dataset Shape

The code assumes a dataset root containing many window directories, each with:

- `train.npy` (float series)
- `train_label.npy` (binary labels in `{0,1}`)
- `train_timestamp.npy` (int timestamps)
- `test.npy` (float series)
- `test_timestamp.npy` (int timestamps)
- `info.json` (metadata; notably includes `test set anomaly ratio`, `test_seq_len`, and identifiers like `metric_type`, `case_name`)

The `info.json` metadata is also used to infer the target number of positives (`k`) for the top‑k calibrator.

---

## Core Building Blocks

### 1) I/O and window abstraction (`io_utils.py`)

- Defines a `WindowData` dataclass that centralizes:
  - raw arrays (`train`, `train_label`, `train_ts`, `test`, `test_ts`)
  - `info.json` metadata
  - derived convenience fields like `test_k`, `metric_type`, `metric_name`
- Provides:
  - `list_window_dirs(dataset_root)` → stable ordering by 3-digit prefix
  - `load_window(window_dir)` → reads arrays + `info.json`, validates label binarity, computes `test_k`

**Pattern to reuse:** treat every problem instance as a “window object” that travels through the pipeline unchanged.

### 2) Deterministic calibration to a fixed anomaly budget (`calibrate.py`)

This module handles score → label conversion with emphasis on **determinism**.

- `topk_from_score(score, k)`:
  - outputs a binary vector with **exactly** `k` ones (clamped to `[0, n]`)
  - deterministic tie-breaking using `np.lexsort` on `(-score, index)`
- `read_test_k(window_info_dict)`:
  - `k = round(test_set_anomaly_ratio * test_seq_len)` with clamping
- Fallback calibration modes are included:
  - `train_f1_threshold(train_score, train_label)` tunes a threshold to maximize train F1
  - `calibrate_window(..., mode=...)` unifies top‑k vs threshold-based approaches

**Pattern to reuse:** if your evaluation metric is top‑k-like (or the dataset fixes anomaly prevalence), *calibration becomes a separate step* from modeling.

### 3) Robust, rolling feature library (`features.py`)

A compact feature toolkit designed for noisy time-series.

Key properties:

- **NaN/Inf-safe** via `_safe()` (non-finite → 0)
- Rolling features via `rolling(x, w, fn)`:
  - preferred implementation uses `pandas.Series.rolling(..., center=True, min_periods=1)`
  - fallback is a pure-numpy loop when pandas is unavailable
- Common derived features:
  - `rolling_mean/std/median/mad/min/max`
  - `first_diff`, `second_diff`
  - `ewma`
  - `percentile_rank` / `ecdf_score` vs a reference distribution

It also exposes `point_feature_matrix(train, ...)` which:

- builds a **train point feature matrix** and
- returns a closure `apply_fn(test, ...)` to apply the same train-derived transforms to test.

**Pattern to reuse:** compute train summary stats once, then provide a “feature closure” that guarantees consistent transforms for test.

### 4) Pointwise + static features for pooled training (`pooled_features.py`)

This module creates **one feature vector per time point**, then appends a **window-static block** to each row.

- Per-point features:
  - raw value, robust and standard z-scores
  - diffs
  - rolling stats at multiple horizons (5/11/21)
  - EWMA residuals
  - percentile rank vs train
  - positional features (normalized index)
  - time features (time-of-day and day-of-week proxies)
- Static per-window features (broadcast to all points):
  - metric type one-hot
  - “service” one-hot based on parsing `case_name`
  - normalized `intervals` plus anomaly ratio metadata

A global list `TOP_SERVICES` is populated externally by a helper that counts services across window dirs.

**Pattern to reuse:** concatenate (point_features || static_window_features) so a single classifier can learn both local patterns and coarse window context.

---

## Two Training Strategies

### A) One global model across all windows (`pooled_supervised.py`)

High-level logic:

1. Aggregate `X_train` / `y_train` from **all windows that contain at least one positive label**.
2. Train a single `HistGradientBoostingClassifier` with sample weighting to balance positives.
3. For each test window:
   - compute `predict_proba` per point
   - smooth and/or postprocess (optional)
   - `topk_from_score(prob, k)` to match the required anomaly budget

It also includes a **time-respecting validation scheme**:

- for each labeled window, the first 70% of train is used for training pool and the last 30% is used for validation
- validation uses **per-window top‑k** with `k = sum(val_labels)` so each window’s anomaly count is respected

Important note: `pooled_supervised.py` imports `metrics.binary_f1` and `postprocess.apply_postprocess`, but `metrics.py` and `postprocess.py` are not present in this folder snapshot. If you want to run `pooled_supervised.py` as-is, you’ll need to provide those modules (or remove those optional features).

**Pattern to reuse:** pooled learning over a heterogeneous dataset + time-respecting split that mirrors a leaderboard setting.

### B) One model per metric type (pooled-by-type) + ensembling (`metric_pooled_v2.py`, `metric_pooled_v5.py`, `metric_pooled_v8.py`)

These scripts train a separate pooled model **per `metric_type`**, to reduce heterogeneity.

Common structure across versions:

- Group windows by `metric_type`.
- Build training pool from windows with at least one positive label.
- Train an ensemble (multiple seeds and/or multiple model families).
- Score each test window with the model(s) for its `metric_type`.
- Smooth probabilities.
- Calibrate to **exact top‑k** labels.

This design isolates metric-specific behavior (e.g., Latency vs QPS) and enables specialized smoothing settings.

---

## Versioned “Pooled Metric” Scripts (Evolution)

### `metric_pooled_v2.py` — extra long-range features + seed ensemble + smoothing

- Builds features as:
  - `pooled_features.build_pointwise_matrix` + extra long-range rolling features (w=41)
  - local volatility ratio and rolling min/max deviation features
- Fits `HistGradientBoostingClassifier` multiple times with different random seeds.
- Averages predicted probabilities across seeds.
- Applies rolling-mean smoothing with blend `alpha`.
- Predicts `topk_from_score(prob, k)`.

Notable engineering choice: deterministic behavior in the calibrator, and “small code paths” for edge cases (`k==0`, no models).

### `metric_pooled_v5.py` — adds within-test distribution-shift features + model-family blend

Adds features to make the model more sensitive to **train→test regime shifts**, especially for windows with **no labeled anomalies**.

Feature block includes:

- rank/percentile within the test window
- robust z-score relative to test distribution
- flags for being outside train min/max support + distances beyond support
- broadcasted scalars capturing mean shift, std ratio, median shift

It blends multiple model families:

- HGBT ensemble + optional RandomForest ensemble + optional LogisticRegression
- linear probability mixture with tunable weights

### `metric_pooled_v8.py` — probability average of two pipelines

Implements two parallel pipelines per metric type:

- P1: “v023-style” (`build_feats_extra`)
- P2: “v024-style” (`build_feats_shift`)

Then combines probabilities:

- `prob = (1 - w_shift) * prob_P1 + w_shift * prob_P2`
- smoothing + top‑k calibration

Key insight encoded in the header docs: **probability averaging** performed better than rank averaging because rank normalization + smoothing harmed calibration.

---

## Copyable Patterns for Another Agent

### Pattern: end-to-end scoring skeleton

Use this structure as a template:

1. `window_dirs = list_window_dirs(dataset_root)`
2. Fit phase:
   - load windows
   - filter to labeled windows (or create pseudo-labels)
   - build `X_train`, `y_train`
   - fit classifier(s)
3. Predict phase:
   - for each window:
     - compute `prob = model.predict_proba(X_test)[:,1]`
     - optional smoothing
     - `pred = topk_from_score(prob, k)`
4. Write JSON: `{ "predictions": { window_id: [0/1,...] } }`

### Pattern: keep feature builders pure

Feature builder functions here are mostly “pure”:

- inputs: `WindowData`, `series_kind` (`"train"` or `"test"`)
- output: `np.ndarray` of features

This makes it easy to:

- swap feature sets by version
- ensemble feature sets (v8)
- unit test feature construction separately

### Pattern: handle edge cases explicitly

These scripts consistently short-circuit:

- `k == 0` → all-zero predictions
- no model available for a metric type → all-zero predictions
- no positives in a training pool → skip training that pool

This avoids fragile behavior in heterogeneous datasets.

---

## Practical Notes / Dependencies

- Hard dependencies: `numpy`
- Optional: `pandas` (speeds up rolling computations in `features.py`)
- Model training scripts require: `scikit-learn`
- `pooled_supervised.py` references `metrics` and `postprocess` modules that are not present in this folder snapshot.

---

## If You Extend This Code

The simplest, most aligned extension is:

- add a new `build_feats_*` function that concatenates additional pointwise or window-static features
- train per-metric-type pools and ensemble over seeds
- keep calibration and smoothing unchanged to preserve submission semantics
