# Time Series Anomaly Detection Pipeline

A supervised time-series anomaly detection pipeline for multi-window datasets, evolved through 71 iterative experiments.

## Overview

This repository contains a single evolving pipeline (`pipeline.py`) tracked through 71 git commits, each representing a version from v1 to v71. The final implementation (v71) is a per-metric-type ensemble of tree-based models with extensive feature engineering, pseudo-labeling, and post-processing.

## Dataset Structure

The pipeline expects a `student_dataset/` directory containing 1000 window folders. Each folder has:

- `train.npy` — float time-series (training data)
- `train_label.npy` — binary labels in `{0, 1}`
- `test.npy` — float time-series (test data)
- `info.json` — metadata including `metric_type`, `test set anomaly ratio`, etc.

## Repository Structure

```
.
├── pipeline.py              # Main submission pipeline (evolves through git history)
├── validation.py            # Time-split validation harness
├── cross_validation.py      # Cross-window leave-N-out validation
├── precompute_tda.py        # Precompute TDA cache (run once)
├── validate_submission.py   # Format checker for submission.json
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Running the Pipeline

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Precompute TDA cache

If using versions v61+, precompute topological features once:

```bash
python precompute_tda.py
```

### 3. Run the pipeline

```bash
python pipeline.py
```

This generates `submission.json` with binary predictions for all 1000 test windows.

### 4. Validate the submission

```bash
python validate_submission.py
```

## Chronological Evolution

The git history of `pipeline.py` tracks our iterative development:

| Phase | Versions | Key Idea |
|-------|----------|----------|
| Unsupervised ensemble | v1–v5 | Statistical scorers, isolation forest, autoencoder |
| CNN experiments | v6–v12 | 1D CNN cross-window scorer, per-metric routing |
| GBM & stacking | v13–v17 | LightGBM, meta-classifiers |
| Segment & TTA | v18–v21 | Segment-level classification, test-time adaptation |
| Metadata & routing | v22–v25 | Interval-based routing, submission voting |
| Deep learning | v26–v30 | Transformer, anomaly transformer |
| Investigation | v31–v36 | Failure-mode analysis, post-processing fixes |
| Friend replication | v37–v39 | Replicating a 0.645 LB approach |
| Centered rolling | v40–v42 | **Key fix**: centered rolling statistics |
| Two-pipeline blend | v43–v49 | P1 standard + P2 shift-detection blend |
| Feature expansion | v50–v52 | Extended rolling min/max |
| Pseudo-labeling | v53–v57 | **Breakthrough**: transductive pseudo-labeling |
| Global context | v58 | Cross-window z-score features |
| Spectral & TDA | v59–v64 | FFT, persistent homology, wavelets |
| Matrix profile | v65–v67 | Stumpy matrix profile + CUSUM |
| Seasonal & AR | v68 | STL decomposition + AR(1) residuals |
| Combined & catch22 | v69–v71 | catch22, complexity, per-metric rules, CatBoost |

## Validation

Two validation strategies are provided:

- **validation.py**: 70/30 time-split within held-out windows (fast but optimistic)
- **cross_validation.py**: Leave-N-out whole windows (slower but closer to leaderboard)

## Dependencies

- numpy
- pandas
- scikit-learn
- stumpy
- catboost
- pycatch22
- statsmodels
- ripser
- persim
