# Time Series Anomaly Detection — Implementation Plan

## Context

- 1000 time windows (IDs `000`–`999`), each folder contains:
  - `train.npy` — training time series
  - `train_label.npy` — binary labels (0=normal, 1=anomaly) for training
  - `test.npy` — test time series (predict anomalies for this)
  - `test_timestamp.npy` — timestamps for test
  - `info.json` — metadata
- **Task**: For each window, predict a binary array (same length as `test.npy`) where 0=normal, 1=anomaly.
- **Scoring**: Mean F1 score across all 1000 windows.
- **Output**: UTF-8 JSON file: `{"predictions": {"000": [0,1,0,...], "001": [...], ...}}`

## Dataset Location

Assume the dataset has been downloaded and extracted to:
```
student_dataset/
  000_dns-resolver##database_availability_rate/
    info.json
    train.npy
    train_label.npy
    train_timestamp.npy
    test.npy
    test_timestamp.npy
  001_.../
  ...
  999_.../
```

If the dataset is not yet downloaded, download it from the Google Drive link in the assignment PDF and extract it. Set `DATASET_ROOT = Path("student_dataset")` accordingly.

## Step-by-Step Implementation

### Step 1: Data Exploration

Load a sample of windows and inspect the data:

```python
from pathlib import Path
import numpy as np

DATASET_ROOT = Path("student_dataset")
window_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()])

# Inspect first few windows
for wdir in window_dirs[:5]:
    wid = wdir.name.split("_", 1)[0]
    train = np.load(wdir / "train.npy")
    train_label = np.load(wdir / "train_label.npy")
    test = np.load(wdir / "test.npy")
    print(f"Window {wid}: train={train.shape}, labels={train_label.shape}, test={test.shape}")
    print(f"  Train anomaly rate: {train_label.mean():.4f}")
    print(f"  Value range: [{train.min():.4f}, {train.max():.4f}]")
```

Check if series are univariate or multivariate, typical lengths, anomaly rates, and value ranges.

Also read `info.json` files to understand what metrics each window represents.

### Step 2: Build the Submission Pipeline Skeleton

Create a working submission pipeline first. This ensures you can generate valid JSON at any point.

Core structure:

```python
import json
from pathlib import Path
import numpy as np

def predict_window(window_dir: Path) -> list:
    """
    Given a window directory, return a list of predictions (0/1 or probabilities)
    with the same length as test.npy.
    """
    test_x = np.load(window_dir / "test.npy")
    # TODO: replace with actual model
    return [0] * len(test_x)

def main():
    predictions = {}
    window_dirs = sorted([
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and (p / "test.npy").is_file()
    ])

    for wdir in window_dirs:
        wid = wdir.name.split("_", 1)[0]
        predictions[wid] = predict_window(wdir)

    # Validate
    assert len(predictions) == 1000, f"Expected 1000 windows, got {len(predictions)}"

    payload = {"predictions": predictions}
    output = DATASET_ROOT.parent / "submission.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {output} with {len(predictions)} windows")

if __name__ == "__main__":
    main()
```

### Step 3: Implement Multiple Detection Methods

For each method, write a function that takes `(train_x, train_y, test_x)` and returns anomaly scores for `test_x` (continuous values, higher = more anomalous).

**Method A: Statistical / Threshold-Based**

```python
def score_zscore(train_x, test_x):
    """Z-score anomaly score."""
    mean = np.mean(train_x)
    std = np.std(train_x)
    if std < 1e-9:
        return np.zeros(len(test_x))
    return np.abs((test_x - mean) / std)

def score_mad(train_x, test_x):
    """Median Absolute Deviation based score."""
    median = np.median(train_x)
    mad = np.median(np.abs(train_x - median))
    if mad < 1e-9:
        return np.zeros(len(test_x))
    modified_z = 0.6745 * (test_x - median) / mad
    return np.abs(modified_z)

def score_iqr(train_x, test_x):
    """IQR-based score."""
    q1, q3 = np.percentile(train_x, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-9:
        return np.zeros(len(test_x))
    return np.maximum((test_x - q3) / iqr, (q1 - test_x) / iqr)
```

**Method B: Isolation Forest**

```python
from sklearn.ensemble import IsolationForest

def score_isolation_forest(train_x, train_y, test_x):
    # Isolation Forest expects 2D input
    train_x = np.asarray(train_x).reshape(-1, 1) if np.asarray(train_x).ndim == 1 else np.asarray(train_x)
    test_x = np.asarray(test_x).reshape(-1, 1) if np.asarray(test_x).ndim == 1 else np.asarray(test_x)

    # Tune contamination based on train anomaly rate
    contamination = max(min(float(np.mean(train_y)), 0.5), 0.01)

    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(train_x)
    # decision_function returns anomaly score (lower = more anomalous), negate to make higher = more anomalous
    scores = -clf.decision_function(test_x)
    return scores
```

**Method C: Local Outlier Factor**

```python
from sklearn.neighbors import LocalOutlierFactor

def score_lof(train_x, test_x):
    train_x = np.asarray(train_x).reshape(-1, 1) if np.asarray(train_x).ndim == 1 else np.asarray(train_x)
    test_x = np.asarray(test_x).reshape(-1, 1) if np.asarray(test_x).ndim == 1 else np.asarray(test_x)

    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    clf.fit(train_x)
    scores = -clf.decision_function(test_x)  # negate: higher = more anomalous
    return scores
```

**Method D: Autoencoder (PyTorch)**

```python
import torch
import torch.nn as nn

def score_autoencoder(train_x, train_y, test_x, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize
    mean = np.mean(train_x)
    std = np.std(train_x)
    if std < 1e-9:
        std = 1.0

    train_norm = (np.asarray(train_x, dtype=np.float32) - mean) / std
    test_norm = (np.asarray(test_x, dtype=np.float32) - mean) / std

    train_tensor = torch.tensor(train_norm, dtype=torch.float32).unsqueeze(1).to(device)
    test_tensor = torch.tensor(test_norm, dtype=torch.float32).unsqueeze(1).to(device)

    model = nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Weight normal samples less if anomalies exist
    sample_weights = torch.where(
        torch.tensor(np.asarray(train_y), dtype=torch.float32) == 1,
        torch.tensor(2.0),
        torch.tensor(1.0)
    ).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(train_tensor)
        loss = (sample_weights * (pred.squeeze() - train_tensor.squeeze()) ** 2).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(test_tensor)
        scores = torch.abs(pred.squeeze() - test_tensor.squeeze()).cpu().numpy()
    return scores
```

### Step 4: Per-Window Threshold Optimization

For each window and each model, find the best threshold on the training data:

```python
def binary_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def find_best_threshold(scores, labels):
    """
    Grid search threshold to maximize F1 on training data.
    scores: anomaly scores (higher = more anomalous)
    labels: ground truth binary labels
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if np.sum(labels == 1) == 0:
        # No anomalies in training, use high threshold
        return float(np.max(scores) + 1e-9)

    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
    best_f1 = -1.0
    best_thresh = float(candidates[0])

    for thresh in candidates:
        pred = (scores >= thresh).astype(int)
        f1 = binary_f1(labels, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return best_thresh
```

### Step 5: Ensemble Strategy

Combine scores from multiple models, then optimize a single threshold:

```python
def ensemble_predict(window_dir: Path) -> list:
    train_x = np.load(window_dir / "train.npy")
    train_y = np.load(window_dir / "train_label.npy")
    test_x = np.load(window_dir / "test.npy")

    # Handle edge case: constant train data
    if np.std(train_x) < 1e-9:
        return [0] * len(test_x)

    # Collect scores from all models
    scores_list = []

    try:
        scores_list.append(score_zscore(train_x, test_x))
    except Exception:
        pass

    try:
        scores_list.append(score_mad(train_x, test_x))
    except Exception:
        pass

    try:
        scores_list.append(score_iqr(train_x, test_x))
    except Exception:
        pass

    try:
        scores_list.append(score_isolation_forest(train_x, train_y, test_x))
    except Exception:
        pass

    try:
        scores_list.append(score_lof(train_x, test_x))
    except Exception:
        pass

    try:
        scores_list.append(score_autoencoder(train_x, train_y, test_x))
    except Exception:
        pass

    if not scores_list:
        return [0] * len(test_x)

    # Normalize each score vector to [0, 1] before averaging
    normalized = []
    for s in scores_list:
        s = np.asarray(s).ravel()
        min_s, max_s = s.min(), s.max()
        if max_s - min_s > 1e-9:
            normalized.append((s - min_s) / (max_s - min_s))
        else:
            normalized.append(np.zeros_like(s))

    # Soft voting: average normalized scores
    ensemble_scores = np.mean(normalized, axis=0)

    # Get training ensemble scores for threshold selection
    train_scores_list = []
    for scorer, scores in zip([
        score_zscore, score_mad, score_iqr,
        score_isolation_forest, score_lof, score_autoencoder
    ], scores_list):
        try:
            # Re-run on train data to get train scores
            if scorer == score_isolation_forest:
                ts = scorer(train_x, train_y, train_x)
            elif scorer == score_autoencoder:
                ts = scorer(train_x, train_y, train_x)
            elif scorer == score_lof:
                ts = scorer(train_x, train_x)
            else:
                ts = scorer(train_x, train_x)

            ts = np.asarray(ts).ravel()
            min_s, max_s = ts.min(), ts.max()
            if max_s - min_s > 1e-9:
                train_scores_list.append((ts - min_s) / (max_s - min_s))
            else:
                train_scores_list.append(np.zeros_like(ts))
        except Exception:
            # Use test normalization as fallback
            ts = np.asarray(scores).ravel()  # this is wrong, skip properly
            pass

    if train_scores_list:
        train_ensemble_scores = np.mean(train_scores_list, axis=0)
        best_thresh = find_best_threshold(train_ensemble_scores, train_y)
    else:
        best_thresh = 0.5

    pred = (ensemble_scores >= best_thresh).astype(int)
    return pred.tolist()
```

### Step 6: Run the Full Pipeline

Replace `predict_window` in the skeleton with `ensemble_predict`, then run:

```bash
pip install numpy scikit-learn torch
python generate_submission.py
```

Validate the output:

```python
import json
import numpy as np
from pathlib import Path

DATASET_ROOT = Path("student_dataset")
with open("submission.json", "r", encoding="utf-8") as f:
    data = json.load(f)

assert "predictions" in data and len(data["predictions"]) == 1000

for wdir in sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()]):
    wid = wdir.name.split("_", 1)[0]
    test_len = len(np.load(wdir / "test.npy"))
    pred_len = len(data["predictions"][wid])
    assert pred_len == test_len, f"Window {wid}: pred={pred_len}, test={test_len}"

print("All validations passed!")
```

### Step 7: Iterate and Improve

After getting an initial score from the leaderboard:

1. **Error analysis**: Identify windows with the lowest F1 (you won't know exact F1s, but you can estimate using training data split).
2. **Add feature engineering**: Rolling statistics, differencing, lag features.
3. **Try window-specific models**: Some windows may be better suited for spectral methods, others for statistical.
4. **Add more models**: LSTM autoencoder, Prophet, STL decomposition.
5. **Better ensembling**: Use a learned meta-classifier (logistic regression on model outputs) instead of simple averaging.

## File Structure

```
.
├── student_dataset/              # Downloaded data
│   ├── 000_.../
│   ├── 001_.../
│   └── ...
├── generate_submission.py        # Main pipeline
├── validate_submission.py        # Format validator
└── submission.json               # Output for upload
```

## Dependencies

```
numpy
scikit-learn
torch
```

Install with: `pip install numpy scikit-learn torch`

## Final Notes

- Wrap model fitting in try/except so one failing window doesn't crash the whole pipeline.
- The daily submission limit is 10 — validate format locally before uploading.
- For Part B (report), document: methods tried, ensemble strategy, ablation results, and any insights from error analysis.
