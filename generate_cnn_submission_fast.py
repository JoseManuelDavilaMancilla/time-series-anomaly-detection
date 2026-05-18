"""
Fast CNN submission generator — lightweight variant for quick iteration.

- 1 seed (instead of 3)
- 5 epochs (instead of 20)
- Smaller batch size for faster convergence

Usage:
    uv run python generate_cnn_submission_fast.py -o submission_cnn_fast.json
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from validation import all_window_dirs, load_window

METRIC_TYPES = ("Count", "ErrorCount", "LatencySecond", "QPS",
                "ResourceUtilizationRate", "SuccessRate")


class SmallCNN(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)


def _make_windows(x, y, context=32):
    n = len(x)
    windows = np.empty((n, context), dtype=np.float32)
    x0 = float(x[0]) if len(x) > 0 else 0.0
    for i in range(n):
        start = max(0, i - context + 1)
        seg = x[start : i + 1].astype(np.float32, copy=False)
        if len(seg) < context:
            pad_len = context - len(seg)
            seg = np.concatenate([np.full(pad_len, x0, dtype=np.float32), seg])
        windows[i] = seg
    return windows, y.astype(np.int64)


def _normalize(windows):
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + 1e-9
    return (windows - means) / stds


def train_cnn(raw_list, labels_list, seed=0, epochs=5, batch_size=512, lr=1e-3):
    all_w, all_l = [], []
    for x, y in zip(raw_list, labels_list):
        if len(x) == 0 or len(y) == 0 or len(x) != len(y):
            continue
        w, l = _make_windows(x, y, 32)
        all_w.append(w)
        all_l.append(l)
    if not all_w:
        return None

    X = np.vstack(all_w)
    y = np.hstack(all_l)
    X = _normalize(X)

    device = torch.device("cpu")
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device).unsqueeze(1)

    pos_ratio = float(y_t.mean())
    pos_weight = float((1.0 - pos_ratio) / (pos_ratio + 1e-9))
    sample_weights = torch.where(y_t == 1.0, pos_weight, 1.0)

    model = SmallCNN(seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")

    n_samples = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            bx, by, bw = X_t[idx], y_t[idx], sample_weights[idx]
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss = (loss * bw).mean()
            loss.backward()
            optimizer.step()

    return model


def predict_cnn(model, x, batch_size=512):
    if model is None:
        return np.zeros(len(x), dtype=np.float32)
    n = len(x)
    windows = np.empty((n, 32), dtype=np.float32)
    x0 = float(x[0]) if n > 0 else 0.0
    for i in range(n):
        start = max(0, i - 32 + 1)
        seg = x[start : i + 1].astype(np.float32, copy=False)
        if len(seg) < 32:
            seg = np.concatenate([np.full(32 - len(seg), x0, dtype=np.float32), seg])
        windows[i] = seg
    windows = _normalize(windows)
    X_t = torch.from_numpy(windows).float()

    probs = []
    with torch.no_grad():
        model.eval()
        for i in range(0, n, batch_size):
            p = model(X_t[i : i + batch_size]).cpu().numpy().flatten()
            probs.append(p)
    return np.concatenate(probs).astype(np.float32)


def build_training_data(window_dirs, target_mt):
    raw_list, labels_list = [], []
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") != target_mt:
            continue
        try:
            train_y = np.load(wdir / "train_label.npy")
        except FileNotFoundError:
            continue
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        raw_list.append(train_x)
        labels_list.append(train_y)
    return raw_list, labels_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=Path("submission_cnn_fast.json"))
    args = parser.parse_args()

    window_dirs = all_window_dirs()
    print(f"Training fast CNNs for {len(METRIC_TYPES)} metric types...")

    cnn_models = {}
    for mt in METRIC_TYPES:
        raw_list, labels_list = build_training_data(window_dirs, mt)
        print(f"  [{mt}] {len(raw_list)} windows", end="", flush=True)
        if not raw_list:
            cnn_models[mt] = None
            print(" -> skip")
            continue
        t0 = time.time()
        model = train_cnn(raw_list, labels_list, seed=42, epochs=5, batch_size=512, lr=1e-3)
        print(f" -> {time.time()-t0:.1f}s")
        cnn_models[mt] = model

    print(f"\nGenerating predictions on 1000 test windows...")
    preds = {}
    t0 = time.time()
    for i, wdir in enumerate(window_dirs, 1):
        w = load_window(wdir)
        mt = w.info.get("metric_type", "Unknown")
        if mt not in cnn_models or cnn_models[mt] is None:
            preds[w.wid] = [0] * len(w.test_x)
            continue

        n = len(w.test_x)
        k = max(0, min(int(round(n * float(w.info.get("test set anomaly ratio", 0.0)))), n))
        if k == 0:
            preds[w.wid] = [0] * n
            continue

        combined = np.concatenate([w.train_x, w.test_x])
        cnn_probs = predict_cnn(cnn_models[mt], combined)
        test_probs = cnn_probs[len(w.train_x):]

        order = np.lexsort((np.arange(n), -test_probs))
        pred = np.zeros(n, dtype=int)
        pred[order[:k]] = 1
        preds[w.wid] = pred.tolist()

        if i % 100 == 0:
            print(f"  {i}/1000 ({time.time()-t0:.0f}s)")

    assert len(preds) == 1000
    args.output.write_text(
        json.dumps({"predictions": preds}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
