"""
cnn_addon.py — Lightweight 1D CNN ensemble add-on for v71 pipeline.

Trains 3 small 1D CNNs (different seeds) on per-metric-type pooled raw time-series data.
Input: sliding windows of length 32 over the raw time series.
Output: averaged anomaly probability across the 3 seeds.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


class SmallCNN(nn.Module):
    """Lightweight 1D CNN: ~465 parameters.

    Architecture:
        Conv1d(1, 16, kernel=3) → ReLU → Conv1d(16, 8, kernel=3) → ReLU
        → GlobalAvgPool1d → Linear(8, 1) → Sigmoid

    Input:  (batch, 32)  — sliding window of raw time series
    Output: (batch, 1)   — anomaly probability
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pool over time dim
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 32) -> (batch, 1, 32)
        x = x.unsqueeze(1)
        return self.net(x)


def _make_windows(x: np.ndarray, y: np.ndarray, context: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of length `context` from raw series x.

    For point i, window = x[max(0, i-context+1):i+1], left-padded with x[0]
    if i < context-1.  Label = y[i].

    Returns:
        windows: (n, context) float32
        labels:  (n,) int64
    """
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


def _normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Per-window z-score normalization."""
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + 1e-9
    return (windows - means) / stds


class CNNEnsemble:
    """Trains 3 SmallCNNs on per-metric-type pooled raw time-series data."""

    def __init__(
        self,
        context: int = 32,
        n_seeds: int = 3,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.context = context
        self.n_seeds = n_seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.models: List[SmallCNN] = []

    def fit(self, raw_series_list: List[np.ndarray], labels_list: List[np.ndarray]) -> "CNNEnsemble":
        """Fit on pooled raw time series.

        Args:
            raw_series_list: list of 1D arrays (train_x for each window)
            labels_list:     list of 1D arrays (train_y for each window)
        """
        all_windows, all_labels = [], []
        for x, y in zip(raw_series_list, labels_list):
            if len(x) == 0 or len(y) == 0 or len(x) != len(y):
                continue
            w, l = _make_windows(x, y, self.context)
            all_windows.append(w)
            all_labels.append(l)

        if not all_windows:
            self.models = []
            return self

        X = np.vstack(all_windows)
        y = np.hstack(all_labels)
        X = _normalize_windows(X)

        # Convert to tensors
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).float().to(self.device).unsqueeze(1)

        # Class-balanced weight
        pos_ratio = float(y_t.mean())
        pos_weight = float((1.0 - pos_ratio) / (pos_ratio + 1e-9))
        sample_weights = torch.where(y_t == 1.0, pos_weight, 1.0)

        n_samples = len(X_t)
        self.models = []
        for seed in range(self.n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SmallCNN(seed=seed).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.BCELoss(reduction="none")

            for epoch in range(self.epochs):
                perm = torch.randperm(n_samples)
                epoch_loss = 0.0
                for i in range(0, n_samples, self.batch_size):
                    batch_idx = perm[i : i + self.batch_size]
                    batch_x = X_t[batch_idx]
                    batch_y = y_t[batch_idx]
                    batch_w = sample_weights[batch_idx]

                    optimizer.zero_grad()
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    loss = (loss * batch_w).mean()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(batch_x)

            self.models.append(model)

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities for each point in x.

        Args:
            x: 1D array of raw time series values

        Returns:
            probs: 1D array of probabilities, same length as x
        """
        if not self.models:
            return np.zeros(len(x), dtype=np.float32)

        n = len(x)
        windows = np.empty((n, self.context), dtype=np.float32)
        x0 = float(x[0]) if n > 0 else 0.0
        for i in range(n):
            start = max(0, i - self.context + 1)
            seg = x[start : i + 1].astype(np.float32, copy=False)
            if len(seg) < self.context:
                pad_len = self.context - len(seg)
                seg = np.concatenate([np.full(pad_len, x0, dtype=np.float32), seg])
            windows[i] = seg

        windows = _normalize_windows(windows)
        X_t = torch.from_numpy(windows).float().to(self.device)

        all_probs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                probs = []
                for i in range(0, n, self.batch_size):
                    batch = X_t[i : i + self.batch_size]
                    p = model(batch).cpu().numpy().flatten()
                    probs.append(p)
                all_probs.append(np.concatenate(probs))

        return np.mean(all_probs, axis=0).astype(np.float32)
