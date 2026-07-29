"""Windowed dataset with fold-local normalization (PLAN.md section 11).

The whole panel is small enough (~560k rows x 43 float32 features = 96 MB) to live
on the GPU. Windows are gathered by index at batch time instead of being
materialised, which would need ~19 GB.

Normalization is fit on training rows ONLY: clip limits at train quantiles, then
robust scaling by train median / IQR. Targets are scaled the same way and inverted
before any metric is computed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.features import FEATURE_SETS
from src.splits import assign_split

CLIP_Q = 0.001  # clip at train 0.1% / 99.9% quantiles


class Panel:
    """Immutable per-symbol arrays plus window-eligibility bookkeeping."""

    def __init__(self, panel: pd.DataFrame, feature_set: str, target: str = "y_return"):
        self.features = FEATURE_SETS[feature_set]
        self.target = target
        panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        self.panel = panel

        X = panel[self.features].to_numpy(np.float32)
        y = panel[target].to_numpy(np.float32)

        row_ok = np.isfinite(X).all(axis=1)
        tgt_ok = np.isfinite(y)

        # a window ending at i is eligible only if all L rows in it are finite and
        # belong to the same symbol
        L = C.CONTEXT_LEN
        sym_codes = panel["symbol"].astype("category").cat.codes.to_numpy()
        run_ok = _rolling_all(row_ok, L)
        same_sym = np.zeros(len(panel), bool)
        same_sym[L - 1:] = sym_codes[L - 1:] == sym_codes[: len(panel) - L + 1]

        self.eligible = run_ok & same_sym & tgt_ok
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        self.timestamp = panel["timestamp"].to_numpy()
        self.symbol = panel["symbol"].to_numpy()

    def fold_indices(self, fold: dict) -> dict[str, np.ndarray]:
        split = assign_split(self.panel["timestamp"], fold).to_numpy()
        return {s: np.flatnonzero(self.eligible & (split == s)) for s in ("train", "val", "test")}

    def window_indices(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        gap = pd.Timedelta(hours=C.HORIZON)
        ts = self.panel["timestamp"]
        m = (ts >= start) & (ts < end - gap)
        return np.flatnonzero(self.eligible & m.to_numpy())


def _rolling_all(mask: np.ndarray, L: int) -> np.ndarray:
    """True at i iff mask[i-L+1 .. i] are all True."""
    cs = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    out = np.zeros(len(mask), bool)
    out[L - 1:] = (cs[L:] - cs[: len(mask) - L + 1]) == L
    return out


class Normalizer:
    """Fit on train rows only, then frozen (PLAN.md section 11 steps 1-4)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
        # feature stats over the rows actually consumed by training windows
        rows = _covered_rows(train_idx, C.CONTEXT_LEN, len(X))
        Xt = X[rows]
        self.lo = np.quantile(Xt, CLIP_Q, axis=0).astype(np.float32)
        self.hi = np.quantile(Xt, 1 - CLIP_Q, axis=0).astype(np.float32)
        Xc = np.clip(Xt, self.lo, self.hi)
        self.med = np.median(Xc, axis=0).astype(np.float32)
        iqr = (np.quantile(Xc, 0.75, axis=0) - np.quantile(Xc, 0.25, axis=0)).astype(np.float32)
        self.scale = np.where(iqr < 1e-8, 1.0, iqr).astype(np.float32)

        yt = y[train_idx]
        self.y_med = np.float32(np.median(yt))
        y_iqr = np.float32(np.quantile(yt, 0.75) - np.quantile(yt, 0.25))
        self.y_scale = np.float32(max(y_iqr, 1e-8))

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return ((np.clip(X, self.lo, self.hi) - self.med) / self.scale).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_med) / self.y_scale).astype(np.float32)

    def inverse_y(self, z):
        return z * self.y_scale + self.y_med


def _covered_rows(idx: np.ndarray, L: int, n: int) -> np.ndarray:
    """Rows touched by any window ending at an index in idx."""
    covered = np.zeros(n, bool)
    for s in range(L):
        covered[idx - s] = True
    return np.flatnonzero(covered)


class GPUWindows:
    """Holds normalized features/targets on-device; yields windowed batches."""

    def __init__(self, X: np.ndarray, y: np.ndarray, device: torch.device):
        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.device = device
        self.offsets = torch.arange(-(C.CONTEXT_LEN - 1), 1, device=device)

    def gather(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        win = idx.unsqueeze(1) + self.offsets  # [B, L]
        return self.X[win], self.y[idx]

    def iterate(self, idx: np.ndarray, batch_size: int, shuffle: bool,
                generator: torch.Generator | None = None):
        t = torch.from_numpy(idx).to(self.device)
        order = torch.randperm(len(t), device=self.device, generator=generator) if shuffle \
            else torch.arange(len(t), device=self.device)
        for i in range(0, len(t), batch_size):
            yield self.gather(t[order[i: i + batch_size]])
