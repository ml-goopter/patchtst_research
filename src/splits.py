"""Time-based splitting (PLAN.md section 10).

Development = oldest 82%, final holdout = newest 18%, never touched until the end.
Within development: 5 expanding walk-forward folds, each with a 3-month validation
window and a 3-month test window. Every boundary is purged by HORIZON candles so no
training target can see into validation, and no validation target into test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def timeline_bounds(panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    t0, t1 = panel["timestamp"].min(), panel["timestamp"].max()
    span = t1 - t0
    dev_end = t0 + span * (1 - C.HOLDOUT_FRACTION)
    dev_end = dev_end.normalize().replace(day=1)  # snap to a month boundary
    return t0, dev_end, t1


def make_folds(panel: pd.DataFrame) -> list[dict]:
    """Expanding folds, oldest first. Fold i's test window sits immediately before
    fold i+1's validation window."""
    t0, dev_end, t1 = timeline_bounds(panel)
    mo = pd.DateOffset(months=1)
    folds = []
    for k in range(C.N_FOLDS):
        back = (C.N_FOLDS - 1 - k) * C.TEST_MONTHS
        test_end = dev_end - back * mo
        test_start = test_end - C.TEST_MONTHS * mo
        val_start = test_start - C.VAL_MONTHS * mo
        folds.append(dict(
            fold=k + 1,
            train_start=t0, train_end=val_start,
            val_start=val_start, val_end=test_start,
            test_start=test_start, test_end=test_end,
        ))
    return folds


def holdout_window(panel: pd.DataFrame) -> dict:
    _, dev_end, t1 = timeline_bounds(panel)
    return dict(fold="holdout", holdout_start=dev_end, holdout_end=t1 + pd.Timedelta(hours=1))


def assign_split(ts: pd.Series, fold: dict) -> pd.Series:
    """Label each timestamp train/val/test/purged for one fold.

    A sample at time t consumes candles up to t+HORIZON, so the last HORIZON hours
    of every window are purged rather than used.
    """
    gap = pd.Timedelta(hours=C.HORIZON)
    out = pd.Series("unused", index=ts.index, dtype=object)
    out[(ts >= fold["train_start"]) & (ts < fold["train_end"] - gap)] = "train"
    out[(ts >= fold["val_start"]) & (ts < fold["val_end"] - gap)] = "val"
    out[(ts >= fold["test_start"]) & (ts < fold["test_end"] - gap)] = "test"
    return out


if __name__ == "__main__":
    panel = pd.read_parquet(C.DATA_PROC / "panel.parquet", columns=["timestamp"])
    t0, dev_end, t1 = timeline_bounds(panel)
    print(f"timeline   {t0.date()} -> {t1.date()}")
    print(f"dev ends   {dev_end.date()}   holdout = {dev_end.date()} -> {t1.date()}\n")
    for f in make_folds(panel):
        print(f"fold {f['fold']}  train {f['train_start'].date()}..{f['train_end'].date()}  "
              f"val {f['val_start'].date()}..{f['val_end'].date()}  "
              f"test {f['test_start'].date()}..{f['test_end'].date()}")
