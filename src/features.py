"""Causal feature engineering (PLAN.md sections 5-6) and target construction (8-9).

Every feature at index t uses only candles <= t. Every target at index t uses only
candles > t. `build_panel` asserts this holds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

EPS = C.EPS

# ------------------------------------------------------------------ groups
GROUP_A = ["close_return", "open_gap", "high_return", "low_return", "candle_body",
           "candle_range", "close_location", "upper_wick", "lower_wick"]
GROUP_B = ["return_2", "return_4", "return_8", "return_12", "return_24", "return_48", "return_168"]
GROUP_C = ["log_volume", "volume_change", "volume_zscore_24", "volume_zscore_72",
           "volume_zscore_168", "dollar_volume", "relative_volume"]
GROUP_D = ["realized_volatility_4", "realized_volatility_12", "realized_volatility_24",
           "realized_volatility_48", "realized_volatility_168", "average_range_12",
           "average_range_24", "average_range_168", "normalized_range"]
GROUP_E = ["return_x_volume_zscore", "range_x_volume_zscore", "signed_volume_change"]
GROUP_F = ["btc_return", "eth_return", "asset_minus_btc_return", "btc_realized_volatility"]
GROUP_G = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]

FEATURE_SETS = {
    "F0": ["close_return"],
    "F1": GROUP_A + GROUP_C,
    "F2": GROUP_A + GROUP_C + GROUP_B + GROUP_D + GROUP_E,
    "F3": GROUP_A + GROUP_C + GROUP_B + GROUP_D + GROUP_E + GROUP_F,
    "F4": GROUP_A + GROUP_C + GROUP_B + GROUP_D + GROUP_E + GROUP_F + GROUP_G,
}

ALL_FEATURES = FEATURE_SETS["F4"]

TARGETS = ["y_return", "y_volatility", "y_mae", "y_mfe", "y_above_cost", "y_regime"]


def _zscore(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std(ddof=0)
    return (s - m) / (sd + EPS)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol features. df must be on a strict hourly grid, sorted ascending."""
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    c1 = c.shift(1)
    out = pd.DataFrame(index=df.index)

    # --- group A: candle price structure
    out["close_return"] = np.log(c / c1)
    out["open_gap"] = np.log(o / c1)
    out["high_return"] = np.log(h / c1)
    out["low_return"] = np.log(l / c1)
    out["candle_body"] = np.log(c / o)
    out["candle_range"] = np.log(h / l)
    out["close_location"] = (c - l) / np.maximum(h - l, EPS)
    out["upper_wick"] = (h - np.maximum(o, c)) / c1
    out["lower_wick"] = (np.minimum(o, c) - l) / c1

    # --- group B: momentum
    for k in (2, 4, 8, 12, 24, 48, 168):
        out[f"return_{k}"] = np.log(c / c.shift(k))

    # --- group C: volume
    out["log_volume"] = np.log1p(v)
    out["volume_change"] = np.log((v + EPS) / (v.shift(1) + EPS))
    for w in (24, 72, 168):
        out[f"volume_zscore_{w}"] = _zscore(out["log_volume"], w)
    out["dollar_volume"] = np.log1p(c * v)
    out["relative_volume"] = np.log((v + EPS) / (v.rolling(24, min_periods=24).mean() + EPS))

    # --- group D: volatility
    r = out["close_return"]
    r2 = r.pow(2)
    for w in (4, 12, 24, 48, 168):
        out[f"realized_volatility_{w}"] = np.sqrt(r2.rolling(w, min_periods=w).sum())
    for w in (12, 24, 168):
        out[f"average_range_{w}"] = out["candle_range"].rolling(w, min_periods=w).mean()
    out["normalized_range"] = (h - l) / c

    # per-candle trailing vol, used by the frozen regime threshold (section 9)
    out["trailing_vol_per_candle"] = np.sqrt(r2.rolling(24, min_periods=24).mean())

    # --- group E: price-volume interaction
    vz = out["volume_zscore_24"]
    out["return_x_volume_zscore"] = r * vz
    out["range_x_volume_zscore"] = out["normalized_range"] * vz
    out["signed_volume_change"] = np.sign(r) * out["volume_change"].abs()

    # --- group G: calendar
    ts = df["timestamp"]
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    return out


def compute_targets(df: pd.DataFrame, trailing_vol: pd.Series) -> pd.DataFrame:
    """Forward-looking targets over PLAN.md's 4-candle horizon."""
    H = C.HORIZON
    c, h, l = df["close"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)

    out["y_return"] = np.log(c.shift(-H) / c)

    # future path of 1-step log returns: r_{t+1} .. r_{t+H}
    step = np.log(c / c.shift(1))
    fwd = [step.shift(-i) for i in range(1, H + 1)]
    fwd_sq = sum(x.pow(2) for x in fwd)
    fwd_abs = sum(x.abs() for x in fwd)
    out["y_volatility"] = np.sqrt(fwd_sq)

    out["y_mae"] = pd.concat([np.log(l.shift(-i) / c) for i in range(1, H + 1)], axis=1).min(axis=1)
    out["y_mfe"] = pd.concat([np.log(h.shift(-i) / c) for i in range(1, H + 1)], axis=1).max(axis=1)

    out["y_above_cost"] = (out["y_return"] > C.COST_THRESHOLD).astype(float)

    # --- regime (frozen definition, PLAN.md section 9)
    path_eff = out["y_return"].abs() / (fwd_abs + EPS)
    trend_thr = np.maximum(C.COST_THRESHOLD, 0.5 * trailing_vol * np.sqrt(H))
    trending = path_eff >= C.PATH_EFFICIENCY_MIN
    up = (out["y_return"] > trend_thr) & trending
    down = (out["y_return"] < -trend_thr) & trending
    regime = np.full(len(df), C.REGIME_CLASSES.index("sideways"), dtype=float)
    regime[up.to_numpy()] = C.REGIME_CLASSES.index("uptrend")
    regime[down.to_numpy()] = C.REGIME_CLASSES.index("downtrend")
    out["y_regime"] = regime
    out["path_efficiency"] = path_eff
    out["trend_threshold"] = trend_thr

    # any NaN in the forward window invalidates every target at t
    bad = out["y_return"].isna() | out["y_volatility"].isna() | out["y_mae"].isna() | out["y_mfe"].isna()
    out.loc[bad, TARGETS + ["path_efficiency", "trend_threshold"]] = np.nan
    return out


def build_panel() -> pd.DataFrame:
    """Load every symbol, attach features, benchmark context and targets."""
    raw = {s: pd.read_parquet(C.DATA_RAW / f"{s}.parquet") for s in C.SYMBOLS}

    # benchmark series, computed once on the benchmark symbols themselves
    bench = {}
    for b in C.BENCHMARK_SYMBOLS:
        bdf = raw[b]
        bc = bdf["close"]
        br = np.log(bc / bc.shift(1))
        bench[b] = pd.DataFrame({
            "timestamp": bdf["timestamp"],
            f"{b[:3].lower()}_return": br,
            f"{b[:3].lower()}_realized_volatility": np.sqrt(br.pow(2).rolling(24, min_periods=24).sum()),
        })

    frames = []
    for sym, df in raw.items():
        df = df.reset_index(drop=True)
        feats = compute_features(df)
        tgts = compute_targets(df, feats["trailing_vol_per_candle"])

        keep = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
        part = pd.concat([keep, feats, tgts], axis=1)
        for b in C.BENCHMARK_SYMBOLS:
            part = part.merge(bench[b], on="timestamp", how="left")
        part["asset_minus_btc_return"] = part["close_return"] - part["btc_return"]
        frames.append(part)

    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    return panel.reset_index(drop=True)


# ------------------------------------------------------------------- checks
def leakage_checks(panel: pd.DataFrame) -> None:
    """Recompute a few values by hand and assert the causal direction is right."""
    d = panel[panel.symbol == "BTCUSDT"].reset_index(drop=True)
    i = 40000
    c = d["close"]

    assert abs(d["close_return"][i] - np.log(c[i] / c[i - 1])) < 1e-12
    assert abs(d["return_24"][i] - np.log(c[i] / c[i - 24])) < 1e-12
    assert abs(d["y_return"][i] - np.log(c[i + C.HORIZON] / c[i])) < 1e-12

    fwd = [np.log(c[i + k] / c[i + k - 1]) for k in range(1, C.HORIZON + 1)]
    assert abs(d["y_volatility"][i] - np.sqrt(sum(x**2 for x in fwd))) < 1e-12
    assert abs(d["y_mae"][i] - min(np.log(d["low"][i + k] / c[i]) for k in range(1, C.HORIZON + 1))) < 1e-12
    assert abs(d["y_mfe"][i] - max(np.log(d["high"][i + k] / c[i]) for k in range(1, C.HORIZON + 1))) < 1e-12

    # a trailing feature must not change when the future is deleted
    trunc = d.iloc[: i + 1].copy()
    f2 = compute_features(trunc)
    for col in ["close_return", "return_168", "realized_volatility_168", "volume_zscore_168",
                "average_range_168", "relative_volume", "trailing_vol_per_candle"]:
        a, b = d[col][i], f2[col].iloc[i]
        assert np.isclose(a, b, rtol=1e-10, atol=1e-14), f"{col} leaks future info: {a} vs {b}"
    print("leakage checks passed")


if __name__ == "__main__":
    panel = build_panel()
    leakage_checks(panel)
    panel.to_parquet(C.DATA_PROC / "panel.parquet", index=False)
    print(f"panel: {len(panel):,} rows x {panel.shape[1]} cols")
    print(f"feature sets: " + ", ".join(f"{k}={len(v)}" for k, v in FEATURE_SETS.items()))
    valid = panel[ALL_FEATURES + TARGETS].notna().all(axis=1)
    print(f"rows with all F4 features + targets: {valid.sum():,} ({100*valid.mean():.1f}%)")
    print("\nregime distribution:")
    print(panel["y_regime"].map(dict(enumerate(C.REGIME_CLASSES))).value_counts(normalize=True).round(4))
    print(f"\nabove-cost base rate: {panel['y_above_cost'].mean():.4f}")
