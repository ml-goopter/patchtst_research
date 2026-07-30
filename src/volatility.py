"""Competing models for realized 4h volatility (PLAN.md section 17).

Section 17 is the one place where PatchTST clears its reference by a wide margin
(pearson +0.46, QLIKE 1.24 vs 1.90). But that reference is a constant -- the same
mistake that made stage one look good until `src/baselines.py` put a tree beside it.
This does for volatility what baselines.py did for the mean and quantiles.py did
for the distribution: fits cheap CPU competitors on exactly the folds, features,
purging and train-only normalization the transformer used.

  rv4_raw        predict the trailing 4-candle realized volatility, unscaled.
                 Pure persistence: the same functional form as the target, one
                 horizon earlier, with nothing fitted at all.
  rv4_ols        the same signal with a train-fitted intercept and slope in log space
  har_ols        HAR-RV: OLS on log realized volatility at 4 / 24 / 168 candles
  gbm_vol_last   gradient boosting on the F2 features at time t only (35 features)
  gbm_vol_lags   the same over 10 log-spaced lags across the window (350 features)
  static_median  the fold's TRAINING-window median volatility, repeated

Everything is fitted on log volatility, scaled by `TargetNormalizer` exactly as the
multi-task head's volatility target is, and inverted with exp() to raw units before
any metric. That makes the point forecasts directly comparable: both families
regress the conditional centre of log volatility and exponentiate, so both report a
median-like forecast and neither gets a mean-vs-median advantage over the other.

`gbm_vol_lags` exists to answer the same question the return task asked and
answered negatively: is there information in the 256-candle context that the last
candle does not already carry?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.baselines import LAG_OFFSETS, _design
from src.datamodule import Normalizer, Panel, TargetNormalizer
from src.metrics import volatility_metrics
from src.quantiles import _fit_staged
from src.splits import make_folds

TARGET = "y_volatility"
HAR_FEATURES = ["realized_volatility_4", "realized_volatility_24", "realized_volatility_168"]
MAX_ITER = 400
# Six rows in the panel have four consecutive flat candles, so realized volatility
# is exactly zero. Persistence then forecasts zero and QLIKE -- a ratio of variances
# -- is unbounded: one such row took a fold's validation QLIKE to 1.6e7. Every
# prediction is clipped at the fold's train 0.1% quantile of the target, the same
# 0.1% clip convention datamodule.Normalizer applies to features, and applied to all
# models identically so the floor cannot favour one.
PRED_FLOOR_Q = 0.001


def _log(v: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(v, 0.0) + C.EPS)


def _ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.hstack([design, np.ones((len(design), 1))])
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _ols_predict(design: np.ndarray, w: np.ndarray) -> np.ndarray:
    return design @ w[:-1] + w[-1]


def run_fold(panel_df: pd.DataFrame, fold: dict, feature_set: str = "F2",
             tag: str = "volatility", verbose: bool = True) -> tuple[list[dict], pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    pan = Panel(panel_df, feature_set, target=TARGET)
    idxs = pan.fold_indices(fold)
    xnorm = Normalizer(pan.X, pan.y, idxs["train"])
    # log + robust scaling, fit on train rows only -- the same transform the
    # multi-task volatility head is trained through (datamodule.TargetNormalizer)
    tnorm = TargetNormalizer(pan.y[:, None], idxs["train"], [TARGET])
    Xs = xnorm.transform_X(pan.X)
    zs = tnorm.transform(pan.y[:, None])[:, 0].astype(np.float64)

    preds: dict[str, dict[str, np.ndarray]] = {}
    fitted: dict[str, dict] = {}
    t0 = time.time()

    # --- persistence: trailing realized volatility, raw and train-rescaled in log space
    if verbose:
        print("    persistence / HAR", flush=True)
    rv4 = pan.X[:, pan.features.index("realized_volatility_4")].astype(np.float64)
    preds["rv4_raw"] = {s: rv4[idxs[s]] for s in ("val", "test")}  # already in vol units

    lrv4 = {s: _log(rv4[idxs[s]])[:, None] for s in ("train", "val", "test")}
    w = _ols(lrv4["train"], zs[idxs["train"]])
    preds["rv4_ols"] = {s: tnorm.inverse(_ols_predict(lrv4[s], w), TARGET) for s in ("val", "test")}
    fitted["rv4_ols"] = {"slope_scaled": float(w[0]), "intercept_scaled": float(w[1])}

    # --- HAR-RV: the standard volatility benchmark, short / daily / weekly components
    har = {s: np.column_stack([_log(pan.X[idxs[s], pan.features.index(c)].astype(np.float64))
                               for c in HAR_FEATURES]) for s in ("train", "val", "test")}
    w = _ols(har["train"], zs[idxs["train"]])
    preds["har_ols"] = {s: tnorm.inverse(_ols_predict(har[s], w), TARGET) for s in ("val", "test")}
    fitted["har_ols"] = {"coefs_scaled": [float(x) for x in w], "features": HAR_FEATURES}

    # --- the fold's unconditional training-window median (no test data touched)
    med = float(np.median(pan.y[idxs["train"]]))
    preds["static_median"] = {s: np.full(len(idxs[s]), med) for s in ("val", "test")}
    fitted["static_median"] = {"train_median": med}

    # --- gradient boosting, at time t and over the same log-spaced lags
    for name, offs in (("gbm_vol_last", [0]), ("gbm_vol_lags", LAG_OFFSETS)):
        if verbose:
            print(f"    {name} ({len(offs)*Xs.shape[1]} features)", flush=True)
        D = {s: _design(Xs, idxs[s], offs) for s in ("train", "val", "test")}

        def make():
            return HistGradientBoostingRegressor(
                loss="squared_error", max_iter=MAX_ITER, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                early_stopping=False, random_state=0)

        # selected on validation MSE in the space the model is fitted in, so the
        # choice of stopping point cannot be tuned toward any one report metric
        best = _fit_staged(make, lambda m, X, y: float(np.mean((m.predict(X) - y) ** 2)),
                           D["train"], zs[idxs["train"]], D["val"], zs[idxs["val"]],
                           verbose, name)
        preds[name] = {s: tnorm.inverse(best["model"].predict(D[s]), TARGET)
                       for s in ("val", "test")}
        fitted[name] = {"n_iter": best["n_iter"], "val_mse_scaled": best["score"],
                        "n_features": len(offs) * Xs.shape[1]}

    # --- score everything in raw volatility units
    floor = float(np.quantile(pan.y[idxs["train"]], PRED_FLOOR_Q))
    results, frames = [], []
    for name, ps in preds.items():
        r = {"tag": tag, "model": name, "fold": fold["fold"], "feature_set": feature_set,
             "target": TARGET, "n_train": len(idxs["train"]), "n_val": len(idxs["val"]),
             "n_test": len(idxs["test"]), "pred_floor": floor, "fitted": fitted.get(name, {})}
        for split, p in ps.items():
            p = np.maximum(p, floor)
            actual = pan.y[idxs[split]].astype(np.float64)
            r[split] = volatility_metrics(p, actual)
            frames.append(pd.DataFrame({
                "timestamp": pan.timestamp[idxs[split]],
                "symbol": pan.symbol[idxs[split]],
                "pred_volatility": p,
                "actual_volatility": actual,
                "split": split, "fold": fold["fold"], "model": name,
                "feature_set": feature_set, "tag": tag,
            }))
        results.append(r)

    if verbose:
        print(f"    fold {fold['fold']} done in {(time.time()-t0)/60:.1f}m", flush=True)
    return results, pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="F2")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--tag", default="volatility")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    panel_df = pd.read_parquet(C.DATA_PROC / "panel.parquet")
    folds = {f["fold"]: f for f in make_folds(panel_df)}
    out_dir = C.RUNS / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_res, all_preds = [], []
    for fid in [int(x) for x in a.folds.split(",")]:
        print(f"\n=== fold {fid} ===", flush=True)
        res, pr = run_fold(panel_df, folds[fid], a.feature_set, a.tag, verbose=not a.quiet)
        all_res += res
        all_preds.append(pr)
        with open(out_dir / "summary.jsonl", "a") as fh:
            for r in res:
                fh.write(json.dumps(r, default=str) + "\n")

    pd.concat(all_preds, ignore_index=True).to_parquet(
        out_dir / "predictions.parquet", index=False)
    print(f"\nwrote {out_dir}/predictions.parquet and summary.jsonl")

    t = pd.DataFrame([{"model": r["model"], "fold": r["fold"],
                       "qlike": r["test"]["qlike"], "pearson": r["test"]["pearson"]}
                      for r in all_res])
    print("\ntest QLIKE by fold:")
    print(t.pivot(index="model", columns="fold", values="qlike").round(4).to_string())
    print("\ntest pearson by fold:")
    print(t.pivot(index="model", columns="fold", values="pearson").round(4).to_string())


if __name__ == "__main__":
    main()
