"""Competing baselines for the expected-return task (PLAN.md section 4a).

The zero predictor in metrics.py is a sanity floor, not a competitor. These are the
models that tell you whether PatchTST is earning its parameters:

  momentum_raw   predict the trailing 4h return, unscaled (pure persistence)
  momentum_ols   the same signal with a train-fitted intercept and slope
  ridge_last     ridge on the F2 features at time t only (no history)
  ridge_lags     ridge on the F2 features at 10 log-spaced lags across the window
  gbm_last       gradient boosting on the F2 features at time t only

Every baseline sees the same folds, the same purged boundaries and the same
train-only normalization as the transformer, and is scored by the same metrics on
raw log-return units.
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
from src.datamodule import Normalizer, Panel
from src.metrics import decile_monotonicity, expected_return_metrics
from src.splits import make_folds

# log-spaced offsets back from the window end; 0 is time t, 255 the oldest candle
LAG_OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64, 128, C.CONTEXT_LEN - 1]
ALPHAS = np.logspace(-2, 10, 25)
CHUNK = 50_000


def _design(X: np.ndarray, idx: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Rows of X at idx-offset for each offset, concatenated. [n, len(offsets)*p]"""
    return np.concatenate([X[idx - o] for o in offsets], axis=1)


def _gram(X: np.ndarray, idx: np.ndarray, y: np.ndarray, offsets: list[int]) -> tuple:
    """Accumulate X'X and X'y in chunks so the full design never materialises."""
    p = X.shape[1] * len(offsets) + 1  # +1 intercept
    xtx = np.zeros((p, p))
    xty = np.zeros(p)
    for i in range(0, len(idx), CHUNK):
        sl = idx[i: i + CHUNK]
        d = _design(X, sl, offsets).astype(np.float64)
        d = np.hstack([d, np.ones((len(d), 1))])
        xtx += d.T @ d
        xty += d.T @ y[sl]
    return xtx, xty


def _ridge_fit(xtx, xty, alpha: float) -> np.ndarray:
    a = np.eye(len(xty)) * alpha
    a[-1, -1] = 0.0  # never penalise the intercept
    return np.linalg.solve(xtx + a, xty)


def _ridge_predict(X, idx, offsets, w) -> np.ndarray:
    out = np.empty(len(idx))
    for i in range(0, len(idx), CHUNK):
        sl = idx[i: i + CHUNK]
        d = _design(X, sl, offsets).astype(np.float64)
        out[i: i + CHUNK] = d @ w[:-1] + w[-1]
    return out


def _fit_ridge(Xs, idxs, ys, offsets, verbose):
    """Fit at every alpha, pick the one with the lowest validation MSE."""
    xtx, xty = _gram(Xs, idxs["train"], ys, offsets)
    best = {"mse": np.inf}
    for alpha in ALPHAS:
        w = _ridge_fit(xtx, xty, alpha)
        vp = _ridge_predict(Xs, idxs["val"], offsets, w)
        mse = float(np.mean((vp - ys[idxs["val"]]) ** 2))
        if mse < best["mse"]:
            best = {"mse": mse, "alpha": float(alpha), "w": w}
    if verbose:
        print(f"      alpha={best['alpha']:.4g}  val_mse={best['mse']:.5f}", flush=True)
    return best


def _fit_gbm(Xs, idxs, ys, offsets, verbose):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=0,
    )
    # explicit validation: fit on train, score on val each 25 iters via warm start
    tr = _design(Xs, idxs["train"], offsets)
    va, vy = _design(Xs, idxs["val"], offsets), ys[idxs["val"]]
    best = {"mae": np.inf, "n_iter": 0, "model": None}
    for n in range(25, 401, 25):
        m.set_params(max_iter=n, warm_start=True)
        m.fit(tr, ys[idxs["train"]])
        mae = float(np.mean(np.abs(m.predict(va) - vy)))
        if mae < best["mae"] - 1e-9:
            best = {"mae": mae, "n_iter": n, "model": m}
        elif n - best["n_iter"] >= 75:
            break
    if verbose:
        print(f"      n_iter={best['n_iter']}  val_mae={best['mae']:.5f}", flush=True)
    return best


def run_fold(panel_df: pd.DataFrame, fold: dict, feature_set: str = "F2",
             tag: str = "baselines", verbose: bool = True) -> tuple[list[dict], pd.DataFrame]:
    pan = Panel(panel_df, feature_set)
    idxs = pan.fold_indices(fold)
    norm = Normalizer(pan.X, pan.y, idxs["train"])
    Xs = norm.transform_X(pan.X)
    ys = norm.transform_y(pan.y)

    mom_col = pan.features.index("return_4")
    preds: dict[str, dict[str, np.ndarray]] = {}
    fitted: dict[str, dict] = {}
    t0 = time.time()

    # --- momentum: trailing 4h return, raw and train-rescaled
    if verbose:
        print("    momentum", flush=True)
    mom = {s: Xs[idxs[s], mom_col].astype(np.float64) for s in ("val", "test")}
    mom_tr = Xs[idxs["train"], mom_col].astype(np.float64)
    # raw persistence lives in log-return units, so undo the feature scaling only
    raw_scale = float(norm.scale[mom_col])
    raw_med = float(norm.med[mom_col])
    preds["momentum_raw"] = {s: norm.transform_y(mom[s] * raw_scale + raw_med) for s in mom}
    b = np.polyfit(mom_tr, ys[idxs["train"]], 1)
    preds["momentum_ols"] = {s: np.polyval(b, mom[s]) for s in mom}
    fitted["momentum_ols"] = {"slope_scaled": float(b[0]), "intercept_scaled": float(b[1])}

    # --- ridge at time t, and ridge over log-spaced lags
    for name, offs in (("ridge_last", [0]), ("ridge_lags", LAG_OFFSETS)):
        if verbose:
            print(f"    {name} ({len(offs)*Xs.shape[1]} features)", flush=True)
        best = _fit_ridge(Xs, idxs, ys, offs, verbose)
        preds[name] = {s: _ridge_predict(Xs, idxs[s], offs, best["w"]) for s in ("val", "test")}
        fitted[name] = {"alpha": best["alpha"], "n_features": len(offs) * Xs.shape[1]}

    # --- gradient boosting, at time t and over the same log-spaced lags
    for name, offs in (("gbm_last", [0]), ("gbm_lags", LAG_OFFSETS)):
        if verbose:
            print(f"    {name} ({len(offs)*Xs.shape[1]} features)", flush=True)
        g = _fit_gbm(Xs, idxs, ys, offs, verbose)
        preds[name] = {s: g["model"].predict(_design(Xs, idxs[s], offs)).astype(np.float64)
                       for s in ("val", "test")}
        fitted[name] = {"n_iter": g["n_iter"], "n_features": len(offs) * Xs.shape[1]}

    # --- score everything in raw log-return units
    results, frames = [], []
    for name, ps in preds.items():
        r = {"tag": tag, "model": name, "fold": fold["fold"], "feature_set": feature_set,
             "n_train": len(idxs["train"]), "n_val": len(idxs["val"]),
             "n_test": len(idxs["test"]), "fitted": fitted.get(name, {})}
        for split, p_s in ps.items():
            p, y = norm.inverse_y(p_s), norm.inverse_y(ys[idxs[split]])
            r[split] = expected_return_metrics(p, y, C.COST_THRESHOLD)
            r[split].update(decile_monotonicity(p, y))
            frames.append(pd.DataFrame({
                "timestamp": pan.timestamp[idxs[split]],
                "symbol": pan.symbol[idxs[split]],
                "predicted_log_return": p,
                "actual_log_return": y,
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
    ap.add_argument("--tag", default="baselines")
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
                       "pearson": r["test"]["pearson"], "r2_vs_zero": r["test"]["r2_vs_zero"],
                       "decile_spread": r["test"]["decile_spread"]} for r in all_res])
    print("\ntest pearson by fold:")
    print(t.pivot(index="model", columns="fold", values="pearson").round(4).to_string())


if __name__ == "__main__":
    main()
