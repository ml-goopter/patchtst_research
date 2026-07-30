"""Quantile + probability baselines for the return distribution (PLAN.md 14-16).

STATUS.md step 2. Before spending ~33 GPU-hours on a Student-t PatchTST head, find
out what a gradient-boosted tree on a single candle already delivers for the
distributional outputs. This fits, on exactly the folds, features, purging and
train-only normalization stage one used:

  q{05..95}       HistGradientBoostingRegressor(loss="quantile") per quantile
  p_above_cost    HistGradientBoostingClassifier on 1[y > COST_THRESHOLD]

Quantiles are fit independently, so they can cross; they are sorted row-wise
afterwards, which is the standard rearrangement fix and can only reduce pinball
loss. The scaler is affine and increasing, so a quantile in scaled space inverts
to the same quantile in log-return units.

The classifier and the quantile grid give two independent estimates of
P(y > cost). Reporting both is the point: if they disagree, neither is trustworthy.
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
from src.datamodule import Normalizer, Panel
from src.splits import make_folds

# dense enough that the grid-approximated CRPS is not dominated by discretisation
QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
MAX_ITER, STEP, PATIENCE = 400, 25, 75


def _pinball(pred: np.ndarray, actual: np.ndarray, tau: float) -> float:
    d = actual - pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def _fit_staged(make_model, score, tr_X, tr_y, va_X, va_y, verbose: bool, what: str):
    """Grow the ensemble in STEP-iteration chunks, keep the best validation score."""
    m = make_model()
    best = {"score": np.inf, "n_iter": 0, "model": None}
    for n in range(STEP, MAX_ITER + 1, STEP):
        m.set_params(max_iter=n, warm_start=True)
        m.fit(tr_X, tr_y)
        s = score(m, va_X, va_y)
        if s < best["score"] - 1e-9:
            best = {"score": s, "n_iter": n, "model": m}
        elif n - best["n_iter"] >= PATIENCE:
            break
    if verbose:
        print(f"      {what:14s} n_iter={best['n_iter']:3d}  val={best['score']:.6f}", flush=True)
    return best


def run_fold(panel_df: pd.DataFrame, fold: dict, feature_set: str, offsets: list[int],
             tag: str, verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    pan = Panel(panel_df, feature_set)
    idxs = pan.fold_indices(fold)
    norm = Normalizer(pan.X, pan.y, idxs["train"])
    Xs, ys = norm.transform_X(pan.X), norm.transform_y(pan.y)

    D = {s: _design(Xs, idxs[s], offsets) for s in ("train", "val", "test")}
    Y = {s: ys[idxs[s]].astype(np.float64) for s in ("train", "val", "test")}
    t0 = time.time()

    # --- one regressor per quantile
    cols, fitted = {}, {}
    for tau in QUANTILES:
        def make(tau=tau):
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=tau, max_iter=MAX_ITER, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                early_stopping=False, random_state=0)

        best = _fit_staged(make, lambda m, X, y, tau=tau: _pinball(m.predict(X), y, tau),
                           D["train"], Y["train"], D["val"], Y["val"], verbose, f"q{tau:.2f}")
        name = f"q{int(round(tau * 100)):02d}"
        cols[name] = {s: best["model"].predict(D[s]) for s in ("val", "test")}
        fitted[name] = {"n_iter": best["n_iter"], "val_pinball_scaled": best["score"]}

    # --- classifier for P(y > cost); the threshold lives in raw units
    cost_s = float(norm.transform_y(np.array([C.COST_THRESHOLD], np.float32))[0])
    lab = {s: (Y[s] > cost_s).astype(int) for s in ("train", "val", "test")}

    def make_clf():
        return HistGradientBoostingClassifier(
            max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, random_state=0)

    def logloss(m, X, y):
        p = np.clip(m.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best = _fit_staged(make_clf, logloss, D["train"], lab["train"], D["val"], lab["val"],
                       verbose, "p_above_cost")
    fitted["p_above_cost"] = {"n_iter": best["n_iter"], "val_logloss": best["score"],
                              "train_base_rate": float(lab["train"].mean())}

    # --- assemble, inverting to raw log-return units
    frames = []
    qnames = [f"q{int(round(t * 100)):02d}" for t in QUANTILES]
    for split in ("val", "test"):
        q = np.column_stack([cols[n][split] for n in qnames])
        q = np.sort(q, axis=1)  # rearrangement: independent fits can cross
        q = norm.inverse_y(q)
        d = pd.DataFrame(q, columns=qnames)
        d["p_above_cost"] = best["model"].predict_proba(D[split])[:, 1]
        d.insert(0, "actual_log_return", norm.inverse_y(Y[split]))
        d.insert(0, "split", split)
        d.insert(0, "fold", fold["fold"])
        d.insert(0, "symbol", pan.symbol[idxs[split]])
        d.insert(0, "timestamp", pan.timestamp[idxs[split]])
        frames.append(d)

    res = {"tag": tag, "fold": fold["fold"], "feature_set": feature_set,
           "n_offsets": len(offsets), "quantiles": QUANTILES,
           "n_train": len(idxs["train"]), "n_val": len(idxs["val"]),
           "n_test": len(idxs["test"]), "fitted": fitted,
           "minutes": round((time.time() - t0) / 60, 2)}
    if verbose:
        print(f"    fold {fold['fold']} done in {res['minutes']:.1f}m", flush=True)
    return res, pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="F2")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--offsets", default="last", choices=["last", "lags"])
    ap.add_argument("--tag", default="quantiles")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    offsets = [0] if a.offsets == "last" else LAG_OFFSETS
    panel_df = pd.read_parquet(C.DATA_PROC / "panel.parquet")
    folds = {f["fold"]: f for f in make_folds(panel_df)}
    out_dir = C.RUNS / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    for fid in [int(x) for x in a.folds.split(",")]:
        print(f"\n=== fold {fid} ({a.offsets}) ===", flush=True)
        r, p = run_fold(panel_df, folds[fid], a.feature_set, offsets, a.tag,
                        verbose=not a.quiet)
        preds.append(p)
        with open(out_dir / "summary.jsonl", "a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")

    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet", index=False)
    print(f"\nwrote {out_dir}/predictions.parquet and summary.jsonl")


if __name__ == "__main__":
    main()
