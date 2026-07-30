"""Expected-return evaluation (PLAN.md section 13).

Everything operates on raw log-return units, never on scaled values.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def huber(err: np.ndarray, delta: float) -> float:
    a = np.abs(err)
    return float(np.mean(np.where(a <= delta, 0.5 * err**2, delta * (a - 0.5 * delta))))


def expected_return_metrics(pred: np.ndarray, actual: np.ndarray, cost: float) -> dict:
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    err = pred - actual
    n = len(pred)

    out = {
        "n": n,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "huber": huber(err, delta=float(np.std(actual))),
        "pred_std": float(np.std(pred)),
        "actual_std": float(np.std(actual)),
        "prediction_bias": float(np.mean(err)),
    }

    if np.std(pred) < 1e-14:
        out["pearson"] = out["spearman"] = 0.0
        out["pearson_p"] = out["spearman_p"] = 1.0
    else:
        r, rp = stats.pearsonr(pred, actual)
        s, sp = stats.spearmanr(pred, actual)
        out.update(pearson=float(r), pearson_p=float(rp),
                   spearman=float(s), spearman_p=float(sp))

    # directional accuracy, all predictions and economically meaningful ones only
    nz = np.sign(pred) != 0
    out["directional_accuracy"] = float(np.mean(np.sign(pred[nz]) == np.sign(actual[nz]))) if nz.any() else np.nan
    big = np.abs(pred) > cost
    out["n_above_cost"] = int(big.sum())
    out["directional_accuracy_above_cost"] = (
        float(np.mean(np.sign(pred[big]) == np.sign(actual[big]))) if big.sum() >= 30 else np.nan
    )

    # calibration: actual = intercept + slope * predicted
    if np.std(pred) > 1e-14:
        slope, intercept, *_ = stats.linregress(pred, actual)
        out["calibration_slope"] = float(slope)
        out["calibration_intercept"] = float(intercept)
    else:
        out["calibration_slope"] = np.nan
        out["calibration_intercept"] = np.nan

    # sanity reference: always predict zero (PLAN.md section 4)
    out["baseline_zero_mae"] = float(np.mean(np.abs(actual)))
    out["baseline_zero_rmse"] = float(np.sqrt(np.mean(actual**2)))
    out["rmse_vs_zero_ratio"] = out["rmse"] / out["baseline_zero_rmse"]
    # fraction of realized variance explained relative to the zero predictor
    out["r2_vs_zero"] = 1.0 - float(np.mean(err**2)) / float(np.mean(actual**2))
    return out


def qlike(actual: np.ndarray, pred: np.ndarray) -> float:
    """QLIKE on variance; robust to noise in the volatility proxy.

    Penalises proportional error, so it is not dominated by the few extreme
    realizations that RMSE is. Zero for a perfect forecast, positive otherwise.
    """
    va = np.maximum(np.asarray(actual, float), 1e-8) ** 2
    vp = np.maximum(np.asarray(pred, float), 1e-8) ** 2
    r = va / vp
    return float(np.mean(r - np.log(r) - 1.0))


def volatility_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Point-forecast accuracy for realized volatility (PLAN.md section 17).

    Both a level view (MAE/RMSE, in log-return units) and a proportional view
    (RMSE on logs, QLIKE), because a forecast can win one and lose the other:
    volatility spans two orders of magnitude, so RMSE is decided almost entirely
    by the extreme quartile while QLIKE weights a 2x error the same everywhere.
    """
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    err = pred - actual
    lp = np.log(np.maximum(pred, 1e-8))
    la = np.log(np.maximum(actual, 1e-8))
    out = {
        "n": len(pred),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rmse_log": float(np.sqrt(np.mean((lp - la) ** 2))),
        "qlike": qlike(actual, pred),
        "bias": float(np.mean(err)),
        "mean_pred": float(np.mean(pred)),
        "mean_actual": float(np.mean(actual)),
    }
    if np.std(pred) < 1e-14:  # a constant forecast has no rank information
        out["pearson"] = out["spearman"] = np.nan
    else:
        out["pearson"] = float(stats.pearsonr(pred, actual).statistic)
        out["spearman"] = float(stats.spearmanr(pred, actual).statistic)
    return out


def decile_monotonicity(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> dict:
    """Do larger predicted returns correspond to larger realized returns?
    (PLAN.md section 23, expected-return criterion 4)"""
    if len(pred) < n_bins * 20 or np.std(pred) < 1e-14:
        return {"deciles": [], "decile_spread": np.nan, "decile_spearman": np.nan}
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    b = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "n": int(m.sum()),
                     "mean_pred": float(pred[m].mean()),
                     "mean_actual": float(actual[m].mean())})
    means = np.array([r["mean_actual"] for r in rows])
    rho = stats.spearmanr(np.arange(n_bins), means).statistic
    return {"deciles": rows, "decile_spread": float(means[-1] - means[0]),
            "decile_spearman": float(rho)}
