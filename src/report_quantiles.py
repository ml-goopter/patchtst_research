"""Return-distribution and probability validation (PLAN.md sections 15 and 16).

Scores the quantile/probability GBM from src/quantiles.py, and scores a static
reference beside it on identical rows. The static reference predicts the training
window's unconditional quantiles and base rate for every timestamp -- it is the
distributional analogue of the zero predictor. Coverage and Brier score are easy
to achieve without any conditional skill, so a number that the static reference
matches is not evidence of a working distribution model.

NLL needs a density, and independent quantile fits only give knots. The density
here is piecewise-uniform between knots with exponential tails outside q05/q95.
That is an assumption laid on top of the model, so NLL is reported as
assumption-dependent and CRPS/pinball/coverage carry the argument.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.quantiles import QUANTILES

QN = [f"q{int(round(t * 100)):02d}" for t in QUANTILES]
TAUS = np.array(QUANTILES)
# PLAN.md section 16 buckets, plus the sub-0.50 range the plan omits: the
# above-cost base rate is 0.46, so most predictions land below 0.50.
BUCKETS = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
           (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.0)]


def pinball(pred: np.ndarray, actual: np.ndarray, tau: float) -> float:
    d = actual - pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def crps_from_quantiles(Q: np.ndarray, y: np.ndarray) -> float:
    """CRPS = 2 * integral of pinball over tau. Trapezoid on the fitted grid.

    The grid stops at 0.05/0.95, so the tails are not integrated and this is a
    lower bound on the true CRPS. Both models are scored the same way, so the
    comparison between them is unaffected."""
    losses = np.array([pinball(Q[:, i], y, t) for i, t in enumerate(TAUS)])
    return float(2 * np.trapezoid(losses, TAUS))


def nll_piecewise(Q: np.ndarray, y: np.ndarray) -> float:
    """Piecewise-uniform density between quantile knots, exponential tails."""
    n, k = Q.shape
    dq = np.diff(Q, axis=1)
    dtau = np.diff(TAUS)
    dens = np.where(dq > 1e-12, dtau[None, :] / np.maximum(dq, 1e-12), 1e-12)  # [n, k-1]

    idx = np.clip(np.sum(y[:, None] >= Q, axis=1) - 1, -1, k - 1)
    out = np.empty(n)
    inside = (idx >= 0) & (idx <= k - 2)
    out[inside] = dens[np.arange(n)[inside], idx[inside]]

    # exponential tails with scale set by the outermost fitted band
    lo = idx < 0
    if lo.any():
        s = np.maximum(Q[lo, 1] - Q[lo, 0], 1e-12)
        out[lo] = TAUS[0] / s * np.exp(-(Q[lo, 0] - y[lo]) / s)
    hi = idx > k - 2
    if hi.any():
        s = np.maximum(Q[hi, -1] - Q[hi, -2], 1e-12)
        out[hi] = (1 - TAUS[-1]) / s * np.exp(-(y[hi] - Q[hi, -1]) / s)
    return float(-np.mean(np.log(np.maximum(out, 1e-300))))


def prob_above_from_quantiles(Q: np.ndarray, thr: float) -> np.ndarray:
    """Interpolate the fitted quantile function to get P(y > thr)."""
    out = np.empty(len(Q))
    for i in range(len(Q)):
        out[i] = 1.0 - np.interp(thr, Q[i], TAUS, left=0.0, right=1.0)
    return out


def dist_metrics(Q: np.ndarray, y: np.ndarray) -> dict:
    p10, p50, p90 = Q[:, QN.index("q10")], Q[:, QN.index("q50")], Q[:, QN.index("q90")]
    return {
        "n": len(y),
        "p10_coverage": float(np.mean(y <= p10)),
        "p50_coverage": float(np.mean(y <= p50)),
        "p90_coverage": float(np.mean(y <= p90)),
        "central80_coverage": float(np.mean((y >= p10) & (y <= p90))),
        "pinball_p10": pinball(p10, y, 0.10),
        "pinball_p50": pinball(p50, y, 0.50),
        "pinball_p90": pinball(p90, y, 0.90),
        "crps": crps_from_quantiles(Q, y),
        "nll": nll_piecewise(Q, y),
        "mean_interval_width": float(np.mean(p90 - p10)),
        "median_interval_width": float(np.median(p90 - p10)),
    }


def prob_metrics(p: np.ndarray, ev: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    out = {
        "n": len(ev),
        "base_rate": float(ev.mean()),
        "mean_pred": float(p.mean()),
        "brier": float(np.mean((p - ev) ** 2)),
        "log_loss": float(-np.mean(ev * np.log(p) + (1 - ev) * np.log(1 - p))),
    }
    # expected calibration error over 10 equal-count bins
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[-1] += 1e-9
    b = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    ece = 0.0
    for k in range(10):
        m = b == k
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - ev[m].mean())
    out["ece"] = float(ece)
    return out


def static_reference(preds: pd.DataFrame, panel: pd.DataFrame, folds) -> pd.DataFrame:
    """Training-window unconditional quantiles and base rate, per fold."""
    from src.splits import assign_split

    rows = []
    for f in folds:
        sp = assign_split(panel["timestamp"], f)
        ytr = panel.loc[(sp == "train").to_numpy(), "y_return"].dropna().to_numpy(float)
        q = np.quantile(ytr, TAUS)
        rate = float((ytr > C.COST_THRESHOLD).mean())
        sub = preds[preds.fold == f["fold"]]
        d = pd.DataFrame(np.tile(q, (len(sub), 1)), columns=QN, index=sub.index)
        d["p_above_cost"] = rate
        d["actual_log_return"] = sub.actual_log_return.to_numpy()
        d["fold"] = f["fold"]
        rows.append(d)
    return pd.concat(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="quantiles")
    ap.add_argument("--out", default="quantiles_report.txt")
    a = ap.parse_args()

    from src.splits import make_folds

    pr = pd.read_parquet(C.RUNS / a.tag / "predictions.parquet")
    pr = pr[pr.split == "test"].reset_index(drop=True)
    panel = pd.read_parquet(C.DATA_PROC / "panel.parquet", columns=["timestamp", "symbol", "y_return"])
    folds = make_folds(panel)
    st = static_reference(pr, panel, folds)

    models = {"gbm_quantile": pr, "static": st}
    out, rows = [], []
    P = out.append

    P("=" * 96)
    P("RETURN DISTRIBUTION + PROBABILITY VALIDATION (PLAN.md sections 15, 16)")
    P("=" * 96)
    P("")
    P("Quantile and probability GBM on the F2 features at time t only, same folds,")
    P("purging and train-only normalization as stage one. 'static' predicts the training")
    P("window's unconditional quantiles and base rate for every row -- the distributional")
    P("equivalent of the zero predictor. Test splits only.")
    P("")

    P("-" * 96)
    P("section 15 - quantile validation, pooled over 5 test windows")
    P("-" * 96)
    P(f"{'model':>14} {'n':>8} {'p10cov':>8} {'p50cov':>8} {'p90cov':>8} {'c80cov':>8} "
      f"{'CRPS':>9} {'NLL':>8} {'width':>9}")
    for name, d in models.items():
        Q = d[QN].to_numpy(float)
        y = d.actual_log_return.to_numpy(float)
        m = dist_metrics(Q, y)
        P(f"{name:>14} {m['n']:>8} {m['p10_coverage']:>8.4f} {m['p50_coverage']:>8.4f} "
          f"{m['p90_coverage']:>8.4f} {m['central80_coverage']:>8.4f} "
          f"{m['crps']:>9.6f} {m['nll']:>8.3f} {1e4*m['mean_interval_width']:>8.1f}b")
        rows.append({"model": name, "scope": "pooled", **m})
    P(f"{'target':>14} {'':>8} {0.10:>8.4f} {0.50:>8.4f} {0.90:>8.4f} {0.80:>8.4f}")
    P("")
    P("pinball loss (lower is better), pooled")
    P(f"{'model':>14} {'p10':>12} {'p50':>12} {'p90':>12}")
    for name, d in models.items():
        m = dist_metrics(d[QN].to_numpy(float), d.actual_log_return.to_numpy(float))
        P(f"{name:>14} {m['pinball_p10']:>12.6f} {m['pinball_p50']:>12.6f} "
          f"{m['pinball_p90']:>12.6f}")

    P("")
    P("central-80% coverage and mean interval width by fold")
    P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>16}" for k in range(1, 6)))
    for name, d in models.items():
        cells = []
        for k in range(1, 6):
            s = d[d.fold == k]
            m = dist_metrics(s[QN].to_numpy(float), s.actual_log_return.to_numpy(float))
            cells.append(f"{m['central80_coverage']:.3f}/{1e4*m['mean_interval_width']:.0f}b")
            rows.append({"model": name, "scope": f"fold{k}", **m})
        P(f"{name:>14} " + " ".join(f"{c:>16}" for c in cells))

    # ------------------------------------------------------------- section 16
    P("")
    P("-" * 96)
    P("section 16 - probability validation, P(return > cost), pooled")
    P("-" * 96)
    ev = (pr.actual_log_return.to_numpy(float) > C.COST_THRESHOLD).astype(int)
    q_derived = prob_above_from_quantiles(pr[QN].to_numpy(float), C.COST_THRESHOLD)
    probs = {
        "gbm_classifier": pr.p_above_cost.to_numpy(float),
        "gbm_from_quantiles": q_derived,
        "static": st.p_above_cost.to_numpy(float),
    }
    P(f"{'model':>20} {'n':>8} {'base':>7} {'mean_p':>7} {'brier':>9} {'logloss':>9} {'ECE':>8}")
    for name, p in probs.items():
        m = prob_metrics(p, ev)
        P(f"{name:>20} {m['n']:>8} {m['base_rate']:>7.4f} {m['mean_pred']:>7.4f} "
          f"{m['brier']:>9.5f} {m['log_loss']:>9.5f} {m['ece']:>8.5f}")
        rows.append({"model": name, "scope": "prob_pooled", **m})

    P("")
    P("calibration of the classifier (PLAN.md section 16 buckets, extended below 0.50)")
    P(f"{'bucket':>14} {'n':>8} {'mean_pred':>10} {'actual':>8} {'gap':>8}")
    p = probs["gbm_classifier"]
    for lo, hi in BUCKETS:
        m = (p >= lo) & (p < hi)
        if m.sum() < 30:
            P(f"{f'{lo:.2f}-{hi:.2f}':>14} {int(m.sum()):>8} {'-':>10} {'-':>8} {'-':>8}")
            continue
        mp, ac = float(p[m].mean()), float(ev[m].mean())
        P(f"{f'{lo:.2f}-{hi:.2f}':>14} {int(m.sum()):>8} {mp:>10.4f} {ac:>8.4f} {ac-mp:>+8.4f}")
        rows.append({"model": "gbm_classifier", "scope": f"bucket_{lo:.2f}_{hi:.2f}",
                     "n": int(m.sum()), "mean_pred": mp, "actual": ac})

    P("")
    P(f"cost threshold = {C.COST_THRESHOLD} log-return units.")
    P("A static model matching the conditional one on a metric means that metric carries")
    P("no evidence of conditional skill.")

    txt = "\n".join(out)
    print(txt)
    (C.REPORTS / a.out).write_text(txt + "\n")
    pd.DataFrame(rows).to_csv(C.REPORTS / "quantiles_metrics.csv", index=False)
    print(f"\nwrote {C.REPORTS}/{a.out} and quantiles_metrics.csv")


if __name__ == "__main__":
    main()
