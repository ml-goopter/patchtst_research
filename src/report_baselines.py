"""Compare PatchTST against the competing baselines (PLAN.md section 4a).

PatchTST is represented by its seed-averaged pooled test predictions, which is the
form it passed section 23 in; each baseline is deterministic so it needs no averaging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import decile_monotonicity, expected_return_metrics

KEYS = ["pearson", "spearman", "r2_vs_zero", "rmse_vs_zero_ratio", "calibration_slope",
        "directional_accuracy", "decile_spread", "decile_spearman"]


def _score(df: pd.DataFrame) -> dict:
    p = df.predicted_log_return.to_numpy(float)
    y = df.actual_log_return.to_numpy(float)
    r = expected_return_metrics(p, y, C.COST_THRESHOLD)
    r.update(decile_monotonicity(p, y))
    return r


def _top_decile_net(df: pd.DataFrame) -> float:
    """Mean realized return of the top predicted decile, minus round-trip cost."""
    p = df.predicted_log_return.to_numpy(float)
    cut = np.quantile(p, 0.9)
    return float(df.actual_log_return.to_numpy(float)[p >= cut].mean() - C.COST_THRESHOLD)


def main() -> None:
    base = pd.read_parquet(C.RUNS / "baselines" / "predictions.parquet")
    base = base[base.split == "test"]
    tst = pd.read_parquet(C.REPORTS / "stage1_pooled_predictions.parquet")
    tst["model"] = "patchtst"

    frames = [base[["model", "fold", "symbol", "timestamp",
                    "predicted_log_return", "actual_log_return"]], tst]
    allp = pd.concat(frames, ignore_index=True)

    out, rows = [], []
    P = out.append

    P("=" * 92)
    P("BASELINE COMPARISON - expected 4h log return, test splits only (PLAN.md section 4a)")
    P("=" * 92)

    models = ["patchtst"] + sorted(base.model.unique())
    n_tst = len(tst)

    P("")
    P("pooled over all 5 test windows")
    P(f"{'model':>14} {'n':>8} {'pearson':>9} {'spearman':>9} {'R2vs0':>9} {'dir%':>7} "
      f"{'decile_sp':>10} {'top10_net':>10} {'calib':>7}")
    for m in models:
        d = allp[allp.model == m]
        r = _score(d)
        net = _top_decile_net(d)
        P(f"{m:>14} {len(d):>8} {r['pearson']:>+9.4f} {r['spearman']:>+9.4f} "
          f"{r['r2_vs_zero']:>+9.5f} {100*r['directional_accuracy']:>7.2f} "
          f"{1e4*r['decile_spread']:>9.1f}b {1e4*net:>9.1f}b {r['calibration_slope']:>7.3f}")
        rows.append({"model": m, "scope": "pooled", "n": len(d),
                     "top_decile_net_bps": 1e4 * net, **{k: r[k] for k in KEYS}})

    P("")
    P("test pearson by fold")
    P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>9}" for k in range(1, 6)) + f" {'mean':>9}")
    for m in models:
        vals = []
        for k in range(1, 6):
            d = allp[(allp.model == m) & (allp.fold == k)]
            r = _score(d)
            vals.append(r["pearson"])
            rows.append({"model": m, "scope": f"fold{k}", "n": len(d),
                         "top_decile_net_bps": 1e4 * _top_decile_net(d),
                         **{key: r[key] for key in KEYS}})
        P(f"{m:>14} " + " ".join(f"{v:>+9.4f}" for v in vals) + f" {np.mean(vals):>+9.4f}")

    P("")
    P("top-decile mean realized return net of cost, by fold (bps)")
    P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>9}" for k in range(1, 6)) + f" {'mean':>9}")
    for m in models:
        vals = [1e4 * _top_decile_net(allp[(allp.model == m) & (allp.fold == k)])
                for k in range(1, 6)]
        P(f"{m:>14} " + " ".join(f"{v:>+9.1f}" for v in vals) + f" {np.mean(vals):>+9.1f}")

    # --- does the transformer add anything on top of the best baseline?
    P("")
    P("-" * 92)
    P("incremental value of PatchTST over each baseline")
    P("-" * 92)
    P("aligned on (symbol, timestamp); 'corr' is between the two prediction series,")
    P("'resid_pearson' is PatchTST vs the part of the actual return the baseline missed.")
    P("")
    P(f"{'baseline':>14} {'n_common':>9} {'corr':>8} {'resid_pearson':>14} {'combo_pearson':>14}")
    for m in sorted(base.model.unique()):
        b = base[base.model == m][["symbol", "timestamp", "predicted_log_return"]]
        j = tst.merge(b, on=["symbol", "timestamp"], suffixes=("_tst", "_base"))
        if not len(j):
            continue
        pt = j.predicted_log_return_tst.to_numpy(float)
        pb = j.predicted_log_return_base.to_numpy(float)
        y = j.actual_log_return.to_numpy(float)
        corr = float(np.corrcoef(pt, pb)[0, 1])
        # regress actual on the baseline, then correlate PatchTST with what is left
        bb = np.polyfit(pb, y, 1)
        resid = y - np.polyval(bb, pb)
        rp = float(np.corrcoef(pt, resid)[0, 1])
        z = lambda a: (a - a.mean()) / a.std()
        cp = float(np.corrcoef(z(pt) + z(pb), y)[0, 1])
        P(f"{m:>14} {len(j):>9} {corr:>+8.3f} {rp:>+14.4f} {cp:>+14.4f}")
        rows.append({"model": f"patchtst_vs_{m}", "scope": "incremental", "n": len(j),
                     "pred_corr": corr, "resid_pearson": rp, "combo_pearson": cp})

    P("")
    P(f"n patchtst = {n_tst:,} (seed-averaged over 5 seeds); baselines are deterministic.")
    P("Nominal significance is inflated ~4x by overlapping 4h targets and cross-asset")
    P("correlation, so treat small pearson gaps between models as noise.")

    txt = "\n".join(out)
    print(txt)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "baselines_report.txt").write_text(txt + "\n")
    pd.DataFrame(rows).to_csv(C.REPORTS / "baselines_comparison.csv", index=False)
    print(f"\nwrote {C.REPORTS}/baselines_report.txt and baselines_comparison.csv")


if __name__ == "__main__":
    main()
