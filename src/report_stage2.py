"""Stage-two evaluation (PLAN.md sections 13, 15-20).

Scores the Student-t and multi-task runs against each other, against stage one's
Huber model, and against the quantile GBM -- which is the bar that matters, since
it delivers the same distributional outputs for 1.4 minutes of CPU per fold.

Everything is seed-averaged over 5 seeds and pooled over the 5 test windows, the
same form stage one passed section 23 in. Averaging quantiles across seeds is
legitimate here: each seed predicts the same fixed quantile levels, so the mean of
the p10 predictions is a p10 prediction. Averaging *parameters* would not be, and
is not done.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import decile_monotonicity, expected_return_metrics, qlike, volatility_metrics
from src.report_quantiles import QN, dist_metrics, prob_metrics

KEY = ["fold", "symbol", "timestamp"]

# section 17, in reporting order: the transformer, then what CPU minutes buy
VOL_MODELS = ["gbm_vol_last", "gbm_vol_lags", "har_ols", "rv4_ols", "rv4_raw", "static_median"]


def load(tag: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(C.RUNS / tag / "*" / "predictions.parquet")))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d.split == "test"]
    num = [c for c in d.columns if d[c].dtype.kind == "f" and c not in ("fold", "seed")]
    agg = {c: (c, "mean") for c in num}
    out = d.groupby(KEY, as_index=False).agg(n_seeds=("seed", "nunique"), **agg)
    if "actual_regime" in d.columns:
        r = d.groupby(KEY, as_index=False).actual_regime.first()
        out = out.merge(r, on=KEY)
    return out


def top_decile_net(p: np.ndarray, y: np.ndarray) -> float:
    return float(y[p >= np.quantile(p, 0.9)].mean() - C.COST_THRESHOLD)


def vol_buckets(v: np.ndarray) -> tuple[np.ndarray, list[str]]:
    e = np.quantile(v, [0.25, 0.50, 0.75])
    return np.digitize(v, e), ["low", "medium", "high", "extreme"]


def load_volatility(mt: pd.DataFrame) -> pd.DataFrame:
    """CPU volatility competitors (src/volatility.py), one column per model.

    Inner-joined onto the multi-task rows: the multi-task sample is the smaller of
    the two (it also requires a defined regime label), and every model has to be
    scored on identical rows for the comparison to mean anything.
    """
    f = C.RUNS / "volatility" / "predictions.parquet"
    if not f.exists():
        return pd.DataFrame()
    v = pd.read_parquet(f)
    v = v[v.split == "test"]
    w = v.pivot_table(index=KEY, columns="model", values="pred_volatility").reset_index()
    w.columns.name = None
    return mt[KEY + ["pred_volatility", "actual_volatility"]].merge(w, on=KEY, how="inner")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage2_report.txt")
    a = ap.parse_args()

    st = load("studentt")
    mt = load("multitask")
    h = pd.read_parquet(C.REPORTS / "stage1_pooled_predictions.parquet")
    g = pd.read_parquet(C.RUNS / "quantiles" / "predictions.parquet")
    g = g[g.split == "test"]

    out, rows = [], []
    P = out.append
    P("=" * 100)
    P("STAGE TWO - distributional and multi-task PatchTST (PLAN.md sections 13, 15-20)")
    P("=" * 100)
    P("")
    P(f"studentt  {len(st):,} seed-averaged rows over {int(st.n_seeds.max())} seeds")
    P(f"multitask {len(mt):,} seed-averaged rows over {int(mt.n_seeds.max())} seeds")
    P("Test splits only. The final holdout remains untouched.")
    P("")

    # ------------------------------------------------- section 13, mean prediction
    P("-" * 100)
    P("section 13 - expected return, pooled over 5 test windows")
    P("-" * 100)
    P(f"{'model':>14} {'n':>8} {'pearson':>9} {'spearman':>9} {'R2vs0':>10} {'dir%':>7} "
      f"{'decile_sp':>10} {'top10_net':>10} {'calib':>7}")
    series = [
        ("studentt", st.predicted_log_return, st.actual_log_return),
        ("multitask", mt.predicted_log_return, mt.actual_log_return),
        ("stage1_huber", h.predicted_log_return, h.actual_log_return),
        ("gbm_q50", g.q50, g.actual_log_return),
    ]
    for name, p, y in series:
        p, y = np.asarray(p, float), np.asarray(y, float)
        r = expected_return_metrics(p, y, C.COST_THRESHOLD)
        r.update(decile_monotonicity(p, y))
        P(f"{name:>14} {len(p):>8} {r['pearson']:>+9.4f} {r['spearman']:>+9.4f} "
          f"{r['r2_vs_zero']:>+10.5f} {100*r['directional_accuracy']:>7.2f} "
          f"{1e4*r['decile_spread']:>9.1f}b {1e4*top_decile_net(p, y):>9.1f}b "
          f"{r['calibration_slope']:>7.3f}")
        rows.append({"section": "13", "model": name, "pearson": r["pearson"],
                     "spearman": r["spearman"], "r2_vs_zero": r["r2_vs_zero"],
                     "decile_spread": r["decile_spread"],
                     "top_decile_net": top_decile_net(p, y)})

    P("")
    P("test pearson by fold")
    P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>9}" for k in range(1, 6)) + f" {'mean':>9}")
    for name, dfp, pc, yc in [("studentt", st, "predicted_log_return", "actual_log_return"),
                              ("multitask", mt, "predicted_log_return", "actual_log_return"),
                              ("stage1_huber", h, "predicted_log_return", "actual_log_return"),
                              ("gbm_q50", g, "q50", "actual_log_return")]:
        v = []
        for k in range(1, 6):
            s = dfp[dfp.fold == k]
            v.append(float(np.corrcoef(s[pc], s[yc])[0, 1]))
        P(f"{name:>14} " + " ".join(f"{x:>+9.4f}" for x in v) + f" {np.mean(v):>+9.4f}")

    # ------------------------------------------------------ section 15, quantiles
    P("")
    P("-" * 100)
    P("section 15 - return-distribution validation")
    P("-" * 100)
    P(f"{'model':>14} {'p10cov':>8} {'p50cov':>8} {'p90cov':>8} {'c80cov':>8} "
      f"{'CRPS':>10} {'NLL':>8} {'pin_p10':>9} {'pin_p50':>9} {'pin_p90':>9} {'width':>9}")
    for name, d, yc in [("studentt", st, "actual_log_return"),
                        ("multitask", mt, "actual_log_return"),
                        ("gbm_quantile", g, "actual_log_return")]:
        m = dist_metrics(d[QN].to_numpy(float), d[yc].to_numpy(float))
        P(f"{name:>14} {m['p10_coverage']:>8.4f} {m['p50_coverage']:>8.4f} "
          f"{m['p90_coverage']:>8.4f} {m['central80_coverage']:>8.4f} {m['crps']:>10.6f} "
          f"{m['nll']:>8.3f} {m['pinball_p10']:>9.6f} {m['pinball_p50']:>9.6f} "
          f"{m['pinball_p90']:>9.6f} {1e4*m['mean_interval_width']:>8.1f}b")
        rows.append({"section": "15", "model": name, **{k: v for k, v in m.items()}})
    P(f"{'target':>14} {0.10:>8.4f} {0.50:>8.4f} {0.90:>8.4f} {0.80:>8.4f}")

    P("")
    P("central-80 coverage / mean width by fold")
    P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>16}" for k in range(1, 6)))
    for name, d in [("studentt", st), ("multitask", mt), ("gbm_quantile", g)]:
        cells = []
        for k in range(1, 6):
            s = d[d.fold == k]
            m = dist_metrics(s[QN].to_numpy(float), s.actual_log_return.to_numpy(float))
            cells.append(f"{m['central80_coverage']:.3f}/{1e4*m['mean_interval_width']:.0f}b")
        P(f"{name:>14} " + " ".join(f"{c:>16}" for c in cells))

    # ---------------------------------------------------- section 16, probability
    P("")
    P("-" * 100)
    P("section 16 - probability validation, P(return > cost)")
    P("-" * 100)
    P(f"{'model':>14} {'base':>7} {'mean_p':>7} {'brier':>9} {'logloss':>9} {'ECE':>8} "
      f"{'p_range':>16}")
    for name, p, y in [("studentt", st.p_above_cost, st.actual_log_return),
                       ("multitask", mt.p_above_cost, mt.actual_log_return),
                       ("gbm_clf", g.p_above_cost, g.actual_log_return)]:
        p, y = np.asarray(p, float), np.asarray(y, float)
        m = prob_metrics(p, (y > C.COST_THRESHOLD).astype(int))
        P(f"{name:>14} {m['base_rate']:>7.4f} {m['mean_pred']:>7.4f} {m['brier']:>9.5f} "
          f"{m['log_loss']:>9.5f} {m['ece']:>8.5f} {f'{p.min():.3f}-{p.max():.3f}':>16}")
        rows.append({"section": "16", "model": name, **m})

    P("")
    P("calibration by predicted-probability quintile (studentt / multitask / gbm)")
    P(f"{'quintile':>10} " + " ".join(f"{n:>22}" for n in ("studentt", "multitask", "gbm_clf")))
    qs = {}
    for name, p, y in [("studentt", st.p_above_cost, st.actual_log_return),
                       ("multitask", mt.p_above_cost, mt.actual_log_return),
                       ("gbm_clf", g.p_above_cost, g.actual_log_return)]:
        p, y = np.asarray(p, float), np.asarray(y, float)
        ev = (y > C.COST_THRESHOLD).astype(int)
        b = np.clip(np.digitize(p, np.quantile(p, np.linspace(0, 1, 6))[1:-1]), 0, 4)
        qs[name] = [(p[b == k].mean(), ev[b == k].mean()) for k in range(5)]
    for k in range(5):
        P(f"{k:>10} " + " ".join(f"{f'{qs[n][k][0]:.4f} -> {qs[n][k][1]:.4f}':>22}"
                                 for n in qs))

    # ----------------------------------------------------- section 17, volatility
    if "pred_volatility" in mt.columns:
        P("")
        P("-" * 100)
        P("section 17 - volatility validation, PatchTST vs GBM")
        P("-" * 100)
        v = load_volatility(mt)
        have_run = not v.empty
        if not have_run:  # nothing to compare against; keep the original single-model view
            v = mt[KEY + ["pred_volatility", "actual_volatility"]].copy()
            v["static_median"] = float(np.median(v.actual_volatility))
            cols = ["static_median"]
            P("runs/volatility not found -- run src/volatility.py for the GBM comparison.")
            P("static_median here is the pooled TEST median, not a train-only reference.")
        else:
            cols = [m for m in VOL_MODELS if m in v.columns]
            P(f"aligned on {len(v):,} rows common to the multi-task run and src/volatility.py "
              f"({len(mt):,} multi-task rows)")
            P("competitors are CPU-only, fitted per fold on the F2 features with the same folds,")
            P("purging and train-only normalization; static_median is the fold's TRAIN median.")
            P("All models regress log volatility and exponentiate, so all report a median-like")
            P("point forecast and none gets a mean-vs-median advantage. One asymmetry remains:")
            P("the multi-task checkpoint is selected on validation return NLL (PLAN.md section")
            P("20), not on volatility loss, while every competitor is selected on its own task.")
            P("Competitor predictions are clipped at the fold's train 0.1% quantile of realized")
            P("volatility, since persistence can forecast exactly zero and QLIKE is a variance")
            P("ratio. The clip never binds for the transformer: its lowest prediction is 0.0017.")
        av = v.actual_volatility.to_numpy(float)
        named = [("multitask", "pred_volatility")] + [(c, c) for c in cols]
        P("")
        P(f"{'model':>14} {'MAE':>10} {'RMSE':>10} {'RMSE_log':>10} {'QLIKE':>10} "
          f"{'pearson':>9} {'spearman':>9}")
        for name, col in named:
            m = volatility_metrics(v[col].to_numpy(float), av)
            P(f"{name:>14} {m['mae']:>10.6f} {m['rmse']:>10.6f} {m['rmse_log']:>10.6f} "
              f"{m['qlike']:>10.6f} {m['pearson']:>+9.4f} {m['spearman']:>+9.4f}")
            rows.append({"section": "17", "model": name, "mae": m["mae"], "rmse": m["rmse"],
                         "rmse_log": m["rmse_log"], "qlike": m["qlike"],
                         "pearson": m["pearson"], "spearman": m["spearman"]})

        if have_run:
            P("static_median is constant within a fold, so its pooled correlation is an artefact")
            P("of the median moving between folds; per fold it is undefined.")
        P("")
        P("test QLIKE by fold (lower is better)")
        P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>9}" for k in range(1, 6)) + f" {'mean':>9}")
        for name, col in named:
            x = [qlike(s.actual_volatility.to_numpy(float), s[col].to_numpy(float))
                 for s in (v[v.fold == k] for k in range(1, 6))]
            P(f"{name:>14} " + " ".join(f"{q:>9.4f}" for q in x) + f" {np.mean(x):>9.4f}")

        P("")
        P("test pearson by fold")
        P(f"{'model':>14} " + " ".join(f"{f'fold{k}':>9}" for k in range(1, 6)) + f" {'mean':>9}")
        for name, col in named:
            x = []
            for k in range(1, 6):
                s = v[v.fold == k]
                p = s[col].to_numpy(float)
                x.append(np.nan if np.std(p) < 1e-14
                         else float(np.corrcoef(p, s.actual_volatility.to_numpy(float))[0, 1]))
            P(f"{name:>14} " + " ".join(f"{q:>+9.4f}" for q in x) + f" {np.mean(x):>+9.4f}")

        P("")
        P("by realized-volatility bucket (quartiles of realized volatility)")
        P(f"{'model':>14} {'bucket':>9} {'n':>8} {'mean_actual':>12} {'mean_pred':>11} "
          f"{'bias':>10} {'pearson':>9}")
        b, bnames = vol_buckets(av)
        for name, col in named:
            if name == "static_median":
                continue
            pv = v[col].to_numpy(float)
            for k, nm in enumerate(bnames):
                m = b == k
                P(f"{name:>14} {nm:>9} {int(m.sum()):>8} {av[m].mean():>12.6f} "
                  f"{pv[m].mean():>11.6f} {pv[m].mean()-av[m].mean():>+10.6f} "
                  f"{np.corrcoef(pv[m], av[m])[0,1]:>+9.4f}")

    # ------------------------------------------------------ section 18, MAE / MFE
    if "pred_mae" in mt.columns:
        P("")
        P("-" * 100)
        P("section 18 - excursion validation (multitask only)")
        P("-" * 100)
        P(f"{'target':>10} {'MAE_err':>10} {'RMSE':>10} {'bias':>11} {'pearson':>9} "
          f"{'spearman':>9}")
        for nm, pc, ac in [("MAE", "pred_mae", "actual_mae"), ("MFE", "pred_mfe", "actual_mfe")]:
            p, y = mt[pc].to_numpy(float), mt[ac].to_numpy(float)
            P(f"{nm:>10} {np.mean(np.abs(p-y)):>10.6f} {np.sqrt(np.mean((p-y)**2)):>10.6f} "
              f"{np.mean(p-y):>+11.6f} {np.corrcoef(p,y)[0,1]:>+9.4f} "
              f"{stats.spearmanr(p,y).statistic:>+9.4f}")
            rows.append({"section": "18", "model": nm, "bias": float(np.mean(p - y)),
                         "pearson": float(np.corrcoef(p, y)[0, 1])})
        P("")
        P("dangerous errors: predicted downside small, actual downside large")
        pm, am = mt.pred_mae.to_numpy(float), mt.actual_mae.to_numpy(float)
        for thr in (2.0, 3.0):
            bad = am < thr * pm  # actual drawdown more than thr x the predicted
            P(f"  actual MAE worse than {thr:.0f}x predicted: {100*bad.mean():.2f}% of rows "
              f"(mean actual {am[bad].mean():+.4f} vs predicted {pm[bad].mean():+.4f})")

    # --------------------------------------------------------- section 19, regime
    if "p_uptrend" in mt.columns:
        P("")
        P("-" * 100)
        P("section 19 - regime validation (multitask only)")
        P("-" * 100)
        Pm = mt[[f"p_{c}" for c in C.REGIME_CLASSES]].to_numpy(float)
        Pm = Pm / Pm.sum(axis=1, keepdims=True)
        yt = mt.actual_regime.to_numpy(int)
        oh = np.eye(len(C.REGIME_CLASSES))[yt]
        base = oh.mean(axis=0)
        ll = float(-np.mean(np.log(np.clip(Pm[np.arange(len(yt)), yt], 1e-12, 1))))
        ll0 = float(-np.mean(np.log(np.clip(np.tile(base, (len(yt), 1))[np.arange(len(yt)), yt],
                                            1e-12, 1))))
        P(f"multiclass log loss   {ll:.5f}   (base-rate reference {ll0:.5f})")
        P(f"multiclass Brier      {np.mean(np.sum((Pm-oh)**2, axis=1)):.5f}   "
          f"(base-rate reference {np.mean(np.sum((base-oh)**2, axis=1)):.5f})")
        pred = Pm.argmax(axis=1)
        P("")
        P(f"{'class':>12} {'n':>8} {'base':>7} {'precision':>10} {'recall':>8} {'F1':>8} "
          f"{'mean_p':>8}")
        f1s, recs = [], []
        for i, cn in enumerate(C.REGIME_CLASSES):
            tp = int(((pred == i) & (yt == i)).sum())
            prec = tp / max((pred == i).sum(), 1)
            rec = tp / max((yt == i).sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            f1s.append(f1); recs.append(rec)
            P(f"{cn:>12} {int((yt==i).sum()):>8} {base[i]:>7.4f} {prec:>10.4f} {rec:>8.4f} "
              f"{f1:>8.4f} {Pm[:,i].mean():>8.4f}")
        P("")
        P(f"macro F1 {np.mean(f1s):.4f}   balanced accuracy {np.mean(recs):.4f}   "
          f"accuracy {np.mean(pred==yt):.4f}   majority-class accuracy {base.max():.4f}")
        P("")
        P("confusion matrix (rows = actual, cols = predicted)")
        P(f"{'':>12} " + " ".join(f"{c:>12}" for c in C.REGIME_CLASSES))
        for i, cn in enumerate(C.REGIME_CLASSES):
            P(f"{cn:>12} " + " ".join(f"{int(((yt==i)&(pred==j)).sum()):>12}"
                                      for j in range(len(C.REGIME_CLASSES))))

    # ------------------------------------------------- section 20, does it cost us
    P("")
    P("-" * 100)
    P("section 20 - do the auxiliary tasks damage the return output?")
    P("-" * 100)
    j = st[KEY + ["predicted_log_return", "actual_log_return"]].merge(
        mt[KEY + ["predicted_log_return"]], on=KEY, suffixes=("_st", "_mt"))
    y = j.actual_log_return.to_numpy(float)
    p_st = j.predicted_log_return_st.to_numpy(float)
    p_mt = j.predicted_log_return_mt.to_numpy(float)
    P(f"aligned on {len(j):,} common rows")
    P(f"  studentt  pearson {np.corrcoef(p_st,y)[0,1]:+.4f}")
    P(f"  multitask pearson {np.corrcoef(p_mt,y)[0,1]:+.4f}")
    P(f"  correlation between the two prediction series {np.corrcoef(p_st,p_mt)[0,1]:+.4f}")
    P("")
    P("PLAN.md section 20 keeps auxiliary tasks only when they do not materially")
    P("reduce expected-return or return-distribution accuracy.")

    txt = "\n".join(out)
    print(txt)
    (C.REPORTS / a.out).write_text(txt + "\n")
    pd.DataFrame(rows).to_csv(C.REPORTS / "stage2_metrics.csv", index=False)
    print(f"\nwrote {C.REPORTS}/{a.out} and stage2_metrics.csv")


if __name__ == "__main__":
    main()
