"""Honest ensemble check: fit the blend on validation, score it on test.

STATUS.md reports combo_pearson = +0.0794 for a z-score sum of PatchTST and
gbm_last. That number is fitted and scored on the same test rows -- the z-score
means and standard deviations come from the test series itself. This script
refits every blend weight on the fold's validation window only, freezes it, and
applies it to the test window, so the reported number is one a live system could
actually have produced.

Blends, all fitted per fold on validation:

  zsum      z-score each prediction with validation mean/sd, then add
  ols       actual ~ 1 + patchtst + baseline, least squares
  ols_nn    the same with negative weights clipped to zero, then refit intercept
  ols_all   actual ~ 1 + patchtst + every baseline that is not pure momentum

No training, no GPU, no panel data -- this reads existing prediction files only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import decile_monotonicity, expected_return_metrics

KEY = ["fold", "symbol", "timestamp"]
# momentum_raw is the unscaled persistence signal; it and its OLS rescaling carry
# the same information, so only the rescaled one enters the multi-model blend.
ALL_BLEND = ["gbm_last", "gbm_lags", "ridge_last", "ridge_lags", "momentum_ols"]


def load_patchtst(tag: str = "stage1") -> pd.DataFrame:
    """Seed-averaged PatchTST predictions for both val and test splits."""
    frames = [pd.read_parquet(p) for p in sorted((C.RUNS / tag).glob("*/predictions.parquet"))]
    if not frames:
        raise SystemExit(f"no predictions under {C.RUNS / tag}")
    df = pd.concat(frames, ignore_index=True)
    n_seeds = df.groupby(KEY + ["split"]).size().max()
    out = (df.groupby(KEY + ["split"], as_index=False)
             .agg(patchtst=("predicted_log_return", "mean"),
                  actual_log_return=("actual_log_return", "first"),
                  n_seeds=("predicted_log_return", "size")))
    if out.n_seeds.nunique() != 1:
        print(f"  warning: uneven seed coverage {sorted(out.n_seeds.unique())}", flush=True)
    print(f"  patchtst: {len(out):,} rows, seed-averaged over {n_seeds} seeds")
    return out.drop(columns="n_seeds")


def load_baselines(tag: str = "baselines") -> pd.DataFrame:
    b = pd.read_parquet(C.RUNS / tag / "predictions.parquet")
    wide = b.pivot_table(index=KEY + ["split"], columns="model",
                         values="predicted_log_return").reset_index()
    wide.columns.name = None
    return wide


def _zsum(tr: dict[str, np.ndarray], te: dict[str, np.ndarray], cols: list[str]) -> np.ndarray:
    """Sum of z-scores, with mean and sd taken from the validation split only."""
    return sum((te[c] - tr[c].mean()) / tr[c].std() for c in cols)


def _ols(tr: dict[str, np.ndarray], te: dict[str, np.ndarray], cols: list[str],
         y_tr: np.ndarray, non_negative: bool = False) -> tuple[np.ndarray, dict]:
    A = np.column_stack([tr[c] for c in cols] + [np.ones(len(y_tr))])
    w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    if non_negative and (w[:-1] < 0).any():
        w[:-1] = np.clip(w[:-1], 0, None)
        # refit the intercept so the blend stays unbiased on validation
        w[-1] = float(y_tr.mean() - sum(w[i] * tr[c].mean() for i, c in enumerate(cols)))
    B = np.column_stack([te[c] for c in cols] + [np.ones(len(te[cols[0]]))])
    return B @ w, {c: float(w[i]) for i, c in enumerate(cols)} | {"intercept": float(w[-1])}


def _top_decile_net(pred: np.ndarray, actual: np.ndarray) -> float:
    cut = np.quantile(pred, 0.9)
    return float(actual[pred >= cut].mean() - C.COST_THRESHOLD)


def _score(pred: np.ndarray, actual: np.ndarray, unitless: bool) -> dict:
    r = expected_return_metrics(pred, actual, C.COST_THRESHOLD)
    r.update(decile_monotonicity(pred, actual))
    r["top_decile_net"] = _top_decile_net(pred, actual)
    if unitless:  # z-score sums have no return units, so scale metrics are meaningless
        for k in ("mae", "rmse", "huber", "r2_vs_zero", "rmse_vs_zero_ratio",
                  "prediction_bias", "calibration_slope"):
            r[k] = np.nan
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--baseline-tag", default="baselines")
    ap.add_argument("--out", default="ensemble_report.txt")
    a = ap.parse_args()

    print("loading predictions", flush=True)
    pt = load_patchtst(a.tag)
    bl = load_baselines(a.baseline_tag)
    df = pt.merge(bl, on=KEY + ["split"], how="inner")
    base_models = [c for c in bl.columns if c not in KEY + ["split"]]
    print(f"  baselines: {base_models}")
    print(f"  merged:    {len(df):,} rows "
          f"({df[df.split=='val'].shape[0]:,} val / {df[df.split=='test'].shape[0]:,} test)")

    folds = sorted(df.fold.unique())
    # one blend definition -> per-fold test predictions, concatenated for pooling
    pooled: dict[str, list[pd.DataFrame]] = {}
    weights_rows, fold_rows = [], []

    for f in folds:
        d_tr = df[(df.fold == f) & (df.split == "val")]
        d_te = df[(df.fold == f) & (df.split == "test")]
        tr = {c: d_tr[c].to_numpy(float) for c in ["patchtst"] + base_models}
        te = {c: d_te[c].to_numpy(float) for c in ["patchtst"] + base_models}
        y_tr = d_tr.actual_log_return.to_numpy(float)
        y_te = d_te.actual_log_return.to_numpy(float)
        meta = d_te[KEY].reset_index(drop=True)

        def emit(name: str, pred: np.ndarray, unitless: bool, w: dict | None = None) -> None:
            pooled.setdefault(name, []).append(
                meta.assign(pred=pred, actual=y_te, unitless=unitless))
            r = _score(pred, y_te, unitless)
            fold_rows.append({"blend": name, "fold": f, "n": len(pred),
                              "pearson": r["pearson"], "spearman": r["spearman"],
                              "decile_spread": r["decile_spread"],
                              "top_decile_net": r["top_decile_net"],
                              "r2_vs_zero": r["r2_vs_zero"]})
            if w:
                weights_rows.append({"blend": name, "fold": f, **w})

        # --- the standalone models, for reference on identical rows
        emit("patchtst", te["patchtst"], False)
        for m in base_models:
            emit(m, te[m], False)

        # --- pairwise blends of PatchTST with each baseline
        for m in base_models:
            cols = ["patchtst", m]
            emit(f"zsum:{m}", _zsum(tr, te, cols), True)
            p, w = _ols(tr, te, cols, y_tr)
            emit(f"ols:{m}", p, False, w)
            p, w = _ols(tr, te, cols, y_tr, non_negative=True)
            emit(f"ols_nn:{m}", p, False, w)

        # --- everything at once
        cols = ["patchtst"] + [m for m in ALL_BLEND if m in base_models]
        p, w = _ols(tr, te, cols, y_tr)
        emit("ols_all", p, False, w)
        p, w = _ols(tr, te, cols, y_tr, non_negative=True)
        emit("ols_all_nn", p, False, w)
        # what the baselines achieve without PatchTST -- the real question is whether
        # adding the transformer to a baseline stack beats the stack alone
        cols_nb = [m for m in ALL_BLEND if m in base_models]
        p, w = _ols(tr, te, cols_nb, y_tr)
        emit("ols_all_no_patchtst", p, False, w)
        print(f"  fold {f}: {len(y_tr):,} val -> {len(y_te):,} test", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    pooled_rows = []
    for name, frames in pooled.items():
        d = pd.concat(frames, ignore_index=True)
        r = _score(d.pred.to_numpy(float), d.actual.to_numpy(float), bool(d.unitless.iloc[0]))
        pooled_rows.append({"blend": name, "n": len(d), **{
            k: r[k] for k in ("pearson", "spearman", "r2_vs_zero", "decile_spread",
                              "top_decile_net", "calibration_slope",
                              "directional_accuracy")}})
    pooled_df = pd.DataFrame(pooled_rows)

    # ------------------------------------------------------------------ report
    out = []
    P = out.append
    P("=" * 96)
    P("ENSEMBLE CHECK - blend weights fitted on validation, scored on test (STATUS.md step 3)")
    P("=" * 96)
    P("")
    P("Every weight below comes from the fold's validation window only. The z-score means")
    P("and sds for 'zsum' are validation statistics, so nothing here is fitted on test.")
    P("Metrics in raw log-return units except for zsum rows, which are unitless (nan).")
    P("")

    order = (["patchtst"] + sorted(base_models)
             + sorted(k for k in pooled if k.startswith("zsum:"))
             + sorted(k for k in pooled if k.startswith("ols:"))
             + sorted(k for k in pooled if k.startswith("ols_nn:"))
             + ["ols_all", "ols_all_nn", "ols_all_no_patchtst"])
    pooled_df = pooled_df.set_index("blend").loc[order].reset_index()

    P("pooled over all 5 test windows")
    P(f"{'blend':>24} {'n':>8} {'pearson':>9} {'spearman':>9} {'R2vs0':>10} "
      f"{'decile_sp':>10} {'top10_net':>10} {'calib':>7}")
    ref = float(pooled_df.loc[pooled_df.blend == "patchtst", "pearson"].iloc[0])
    for _, r in pooled_df.iterrows():
        mark = ""
        if r.blend.startswith(("zsum", "ols")):
            mark = " *" if r.pearson > ref else ""
        P(f"{r.blend:>24} {int(r.n):>8} {r.pearson:>+9.4f} {r.spearman:>+9.4f} "
          f"{r.r2_vs_zero:>+10.5f} {1e4*r.decile_spread:>9.1f}b "
          f"{1e4*r.top_decile_net:>9.1f}b {r.calibration_slope:>7.3f}{mark}")
    P("")
    P("* = pooled pearson above PatchTST alone.")

    P("")
    P("test pearson by fold")
    piv = fold_df.pivot(index="blend", columns="fold", values="pearson").loc[order]
    P(f"{'blend':>24} " + " ".join(f"{f'fold{k}':>9}" for k in piv.columns) + f" {'mean':>9}")
    for name, row in piv.iterrows():
        P(f"{name:>24} " + " ".join(f"{v:>+9.4f}" for v in row) + f" {row.mean():>+9.4f}")

    P("")
    P("top-decile mean realized return net of cost, by fold (bps)")
    piv = fold_df.pivot(index="blend", columns="fold", values="top_decile_net").loc[order]
    P(f"{'blend':>24} " + " ".join(f"{f'fold{k}':>9}" for k in piv.columns) + f" {'mean':>9}")
    for name, row in piv.iterrows():
        P(f"{name:>24} " + " ".join(f"{1e4*v:>+9.1f}" for v in row)
          + f" {1e4*row.mean():>+9.1f}")

    P("")
    P("-" * 96)
    P("fitted blend weights (validation, per fold)")
    P("-" * 96)
    w = pd.DataFrame(weights_rows)
    for name in ["ols:gbm_last", "ols:gbm_lags", "ols_all", "ols_all_no_patchtst"]:
        sub = w[w.blend == name].drop(columns="blend").set_index("fold")
        sub = sub.dropna(axis=1, how="all")
        P("")
        P(f"{name}")
        P(sub.round(4).to_string())
    P("")
    P("A weight near zero on patchtst means validation could not find a use for it.")

    txt = "\n".join(out)
    print()
    print(txt)
    (C.REPORTS / a.out).write_text(txt + "\n")
    fold_df.to_csv(C.REPORTS / "ensemble_by_fold.csv", index=False)
    pooled_df.to_csv(C.REPORTS / "ensemble_pooled.csv", index=False)
    w.to_csv(C.REPORTS / "ensemble_weights.csv", index=False)
    print(f"\nwrote {C.REPORTS}/{a.out}, ensemble_by_fold.csv, ensemble_pooled.csv, "
          f"ensemble_weights.csv")


if __name__ == "__main__":
    main()
