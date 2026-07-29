"""Aggregate stage-one runs and check them against the acceptance criteria
(PLAN.md sections 13 and 23).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import decile_monotonicity, expected_return_metrics


def load_runs(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = C.RUNS / tag
    rows, preds = [], []
    for d in sorted(root.glob("*/")):
        rj, pq = d / "result.json", d / "predictions.parquet"
        if not (rj.exists() and pq.exists()):
            continue
        r = json.loads(rj.read_text())
        for split in ("val", "test"):
            rows.append({"fold": r["fold"], "seed": r["seed"], "feature_set": r["feature_set"],
                         "loss": r["loss"], "split": split, "best_epoch": r["best_epoch"],
                         "n_params": r["n_params"], "minutes": r["train_minutes"],
                         **{k: v for k, v in r[split].items() if k != "deciles"}})
        preds.append(pd.read_parquet(pq))
    if not rows:
        raise SystemExit(f"no completed runs under {root}")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def fmt(v, p=6):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def per_fold_table(m: pd.DataFrame) -> str:
    t = m[m.split == "test"].sort_values(["fold", "seed"])
    lines = [f"{'fold':>4} {'seed':>4} {'n':>7} {'RMSE':>10} {'zeroRMSE':>10} {'R2vs0':>9} "
             f"{'pearson':>9} {'spearman':>9} {'dir%':>7} {'dir>c%':>7} {'bias':>10} {'slope':>8}"]
    for _, r in t.iterrows():
        lines.append(
            f"{int(r.fold):>4} {int(r.seed):>4} {int(r.n):>7} {r.rmse:>10.6f} "
            f"{r.baseline_zero_rmse:>10.6f} {r.r2_vs_zero:>+9.5f} {r.pearson:>+9.4f} "
            f"{r.spearman:>+9.4f} {100*r.directional_accuracy:>7.2f} "
            f"{100*r.directional_accuracy_above_cost if np.isfinite(r.directional_accuracy_above_cost) else float('nan'):>7.2f} "
            f"{r.prediction_bias:>+10.2e} {r.calibration_slope:>8.3f}")
    return "\n".join(lines)


def acceptance(m: pd.DataFrame, pooled: dict, per_asset: pd.DataFrame,
               per_fold_pooled: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """PLAN.md section 23, expected-return block."""
    t = m[m.split == "test"]
    checks = []

    rmse_ratio = t.rmse / t.baseline_zero_rmse
    checks.append((
        "Prediction error stable across walk-forward periods",
        bool(rmse_ratio.std() < 0.05 and rmse_ratio.max() < 1.05),
        f"RMSE/zero-RMSE per fold: {', '.join(f'{v:.4f}' for v in rmse_ratio)} "
        f"(sd {rmse_ratio.std():.4f})"))

    checks.append((
        "Pearson and Spearman correlations positive",
        bool((t.pearson > 0).all() and (t.spearman > 0).all()),
        f"pearson {', '.join(f'{v:+.4f}' for v in t.pearson)} | "
        f"spearman {', '.join(f'{v:+.4f}' for v in t.spearman)}"))

    bias_tol = 0.1 * t.actual_std.mean()
    checks.append((
        "Prediction not consistently biased",
        bool(np.abs(t.prediction_bias.mean()) < bias_tol and
             not (np.sign(t.prediction_bias).nunique() == 1 and np.abs(t.prediction_bias).min() > bias_tol)),
        f"mean bias {t.prediction_bias.mean():+.3e} vs 10% of actual sd ({bias_tol:.3e})"))

    checks.append((
        "Larger predicted returns -> larger realized returns",
        bool(np.isfinite(pooled.get("decile_spearman", np.nan)) and pooled["decile_spearman"] > 0.6),
        f"decile rank corr {fmt(pooled.get('decile_spearman'), 3)}, "
        f"top-bottom spread {fmt(pooled.get('decile_spread'), 6)}"))

    n_seeds = t.seed.nunique()
    if n_seeds >= 5:
        spread = t.groupby("fold").pearson.std()
        ok = bool((spread < 0.02).all() and t.groupby("fold").pearson.mean().gt(0).all())
        detail = f"per-fold pearson sd across {n_seeds} seeds: {', '.join(f'{v:.4f}' for v in spread)}"
    else:
        ok, detail = False, f"only {n_seeds} seed(s) run; criterion requires 5"
    checks.append(("Stable across at least five random seeds", ok, detail))

    pos_assets = (per_asset.pearson > 0).sum()
    pos_folds = (per_fold_pooled.pearson > 0).sum()
    checks.append((
        "Not produced by one asset or one short period",
        bool(pos_assets >= 0.7 * len(per_asset) and pos_folds >= 4),
        f"{pos_assets}/{len(per_asset)} assets and {pos_folds}/{len(per_fold_pooled)} folds "
        f"have positive Pearson"))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage1")
    a = ap.parse_args()

    m, preds = load_runs(a.tag)
    test = preds[preds.split == "test"]
    cost = C.COST_THRESHOLD

    out = []
    P = out.append
    P("=" * 100)
    P(f"STAGE ONE - EXPECTED 4h LOG RETURN vs REALIZED  (tag={a.tag})")
    P("=" * 100)
    fs = sorted(m.feature_set.unique())
    P(f"feature set(s): {', '.join(fs)} | loss: {', '.join(sorted(m.loss.unique()))} | "
      f"folds: {sorted(m.fold.unique())} | seeds: {sorted(m.seed.unique())}")
    P(f"model params: {int(m.n_params.iloc[0]):,} | mean train time: {m.minutes.mean():.1f} min/run")
    P(f"cost threshold: {cost:.5f} log-return ({1e4*cost:.0f} bps round trip)")
    P("")

    P("-" * 100)
    P("PER-FOLD TEST RESULTS")
    P("-" * 100)
    P(per_fold_table(m))
    P("")

    # pooled over every test prediction (averaging seeds per (symbol, timestamp))
    pooled_df = (test.groupby(["symbol", "timestamp", "fold"], as_index=False)
                     .agg(predicted_log_return=("predicted_log_return", "mean"),
                          actual_log_return=("actual_log_return", "mean")))
    pv, av = pooled_df.predicted_log_return.to_numpy(), pooled_df.actual_log_return.to_numpy()
    pooled = expected_return_metrics(pv, av, cost)
    pooled.update(decile_monotonicity(pv, av))

    P("-" * 100)
    P("POOLED ACROSS ALL TEST FOLDS")
    P("-" * 100)
    for k in ["n", "mae", "rmse", "huber", "pearson", "spearman", "directional_accuracy",
              "directional_accuracy_above_cost", "n_above_cost", "prediction_bias",
              "calibration_slope", "calibration_intercept", "pred_std", "actual_std"]:
        v = pooled.get(k)
        P(f"  {k:34s} {v if isinstance(v,int) else fmt(v)}")
    P("")
    P("  sanity reference - always predict zero (PLAN.md section 4):")
    P(f"  {'baseline_zero_mae':34s} {fmt(pooled['baseline_zero_mae'])}   "
      f"(model MAE {fmt(pooled['mae'])})")
    P(f"  {'baseline_zero_rmse':34s} {fmt(pooled['baseline_zero_rmse'])}   "
      f"(model RMSE {fmt(pooled['rmse'])})")
    P(f"  {'R2 vs zero predictor':34s} {fmt(pooled['r2_vs_zero'])}  "
      f"({'model beats' if pooled['r2_vs_zero'] > 0 else 'model LOSES to'} the zero predictor)")
    P("")

    P("-" * 100)
    P("PREDICTION DECILES (does a bigger prediction mean a bigger realized return?)")
    P("-" * 100)
    P(f"{'decile':>7} {'n':>8} {'mean predicted':>16} {'mean actual':>14}")
    for d in pooled["deciles"]:
        P(f"{d['bin']:>7} {d['n']:>8} {d['mean_pred']:>+16.6f} {d['mean_actual']:>+14.6f}")
    P(f"  rank correlation {fmt(pooled['decile_spearman'],3)} | "
      f"top-minus-bottom realized spread {fmt(pooled['decile_spread'])}")
    P("")

    rows = []
    for sym, g in pooled_df.groupby("symbol"):
        r = expected_return_metrics(g.predicted_log_return.to_numpy(), g.actual_log_return.to_numpy(), cost)
        rows.append({"symbol": sym, **{k: r[k] for k in
                     ["n", "rmse", "pearson", "spearman", "directional_accuracy", "r2_vs_zero"]}})
    per_asset = pd.DataFrame(rows)
    P("-" * 100)
    P("PER-ASSET (pooled test folds)")
    P("-" * 100)
    P(f"{'symbol':>10} {'n':>7} {'RMSE':>10} {'pearson':>9} {'spearman':>9} {'dir%':>7} {'R2vs0':>9}")
    for _, r in per_asset.iterrows():
        P(f"{r.symbol:>10} {int(r.n):>7} {r.rmse:>10.6f} {r.pearson:>+9.4f} {r.spearman:>+9.4f} "
          f"{100*r.directional_accuracy:>7.2f} {r.r2_vs_zero:>+9.5f}")
    P("")

    rows = []
    for fold, g in pooled_df.groupby("fold"):
        r = expected_return_metrics(g.predicted_log_return.to_numpy(), g.actual_log_return.to_numpy(), cost)
        rows.append({"fold": fold, **{k: r[k] for k in ["n", "rmse", "pearson", "spearman", "r2_vs_zero"]}})
    per_fold_pooled = pd.DataFrame(rows)

    P("=" * 100)
    P("ACCEPTANCE CRITERIA - EXPECTED RETURN (PLAN.md section 23)")
    P("=" * 100)
    checks = acceptance(m, pooled, per_asset, per_fold_pooled)
    for name, ok, detail in checks:
        P(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        P(f"         {detail}")
    n_pass = sum(ok for _, ok, _ in checks)
    P("")
    P(f"  {n_pass}/{len(checks)} criteria met -> "
      f"{'ACCEPT stage one' if n_pass == len(checks) else 'DO NOT proceed to stage two'}")
    P("=" * 100)

    text = "\n".join(out)
    print(text)
    (C.REPORTS / f"{a.tag}_report.txt").write_text(text)
    per_asset.to_csv(C.REPORTS / f"{a.tag}_per_asset.csv", index=False)
    m.to_csv(C.REPORTS / f"{a.tag}_per_run.csv", index=False)
    pooled_df.to_parquet(C.REPORTS / f"{a.tag}_pooled_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
