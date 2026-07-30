"""Shuffled-target control (PLAN.md section 4a, STATUS.md step 1).

What correlation does this pipeline manufacture when there is provably no signal?
Every run scored here is the stage-one pipeline, unchanged, on a target whose link
to the features has been destroyed. Whatever spread these runs show is the noise
floor that the real numbers must clear.

Two nulls, and the gap between them is the point:

  iid    target permuted independently within each symbol. Destroys the target's
         autocorrelation and cross-asset correlation along with the signal, so
         each row behaves like an independent sample and the null looks tight.
  shift  target moved back by one common time offset. Autocorrelation and
         cross-asset correlation survive exactly; only the alignment with the
         features is destroyed. This is the null that matches the real data,
         where 4h targets overlap 4-to-1 across 10 correlated assets.

Reporting the iid null alone would understate the noise floor.
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

METRICS = ["pearson", "spearman", "decile_spread", "r2_vs_zero", "directional_accuracy"]


def load_runs(tag: str) -> pd.DataFrame:
    """Read per-run result.json rather than the batch summary.

    train.py appends summary.jsonl only after a whole --shuffle-seeds batch
    finishes, but run_one writes result.json as each run completes. Reading the
    per-run files means an interrupted batch still reports every draw that
    actually finished, so the GPU can be freed between draws without losing work.
    """
    rows = []
    for p in sorted((C.RUNS / tag).glob("*/*/result.json")):
        r = json.loads(p.read_text())
        # the draw directory carries the mode when the run predates the json fields
        mode, _, draw = p.parent.parent.name.partition("_draw")
        rows.append({
            "mode": r.get("shuffle_target", mode),
            "draw": int(r["shuffle_seed"] if r.get("shuffle_seed") is not None else draw),
            "fold": r["fold"], "seed": r["seed"],
            "best_epoch": r["best_epoch"], "n_test": r["test"]["n"],
            **{m: r["test"][m] for m in METRICS},
        })
    if not rows:
        raise SystemExit(f"no completed runs under {C.RUNS / tag}")
    d = pd.DataFrame(rows)
    return d.drop_duplicates(subset=["mode", "draw", "fold", "seed"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="shuffle")
    ap.add_argument("--out", default="shuffle_report.txt")
    a = ap.parse_args()

    null = load_runs(a.tag)
    real = pd.read_csv(C.REPORTS / "stage1_per_run.csv")
    real = real[real.split == "test"]

    out = []
    P = out.append
    P("=" * 96)
    P("SHUFFLED-TARGET CONTROL - the null this pipeline produces on noise")
    P("=" * 96)
    P("")
    P("Identical pipeline, identical config, identical folds. Only the target is")
    P("randomised. Every number below is on the fold's held-out test window.")
    P("")

    for fold in sorted(null.fold.unique()):
        nf = null[null.fold == fold]
        rf = real[real.fold == fold]
        P("-" * 96)
        P(f"fold {fold}")
        P("-" * 96)
        P(f"{'series':>28} {'n':>4} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9} "
          f"{'|>=real|':>9}")

        real_mean = float(rf.pearson.mean())
        for mode in ["shift", "iid"]:
            m = nf[nf["mode"] == mode]
            if not len(m):
                continue
            v = m.pearson.to_numpy(float)
            ge = int((v >= real_mean).sum())
            P(f"{f'null pearson ({mode})':>28} {len(v):>4} {v.mean():>+9.4f} {v.std(ddof=1):>8.4f} "
              f"{v.min():>+9.4f} {v.max():>+9.4f} {f'{ge}/{len(v)}':>9}")
        v = rf.pearson.to_numpy(float)
        P(f"{'REAL pearson (5 seeds)':>28} {len(v):>4} {v.mean():>+9.4f} {v.std(ddof=1):>8.4f} "
          f"{v.min():>+9.4f} {v.max():>+9.4f} {'-':>9}")

        # how far above the null is the real result, in null standard deviations
        P("")
        for mode in ["shift", "iid"]:
            m = nf[nf["mode"] == mode]
            if len(m) < 3:
                continue
            v = m.pearson.to_numpy(float)
            sd, mu = v.std(ddof=1), v.mean()
            z = (real_mean - mu) / sd
            # sd of a mean-zero correlation over n_eff independent samples is
            # 1/sqrt(n_eff); invert to see what this null implies about sample size
            n_eff = 1.0 / sd**2
            n_nom = int(nf.n_test.iloc[0])
            P(f"  {mode:>5}: real is {z:+.2f} null sd above the null mean. "
              f"null sd {sd:.4f} implies n_eff ~ {n_eff:,.0f} "
              f"vs {n_nom:,} nominal rows ({n_nom/n_eff:.0f}x inflation)")
        P("")

        P(f"{'':>28} {'decile_sp':>11} {'dir%':>8} {'R2vs0':>10} {'best_ep':>8}")
        for mode in ["shift", "iid"]:
            m = nf[nf["mode"] == mode]
            if not len(m):
                continue
            P(f"{f'null mean ({mode})':>28} {1e4*m.decile_spread.mean():>10.1f}b "
              f"{100*m.directional_accuracy.mean():>8.2f} {m.r2_vs_zero.mean():>+10.5f} "
              f"{m.best_epoch.mean():>8.1f}")
        P(f"{'REAL mean':>28} {1e4*rf.decile_spread.mean():>10.1f}b "
          f"{100*rf.directional_accuracy.mean():>8.2f} {rf.r2_vs_zero.mean():>+10.5f} "
          f"{rf.best_epoch.mean():>8.1f}")
        P("")

    P("-" * 96)
    P("every null draw")
    P("-" * 96)
    P(f"{'mode':>7} {'draw':>5} {'fold':>5} {'seed':>5} {'pearson':>9} {'spearman':>9} "
      f"{'decile_sp':>11} {'best_ep':>8}")
    for _, r in null.sort_values(["fold", "mode", "draw"]).iterrows():
        P(f"{r['mode']:>7} {int(r.draw):>5} {int(r.fold):>5} {int(r.seed):>5} "
          f"{r.pearson:>+9.4f} {r.spearman:>+9.4f} {1e4*r.decile_spread:>10.1f}b "
          f"{int(r.best_epoch):>8}")

    P("")
    P("A null centred near zero with a wide spread does not mean the pipeline is broken.")
    P("It means correlation on overlapping, cross-correlated targets is a noisy statistic,")
    P("and single-fold results must clear that spread before they mean anything.")

    txt = "\n".join(out)
    print(txt)
    (C.REPORTS / a.out).write_text(txt + "\n")
    null.to_csv(C.REPORTS / "shuffle_draws.csv", index=False)
    print(f"\nwrote {C.REPORTS}/{a.out} and shuffle_draws.csv")


if __name__ == "__main__":
    main()
