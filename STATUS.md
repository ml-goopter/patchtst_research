# Status / handoff

Last updated 2026-07-29. Read `PLAN.md` for the frozen plan, `README.md` for layout,
hardware limits and leakage controls. This file is only: what is done, what the
numbers actually say, and what to do next.

---

## Where the project is

Stage one (expected 4h log return) is **complete and evaluated**. Stage two has not
started. The final holdout (2025-04 → 2026-06) has **not been touched by any run** —
keep it that way until stage one's model choice is settled.

| item | state |
| --- | --- |
| data, features, targets | done — all six targets already in `data/processed/panel.parquet` |
| stage one, F2 + Huber, 5 folds x 5 seeds | done — 25 runs, `runs/stage1/` |
| section 23 acceptance (original six) | **6/6 pass** — `reports/stage1_report.txt` |
| section 4a competing baselines | done — `reports/baselines_report.txt` |
| shuffled-target control | **NOT RUN** — blocking, see next steps |
| stage two (sections 14–20) | not started |
| section 21 ablations | not started |
| holdout evaluation | not started, deliberately |

---

## What the numbers say

Stage one passes its original acceptance criteria, and PatchTST **loses to a
gradient-boosted tree on a single candle**. Both statements are true; do not report
one without the other.

Pooled test predictions, n=109,240, five test windows 2024-01 → 2025-04:

| model | pearson | R² vs zero | decile spread | top-10% net of cost |
| --- | --- | --- | --- | --- |
| gbm_last (35 feats, **no history**) | **+0.0674** | **+0.00249** | 24.9 bps | +9.0 bps |
| gbm_lags | +0.0605 | +0.00099 | 26.3 bps | +8.5 bps |
| **patchtst** (seed-averaged) | +0.0596 | +0.00202 | **38.4 bps** | +10.8 bps |
| ridge_lags | +0.0413 | +0.00097 | 23.5 bps | +0.7 bps |
| ridge_last | +0.0345 | +0.00119 | 11.9 bps | −1.5 bps |
| momentum_ols | +0.0120 | +0.00007 | 1.8 bps | −4.0 bps |
| momentum_raw | −0.0140 | −0.98705 | −3.5 bps | −5.9 bps |

Mean per-fold pearson: gbm_last +0.0690, gbm_lags +0.0630, patchtst +0.0581.
The GBM wins 3 of 5 folds and every pooled metric except decile spread.

PatchTST is **not redundant** with it, though:

| baseline | pred corr | resid pearson | combo pearson |
| --- | --- | --- | --- |
| gbm_last | +0.278 | **+0.0410** | **+0.0794** |
| gbm_lags | +0.373 | +0.0371 | +0.0724 |

Correlation with the GBM is only 0.28, PatchTST still correlates +0.041 with the
part of the actual return the GBM misses, and a naive z-score sum of the two beats
either alone. Current best reading: **PatchTST is an ensemble component, not a
standalone forecaster.**

### Things that are easy to get wrong when reporting this

- **Ranking skill, not magnitude skill.** R² vs zero is +0.002 only *after*
  seed-averaging. Per individual run, 20 of 25 have RMSE above the zero predictor's.
  Calibration slope 0.60 → predictions are over-scaled by ~1.7x.
- **Two different per-fold pearsons exist.** Mean-of-per-seed-pearson
  (+0.0348, +0.0297, +0.0225, +0.0884, +0.0693) vs pearson-of-seed-averaged-series
  (+0.0419, +0.0396, +0.0269, +0.1016, +0.0807). Both correct, different questions.
  `reports/baselines_report.txt` uses the second; `reports/stage1_report.txt` the first.
- **Nominal t-stats are inflated ~4x.** Overlapping 4h targets (~4x) and cross-asset
  correlation (~3–4x) mean 109,240 rows are nowhere near 109,240 independent samples.
  Treat pearson gaps under ~0.02 between models as noise.
- **Fold 4 carries the economics.** Top-decile net of the 11 bps cost, by fold:
  +5.6, −2.7, +4.8, **+41.1**, −4.8 bps. Three of five folds clear cost.
- **Not just "crypto goes up".** In folds 1 and 5 mean prediction has the opposite
  sign to the period's actual drift and correlation is still positive.

---

## Next steps, in order

Do **not** start stage two compute before 1 and 2.

**1. Shuffled-target control — ~40 min, one fold, GPU. Blocking.**
Refit one fold with `y_return` randomly permuted within each symbol. Establishes the
null distribution this pipeline produces on noise. If shuffled targets yield pearson
±0.03 on 109k overlapping samples, a chunk of the +0.06 is artifact and everything
above needs re-reading. Cheapest check with the highest chance of invalidating the
result. Now a section 23 criterion.

**2. Quantile GBM — under an hour, CPU.**
`HistGradientBoostingRegressor(loss="quantile")` at 0.1/0.5/0.9 plus a classifier for
P(return > cost), same folds. Delivers all of sections 15 and 16 immediately, and
tells you what a Student-t PatchTST head has to beat before spending 33 GPU-hours on
one. Extend `src/baselines.py`.

**3. Ensemble check — free, no training.**
The +0.0794 combo is a naive z-score sum on existing predictions. Fit the blend
weight on validation, score on test. If it holds, PatchTST's role is settled and it
changes what stage two is for.

**Then** stage two, 50-run / ~33 h path (section 14 Student-t 5x5, then section 20
multi-task 5x5). Per-run cost is unchanged from stage one — the head goes from 1 to
3–5 outputs, under 0.5% of FLOPs — so budget ~40 min/run including NLL needing a
couple more epochs. Known risk: the Student-t `df` parameter diverging on fat-tailed
4h crypto returns; plan on softplus floors and df clamping.

**Deferred:** section 21 ablations. `head="mean"` is the interesting one — 31x fewer
head params (4,480 → 1 instead of 138,880 → 1, currently 19% of the model). Given
PatchTST barely beats a single-candle GBM, the flatten head over 256 candles looks
like capacity in the wrong place. ~3 h as a 5-fold single-seed probe.

---

## Commands

```bash
# stage one (25 runs, ~15 h)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python src/train.py --feature-set F2 --loss huber --folds 1,2,3,4,5 --seeds 0,1,2,3,4
.venv/bin/python src/report.py --tag stage1

# baselines (~1 min/fold, CPU only)
.venv/bin/python src/baselines.py --folds 1,2,3,4,5 --tag baselines
.venv/bin/python src/report_baselines.py

# fold windows
.venv/bin/python src/splits.py
```

## Operational notes

- Use `.venv` for everything. Do not install into system python.
- Baselines are CPU-only and can run concurrently with GPU training.
- Timings below are from the original development GPU. Run `src/bench.py` before
  sizing a sweep on different hardware.
- The user wants **concise output**: numbers and verdict, not the reasoning around
  them. No section headers in chat replies, no restating background, no extended
  caveat addenda unless they change the conclusion.

## Files worth knowing

```
reports/stage1_report.txt              section 23 acceptance, per-fold table
reports/stage1_pooled_predictions.parquet   seed-averaged test preds (the ensemble input)
reports/baselines_report.txt           section 4a comparison + incremental value
reports/baselines_comparison.csv       same, machine-readable
runs/stage1/<F2_huber_foldN_seedS>/    predictions.parquet, result.json, model.pt
runs/baselines/                        predictions.parquet, summary.jsonl
src/baselines.py                       six competing models, chunked ridge, GBM
src/report_baselines.py                pooled + per-fold + incremental-value report
```
