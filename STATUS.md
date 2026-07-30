# Status / handoff

Last updated 2026-07-30 (volatility comparison added; shuffle control still paused
mid-run). Read `PLAN.md` for the frozen plan,
`README.md` for layout, hardware limits and leakage controls. This file is only:
what is done, what the numbers actually say, and what to do next.

**Paused deliberately with the shuffled-target control 4 draws of 15 complete.**
Resume command at the bottom of "Next steps". The GPU is free.

---

## Where the project is

Stage one (expected 4h log return) is **complete and evaluated**. Stage two's runs are
complete and reported in `reports/RESULTS.md`. The final holdout (2025-04 → 2026-06)
has **not been touched by any run** — keep it that way until the model choice is
settled.

| item | state |
| --- | --- |
| data, features, targets | done — all six targets already in `data/processed/panel.parquet` |
| stage one, F2 + Huber, 5 folds x 5 seeds | done — 25 runs, `runs/stage1/` |
| section 23 acceptance (original six) | **6/6 pass** — `reports/stage1_report.txt` |
| section 4a competing baselines | done — `reports/baselines_report.txt` |
| shuffled-target control | **4 of 15 draws** — `reports/shuffle_report.txt`, preliminary and alarming |
| ensemble check (was step 3) | done — `reports/ensemble_report.txt` |
| quantile + probability GBM (was step 2) | done — `reports/quantiles_report.txt` |
| stage two (sections 14–20) | runs complete, 50 runs in `runs/{studentt,multitask}/`; numbers in `reports/RESULTS.md` and `reports/stage2_report.txt`. The narrative sections of this file below still predate them. |
| section 17 volatility competitors | done — `src/volatility.py`, scored in `reports/stage2_report.txt` section 17 |
| section 21 ablations | not started |
| holdout evaluation | not started, deliberately |

`data/` is gitignored and was regenerated on 2026-07-29 from `src/download.py` +
`src/features.py`. It reproduces exactly: 557,796 rows, leakage checks pass, and
re-running fold 5 of the baselines returns every recorded digit (gbm_last +0.0787,
ridge_lags +0.0707, ridge_last +0.0336).

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

That +0.0794 was fitted and scored on the same test rows. **It survives an honest
protocol** — `src/ensemble.py`, every weight from the fold's validation window,
applied to test (`reports/ensemble_report.txt`):

| blend | fold1 | fold2 | fold3 | fold4 | fold5 | mean | pooled | top10 net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patchtst | +.0419 | +.0396 | +.0269 | +.1016 | +.0807 | +.0581 | +0.0596 | 10.8b |
| gbm_last | +.0836 | +.0487 | +.0799 | +.0542 | +.0787 | +.0690 | +0.0674 | 9.0b |
| **zsum:gbm_last** | +.0816 | +.0548 | +.0666 | +.0937 | +.0996 | **+.0793** | **+0.0808** | **14.8b** |

Never the worst in any fold, and far more stable than PatchTST alone (range
.055–.100 vs .027–.102). Adding PatchTST to a stack of all five baselines lifts it
from +0.0510 to +0.0692.

**Equal weights beat fitted weights.** Every validation-fitted OLS blend does worse
than the naive z-score sum (`ols:gbm_last` +0.0611 vs `zsum` +0.0808) — fold 4 fits
a *negative* PatchTST weight that then hurts on test. Do not "optimize" this blend.

## The distributional targets are already met — by a tree

`src/quantiles.py`, 7 min of CPU, GBM on the F2 features at time t only, same folds
and purging (`reports/quantiles_report.txt`). "static" predicts the training
window's unconditional quantiles for every row — the distributional zero predictor:

| | p10cov | p50cov | p90cov | c80cov | CRPS | mean width |
| --- | --- | --- | --- | --- | --- | --- |
| gbm_quantile | 0.1080 | 0.4972 | 0.8938 | 0.7859 | **0.008532** | **356 bps** |
| static | 0.0826 | 0.5005 | 0.9194 | 0.8368 | 0.008852 | 414 bps |
| PLAN target | 0.10 | 0.50 | 0.90 | 0.80 | | |

Better calibrated *and* 58 bps narrower, and it wins every pinball loss. **This
clears all six of PLAN.md section 23's return-distribution criteria.** Section 14's
Student-t PatchTST head has to beat this, not the static reference.

The probability head is the weak part: Brier 0.24715 vs static 0.24859, log loss
0.68744 vs 0.69033. Real but tiny. Only 136 of 109,240 rows ever predict above 0.65
and none above 0.80, so PLAN section 22's "probability above cost >= 0.60" trading
rule would fire on ~1% of rows.

**History adds nothing here either.** The same fit over 10 log-spaced lags
(`--offsets lags`, 350 features, ~9x the runtime, `reports/quantiles_lags_report.txt`)
scores CRPS 0.008539 vs 0.008532 and Brier 0.24718 vs 0.24715 — indistinguishable,
marginally worse. This mirrors gbm_last ≥ gbm_lags on the mean task, and the
volatility comparison below makes it three. Three independent model families now say
the 4h-ahead information is in the most recent candle, which is the strongest
argument yet against spending capacity on a 256-candle context.

### Things that are easy to get wrong when reporting this

- **Ranking skill, not magnitude skill.** R² vs zero is +0.002 only *after*
  seed-averaging. Per individual run, 20 of 25 have RMSE above the zero predictor's.
  Calibration slope 0.60 → predictions are over-scaled by ~1.7x.
- **Two different per-fold pearsons exist.** Mean-of-per-seed-pearson
  (+0.0348, +0.0297, +0.0225, +0.0884, +0.0693) vs pearson-of-seed-averaged-series
  (+0.0419, +0.0396, +0.0269, +0.1016, +0.0807). Both correct, different questions.
  `reports/baselines_report.txt` uses the second; `reports/stage1_report.txt` the first.
- **Nominal t-stats are inflated ~4x** — and the partial shuffled control says
  **~17x**, which would make the honest noise threshold ~0.06 rather than ~0.02.
  Overlapping 4h targets and cross-asset correlation mean 109,240 rows are nowhere
  near 109,240 independent samples. Until the control finishes, treat pearson gaps
  under ~0.02 as noise and know that number is probably far too generous.
- **Fold 4 carries the economics.** Top-decile net of the 11 bps cost, by fold:
  +5.6, −2.7, +4.8, **+41.1**, −4.8 bps. Three of five folds clear cost.
- **Not just "crypto goes up".** In folds 1 and 5 mean prediction has the opposite
  sign to the period's actual drift and correlation is still positive.

---

## The volatility target goes the same way

`src/volatility.py`, 7.5 min of CPU for five folds, scored in `reports/stage2_report.txt`
section 17 against the multi-task head's volatility output (109,240 common test rows):

| model | QLIKE | pearson | MAE |
| --- | --- | --- | --- |
| multitask PatchTST | 1.2417 | +0.4645 | 0.006727 |
| **gbm_vol_last** (35 feats, no history) | **0.9599** | **+0.6004** | 0.005922 |
| gbm_vol_lags (350 feats) | 0.9582 | +0.5996 | 0.005912 |
| har_ols (3 features) | 1.0175 | +0.5532 | 0.006205 |
| rv4_ols (trailing 4h vol, rescaled) | 1.2248 | +0.4703 | 0.006709 |
| static_median (fold-train median) | 1.7772 | n/a | 0.007729 |

PatchTST beats a constant and loses to everything that is fitted — including a
two-parameter rescaling of trailing volatility, which it ties. It loses pearson in
5 folds of 5 and QLIKE in 4 of 5. RESULTS.md called volatility "the one clear
success"; it is a clear success *for the target*, not for the model. Section 6 of
`reports/RESULTS.md` has been rewritten accordingly.

The 350-feature lag design ties the 35-feature one here too — a third independent
family finding nothing in the 256-candle context beyond the last candle.

`y_mae` and `y_mfe` (section 18) still have no competing-model control at all. That
is now the cheapest outstanding check in the project.

## The shuffled-target control — partial, and it dominates everything else

`src/train.py --shuffle-target {shift,iid}` refits the pipeline unchanged on a
target whose link to the features has been destroyed. Fold 5, seed 0, 4 draws of a
planned 15. **Do not plan stage two around the stage-one numbers until this is
finished.**

| series | n | mean | sd | min | max |
| --- | --- | --- | --- | --- | --- |
| null pearson (shift) | 4 | **+0.0200** | 0.0281 | −0.0098 | +0.0481 |
| REAL pearson (5 seeds) | 5 | +0.0693 | 0.0063 | +0.0601 | +0.0768 |

Real fold 5 is only **+1.76 null sd** above the null mean. The null sd implies
n_eff ≈ 1,269 against 21,560 nominal test rows — a **17x** inflation of nominal
significance, well beyond the ~4x assumed below. If this holds at 15 draws, the
"gaps under 0.02 are noise" rule becomes more like 0.06, and every model in the
section 4a table collapses into one indistinguishable group.

Two caveats before anyone acts on it: 4 draws puts ±50% error on that sd, and this
is one fold. Fold 4 (+0.0884) has more headroom than fold 5.

**The ranking metrics separate far better than pearson does**, which is the more
useful reading so far:

| | pearson | decile spread | R² vs zero | dir % |
| --- | --- | --- | --- | --- |
| null mean (4 draws) | +0.0200 | 7.0 bps | −0.00979 | 50.72 |
| real mean (5 seeds) | +0.0693 | 52.5 bps | +0.00178 | 52.13 |

Decile spread is 7.5x apart and R² vs zero has the right sign in every real run and
the wrong sign in every null draw. Pearson may simply be the wrong statistic for
overlapping cross-correlated targets.

### Two nulls, and why both

`shift` moves the target back by one common time offset: autocorrelation (lag-1
0.7358 vs 0.7360) and cross-asset correlation (0.638 vs 0.643) survive exactly,
only feature alignment dies. `iid` — the permutation PLAN.md section 4a literally
specifies — destroys all three at once, driving lag-1 to −0.004 and cross-asset to
0.000, which makes every row look independent and reports a **falsely tight null**.
Report `shift`; report `iid` beside it to show the gap. The `iid` draws have not
run yet.

---

## Next steps, in order

**1. Finish the shuffled-target control — ~2 h GPU. Still blocking.**
6 shift draws and 5 iid draws remain. Runs are ~10 min each on the 3090.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python src/train.py \
  --feature-set F2 --loss huber --folds 5 --seeds 0 --tag shuffle \
  --shuffle-target shift --shuffle-seeds 4,5,6,7,8,9 --quiet
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python src/train.py \
  --feature-set F2 --loss huber --folds 5 --seeds 0 --tag shuffle \
  --shuffle-target iid --shuffle-seeds 0,1,2,3,4 --quiet
.venv/bin/python src/report_shuffle.py
```

`report_shuffle.py` reads per-run `result.json`, not the batch summary, so an
interrupted batch still reports every draw that finished. Safe to stop between
draws; a kill costs only the in-flight run.

If the null holds near +0.02 ± 0.03, the next move is **not** stage two. It is to
re-run the section 4a comparison with the null subtracted and decide whether any
model in it is distinguishable from noise on pearson — and if not, to re-centre the
whole evaluation on decile spread and R² vs zero, which do separate.

**2. Then** stage two, 50 runs. On this 3090 that is **~8–9 h**, not the 33 h below:
fold 5 measures 10.3 min/run vs 41.8 min on the 1070, a 4.1x speedup. Section 14
Student-t 5x5, then section 20 multi-task 5x5. Per-run cost is unchanged from stage
one — the head goes from 1 to 3–5 outputs, under 0.5% of FLOPs. Known risk: the
Student-t `df` parameter diverging on fat-tailed 4h crypto returns; plan on softplus
floors and df clamping. **Whatever it produces must beat the quantile GBM below,
which cost 7 minutes of CPU.**

Worth benchmarking first: `TRAIN["amp"]=False` and `batch_size=128` were tuned for
the 1070's 6.5 GB and its 1/64-rate FP16. Ampere reverses both, and only 4.9 GB of
24 GB is in use. Do not change them until the shuffle control is done — the null is
only valid on the identical config.

**Promoted from deferred:** section 21 ablations, specifically `head="mean"` — 31x
fewer head params (4,480 → 1 instead of 138,880 → 1, currently 19% of the model).
Now ~45 min as a 5-fold single-seed probe on the 3090, not 3 h. Two independent
model families (GBM mean, GBM quantile) find no value in history beyond the last
candle, so a flatten head over 256 candles is very likely capacity in the wrong
place. Cheap, and it may explain the whole result.

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

# quantile + probability GBM (~1.4 min/fold, CPU only); --offsets lags is ~9x slower
.venv/bin/python src/quantiles.py --folds 1,2,3,4,5 --tag quantiles
.venv/bin/python src/report_quantiles.py

# volatility competitors for section 17 (~1.5 min/fold, CPU only); scored by report_stage2
.venv/bin/python src/volatility.py --folds 1,2,3,4,5 --tag volatility
.venv/bin/python src/report_stage2.py

# ensemble: validation-fitted blends scored on test (no training, no GPU, seconds)
.venv/bin/python src/ensemble.py

# shuffled-target null (see "Next steps" for the resume command)
.venv/bin/python src/report_shuffle.py

# rebuild data/ from scratch (~3 min; data/ is gitignored)
.venv/bin/python src/download.py && .venv/bin/python src/features.py

# fold windows
.venv/bin/python src/splits.py
```

## Operational notes

- Use `.venv` for everything. Do not install into system python.
- Baselines, quantiles and the ensemble are CPU-only and run concurrently with GPU
  training. `--offsets lags` saturates every core; `nice` it if sharing the machine.
- **Current GPU is an RTX 3090, 24 GB.** Fold 5 runs at 10.3 min vs 41.8 min on the
  original 1070 — 4.1x. Every timing in this file is 3090-based unless it says
  otherwise. `config.py` is still tuned for the 1070 (`amp=False`, `batch_size=128`,
  4.9 GB of 24 GB used); run `src/bench.py` before resizing, and not before the
  shuffle control finishes.
- The user wants **concise output**: numbers and verdict, not the reasoning around
  them. No section headers in chat replies, no restating background, no extended
  caveat addenda unless they change the conclusion.
- The user wants **concise output**: numbers and verdict, not the reasoning around
  them. No section headers in chat replies, no restating background, no extended
  caveat addenda unless they change the conclusion.

## Files worth knowing

```
reports/stage1_report.txt              section 23 acceptance, per-fold table
reports/stage1_pooled_predictions.parquet   seed-averaged test preds (the ensemble input)
reports/baselines_report.txt           section 4a comparison + incremental value
reports/baselines_comparison.csv       same, machine-readable
reports/ensemble_report.txt            validation-fitted blends scored on test
reports/ensemble_{pooled,by_fold,weights}.csv   same, machine-readable
reports/quantiles_report.txt           sections 15-16 vs a static reference
reports/quantiles_lags_report.txt      same with 10 lags; no better than one candle
reports/shuffle_report.txt             the null on randomised targets (PARTIAL, 4/15)
reports/shuffle_draws.csv              one row per null draw
runs/stage1/<F2_huber_foldN_seedS>/    predictions.parquet, result.json, model.pt
runs/shuffle/<mode>_draw<k>/<run>/     one directory per null draw
runs/baselines/, runs/quantiles/       predictions.parquet, summary.jsonl
src/baselines.py                       six competing models, chunked ridge, GBM
src/report_baselines.py                pooled + per-fold + incremental-value report
src/quantiles.py                       11-quantile GBM + P(>cost) classifier
src/volatility.py                      section 17 competitors: persistence, HAR-RV, GBM
runs/volatility/                       predictions.parquet, summary.jsonl
src/ensemble.py                        zsum / OLS / non-negative blends, val-fitted
src/train.py:shuffle_target            the two nulls; read the docstring before using
```
