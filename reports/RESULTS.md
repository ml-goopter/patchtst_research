# PatchTST validation — results

**Date:** 2026-07-30
**Scope:** PLAN.md sections 4a, 13, 15–20, plus the shuffled-target control and ensemble check.
**Status:** Stage one and stage two complete on the development timeline. The final holdout has not been touched.

---

## 1. Verdict

The model does not predict direction. It does predict risk.

| question | answer |
|---|---|
| Can it predict the sign/size of the next 4h return? | No — pooled pearson +0.047 to +0.060, and a plain gradient-boosted tree on time-`t` features does as well or better. |
| Did stage two's distributional head improve the point forecast? | No — it made it worse (+0.0596 → +0.0476) and left it badly over-scaled (calibration slope 0.44 vs 0.60). |
| Is the predicted return distribution better than a simple quantile GBM? | No — the GBM wins CRPS, all three pinball losses, Brier and ECE. |
| Are its probability estimates useful? | No — Brier 0.24917 is *worse* than the 0.24859 you get from quoting the base rate. |
| Can it predict realized volatility? | **Yes** — pearson +0.4645, QLIKE 1.242 vs 1.898 for a static reference. |
| Can it predict the size of intra-horizon excursions? | **Partly** — MAE +0.2788, MFE +0.3313, but it lowballs the dangerous tail badly. |
| Can it classify market regime? | No — accuracy 0.5038 vs 0.5065 for always answering "sideways". |
| Do the auxiliary tasks cost anything? | No — +0.0476 vs +0.0469, series correlated +0.9374. Keep them. |

Volatility is predicted roughly **ten times better than direction**. That is the only result in this project that clears its reference by a wide, stable margin.

---

## 2. Data and splits

### Panel

| property | value |
|---|---|
| Market | Binance spot USDT, 10 symbols |
| Symbols | BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LTC, LINK, AVAX |
| Interval | 1h candles |
| Full range | 2020-01-01 00:00 UTC → 2026-06-30 23:00 UTC |
| Total rows | 557,796 |
| Missing candles | 0.037%–0.054% per symbol |
| Target horizon | 4 candles (4h) |
| Context length | 256 candles |
| Feature set | F2, 35 channels |
| Cost threshold | 0.0011 log-return (11 bps round trip) |

SOL starts 2020-08-11 and AVAX starts 2020-09-22; both list later than the rest of the panel.

### Timeline partition

| segment | range | use |
|---|---|---|
| Development | 2020-01-01 → 2025-04-01 | all 5 walk-forward folds |
| **Final holdout (18%)** | **2025-04-01 → 2026-06-30** | **untouched — never loaded, never scored** |

### Walk-forward folds

Expanding train window, 3-month validation, 3-month test, with a `HORIZON`-length purge at every boundary so no window straddles a split.

| fold | train | rows | validation | rows | test | rows |
|---|---|---|---|---|---|---|
| 1 | 2020-01-01 → 2023-10-01 | 316,796 | 2023-10-01 → 2024-01-01 | 22,040 | 2024-01-01 → 2024-04-01 | 21,800 |
| 2 | 2020-01-01 → 2024-01-01 | 338,876 | 2024-01-01 → 2024-04-01 | 21,800 | 2024-04-01 → 2024-07-01 | 21,800 |
| 3 | 2020-01-01 → 2024-04-01 | 360,716 | 2024-04-01 → 2024-07-01 | 21,800 | 2024-07-01 → 2024-10-01 | 22,040 |
| 4 | 2020-01-01 → 2024-07-01 | 382,556 | 2024-07-01 → 2024-10-01 | 22,040 | 2024-10-01 → 2025-01-01 | 22,040 |
| 5 | 2020-01-01 → 2024-10-01 | 404,636 | 2024-10-01 → 2025-01-01 | 22,040 | 2025-01-01 → 2025-04-01 | 21,560 |

**Pooled test set: 109,240 rows.** Every metric in this document is on test windows only.

Normalization is fit on training rows only — clip at the train 0.1%/99.9% quantiles, then robust-scale by train median and IQR. Targets are scaled the same way and inverted before any metric is computed.

### Models trained

| tag | task | outputs | params | seeds × folds | mean time/run |
|---|---|---|---|---|---|
| `stage1` | Huber point forecast | 1 | 740,167 | 5 × 5 | 38.2 min |
| `studentt` | Student-t return distribution | 2 + global df | 879,049 | 5 × 5 | 10.4 min |
| `multitask` | Student-t + volatility + MAE + MFE + regime | 8 + global df | 1,712,335 | 5 × 5 | 10.1 min |

All PatchTST numbers are averaged over seeds 0–4. Baselines are deterministic. Stage-two runs were faster per run because they moved to a 3090; the architecture is unchanged.

---

## 3. Expected return — every model, side by side (PLAN §4a, §13)

Pooled over all 5 test windows, n = 109,240.

| model | pearson | spearman | R² vs 0 | dir % | decile spread | top-10% net | calib slope |
|---|---|---|---|---|---|---|---|
| **gbm_last** (baseline) | **+0.0674** | +0.0639 | **+0.00249** | 52.33 | 24.9b | +9.0b | 0.601 |
| gbm_q50 (quantile GBM) | +0.0673 | +0.0639 | +0.00248 | 52.55 | 24.8b | +8.9b | 0.601 |
| gbm_lags (baseline) | +0.0605 | +0.0673 | +0.00099 | 52.58 | 26.3b | +8.5b | 0.542 |
| **stage1_huber (PatchTST)** | +0.0596 | +0.0773 | +0.00202 | 52.27 | **38.4b** | **+10.8b** | 0.602 |
| **studentt (PatchTST)** | +0.0476 | +0.0768 | −0.00156 | 52.62 | 31.2b | +4.1b | 0.441 |
| **multitask (PatchTST)** | +0.0469 | **+0.0771** | −0.00237 | **52.64** | 27.2b | +2.4b | 0.410 |
| ridge_lags | +0.0413 | +0.0405 | +0.00097 | 51.04 | 23.5b | +0.7b | 0.602 |
| ridge_last | +0.0345 | +0.0343 | +0.00119 | 50.81 | 11.9b | −1.5b | 0.879 |
| momentum_ols | +0.0120 | +0.0334 | +0.00007 | 51.06 | 1.8b | −4.0b | 0.584 |
| momentum_raw | −0.0140 | −0.0362 | −0.98705 | 48.03 | −3.5b | −5.9b | −0.014 |

Reading this table:

- **A CPU gradient-boosted tree on time-`t` features matches or beats the transformer on linear correlation.** `gbm_last` takes minutes to fit; PatchTST takes 38 min/run × 25 runs.
- **PatchTST's advantage is in rank, not level.** It has the best spearman and by far the widest decile spread (38.4b vs 24.9b) and the best top-decile net return (+10.8b). Whatever it knows is ordering information, not magnitude.
- **Stage two damaged the point forecast.** Both distributional variants have *negative* R² versus predicting zero, meaning their point estimates are worse than a constant. The calibration slope falls from 0.60 to 0.41–0.44 — when the model predicts a large move, the realized move is only ~42% as large.

### By fold

| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean |
|---|---|---|---|---|---|---|
| stage1_huber | +0.0419 | +0.0396 | +0.0269 | **+0.1016** | **+0.0807** | +0.0581 |
| studentt | +0.0493 | +0.0358 | +0.0249 | +0.0799 | +0.0584 | +0.0497 |
| multitask | +0.0528 | +0.0349 | +0.0293 | +0.0664 | +0.0560 | +0.0479 |
| gbm_last | +0.0836 | +0.0487 | +0.0801 | +0.0542 | +0.0784 | +0.0689 |
| gbm_lags | +0.1005 | +0.0285 | +0.0754 | +0.0558 | +0.0549 | +0.0630 |
| ridge_lags | +0.0534 | −0.0200 | +0.0387 | +0.0626 | +0.0707 | +0.0411 |

Two things stand out. **The fold-to-fold spread is larger than the gap between models** — `gbm_lags` ranges from +0.1005 to +0.0285 across adjacent quarters. And **stage two's damage concentrates exactly where stage one had signal**: folds 4 and 5, where Huber reached +0.1016 and +0.0807, drop to +0.0799/+0.0584. The NLL objective spent capacity on spread and tails instead of location.

### Top-decile realized return net of cost, by fold (bps)

| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean |
|---|---|---|---|---|---|---|
| stage1_huber | +5.6 | −2.7 | +4.8 | **+41.1** | −4.8 | +8.8 |
| gbm_last | +32.9 | −9.0 | +2.3 | +19.2 | +8.1 | +10.7 |
| gbm_lags | +35.2 | −17.9 | +5.7 | +26.8 | +3.7 | +10.7 |
| ridge_lags | +16.3 | −21.8 | −0.4 | +33.9 | +14.2 | +8.4 |

Every model loses money in fold 2. PatchTST's headline +8.8 bps mean is carried almost entirely by one quarter (+41.1 in fold 4); its other four folds average +0.7 bps, which does not clear the 11 bps cost assumption in any meaningful sense. This is not a stable edge.

### Per-asset (stage one)

| symbol | n | pearson | spearman | dir % | R² vs 0 |
|---|---|---|---|---|---|
| DOGEUSDT | 10,924 | +0.0838 | +0.0973 | 52.27 | +0.00707 |
| XRPUSDT | 10,924 | +0.0694 | +0.0852 | 52.29 | +0.00430 |
| ADAUSDT | 10,924 | +0.0667 | +0.0900 | 53.20 | +0.00420 |
| BNBUSDT | 10,924 | +0.0655 | +0.0787 | 52.42 | −0.00188 |
| LTCUSDT | 10,924 | +0.0619 | +0.0744 | 51.59 | +0.00197 |
| ETHUSDT | 10,924 | +0.0560 | +0.0780 | 52.87 | −0.00198 |
| BTCUSDT | 10,924 | +0.0557 | +0.0678 | 52.22 | −0.00828 |
| AVAXUSDT | 10,924 | +0.0522 | +0.0768 | 52.00 | +0.00140 |
| LINKUSDT | 10,924 | +0.0520 | +0.0643 | 51.41 | +0.00119 |
| SOLUSDT | 10,924 | +0.0417 | +0.0644 | 52.41 | −0.00110 |

Correlation is weakest on the two largest, most efficient assets (BTC, ETH — both with negative R²) and strongest on the smaller, noisier ones. That is the expected shape if the signal is a microstructure or liquidity effect rather than a macro one.

---

## 4. Return distribution (PLAN §15)

Pooled, n = 109,240.

| model | p10 cov | p50 cov | p90 cov | central-80 cov | CRPS | pinball p10 | pinball p50 | pinball p90 | mean width |
|---|---|---|---|---|---|---|---|---|---|
| **gbm_quantile** | **0.1080** | 0.4972 | 0.8938 | 0.7859 | **0.008532** | **0.003095** | **0.005978** | **0.002921** | 356.4b |
| studentt | 0.1105 | 0.5060 | **0.8974** | 0.7869 | 0.008686 | 0.003202 | 0.006005 | 0.003115 | 358.3b |
| multitask | 0.1085 | 0.4994 | 0.8958 | **0.7874** | 0.008686 | 0.003199 | 0.006009 | 0.003109 | 358.7b |
| static reference | 0.0826 | 0.5005 | 0.9194 | 0.8368 | 0.008852 | 0.003370 | 0.006008 | 0.003344 | 414.4b |
| *target* | *0.1000* | *0.5000* | *0.9000* | *0.8000* | | | | | |

Coverage is good for all three conditional models — every quantile lands within ~1 point of nominal, and all three produce meaningfully tighter intervals than the static reference (356–359b vs 414b) while covering *closer* to target. That part works.

But the Student-t head **loses to the quantile GBM on every sharpness metric**: CRPS, and all three pinball losses. The transformer's distributional output is strictly dominated by a model that only sees features at time `t`.

### Central-80 coverage / mean width by fold

| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---|---|---|---|---|
| studentt | 0.792 / 358b | 0.803 / 311b | 0.803 / 342b | 0.798 / 404b | **0.736** / 376b |
| multitask | 0.793 / 359b | 0.806 / 310b | 0.804 / 346b | 0.792 / 399b | **0.742** / 381b |
| gbm_quantile | 0.795 / 359b | 0.794 / 302b | 0.776 / 310b | 0.781 / 402b | 0.783 / 409b |

Fold 5 is where both PatchTST variants break. Coverage drops to 0.736/0.742 against a 0.80 target — the intervals are too narrow in the most recent, most volatile quarter, which is exactly the regime where an under-covering interval is most costly. The GBM widened its intervals in fold 5 (409b) while PatchTST did not (376b).

### Fitted tail index

The Student-t degrees-of-freedom parameter is a single global learned value (per-sample df collapsed onto its floor — see `src/dist.py`). It settled at **2.65–2.85** (studentt) and **2.75–2.89** (multitask), against an unconditional MLE of 2.12. **Floor and cap bound on 0 of 50 runs**, so the parameterization is stable and the value is data-driven. Mean test return NLL: 1.3784 (studentt), 1.3767 (multitask).

A df near 2.8 means the fitted return distribution has finite variance but very fat tails — consistent with 4h crypto returns.

---

## 5. Probability of clearing cost (PLAN §16)

P(4h return > 11 bps). Base rate 0.4623.

| model | mean p | Brier | log loss | ECE | predicted range |
|---|---|---|---|---|---|
| **gbm_classifier** | 0.4600 | **0.24715** | **0.68744** | 0.01393 | 0.173–0.753 |
| gbm_from_quantiles | 0.4592 | 0.24716 | 0.68749 | **0.01360** | — |
| **static (base rate)** | 0.4640 | **0.24859** | 0.69033 | 0.01254 | — |
| studentt | 0.4696 | 0.24917 | 0.69184 | 0.03132 | 0.048–0.889 |
| multitask | 0.4646 | 0.24938 | 0.69234 | 0.03336 | 0.065–0.909 |

**Both PatchTST variants score worse than the static base rate on Brier and log loss.** A model that ignores the features entirely and outputs 0.4623 for every row beats them. This output has no demonstrated value.

The failure mode is over-confidence, visible in the calibration table:

| quintile | studentt pred → actual | multitask pred → actual | gbm pred → actual |
|---|---|---|---|
| 0 | 0.3568 → 0.4106 | 0.3462 → 0.4069 | 0.3811 → 0.4129 |
| 1 | 0.4328 → 0.4370 | 0.4256 → 0.4424 | 0.4296 → 0.4356 |
| 2 | 0.4719 → 0.4622 | 0.4667 → 0.4617 | 0.4584 → 0.4612 |
| 3 | 0.5094 → 0.4858 | 0.5064 → 0.4869 | 0.4893 → 0.4785 |
| 4 | **0.5773 → 0.5158** | **0.5781 → 0.5134** | 0.5414 → 0.5232 |

PatchTST spreads its probabilities from 0.048 to 0.889 while the truth spans 0.41 to 0.52. The GBM keeps to 0.173–0.753 and is correspondingly better calibrated (ECE 0.014 vs 0.031–0.033). The rank ordering is real in all three — the top quintile does outperform the bottom — but PatchTST's *magnitudes* are fiction.

---

## 6. Volatility (PLAN §17) — the one clear success

Realized 4h volatility, multitask model only.

| model | MAE | RMSE | RMSE (log) | QLIKE | pearson | spearman |
|---|---|---|---|---|---|---|
| **multitask** | **0.006727** | **0.011265** | **0.618675** | **1.241654** | **+0.4645** | **+0.4887** |
| static median | 0.007695 | 0.012861 | 0.717897 | 1.898271 | n/a | n/a |

Beats the static reference on every metric, with QLIKE — the metric that actually matters for volatility, since it penalizes proportional error — improving 35%. Correlation of +0.46 is an order of magnitude above anything achieved on returns.

### But it fails in the tail

| bucket | n | mean actual | mean predicted | bias | pearson |
|---|---|---|---|---|---|
| low | 27,310 | 0.004825 | 0.009170 | **+0.004344** | +0.2367 |
| medium | 27,310 | 0.009042 | 0.011136 | +0.002094 | +0.1267 |
| high | 27,310 | 0.014306 | 0.012781 | −0.001524 | +0.1101 |
| extreme | 27,310 | 0.030344 | 0.015714 | **−0.014630** | +0.2964 |

The prediction is compressed toward the middle. In calm conditions it forecasts roughly double the realized volatility; in the most violent quarter of the sample it forecasts **half** — 0.0157 against an actual 0.0303. The rank ordering across buckets is correct, so the model knows *which* periods are dangerous. It does not know *how* dangerous, and it errs on the wrong side.

---

## 7. Excursions (PLAN §18)

Maximum adverse and favourable excursion within the 4h horizon, multitask only.

| target | MAE of error | RMSE | bias | pearson | spearman |
|---|---|---|---|---|---|
| MAE (drawdown) | 0.009503 | 0.016628 | +0.001650 | +0.2788 | +0.2972 |
| MFE (run-up) | 0.008413 | 0.014006 | −0.001401 | +0.3313 | +0.3150 |

Both are far better than the return forecast (+0.28/+0.33 vs +0.047). Both are far worse than volatility (+0.46).

### Dangerous errors

| condition | share of rows | mean actual | mean predicted |
|---|---|---|---|
| Actual drawdown worse than **2×** predicted | **16.20%** | −0.0381 | −0.0112 |
| Actual drawdown worse than **3×** predicted | **7.26%** | −0.0506 | −0.0106 |

One row in six sees a drawdown more than twice as deep as forecast. One in fourteen sees more than three times — a 5.1% average realized drawdown against a 1.1% warning. A position sized on this forecast would be under-hedged in precisely the situations that matter, and the same tail compression seen in §6 is the cause.

---

## 8. Regime classification (PLAN §19)

Three classes: downtrend / sideways / uptrend. Multitask only.

| metric | model | reference |
|---|---|---|
| Multiclass log loss | 1.02813 | 1.03499 (base rate) |
| Multiclass Brier | 0.61619 | 0.62161 (base rate) |
| Accuracy | 0.5038 | **0.5065** (majority class) |
| Macro F1 | 0.2632 | — |
| Balanced accuracy | 0.3431 | 0.3333 (chance) |

| class | n | base rate | precision | recall | F1 | mean p |
|---|---|---|---|---|---|---|
| downtrend | 26,377 | 0.2415 | 0.3746 | **0.0435** | 0.0779 | 0.2370 |
| sideways | 55,335 | 0.5065 | 0.5112 | **0.9621** | 0.6676 | 0.5147 |
| uptrend | 27,528 | 0.2520 | 0.3210 | **0.0236** | 0.0440 | 0.2483 |

Confusion matrix (rows = actual, columns = predicted):

| | downtrend | sideways | uptrend |
|---|---|---|---|
| **downtrend** | 1,147 | 24,730 | 500 |
| **sideways** | 1,221 | 53,239 | 875 |
| **uptrend** | 694 | 26,184 | 650 |

The model answers "sideways" for 96.2% of rows and is **less accurate than a constant "sideways" rule** (0.5038 vs 0.5065). Log loss improves on the base rate by 0.7%, and balanced accuracy sits 1 point above chance. Precision on the directional classes is above base rate (0.37 and 0.32 vs 0.24 and 0.25), so the few directional calls it does make are better than random — but at recall of 4.4% and 2.4% they are too rare to be useful. This output should not be shipped.

---

## 9. Do the auxiliary tasks hurt the return forecast? (PLAN §20)

Aligned on 109,240 common rows:

| | pearson |
|---|---|
| studentt (return only) | +0.0476 |
| multitask (5 tasks) | +0.0469 |
| correlation between the two prediction series | **+0.9374** |

A 0.0007 difference in pearson, well inside seed noise, and the two prediction series are 94% correlated. **The auxiliary heads are free.** PLAN §20's criterion — keep auxiliary tasks unless they materially reduce return accuracy — is satisfied, so the multitask model is the one to keep. It gives away nothing on returns and adds the volatility and excursion outputs, which are the only outputs that work.

---

## 10. Ensemble check (blend weights fitted on validation, scored on test)

Every weight comes from the fold's own validation window. Nothing is fitted on test.

| blend | pearson | spearman | decile spread | top-10% net |
|---|---|---|---|---|
| **zsum: patchtst + gbm_last** | **+0.0808** | **+0.0925** | 44.0b | +14.8b |
| zsum: patchtst + gbm_lags | +0.0749 | +0.0895 | 40.8b | +13.6b |
| zsum: patchtst + ridge_lags | +0.0705 | +0.0750 | **47.5b** | **+16.9b** |
| ols_all_nn (all models) | +0.0692 | +0.0795 | 40.3b | +16.1b |
| gbm_last alone | +0.0674 | +0.0639 | 24.9b | +9.0b |
| patchtst alone | +0.0596 | +0.0773 | 38.4b | +10.8b |
| ols_all_no_patchtst | +0.0510 | +0.0508 | 25.9b | +8.9b |

This is the most genuinely positive result for PatchTST in the project. A simple validation-fitted z-score sum of PatchTST and `gbm_last` reaches +0.0808 pooled — meaningfully above either component (+0.0596, +0.0674) — and improves top-decile net return to +14.8 bps. Dropping PatchTST from the full blend costs it (+0.0510 vs +0.0692).

The residual analysis says the same thing: PatchTST's predictions correlate only +0.278 with `gbm_last`, and PatchTST still reaches +0.0410 against the part of the return the GBM missed. **The two models see different things.** PatchTST's contribution is as a diversifier, not as a standalone forecaster.

Caveat: the fitted OLS weights are unstable across folds (the `ols_all` weight on PatchTST swings from −0.13 to +0.88, and ridge_lags from −4.42 to +1.51), which is why the unfitted `zsum` blend outperforms the fitted ones. Do not trust the specific weights; trust the fact that a naive equal blend helps.

---

## 11. Is any of this significant? The shuffled-target control

The pipeline was re-run end to end with only the target randomised — same folds, same config, same purging. The `shift` mode applies a common calendar offset of 30–180 days to every symbol, which destroys the signal while preserving both autocorrelation (lag-1 0.7358 vs 0.7360 real) and cross-asset correlation (0.638 vs 0.643 real). This matters: a naive within-symbol permutation destroys both and produces a falsely tight null.

**This control is incomplete — 4 of 15 planned draws, fold 5 only.** It was stopped early by decision.

| series | n | mean | sd | min | max |
|---|---|---|---|---|---|
| null pearson (shift) | 4 | +0.0200 | 0.0281 | −0.0098 | +0.0481 |
| **real pearson (5 seeds)** | 5 | **+0.0693** | 0.0063 | +0.0601 | +0.0768 |

| | decile spread | dir % | R² vs 0 |
|---|---|---|---|
| null mean | 7.0b | 50.72 | −0.00979 |
| real mean | **52.5b** | 52.13 | **+0.00178** |

The real result sits **1.76 null standard deviations** above the null mean — suggestive, not conclusive, and not the 3σ you would want.

The important number is the implied effective sample size. A null sd of 0.0281 implies **n_eff ≈ 1,269 against 21,560 nominal rows — a 17× inflation of significance**, far worse than the ~4× previously assumed from target overlap alone. This means *every* pearson gap in this document is far less meaningful than its nominal n suggests. It reframes the whole results table: the differences between PatchTST, the GBMs, and ridge are mostly inside the noise band.

The ranking metrics separate much better than pearson does. Decile spread is 52.5b real against 7.0b null, and R² versus zero has the right sign in 5 of 5 real runs and the wrong sign in 4 of 4 null runs. **If there is real signal here, it is in the ranking, not the level** — which is consistent with §3, where PatchTST's spearman and decile spread beat the GBMs while its pearson does not.

---

## 12. PLAN §23 acceptance criteria

| criterion | result | verdict |
|---|---|---|
| p10 coverage near 10% | 0.1105 | pass |
| p50 coverage near 50% | 0.5060 | pass |
| p90 coverage near 90% | 0.8974 | pass |
| Central-80 coverage in 75–85% | 0.7869 pooled | pass (fails in fold 5 at 0.736) |
| Intervals not excessively wide | 358b vs 414b static | pass |
| Probability Brier and calibration acceptable | 0.24917 vs 0.24859 static | **fail** |
| Expected return beats zero predictor | R² −0.00156 | **fail** (stage one passed at +0.00202) |
| Volatility beats static reference | QLIKE 1.242 vs 1.898 | pass |
| Auxiliary tasks do not damage return output | +0.0476 vs +0.0469 | pass |
| Regime beats base rate | accuracy 0.5038 vs 0.5065 | **fail** |

---

## 13. Limitations

1. **The shuffled-target control is 4 draws on one fold, not 15 across all folds.** Every significance claim rests on a null estimated from four numbers. The 17× significance inflation it implies is itself imprecisely estimated.
2. **The final holdout has never been evaluated.** Everything here is development-set performance, and the fold-to-fold instability suggests holdout results could differ substantially.
3. **Fold-to-fold variance exceeds model-to-model variance.** Ranking models on pooled pearson is not well supported by the data.
4. **Single market, single horizon.** 10 crypto symbols, 4h horizon, one exchange. No claim about generality.
5. **Costs are modelled as a flat 11 bps** with no market impact, no slippage scaling with size, and no borrow or funding cost. The top-decile net returns are optimistic.
6. **No hyperparameter search was run.** The architecture is PLAN's specification with one measured change (global rather than per-sample `df`). A tuned model might do better — but the shuffled-target control means a tuned model would also be much easier to overfit to the validation windows than the nominal sample size suggests.
7. **Volatility and excursion targets were never given a shuffled-target control.** Their much larger effect sizes make them likelier to be real, but this has not been tested.

---

## 14. Recommendation

**Do not deploy the return forecast or the regime classifier.** The return point forecast is beaten by a CPU tree model, its probability output is beaten by the base rate, and the regime output is beaten by a constant. Stage two's distributional head made the point forecast worse, so if a return forecast is wanted at all, use the stage-one Huber model or `gbm_last`, not the Student-t.

**The volatility head is worth pursuing.** +0.46 correlation and a 35% QLIKE improvement is a real result by a wide margin. Before it could be used it needs its tail compression fixed — the current model halves its forecast in the most extreme quarter of the sample. Training on log-volatility with an asymmetric loss, or explicitly modelling the conditional upper quantile rather than the conditional mean, are the obvious next steps.

**Keep the multitask architecture.** It costs nothing on returns and produces the only two outputs that work.

**The remaining honest use for the return model is as an ensemble component.** Blended with `gbm_last` it reaches +0.0808, above either alone, and it is only +0.278 correlated with that model — it carries information the tree does not. That is a real if modest finding, and it is the strongest claim this project supports.

**Before anything else, finish the shuffled-target control.** At 4 draws the null is too coarse to sustain any of the above. The remaining 11 draws are ~40 minutes of GPU time and would either confirm or dissolve the fold-5 result that most of this rests on.

---

## 15. Artifacts

| file | contents |
|---|---|
| `reports/stage1_report.txt` | stage one, per-fold, per-seed, per-asset |
| `reports/baselines_report.txt` | §4a baseline comparison, incremental value |
| `reports/stage2_report.txt` | §13, §15–20 |
| `reports/stage2_metrics.csv` | machine-readable stage-two metrics |
| `reports/quantiles_report.txt` | §15/§16 quantile and probability GBM |
| `reports/quantiles_lags_report.txt` | same, lagged-feature variant |
| `reports/ensemble_report.txt` | validation-fitted blends, per-fold weights |
| `reports/shuffle_report.txt` | shuffled-target null, 4 draws |
| `reports/stage1_pooled_predictions.parquet` | pooled test predictions |
| `runs/{stage1,studentt,multitask,quantiles,baselines,shuffle}/` | per-run checkpoints and `result.json` |
