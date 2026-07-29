# PatchTST validation

Implementation of `PLAN.md`. Stage one only: does a predicted 4-hour log return
match the realized 4-hour log return on unseen timestamps?

## Layout

```
PLAN.md              the frozen plan
config.py            all thresholds, splits and model config (frozen before evaluation)
src/download.py      Binance hourly OHLCV -> data/raw/*.parquet
src/features.py      feature groups A-G, targets, leakage assertions -> data/processed/panel.parquet
src/splits.py        dev/holdout split + 5 expanding walk-forward folds
src/datamodule.py    windowing, fold-local normalization, GPU-resident batching
src/model.py         PatchTST (RevIN + channel attention as ablation switches)
src/train.py         one run = (feature_set, fold, seed)
src/metrics.py       PLAN.md section 13 metrics
src/report.py        aggregation + PLAN.md section 23 acceptance criteria
src/baselines.py     PLAN.md section 4a competing baselines (CPU only)
src/report_baselines.py  baseline comparison + incremental value of PatchTST
src/bench.py         GPU throughput benchmark
STATUS.md            current state, what the numbers say, next steps
```

Run order:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/download.py
.venv/bin/python src/features.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python src/train.py --feature-set F2 --loss huber --folds 1,2,3,4,5 --seeds 0
.venv/bin/python src/report.py --tag stage1
.venv/bin/python src/baselines.py --folds 1,2,3,4,5 --tag baselines
.venv/bin/python src/report_baselines.py
```

## Decisions the plan left open

**Market.** The plan does not name an asset class. Feature group F lists crypto
benchmarks and the acceptance criteria require several assets, so this uses a
10-symbol Binance spot USDT panel (BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LTC, LINK,
AVAX), hourly, 2020-01 to 2026-06. 557,796 rows, 0.05% missing candles.

**Cost threshold.** 0.0011 log-return (~11 bps round trip): 0.045% taker fee per
leg plus ~1 bp spread/slippage per leg. Frozen in `config.py` before evaluation.

**Feature group E** (price-volume interaction) is not assigned to a version in the
plan's F0-F5 table. It needs both returns and volume z-scores, so it sits in F2.
F2 = 35 features.

**Gaps.** Missing candles are kept as explicit NaNs on a strict hourly grid. Any
window or target touching a NaN is dropped rather than imputed — 95.6% of rows
survive with all F4 features and targets present.

## Throughput

Channel-independent patching makes the effective sequence batch
`batch_size * n_features` (128 x 35 = 4,480 sequences), so memory scales with the
feature set, not just the batch size. `src/bench.py` measures s/step, samples/s and
peak VRAM for a given config — run it before sizing a sweep on new hardware.

A naive 5-fold x 5-seed sweep over ~300k windows/epoch is expensive on any single
GPU. Instead each epoch trains on every 4th window and rotates the offset
(`TRAIN["sample_stride"]`), so every window is still used, just spread across
epochs. Adjacent windows share 255/256 candles and their 4h targets are almost
perfectly autocorrelated, so this costs very little independent information.

`TRAIN["amp"]` defaults to off. Mixed precision is a large win on Ampere and newer
and a loss on pre-Volta cards, so benchmark it rather than assuming.

## Leakage controls

- Every feature at index `t` uses only candles `<= t`; every target uses only
  candles `> t`. `src/features.py:leakage_checks` recomputes features on a
  truncated series and asserts they are unchanged.
- Clip limits and robust scalers are fit on training rows only, then frozen.
  Predictions are inverted to raw log-return units before any metric.
- Each split boundary is purged by `HORIZON` candles, so no training target can
  see into validation and no validation target into test.
- The newest 18% of the timeline (2025-04 onward) is holdout and is not touched
  by any run reported here.
