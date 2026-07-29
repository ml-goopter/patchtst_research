"""Frozen experiment configuration.

Every constant that affects target definitions or splits is fixed here BEFORE any
model is evaluated (PLAN.md sections 9, 10, 22).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

for _d in (DATA_RAW, DATA_PROC, RUNS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- market data
# Crypto panel. PLAN.md leaves the market unspecified; feature group F names
# btc/eth benchmarks, and acceptance criteria require multiple assets, so we use
# a liquid Binance spot USDT panel with BTC/ETH as the benchmark context.
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
]
BENCHMARK_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

INTERVAL = "1h"
START_MONTH = "2020-01"
END_MONTH = "2026-06"  # last fully-closed month

# ------------------------------------------------------------------- horizon
HORIZON = 4  # candles ahead (4 x 1h = 4h)
CONTEXT_LEN = 256

# --------------------------------------------------------- frozen thresholds
EPS = 1e-12

# Round-trip cost: taker fee 0.045% x 2 legs + ~0.01% spread/slippage per leg.
COST_THRESHOLD = 0.0011  # log-return units, ~11 bps round trip

# Regime labelling (PLAN.md section 9). FROZEN.
PATH_EFFICIENCY_MIN = 0.35
REGIME_CLASSES = ["downtrend", "sideways", "uptrend"]

# ------------------------------------------------------------------- splits
HOLDOUT_FRACTION = 0.18  # newest 18% untouched until the very end
N_FOLDS = 5
VAL_MONTHS = 3
TEST_MONTHS = 3

# ------------------------------------------------------------------- model
MODEL = dict(
    context_len=CONTEXT_LEN,
    patch_len=16,
    stride=8,
    n_layers=3,
    d_model=128,
    n_heads=4,
    d_ff=512,
    dropout=0.10,
    revin=True,
)

TRAIN = dict(
    # Channel-independent patching means the effective sequence batch is
    # batch_size * n_features, so VRAM scales with the feature count, not just
    # the batch. 128 x 35 channels fits the ~6.5 GB free on the 1070.
    batch_size=128,
    eval_batch_size=128,
    lr=3e-4,
    weight_decay=1e-4,
    max_epochs=14,
    patience=3,
    grad_clip=1.0,
    # Adjacent windows share 255/256 candles and their 4h targets are almost
    # perfectly autocorrelated, so consecutive samples carry little independent
    # information. Each epoch trains on every Nth window and rotates the offset,
    # so all windows are still used -- just spread across epochs. This is a
    # throughput measure, not a data reduction.
    sample_stride=4,
    # GTX 1070 is Pascal (sm_61): native FP16 math runs at 1/64 rate. Measured
    # AMP at 2.9x SLOWER than FP32 here, and torch.compile is unavailable
    # (Triton needs CUDA capability >= 7.0). FP32 eager is the ceiling.
    amp=False,
)

SEEDS = [0, 1, 2, 3, 4]
