"""Download hourly OHLCV from the Binance public data archive.

data.binance.vision ships monthly CSV zips. Two gotchas handled here:
  * files from 2025 onward stamp open_time in MICROseconds, earlier ones in ms
  * files from 2025 onward carry a header row
"""
from __future__ import annotations

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _months(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m") for d in pd.period_range(start, end, freq="M").to_timestamp()]


def _fetch_month(symbol: str, month: str) -> pd.DataFrame | None:
    url = f"{BASE}/{symbol}/{C.INTERVAL}/{symbol}-{C.INTERVAL}-{month}.zip"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None  # symbol not listed yet
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    header = 0 if not first.split(",")[0].strip().isdigit() else None
    df = pd.read_csv(io.StringIO(raw), header=header, names=COLS if header is None else None)
    df.columns = COLS[: len(df.columns)]
    # normalise ms vs us epochs -> ms
    ot = df["open_time"].astype("int64")
    df["open_time"] = ot.where(ot < 10**14, ot // 1000)
    return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def download_symbol(symbol: str) -> pd.DataFrame:
    out = C.DATA_RAW / f"{symbol}.parquet"
    if out.exists():
        return pd.read_parquet(out)

    months = _months(C.START_MONTH, C.END_MONTH)
    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(lambda m: _fetch_month(symbol, m), months))
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        raise RuntimeError(f"no data for {symbol}")

    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"]).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # reindex onto a strict hourly grid so gaps are explicit NaNs, not silent jumps
    grid = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq="1h", tz="UTC")
    df = df.set_index("timestamp").reindex(grid)
    df.index.name = "timestamp"
    df = df.reset_index()
    df["symbol"] = symbol

    df.to_parquet(out, index=False)
    return df


def main() -> None:
    summary = []
    for sym in C.SYMBOLS:
        df = download_symbol(sym)
        n_missing = int(df["close"].isna().sum())
        summary.append(dict(
            symbol=sym,
            rows=len(df),
            start=str(df["timestamp"].iloc[0]),
            end=str(df["timestamp"].iloc[-1]),
            missing_candles=n_missing,
            missing_pct=round(100 * n_missing / len(df), 4),
        ))
        print(f"{sym:10s} {len(df):7d} rows  {df['timestamp'].iloc[0].date()} -> "
              f"{df['timestamp'].iloc[-1].date()}  missing={n_missing}")
    pd.DataFrame(summary).to_csv(C.REPORTS / "data_summary.csv", index=False)


if __name__ == "__main__":
    main()
