from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.common.config import load_yaml
from src.common.paths import run_dirs, ensure_dir


def _as_date(s: str) -> pd.Timestamp:
    # Accept YYYY-MM-DD, return normalized Timestamp (naive)
    return pd.to_datetime(s).normalize()


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV from yfinance for [start, end] inclusive.
    yfinance 'end' is exclusive, so we add 1 day.
    """
    start_dt = _as_date(start)
    end_dt_exclusive = _as_date(end) + pd.Timedelta(days=1)

    df = yf.download(
        tickers=tickers,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt_exclusive.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("yfinance returned an empty dataframe (check tickers/window/network).")

    # yfinance returns:
    # - MultiIndex columns when multiple tickers
    # - Single-index columns when one ticker
    out_rows = []
    if isinstance(df.columns, pd.MultiIndex):
        for tkr in tickers:
            if tkr not in df.columns.get_level_values(0):
                continue
            sub = df[tkr].copy()
            sub["ticker"] = tkr
            out_rows.append(sub.reset_index())
        prices = pd.concat(out_rows, ignore_index=True)
    else:
        prices = df.reset_index()
        prices["ticker"] = tickers[0]

    prices.rename(columns={"Date": "date"}, inplace=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    # Standardize column names
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    prices.rename(columns=col_map, inplace=True)

    prices["source"] = "yfinance"
    return prices[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., config/pilot.yaml)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = cfg["project"]["data_dir"]
    start = cfg["window"]["start"]
    end = cfg["window"]["end"]
    tickers = [t["symbol"] for t in cfg["tickers"]]

    dirs = run_dirs(data_dir, run_id)
    raw_dir = ensure_dir(dirs["raw"] / "yfinance")
    proc_dir = ensure_dir(dirs["processed"])

    prices = download_prices(tickers, start, end)

    # Save processed
    out_parquet = proc_dir / "prices_daily.parquet"
    prices.to_parquet(out_parquet, index=False)

    # Save a small raw-ish CSV snapshot for quick inspection
    out_csv = raw_dir / f"prices_{start}_to_{end}.csv".replace(":", "-")
    prices.to_csv(out_csv, index=False)

    print(f"[OK] Saved {len(prices):,} rows -> {out_parquet}")
    print(f"[OK] Snapshot CSV -> {out_csv}")


if __name__ == "__main__":
    main()
