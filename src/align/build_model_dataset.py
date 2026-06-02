from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.common.config import load_cfg


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = Path(cfg["project"]["data_dir"])
    processed = data_dir / "processed" / run_id

    features_path = processed / "features_daily.parquet"
    prices_path = processed / "prices_daily.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing: {features_path} (run build_features_daily first)")
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing: {prices_path}")

    feats = pd.read_parquet(features_path)
    prices = pd.read_parquet(prices_path)

    f_ticker = _find_col(feats, ["ticker"])
    f_date = _find_col(feats, ["date"])
    if not (f_ticker and f_date):
        raise ValueError(f"features_daily schema unexpected. cols={list(feats.columns)}")

    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    px_close = _find_col(prices, ["adj_close", "close"])
    if not (px_ticker and px_date and px_close):
        raise ValueError(f"prices_daily schema unexpected. cols={list(prices.columns)}")

    feats = feats.copy()
    feats["ticker"] = feats[f_ticker].astype(str)
    feats["date"] = pd.to_datetime(feats[f_date], errors="coerce").dt.date

    prices = prices.copy()
    prices["ticker"] = prices[px_ticker].astype(str)
    prices["date"] = pd.to_datetime(prices[px_date], errors="coerce").dt.date
    prices["close"] = pd.to_numeric(prices[px_close], errors="coerce")
    prices = prices.sort_values(["ticker", "date"])

    g = prices.groupby("ticker", group_keys=False)
    prices["target_ret_t1"] = g["close"].pct_change().shift(-1)  # return from t to t+1, aligned at t
    prices["target_up_t1"] = (prices["target_ret_t1"] > 0).astype(int)

    targets = prices[["ticker", "date", "target_ret_t1", "target_up_t1"]].copy()

    ds = feats.merge(targets, on=["ticker", "date"], how="left")
    ds = ds.dropna(subset=["target_ret_t1"]).reset_index(drop=True)

    out_path = processed / "model_dataset.parquet"
    ds.to_parquet(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
