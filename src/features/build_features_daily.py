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
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = Path(cfg["project"]["data_dir"])
    processed = data_dir / "processed" / run_id

    prices_path = processed / "prices_daily.parquet"
    news_daily_path = processed / "news_daily.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing: {prices_path}")
    if not news_daily_path.exists():
        raise FileNotFoundError(f"Missing: {news_daily_path}")

    prices = pd.read_parquet(prices_path)
    news_daily = pd.read_parquet(news_daily_path)

    # --- column detection (tolerant) ---
    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    px_close = _find_col(prices, ["adj_close", "close"])
    px_vol = _find_col(prices, ["volume", "vol"])

    if not (px_ticker and px_date and px_close):
        raise ValueError(f"prices_daily schema unexpected. cols={list(prices.columns)}")

    # --- normalize prices ---
    prices = prices.copy()
    prices["ticker"] = prices[px_ticker].astype(str)
    prices["date"] = pd.to_datetime(prices[px_date], errors="coerce").dt.date
    prices["close"] = pd.to_numeric(prices[px_close], errors="coerce")
    prices["volume"] = pd.to_numeric(prices[px_vol], errors="coerce") if px_vol else np.nan

    prices = prices.dropna(subset=["ticker", "date"]).copy()

    # Guard: no duplicate (ticker,date)
    # (If yfinance ever produces dupes, keep the last row deterministically.)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)

    # Guard: ensure sorted within ticker
    # (Rolling assumes correct chronological ordering.)
    # This is a cheap sanity check:
    if not prices.sort_values(["ticker", "date"]).index.equals(prices.index):
        raise AssertionError("prices not sorted by ticker,date after sorting step (unexpected).")

    # --- rolling features (NO LOOKAHEAD) ---
    # Use transform/apply patterns so index matches the frame index.
    g_close = prices.groupby("ticker")["close"]
    prices["px_ret_1d"] = g_close.pct_change()               # uses t and t-1 only
    prices["px_ret_5d"] = g_close.pct_change(5)              # uses t and t-5 only

    # volatility over last 5 returns up to t (not centered => no future)
    # min_periods=5 avoids partial-window leakage-like noise; can relax if you prefer.
    prices["px_volatility_5d"] = prices.groupby("ticker")["px_ret_1d"].transform(
        lambda s: s.rolling(window=5, min_periods=5).std()
    )

    # volume change uses only past value
    prices["px_volume_chg_1d"] = prices.groupby("ticker")["volume"].pct_change()

    feats = prices[
        ["ticker", "date", "close", "volume", "px_ret_1d", "px_ret_5d", "px_volatility_5d", "px_volume_chg_1d"]
    ].copy()
    feats = feats.rename(columns={"close": "px_close", "volume": "px_volume"})

    # --- merge Guardian-compliant daily aggregates (no headline text needed) ---
    nd_ticker = _find_col(news_daily, ["ticker", "symbol"])
    nd_date = _find_col(news_daily, ["assigned_date", "date", "effective_date", "local_date"])
    if not (nd_ticker and nd_date):
        raise ValueError(f"news_daily schema unexpected. cols={list(news_daily.columns)}")

    news_daily = news_daily.copy()
    news_daily["ticker"] = news_daily[nd_ticker].astype(str)
    news_daily["date"] = pd.to_datetime(news_daily[nd_date], errors="coerce").dt.date

    n_col = _find_col(news_daily, ["n_articles", "n_items", "count", "n"])
    vader_mean = _find_col(news_daily, ["sent_vader_mean", "vader_mean", "mean_sent_vader"])
    vader_sum = _find_col(news_daily, ["sent_vader_sum", "vader_sum", "sum_sent_vader"])
    finbert_pos = _find_col(news_daily, ["finbert_pos_mean"])
    finbert_neg = _find_col(news_daily, ["finbert_neg_mean"])
    finbert_neu = _find_col(news_daily, ["finbert_neu_mean"])
    finbert_score = _find_col(news_daily, ["finbert_score_mean"])

    if not n_col:
        news_daily["n_articles"] = 1
        n_col = "n_articles"
    if not vader_mean:
        news_daily["sent_vader_mean"] = np.nan
        vader_mean = "sent_vader_mean"
    if not vader_sum:
        news_daily["sent_vader_sum"] = np.nan
        vader_sum = "sent_vader_sum"

    keep_cols = ["ticker", "date", n_col, vader_mean, vader_sum]
    rename_map = {
        n_col: "n_articles",
        vader_mean: "sent_vader_mean",
        vader_sum: "sent_vader_sum",
    }
    for col_name, out_name in [
        (finbert_pos, "finbert_pos_mean"),
        (finbert_neg, "finbert_neg_mean"),
        (finbert_neu, "finbert_neu_mean"),
        (finbert_score, "finbert_score_mean"),
    ]:
        if col_name:
            keep_cols.append(col_name)
            rename_map[col_name] = out_name

    nd_small = news_daily[keep_cols].rename(columns=rename_map)

    feats = feats.sort_values(["ticker", "date"]).reset_index(drop=True)
    nd_small = nd_small.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Guard: news_daily should already be aggregated to one row per (ticker,date)
    # If duplicates slip through, keep the last row deterministically.
    nd_small = nd_small.drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)

    # Prices define the full trading calendar; news is left-joined.
    feats = feats.merge(
        nd_small,
        on=["ticker", "date"],
        how="left",
        suffixes=("", "_news"),
    )

    # Encode missing-news explicitly
    feats["n_articles"] = pd.to_numeric(feats["n_articles"], errors="coerce").fillna(0).astype(int)
    feats["has_news"] = (feats["n_articles"] > 0).astype(int)

    # Fill sentiment aggregates with 0 on no-news days (paired with has_news/n_articles)
    feats["sent_vader_mean"] = pd.to_numeric(feats["sent_vader_mean"], errors="coerce").fillna(0.0)
    feats["sent_vader_sum"] = pd.to_numeric(feats["sent_vader_sum"], errors="coerce").fillna(0.0)

    for c in ["finbert_pos_mean", "finbert_neg_mean", "finbert_neu_mean", "finbert_score_mean"]:
        if c not in feats.columns:
            feats[c] = np.nan
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

    # Guardrail: ensure no duplicate ticker-date rows after merge
    assert not feats.duplicated(["ticker", "date"]).any(), "Duplicate rows after merge"

    out_path = processed / "features_daily.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
