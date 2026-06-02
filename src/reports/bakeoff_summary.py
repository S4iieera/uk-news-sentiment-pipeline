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


def _read_candidate_summary(cfg_path: str, candidate_ticker: str) -> dict:
    cfg = load_cfg(cfg_path)
    run_id = cfg["project"]["run_id"]
    data_dir = Path(cfg["project"]["data_dir"])
    proc = data_dir / "processed" / run_id

    prices_path = proc / "prices_daily.parquet"
    news_daily_path = proc / "news_daily.parquet"
    news_items_path = proc / "news_items.parquet"
    inspection_path = proc / "inspection" / "inspection_items.csv"

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing: {prices_path}")
    if not news_daily_path.exists():
        raise FileNotFoundError(f"Missing: {news_daily_path}")

    prices = pd.read_parquet(prices_path)
    news_daily = pd.read_parquet(news_daily_path)

    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    if not (px_ticker and px_date):
        raise ValueError(f"prices_daily schema unexpected. cols={list(prices.columns)}")

    nd_ticker = _find_col(news_daily, ["ticker", "symbol"])
    nd_date = _find_col(news_daily, ["date", "assigned_date", "effective_date", "local_date"])
    nd_n = _find_col(news_daily, ["n_articles", "n_items", "count", "n"])
    if not (nd_ticker and nd_date):
        raise ValueError(f"news_daily schema unexpected. cols={list(news_daily.columns)}")
    if not nd_n:
        news_daily["_n_articles_tmp"] = 1
        nd_n = "_n_articles_tmp"

    px = prices.copy()
    px["ticker"] = px[px_ticker].astype(str)
    px["date"] = pd.to_datetime(px[px_date], errors="coerce").dt.date
    px = px.loc[px["ticker"] == candidate_ticker].copy()

    nd = news_daily.copy()
    nd["ticker"] = nd[nd_ticker].astype(str)
    nd["date"] = pd.to_datetime(nd[nd_date], errors="coerce").dt.date
    nd["n_articles"] = pd.to_numeric(nd[nd_n], errors="coerce").fillna(0)
    nd = nd.loc[nd["ticker"] == candidate_ticker].copy()

    trading_days = int(px["date"].nunique())

    nd_news = nd.loc[nd["n_articles"] > 0].copy()
    days_with_news = int(nd_news["date"].nunique())
    coverage_pct = float(days_with_news / trading_days) if trading_days > 0 else 0.0

    median_articles_per_news_day = float(nd_news["n_articles"].median()) if not nd_news.empty else 0.0
    total_articles = int(nd_news["n_articles"].sum()) if not nd_news.empty else 0

    query_expr = ""
    item_rows = 0
    if news_items_path.exists():
        ni = pd.read_parquet(news_items_path)
        ni_ticker = _find_col(ni, ["ticker"])
        ni_query = _find_col(ni, ["query_expr", "query_used", "query"])
        if ni_ticker:
            ni = ni.loc[ni[ni_ticker].astype(str) == candidate_ticker].copy()
            item_rows = int(len(ni))
            if ni_query and not ni.empty:
                uniq = ni[ni_query].dropna().astype(str).unique().tolist()
                query_expr = " | ".join(uniq)

    inspection_rows = 0
    if inspection_path.exists():
        insp = pd.read_csv(inspection_path)
        if "ticker" in insp.columns:
            inspection_rows = int((insp["ticker"].astype(str) == candidate_ticker).sum())

    return {
        "run_id": run_id,
        "candidate_ticker": candidate_ticker,
        "trading_days": trading_days,
        "days_with_news": days_with_news,
        "coverage_pct": round(coverage_pct, 4),
        "median_articles_per_news_day": round(median_articles_per_news_day, 4),
        "total_articles": total_articles,
        "news_item_rows": item_rows,
        "inspection_rows": inspection_rows,
        "query_expr": query_expr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", required=True, help="e.g. config/pilot_5tick_bakeoff_dge.yaml")
    ap.add_argument("--ticker-a", required=True, help="candidate ticker in config-a, e.g. DGE.L")
    ap.add_argument("--config-b", required=True, help="e.g. config/pilot_5tick_bakeoff_expn.yaml")
    ap.add_argument("--ticker-b", required=True, help="candidate ticker in config-b, e.g. EXPN.L")
    args = ap.parse_args()

    row_a = _read_candidate_summary(args.config_a, args.ticker_a)
    row_b = _read_candidate_summary(args.config_b, args.ticker_b)

    out = pd.DataFrame([row_a, row_b])

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "bakeoff_summary.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()