from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import requests

from src.common.config import load_cfg
from src.common.paths import ensure_dir, run_dirs


GUARDIAN_ITEM_URL = "https://content.guardianapis.com/{guardian_id}"


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    px_close = _find_col(prices, ["adj_close", "close"])

    if not (px_ticker and px_date and px_close):
        raise ValueError(f"prices_daily schema unexpected. cols={list(prices.columns)}")

    out = prices.copy()
    out["ticker"] = out[px_ticker].astype(str)
    out["trade_date"] = _to_date_series(out[px_date])
    out["close"] = pd.to_numeric(out[px_close], errors="coerce")

    out = (
        out.dropna(subset=["ticker", "trade_date", "close"])
        .sort_values(["ticker", "trade_date"])
        .drop_duplicates(["ticker", "trade_date"], keep="last")
        .reset_index(drop=True)
    )

    g = out.groupby("ticker", group_keys=False)
    out["same_day_return"] = g["close"].pct_change()
    out["next_day_return"] = g["same_day_return"].shift(-1)

    return out[["ticker", "trade_date", "close", "same_day_return", "next_day_return"]].copy()


def _prepare_items(news_items: pd.DataFrame, manifest: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ni = news_items.copy()

    col_ticker = _find_col(ni, ["ticker"])
    col_provider = _find_col(ni, ["provider"])
    col_guardian_id = _find_col(ni, ["guardian_id", "id"])
    col_hash = _find_col(ni, ["item_hash", "sha1"])
    col_url = _find_col(ni, ["web_url", "url"])
    col_pub_utc = _find_col(ni, ["published_at_utc"])
    col_pub_lon = _find_col(ni, ["published_at_london"])
    col_assigned = _find_col(ni, ["assigned_date", "effective_date", "date"])
    col_query = _find_col(ni, ["query_expr", "query_used", "query"])
    col_sent = _find_col(ni, ["sent_vader", "sentiment_score", "vader_score"])
    col_label = _find_col(ni, ["vader_label", "sentiment_label"])

    if not (col_ticker and col_pub_lon and col_assigned):
        raise ValueError(f"news_items schema unexpected. cols={list(ni.columns)}")

    # Optional manifest enrichment if some fields are missing
    if manifest is not None and not manifest.empty:
        mf = manifest.copy()

        mf_guardian = _find_col(mf, ["guardian_id"])
        mf_hash = _find_col(mf, ["item_hash", "sha1"])
        mf_url = _find_col(mf, ["web_url", "url"])
        mf_pub_lon = _find_col(mf, ["published_at_london"])
        mf_assigned = _find_col(mf, ["assigned_date", "effective_date"])

        join_keys = []
        if col_guardian_id and mf_guardian:
            join_keys = [(col_guardian_id, mf_guardian)]
        elif col_hash and mf_hash:
            join_keys = [(col_hash, mf_hash)]

        if join_keys:
            left_key, right_key = join_keys[0]
            keep_cols = [c for c in [right_key, mf_url, mf_pub_lon, mf_assigned] if c]
            mf_small = mf[keep_cols].drop_duplicates(subset=[right_key]).copy()
            ni = ni.merge(mf_small, left_on=left_key, right_on=right_key, how="left", suffixes=("", "_manifest"))

            if not col_url and mf_url:
                col_url = f"{mf_url}_manifest" if f"{mf_url}_manifest" in ni.columns else mf_url

    out = pd.DataFrame()
    out["ticker"] = ni[col_ticker].astype(str)
    out["provider"] = ni[col_provider] if col_provider else "guardian"
    out["guardian_id"] = ni[col_guardian_id] if col_guardian_id else ""
    out["item_hash"] = ni[col_hash] if col_hash else ""
    out["url"] = ni[col_url] if col_url else ""
    out["published_at_utc"] = pd.to_datetime(ni[col_pub_utc], errors="coerce", utc=True) if col_pub_utc else pd.NaT
    out["published_at_london"] = pd.to_datetime(ni[col_pub_lon], errors="coerce")
    out["assigned_date"] = _to_date_series(ni[col_assigned])
    out["local_date"] = out["published_at_london"].dt.date
    out["query_expr"] = ni[col_query] if col_query else ""
    out["sentiment_score"] = pd.to_numeric(ni[col_sent], errors="coerce") if col_sent else np.nan
    out["sentiment_label"] = ni[col_label] if col_label else ""

    # Keep only valid rows for inspection
    out = out.dropna(subset=["ticker", "published_at_london", "assigned_date"]).reset_index(drop=True)
    return out


def _align_items_to_prices(items: pd.DataFrame, prices_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Map each item's assigned_date to the first trading date on or after assigned_date
    for the same ticker. This is for inspection only and does not alter upstream model logic.
    """
    out_parts = []

    prices_ret = prices_ret.copy()
    prices_ret["trade_date_ts"] = pd.to_datetime(prices_ret["trade_date"])
    items = items.copy()
    items["assigned_date_ts"] = pd.to_datetime(items["assigned_date"])

    for ticker, item_sub in items.groupby("ticker", sort=False):
        px_sub = prices_ret.loc[prices_ret["ticker"] == ticker].sort_values("trade_date_ts").reset_index(drop=True)
        item_sub = item_sub.sort_values("assigned_date_ts").reset_index(drop=True).copy()

        if px_sub.empty:
            item_sub["aligned_trade_date"] = pd.NaT
            item_sub["same_day_return"] = np.nan
            item_sub["next_day_return"] = np.nan
            item_sub["assigned_to_trade_gap_days"] = np.nan
            out_parts.append(item_sub)
            continue

        px_dates = px_sub["trade_date_ts"].to_numpy(dtype="datetime64[ns]")
        item_dates = item_sub["assigned_date_ts"].to_numpy(dtype="datetime64[ns]")

        idx = np.searchsorted(px_dates, item_dates, side="left")
        valid = idx < len(px_sub)

        aligned_trade_date = np.array([np.datetime64("NaT")] * len(item_sub), dtype="datetime64[ns]")
        same_day = np.full(len(item_sub), np.nan)
        next_day = np.full(len(item_sub), np.nan)

        valid_idx = np.where(valid)[0]
        px_idx = idx[valid]
        aligned_trade_date[valid_idx] = px_dates[px_idx]
        same_day[valid_idx] = px_sub["same_day_return"].to_numpy()[px_idx]
        next_day[valid_idx] = px_sub["next_day_return"].to_numpy()[px_idx]

        item_sub["aligned_trade_date"] = pd.to_datetime(aligned_trade_date).date
        item_sub["same_day_return"] = same_day
        item_sub["next_day_return"] = next_day

        aligned_ts = pd.to_datetime(item_sub["aligned_trade_date"], errors="coerce")
        item_sub["assigned_to_trade_gap_days"] = (aligned_ts - item_sub["assigned_date_ts"]).dt.days

        out_parts.append(item_sub)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else items.copy()
    return out


def _fetch_guardian_headline(guardian_id: str, api_key: str, session: requests.Session) -> str:
    if not guardian_id:
        return ""

    url = GUARDIAN_ITEM_URL.format(guardian_id=guardian_id)
    params = {
        "api-key": api_key,
        "show-fields": "headline",
    }

    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    content = (data.get("response", {}) or {}).get("content", {}) or {}
    fields = content.get("fields", {}) or {}
    return str(fields.get("headline") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--run-id", default=None, help="Optional override for run_id")
    ap.add_argument("--with-headlines-temp", action="store_true", help="Create ephemeral TEMP CSV with headline text re-fetched at runtime")
    ap.add_argument("--purge-temp", action="store_true", help="Delete TEMP CSV after writing it (useful for one-off checks)")
    ap.add_argument("--sleep-seconds", type=float, default=1.05, help="Delay between Guardian item fetches for TEMP headline reconstruction")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_id = args.run_id or cfg["project"]["run_id"]
    data_dir = cfg["project"]["data_dir"]

    dirs = run_dirs(data_dir, run_id)
    processed = dirs["processed"]
    inspection_dir = ensure_dir(processed / "inspection")

    news_items_path = processed / "news_items.parquet"
    news_daily_path = processed / "news_daily.parquet"         # read-presence only
    manifest_path = processed / "manifest_news.csv"
    prices_path = processed / "prices_daily.parquet"
    features_path = processed / "features_daily.parquet"       # optional / informational
    model_path = processed / "model_dataset.parquet"           # optional / informational

    if not news_items_path.exists():
        raise FileNotFoundError(f"Missing: {news_items_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing: {prices_path}")

    news_items = pd.read_parquet(news_items_path)
    prices = pd.read_parquet(prices_path)
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None

    # Optional existence checks only, so you can see current stage coverage in logs if needed
    _ = news_daily_path.exists()
    _ = features_path.exists()
    _ = model_path.exists()

    items = _prepare_items(news_items, manifest=manifest)
    prices_ret = _prepare_prices(prices)
    inspection = _align_items_to_prices(items, prices_ret)

    inspection["manual_relevance"] = ""
    inspection["review_notes"] = ""

    # Column order for human inspection
    desired_cols = [
        "ticker",
        "provider",
        "guardian_id",
        "item_hash",
        "url",
        "published_at_utc",
        "published_at_london",
        "local_date",
        "assigned_date",
        "aligned_trade_date",
        "assigned_to_trade_gap_days",
        "query_expr",
        "sentiment_score",
        "sentiment_label",
        "same_day_return",
        "next_day_return",
        "manual_relevance",
        "review_notes",
    ]
    inspection = inspection[[c for c in desired_cols if c in inspection.columns]].copy()
    inspection = inspection.sort_values(["ticker", "aligned_trade_date", "published_at_london"]).reset_index(drop=True)

    out_csv = inspection_dir / "inspection_items.csv"
    inspection.to_csv(out_csv, index=False)

    print(f"[OK] read: {news_items_path}")
    if manifest_path.exists():
        print(f"[OK] read: {manifest_path}")
    print(f"[OK] read: {prices_path}")
    print(f"[OK] wrote: {out_csv}")

    if args.with_headlines_temp:
        api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GUARDIAN_API_KEY is required for --with-headlines-temp")

        session = requests.Session()
        temp_df = inspection.copy()
        headlines = []

        for i, gid in enumerate(temp_df["guardian_id"].fillna("").astype(str).tolist(), start=1):
            try:
                headline = _fetch_guardian_headline(gid, api_key, session)
            except Exception:
                headline = ""
            headlines.append(headline)

            if i < len(temp_df):
                time.sleep(args.sleep_seconds)

        temp_df["headline_text"] = headlines

        temp_path = inspection_dir / "inspection_items_with_headlines_TEMP.csv"
        temp_df.to_csv(temp_path, index=False)
        print(f"[OK] wrote TEMP file: {temp_path}")

        if args.purge_temp:
            temp_path.unlink(missing_ok=True)
            print(f"[OK] purged TEMP file: {temp_path}")


if __name__ == "__main__":
    main()