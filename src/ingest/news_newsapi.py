"""
DEPRECATED / ARCHIVED ATTEMPT
NewsAPI returns HTTP 426 (Upgrade Required) due to plan restriction.
Kept for evidence and progression.
"""


from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
import os

from src.common.config import load_yaml
from src.common.hashing import sha1_text
from src.common.paths import run_dirs, ensure_dir


def parse_cutoff(cutoff_str: str) -> dtime:
    hh, mm = cutoff_str.split(":")
    return dtime(int(hh), int(mm))


def compute_effective_date(published_london: pd.Series, cutoff: dtime) -> pd.Series:
    """
    published_london: tz-aware datetime series in Europe/London
    Rule: headlines strictly AFTER cutoff (e.g., > 16:30) get assigned to t+1.
    """
    # Normalize to local midnight (still tz-aware), then add 1 day if time > cutoff.
    local_midnight = published_london.dt.normalize()
    after_cutoff = published_london.dt.time > cutoff
    effective_midnight = local_midnight + pd.to_timedelta(after_cutoff.astype(int), unit="D")
    return effective_midnight.dt.date


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


@dataclass
class NewsAPIConfig:
    endpoint: str
    language: str
    page_size: int
    sort_by: str


def fetch_newsapi_everything(
    api_key: str,
    cfg: NewsAPIConfig,
    query: str,
    date_from: str,
    date_to: str,
    max_pages: int = 50,
    sleep_s: float = 1.1,
) -> list[dict[str, Any]]:
    """
    Fetch pages from NewsAPI Everything endpoint.
    Returns list of page payloads (raw JSON dicts).
    """
    headers = {"X-Api-Key": api_key}
    all_pages: list[dict[str, Any]] = []

    page = 1
    while page <= max_pages:
        params = {
            "q": query,
            "from": date_from,
            "to": date_to,
            "language": cfg.language,
            "pageSize": cfg.page_size,
            "page": page,
            "sortBy": cfg.sort_by,
        }
        resp = requests.get(cfg.endpoint, headers=headers, params=params, timeout=30)

        # basic backoff handling
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = min(30, 2 ** min(page, 5))
            time.sleep(wait)
            continue

        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "ok":
            raise RuntimeError(f"NewsAPI returned non-ok status: {payload}")

        all_pages.append(payload)

        total_results = int(payload.get("totalResults", 0))
        articles = payload.get("articles", [])
        if not articles:
            break

        # Stop when we’ve fetched all pages (best-effort)
        fetched_so_far = page * cfg.page_size
        if fetched_so_far >= total_results:
            break

        page += 1
        time.sleep(sleep_s)

    return all_pages


def pages_to_clean_df(
    pages: list[dict[str, Any]],
    ticker: str,
    provider: str,
    timezone_str: str,
    cutoff_time: str,
    query_used: str,
) -> pd.DataFrame:
    cutoff = parse_cutoff(cutoff_time)

    rows = []
    for page_idx, payload in enumerate(pages, start=1):
        for a in payload.get("articles", []):
            title = (a.get("title") or "").strip()
            source_name = ((a.get("source") or {}).get("name") or "").strip()
            url = (a.get("url") or "").strip()
            published_at = (a.get("publishedAt") or "").strip()

            if not title or not published_at:
                continue

            key = f"{title}|{source_name}|{published_at}"
            h = sha1_text(key)

            rows.append(
                {
                    "ticker": ticker,
                    "title": title,
                    "source": source_name,
                    "url": url,
                    "published_at": published_at,  # raw string
                    "sha1": h,
                    "provider": provider,
                    "query": query_used,
                    "page": page_idx,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Parse datetime UTC then convert to Europe/London
    df["published_at_utc"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at_utc"]).copy()
    df["published_at_london"] = df["published_at_utc"].dt.tz_convert(timezone_str)

    # Local date + effective date (cutoff rule)
    df["local_date"] = df["published_at_london"].dt.date
    df["effective_date"] = compute_effective_date(df["published_at_london"], cutoff)

    # De-dup within run by sha1
    df = df.drop_duplicates(subset=["sha1"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., config/pilot.yaml)")
    ap.add_argument("--max-pages", type=int, default=50, help="Safety cap for NewsAPI pages per ticker")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not found. Put it in .env (and do not commit).")

    cfg = load_yaml(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = cfg["project"]["data_dir"]
    tz = cfg["project"]["timezone"]
    cutoff_time = cfg["project"]["cutoff_time"]
    start = cfg["window"]["start"]
    end = cfg["window"]["end"]

    provider_cfg = cfg["providers"]["newsapi"]
    newsapi_cfg = NewsAPIConfig(
        endpoint=provider_cfg["endpoint"],
        language=provider_cfg["language"],
        page_size=int(provider_cfg["page_size"]),
        sort_by=provider_cfg["sort_by"],
    )

    dirs = run_dirs(data_dir, run_id)
    raw_dir = ensure_dir(dirs["raw"] / "newsapi")
    proc_dir = ensure_dir(dirs["processed"])

    all_clean = []
    for t in cfg["tickers"]:
        symbol = t["symbol"]
        query = t["news_query"]

        pages = fetch_newsapi_everything(
            api_key=api_key,
            cfg=newsapi_cfg,
            query=query,
            date_from=start,
            date_to=end,
            max_pages=args.max_pages,
        )

        # Save raw pages
        stamp = f"{symbol}_{start}_to_{end}"
        out_raw = raw_dir / f"{safe_filename(stamp)}.json"
        with out_raw.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ticker": symbol,
                    "query": query,
                    "from": start,
                    "to": end,
                    "pages": pages,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        clean_df = pages_to_clean_df(
            pages=pages,
            ticker=symbol,
            provider="newsapi",
            timezone_str=tz,
            cutoff_time=cutoff_time,
            query_used=query,
        )
        all_clean.append(clean_df)

        print(f"[OK] {symbol}: pages={len(pages)} rows_clean={len(clean_df):,} raw_saved={out_raw.name}")

    combined = pd.concat(all_clean, ignore_index=True) if all_clean else pd.DataFrame()
    out_parquet = proc_dir / "news_clean.parquet"
    combined.to_parquet(out_parquet, index=False)

    print(f"[OK] Saved combined news -> {out_parquet} (rows={len(combined):,})")


if __name__ == "__main__":
    main()
