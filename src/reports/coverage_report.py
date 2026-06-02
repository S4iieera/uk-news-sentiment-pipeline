from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.common.config import load_cfg


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _ensure_datetime(df: pd.DataFrame, col: str, tz: Optional[str] = None) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce")
    # If it's tz-naive and a tz is provided, localize (common for "date" columns)
    if tz and getattr(s.dt, "tz", None) is None:
        try:
            s = s.dt.tz_localize(tz)
        except Exception:
            # If already tz-aware or localization fails, keep as-is
            pass
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = Path(cfg["project"]["data_dir"])
    tz = cfg["project"].get("timezone", "Europe/London")

    processed = data_dir / "processed" / run_id
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    prices_path = processed / "prices_daily.parquet"
    news_daily_path = processed / "news_daily.parquet"

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing: {prices_path}")
    if not news_daily_path.exists():
        raise FileNotFoundError(f"Missing: {news_daily_path}")

    prices = pd.read_parquet(prices_path)
    news_daily = pd.read_parquet(news_daily_path)

    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    if not px_ticker or not px_date:
        raise ValueError(f"prices_daily schema unexpected. cols={list(prices.columns)}")

    nd_ticker = _find_col(news_daily, ["ticker", "symbol"])
    nd_date = _find_col(news_daily, ["date", "assigned_date", "effective_date", "local_date"])
    nd_n = _find_col(news_daily, ["n_articles", "n_items", "count", "n"])
    if not nd_ticker or not nd_date:
        raise ValueError(f"news_daily schema unexpected. cols={list(news_daily.columns)}")
    if not nd_n:
        # If counts not present, treat any row as "has news"
        news_daily["_n_articles_tmp"] = 1
        nd_n = "_n_articles_tmp"

    prices["_date"] = pd.to_datetime(prices[px_date], errors="coerce").dt.date
    news_daily["_date"] = pd.to_datetime(news_daily[nd_date], errors="coerce").dt.date

    out_rows = []
    for ticker in sorted(set(prices[px_ticker].dropna().astype(str))):
        px_dates = sorted(set(prices.loc[prices[px_ticker].astype(str) == ticker, "_date"]))
        if not px_dates:
            continue

        nd = news_daily.loc[news_daily[nd_ticker].astype(str) == ticker].copy()
        nd_has = nd.loc[pd.to_numeric(nd[nd_n], errors="coerce").fillna(0) > 0]
        news_dates = set(nd_has["_date"].dropna().tolist())

        missing = [d for d in px_dates if d not in news_dates]
        cov_pct = 0.0 if len(px_dates) == 0 else (len(px_dates) - len(missing)) / len(px_dates) * 100.0

        out_rows.append(
            {
                "run_id": run_id,
                "ticker": ticker,
                "n_price_days": len(px_dates),
                "n_news_days": len(px_dates) - len(missing),
                "coverage_pct": round(cov_pct, 2),
                "missing_days_count": len(missing),
                "first_price_day": str(px_dates[0]),
                "last_price_day": str(px_dates[-1]),
                "missing_days_sample": ";".join(map(str, missing[:15])),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["coverage_pct", "ticker"], ascending=[True, True])

    csv_path = reports_dir / "coverage_report.csv"
    md_path = reports_dir / "coverage_report.md"
    out.to_csv(csv_path, index=False)

    # short markdown
    lines = []
    lines.append(f"# Coverage report ({run_id})")
    lines.append("")
    lines.append(f"- prices_daily: `{prices_path.as_posix()}`")
    lines.append(f"- news_daily: `{news_daily_path.as_posix()}`")
    lines.append("")
    if out.empty:
        lines.append("No rows found (unexpected).")
    else:
        lines.append(out[["ticker", "n_price_days", "n_news_days", "coverage_pct", "missing_days_count"]].to_markdown(index=False))
        lines.append("")
        worst = out.iloc[0].to_dict()
        lines.append(f"Lowest coverage: **{worst['ticker']}** = {worst['coverage_pct']}% (missing {worst['missing_days_count']} days).")
        if worst.get("missing_days_sample"):
            lines.append(f"Missing sample: {worst['missing_days_sample']}")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
    