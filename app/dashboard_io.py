from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "model_v2_core3_1y_vader_finbert_guardian"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_processed_dir() -> Path:
    return repo_root() / "data" / "processed"


def reports_dir() -> Path:
    return repo_root() / "reports"


def run_dir(run_id: str) -> Path:
    return data_processed_dir() / run_id


def list_available_runs() -> list[str]:
    """List completed run folders without executing any pipeline step."""
    base = data_processed_dir()
    if not base.exists():
        return []

    runs = []
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        if (path / "run_metadata.json").exists() or (path / "prices_daily.parquet").exists():
            runs.append(path.name)
    return runs


def default_run_id(runs: list[str]) -> str | None:
    if DEFAULT_RUN_ID in runs:
        return DEFAULT_RUN_ID
    return runs[0] if runs else None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        found = lowered.get(candidate.lower())
        if found:
            return found
    return None


def load_run_metadata(run_id: str) -> dict[str, Any] | None:
    return read_json(run_dir(run_id) / "run_metadata.json")


def infer_run_info(run_id: str) -> dict[str, Any]:
    """Best-effort run summary from metadata and run-specific parquet files."""
    info: dict[str, Any] = {
        "run_id": run_id,
        "provider": None,
        "timezone": None,
        "cutoff_time": None,
        "tickers": [],
        "window_start": None,
        "window_end": None,
        "sentiment": {},
    }

    meta = load_run_metadata(run_id)
    if meta:
        info["provider"] = meta.get("provider")
        info["timezone"] = meta.get("timezone")
        info["cutoff_time"] = meta.get("cutoff_time") or meta.get("cutoff_local_time")
        info["window_start"] = meta.get("start_date")
        info["window_end"] = meta.get("end_date")
        if isinstance(meta.get("tickers"), list):
            info["tickers"] = meta["tickers"]
        if isinstance(meta.get("sentiment"), dict):
            info["sentiment"] = meta["sentiment"]

    prices = read_parquet(run_dir(run_id) / "prices_daily.parquet")
    if prices is not None and not prices.empty:
        ticker_col = _find_col(prices, ["ticker", "symbol"])
        date_col = _find_col(prices, ["date", "trade_date", "datetime"])
        if ticker_col:
            info["tickers"] = sorted(prices[ticker_col].dropna().astype(str).unique().tolist())
        if date_col:
            dates = pd.to_datetime(prices[date_col], errors="coerce").dropna()
            if not dates.empty:
                info["window_start"] = str(dates.min().date())
                info["window_end"] = str(dates.max().date())

    news_items = read_parquet(run_dir(run_id) / "news_items.parquet")
    if news_items is not None and not news_items.empty:
        provider_col = _find_col(news_items, ["provider"])
        if provider_col and info["provider"] is None:
            providers = news_items[provider_col].dropna().astype(str).unique().tolist()
            info["provider"] = providers[0] if providers else None

    return info


def compute_coverage_for_run(run_id: str) -> pd.DataFrame | None:
    """Compute coverage from run-specific artefacts instead of global reports.

    Coverage is measured against the price trading calendar: a news day only counts
    if the ticker/date pair exists in prices_daily.parquet.
    """
    rdir = run_dir(run_id)
    prices = read_parquet(rdir / "prices_daily.parquet")
    news_daily = read_parquet(rdir / "news_daily.parquet")

    if prices is None or prices.empty:
        return None

    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    if not px_ticker or not px_date:
        return None

    px = prices.copy()
    px["ticker"] = px[px_ticker].astype(str)
    px["date"] = pd.to_datetime(px[px_date], errors="coerce").dt.date
    px_pairs = px.dropna(subset=["date"])[["ticker", "date"]].drop_duplicates()
    trading = px_pairs.groupby("ticker")["date"].nunique().rename("price_days")

    if news_daily is None or news_daily.empty:
        news_pairs = pd.DataFrame(columns=["ticker", "date"])
    else:
        nd_ticker = _find_col(news_daily, ["ticker", "symbol"])
        nd_date = _find_col(news_daily, ["date", "assigned_date", "effective_date", "local_date"])
        nd_n = _find_col(news_daily, ["n_articles", "n_items", "count", "n"])
        if not nd_ticker or not nd_date:
            news_pairs = pd.DataFrame(columns=["ticker", "date"])
        else:
            nd = news_daily.copy()
            nd["ticker"] = nd[nd_ticker].astype(str)
            nd["date"] = pd.to_datetime(nd[nd_date], errors="coerce").dt.date
            if nd_n:
                nd["n_articles"] = pd.to_numeric(nd[nd_n], errors="coerce").fillna(0)
                nd = nd.loc[nd["n_articles"] > 0].copy()
            news_pairs = nd.dropna(subset=["date"])[["ticker", "date"]].drop_duplicates()

    if news_pairs.empty:
        news = pd.Series(dtype="int64", name="news_days")
    else:
        aligned_news_pairs = news_pairs.merge(px_pairs, on=["ticker", "date"], how="inner")
        news = aligned_news_pairs.groupby("ticker")["date"].nunique().rename("news_days")

    out = pd.concat([trading, news], axis=1).fillna(0).reset_index()
    out["price_days"] = out["price_days"].astype(int)
    out["news_days"] = out["news_days"].astype(int)
    out["coverage_pct"] = (
        out["news_days"].div(out["price_days"]).where(out["price_days"] > 0, 0).mul(100).round(2)
    )
    return out.sort_values("ticker").reset_index(drop=True)

def load_global_coverage_report() -> pd.DataFrame | None:
    return read_csv(reports_dir() / "coverage_report.csv")


def load_global_metrics() -> pd.DataFrame | None:
    return read_csv(reports_dir() / "pilot_metrics_table.csv")


def load_prices(run_id: str) -> pd.DataFrame | None:
    return read_parquet(run_dir(run_id) / "prices_daily.parquet")


def load_features(run_id: str) -> pd.DataFrame | None:
    return read_parquet(run_dir(run_id) / "features_daily.parquet")


def load_model_dataset(run_id: str) -> pd.DataFrame | None:
    return read_parquet(run_dir(run_id) / "model_dataset.parquet")


def load_inspection(run_id: str) -> pd.DataFrame | None:
    return read_csv(run_dir(run_id) / "inspection" / "inspection_items.csv")


def get_metrics_for_display(run_id: str) -> tuple[pd.DataFrame | None, str]:
    """Load global/latest metrics and explain whether run matching is verifiable."""
    metrics = load_global_metrics()
    if metrics is None or metrics.empty:
        return None, "No reports/pilot_metrics_table.csv found."

    if "run_id" in metrics.columns:
        run_ids = metrics["run_id"].dropna().astype(str).unique().tolist()
        if len(run_ids) == 1 and run_ids[0] == run_id:
            return metrics.copy(), "Loaded metrics from reports/pilot_metrics_table.csv; run_id matches selected run."
        if run_ids and run_id not in run_ids:
            return metrics.copy(), (
                "Warning: reports/pilot_metrics_table.csv appears to belong to a different run "
                f"({', '.join(run_ids)}), not the selected run ({run_id})."
            )
        return (
            metrics.loc[metrics["run_id"].astype(str) == run_id].copy(),
            "Loaded matching rows from reports/pilot_metrics_table.csv for the selected run.",
        )

    return metrics.copy(), (
        "Warning: reports/pilot_metrics_table.csv has no run_id column, so it is treated as global/latest only."
    )

