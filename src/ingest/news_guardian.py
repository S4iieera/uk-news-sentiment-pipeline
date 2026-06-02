# src/ingest/news_guardian.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

from zoneinfo import ZoneInfo

# Optional .env support (recommended)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


LONDON_TZ = "Europe/London"
DEFAULT_CUTOFF = "16:30"
DEFAULT_BASE_URL = "https://content.guardianapis.com/search"


def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _quote_guardian_term(term: str) -> str:
    t = (term or "").strip()
    if not t:
        return ""
    # strip outer quotes if user already wrote them
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
    # escape quotes for safety
    t = t.replace('"', '\\"')
    return f'"{t}"'


def build_guardian_query_expr(aliases) -> str:
    terms = [_quote_guardian_term(a) for a in _ensure_list(aliases)]
    terms = [t for t in terms if t]
    if not terms:
        raise ValueError("No aliases provided for Guardian query.")
    # Deterministic OR expression
    return "(" + " OR ".join(terms) + ")"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _get_nested(cfg: Dict[str, Any], paths: List[Tuple[str, ...]], default: Any = None) -> Any:
    """
    Try multiple possible key-paths and return the first one found.
    Example paths: [("window","start"), ("start_date",)]
    """
    for p in paths:
        cur: Any = cfg
        ok = True
        for k in p:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_hhmm(s: str) -> Tuple[int, int]:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid cutoff time '{s}'. Expected HH:MM.")
    return int(parts[0]), int(parts[1])


def to_london(dt: datetime) -> datetime:
    """
    Ensure tz-aware and convert to Europe/London.
    Guardian returns ISO timestamps with timezone (often 'Z'); we preserve correctness.
    """
    if dt.tzinfo is None:
        # Treat naive as UTC if it ever occurs (shouldn't for Guardian).
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo(LONDON_TZ))


def assign_date_with_cutoff(published_london: datetime, cutoff_hhmm: str = DEFAULT_CUTOFF) -> date:
    """
    If published strictly AFTER cutoff time (local London), assign to t+1; else assign to same day.
    """
    hh, mm = _parse_hhmm(cutoff_hhmm)
    cutoff_dt = published_london.replace(hour=hh, minute=mm, second=0, microsecond=0)
    base_day = published_london.date()
    if published_london > cutoff_dt:
        return base_day + timedelta(days=1)
    return base_day


def validate_cutoff_assignments(df_items: pd.DataFrame, cutoff_hhmm: str = DEFAULT_CUTOFF) -> None:
    """
    Validates:
    - assigned_date matches cutoff rule based on published_at_london
    Throws ValueError if any violations exist.
    """
    if df_items.empty:
        return

    required = {"published_at_london", "assigned_date"}
    missing = required - set(df_items.columns)
    if missing:
        raise ValueError(f"Missing required columns for validation: {sorted(missing)}")

    bad_rows = []
    for i, row in df_items.iterrows():
        pub = pd.to_datetime(row["published_at_london"], utc=False)
        if getattr(pub, "tzinfo", None) is None:
            # If stored without tz info, interpret as London local
            pub = pub.replace(tzinfo=ZoneInfo(LONDON_TZ))
        else:
            pub = pub.astimezone(ZoneInfo(LONDON_TZ))

        expected = assign_date_with_cutoff(pub.to_pydatetime(), cutoff_hhmm=cutoff_hhmm)
        got = pd.to_datetime(row["assigned_date"]).date()
        if got != expected:
            bad_rows.append((i, row.get("ticker"), pub.isoformat(), str(got), str(expected)))

    if bad_rows:
        msg = "Cutoff assignment violations (idx, ticker, published_london, got, expected):\n"
        msg += "\n".join([str(x) for x in bad_rows[:25]])
        raise ValueError(msg)


@dataclass
class GuardianConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    page_size: int = 200
    max_pages: int = 10
    max_items: int = 10_000
    rate_limit_per_sec: float = 1.0
    max_retries: int = 5
    backoff_base_sec: float = 1.0
    lang: str = "en"
    order_by: str = "newest"
    show_fields: str = "headline"  # keep minimal; headline text only used in-memory


def _ensure_vader() -> "SentimentIntensityAnalyzer":
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        # Check resource availability
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    return SentimentIntensityAnalyzer()


def _safe_pkg_version(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not_installed"


def _resolve_finbert_device(device_pref: str) -> str:
    import torch

    pref = (device_pref or "auto").lower()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _load_finbert(model_id: str, device_pref: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _resolve_finbert_device(device_pref)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    return tokenizer, model, device


def _canonical_finbert_label(label: str) -> str:
    text = str(label).strip().lower()
    if "pos" in text:
        return "positive"
    if "neg" in text:
        return "negative"
    if "neu" in text:
        return "neutral"
    return text


def _finbert_label_order(model) -> List[str]:
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    if id2label:
        ordered = []
        for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0])):
            ordered.append(_canonical_finbert_label(str(label)))
        if len(ordered) >= 3:
            return ordered[:3]
    return ["positive", "negative", "neutral"]


def _score_finbert_batch(texts, tokenizer, model, device: str):
    import torch

    if not texts:
        return []

    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    if device == "cuda":
        enc = {k: v.to("cuda") for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

    labels = _finbert_label_order(model)
    out = []
    for p in probs:
        scores = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }
        for label, prob in zip(labels, p):
            if label in scores:
                scores[label] = float(prob)

        best_idx = int(np.argmax(p))
        best_label = labels[best_idx] if best_idx < len(labels) else "neutral"
        out.append(
            {
                "finbert_label": best_label,
                "finbert_pos": scores["positive"],
                "finbert_neg": scores["negative"],
                "finbert_neu": scores["neutral"],
                "finbert_score": float(scores["positive"] - scores["negative"]),
            }
        )
    return out


def _build_query_for_ticker(t: Dict[str, Any]) -> str:
    # Prefer explicit news_query if present
    nq = t.get("news_query") or t.get("query") or t.get("guardian_query")
    if isinstance(nq, str) and nq.strip():
        return nq.strip()

    aliases = t.get("aliases") or t.get("alias") or t.get("keywords")
    if isinstance(aliases, list) and aliases:
        parts = []
        for a in aliases:
            if not isinstance(a, str):
                continue
            a = a.strip()
            if not a:
                continue
            # Quote multi-word aliases for Guardian q parameter
            if " " in a:
                parts.append(f"\"{a}\"")
            else:
                parts.append(a)
        if parts:
            return " OR ".join(parts)

    # Fallback: ticker itself
    return str(t.get("ticker") or t.get("symbol") or "").strip() or "FTSE"


def _ticker_symbol(t: Dict[str, Any]) -> str:
    return str(t.get("ticker") or t.get("symbol") or t.get("yfinance") or t.get("yf") or "").strip()


def _respect_rate_limit(last_call_ts: float, rate_limit_per_sec: float) -> float:
    # Ensure <= 1/rate seconds between calls
    min_interval = 1.0 / max(rate_limit_per_sec, 0.001)
    now = time.time()
    elapsed = now - last_call_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.time()


def _request_with_retries(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    cfg: GuardianConfig,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    attempt = 0
    last_err: Optional[Exception] = None

    while attempt < cfg.max_retries:
        attempt += 1
        try:
            resp = session.get(url, params=params, timeout=timeout_sec)
            # Respect 429 Retry-After if present
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        time.sleep(float(ra))
                    except Exception:
                        pass
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            # Exponential backoff + small jitter
            sleep_s = cfg.backoff_base_sec * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(min(sleep_s, 20.0))

    raise RuntimeError(f"Guardian request failed after {cfg.max_retries} retries: {last_err}") from last_err


def guardian_search(
    session: requests.Session,
    gcfg: GuardianConfig,
    q: str,
    start_date: date,
    end_date: date,
    max_pages: int,
    max_items: int,
    write_raw_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of raw Guardian result objects (as dicts).
    """
    items: List[Dict[str, Any]] = []
    page = 1
    last_call = 0.0

    while page <= max_pages and len(items) < max_items:
        last_call = _respect_rate_limit(last_call, gcfg.rate_limit_per_sec)

        params = {
            "q": q,
            "from-date": start_date.isoformat(),
            "to-date": end_date.isoformat(),
            "page": page,
            "page-size": min(int(gcfg.page_size), 200),
            "api-key": gcfg.api_key,
            "order-by": gcfg.order_by,
            "lang": gcfg.lang,
            "show-fields": gcfg.show_fields,
        }

        data = _request_with_retries(session, gcfg.base_url, params=params, cfg=gcfg)

        # Optional ephemeral raw write (contains headline text; purge by default)
        if write_raw_dir is not None:
            write_raw_dir.mkdir(parents=True, exist_ok=True)
            out = write_raw_dir / f"search_page_{page:04d}.json"
            out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        resp = data.get("response", {})
        results = resp.get("results", []) or []
        if not results:
            break

        items.extend(results)

        # Stop if Guardian indicates no more pages
        pages_total = resp.get("pages")
        if isinstance(pages_total, int) and page >= pages_total:
            break

        page += 1

    return items[:max_items]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML, e.g. config/pilot.yaml")
    ap.add_argument("--run-id", default=None, help="Optional override for run_id (default: from YAML)")
    ap.add_argument(
        "--write-raw",
        action="store_true",
        help="Write raw Guardian JSON temporarily to data/raw/<run_id>/guardian (contains headline text!)",
    )
    ap.add_argument(
        "--purge-raw",
        action="store_true",
        help="Delete raw Guardian JSON folder at end of run (recommended; use with --write-raw)",
    )
    args = ap.parse_args()

    if load_dotenv is not None:
        load_dotenv()  # load .env if present

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ---- Canonical keys (match prices ingest + tests) ----
    project = cfg.get("project", {}) or {}
    window = cfg.get("window", {}) or {}

    run_id = args.run_id or project.get("run_id")
    if not run_id:
        # Backward-compatible fallback (but your pipeline should use project.run_id)
        run_id = (
            _get_nested(cfg, [("run_id",), ("run", "id"), ("run", "run_id")], default=None)
            or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

    data_dir = Path(project.get("data_dir", "data"))
    tz = str(project.get("timezone", LONDON_TZ))
    cutoff = str(project.get("cutoff_time", DEFAULT_CUTOFF))

    start_s = window.get("start")
    end_s = window.get("end")
    if not start_s or not end_s:
        raise ValueError("Missing window.start/window.end in YAML (expected YYYY-MM-DD).")

    start_d = _parse_yyyy_mm_dd(str(start_s))
    end_d = _parse_yyyy_mm_dd(str(end_s))

    # Canonical run directories
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # Guardian provider config
    api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GUARDIAN_API_KEY. Put it in your .env (not committed) or environment variables.")

    gblock = _get_nested(cfg, [("news", "guardian"), ("guardian",), ("providers", "guardian")], default={}) or {}
    gcfg = GuardianConfig(
        api_key=api_key,
        base_url=str(gblock.get("base_url") or DEFAULT_BASE_URL),
        page_size=int(gblock.get("page_size") or 200),
        max_pages=int(gblock.get("max_pages") or 10),
        max_items=int(gblock.get("max_items") or 10_000),
        rate_limit_per_sec=float(gblock.get("rate_limit_per_sec") or 1.0),
        max_retries=int(_get_nested(gblock, [("retry", "max_retries")], default=5)),
        backoff_base_sec=float(_get_nested(gblock, [("retry", "backoff_base_sec")], default=1.0)),
        lang=str(gblock.get("lang") or "en"),
        order_by=str(gblock.get("order_by") or "newest"),
        show_fields=str(gblock.get("show_fields") or "headline"),
    )

    sentiment_cfg = cfg.get("sentiment", {}) or {}
    use_vader = bool(sentiment_cfg.get("use_vader", True))
    use_finbert = bool(sentiment_cfg.get("use_finbert", False))

    finbert_cfg = sentiment_cfg.get("finbert", {}) or {}
    finbert_model_id = str(finbert_cfg.get("model_id", "ProsusAI/finbert"))
    finbert_batch_size = int(finbert_cfg.get("batch_size", 32))
    finbert_device_pref = str(finbert_cfg.get("device", "auto"))
    finbert_cache_filename = str(finbert_cfg.get("cache_filename", "finbert_cache.parquet"))

    # Tickers
    tickers = _get_nested(cfg, [("tickers",), ("universe", "tickers")], default=None)
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("No tickers found in YAML. Expected top-level 'tickers: [ ... ]' list.")

    # Output dirs
    out_dir = processed_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    finbert_cache_path = out_dir / finbert_cache_filename
    if finbert_cache_path.exists():
        finbert_cache = pd.read_parquet(finbert_cache_path)
    else:
        finbert_cache = pd.DataFrame(
            columns=[
                "item_hash",
                "finbert_model_id",
                "guardian_id",
                "published_at_utc",
                "finbert_label",
                "finbert_pos",
                "finbert_neg",
                "finbert_neu",
                "finbert_score",
            ]
        )

    finbert_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not finbert_cache.empty:
        for _, row in finbert_cache.iterrows():
            finbert_lookup[(str(row["item_hash"]), str(row["finbert_model_id"]))] = {
                "finbert_label": row.get("finbert_label"),
                "finbert_pos": row.get("finbert_pos"),
                "finbert_neg": row.get("finbert_neg"),
                "finbert_neu": row.get("finbert_neu"),
                "finbert_score": row.get("finbert_score"),
            }

    tokenizer = None
    finbert_model = None
    finbert_device = None
    if use_finbert:
        tokenizer, finbert_model, finbert_device = _load_finbert(finbert_model_id, finbert_device_pref)

    raw_out_dir = raw_dir / run_id / "guardian" if args.write_raw else None
    if raw_out_dir is not None:
        raw_out_dir.mkdir(parents=True, exist_ok=True)

    # VADER
    vader = _ensure_vader() if use_vader else None

    session = requests.Session()

    rows: List[Dict[str, Any]] = []
    total_seen = 0

    for t in tickers:
        if not isinstance(t, dict):
            continue
        sym = _ticker_symbol(t)
        if not sym:
            continue

        ticker_cfg = t

        aliases = (
            ticker_cfg.get("aliases")
            or ticker_cfg.get("news_aliases")
            or ticker_cfg.get("guardian_aliases")
        )

        # Backward-compat: if older config had a raw query string, allow it
        raw_q = (
            ticker_cfg.get("news_query")
            or ticker_cfg.get("guardian_q")
            or ticker_cfg.get("query")
            or ticker_cfg.get("guardian_query")
        )

        if raw_q:
            query_expr = str(raw_q).strip()
        else:
            try:
                query_expr = build_guardian_query_expr(aliases)
            except ValueError:
                # Backward-compat safety: preserve baseline behaviour if aliases missing
                query_expr = _build_query_for_ticker(ticker_cfg)

        q = query_expr
        print(f"[guardian] ticker={sym} q={query_expr} window={start_d}..{end_d}")

        raw_results = guardian_search(
            session=session,
            gcfg=gcfg,
            q=q,
            start_date=start_d,
            end_date=end_d,
            max_pages=gcfg.max_pages,
            max_items=max(1, gcfg.max_items - total_seen),
            write_raw_dir=raw_out_dir / sym if raw_out_dir is not None else None,
        )

        for item in raw_results:
            guardian_id = item.get("id")
            web_url = item.get("webUrl")
            web_pub = item.get("webPublicationDate")
            fields = item.get("fields") or {}
            headline = fields.get("headline") or item.get("webTitle") or ""

            if not guardian_id or not web_url or not web_pub:
                continue

            # Parse publication datetime (Guardian returns ISO 8601)
            pub_dt = datetime.fromisoformat(str(web_pub).replace("Z", "+00:00"))
            pub_utc = pub_dt.astimezone(ZoneInfo("UTC"))
            pub_lon = pub_dt.astimezone(ZoneInfo(tz))

            assigned = assign_date_with_cutoff(pub_lon, cutoff_hhmm=cutoff)

            item_hash = _sha1(f"guardian:{guardian_id}:{sym}")
            row = {
                "provider": "guardian",
                "guardian_id": guardian_id,
                "web_url": web_url,
                # store as tz-aware datetimes / dates (NOT strings)
                "published_at_utc": pub_utc,
                "published_at_london": pub_lon,
                "assigned_date": assigned,
                "ticker": sym,
                "query_used": q,
                "query_expr": query_expr,
                "item_hash": item_hash,
            }

            if use_vader and vader is not None:
                scores = vader.polarity_scores(str(headline))
                compound = float(scores.get("compound", 0.0))
                if compound > 0.05:
                    label = "pos"
                elif compound < -0.05:
                    label = "neg"
                else:
                    label = "neu"
                row["sent_vader"] = compound
                row["vader_label"] = label
            else:
                row["sent_vader"] = np.nan
                row["vader_label"] = ""

            if use_finbert:
                cached = finbert_lookup.get((item_hash, finbert_model_id))
                if cached is not None:
                    row.update(cached)
                else:
                    # Keep headline text in memory only long enough to score FinBERT.
                    row["_headline_temp"] = str(headline)

            rows.append(row)

        total_seen += len(raw_results)
        if total_seen >= gcfg.max_items:
            print(f"[guardian] reached max_items={gcfg.max_items}, stopping.")
            break

    df_items = pd.DataFrame(rows)
    if not df_items.empty:
        df_items = df_items.drop_duplicates(subset=["item_hash"]).reset_index(drop=True)
        # Normalize dtypes (helps parquet roundtrip + tz-aware tests)
        df_items["published_at_utc"] = pd.to_datetime(df_items["published_at_utc"], utc=True, errors="raise")
        df_items["published_at_london"] = pd.to_datetime(df_items["published_at_london"], errors="raise")

        if use_finbert:
            pending_idx = []
            if "_headline_temp" in df_items.columns:
                pending_idx = df_items.index[df_items["_headline_temp"].notna()].tolist()

            if pending_idx:
                new_cache_rows = []
                for start_i in range(0, len(pending_idx), finbert_batch_size):
                    batch_idx = pending_idx[start_i:start_i + finbert_batch_size]
                    batch_texts = df_items.loc[batch_idx, "_headline_temp"].astype(str).tolist()
                    batch_scores = _score_finbert_batch(batch_texts, tokenizer, finbert_model, finbert_device)

                    for idx, scored in zip(batch_idx, batch_scores):
                        for key, value in scored.items():
                            df_items.at[idx, key] = value
                        new_cache_rows.append(
                            {
                                "item_hash": df_items.at[idx, "item_hash"],
                                "finbert_model_id": finbert_model_id,
                                "guardian_id": df_items.at[idx, "guardian_id"],
                                "published_at_utc": df_items.at[idx, "published_at_utc"],
                                "finbert_label": df_items.at[idx, "finbert_label"],
                                "finbert_pos": df_items.at[idx, "finbert_pos"],
                                "finbert_neg": df_items.at[idx, "finbert_neg"],
                                "finbert_neu": df_items.at[idx, "finbert_neu"],
                                "finbert_score": df_items.at[idx, "finbert_score"],
                            }
                        )

                if new_cache_rows:
                    finbert_cache = pd.concat([finbert_cache, pd.DataFrame(new_cache_rows)], ignore_index=True)
                    finbert_cache = finbert_cache.drop_duplicates(
                        subset=["item_hash", "finbert_model_id"],
                        keep="last",
                    )
                    finbert_cache.to_parquet(finbert_cache_path, index=False)

            if "_headline_temp" in df_items.columns:
                df_items = df_items.drop(columns=["_headline_temp"])

    if df_items.empty:
        print("[guardian] No items found. Writing empty outputs.")
    else:
        # Safety validation: cutoff assignment must be consistent
        validate_cutoff_assignments(df_items, cutoff_hhmm=cutoff)

    # Write long-lived outputs (NO headline text)
    items_path = out_dir / "news_items.parquet"
    daily_path = out_dir / "news_daily.parquet"
    manifest_path = out_dir / "manifest_news.csv"

    df_items.to_parquet(items_path, index=False)

    if df_items.empty:
        daily_cols = ["ticker", "assigned_date", "n_articles", "vader_mean", "vader_sum", "pos", "neg", "neu", "vader_std"]
        if use_finbert:
            daily_cols.extend(["finbert_pos_mean", "finbert_neg_mean", "finbert_neu_mean", "finbert_score_mean"])
        df_daily = pd.DataFrame(columns=daily_cols)
    else:
        df_items["assigned_date"] = pd.to_datetime(df_items["assigned_date"]).dt.date.astype(str)

        agg_map = {
            "n_articles": ("guardian_id", "count"),
            "vader_mean": ("sent_vader", "mean"),
            "vader_sum": ("sent_vader", "sum"),
            "vader_std": ("sent_vader", "std"),
            "pos": ("vader_label", lambda x: int((x == "pos").sum())),
            "neg": ("vader_label", lambda x: int((x == "neg").sum())),
            "neu": ("vader_label", lambda x: int((x == "neu").sum())),
        }
        if use_finbert:
            agg_map.update(
                {
                    "finbert_pos_mean": ("finbert_pos", "mean"),
                    "finbert_neg_mean": ("finbert_neg", "mean"),
                    "finbert_neu_mean": ("finbert_neu", "mean"),
                    "finbert_score_mean": ("finbert_score", "mean"),
                }
            )

        daily = df_items.groupby(["ticker", "assigned_date"], as_index=False).agg(**agg_map)
        df_daily = daily

    df_daily.to_parquet(daily_path, index=False)

    # Compatibility output for existing alignment/leakage test
    clean_path = out_dir / "news_clean.parquet"

    if df_items.empty:
        df_clean = pd.DataFrame(columns=["ticker", "published_at_london", "local_date", "effective_date", "sha1"])
    else:
        df_clean = df_items[["ticker", "published_at_london", "assigned_date", "item_hash"]].copy()
        # ensure tz-aware datetime dtype survives parquet roundtrip
        df_clean["published_at_london"] = pd.to_datetime(df_clean["published_at_london"], errors="raise")
        df_clean["local_date"] = df_clean["published_at_london"].dt.date
        df_clean["effective_date"] = pd.to_datetime(df_clean["assigned_date"]).dt.date
        df_clean.rename(columns={"item_hash": "sha1"}, inplace=True)
        df_clean = df_clean[["ticker", "published_at_london", "local_date", "effective_date", "sha1"]]
        df_clean = df_clean.drop_duplicates(subset=["sha1"]).reset_index(drop=True)

    df_clean.to_parquet(clean_path, index=False)
    print(f"[OK] wrote {clean_path}")


    # Manifest CSV (IDs/URLs/timestamps only)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["guardian_id", "web_url", "published_at_london", "assigned_date", "ticker", "item_hash", "query_expr"],
        )
        w.writeheader()
        for _, r in df_items.iterrows():
            w.writerow(
                {
                    "guardian_id": r.get("guardian_id"),
                    "web_url": r.get("web_url"),
                    "published_at_london": (
                        r.get("published_at_london").isoformat() if hasattr(r.get("published_at_london"), "isoformat") else r.get("published_at_london")
                    ),
                    "assigned_date": (
                        r.get("assigned_date").isoformat() if hasattr(r.get("assigned_date"), "isoformat") else r.get("assigned_date")
                    ),
                    "ticker": r.get("ticker"),
                    "item_hash": r.get("item_hash"),
                    "query_expr": r.get("query_expr"),
                }
            )

    # Store run metadata (safe, no content)
    meta = {
        "run_id": run_id,
        "provider": "guardian",
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "timezone": tz,
        "cutoff_local_time": cutoff,
        "config_path": str(cfg_path),
        "config_sha1": _sha1(cfg_path.read_text(encoding="utf-8")),
        "counts": {
            "items": int(len(df_items)),
            "daily_rows": int(len(df_daily)),
        },
    }
    meta["sentiment"] = {
        "use_vader": use_vader,
        "use_finbert": use_finbert,
        "finbert_model_id": finbert_model_id if use_finbert else None,
        "transformers_version": _safe_pkg_version("transformers") if use_finbert else None,
        "torch_version": _safe_pkg_version("torch") if use_finbert else None,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] wrote {items_path}")
    print(f"[OK] wrote {daily_path}")
    print(f"[OK] wrote {manifest_path}")

    # Ephemeral raw purge
    if args.write_raw and args.purge_raw and raw_out_dir is not None:
        shutil.rmtree(raw_out_dir, ignore_errors=True)
        print(f"[OK] purged raw dir: {raw_out_dir}")


if __name__ == "__main__":
    main()
