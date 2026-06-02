from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.common.config import load_cfg


DESCRIPTION_HINTS: Dict[str, str] = {
    "ticker": "Instrument ticker (e.g., HSBA.L).",
    "date": "Trading day / feature day (local date).",
    "published_at_london": "Publish timestamp converted to Europe/London.",
    "effective_date": "Cutoff-adjusted assigned date (after 16:30 -> t+1).",
    "local_date": "Date portion of published_at_london.",
    "sha1": "Deterministic item hash for de-duplication/provenance.",
    "n_articles": "Number of news items aggregated into the day.",
    "sent_vader_mean": "Mean VADER compound sentiment aggregated daily.",
    "sent_vader_sum": "Sum VADER compound sentiment aggregated daily.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = Path(cfg["project"]["data_dir"])
    processed = data_dir / "processed" / run_id

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        "prices_daily.parquet",
        "news_items.parquet",
        "news_daily.parquet",
        "news_clean.parquet",
        "features_daily.parquet",
        "model_dataset.parquet",
    ]

    rows: List[dict] = []
    for fname in candidates:
        fpath = processed / fname
        if not fpath.exists():
            rows.append(
                {
                    "file": fname,
                    "column": "(missing)",
                    "dtype": "",
                    "description_stub": "File not present at time of dictionary generation.",
                }
            )
            continue

        df = pd.read_parquet(fpath)
        for col in df.columns:
            dtype = str(df[col].dtype)
            rows.append(
                {
                    "file": fname,
                    "column": col,
                    "dtype": dtype,
                    "description_stub": DESCRIPTION_HINTS.get(col, ""),
                }
            )

    out = pd.DataFrame(rows)
    out_path = reports_dir / "data_dictionary.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()