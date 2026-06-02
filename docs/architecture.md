# Architecture

The project is organised as a set of command-line pipeline stages plus a read-only Streamlit dashboard.

```text
config/*.yaml
    |
    v
src/ingest/prices_yfinance.py      -> prices_daily.parquet
src/ingest/news_guardian.py        -> news_items.parquet, news_daily.parquet
    |
    v
src/features/build_features_daily.py
    |
    v
features_daily.parquet
    |
    v
src/align/build_model_dataset.py
    |
    v
model_dataset.parquet
    |
    v
src/eval/pilot_metrics.py          -> reports/*.csv, reports/*.json
src/reports/*                      -> coverage and data dictionary outputs
src/inspection/*                   -> inspection rows for dashboard
    |
    v
app/app.py                         -> read-only dashboard
```

## Source Layout

```text
src/common/      Config, paths, hashing helpers
src/ingest/      Price and news provider ingestion
src/features/    Daily feature construction
src/align/       Target alignment for day t -> t+1 modelling
src/eval/        Temporal holdout model comparison
src/reports/     Coverage and metadata reporting
src/inspection/  Human-inspection rows for dashboard display
app/             Streamlit dashboard
tests/           Cutoff and alignment tests
```

## Dashboard Boundary

The Streamlit app loads saved artefacts from `data/processed/` and `reports/`.

It does not:

- call Guardian or NewsAPI
- download price data
- score VADER or FinBERT
- train models
- rerun evaluation

This keeps the dashboard as an inspection layer rather than a hidden pipeline runner.

## Reproducibility Boundary

The public repository keeps source code, configs, tests, docs, and small report outputs.

Generated data is intentionally excluded:

```text
data/raw/
data/processed/
```

This avoids publishing API-derived artefacts while still making the pipeline auditable.
