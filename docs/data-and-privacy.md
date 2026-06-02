# Data And Privacy

This repository is designed to keep private credentials, raw downloads, and provider-derived data out of the public GitHub project.

## API Keys

The local `.env` file must not be committed.

Use `.env.example` as a template:

```text
GUARDIAN_API_KEY=your_guardian_key_here
NEWSAPI_KEY=your_newsapi_key_here
```

`NEWSAPI_KEY` is retained only for legacy experiments. The implemented news ingestion uses The Guardian Open Platform.

## Generated Data

The following folders are excluded from Git:

```text
data/raw/
data/processed/
```

These folders may contain:

- yfinance price snapshots
- Guardian-derived metadata
- run-specific parquet artefacts
- FinBERT score caches
- inspection outputs

They should remain local unless a deliberately small and sanitized sample dataset is created.

## Raw News Content

The Guardian ingestion code keeps headline text only long enough to score sentiment. Long-lived outputs are designed to store metadata, timestamps, URLs, hashes, sentiment scores, and aggregates rather than raw article text.

Raw Guardian JSON should not be published. If raw writes are used for debugging, they should be purged after the run.

## Local Files To Keep Private

Do not publish:

- `.env`
- local virtual environments
- `data/raw/`
- `data/processed/`
- temporary DOCX/report check folders
- university submission packages
- raw provider JSON
- API-key-bearing logs
