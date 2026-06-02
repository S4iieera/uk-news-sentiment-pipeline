# Coverage report (model_v2_core3_1y_vader_finbert_guardian)

- prices_daily: `data/processed/model_v2_core3_1y_vader_finbert_guardian/prices_daily.parquet`
- news_daily: `data/processed/model_v2_core3_1y_vader_finbert_guardian/news_daily.parquet`

| ticker   |   n_price_days |   n_news_days |   coverage_pct |   missing_days_count |
|:---------|---------------:|--------------:|---------------:|---------------------:|
| AZN.L    |            253 |            72 |          28.46 |                  181 |
| HSBA.L   |            253 |           108 |          42.69 |                  145 |
| BP.L     |            253 |           109 |          43.08 |                  144 |

Lowest coverage: **AZN.L** = 28.46% (missing 181 days).
Missing sample: 2024-11-01;2024-11-04;2024-11-05;2024-11-07;2024-11-08;2024-11-14;2024-11-18;2024-11-21;2024-11-22;2024-11-25;2024-11-26;2024-11-27;2024-11-29;2024-12-02;2024-12-03