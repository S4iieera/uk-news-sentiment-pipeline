# Methodology

This project tests whether daily news sentiment features add useful signal over a price-only baseline for short-horizon UK equity movement modelling.

It is framed as a reproducible pipeline and evaluation study, not as a trading system.

## Data Sources

Price data comes from `yfinance` as daily OHLCV rows.

News metadata comes from The Guardian Open Platform. NewsAPI was tested early in the project but replaced because the available access level was not suitable for repeated data pulls.

## Time Window And Universe

Final modelling window:

```text
2024-11-01 to 2025-10-31
```

The project first used an exploratory 8-ticker run to inspect Guardian coverage and retrieval quality. The final modelling subset is restricted to:

```text
HSBA.L
BP.L
AZN.L
```

## Timestamp Alignment

Guardian publication timestamps are converted to `Europe/London`.

Articles published strictly after `16:30` London time are assigned to the next effective date. Articles at or before `16:30` remain on the same date.

This cutoff is used to reduce look-ahead risk when aligning news to daily market features.

## Prediction Target

Features from day `t` are used to predict next-day movement at `t+1`.

The classification target is whether the next-day return is positive. The regression target is the next-day return.

## Sentiment Features

VADER is used as a transparent lexicon baseline.

FinBERT is used as a finance-domain transformer comparison. The final comparison includes:

- `price_only`
- `price_plus_vader`
- `price_plus_finbert`

## Coverage Policies

Policy A keeps no-news days and represents them explicitly with `has_news` and `n_articles`.

Policy B drops rows without news.

The two policies make the coverage tradeoff visible: keeping all trading days preserves the price calendar, while dropping no-news days focuses only on dates where Guardian retrieval found relevant metadata.

## Evaluation

The project uses temporal holdout splits rather than random shuffling.

Classification metrics:

- ROC-AUC
- balanced accuracy
- F1 score

Regression metrics:

- R2
- MAE
- RMSE

The regression results are weak, and the classification results show only small differences between variants. The results are useful for understanding the pipeline and evaluation setup, but they should not be treated as evidence of a reliable market predictor.
