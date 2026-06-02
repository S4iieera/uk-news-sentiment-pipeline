# Results

The final results are deliberately presented with caution. This project does not show reliable market prediction. It shows a reproducible way to collect data, align timestamps, build sentiment features, and test whether those features add useful signal over a price-only baseline.

## Final Run

```text
model_v2_core3_1y_vader_finbert_guardian
```

Final modelling subset:

```text
HSBA.L
BP.L
AZN.L
```

Window:

```text
2024-11-01 to 2025-10-31
```

## Policy A

Policy A keeps no-news days.

| Variant | ROC-AUC | Balanced accuracy | F1 | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| price_only | 0.509 | 0.519 | 0.629 | -0.013 | 0.0096 | 0.0147 |
| price_plus_vader | 0.528 | 0.524 | 0.610 | -0.061 | 0.0098 | 0.0151 |
| price_plus_finbert | 0.512 | 0.489 | 0.553 | -0.040 | 0.0098 | 0.0149 |

Policy A gives a small ROC-AUC increase for VADER over the price-only baseline, but the difference is modest and does not translate into a clear overall advantage.

## Policy B

Policy B drops no-news rows.

| Variant | ROC-AUC | Balanced accuracy | F1 | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| price_only | 0.506 | 0.504 | 0.532 | -0.024 | 0.0113 | 0.0183 |
| price_plus_vader | 0.510 | 0.492 | 0.489 | -0.103 | 0.0118 | 0.0190 |
| price_plus_finbert | 0.536 | 0.519 | 0.538 | -0.071 | 0.0117 | 0.0187 |

Policy B gives a small ROC-AUC and balanced-accuracy lift for FinBERT, but the sample is smaller because no-news rows are removed.

## Interpretation

The results are mixed:

- classification metrics show small and inconsistent differences
- regression metrics are weak, with negative R2 values
- FinBERT does not clearly dominate VADER across both policies
- the coverage policy materially affects the evaluation set

The conclusion is that the pipeline is useful for controlled experimentation and auditability, but the saved results do not support strong claims about trading performance.
