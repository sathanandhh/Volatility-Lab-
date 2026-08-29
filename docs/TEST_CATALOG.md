# Test Catalog

> Every statistical test used in pre-flight and diagnostics.

## Pre-flight tests

| # | Test | Null | Block threshold | Why |
|---|---|---|---|---|
| 1 | Sample size | — | n < 300 | Insufficient for MLE |
| 2 | Missing data | — | Gap > 30 days | Data-feed issues |
| 3 | Outliers (Hampel) | — | > 20 obs > 5σ | Influences MLE |
| 4 | Zero/infinity | — | Any inf | Log returns undefined |
| 5 | ADF | Unit root | p > 0.10 | Returns non-stationary |
| 6 | KPSS | Stationary | p < 0.05 | Non-stationary |
| 7 | CUSUM (r²) | No break | > 1.36 | Structural break |
| 8 | Engle ARCH-LM | No ARCH | p > 0.10 | **GARCH unjustified** |
| 9 | Ljung-Box (r²) | No clustering | p > 0.10 | No clustering |
| 10 | Jarque-Bera | Normal | p < 0.001 | Use Student-t |
| 11 | Shapiro-Wilk | Normal | p < 0.05 | Cross-check JB |
| 12 | Sign-bias | No leverage | p < 0.05 | Try GJR/EGARCH |

## Post-fit diagnostics

| Test | Null | Pass | Failure → |
|---|---|---|---|
| Ljung-Box z | No serial corr in std resid | p ≥ 0.05 | Add AR lags |
| Ljung-Box z² | No remaining ARCH | p ≥ 0.05 | Increase q |
| ARCH-LM on z | No remaining ARCH | p ≥ 0.05 | Misspecified |
| Jarque-Bera on z | Normality of std resid | p ≥ 0.05 | Switch dist |
| Sign-bias on z | No leverage in resid | p ≥ 0.05 | Try GJR/EGARCH |
| Nyblom stability | Parameter stability | Stable | Regime split |

## Backtest tests

| Test | Null | Pass | Failure → |
|---|---|---|---|
| Kupiec POF | Breach rate = expected | p ≥ 0.05 | Re-tune dist/window |
| Christoffersen | Breaches independent | p ≥ 0.05 | Shorten refit |
| Conditional coverage | POF + independence | p ≥ 0.05 | Combined |
| Basel Traffic Light | ≤ 4 breaches / 250 | Green | Yellow/Red → multiplier |
| Dynamic Quantile | Coverage + independence | p ≥ 0.05 | Most powerful single test |
| Diebold-Mariano | Equal accuracy | p ≥ 0.05 | Prefer lower-loss model |

## Information criteria

| Criterion | Formula | Penalty | When |
|---|---|---|---|
| AIC | -2LL + 2k | Light | Prediction (default) |
| BIC | -2LL + k log n | Heavy | True model |
| HQIC | -2LL + 2k log log n | Medium | Compromise |

## Accuracy metrics

| Metric | Formula | Lower better |
|---|---|---|
| QLIKE | mean[log σ² + r²/σ²] | ✓ (preferred) |
| Vol RMSE | sqrt(mean[(r²-σ²)²]) | ✓ |
| MAE | mean[\|r²-σ²\|] | ✓ |
| MAFE | mean[\|r-σ\|/σ] | ✓ |
