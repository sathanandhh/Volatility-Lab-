# Statistical Test Catalog

## Pre-flight tests (run BEFORE model fitting)

| Test | Null hypothesis | Block threshold | Why it matters |
|---|---|---|---|
| Sample size | n ≥ 300 (GARCH) / 500 (EGARCH) | n < 300 | Insufficient for stable MLE. |
| Missing data | No gaps > 5 obs | Gap > 10 | Resampling artifacts; consider imputation. |
| Outliers | No obs > 5σ (Hampel) | > 10 obs > 5σ | Influences MLE disproportionately. |
| Zero/Infinity | No bad values | Any present | WTI April-2020 case; log returns undefined. |
| ADF | Unit root | p > 0.05 | Returns should be stationary. |
| KPSS | Stationarity | p < 0.05 | Cross-check with ADF. |
| Bai-Perron | No structural breaks | Break with > 30% sample | Regime split needed. |
| Engle ARCH-LM | No ARCH effect | p > 0.05 | **If pass, GARCH is unjustified.** |
| Ljung-Box r² | No serial dependence in r² | p > 0.05 | Volatility clustering absent. |
| Jarque-Bera | Normality | p > 0.05 (warn) | Use Student-t if rejected. |
| Shapiro-Wilk | Normality | p > 0.05 (warn) | Cross-check JB. |
| Sign-bias | No leverage | p > 0.05 | Try GJR/EGARCH if rejected. |

## Post-fit diagnostics (run AFTER model fitting)

| Test | Null | Pass threshold | Failure → |
|---|---|---|---|
| Ljung-Box z (lag 10, 20) | No serial corr in std resid | p ≥ 0.05 | Add AR lags to mean. |
| Ljung-Box z² (lag 10, 20) | No remaining ARCH | p ≥ 0.05 | Increase q; try FIGARCH. |
| ARCH-LM on z | No remaining ARCH | p ≥ 0.05 | Model misspecified. |
| Jarque-Bera on z | Normality of std resid | p ≥ 0.05 | Switch distribution. |
| Shapiro-Wilk on z | Normality | p ≥ 0.05 | Cross-check. |
| Anderson-Darling on z | Normality (tail-sensitive) | p ≥ 0.05 | Tail-sensitive check. |
| Sign-bias on z | No leverage in resid | p ≥ 0.05 | Try GJR/EGARCH. |
| Nyblom stability | Parameter stability | All stats < critical | Regime split. |

## Backtest tests (run AFTER VaR forecasting)

| Test | Null | Pass threshold | Failure → |
|---|---|---|---|
| Kupiec POF | Breach rate = expected | p ≥ 0.05 | Re-tune distribution or window. |
| Christoffersen independence | Breaches not clustered | p ≥ 0.05 | Shorten refit interval. |
| Conditional coverage | Both POF + independence | p ≥ 0.05 | Combined verdict. |
| Basel Traffic Light | ≤ 4 breaches (250 obs) | Green zone | Yellow/Red → capital multiplier. |
| Engle-Manganelli DQ | Coverage + independence (regression) | p ≥ 0.05 | Most powerful single test. |
| Diebold-Mariano | Equal forecast accuracy | p ≥ 0.05 | If rejected, prefer lower-loss model. |

## Information criteria

| Criterion | Formula | When to prefer |
|---|---|---|
| AIC | -2 log L + 2k | Default; better for prediction. |
| BIC | -2 log L + k log n | Heavier penalty; better for true model. |
| HQIC | -2 log L + 2k log log n | Compromise between AIC and BIC. |

## Accuracy metrics (out-of-sample)

| Metric | Formula | Lower is better? |
|---|---|---|
| QLIKE | mean[log(σ²) + r²/σ²] | Yes — preferred for vol. |
| Volatility RMSE | sqrt(mean[(r² - σ²)²]) | Yes. |
| MAE | mean[|r² - σ²|] | Yes. |
| MAFE | mean[|r - σ|/σ] | Yes. |
