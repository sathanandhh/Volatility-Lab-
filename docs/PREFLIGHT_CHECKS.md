# Pre-flight Checks

> Every gate explained. See also: [TEST_CATALOG.md](TEST_CATALOG.md) for the full statistical test reference.

## Why pre-flight?

GARCH models are **not** universally applicable. If the data has no
ARCH effect, fitting GARCH is statistical malpractice. If the data has
structural breaks, a single-regime model will produce misleading
forecasts. The pre-flight layer catches these issues **before** any
model is fit, and blocks the fit if necessary.

## The 12 checks

### 1. Sample size

| | |
|---|---|
| **Test** | Count of observations |
| **Pass** | n ≥ 500 |
| **Warn** | 300 ≤ n < 500 |
| **Block** | n < 300 |
| **Why** | MLE needs enough data for stable parameter estimation. EGARCH is especially data-hungry. |

### 2. Missing data

| | |
|---|---|
| **Test** | Consecutive-day gaps in the index |
| **Pass** | No gap > 10 days |
| **Warn** | Gap > 10 days or > 5% missing |
| **Block** | (never blocks — only warns) |
| **Why** | Large gaps suggest data-feed issues, holidays, or delistings. |

### 3. Outliers

| | |
|---|---|
| **Test** | Hampel filter (rolling median + MAD, window=21) |
| **Pass** | ≤ 20 outliers > 5 MAD |
| **Warn** | > 20 outliers or > 2% of sample |
| **Block** | (never blocks) |
| **Why** | Outliers disproportionately influence MLE. Consider winsorizing or Student-t. |

### 4. Zero / infinity

| | |
|---|---|
| **Test** | Count of `inf`, `nan`, and `0` values |
| **Pass** | No inf, < 10% zeros |
| **Warn** | > 10% zeros (illiquid series) |
| **Block** | Any inf (WTI April-2020 case) |
| **Why** | Log returns are undefined for non-positive prices. |

### 5. Stationarity

| | |
|---|---|
| **Test** | ADF (H0: unit root) + KPSS (H0: stationary) |
| **Pass** | ADF p < 0.10 or KPSS p > 0.05 |
| **Warn** | Both tests suggest non-stationarity |
| **Block** | (never blocks — only warns) |
| **Why** | Returns should be stationary. If not, difference the series. |

### 6. Structural break

| | |
|---|---|
| **Test** | CUSUM on squared returns (Brownian-bridge asymptotics) |
| **Pass** | Max CUSUM ≤ 1.36 (5% critical value) |
| **Warn** | Max CUSUM > 1.36 |
| **Block** | (never blocks) |
| **Why** | A regime change (e.g. COVID crash) invalidates single-regime models. Split the sample or use a dummy. |

### 7. ARCH effect (Engle LM) — THE MOST CRITICAL CHECK

| | |
|---|---|
| **Test** | Engle ARCH-LM (lag=10) |
| **Pass** | p < 0.10 (ARCH effect present) |
| **Warn** | (never warns) |
| **Block** | p ≥ 0.10 (no ARCH effect) |
| **Why** | If there is no ARCH effect, the entire GARCH family is statistically unjustified. Use EWMA or constant volatility instead. |

### 8. Volatility clustering

| | |
|---|---|
| **Test** | Ljung-Box on r² (lag=10) |
| **Pass** | p < 0.10 (clustering present) |
| **Warn** | (never warns) |
| **Block** | p ≥ 0.10 (no clustering) |
| **Why** | GARCH models volatility clustering. If there is none, GARCH adds no value. |

### 9. Normality

| | |
|---|---|
| **Test** | Jarque-Bera + Shapiro-Wilk |
| **Pass** | JB p ≥ 0.05 |
| **Warn** | 0.001 < JB p < 0.05 |
| **Block** | JB p < 0.001 |
| **Why** | If normality is strongly rejected, Normal innovations are inappropriate. Use Student-t. |

### 10. Mean specification

| | |
|---|---|
| **Test** | Ljung-Box on constant-mean residuals (lag=10) |
| **Pass** | p ≥ 0.05 (constant mean adequate) |
| **Warn** | p < 0.05 (serial correlation in mean) |
| **Block** | (never blocks) |
| **Why** | If the mean has serial correlation, add AR/ARMA terms. |

### 11. Leverage asymmetry

| | |
|---|---|
| **Test** | Sign-bias test (regress r² on indicator(r<0)) |
| **Pass** | p < 0.05 (leverage detected) |
| **Warn** | (never warns) |
| **Block** | (never blocks) |
| **Why** | If leverage is detected, symmetric GARCH misses it. Try GJR-GARCH or EGARCH. |

### 12. Frequency adequacy

| | |
|---|---|
| **Test** | Infer frequency from median gap; check min obs |
| **Pass** | Daily ≥ 250, Weekly ≥ 60, Monthly ≥ 36 |
| **Warn** | Below minimum for the inferred frequency |
| **Block** | (never blocks) |
| **Why** | Weekly/monthly data needs more history for stable MLE. |

## Gate aggregation

```
If any check = BLOCK → overall = BLOCK (models.fit refuses)
Else if any check = WARN → overall = WARN (models.fit proceeds with caveats)
Else → overall = PASS
```

## What happens when blocked?

- `models.fit` returns a `blocked` status with the list of blocking checks.
- The agent should call `preflight.explain_gate(check_name)` to understand
  each block and propose a fix.
- After fixing (e.g. switching distribution, extending history), re-run
  `preflight.run` and try again.
- `force=True` bypasses blocks but logs the decision for audit.
