# Model Catalog

> When to use which model. See also: [DECISION_RULES.md](DECISION_RULES.md).

## Univariate

| Model | Family | Asymmetry | Persistence | Best for |
|---|---|---|---|---|
| ARCH(p) | ARCH | None | Short | Teaching; short memory |
| GARCH(1,1) | GARCH | None | β + α | Industry workhorse |
| GJR-GARCH | GARCH | Threshold (γ>0) | β + α + γ/2 | Leverage detected |
| EGARCH | EGARCH | Smooth (γ<0) | β | Leverage + positivity |
| FIGARCH | FIGARCH | None | Long memory | Long-memory vol |
| IGARCH | IGARCH | None | = 1 (non-stat.) | Use with extreme caution |
| EWMA | — | None | λ (imposed) | RiskMetrics; no MLE |
| HARCH | HARCH | None | Multi-scale | Long memory via multi-lag |
| Realized-GARCH | — | — | — | Intraday data available |
| HEAVY | — | — | — | High-frequency-based |

## Multivariate

| Model | Best for |
|---|---|
| DCC-GARCH | Time-varying cross-asset correlations |
| BEKK | Positive-definite covariance without constraints |
| GO-GARCH | Orthogonal factors drive covariance |
| Copula-GARCH | Flexible dependence structure |

## Stochastic volatility

| Model | Best for |
|---|---|
| Heston | Option pricing (affine SV) |
| Heston-Nandi | GARCH-like SV with closed-form option pricing |
| SV-Jumps | Jump-diffusion volatility |

## Machine learning

| Model | Best for |
|---|---|
| LSTM-Vol | Nonlinear patterns in squared returns |
| Transformer-Vol | Long-context attention |
| TFT | Multi-horizon with covariates |

## Innovation distributions

| Distribution | Skew | Kurtosis | Best for |
|---|---|---|---|
| Normal | 0 | 0 | Residuals pass JB (rare in finance) |
| Student-t | 0 | > 0 (ν) | Symmetric heavy tails |
| GED | 0 | Parameter | Tails heavier or lighter than t |
| Skew-t | ≠ 0 | > 0 | Skewed + heavy-tailed |
| JSU | Parameter | Parameter | Very heavy tails with skew |
| GHD | Parameter | Parameter | Maximum flexibility |
