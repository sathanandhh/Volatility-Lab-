# Volatility Model Catalog

## Univariate ARCH-family

| Model | Equation (sketch) | Asymmetry | Persistence | Notes |
|---|---|---|---|---|
| ARCH(p) | σ²_t = ω + Σ α_i ε²_{t-i} | None | Finite (short) | Engle 1982. Forgets quickly. |
| GARCH(p,q) | σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} | None | β + α | Bollerslev 1986. Industry workhorse. |
| GJR-GARCH | σ²_t = ω + (α + γ I) ε²_{t-1} + β σ²_{t-1} | Threshold (γ > 0 = leverage) | β + α + γ/2 | Glosten-Jagannathan-Runkle 1993. |
| EGARCH | log σ²_t = ω + β log σ²_{t-1} + α(|z|-E|z|) + γ z | Smooth (γ < 0 = leverage) | β | Nelson 1991. Variance always positive. |
| IGARCH | β + α = 1 (forced) | None | = 1 (non-stationary) | Use with extreme caution. |
| FIGARCH | Fractionally integrated GARCH | None | Long memory | Captures long-memory vol. |
| EWMA | σ²_t = λ σ²_{t-1} + (1-λ) ε²_{t-1} | None | λ (imposed) | RiskMetrics 1996. No MLE needed. |

## Multivariate

| Model | Use case |
|---|---|
| DCC-GARCH | Time-varying correlations between assets. Engle 2002. |
| BEKK | Positive-definite covariance without constraints. Engle-Kroner 1995. |
| GO-GARCH | Orthogonal factors drive covariance. van der Weide 2002. |
| Copula-GARCH | Marginals GARCH; dependence via copula. |

## Realized / High-frequency

| Model | Use case |
|---|---|
| Realized GARCH | Uses intraday realized variance as input. Hansen-Huang-Shek 2012. |
| HEAVY | HAR-type with HEAVY regressors. Shephard-Andersen 2009. |
| HAR-RV | Heterogeneous AR of realized variance. Corsi 2009. |

## Stochastic volatility

| Model | Use case |
|---|---|
| Heston | Affine SV; closed-form option pricing. |
| Heston-Nandi | GARCH-like SV with closed-form option pricing. |
| SV with jumps | Adds Poisson jump component. |

## Machine learning

| Model | Use case |
|---|---|
| LSTM vol | Sequence model on squared returns. |
| Transformer vol | Attention-based; long-context vol. |
| TFT (Temporal Fusion Transformer) | Multi-horizon forecasting with covariates. |

## Distribution catalog (see distributions/catalog for details)

normal, t (Student), ged (Generalized Error), skewt (Skew-Student),
jsu (Johnson SU), ghd (Generalized Hyperbolic).

## Selection heuristics

- If pre-flight ARCH-LM fails → use EWMA or static volatility (NOT GARCH).
- If leverage detected (sign-bias p < 0.05) → prefer GJR-GARCH or EGARCH.
- If long memory detected → try FIGARCH.
- If multivariate (≥2 assets) → DCC-GARCH.
- If intraday data available → Realized GARCH or HEAVY.
- If pricing options → Heston or Heston-Nandi.
