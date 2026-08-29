# Volatility Analytics — Master Methodology

> Audience: LLM agents and human practitioners using the Volatility MCP server.

## The eight-stage workflow

The Volatility MCP drives every analysis through eight connected stages.
Maximum Likelihood Estimation is the bridge between the historical
return series and each future conditional-variance path.

### Stage 1 · Obtain adjusted closing prices
Select market, asset, history and return frequency. Remove missing
observations. Use auto-adjusted prices where available.

### Stage 2 · Calculate log returns
Transform prices into returns: r_t = 100 × ln(P_t / P_{t-1}).
Log returns are time-additive and symmetric, which simplifies aggregation.

### Stage 3 · Run pre-flight gates (BEFORE any model)
Twelve statistical checks determine whether GARCH is appropriate at all.
The most important is the Engle ARCH-LM test: if it fails to reject the
null of no ARCH effect, the entire GARCH family is statistically
unjustified and the server blocks further model fitting.

### Stage 4 · Optimize inputs
Rather than guessing p, q, distribution, and window size, sweep each
input space and pick the optimum by information or out-of-sample
criteria:
  - (p, q): minimum AIC
  - distribution: minimum AIC
  - mean specification: Ljung-Box p > 0.05 on residuals
  - rolling window: minimum QLIKE
  - refit frequency: maximum conditional-coverage p-value

### Stage 5 · Specify competing variance equations
Fit ARCH, symmetric GARCH, threshold GJR-GARCH, log-variance EGARCH
(and optionally FIGARCH, IGARCH, EWMA, multivariate DCC, ML variants).

### Stage 6 · Estimate parameters using MLE
Estimate μ and variance parameters; Student-t models also estimate
degrees of freedom ν. Check convergence flag; non-converged fits are
flagged and excluded from comparison.

### Stage 7 · Run residual diagnostics
Test standardized residuals for:
  - Remaining ARCH effect (ARCH-LM)
  - Remaining serial correlation (Ljung-Box on z and z²)
  - Normality (Jarque-Bera, Shapiro-Wilk, Anderson-Darling)
  - Sign bias (leverage)
  - Parameter stability (Nyblom)

### Stage 8 · Backtest and translate into risk
Walk-forward rolling refit; produce genuine one-step-ahead VaR
forecasts; test coverage (Kupiec POF), independence (Christoffersen),
conditional coverage, Basel traffic light, and Engle-Manganelli DQ.
Finally, express VaR/ES in currency for the illustrative portfolio.

## Core equations

  GARCH(1,1):        σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
  GJR-GARCH:         σ²_t = ω + (α + γ I_{t-1}) ε²_{t-1} + β σ²_{t-1}
  EGARCH:            log(σ²_t) = ω + β log(σ²_{t-1})
                            + α (|z_{t-1}| - E|z|) + γ z_{t-1}
  EWMA:              σ²_t = λ σ²_{t-1} + (1-λ) ε²_{t-1}

EGARCH guarantees positive variance through exponentiation. Multi-step
EGARCH forecasts require simulation because an exact analytic forecast
is unavailable beyond one period.

## VaR and Expected Shortfall

  Normal VaR:        VaR = μ + σ Φ^{-1}(α)
  Student-t VaR:     VaR = μ + σ t_ν^{-1}(α) × √((ν-2)/ν)
  Expected Shortfall: ES = E[L | L > VaR]

Basel MAR33 convention: 97.5% one-tailed ES, 10-day base liquidity
horizon. Simplified here — does not implement stressed calibration,
liquidity buckets, or modellability tests.

## When the feedback loop iterates

The server recommends returning to `optimize.*` or `models.fit` when:
  - Ljung-Box on squared residuals has p < 0.05 (remaining ARCH)
  - ARCH-LM on residuals has p < 0.05
  - Sign-bias test p < 0.05 (try GJR/EGARCH)
  - Jarque-Bera p < 0.01 (try Student-t or skew-t)
  - Kupiec p < 0.05 (under/over-prediction — re-tune window or refit)
  - Christoffersen independence p < 0.05 (breach clustering — shorten refit)
  - Nyblom stability fails (regime split or rolling refit)
