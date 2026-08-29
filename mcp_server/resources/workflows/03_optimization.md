# Workflow 3 · Input Optimization

**Goal:** remove guesswork from model specification.

## Prerequisites
- Workflow 1 complete (preflight overall = pass or warn)
- Workflow 2 optionally started (model family chosen)

## Steps

1. `optimize.optimize_order` — sweep (p, q) by AIC
   - Default: p ∈ [1..3], q ∈ [0..2]
   - Returns full sweep table; agent explains why (1,1) beats (2,1)
2. `optimize.optimize_distribution` — sweep distributions by AIC
   - Default candidates: normal, t, ged, skewt, jsu
   - If preflight normality = block → normal will lose; result confirms
3. `optimize.optimize_mean` — sweep mean specs by residual LB p
   - Default candidates: Constant, AR(1), ARMA(1,1)
4. `optimize.optimize_window` — sweep rolling window by QLIKE
   - Only needed if planning backtest.rolling
5. `optimize.optimize_refit` — sweep refit frequency by CC p-value
   - Default: 1..30
6. `optimize.optimize_horizon` — pick horizon by use case
   - regulatory → 10 days, weekly → 5, monthly → 21

## Heuristics

- AIC differences < 2 → prefer simpler model
- AIC differences 4-7 → less support for more complex
- AIC differences > 10 → complex model strongly supported
- BIC penalizes complexity more heavily → use for true-model selection
- QLIKE preferred over RMSE for vol forecast evaluation (Patton 2011)

## Exit criterion

All optimization results stored on session. `models.fit` will use them
as defaults. Proceed to Workflow 2 (modeling) or Workflow 5 (validation).
