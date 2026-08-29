# Input Optimization

> How each model input is tuned — no guesswork, just information criteria.

## Philosophy

The user (or the LLM agent) should never guess `p`, `q`, distribution,
window size, or refit frequency. Each input has a dedicated optimizer
that sweeps a search space and picks the optimum by an objective
criterion. The full sweep table is returned so the choice can be
explained.

## What gets optimized

| Input | Optimizer | Objective | Search space | Default |
|---|---|---|---|---|
| (p, q) orders | `optimize.order` | AIC (or BIC) | p∈[1..3], q∈[0..2] | (1, 1) |
| Distribution | `optimize.distribution` | AIC | {normal, t, ged, skewt, jsu} | t |
| Mean spec | `optimize.mean` | Ljung-Box p on residuals | {Constant, AR(1), ARMA(1,1)} | Constant |
| Rolling window | `optimize.window` | Out-of-sample QLIKE | 10–63 days, step 5 | 21 |
| Refit frequency | `optimize.refit` | Conditional-coverage p | 1–30, step 5 | 10 |
| Forecast horizon | `optimize.horizon` | Use-case rule | — | 10 (regulatory) |

## AIC vs BIC

| Criterion | Penalty | When to prefer |
|---|---|---|
| AIC | 2k | Prediction (default) |
| BIC | k log(n) | True-model identification |
| HQIC | 2k log log(n) | Compromise |

**Rule of thumb (ΔAIC):**
- Δ < 2: both models have substantial support — prefer simpler
- Δ 4–7: less support for the more complex
- Δ > 10: essentially no support for the more complex

## Distribution selection

The pre-flight normality check provides the initial hint:

| Normality status | Recommended distribution |
|---|---|
| PASS | Normal (but Student-t rarely hurts) |
| WARN | Student-t |
| BLOCK | Student-t (mandatory) or skew-t |

The optimizer confirms this by fitting each candidate and comparing AIC.

## Window selection

QLIKE is the preferred loss function for volatility-forecast evaluation
(Patton 2011). It is robust to noise in the realized-variance proxy
(squared returns).

```
QLIKE = mean[ log(σ²) + r²/σ² ]
```

Lower QLIKE = better forecast. The optimizer sweeps window sizes and
picks the one minimizing QLIKE on an out-of-sample window.

## Refit frequency

The refit optimizer sweeps "refit every N observations" and picks the
N that maximizes the conditional-coverage p-value from the backtest.

- Lower N → adapts faster to regime shifts but costs more compute
- Higher N → faster but may lag regime changes

The optimizer finds the sweet spot automatically.

## Horizon selection

Horizon is not optimized — it's chosen by use case:

| Use case | Horizon |
|---|---|
| regulatory (Basel) | 10 days |
| weekly | 5 days |
| monthly | 21 days |
| stress | 60 days |
| auto | half-life of squared-return ACF |
