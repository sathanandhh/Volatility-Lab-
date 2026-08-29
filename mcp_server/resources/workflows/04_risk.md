# Workflow 4 · Risk Quantification

**Goal:** translate fitted volatility into VaR, ES, and Basel metrics.

## Prerequisites
- Workflow 2 complete (≥1 model fit + diagnostics clean)

## Steps

1. `models.forecast` — produce horizon variance path
2. `risk.compute_var` with method:
   - "normal" — parametric Normal
   - "student_t" — parametric Student-t (default; recommended if preflight normality = block)
   - "cornish_fisher" — CF expansion (skew + kurt adjusted)
   - "historical" — empirical quantile of past returns
   - "fhs" — Filtered Historical Simulation (recommended for tails)
   - "monte_carlo" — simulate from the fitted model
3. `risk.compute_es` with same method
   - Returns ES/VaR ratio — > 1.3 indicates heavy tail
4. `risk.basel_es` — simplified Basel 97.5% 10-day ES
   - Teaching benchmark; not regulatory capital
5. `scenario.apply_market_shock` — what-if stress
6. `scenario.stressed_var` — historical crisis scenario

## Method selection guide

| Situation | Recommended method |
|---|---|
| Preflight normality = pass | normal or student_t |
| Preflight normality = block | student_t (mandatory) |
| Strong skew detected | cornish_fisher or skewt distribution |
| Limited data, want tail realism | fhs |
| Need path-dependent scenarios | monte_carlo |
| Regulatory reference | basel_es |

## Exit criterion

VaR + ES computed for at least one model. Compare methods via ratios
and explain discrepancies to the user.
