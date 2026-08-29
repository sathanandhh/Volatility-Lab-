# Workflow 2 · Modeling

**Goal:** fit one or more ARCH-family models to the loaded returns.

## Prerequisites
- Workflow 1 complete (preflight overall = pass or warn)

## Steps

1. `optimize.optimize_order` → sweep (p, q)
2. `optimize.optimize_distribution` → pick distribution
3. `optimize.optimize_mean` → pick mean specification
4. `models.fit` with optimized (or explicit) inputs
5. `diagnostics.run` → check residual whiteness
6. If any diagnostic fails → return to step 4 with different spec
7. Optionally fit a second/third model for comparison
8. `compare.compare_models` → scorecard with ranks

## What to look for

- Convergence flag (must be true)
- AIC/BIC ranking (lower = better in-sample fit)
- QLIKE ranking (lower = better out-of-sample)
- Ljung-Box on z² p ≥ 0.05 (no remaining ARCH)
- ARCH-LM on z p ≥ 0.05
- Sign-bias on z p ≥ 0.05 (no leverage in residuals)

## Exit criterion

At least one model with:
- Converged = true
- LB(z²) p ≥ 0.05
- ARCH-LM p ≥ 0.05
- AIC ranked among top 2
