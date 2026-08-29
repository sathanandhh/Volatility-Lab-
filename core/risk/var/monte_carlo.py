"""Monte Carlo VaR — simulate future paths from a fitted model."""
from __future__ import annotations

import math
import numpy as np


def var_mc(fit, horizon: int = 1, alpha: float = 0.025,
           n_sims: int = 10_000) -> float:
    """Monte Carlo VaR — simulate horizon return paths from a fitted model."""
    rng = np.random.default_rng(42)
    mu = float(fit.params.get("mu", 0.0))
    sigma = fit.next_vol if hasattr(fit, "next_vol") else float(fit.conditional_volatility.iloc[-1])
    nu = float(fit.params.get("nu", 8.0))
    # Simulate Student-t innovations (unit variance)
    innov = rng.standard_t(nu, size=(n_sims, horizon)) / math.sqrt((nu - 2) / nu)
    horizon_returns = mu * horizon + sigma * np.sqrt(horizon) * innov.mean(axis=1)
    # Better: sum the per-period simulated returns
    horizon_returns = mu * horizon + sigma * innov.sum(axis=1)
    return float(np.quantile(horizon_returns, alpha))
