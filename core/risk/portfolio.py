"""Portfolio VaR decomposition (marginal, component, incremental)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def portfolio_var_decomposition(
    returns: pd.DataFrame, weights: dict[str, float],
    confidence: float = 0.975, portfolio_value: float = 1_000_000.0,
    method: str = "normal",
) -> dict:
    """Decompose portfolio VaR into marginal, component, incremental."""
    cols = [c for c in weights if c in returns.columns]
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    R = returns[cols].dropna()
    cov = R.cov().values
    alpha = 1 - confidence
    z = stats.norm.ppf(confidence)
    portfolio_var = float(np.sqrt(w @ cov @ w) * z)
    # Marginal VaR: ∂VaR/∂w_i = (cov @ w)_i / σ_p * z
    marginal = (cov @ w) / np.sqrt(w @ cov @ w) * z
    component = marginal * w
    # Incremental: VaR without each asset
    incremental = []
    for i, c in enumerate(cols):
        mask = np.ones(len(cols), dtype=bool)
        mask[i] = False
        if mask.sum() == 0:
            incr = 0.0
        else:
            w_red = np.delete(w, i)
            w_red = w_red / w_red.sum() if w_red.sum() > 0 else w_red
            cov_red = np.delete(np.delete(cov, i, 0), i, 1)
            var_red = float(np.sqrt(w_red @ cov_red @ w_red) * z)
            incr = portfolio_var - var_red
        incremental.append({"asset": c, "incremental_var": incr})
    return {
        "method": method,
        "confidence": confidence,
        "portfolio_var_return_pct": portfolio_var * 100,
        "portfolio_var_currency": portfolio_value * portfolio_var,
        "marginal_var": {c: float(m) for c, m in zip(cols, marginal)},
        "component_var": {c: float(cm) for c, cm in zip(cols, component)},
        "incremental_var": incremental,
        "weights": {c: float(wi) for c, wi in zip(cols, w)},
    }
