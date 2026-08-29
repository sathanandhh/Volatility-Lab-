"""Basel 97.5% Expected Shortfall (simplified teaching benchmark)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.risk.es.expected_shortfall import es_student_t, es_normal


def basel_es_97_5(fit, returns: pd.Series, portfolio_value: float = 1_000_000.0,
                  base_horizon: int = 10) -> dict:
    """Simplified Basel 97.5% one-tailed ES at the 10-day base horizon.

    Per Basel MAR33: 97.5% one-tailed ES, 10-day base liquidity horizon.
    This is a teaching benchmark — does NOT implement stressed
    calibration, liquidity buckets, modellability tests, or capital
    multipliers.
    """
    sigma_daily = fit.next_vol if hasattr(fit, "next_vol") else float(
        fit.conditional_volatility.iloc[-1]
    )
    sigma_horizon = sigma_daily * math.sqrt(base_horizon)
    mu = float(fit.params.get("mu", 0.0)) * base_horizon
    nu = float(fit.params.get("nu", 8.0))
    # Basel uses 97.5% confidence → α = 0.025 (loss tail)
    alpha = 0.025
    es_rate = es_student_t(mu, sigma_horizon, alpha, nu) if nu > 2 else es_normal(mu, sigma_horizon, alpha)
    es_currency = portfolio_value * abs(es_rate)
    var_rate = mu + sigma_horizon * stats.t.ppf(alpha, nu) * math.sqrt((nu - 2) / nu) if nu > 2 else mu + sigma_horizon * stats.norm.ppf(alpha)
    var_currency = portfolio_value * abs(var_rate)
    return {
        "model": fit.name,
        "confidence": 0.975,
        "base_horizon_days": base_horizon,
        "es_return_pct": float(es_rate) * 100,
        "es_currency": float(es_currency),
        "var_return_pct": float(var_rate) * 100,
        "var_currency": float(var_currency),
        "es_to_var_ratio": float(abs(es_rate / var_rate)) if var_rate else None,
        "portfolio_value": portfolio_value,
    }
