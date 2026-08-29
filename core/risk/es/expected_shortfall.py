"""Expected Shortfall (CVaR) — parametric and historical."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def es_normal(mu: float, sigma: float, alpha: float) -> float:
    """Normal ES: E[L | L > VaR] = σ φ(z) / α."""
    z = stats.norm.ppf(alpha)
    return mu + sigma * stats.norm.pdf(z) / alpha


def es_student_t(mu: float, sigma: float, alpha: float, nu: float) -> float:
    """Student-t ES with unit-variance standardisation."""
    q = stats.t.ppf(alpha, nu)
    scale = math.sqrt((nu - 2) / nu)
    pdf_q = stats.t.pdf(q, nu)
    es_standardised = (pdf_q * (nu + q ** 2)) / ((nu - 1) * alpha)
    return mu + sigma * scale * es_standardised


def es_historical(returns: pd.Series, alpha: float, horizon: int = 1) -> float:
    """Historical ES — average loss beyond the empirical VaR quantile."""
    r = returns.dropna()
    if horizon > 1:
        r = r.rolling(horizon).sum().dropna()
    if len(r) < 30:
        return float("nan")
    q = r.quantile(alpha)
    tail = r[r <= q]
    return float(tail.mean()) if not tail.empty else float(q)
