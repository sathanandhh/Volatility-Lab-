"""Parametric VaR — Normal, Student-t, Cornish-Fisher."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def var_normal(mu: float, sigma: float, alpha: float) -> float:
    """Normal parametric VaR (return rate; negative = loss)."""
    return mu + sigma * stats.norm.ppf(alpha)


def var_student_t(mu: float, sigma: float, alpha: float, nu: float) -> float:
    """Student-t parametric VaR with unit-variance standardisation."""
    return mu + sigma * stats.t.ppf(alpha, nu) * math.sqrt((nu - 2) / nu)


def var_cornish_fisher(returns: pd.Series, mu: float, sigma: float,
                       alpha: float) -> float:
    """Cornish-Fisher VaR — adjusts for skew and kurtosis."""
    r = returns.dropna()
    s = float(stats.skew(r, bias=False))
    k = float(stats.kurtosis(r, fisher=True, bias=False))
    z = stats.norm.ppf(alpha)
    z_cf = (z + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    return mu + sigma * z_cf
