"""Normality tests on standardized residuals."""
from __future__ import annotations

import pandas as pd


def jarque_bera(series: pd.Series) -> dict:
    from scipy import stats
    s = series.dropna()
    stat, p = stats.jarque_bera(s)
    return {"statistic": float(stat), "p_value": float(p)}


def shapiro_wilk(series: pd.Series) -> dict:
    from scipy import stats
    s = series.dropna()
    if len(s) > 5000:
        s = s.sample(5000, random_state=42)
    if len(s) < 3:
        return {"statistic": float("nan"), "p_value": float("nan")}
    stat, p = stats.shapiro(s)
    return {"statistic": float(stat), "p_value": float(p)}


def anderson_darling(series: pd.Series) -> dict:
    from scipy import stats
    s = series.dropna()
    if len(s) < 8:
        return {"statistic": float("nan"), "p_value": float("nan")}
    result = stats.anderson(s, dist="norm")
    # Approximate p-value via interpolation
    sig_levels = result.significance_level
    crit_vals = result.critical_values
    import numpy as np
    p = float(np.interp(result.statistic, crit_vals, sig_levels)[0])
    # The interpolation isn't perfect; clamp to [0,100]
    p = max(0.0, min(100.0, p)) / 100.0
    return {"statistic": float(result.statistic), "p_value": p}
