"""Out-of-sample accuracy metrics for volatility forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def qlike(realized: pd.Series, forecast_var: pd.Series) -> float:
    """QLIKE loss — preferred volatility-forecast metric (Patton 2011)."""
    v = forecast_var.clip(lower=1e-10)
    r2 = realized.pow(2).reindex(v.index)
    return float(np.mean(np.log(v) + r2 / v))


def volatility_rmse(realized: pd.Series, cond_vol: pd.Series) -> float:
    """RMSE between r² and σ² (realized-variance proxy)."""
    cv = cond_vol.reindex(realized.index)
    r2 = realized.pow(2)
    diff = (r2 - cv.pow(2)).dropna()
    if diff.empty:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(realized: pd.Series, cond_vol: pd.Series) -> float:
    """Mean absolute error between |r| and σ."""
    cv = cond_vol.reindex(realized.index)
    diff = (realized.abs() - cv).dropna()
    if diff.empty:
        return float("nan")
    return float(np.mean(diff.abs()))


def mafe(realized: pd.Series, cond_vol: pd.Series) -> float:
    """Mean absolute forecast error of volatility."""
    cv = cond_vol.reindex(realized.index)
    ratio = (realized.abs() - cv).abs() / cv.clip(lower=1e-10)
    return float(ratio.dropna().mean())
