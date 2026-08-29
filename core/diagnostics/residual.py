"""Residual autocorrelation tests (Ljung-Box)."""
from __future__ import annotations

import pandas as pd


def ljung_box(series: pd.Series, lag: int = 10) -> dict:
    """Ljung-Box test for serial correlation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    s = series.dropna()
    if len(s) < 30:
        return {"statistic": float("nan"), "p_value": float("nan")}
    lb = acorr_ljungbox(s, lags=[lag], return_df=True)
    return {
        "statistic": float(lb.loc[lag, "lb_stat"]),
        "p_value": float(lb.loc[lag, "lb_pvalue"]),
    }


def ljung_box_residuals(std_resid: pd.Series, lag: int = 10) -> dict:
    """Convenience wrapper for std residuals."""
    return ljung_box(std_resid, lag=lag)
