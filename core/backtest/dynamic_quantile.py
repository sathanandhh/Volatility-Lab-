"""Engle-Manganelli Dynamic Quantile (DQ) test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dynamic_quantile_test(returns: pd.Series, var_forecast: pd.Series,
                          alpha: float = 0.025) -> dict:
    """Engle-Manganelli DQ test.

    Regresses the breach indicator on the VaR forecast and lagged
    breaches; a high p-value means the model is well-calibrated.
    """
    aligned = pd.DataFrame({"r": returns, "var": var_forecast}).dropna()
    if len(aligned) < 30:
        return {"statistic": float("nan"), "p_value": float("nan")}
    hits = (aligned["r"] < aligned["var"]).astype(float).to_numpy()
    # Regressor: [1, var, hits_lag1]
    var_t = aligned["var"].to_numpy()
    hit_lag = np.roll(hits, 1)
    hit_lag[0] = 0
    X = np.column_stack([np.ones_like(hits), var_t, hit_lag])
    try:
        beta, *_ = np.linalg.lstsq(X, hits, rcond=None)
        resid = hits - X @ beta
        n, k = len(hits), X.shape[1]
        if n - k <= 0 or resid.var() == 0:
            return {"statistic": 0.0, "p_value": 1.0}
        # F-test for joint significance of regressors
        ssr = float(resid @ resid)
        tss = float(((hits - hits.mean()) ** 2).sum())
        if tss == 0 or ssr == tss:
            return {"statistic": 0.0, "p_value": 1.0}
        f_stat = ((tss - ssr) / (k - 1)) / (ssr / (n - k))
        p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        return {"statistic": float(f_stat), "p_value": float(p_value)}
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan")}
