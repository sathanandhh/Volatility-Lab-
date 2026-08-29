"""Diebold-Mariano test for comparing forecast accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(realized: pd.Series, forecast_a: pd.Series,
                         forecast_b: pd.Series, loss: str = "qlike") -> dict:
    """Diebold-Mariano test. H0: equal forecast accuracy."""
    r2 = realized.pow(2).reindex(forecast_a.index).dropna()
    a = forecast_a.reindex(r2.index).clip(lower=1e-10)
    b = forecast_b.reindex(r2.index).clip(lower=1e-10)
    if loss == "qlike":
        e_a = np.log(a) + r2 / a
        e_b = np.log(b) + r2 / b
    else:  # MSE
        e_a = (r2 - a) ** 2
        e_b = (r2 - b) ** 2
    d = e_a - e_b
    d = d.dropna()
    n = len(d)
    if n < 5:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "mean_loss_diff": float("nan")}
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if var_d == 0:
        return {"statistic": 0.0, "p_value": 1.0, "mean_loss_diff": mean_d}
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return {
        "statistic": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": mean_d,
        "preferred_model": "a" if mean_d < 0 else "b",
    }
