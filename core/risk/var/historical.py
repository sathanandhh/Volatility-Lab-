"""Historical-simulation VaR."""
from __future__ import annotations

import numpy as np
import pandas as pd


def var_historical(returns: pd.Series, alpha: float, horizon: int = 1) -> float:
    """Empirical-quantile VaR from the historical return distribution."""
    r = returns.dropna()
    if horizon > 1:
        # Aggregate to horizon returns
        r = r.rolling(horizon).sum().dropna()
    if len(r) < 30:
        return float("nan")
    return float(r.quantile(alpha))
