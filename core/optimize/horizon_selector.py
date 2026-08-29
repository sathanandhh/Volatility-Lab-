"""Forecast horizon recommendation based on use case."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.optimize.optimizer import OptimizationResult


def select_horizon(returns: pd.Series, target_use: str = "regulatory") -> OptimizationResult:
    """Pick a forecast horizon based on the intended use case."""
    presets = {
        "regulatory": 10, "weekly": 5, "monthly": 21,
        "stress": 60, "auto": None,
    }
    if target_use == "auto":
        # Use squared-return ACF half-life as a guide
        sq = returns.pow(2)
        lags = range(1, 31)
        acf = np.array([sq.autocorr(l) for l in lags])
        valid = acf[acf > 0]
        if len(valid) >= 5:
            slope = np.polyfit(np.arange(1, len(valid) + 1), np.log(valid), 1)[0]
            half_life = -np.log(2) / slope if slope < 0 else 10
            horizon = int(max(5, min(60, round(half_life))))
        else:
            horizon = 10
    else:
        horizon = presets.get(target_use, 10)
    rec = (
        f"Recommended horizon: {horizon} periods ({target_use})."
        + (" Basel 10-day base liquidity horizon." if target_use == "regulatory" else "")
    )
    return OptimizationResult(
        kind="horizon", optimal=int(horizon),
        optimal_score=float(horizon), criterion=target_use,
        sweep_table=[{"target_use": target_use, "horizon": horizon}],
        recommendation=rec,
    )
