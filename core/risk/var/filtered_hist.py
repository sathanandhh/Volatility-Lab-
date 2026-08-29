"""Filtered Historical Simulation (FHS) — bootstraps std residuals × current σ."""
from __future__ import annotations

import numpy as np
import pandas as pd


def var_fhs(std_resid: pd.Series, sigma: float, alpha: float,
            n_samples: int = 10_000) -> float:
    """FHS VaR — resample standardised residuals, scale by current σ."""
    z = std_resid.dropna().to_numpy()
    if len(z) < 30:
        return float("nan")
    rng = np.random.default_rng(42)
    sampled = rng.choice(z, size=n_samples, replace=True)
    losses = sigma * sampled
    return float(np.quantile(losses, alpha))
