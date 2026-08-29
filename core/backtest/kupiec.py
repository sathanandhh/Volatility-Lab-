"""Kupiec POF (Proportion of Failures) test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_test(hits: pd.Series, alpha: float = 0.025) -> tuple[float, float]:
    """Kupiec POF test. H0: observed breach rate = expected (α)."""
    n = int(hits.size)
    if n == 0:
        return float("nan"), float("nan")
    x = int(hits.sum())
    phat = np.clip(x / n, 1e-12, 1 - 1e-12)
    a = np.clip(alpha, 1e-12, 1 - 1e-12)
    ll0 = (n - x) * np.log(1 - a) + x * np.log(a)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    stat = max(0.0, -2 * (ll0 - ll1))
    p = 1 - stats.chi2.cdf(stat, 1)
    return float(stat), float(p)
