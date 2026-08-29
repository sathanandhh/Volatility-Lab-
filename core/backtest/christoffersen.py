"""Christoffersen independence and conditional-coverage tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def christoffersen_test(hits: pd.Series) -> tuple[float, float]:
    """Christoffersen independence test. H0: breaches are independent."""
    h = hits.astype(int).to_numpy()
    if len(h) < 2:
        return float("nan"), float("nan")
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _ll(n: int, p: float) -> float:
        return 0.0 if n == 0 else n * np.log(np.clip(p, 1e-12, 1 - 1e-12))

    ll_ind = (_ll(n00, 1 - pi0) + _ll(n01, pi0) +
              _ll(n10, 1 - pi1) + _ll(n11, pi1))
    ll_dep = _ll(n00 + n10, 1 - pi) + _ll(n01 + n11, pi)
    stat = max(0.0, -2 * (ll_dep - ll_ind))
    p = 1 - stats.chi2.cdf(stat, 1)
    return float(stat), float(p)


def conditional_coverage(hits: pd.Series, alpha: float = 0.025) -> tuple[float, float]:
    """Joint conditional-coverage test (Kupiec + Christoffersen)."""
    from core.backtest.kupiec import kupiec_test
    kup_stat, _ = kupiec_test(hits, alpha)
    ind_stat, _ = christoffersen_test(hits)
    cc_stat = kup_stat + ind_stat
    cc_p = 1 - stats.chi2.cdf(cc_stat, 2)
    return float(cc_stat), float(cc_p), float(kup_stat), float(ind_stat)
