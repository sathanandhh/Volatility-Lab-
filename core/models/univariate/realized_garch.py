"""Realized GARCH — uses intraday realized variance (stub)."""
from __future__ import annotations

import pandas as pd

from core.models.univariate.arch_family import FitResult


def fit_realized_garch(returns: pd.Series, realized_var: pd.Series | None = None) -> FitResult:
    """Stub — requires intraday realized variance input.

    Falls back to standard GARCH(1,1) when realized_var is None.
    """
    from core.models.univariate.arch_family import fit_arch_family
    return fit_arch_family(returns, family="GARCH", p=1, q=1, o=0,
                           distribution="t", mean="Constant")
