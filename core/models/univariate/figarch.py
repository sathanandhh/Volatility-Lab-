"""FIGARCH — fractionally integrated GARCH (long memory)."""
from __future__ import annotations

import pandas as pd

from core.models.univariate.arch_family import FitResult, fit_arch_family


def fit_figarch(returns: pd.Series, p: int = 1, q: int = 1,
                distribution: str = "t") -> FitResult:
    """FIGARCH via arch's vol='FIGARCH'."""
    return fit_arch_family(returns, family="FIGARCH", p=p, q=q, o=0,
                           distribution=distribution, mean="Constant")
