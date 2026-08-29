"""IGARCH — integrated GARCH (persistence = 1)."""
from __future__ import annotations

import pandas as pd

from core.models.univariate.arch_family import FitResult, fit_arch_family


def fit_igarch(returns: pd.Series, p: int = 1, q: int = 1,
               distribution: str = "t") -> FitResult:
    return fit_arch_family(returns, family="IGARCH", p=p, q=q, o=0,
                           distribution=distribution, mean="Constant")
