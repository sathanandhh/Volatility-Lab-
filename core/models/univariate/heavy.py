"""HEAVY model (Shephard-Andersen 2009) — stub."""
from __future__ import annotations

import pandas as pd

from core.models.univariate.arch_family import FitResult


def fit_heavy(returns: pd.Series) -> FitResult:
    """Stub — falls back to HARCH (multi-scale ARCH)."""
    from core.models.univariate.arch_family import fit_arch_family
    return fit_arch_family(returns, family="HARCH", p=3, q=0, o=0,
                           distribution="t", mean="Constant")
