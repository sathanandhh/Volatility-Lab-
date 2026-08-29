"""Risk attribution — decompose portfolio VaR by risk factor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def attribute_var(returns: pd.DataFrame, weights: dict[str, float],
                  confidence: float = 0.975) -> dict:
    """Decompose portfolio VaR into asset contributions (percentage of total)."""
    from core.risk.portfolio import portfolio_var_decomposition
    decomp = portfolio_var_decomposition(returns, weights, confidence=confidence)
    total_component = sum(decomp["component_var"].values())
    return {
        "by_asset": {
            a: {"contribution": v,
                 "share_pct": float(v / total_component * 100) if total_component else 0.0}
            for a, v in decomp["component_var"].items()
        },
        "total_var_pct": decomp["portfolio_var_return_pct"],
    }
