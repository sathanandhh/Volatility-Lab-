"""ARCH-LM test on standardized residuals — remaining ARCH effect."""
from __future__ import annotations

import pandas as pd


def arch_lm_test(std_resid: pd.Series, nlags: int = 10) -> dict:
    from statsmodels.stats.diagnostic import het_arch
    s = std_resid.dropna()
    if len(s) < 50:
        return {"statistic": float("nan"), "p_value": float("nan")}
    lm_stat, lm_p, f_stat, f_p = het_arch(s, nlags=nlags)
    return {
        "statistic": float(lm_stat),
        "p_value": float(lm_p),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_p),
    }
