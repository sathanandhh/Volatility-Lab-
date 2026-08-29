"""EWMA (RiskMetrics 1996) — closed-form, no MLE required."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.models.univariate.arch_family import FitResult, ForecastResult


def fit_ewma(returns: pd.Series, decay: float = 0.94) -> FitResult:
    """Fit EWMA with the given decay (λ). No optimization — closed form."""
    r = returns.dropna()
    var = r.ewm(alpha=1 - decay, adjust=False).var(bias=False)
    cv = np.sqrt(var)
    std_resid = (r / cv).replace([np.inf, -np.inf], np.nan).dropna()
    # Pseudo log-likelihood (Normal innovations) for AIC comparability
    n = len(r)
    var_nz = var.replace(0, np.nan).dropna()
    ll = float(-0.5 * (np.sum(np.log(2 * np.pi) + np.log(var_nz) + r.reindex(var_nz.index)**2 / var_nz)))
    k = 1  # decay is imposed, not estimated
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    return FitResult(
        name=f"EWMA(λ={decay})",
        params={"lambda": float(decay)},
        conditional_volatility=cv,
        std_resid=std_resid,
        loglikelihood=ll,
        aic=float(aic),
        bic=float(bic),
        n_params=k,
        converged=True,
        convergence_flag=0,
        next_vol=float(cv.iloc[-1]),
        extras={"_decay": decay},
    )


def forecast_ewma(fit: FitResult, horizon: int = 10) -> ForecastResult:
    """EWMA forecasts are flat (no mean reversion)."""
    last_vol = fit.next_vol
    var = last_vol ** 2
    path = np.full(horizon, var)
    return ForecastResult(variance_path=path, method="analytic", horizon=horizon)
