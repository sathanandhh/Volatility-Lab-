"""ARCH-family wrappers around the `arch` package.

Provides a uniform interface for fitting and forecasting:
  - fit_arch_family(returns, family, p, q, o, distribution, mean) -> FitResult
  - forecast_arch_family(fit, horizon, method, simulations) -> ForecastResult

FitResult exposes: params, conditional_volatility, std_resid,
loglikelihood, aic, bic, n_params, converged, std_resid, next_vol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.config import get_settings


@dataclass
class FitResult:
    """Uniform fit result for any univariate model."""
    name: str
    params: dict[str, float]
    conditional_volatility: pd.Series
    std_resid: pd.Series
    loglikelihood: float
    aic: float
    bic: float
    n_params: int
    converged: bool
    convergence_flag: int = 0
    next_vol: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def residuals(self) -> pd.Series:
        return self.std_resid


@dataclass
class ForecastResult:
    """Uniform forecast result."""
    variance_path: np.ndarray
    mean_path: np.ndarray | None = None
    method: str = "analytic"
    horizon: int = 1


def fit_arch_family(
    returns: pd.Series,
    family: str = "GARCH",
    p: int = 1, q: int = 1, o: int = 0,
    distribution: str = "t",
    mean: str = "Constant",
) -> FitResult:
    """Fit an ARCH-family model to a percentage-log-returns series."""
    from arch import arch_model
    settings = get_settings()
    r = returns.dropna()
    if len(r) < 50:
        raise ValueError(f"Need ≥50 observations; got {len(r)}.")
    # Map distribution names: "normal" -> "normal", "t" -> "t", etc.
    dist_map = {"normal": "normal", "t": "t", "ged": "ged",
                "skewt": "skewt", "jsu": "jsu"}
    arch_dist = dist_map.get(distribution, distribution)
    spec = arch_model(
        r, mean=mean, vol=family, p=p, q=q, o=o,
        dist=arch_dist, rescale=False,
    )
    fit = spec.fit(disp="off", show_warning=False,
                   options={"maxiter": settings.default_maxiter})
    next_var = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    return FitResult(
        name=f"{family}({p},{q},{o})",
        params={k: float(v) for k, v in fit.params.items()},
        conditional_volatility=fit.conditional_volatility,
        std_resid=pd.Series(fit.std_resid, index=r.index).replace(
            [np.inf, -np.inf], np.nan
        ).dropna(),
        loglikelihood=float(fit.loglikelihood),
        aic=float(fit.aic),
        bic=float(fit.bic),
        n_params=int(len(fit.params)),
        converged=bool(fit.convergence_flag == 0),
        convergence_flag=int(fit.convergence_flag),
        next_vol=math.sqrt(max(next_var, 0.0)),
        extras={"_arch_fit": fit},
    )


def forecast_arch_family(
    fit: FitResult, horizon: int = 10,
    method: str = "analytic", simulations: int = 1000,
) -> ForecastResult:
    """Forecast variance path from a fitted ARCH-family model."""
    arch_fit = fit.extras.get("_arch_fit")
    if arch_fit is None:
        raise ValueError("FitResult does not contain the underlying arch fit object.")
    if method == "simulation" or (fit.name.startswith("EGARCH") and horizon > 1):
        f = arch_fit.forecast(horizon=horizon, reindex=False,
                              method="simulation", simulations=simulations)
    elif method == "bootstrap":
        f = arch_fit.forecast(horizon=horizon, reindex=False,
                              method="bootstrap", simulations=simulations)
    else:
        f = arch_fit.forecast(horizon=horizon, reindex=False, method="analytic")
    var = f.variance.iloc[-1].to_numpy()
    return ForecastResult(variance_path=var, method=method, horizon=horizon)
