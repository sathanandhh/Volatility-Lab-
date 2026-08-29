"""What-if scenario shocks applied to a fitted model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.models.univariate.arch_family import FitResult


@dataclass
class ShockedModel:
    """A fitted model with a one-off shock applied to the latest observation."""
    base_model: FitResult
    shock: float
    next_vol: float
    variance_path: list[float]


def apply_shock(fit: FitResult, shock: float) -> ShockedModel:
    """Apply an exogenous shock ε_t+1 = shock to the fitted model.

    Computes the implied next-period variance via the model's recursion.
    """
    p = fit.params
    last_vol = float(fit.conditional_volatility.iloc[-1])
    last_var = last_vol ** 2
    shock_sq = shock ** 2
    # GARCH(1,1)-style recursion (works as a teaching approximation for all arch_family)
    omega = float(p.get("omega", 0.0))
    alpha = float(p.get("alpha[1]", 0.05))
    beta = float(p.get("beta[1]", 0.90))
    gamma = float(p.get("gamma[1]", 0.0))
    # For GJR: variance gets +gamma*shock^2 if shock < 0
    asym_term = gamma * shock_sq * (1.0 if shock < 0 else 0.0)
    next_var = omega + (alpha + asym_term / shock_sq) * shock_sq + beta * last_var
    next_var = max(next_var, 1e-10)
    # Build an indicative path: variance decays back to long-run
    lr_var = omega / (1 - alpha - beta - gamma / 2) if (1 - alpha - beta - gamma / 2) > 0 else next_var
    path = [next_var]
    for _ in range(9):
        prev = path[-1]
        path.append(omega + alpha * prev * (shock_sq / last_var) + beta * prev)
    return ShockedModel(
        base_model=fit, shock=shock,
        next_vol=math.sqrt(next_var),
        variance_path=path,
    )
